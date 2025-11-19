# streamlit_jobrec_app.py
# Streamlit modular scaffold for: Job Recommendation System Based on Skills
# Run with: streamlit run streamlit_jobrec_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile, json, random, re, os, io, uuid
from typing import List, Dict, Tuple, Any, Optional
import PyPDF2

# your extractor
from skills_extractions import extract_skills_transformers

# try to import canonicalizer (optional)
try:
    from domain_agnostic_normalize import (
        ChromaCanonicalizer,
        extract_skills_from_jd_text,
        generate_candidates,
        normalize_text
    )
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False
    # fallback helpers
    def normalize_text(s: str) -> str:
        if s is None: return ""
        s = s.strip().lower()
        s = re.sub(r'[^a-z0-9\+\#\s\-]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def generate_candidates(raw: str):
        txt = normalize_text(raw)
        return [t for t in re.split(r'[\/,;|]', txt) if t]

# sentence-transformers lazy load
_SBM = None
@st.cache_resource
def get_sbert(model_name: str = "all-MiniLM-L6-v2"):
    global _SBM
    if _SBM is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SBM = SentenceTransformer(model_name)
        except Exception:
            _SBM = None
    return _SBM

# ---------------------------
# DEFAULT JOBS (small demo)
# ---------------------------
DEFAULT_JOBS = [

]

# ---------------------------
# SESSION / JOBS helpers
# ---------------------------
def save_jobs_to_session(jobs: List[Dict]):
    st.session_state['jobs'] = jobs

def get_jobs() -> List[Dict]:
    if 'jobs' not in st.session_state:
        st.session_state['jobs'] = DEFAULT_JOBS.copy()
    return st.session_state['jobs']

def add_job(title, company, description, skills):
    jobs = get_jobs()
    new_id = max([j.get('id', 0) for j in jobs]) + 1 if jobs else 1
    jobs.append({'id': new_id, 'title': title, 'company': company, 'description': description, 'skills': skills})
    save_jobs_to_session(jobs)

def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# ---------------------------
# simple heuristic extractor (fallback)
# ---------------------------
def extract_skills_from_text(text: str) -> List[str]:
    if not text:
        return []
    common_skills = [
        'python','java','c++','react','typescript','sql','postgresql','django','flask','docker',
        'pandas','numpy','scikit-learn','ml','aws','azure','gcp','html','css','javascript'
    ]
    tokens = [t.strip().lower() for t in re.split(r'[^a-zA-Z0-9\+\#]+', text) if t.strip()]
    found = sorted(list(set([s for s in common_skills if s in tokens])))
    return found

# ---------------------------
# normalize LLM output
# ---------------------------
def normalize_skills_output(raw: Any) -> Tuple[List[str], pd.DataFrame]:
    skills_list: List[str] = []
    details_df = pd.DataFrame()
    # unwrap possible tuple/list wrappers
    if isinstance(raw, (tuple,list)) and len(raw)>0 and not all(isinstance(x, (str, dict)) for x in raw):
        for item in raw:
            if isinstance(item, list):
                raw = item
                break
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            raw = parsed
        except Exception:
            pass
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        skills_list = [s.strip().lower() for s in raw if s and isinstance(s,str)]
        skills_list = sorted(list(dict.fromkeys(skills_list)))
        details_df = pd.DataFrame({'skill': skills_list})
        return skills_list, details_df
    if isinstance(raw, list) and all(isinstance(x, dict) for x in raw):
        rows=[]
        for d in raw:
            name = d.get('skill') or d.get('name') or d.get('label') or d.get('text')
            conf = d.get('confidence') or d.get('score') or None
            if name:
                rows.append({'skill': str(name).strip().lower(), 'confidence': conf})
        if rows:
            details_df = pd.DataFrame(rows)
            skills_list = list(dict.fromkeys(details_df['skill'].tolist()))
            return skills_list, details_df
    if isinstance(raw, dict):
        if 'skills' in raw and isinstance(raw['skills'], list):
            return normalize_skills_output(raw['skills'])
    return [], pd.DataFrame()

# ---------------------------
# MATCHING helpers (embedding + overlap)
# ---------------------------
def avg_embedding(model, texts: List[str]) -> Optional[np.ndarray]:
    if not texts:
        return None
    embs = model.encode(texts, convert_to_numpy=True)
    if embs.ndim == 1:
        vec = embs
    else:
        vec = embs.mean(axis=0)
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm and norm != 0 else vec

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

def token_jaccard(a: List[str], b: List[str]) -> float:
    sa = set([normalize_text(x) for x in a if x])
    sb = set([normalize_text(x) for x in b if x])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / (len(sa | sb) + 1e-12)

# ---------------------------
# UI: Sidebar controls + manual JD paste
# ---------------------------
def sidebar_controls():
    st.sidebar.title("Controls")
    uploaded_resume = st.sidebar.file_uploader("Upload resume (txt or pdf)", type=['txt','pdf'])
    uploaded_jobs = st.sidebar.file_uploader("Upload jobs CSV (optional)", type=['csv'])
    top_k = st.sidebar.slider("Number of recommendations", min_value=3, max_value=50, value=5)
    sample_size = st.sidebar.slider("Sample size (for ranking from session jobs)", min_value=1, max_value=50, value=50)
    ranking_model = st.sidebar.selectbox("Embedding model (SBERT)", options=["all-MiniLM-L6-v2"], index=0)
    w_overlap = st.sidebar.slider("Weight: canonical/token overlap", 0.0, 1.0, 0.6)
    w_embed = st.sidebar.slider("Weight: embedding similarity", 0.0, 1.0, 0.4)
    add_pasted = st.sidebar.checkbox("Enable manual JD paste panel", value=True)
    return uploaded_resume, uploaded_jobs, top_k, sample_size, ranking_model, w_overlap, w_embed, add_pasted

# ---------------------------
# Panels
# ---------------------------
def resume_panel(uploaded_resume) -> Tuple[str, List[str]]:
    st.header("Candidate Resume")
    resume_text = ""
    if uploaded_resume is not None:
        if uploaded_resume.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_resume)
        else:
            try:
                bytes_data = uploaded_resume.read()
                resume_text = bytes_data.decode('utf-8', errors='ignore')
            except Exception:
                resume_text = ""
                st.warning("Couldn't decode resume file as text — try pasting plain text or upload a .txt file.")

    if not resume_text:
        st.info("No resume uploaded. Paste resume text below to test the pipeline.")
        resume_text = st.text_area("Paste resume text here", height=240)

    st.subheader("Extracted Skills (ML model)")
    skills: List[str] = []
    details_df = pd.DataFrame()

    if resume_text:
        with st.spinner("Extracting skills with LLM..."):
            try:
                raw_out = extract_skills_transformers(resume_text)
                skills, details_df = normalize_skills_output(raw_out)
                if skills:
                    st.markdown("**Skills (detected):**")
                    st.markdown(''.join([f"- {s}" for s in skills]))
                    if not details_df.empty:
                        cols = [c for c in details_df.columns if c != 'skill']
                        ordered = ['skill'] + cols
                        st.subheader('Skill details')
                        st.dataframe(details_df[ordered])
                else:
                    st.info("No skills extracted by the model. Falling back to heuristic.")
                    skills = extract_skills_from_text(resume_text)
                    if skills:
                        st.markdown("**Fallback skills:**")
                        st.markdown(''.join([f"- {s}" for s in skills]))
            except Exception as e:
                st.warning(f"Could not extract skills with ML model: {e}. Falling back to heuristic.")
                skills = extract_skills_from_text(resume_text)
                if skills:
                    st.markdown("**Fallback skills:**")
                    st.markdown(''.join([f"- {s}" for s in skills]))

    skills = [s.strip().lower() for s in skills if isinstance(s, str) and s.strip()]
    return resume_text, skills

def jobs_panel():
    st.header("Jobs (session)")
    jobs = get_jobs()
    df = pd.DataFrame([{
        'id': j['id'], 'title': j['title'], 'company': j['company'], 'skills': ', '.join(j.get('skills', []))
    } for j in jobs])
    st.dataframe(df)
    if st.button("Add sample job to session"):
        new_id = max([j['id'] for j in jobs]) + 1 if jobs else 1
        jobs.append({
            'id': new_id,
            'title': 'New Role ' + str(new_id),
            'company': 'Company ' + str(new_id),
            'description': '',
            'skills': []
        })
        save_jobs_to_session(jobs)
        # guarded rerun (not all streamlit builds expose experimental_rerun)
        try:
            # st.experimental_rerun()
            st.rerun()
        except Exception:
            pass
    return jobs

# ---------------------------
# Manual JD paste & utilities
# ---------------------------

def manual_jd_panel(enable: bool):
    if not enable:
        return
    st.sidebar.markdown("---")
    st.sidebar.subheader("Paste LinkedIn JD(s)")
    st.sidebar.markdown("Paste one JD or multiple JDs separated by a line containing exactly `===JD===`")
    pasted = st.sidebar.text_area("Paste JD(s) here", height=300)
    title_default = st.sidebar.text_input("Default title (applied to pasted JDs if not present)", value="LinkedIn Role")
    company_default = st.sidebar.text_input("Default company", value="LinkedIn")

    # Sidebar processing toggles
    use_chroma = st.sidebar.checkbox("Auto-canonicalize with Chroma (may be slow first run)", value=False)
    do_semantic = st.sidebar.checkbox("Use semantic dedupe (SBERT, slower)", value=False)

    # helper to try various extractors and normalize output into list of tokens
    def jd_extract_candidates(text: str):
        cands = []
        try:
            if 'extract_skills_from_jd_text' in globals():
                cands = extract_skills_from_jd_text(text, prefer_sectioned=True)
            elif 'generate_candidates' in globals():
                cands = []
                for ln in text.splitlines()[:80]:
                    for g in generate_candidates(ln):
                        if g and g not in cands:
                            cands.append(g)
            else:
                cands = extract_skills_from_text(text)
        except Exception:
            try:
                cands = extract_skills_from_text(text)
            except Exception:
                cands = []

        # normalize tokens if normalize_text exists
        out = []
        for s in cands:
            try:
                n = normalize_text(s) if 'normalize_text' in globals() else str(s).strip().lower()
            except Exception:
                n = str(s).strip().lower()
            if n and n not in out:
                out.append(n)
        return out

    # Initialize Chroma + SBERT once if requested
    chroma_reg = None
    sbert_model = None
    chroma_init_warned = False

    if st.sidebar.button("Add pasted JD(s) to session"):
        if not pasted.strip():
            st.sidebar.warning("Paste one or more JDs first.")
            return

        parts = [p.strip() for p in pasted.split("===JD===") if p.strip()]
        added = 0

        # init resources lazily
        if use_chroma:
            try:
                chroma_reg = ChromaCanonicalizer()
            except Exception as e:
                chroma_reg = None
                st.sidebar.warning(f"Chroma init failed (continuing without canonicalization): {e}")
                chroma_init_warned = True

        if do_semantic:
            try:
                sbert_model = get_sbert()  # uses default model name, returns None if unavailable
                if sbert_model is None:
                    st.sidebar.warning("SBERT model not available; semantic dedupe disabled.")
            except Exception:
                sbert_model = None
                st.sidebar.warning("Failed to load SBERT model; semantic dedupe disabled.")

        # process pasted JDs
        with st.spinner(f"Processing {len(parts)} JD(s) — extracting skills..."):
            for p in parts:
                # heuristics to get a title from the pasted text
                lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
                title = title_default
                company = company_default
                if len(lines) >= 2 and len(lines[0]) < 30 and 'about the job' not in lines[0].lower():
                    title = lines[0]
                elif len(lines) >= 1 and len(lines[0]) < 60 and 'about the job' not in lines[0].lower():
                    title = lines[0]

                # 1) extract raw candidates
                raw = jd_extract_candidates(p)

                # 2) clean & fuzzy-dedupe (uses normalize_text if available)
                try:
                    skills_list = clean_extracted_skills(raw, normalize_fn=(normalize_text if 'normalize_text' in globals() else None),
                                                        fuzzy_thresh=88, semantic_dedupe=False, sbert_model=None)
                except Exception:
                    # fallback minimal normalization
                    skills_list = [s.strip().lower() for s in raw if s and isinstance(s, str)]

                # optional semantic dedupe (slower)
                if do_semantic and sbert_model is not None and skills_list:
                    try:
                        skills_list = clean_extracted_skills(skills_list, normalize_fn=(normalize_text if 'normalize_text' in globals() else None),
                                                            fuzzy_thresh=88, semantic_dedupe=True, sbert_model=sbert_model)
                    except Exception:
                        pass

                # 3) add job to session (add_job assigns id)
                add_job(title=title, company=company, description=p, skills=skills_list)

                # 4) if chroma requested & initialized, canonicalize the skills for this job
                if use_chroma and chroma_reg is not None and skills_list:
                    try:
                        # add_phrases will add/update canonical entries and return mapping phrase->cid
                        add_map = chroma_reg.add_phrases(skills_list, generate_candidates_fn=(generate_candidates if 'generate_candidates' in globals() else None))
                        # update the last appended job with canonical ids
                        jobs = get_jobs()
                        if jobs:
                            # last job should be the one we just appended
                            last = jobs[-1]
                            cids = [add_map.get(s) for s in skills_list if add_map.get(s)]
                            last['canonical_ids'] = cids
                            save_jobs_to_session(jobs)
                    except Exception:
                        # ignore chroma errors for robustness
                        pass

                added += 1

        st.sidebar.success(f"Added {added} JD(s) to session jobs (skills auto-extracted).")

        # guarded rerun (older streamlit may not have experimental_rerun)
        try:
            # st.experimental_rerun()
            st.rerun()
        except Exception:
            # nothing — session_state already updated; UI will reflect changes on next interaction or refresh
            pass
        
# ---------------------------
# Ranking logic (sample + rank)
# ---------------------------
def rank_resume_vs_sampled_jds(resume_skills: List[str], jobs: List[Dict],
                               sample_size: int = 50, model_name: str = "all-MiniLM-L6-v2",
                               w_overlap: float = 0.6, w_embed: float = 0.4, top_k: int = 10):
    # sample (deterministic)
    if len(jobs) <= sample_size:
        sample = jobs.copy()
    else:
        random.seed(42)
        sample = random.sample(jobs, sample_size)

    # prepare candidate lists for JDs (prefer job['skills'] else heuristic/extractor)
    jd_records = []
    for rec in sample:
        cands = rec.get("skills") or []
        if not cands:
            # try using your JD-things extractor if available
            try:
                text = rec.get("description","")
                # prefer extract_skills_from_jd_text if available
                if 'extract_skills_from_jd_text' in globals():
                    cands = extract_skills_from_jd_text(text, prefer_sectioned=True)
                else:
                    cands = extract_skills_from_text(text)
            except Exception:
                cands = extract_skills_from_text(rec.get("description",""))
        # normalize tokens
        cands_norm = [normalize_text(x) for x in cands if normalize_text(x)]
        jd_records.append({'job': rec, 'candidates': cands_norm})

    # canonicalize via Chroma if possible
    reg = None
    mapping_cache = {}
    if CHROMA_AVAILABLE:
        try:
            reg = ChromaCanonicalizer()
        except Exception:
            reg = None

    # if resume_skills are canonical ids (rare in UI) - not likely - here we only have tokens
    # compute resume embedding
    model = get_sbert(model_name)
    if model is None:
        st.warning("SBERT model could not be loaded; embedding-based ranking disabled.")
    resume_emb = model.encode([s for s in resume_skills], convert_to_numpy=True).mean(axis=0) if (model is not None and resume_skills) else None
    if resume_emb is not None:
        resume_emb = resume_emb / (np.linalg.norm(resume_emb) + 1e-12)

    # build JD embedding texts and compute in batch
    jd_texts = [" ".join(rec['candidates']) if rec['candidates'] else normalize_text(rec['job'].get('description') or "") for rec in jd_records]
    jd_embs = None
    if model is not None:
        try:
            jd_embs = model.encode(jd_texts, convert_to_numpy=True)
            norms = np.linalg.norm(jd_embs, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            jd_embs = (jd_embs / norms).astype('float32')
        except Exception:
            jd_embs = None

    # scoring
    results = []
    for idx, rec in enumerate(jd_records):
        job = rec['job']
        cands = rec['candidates']
        # overlap token-based
        overlap = token_jaccard(resume_skills, cands)
        # embed similarity
        embed_sim = 0.0
        if resume_emb is not None and jd_embs is not None:
            emb = jd_embs[idx]
            sim = float(np.dot(resume_emb, emb))
            # convert cosine (-1..1) -> [0,1]
            embed_sim = max(0.0, min(1.0, (sim + 1.0)/2.0))
        final = w_overlap * overlap + w_embed * embed_sim
        results.append({
            "job": job,
            "candidates": cands,
            "overlap": overlap,
            "embed": embed_sim,
            "score": final
        })

    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    return results_sorted[:top_k], results_sorted

# ---------------------------
# Recommendations panel
# ---------------------------
def recommendations_panel(candidate_skills: List[str], jobs: List[Dict], top_k:int,
                          sample_size:int, model_name:str, w_overlap:float, w_embed:float):
    st.header("Recommendations / Ranking")

    st.markdown("You can either use the TF-IDF quick matcher (left) or the `Rank resume vs sampled JDs` for more robust ranking using embeddings + token overlap (right).")
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("TF-IDF quick match")
        if candidate_skills:
            scored = match_jobs_by_skills(candidate_skills, jobs, top_k=top_k)
            for job, score in scored:
                st.markdown(f"**{job['title']}** — {job['company']}")
                st.write(job.get('description','')[:400])
                st.write("Skills:", ', '.join(job.get('skills', [])))
                st.metric("TF-IDF score", f"{score:.3f}")
        else:
            st.info("No skills provided for TF-IDF quick match.")

    with col2:
        st.subheader("Embed + overlap ranking (sample + rank)")
        st.write(f"Sample size for ranking: **{sample_size}**")
        if st.button("Rank resume vs sampled JDs"):
            if not candidate_skills:
                st.warning("No candidate skills found. Paste a resume or enter skills manually.")
                return
            with st.spinner("Sampling JDs and computing rankings..."):
                top_results, all_results = rank_resume_vs_sampled_jds(candidate_skills, jobs, sample_size=sample_size, model_name=model_name, w_overlap=w_overlap, w_embed=w_embed, top_k=top_k)
            st.success("Ranking complete — top results:")
            for r in top_results:
                job = r['job']
                st.markdown(f"**{job.get('title')}** — {job.get('company')}")
                st.write(job.get('description','')[:600])
                st.write("Candidates:", ', '.join(r['candidates'][:30]))
                st.metric("Score", f"{r['score']:.3f}", delta=f"ov:{r['overlap']:.2f} emb:{r['embed']:.2f}")
            # provide download buttons
            buf = io.StringIO()
            for r in all_results:
                rec = {'id': r['job'].get('id'), 'title': r['job'].get('title'), 'company': r['job'].get('company'),
                       'score': r['score'], 'overlap': r['overlap'], 'embed': r['embed'], 'candidates': r['candidates']}
                buf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            buf.seek(0)
            st.download_button("Download ranked JSONL", data=buf.getvalue().encode('utf-8'), file_name="ranked_sampled_jds.jsonl")
            # CSV
            df = pd.DataFrame([{'id': r['job'].get('id'), 'title': r['job'].get('title'), 'company': r['job'].get('company'),
                                'score': r['score'], 'overlap': r['overlap'], 'embed': r['embed'],
                                'candidates': ";".join(r['candidates'])} for r in all_results])
            st.download_button("Download ranked CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="ranked_sampled_jds.csv")

# ---------------------------
# keep your existing TF-IDF matcher
# ---------------------------
def match_jobs_by_skills(candidate_skills: List[str], jobs: List[Dict], top_k:int=10) -> List[Tuple[Dict,float]]:
    job_skill_texts = [ ' '.join(j.get('skills', [])) for j in jobs ]
    cand_text = ' '.join(candidate_skills)
    texts = job_skill_texts + [cand_text]
    try:
        vectorizer = TfidfVectorizer().fit(texts)
        vectors = vectorizer.transform(texts)
        job_vecs = vectors[:-1]
        cand_vec = vectors[-1]
        sims = cosine_similarity(job_vecs, cand_vec.reshape(1, -1)).reshape(-1)
    except Exception:
        # fallback simple overlap
        sims = np.array([token_jaccard(candidate_skills, j.get('skills', [])) for j in jobs])
    scored = list(zip(jobs, sims))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    return scored_sorted[:top_k]

# ---------------------------
# MAIN
# ---------------------------
def main():
    st.set_page_config(page_title='Job Recommender — Skills-based', layout='wide')
    st.title('Job Recommendation System — Skills Based (Streamlit)')

    uploaded_resume, uploaded_jobs, top_k, sample_size, model_name, w_overlap, w_embed, add_pasted = sidebar_controls()

    # load jobs CSV if present
    if uploaded_jobs is not None:
        try:
            dfjobs = pd.read_csv(uploaded_jobs)
            jobs = []
            for _, r in dfjobs.iterrows():
                jobs.append({
                    'id': int(r.get('id', len(jobs)+1)),
                    'title': r.get('title','Untitled'),
                    'company': r.get('company',''),
                    'description': r.get('description',''),
                    'skills': (r.get('skills') or "").split(',') if 'skills' in r else []
                })
            save_jobs_to_session(jobs)
        except Exception as e:
            st.sidebar.error(f"Failed to load jobs CSV: {e}")

    # manual JD paste panel in sidebar
    manual_jd_panel(add_pasted)

    # Panels
    resume_text, candidate_skills = resume_panel(uploaded_resume)
    jobs = jobs_panel()
    recommendations_panel(candidate_skills, jobs, top_k, sample_size, model_name, w_overlap, w_embed)

    st.markdown("---")
    st.caption("This scaffold ranks a single resume vs a sampled set of JDs (paste LinkedIn JDs in the sidebar). Use Chroma canonicalizer if available for better canonical matching.")


# --- paste into your app.py ---
from rapidfuzz import fuzz
import re

# lightweight stopwords for JD cleanup (augment as needed)
_JD_STOPWORDS = {
    "and","or","the","a","an","this","that","these","those","of","for","to","in",
    "with","on","as","by","be","is","are","was","were","will","would","should",
    "our","you","your","we","their","they","it","its","which","at","from","about",
    "including","such","etc","please","apply","role","position","responsibilities",
    "requirements","experience","degree","bachelor","master","proven","strong",
    "ability","ability to","excellent","ability to","preferable","preferred"
}
import re
from rapidfuzz import fuzz

# small stoplist — extend as needed
_JD_STOPWORDS2 = {
    "and","or","the","a","an","this","that","these","those","of","for","to","in",
    "with","on","as","by","be","is","are","was","were","will","would","should",
    "our","you","your","we","their","they","it","its","which","at","from","about",
    "including","such","etc","please","apply","role","position","responsibilities",
    "requirements","experience","degree","bachelor","master","proven","strong",
    "ability","excellent","preferred","preferable","candidate","candidates",
}

_JD_STOPWORDS = _JD_STOPWORDS.union(_JD_STOPWORDS2)

def clean_extracted_skills(raw_tokens, normalize_fn=None,
                           min_len=2, keep_phrases_max_words=6,
                           fuzzy_thresh=88, semantic_dedupe=False, sbert_model=None):
    """
    Clean extractor output -> ordered list of good skill tokens.
    - raw_tokens: list or comma/string
    - normalize_fn: optional normalizer (e.g. normalize_text)
    - fuzzy_thresh: 0-100 for rapidfuzz token-sort dedupe
    - semantic_dedupe: if True and sbert_model provided, perform semantic dedupe
    """
    if isinstance(raw_tokens, str):
        toks = [t.strip() for t in re.split(r'[,;\n/|]', raw_tokens) if t.strip()]
    else:
        toks = [str(t).strip() for t in (raw_tokens or [])]

    norm = normalize_fn if normalize_fn is not None else (lambda s: s.strip().lower())
    cleaned = []
    seen_norm = set()

    def is_techy(tok):
        return bool(re.search(r'[A-Z][a-z]|[\+\#\.]|[0-9]', tok))

    for t in toks:
        if not t:
            continue
        t0 = re.sub(r'\s+', ' ', t).strip()
        t0 = re.sub(r'^[\-\:\•\*\d\.\)\s]+', '', t0)
        if not t0:
            continue
        try:
            n = norm(t0)
        except Exception:
            n = t0.lower().strip()
        if not n:
            continue
        if n in _JD_STOPWORDS or len(n) < min_len:
            continue
        if re.match(r'^[^a-zA-Z0-9]+$', n):
            continue
        # drop single generic verbs/nouns
        if len(n.split()) == 1 and n in {"document","analyze","build","ensure","monitor","design","develop","collaborate","experience","familiarity","work","working"}:
            continue
        if len(n.split()) > keep_phrases_max_words:
            n = " ".join(n.split()[:keep_phrases_max_words])

        # fuzzy dedupe against seen_norm
        best_score = 0
        for s in list(seen_norm):
            score = fuzz.token_sort_ratio(n, s)
            if score > best_score:
                best_score = score
                best = s
        if best_score >= fuzzy_thresh:
            continue

        if n not in seen_norm:
            seen_norm.add(n)
            cleaned.append(n)

    # optional semantic dedupe
    if semantic_dedupe and sbert_model is not None and len(cleaned) > 1:
        try:
            import numpy as np
            embs = sbert_model.encode(cleaned, convert_to_numpy=True)
            norms = np.linalg.norm(embs, axis=1, keepdims=True); norms[norms==0]=1.0
            embs = embs / norms
            keep = []
            keep_embs = []
            for i,e in enumerate(embs):
                if not keep_embs:
                    keep.append(cleaned[i]); keep_embs.append(e); continue
                sims = np.dot(np.vstack(keep_embs), e)
                if float(np.max(sims)) < 0.86:
                    keep.append(cleaned[i]); keep_embs.append(e)
            cleaned = keep
        except Exception:
            pass

    return cleaned

def jd_extract_and_store(jd_text: str, job_dict: dict,
                         use_chroma: bool = True, semantic_dedupe: bool = False,
                         sbert_model=None, chroma_reg=None):
    """
    - jd_text: raw job description
    - job_dict: the job dict in session (will be updated in-place)
    - returns job_dict with 'skills' populated (list)
    """
    # 1) try structured extractor first (if available)
    cands = []
    try:
        if 'extract_skills_from_jd_text' in globals():
            cands = extract_skills_from_jd_text(jd_text, prefer_sectioned=True)
        elif 'generate_candidates' in globals():
            # quick generate over lines
            for ln in jd_text.splitlines()[:60]:
                for g in generate_candidates(ln):
                    if g and g not in cands:
                        cands.append(g)
        else:
            cands = extract_skills_from_text(jd_text)
    except Exception:
        try:
            cands = extract_skills_from_text(jd_text)
        except Exception:
            cands = []

    # 2) clean tokens
    sbert = sbert_model if semantic_dedupe else None
    skills_list = clean_extracted_skills(cands, normalize_fn=normalize_text if 'normalize_text' in globals() else None,
                                         fuzzy_thresh=88, semantic_dedupe=semantic_dedupe, sbert_model=sbert)

    # 3) canonicalize (batch add if chroma available and requested)
    if use_chroma and chroma_reg is not None and skills_list:
        try:
            # reg.add_phrases expects raw phrases; it returns mapping phrase->cid
            add_map = chroma_reg.add_phrases(skills_list, generate_candidates_fn=generate_candidates)
            # convert to canonical ids list
            cids = [add_map.get(s) for s in skills_list if add_map.get(s)]
            job_dict['canonical_ids'] = cids
        except Exception:
            job_dict['canonical_ids'] = job_dict.get('canonical_ids', [])
    else:
        job_dict['canonical_ids'] = job_dict.get('canonical_ids', [])

    job_dict['skills'] = skills_list
    return job_dict

if __name__ == '__main__':
    main()

