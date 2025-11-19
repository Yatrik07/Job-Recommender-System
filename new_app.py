import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import time

# --- 1. SETUP & IMPORTS ---
st.set_page_config(page_title="Universal Job Matcher", layout="wide", page_icon="⚡")

# Import Modules
try:
    from fast_extractor import FastSkillExtractor   # For Resumes
    from fast_extractor_jds import FastJDExtractor  # For JDs
    from analysis_utils import analyze_job_clusters # For Clustering
except ImportError as e:
    st.error(f"CRITICAL: Missing modules. Ensure 'fast_extractor.py', 'fast_extractor_jds.py' and 'analysis_utils.py' are in the folder. Error: {e}")
    st.stop()

# --- 2. CACHED AI MODELS ---
@st.cache_resource
def load_models():
    with st.status("🚀 Loading AI Models...", expanded=True) as status:
        st.write("Loading Resume Extractor...")
        res_extractor = FastSkillExtractor(model_name="urchade/gliner_medium-v2.1")
        
        st.write("Loading JD Extractor...")
        jd_extractor = FastJDExtractor(model_name="urchade/gliner_medium-v2.1")
        
        st.write("Loading Semantic Ranker...")
        ranker = SentenceTransformer("all-MiniLM-L6-v2")
        
        status.update(label="All Models Ready!", state="complete", expanded=False)
    return res_extractor, jd_extractor, ranker

res_extractor, jd_extractor, ranker = load_models()

# --- 3. SESSION STATE ---
if 'jobs' not in st.session_state:
    st.session_state['jobs'] = []
if 'resume' not in st.session_state:
    st.session_state['resume'] = {"text": "", "skills": [], "name": "Candidate"}

# --- 4. CORE FUNCTIONS ---

def parse_resume_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def process_jds(jd_text_input):
    """Uses the JD Extractor (Stricter noise filtering)"""
    raw_jds = jd_text_input.split("===")
    processed = []
    
    my_bar = st.progress(0, text="Extracting requirements from JDs...")
    
    for i, jd in enumerate(raw_jds):
        if len(jd.strip()) < 10: continue 
        
        skills = jd_extractor.extract(jd)
        
        lines = jd.strip().split('\n')
        title = lines[0][:60] if lines else f"Job #{len(st.session_state['jobs'])+i+1}"
        
        processed.append({
            "id": f"job_{int(time.time())}_{i}",
            "title": title,
            "text": jd,
            "skills": skills
        })
        my_bar.progress((i + 1) / len(raw_jds))
        
    my_bar.empty()
    return processed

def extract_common_context(text1, text2, top_n=5):
    """Finds shared phrases for Context Visualization"""
    if not text1 or not text2 or len(text1) < 10 or len(text2) < 10:
        return []

    corpus = [text1, text2]
    try:
        vec = CountVectorizer(ngram_range=(2, 3), stop_words='english', min_df=1)
        X = vec.fit_transform(corpus)
        features = vec.get_feature_names_out()
        
        res_counts = X[0].toarray().flatten()
        jd_counts = X[1].toarray().flatten()
        
        common_indices = np.where((res_counts > 0) & (jd_counts > 0))[0]
        common_phrases = [features[i] for i in common_indices]
        
        unique_phrases = []
        for p in sorted(common_phrases, key=len, reverse=True):
            if not any(p in other for other in unique_phrases):
                unique_phrases.append(p)
                
        return unique_phrases[:top_n]
    except ValueError:
        return []

def calculate_scores(resume_data, jobs_data):
    if not resume_data['skills'] or not jobs_data:
        return []

    res_skill_embeddings = ranker.encode(resume_data['skills'])
    res_text = resume_data.get('text', '')
    res_text_trunc = res_text[:2000] if len(res_text) > 2000 else res_text
    res_text_embedding = ranker.encode(res_text_trunc)
    
    scored_results = []

    for job in jobs_data:
        jd_skills = job.get('skills', [])
        jd_text = job.get('text', '')
        
        # INITIALIZE DEFAULTS (Fixes KeyError)
        skill_ai_score = 0.0
        context_score = 0.0
        keyword_score = 0.0
        missing_skills = []
        match_count = 0
        total_reqs = 0
        common_themes = []

        # 1. Skill Semantic Match
        if jd_skills:
            jd_skill_embeddings = ranker.encode(jd_skills)
            sim_matrix = cosine_similarity(jd_skill_embeddings, res_skill_embeddings)
            max_matches = np.max(sim_matrix, axis=1)
            skill_ai_score = float(np.mean(max_matches))
            
            missing_indices = np.where(max_matches < 0.6)[0]
            missing_skills = [jd_skills[i] for i in missing_indices]
            
            r_set = set([s.lower() for s in resume_data['skills']])
            j_set = set([s.lower() for s in jd_skills])
            overlap = len(r_set.intersection(j_set))
            keyword_score = overlap / len(j_set) if len(j_set) > 0 else 0
            
            total_reqs = len(j_set)
            match_count = total_reqs - len(missing_skills)

        # 2. Context Match
        job_text_trunc = jd_text[:2000] if len(jd_text) > 2000 else jd_text
        if job_text_trunc.strip():
            jd_text_embedding = ranker.encode(job_text_trunc)
            context_score = float(cosine_similarity(
                res_text_embedding.reshape(1, -1), 
                jd_text_embedding.reshape(1, -1)
            )[0][0])
            context_score = max(0.0, context_score)
            common_themes = extract_common_context(res_text, jd_text)

        # Final Weighted Score
        final_score = (skill_ai_score * 0.7) + (context_score * 0.3)

        scored_results.append({
            **job,
            "ai_score": final_score,
            "skill_match": skill_ai_score,
            "context_match": context_score,
            "keyword_score": keyword_score,
            "missing": missing_skills,
            "match_count": match_count,
            "total_reqs": total_reqs,
            "common_themes": common_themes
        })

    return sorted(scored_results, key=lambda x: x['ai_score'], reverse=True)

# --- 5. UI LAYOUT ---

with st.sidebar:
    st.header("1. Candidate Profile")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    
    if uploaded_file:
        file_hash = hash(uploaded_file.getvalue())
        if st.session_state.get('resume_hash') != file_hash:
            with st.spinner("Analyzing Resume..."):
                text = parse_resume_pdf(uploaded_file)
                skills = res_extractor.extract(text)
                st.session_state['resume'] = {"text": text, "skills": skills, "name": uploaded_file.name}
                st.session_state['resume_hash'] = file_hash
            st.success(f"Extracted {len(skills)} skills")

    st.divider()
    
    st.header("2. Job Descriptions")
    tab1, tab2 = st.tabs(["Paste", "Upload"])
    
    with tab1:
        jd_paste = st.text_area("Paste JDs (Split with '===')", height=200)
        if st.button("Process Pasted JDs"):
            if jd_paste:
                new_jobs = process_jds(jd_paste)
                st.session_state['jobs'].extend(new_jobs)
                st.success(f"Added {len(new_jobs)} Jobs")
            else:
                st.warning("Paste text first.")
                
    with tab2:
        uploaded_csv = st.file_uploader("Upload CSV (Must have 'description' col)", type="csv")
        if uploaded_csv:
             if st.button("Process CSV"):
                df = pd.read_csv(uploaded_csv)
                if 'description' in df.columns:
                    texts = df['description'].astype(str).tolist()[:50]
                    batch_text = "\n===\n".join(texts)
                    new_jobs = process_jds(batch_text)
                    st.session_state['jobs'].extend(new_jobs)
                    st.success(f"Added {len(new_jobs)} Jobs from CSV")

    if st.button("Clear All Jobs"):
        st.session_state['jobs'] = []
        st.rerun()

# MAIN DASHBOARD
st.title("📊 Job Match Dashboard")

if st.session_state['resume']['skills']:
    with st.expander(f"👤 **Candidate Profile:** {st.session_state['resume']['name']} ({len(st.session_state['resume']['skills'])} Skills)", expanded=False):
        st.markdown(" ".join([f"`{s}`" for s in st.session_state['resume']['skills']]))
else:
    st.info("👈 Please upload a resume to start.")

if st.session_state['jobs'] and st.session_state['resume']['skills']:
    st.divider()
    st.subheader("🏆 Ranked Recommendations")
    
    ranked_jobs = calculate_scores(st.session_state['resume'], st.session_state['jobs'])
    
    # CLUSTERING FEATURE (Unsupervised)
    use_clusters = st.checkbox("⚡ Group similar jobs (Unsupervised AI)")
    
    if use_clusters and len(ranked_jobs) > 3:
        with st.spinner("Detecting job themes..."):
            # OLD LINE:
            # ranked_jobs = analyze_job_clusters(ranked_jobs, n_clusters=3)
            
            # NEW LINE (Pass the ranker model):
            ranked_jobs = analyze_job_clusters(ranked_jobs, ranker, n_clusters=3)
        
        df_clusters = pd.DataFrame(ranked_jobs)
        if 'cluster_name' in df_clusters.columns:
            st.success("AI detected these job categories:")
            unique_clusters = df_clusters['cluster_name'].unique()
            for ind, cluster in enumerate(unique_clusters):
                with st.expander(f"📂 Category: {ind}", expanded=True):
                    cluster_jobs = [j for j in ranked_jobs if j.get('cluster_name') == cluster]
                    for job in cluster_jobs:
                        col1, col2 = st.columns([1, 4])
                        col1.progress(job['ai_score'], text=f"{job['ai_score']:.0%}")
                        col2.markdown(f"**{job['title']}**")
                        if job['missing']:
                            col2.caption(f"Missing: {', '.join(job['missing'][:3])}")
                        st.divider()
    else:
        # STANDARD LIST VIEW
        for rank, job in enumerate(ranked_jobs):
            with st.container():
                col_score, col_details = st.columns([1, 4])
                
                with col_score:
                    st.write(f"### Rank #{rank+1}")
                    st.progress(job['ai_score'], text=f"AI Match: {job['ai_score']:.1%}")
                    st.caption(f"Keyword Match: {job['keyword_score']:.1%}")
                
                with col_details:
                    st.markdown(f"#### {job['title']}")
                    
                    if job['common_themes']:
                        st.info(f"🧩 **Context Overlap:** {', '.join(job['common_themes'])}")

                    if job['missing']:
                        st.markdown(f"⚠️ **Missing Skills:** {', '.join(job['missing'][:7])} " + (f"...and {len(job['missing'])-7} more" if len(job['missing'])>7 else ""))
                    else:
                        st.success("✅ Perfect Skill Match!")
                    
                    with st.expander("View Job Description"):
                        st.markdown("**Extracted:** " + ", ".join([f"`{s}`" for s in job['skills']]))
                        st.markdown("---")
                        st.text(job['text'])
                
                st.divider()
            
    st.subheader("📥 Export Data")
    df_export = pd.DataFrame(ranked_jobs)
    if not df_export.empty:
        export_cols = ['title', 'ai_score', 'keyword_score', 'match_count', 'total_reqs', 'missing', 'skills']
        # Ensure columns exist before selecting
        valid_cols = [c for c in export_cols if c in df_export.columns]
        df_export = df_export[valid_cols]
        
        if 'missing' in df_export.columns:
            df_export['missing'] = df_export['missing'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        if 'skills' in df_export.columns:
            df_export['skills'] = df_export['skills'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        
        st.download_button(
            label="Download Ranking CSV",
            data=df_export.to_csv(index=False).encode('utf-8'),
            file_name="thesis_job_rankings.csv",
            mime="text/csv"
        )