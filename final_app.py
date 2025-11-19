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

try:
    from fast_extractor import FastSkillExtractor   # Resume Skills
    from fast_extractor_jds import FastJDExtractor  # JD Skills
    from analysis_utils import analyze_job_clusters # Clustering
    from segmentation import TextSegmenter          # Section Splitter
except ImportError as e:
    st.error(f"CRITICAL: Missing modules. Ensure all helper files are in the folder. Error: {e}")
    st.stop()

# --- 2. CACHED AI MODELS ---
@st.cache_resource
def load_models():
    with st.status("🚀 Loading AI Models...", expanded=True) as status:
        st.write("Loading Extractors...")
        res_extractor = FastSkillExtractor(model_name="urchade/gliner_medium-v2.1")
        jd_extractor = FastJDExtractor(model_name="urchade/gliner_medium-v2.1")
        
        st.write("Loading Ranker...")
        ranker = SentenceTransformer("all-MiniLM-L6-v2")
        
        st.write("Loading Segmenter...")
        segmenter = TextSegmenter()
        
        status.update(label="System Ready!", state="complete", expanded=False)
    return res_extractor, jd_extractor, ranker, segmenter

res_extractor, jd_extractor, ranker, segmenter = load_models()

# --- 3. SESSION STATE ---
if 'jobs' not in st.session_state:
    st.session_state['jobs'] = []
if 'resume' not in st.session_state:
    st.session_state['resume'] = {}

# --- 4. CORE FUNCTIONS ---

def parse_resume_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def process_jds(jd_text_input):
    raw_jds = jd_text_input.split("===")
    processed = []
    
    my_bar = st.progress(0, text="Analyzing JDs...")
    
    for i, jd in enumerate(raw_jds):
        if len(jd.strip()) < 10: continue 
        
        # 1. Segment the JD
        sections = segmenter.parse_jd(jd)
        
        # 2. Extract Skills ONLY from Requirements (cleaner signal)
        extract_source = sections['requirements'] if len(sections['requirements']) > 50 else jd
        skills = jd_extractor.extract(extract_source)
        
        lines = jd.strip().split('\n')
        title = lines[0][:60] if lines else f"Job #{len(st.session_state['jobs'])+i+1}"
        
        processed.append({
            "id": f"job_{int(time.time())}_{i}",
            "title": title,
            "text": jd,
            "responsibilities": sections['responsibilities'], 
            "requirements_text": sections['requirements'],    
            "skills": skills                                  
        })
        my_bar.progress((i + 1) / len(raw_jds))
        
    my_bar.empty()
    return processed

def extract_common_context(text1, text2, top_n=5):
    if not text1 or not text2 or len(text1) < 10 or len(text2) < 10:
        return []
    try:
        vec = CountVectorizer(ngram_range=(2, 3), stop_words='english', min_df=1)
        X = vec.fit_transform([text1, text2])
        features = vec.get_feature_names_out()
        
        res_counts = X[0].toarray().flatten()
        jd_counts = X[1].toarray().flatten()
        
        common_indices = np.where((res_counts > 0) & (jd_counts > 0))[0]
        common_phrases = [features[i] for i in common_indices]
        
        unique = sorted(list(set(common_phrases)), key=len, reverse=True)
        final = []
        for p in unique:
            if not any(p in other for other in final):
                final.append(p)
        return final[:top_n]
    except ValueError:
        return []

def calculate_scores(resume_data, jobs_data):
    if not resume_data.get('skills') or not jobs_data:
        return []

    # 1. Embed Resume Skills
    res_skill_embeddings = ranker.encode(resume_data['skills'])
    
    # 2. Embed Resume EXPERIENCE
    res_exp_text = resume_data.get('experience', '')
    if len(res_exp_text) < 50: 
        res_exp_text = resume_data.get('text', '')[:2000]
        
    res_exp_embedding = ranker.encode(res_exp_text)
    
    scored_results = []

    for job in jobs_data:
        jd_skills = job.get('skills', [])
        
        skill_ai_score = 0.0
        exp_score = 0.0
        missing_skills = []
        common_themes = []

        # --- SIGNAL 1: SKILL MATCH ---
        if jd_skills:
            jd_skill_embeddings = ranker.encode(jd_skills)
            sim_matrix = cosine_similarity(jd_skill_embeddings, res_skill_embeddings)
            max_matches = np.max(sim_matrix, axis=1)
            skill_ai_score = float(np.mean(max_matches))
            
            missing_indices = np.where(max_matches < 0.6)[0]
            missing_skills = [jd_skills[i] for i in missing_indices]

        # --- SIGNAL 2: EXPERIENCE MATCH ---
        jd_resp_text = job.get('responsibilities', '')
        if len(jd_resp_text) < 50: 
            jd_resp_text = job.get('text', '')[:2000]
            
        jd_resp_embedding = ranker.encode(jd_resp_text)
        
        exp_score = float(cosine_similarity(
            res_exp_embedding.reshape(1, -1), 
            jd_resp_embedding.reshape(1, -1)
        )[0][0])
        exp_score = max(0.0, exp_score)
        
        common_themes = extract_common_context(res_exp_text, jd_resp_text)

        final_score = (skill_ai_score * 0.7) + (exp_score * 0.3)

        scored_results.append({
            **job,
            "ai_score": final_score,
            "skill_match": skill_ai_score,
            "context_match": exp_score, 
            "missing": missing_skills,
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
            with st.spinner("Parsing & Segmenting Resume..."):
                full_text = parse_resume_pdf(uploaded_file)
                sections = segmenter.parse_resume(full_text)
                skills = res_extractor.extract(full_text)
                
                st.session_state['resume'] = {
                    "text": full_text,
                    "experience": sections['experience'],
                    "skills": skills, 
                    "name": uploaded_file.name
                }
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
                    st.success(f"Added {len(new_jobs)} Jobs")

    if st.button("Clear All Jobs"):
        st.session_state['jobs'] = []
        st.rerun()

# MAIN DASHBOARD
st.title("📊 Job Match Dashboard")

if st.session_state.get('resume') and st.session_state['resume'].get('skills'):
    with st.expander(f"👤 **Candidate:** {st.session_state['resume']['name']}", expanded=False):
        st.caption("Extracted Skills:")
        st.markdown(" ".join([f"`{s}`" for s in st.session_state['resume']['skills']]))
        if st.session_state['resume'].get('experience'):
            st.success("✅ 'Experience' section successfully isolated for semantic matching.")
else:
    st.info("👈 Please upload a resume to start.")

if st.session_state['jobs'] and st.session_state.get('resume', {}).get('skills'):
    st.divider()
    st.subheader("🏆 Ranked Recommendations")
    
    ranked_jobs = calculate_scores(st.session_state['resume'], st.session_state['jobs'])
    
    use_clusters = st.checkbox("⚡ Group similar jobs (Unsupervised AI)")
    
    if use_clusters and len(ranked_jobs) > 3:
        with st.spinner("Detecting job themes..."):
            ranked_jobs = analyze_job_clusters(ranked_jobs, ranker, n_clusters=3)
        
        df_clusters = pd.DataFrame(ranked_jobs)
        if 'cluster_name' in df_clusters.columns:
            st.success("AI detected these job categories:")
            unique_clusters = df_clusters['cluster_name'].unique()
            for cluster in unique_clusters:
                with st.expander(f"📂 Category: {cluster}", expanded=True):
                    cluster_jobs = [j for j in ranked_jobs if j.get('cluster_name') == cluster]
                    for job in cluster_jobs:
                        col1, col2 = st.columns([1, 4])
                        col1.progress(job['ai_score'], text=f"{job['ai_score']:.0%}")
                        col2.markdown(f"**{job['title']}**")
                        st.divider()
    else:
        for rank, job in enumerate(ranked_jobs):
            with st.container():
                col_score, col_details = st.columns([1, 4])
                
                with col_score:
                    st.write(f"### Rank #{rank+1}")
                    st.progress(job['ai_score'], text=f"Total: {job['ai_score']:.1%}")
                    st.caption(f"Skills: {job['skill_match']:.1%} | Exp: {job['context_match']:.1%}")
                
                with col_details:
                    st.markdown(f"#### {job['title']}")
                    
                    if job['common_themes']:
                        st.info(f"🧩 **Experience Match:** {', '.join(job['common_themes'])}")

                    if job['missing']:
                        st.markdown(f"⚠️ **Missing Skills:** {', '.join(job['missing'][:7])} " + (f"...and {len(job['missing'])-7} more" if len(job['missing'])>7 else ""))
                    else:
                        st.success("✅ Perfect Skill Match!")
                    
                    with st.expander("🔍 View Segmentation Analysis (Debug)"):
                        st.markdown("### 1. Extracted Requirements (for Skill Matching)")
                        st.caption("The system extracted skills from this section:")
                        st.text(job.get('requirements_text', 'N/A')[:800] + "...")
                        
                        st.markdown("### 2. Extracted Responsibilities (for Experience Matching)")
                        st.caption("The system compared your resume history against this narrative:")
                        st.text(job.get('responsibilities', 'N/A')[:800] + "...")
                        
                        st.markdown("### 3. Extracted Skills List")
                        st.markdown(", ".join([f"`{s}`" for s in job['skills']]))
                
                st.divider()
            
    st.subheader("📥 Export Data")
    df_export = pd.DataFrame(ranked_jobs)
    if not df_export.empty:
        # ADDED: responsibilities and requirements_text columns
        export_cols = ['title', 'ai_score', 'skill_match', 'context_match', 'missing', 'skills', 'responsibilities', 'requirements_text']
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