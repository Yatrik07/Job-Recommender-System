
# Job Recommender System Based on Skills

### An Intelligent, LLM-Powered Engine for Matching People to the Right Jobs

Finding the right job shouldn’t feel like searching for a needle in a haystack — and yet, it often does.
This project aims to fix that.

The **Job Recommender System** uses modern NLP, semantic understanding, and a hybrid AI pipeline to match resumes and job descriptions in a way that feels *much closer to how a human would evaluate a profile*, but with the speed and scale of automation.

This system doesn’t just scan for keywords.
It understands skills, context, synonyms, responsibilities, and the actual meaning behind the text.

---

## What This System Can Do

### **Extract Skills (Even the Hidden Ones)**

* Reads resumes and job descriptions using a combination of:

  * **Qwen LLM** for deep, contextual understanding
  * **GLiNER NER** for extremely accurate technical skill identification
* Captures implicit skills mentioned through experience or projects, not just explicit keywords.

### **Normalize Skills Automatically**

Different people write skills differently.
“Deep Learning,” “DL,” and “Neural Nets” → all mean the same thing.

The system automatically cleans, standardizes, and organizes these into a canonical skill set, making comparisons accurate and fair.

### **Understand Text Semantically**

Using **Sentence Transformers**, the system converts resumes and job descriptions into dense embeddings that capture meaning — not just words.
This lets the system compare candidates and roles the way a human would understand similarities.

### **Rank Jobs Based on True Match Quality**

Every job recommendation comes with:

* Overall similarity score
* Matching skills
* Missing skills (gap analysis)
* Experience–responsibility alignment
* A clear, ranked list of opportunities

### **Simple, Clean Web Interface**

Built with Streamlit — just upload your resume (PDF), and the system does the rest.

---

## How the System Works (High-Level Overview)

```
Resume PDF
    ↓
Resume Parsing → Skill Extraction → Skill Normalization
    ↓
Embedding Generation → Similarity Scoring
    ↓
Ranked Job Recommendations
```

**Resume Parsing**
Reads resume PDFs, cleans the text, identifies sections like Skills, Experience, Projects, and prepares them for extraction.

**Job Description Processing**
Breaks down job descriptions into Requirements and Responsibilities, ensuring structured comparison.

**Hybrid Skill Extraction (LLM + NER)**
Provides both contextual reasoning and token-level accuracy — best of both worlds.

**Semantic Embeddings**
Transforms skills + full documents into meaningful vector representations.

**Ranking Engine**
Score = weighted combination of skills match + responsibilities match → sorted recommendations.

---

## Repository Structure

```
├── analysis_utils.py              # Core utility logic for JD/resume processing
├── fast_extractor.py              # Fast GLiNER-based skill extractor (resumes)
├── fast_extractor_jds.py          # GLiNER extractor tailored for job descriptions
├── segmentation.py                # Logic for splitting documents into sections
├── final_app.py                   # Streamlit application (main entry point)
├── requirements.txt               # Project dependencies
└── README.md
```

---

## Tech Stack

### Languages & Frameworks

* **Python**
* **Streamlit** (Frontend UI)

### AI / NLP Models

* **Qwen** (contextual extraction)
* **GLiNER** (NER-based skill identification)
* **Sentence Transformers** (semantic embeddings)

### Libraries & Tools

* PyPDF2, pdfplumber
* spaCy, NLTK
* pandas, NumPy
* scikit-learn

---

## Getting Started (Run Locally)

### 1. Clone the repository

```bash
git clone https://github.com/Yatrik07/Job-Recommender-System
cd Job-Recommender-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run final_app.py
```

### 4. Upload your resume

The system will instantly:

* Extract skills
* Compare them with job descriptions
* Generate ranked recommendations

---

## 📌 Final Notes

This project was built with the goal of creating a smarter, more human-like way to match job seekers with relevant opportunities. Instead of relying on outdated keyword matching, it focuses on understanding what candidates actually know and what jobs genuinely require.

If you are exploring AI applications in recruitment, NLP-based document analysis, or semantic similarity systems, this repository should give you a solid, practical foundation to build on. Feel free to fork, experiment, and adapt it to your own workflow or datasets.

If you have any suggestions, ideas, or improvements, I’d love to hear them - feel free to reach out or open a discussion.

Thanks for checking out the project! 😊

