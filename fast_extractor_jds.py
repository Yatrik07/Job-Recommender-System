from gliner import GLiNER
import torch
import re

class FastJDExtractor:
    def __init__(self, model_name="urchade/gliner_medium-v2.1"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = GLiNER.from_pretrained(model_name).to(device)
        except Exception as e:
            print(f"GLiNER load failed: {e}")
            self.model = None

        # Labels tailored for Requirements
        self.labels = [
            "Skill", "Tool", "Technology", "Methodology", 
            "Certification", "Framework", "Library", "Standard"
        ]
        
        # JD-SPECIFIC NOISE FILTER
        self.jd_noise = {
            # HR / Benefits / Perks
            "salary", "benefits", "compensation", "equity", "bonus", "401k", "stock", "options",
            "dental", "medical", "vision", "insurance", "health", "life", "disability",
            "pto", "paid", "time", "off", "vacation", "holiday", "holidays", "leave", "sick",
            "tuition", "reimbursement", "gym", "membership", "snacks", "meals", "catered",
            "culture", "environment", "flexible", "remote", "hybrid", "onsite", "relocation",
            
            # Legal / EEO Statements
            "equal", "opportunity", "employer", "veteran", "disability", "race", "color",
            "religion", "gender", "sexual", "orientation", "identity", "national", "origin",
            "protected", "status", "accommodation", "applicants", "qualified", "consideration",
            "authorization", "sponsorship", "citizen", "citizenship", "visa", "background", "check",
            
            # Generic Header/Footer Words
            "responsibilities", "requirements", "qualifications", "preferred", "plus", "advantage",
            "description", "summary", "role", "position", "location", "type", "duration",
            "contract", "full-time", "part-time", "internship", "willingness", "travel",
            "job", "work", "experience", "year", "years", "degree", "bachelor", "master", "phd",
            "computer", "science", "engineering", "field", "related", "equivalent",
            "knowledge", "understanding", "proficiency", "expertise", "ability", "skills",
            "strong", "proven", "track", "record", "excellent", "good", "demonstrated",
            "communication", "teamwork", "collaboration", "leadership", "problem-solving"
        }
        
        self.job_titles = {
            "intern", "consultant", "lead", "manager", "assistant", "associate",
            "expert", "specialist", "engineer", "analyst", "developer", "scientist",
            "director", "executive", "officer", "administrator", "representative"
        }

    def extract(self, text: str):
        if not self.model or not text:
            return []

        chunk_size = 3000
        target_text = text[:chunk_size]
        chunks = [c.strip() for c in target_text.split('\n') if len(c.split()) > 3]

        final_skills = set()

        for chunk in chunks:
            entities = self.model.predict_entities(chunk, self.labels, threshold=0.35)
            for entity in entities:
                cleaned = self._clean_skill(entity["text"])
                if cleaned:
                    final_skills.add(cleaned)

        return list(final_skills)

    def _clean_skill(self, text):
        text = text.strip()
        
        # Standard Cleaning
        if re.search(r'\b[A-Za-z]\s[A-Za-z]\b', text): 
             text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', text)
        
        text = re.sub(r'\b([A-Z]{2,})\s+([a-z]{2,3})\b', r'\1\2', text) 
        text = re.sub(r'\b([A-Z][a-z])\s+([A-Z][a-z]+)\b', r'\1\2', text) 
        text = re.sub(r'\s+', ' ', text) 
        text = re.sub(r'\s*-\s*', '-', text)
        text = re.sub(r'([A-Z]{2,})\s+s\b', r'\1s', text) 

        text_lower = text.lower()
        tokens = text_lower.split()

        if len(text) < 2 or len(tokens) > 5:
            return None

        # NOISE CHECKS
        if text_lower in self.jd_noise:
            return None
        
        # Strict check for benefits terms
        strict_noise = {"insurance", "benefits", "401k", "dental", "medical", "vision", "vacation", "pto", "gender", "race", "disability", "veteran", "equal", "employer"}
        if any(t in strict_noise for t in tokens):
             return None

        if tokens[-1] in self.job_titles:
            return None
            
        # Bad Starts for JDs
        bad_starts = {
            "proven", "demonstrated", "strong", "excellent", "good", "proficient", 
            "ability", "willingness", "must", "should", "nice", "have", "knowledge",
            "experience", "familiarity", "understanding", "expertise", "responsible",
            "participate", "collaborate", "assist", "support", "manage", "lead"
        }
        if tokens[0] in bad_starts:
            return None
            
        if re.search(r'\d', text) and not re.search(r'(2|3|4)[dD]', text):
             return None

        return text