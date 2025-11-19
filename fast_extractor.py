from gliner import GLiNER
import torch
import re

class FastSkillExtractor:
    def __init__(self, model_name="urchade/gliner_medium-v2.1"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = GLiNER.from_pretrained(model_name).to(device)
        except Exception as e:
            print(f"GLiNER load failed: {e}")
            self.model = None

        self.labels = [
            "Skill", "Tool", "Technology", "Methodology", 
            "Certification", "Standard", "Framework", "Library"
        ]
        
        # 1. JOB TITLES (Suffixes that indicate a Role)
        self.job_titles = {
            "intern", "consultant", "lead", "manager", "assistant", "associate",
            "expert", "specialist", "engineer", "analyst", "developer", "scientist",
            "director", "executive", "officer", "administrator", "representative",
            "coordinator", "technician", "architect", "auditor"
        }

        # 2. GENERAL NOISE
        self.general_noise = {
            # Academic/Personal
            "gpa", "honors", "bs", "ms", "phd", "bachelor", "master", "degree", 
            "university", "graduate", "resume", "curriculum", "vitae", "education",
            "name", "email", "phone", "address", "profile", "summary", "contact",
            
            # Corporate/Business
            "llc", "inc", "ltd", "pvt", "corp", "corporation", "group", "company",
            "limited", "solutions", "technologies", "services", "firm", "systems",
            
            # Fluff Words
            "text", "data", "accuracy", "pipeline", "documents", "job", "description",
            "candidate", "information", "system", "solution", "environment", "application",
            "project", "projects", "work", "experience", "year", "years", "month",
            "support", "supporting", "labs", "grading", "metrics", "analysis",
            "health", "model", "models", "algorithm", "algorithms", "feature",
            "real-time", "monitoring", "detection", "classification", "prediction",
            "processing", "retrieval", "generation", "segmentation", "concepts",
            "practical", "state-of-the-art", "techniques", "tool", "tools",
            "strength", "bias", "efficiency", "topics", "classify", "improving",
            "human", "first", "response", "app", "ai", "based", "predictions",
            "deployment", "interfaces", "components", "video", "footage", "alert",
            "retraining", "statistical", "insights", "findings", "breaches",
            "activity", "activities", "review", "reviews", "policy", "policies",
            "control", "controls", "workflow", "workflows", "record", "records",
            "practice", "practices", "environment", "environments"
        }

        self.all_noise = self.job_titles.union(self.general_noise)

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
        # 1. Formatting Cleanup
        text = text.strip()
        
        # FIX: "Pe nTesting" (Generic PDF split fix)
        # Looks for "SingleLetter Space SingleLetter" pattern to merge
        # e.g. "P e n T e s t i n g" -> "PenTesting"
        # We loop once to catch consecutive splits
        if re.search(r'\b[A-Za-z]\s[A-Za-z]\b', text):
             text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', text)

        # FIX: "MLO ps" -> "MLOps" (Capital word + space + short lowercase suffix)
        # Matches "MLO ps", "DEV ops" -> Merges them.
        text = re.sub(r'\b([A-Z]{2,})\s+([a-z]{2,3})\b', r'\1\2', text)

        # FIX: "Py Torch" / "My SQL" (Specific Pattern: Short Title + Space + Title)
        # Only merges if first word is exactly 2 chars (Py, My, Go) to avoid merging "Data Science"
        text = re.sub(r'\b([A-Z][a-z])\s+([A-Z][a-z]+)\b', r'\1\2', text)

        # Standard cleanup
        text = re.sub(r'\s+', ' ', text) 
        text = re.sub(r'\s*-\s*', '-', text)
        text = re.sub(r'([A-Z]{2,})\s+s\b', r'\1s', text) # Fix "CNN s"

        # 2. Lowercase for filtering
        text_lower = text.lower()
        tokens = text_lower.split()

        # 3. Filter: Length
        # Drop if too short (<2 chars) or too long (>5 words)
        if len(text) < 2 or len(tokens) > 5:
            return None

        # 4. Filter: Logic Checks
        
        # A. Exact Match Blocklist (e.g., "education", "university")
        if text_lower in self.all_noise:
            return None
            
        # B. Ends with Job Title (e.g., "Security Analyst", "Software Engineer")
        if tokens[-1] in self.job_titles:
            return None

        # C. All Noise Rule (e.g., "Weekly Reports", "Business Requirements")
        if all(t in self.general_noise for t in tokens):
            return None

        # 5. Filter: Bad Starter Words (Expanded based on your Resume 2 output)
        # Kills: "communicate findings", "investigate incidents", "optimizing workflow", "volunteer", "skilled"
        bad_starts = {
            "improving", "supporting", "effectively", "automated", "custom", "advanced", 
            "proficient", "statistical", "resolving", "identifying", "communicating", 
            "mitigating", "streamlining", "automating", "generating", "proven", "existing",
            "investigate", "monitor", "manage", "coordinate", "perform", "conduct",
            "utilize", "business", "weekly", "manual", "communicate", "optimizing",
            "development", "volunteer", "skilled", "maintain", "administer", "gathering",
            "collaborate", "assisting", "responsible", "preparing", "creating"
        }
        
        if tokens[0] in bad_starts:
            return None
            
        # 6. Filter Numbers (e.g., "300+ misconfigs")
        # Allows "2D", "3D", "4G" but kills "300"
        if re.search(r'\d', text) and not re.search(r'(2|3|4)[dD]', text):
             return None

        return text