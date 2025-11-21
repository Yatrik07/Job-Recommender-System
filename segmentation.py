import re

class TextSegmenter:
    def __init__(self):
        # --- RESUME HEADERS ---
        self.res_exp_keywords = [
            r'work\s+experience', r'professional\s+experience', r'employment\s+history', 
            r'work\s+history', r'experience', r'employment', r'experience\s*:'
        ]
        self.res_stop_keywords = [
            r'education', r'skills', r'technical\s+skills', r'projects', 
            r'certifications', r'awards', r'languages', r'references', r'achievement'
        ]

        # --- JD HEADERS (Responsibilities) ---
        # Updated regex to handle both straight (') and curly (’) apostrophes
        self.jd_resp_keywords = [
            r'responsibilities', r'what\s+you\s+will\s+do', r'duties', 
            r'role\s+description', r'key\s+responsibilities', r'job\s+description',
            r'what\s+you[\'’]ll\s+do', r'your\s+role', r'position\s+summary', 
            r'about\s+the\s+job', r'what\s+will\s+you\s+do',
            r'what\s+you\s+can\s+expect', r'project\s+overview', r'role\s+summary'
        ]
        
        # --- JD HEADERS (Requirements) ---
        self.jd_req_keywords = [
            r'requirements', r'qualifications', r'what\s+we\s+are\s+looking\s+for', 
            r'who\s+you\s+are', r'skills', r'minimum\s+qualifications', r'preferred\s+qualifications',
            r'expertise', r'what\s+you\s+need', r'what\s+you\s+bring', r'candidate\s+profile',
            r'basic\s+qualifications', r'required\s+skills', r'minimum\s+requirements',
            r'what\s+we[\'’]re\s+looking\s+for', r'what\s+you\s+should\s+have'
        ]

    def extract_section(self, text, start_patterns, stop_patterns):
        if not text: return ""
        
        text_lower = text.lower()
        best_start_idx = -1
        content_start = -1
        
        # 1. Find FIRST occurrence of ANY start pattern
        for pattern in start_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if best_start_idx == -1 or match.start() < best_start_idx:
                     best_start_idx = match.start()
                     content_start = match.end()

        if best_start_idx == -1:
            return "" 

        # 2. Find NEAREST stop pattern
        stop_idx = len(text)
        search_text = text_lower[content_start:]
        nearest_stop_dist = len(search_text)
        
        for pattern in stop_patterns:
            match = re.search(pattern, search_text)
            if match:
                if match.start() < nearest_stop_dist:
                     nearest_stop_dist = match.start()
                     
        stop_idx = content_start + nearest_stop_dist
        return text[content_start:stop_idx].strip()

    def parse_resume(self, text):
        return {
            "experience": self.extract_section(text, self.res_exp_keywords, self.res_stop_keywords),
            "full_text": text
        }

    def parse_jd(self, text):
        # Responsibilities: Stop at Requirements OR Benefits
        # Added 'what we offer' to catch ending blocks
        resp_stop_patterns = self.jd_req_keywords + [
            r'benefits', r'what\s+we\s+offer', r'compensation', r'salary', r'about\s+us'
        ]
        
        resp = self.extract_section(text, self.jd_resp_keywords, resp_stop_patterns)
        
        # Requirements: Stop at Benefits OR EEO/Legal statements
        req_stop_patterns = [
            r'benefits', r'compensation', r'salary', r'about\s+us', 
            r'equal\s+employment', r'disclaimer', r'eeo\s+statement', 
            r'what\s+we\s+offer', r'why\s+this\s+is\s+different'
        ]
        
import re

class TextSegmenter:
    def __init__(self):
        # --- RESUME HEADERS ---
        self.res_exp_keywords = [
            r'work\s+experience', r'professional\s+experience', r'employment\s+history', 
            r'work\s+history', r'experience', r'employment', r'experience\s*:'
        ]
        self.res_stop_keywords = [
            r'education', r'skills', r'technical\s+skills', r'projects', 
            r'certifications', r'awards', r'languages', r'references', r'achievement'
        ]

        # --- JD HEADERS (Responsibilities) ---
        # Updated regex to handle both straight (') and curly (’) apostrophes
        self.jd_resp_keywords = [
            r'responsibilities', r'what\s+you\s+will\s+do', r'duties', 
            r'role\s+description', r'key\s+responsibilities', r'job\s+description',
            r'what\s+you[\'’]ll\s+do', r'your\s+role', r'position\s+summary', 
            r'about\s+the\s+job', r'what\s+will\s+you\s+do',
            r'what\s+you\s+can\s+expect', r'project\s+overview', r'role\s+summary'
        ]
        
        # --- JD HEADERS (Requirements) ---
        self.jd_req_keywords = [
            r'requirements', r'qualifications', r'what\s+we\s+are\s+looking\s+for', 
            r'who\s+you\s+are', r'skills', r'minimum\s+qualifications', r'preferred\s+qualifications',
            r'expertise', r'what\s+you\s+need', r'what\s+you\s+bring', r'candidate\s+profile',
            r'basic\s+qualifications', r'required\s+skills', r'minimum\s+requirements',
            r'what\s+we[\'’]re\s+looking\s+for', r'what\s+you\s+should\s+have'
        ]

    def extract_section(self, text, start_patterns, stop_patterns):
        if not text: return ""
        
        text_lower = text.lower()
        best_start_idx = -1
        content_start = -1
        
        # 1. Find FIRST occurrence of ANY start pattern
        for pattern in start_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if best_start_idx == -1 or match.start() < best_start_idx:
                     best_start_idx = match.start()
                     content_start = match.end()

        if best_start_idx == -1:
            return "" 

        # 2. Find NEAREST stop pattern
        stop_idx = len(text)
        search_text = text_lower[content_start:]
        nearest_stop_dist = len(search_text)
        
        for pattern in stop_patterns:
            match = re.search(pattern, search_text)
            if match:
                if match.start() < nearest_stop_dist:
                     nearest_stop_dist = match.start()
                     
        stop_idx = content_start + nearest_stop_dist
        return text[content_start:stop_idx].strip()

    def parse_resume(self, text):
        return {
            "experience": self.extract_section(text, self.res_exp_keywords, self.res_stop_keywords),
            "full_text": text
        }

    def parse_jd(self, text):
        # Responsibilities: Stop at Requirements OR Benefits
        # Added 'what we offer' to catch ending blocks
        resp_stop_patterns = self.jd_req_keywords + [
            r'benefits', r'what\s+we\s+offer', r'compensation', r'salary', r'about\s+us'
        ]
        
        resp = self.extract_section(text, self.jd_resp_keywords, resp_stop_patterns)
        
        # Requirements: Stop at Benefits OR EEO/Legal statements
        req_stop_patterns = [
            r'benefits', r'compensation', r'salary', r'about\s+us', 
            r'equal\s+employment', r'disclaimer', r'eeo\s+statement', 
            r'what\s+we\s+offer', r'why\s+this\s+is\s+different'
        ]
        
        reqs = self.extract_section(text, self.jd_req_keywords, req_stop_patterns)
        
        # Fallback logic
        if len(resp) < 50: resp = text 
        if len(reqs) < 50: reqs = text
            
        return {
            "responsibilities": resp,
            "requirements": reqs,
            "full_text": text
        }

    def load_llm(self):
        if hasattr(self, 'llm_model') and self.llm_model:
            return

        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_path = os.path.join(os.path.dirname(__file__), "model", "qwen1.7b")
        print(f"Loading LLM from {model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            print("LLM Loaded Successfully.")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            self.llm_model = None

    def parse_jd_llm(self, text):
        self.load_llm()
        if not self.llm_model:
            return self.parse_jd(text) # Fallback

        prompt = f"""You are a helpful assistant. Extract the 'Responsibilities' and 'Requirements' sections from the following Job Description. 
        Return the output in this exact format:
        RESPONSIBILITIES: <content>
        REQUIREMENTS: <content>

        Job Description:
        {text[:3500]}
        
        Output:"""

        messages = [
            {"role": "system", "content": "You are a precise extractor."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature=0.1
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse the response
        resp_match = re.search(r'RESPONSIBILITIES:\s*(.*?)(?=REQUIREMENTS:|$)', response, re.DOTALL | re.IGNORECASE)
        req_match = re.search(r'REQUIREMENTS:\s*(.*?)$', response, re.DOTALL | re.IGNORECASE)
        
        responsibilities = resp_match.group(1).strip() if resp_match else ""
        requirements = req_match.group(1).strip() if req_match else ""
        
        # Fallback if LLM fails to follow format
        if len(responsibilities) < 10 and len(requirements) < 10:
             return self.parse_jd(text)

        return {
            "responsibilities": responsibilities,
            "requirements": requirements,
            "full_text": text
        }