import re, json, logging, torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skills-transformers")

JSON_ARRAY_RE = re.compile(r"(\[.*?\])", re.DOTALL)
model_id = "D:\PycharmProjects\Job-Recommender\model\qwen1.7b"


tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=  "cuda", torch_dtype=torch.float16, trust_remote_code=True)
model.eval()


PROMPT2 = """Extract ONLY atomic skill names, Tools, technique, certifications and Libraries from the resume below.
Rules:
1) Each skill must be SHORT (1–3 words).
2) NO sentences or job duties.
3) ONLY proper skill names: software, libraries, tools, techniques, models.
4) REMOVE duplicates.
4) Output ONLY a JSON list of strings (e.g. ["PyTorch","NumPy"]).
6) No extra text or explanation.

### EXAMPLES
Input: "Technical: PyTorch, TensorFlow, data preprocessing and model training in production"
Output: ["PyTorch","TensorFlow","Data preprocessing"]

Input: "Tools: MySQL/MongoDB, Flask/Django, Git"
Output: ["MySQL","MongoDB","Flask","Django","Git"]

Resume:
\"\"\"\n{resume}\n\"\"\"\n
Answer (JSON only):"""

from typing import Tuple, Optional
from transformers import StoppingCriteria, StoppingCriteriaList

import ast
import json
import re
from typing import Optional, Tuple, Any

def extract_list(text: str, debug: bool = False) -> Optional[Any]:
    """
    Robustly extract the first list-like object found *after* a </think> tag.
    If no </think> is present, falls back to the first top-level list in the string.
    Returns:
      - a Python list on success
      - None on failure (or (None, diagnostics) if debug=True)
    Debug info is returned as a dict when debug=True.
    """
    diagnostics = {"original_len": None, "sanitized_len": None, "tag_found": False,
                   "start_idx": None, "end_idx": None, "list_str_preview": None,
                   "ast_ok": False, "json_ok": False, "error": None, "method": None}

    if not isinstance(text, str):
        diagnostics["error"] = "input not a str"
        return (None, diagnostics) if debug else None

    diagnostics["original_len"] = len(text)

    # --- sanitize: remove control chars except \n\r\t, normalize curly quotes ---
    text = "".join(ch for ch in text if (ord(ch) >= 32 or ch in ("\n", "\r", "\t")))
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    diagnostics["sanitized_len"] = len(text)

    # strip outer matching quotes if whole string is quoted
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = text[1:-1]

    # try find </think> (case-insensitive). If found, search after it; else search entire string
    m = re.search(r"</think\s*>", text, flags=re.IGNORECASE)
    search_start = m.end() if m else 0
    diagnostics["tag_found"] = bool(m)

    # helper: find first top-level list starting at pos
    def find_top_level_list(s: str, pos: int = 0) -> Tuple[Optional[int], Optional[int]]:
        start = s.find('[', pos)
        if start == -1:
            return None, None
        depth = 0
        in_string = False
        string_char = None
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if escape:
                escape = False
                continue
            if in_string:
                if ch == '\\':
                    escape = True
                    continue
                if ch == string_char:
                    in_string = False
                    string_char = None
                continue
            else:
                if ch == '"' or ch == "'":
                    in_string = True
                    string_char = ch
                    continue
                if ch == '[':
                    depth += 1
                    continue
                if ch == ']':
                    depth -= 1
                    if depth == 0:
                        return start, i
                    continue
        return None, None

    start, end = find_top_level_list(text, search_start)

    # fallback: if not found after </think>, try from beginning
    if start is None and search_start != 0:
        start, end = find_top_level_list(text, 0)

    if start is None:
        diagnostics["error"] = "no top-level list found"
        return (None, diagnostics) if debug else None

    diagnostics["start_idx"] = start
    diagnostics["end_idx"] = end
    list_str = text[start:end+1]
    diagnostics["list_str_preview"] = (list_str[:200] + "...") if len(list_str) > 200 else list_str

    # Try safe Python literal parsing first
    try:
        parsed = ast.literal_eval(list_str)
        if isinstance(parsed, list):
            diagnostics["ast_ok"] = True
            diagnostics["method"] = "ast"
            return (parsed, diagnostics) if debug else parsed
        else:
            diagnostics["error"] = "ast parsed, result not a list"
    except Exception as e_ast:
        diagnostics["error"] = f"ast failed: {e_ast}"

    # Try JSON (requires double quotes)
    try:
        parsed_json = json.loads(list_str)
        if isinstance(parsed_json, list):
            diagnostics["json_ok"] = True
            diagnostics["method"] = "json"
            return (parsed_json, diagnostics) if debug else parsed_json
        else:
            diagnostics["error"] = "json parsed, result not a list"
    except Exception as e_json:
        diagnostics["error"] = f"json failed: {e_json}"

    # Last-ditch: try to coerce single quotes to double quotes safely for JSON
    # Replace only unescaped single quotes that are list/string delimiters (naive but practical)
    try:
        # avoid touching escaped quotes
        coerced = re.sub(r"(?<!\\)'", '"', list_str)
        parsed_coerced = json.loads(coerced)
        if isinstance(parsed_coerced, list):
            diagnostics["json_ok"] = True
            diagnostics["method"] = "json_coerced"
            diagnostics["note"] = "coerced single->double quotes"
            return (parsed_coerced, diagnostics) if debug else parsed_coerced
        else:
            diagnostics["error"] = "coerced json parsed, result not a list"
    except Exception as e_coerce:
        diagnostics["error"] = f"coerced json failed: {e_coerce}"

    return (None, diagnostics) if debug else None


def extract_skills_transformers(resume_text: str, model_id: str = model_id):
    """
    Minimal extractor that DISABLES 'thinking' by stopping generation after the first ']'.
    Returns: (decoded_generated_text, parsed_json_or_None, filled_offsets_list).
    Reuses global `tok` and `mdl` if present; otherwise loads them simply.
    """
    # Local stopping criterion that halts when a closing bracket appears
    class StopOnFirstClosingBracket(StoppingCriteria):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        def __call__(self, input_ids, scores) -> bool:
            # decode current tokens and stop if we see ']'
            try:
                text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                return text.count("]") >= 1
            except Exception:
                return False

    # reuse globals if available
    try:
        tokenizer = tok  # type: ignore[name-defined]
        # model = mdl      # type: ignore[name-defined]
        global model
    except Exception:
        # minimal load if globals missing
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
         # choose device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            # prefer fp16 on CUDA for speed & memory
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            model.to(device)

        model.eval()

    # prepare prompt and tokenized inputs
    prompt = PROMPT2.format(resume=resume_text)
    # inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(model.device)
    # input_ids = inputs["input_ids"]
    # input_len = input_ids.shape[1]

    messages = [{"role": "user", "content": prompt + "/no_think"}] # + "/no_think"

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)
    print("inputs: ", inputs)

    input_ids = inputs
    input_len = input_ids.shape[1]


    # generate with stopping criterion (stops immediately after first ']')
    stopping = StoppingCriteriaList([StopOnFirstClosingBracket(tokenizer)])
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=1024,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=False,
            # stopping_criteria=stopping,
        )

    decoded = tokenizer.decode(out.sequences[0][input_len:], skip_special_tokens=True)
    decoded = extract_list(decoded)
    print("Generation completed: " , decoded)
    
    return decoded





