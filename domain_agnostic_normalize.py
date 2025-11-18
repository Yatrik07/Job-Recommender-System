# domain_agnostic_normalize.py
# Requirements:
#   pip install sentence-transformers chromadb numpy rapidfuzz bs4 lxml

import re
import uuid
import json
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from rapidfuzz import fuzz

# ---------- CONFIG ----------
MODEL_NAME = "all-MiniLM-L6-v2"
MERGE_THRESHOLD = 0.86   # cosine threshold to merge into existing canonical
MATCH_THRESHOLD = 0.80   # threshold to consider two canonical embeddings matching
FUZZY_THRESHOLD = 88     # rapidfuzz score (0-100)
COLLECTION_NAME = "skill_registry"
PERSIST_DIR = "./chroma_store"

# ---------- Normalization / candidate generation ----------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.lower()
    s = re.sub(r'[\r\n]+', ' ', s)
    s = re.sub(r'[^a-z0-9\+\#\s\-/,&\|;()\.:]', ' ', s)
    s = s.replace('c++', 'cpp').replace('c#', 'csharp')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

SPLIT_SEPARATORS = r'[\/,&\|;]'

def generate_candidates(raw: str, min_tokens: int = 1) -> List[str]:
    s = normalize_text(raw)
    if not s:
        return []
    parts = re.split(r'[\(\)]', s)
    tokens = []
    for p in parts:
        for chunk in re.split(SPLIT_SEPARATORS, p):
            chunk = chunk.strip()
            if chunk:
                tokens.append(chunk)
    # dedupe preserving order
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ---------- Chroma-backed canonicalizer ----------
class ChromaCanonicalizer:
    def __init__(self,
                 model_name: str = MODEL_NAME,
                 merge_thresh: float = MERGE_THRESHOLD,
                 match_thresh: float = MATCH_THRESHOLD,
                 fuzzy_thresh: int = FUZZY_THRESHOLD,
                 collection_name: str = COLLECTION_NAME,
                 persist_dir: str = PERSIST_DIR,
                 persist: bool = True):
        self.model = SentenceTransformer(model_name)
        self.merge_thresh = merge_thresh
        self.match_thresh = match_thresh
        self.fuzzy_thresh = fuzzy_thresh
        self.persist_flag = persist

        # try PersistentClient if available, fallback to Client()
        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
        except Exception:
            try:
                self.client = chromadb.Client()
            except Exception as e:
                raise RuntimeError("Failed to create Chroma client: " + str(e))

        try:
            self.col = self.client.get_collection(name=collection_name)
        except Exception:
            self.col = self.client.create_collection(name=collection_name)

        # in-memory alias -> cid map for deterministic exact matches
        self.alias_index: Dict[str, str] = {}
        self._load_alias_index_from_chroma()

    def clear_collection(self):
        """Utility for tests: delete all entries (destructive)."""
        try:
            self.col.delete()
        except Exception:
            try:
                self.client.delete_collection(name=self.col.name)
            except Exception:
                pass
            self.col = self.client.create_collection(name=self.col.name)
        self.alias_index = {}

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        return embs.astype('float32')

    def _meta_to_chroma_safe(self, label: str, aliases: List[str]) -> Dict[str, Any]:
        aliases_str = "|".join(aliases) if aliases else ""
        return {'label': label, 'aliases': aliases_str}

    def _load_alias_index_from_chroma(self):
        """Load existing aliases from chroma metadata into in-memory index (best-effort)."""
        try:
            resp = self.col.get(include=['ids', 'metadatas'])
            ids = resp.get('ids', [])
            metas = resp.get('metadatas', [])
            for i, cid in enumerate(ids):
                meta = metas[i] if i < len(metas) else {}
                aliases_s = meta.get('aliases', '')
                if aliases_s:
                    for a in aliases_s.split('|'):
                        if a:
                            self.alias_index[a] = cid
        except Exception:
            pass

    def _register_aliases_local(self, cid: str, aliases: List[str]):
        for a in aliases:
            self.alias_index[a] = cid

    def add_phrases(self, raw_phrases: List[str], generate_candidates_fn=generate_candidates) -> Dict[str, str]:
        """
        Add phrases to the registry. Returns mapping {candidate -> canonical_id}.
        generate_candidates_fn should return normalized tokens (lowercased).
        """
        candidates = []
        for p in raw_phrases:
            candidates.extend(generate_candidates_fn(p))
        if not candidates:
            return {}

        freq = Counter(candidates)
        uniq = [p for p, _ in freq.most_common()]
        embs = self._embed(uniq)
        mapping: Dict[str, str] = {}

        for i, phrase in enumerate(uniq):
            phrase_norm = phrase
            # exact alias lookup
            if phrase_norm in self.alias_index:
                mapping[phrase] = self.alias_index[phrase_norm]
                continue

            emb = embs[i].tolist()

            # try embedding nearest neighbor
            try:
                col_count = self.col.count()
            except Exception:
                col_count = 0

            if col_count > 0:
                try:
                    res = self.col.query(query_embeddings=[emb], n_results=1, include=["metadatas", "distances"])
                    ids = res.get('ids', [[]])[0]
                    metas = res.get('metadatas', [[]])[0]
                except Exception:
                    ids, metas = [], []
                if ids and metas and metas[0]:
                    cid = ids[0]
                    meta0 = metas[0]
                    # fetch stored embedding
                    try:
                        stored = self.col.get(ids=[cid], include=['embeddings'])
                        stored_embs = stored.get('embeddings', [[]])[0]
                        if stored_embs:
                            stored_emb = np.array(stored_embs, dtype='float32')
                            sim = float(np.dot(stored_emb, np.array(emb, dtype='float32')))
                            if sim >= self.merge_thresh:
                                existing_aliases_str = meta0.get('aliases', '')
                                aliases_list = existing_aliases_str.split('|') if existing_aliases_str else []
                                if phrase_norm not in aliases_list:
                                    aliases_list.append(phrase_norm)
                                    meta_for_chroma = self._meta_to_chroma_safe(meta0.get('label', cid), aliases_list)
                                    try:
                                        self.col.add(ids=[cid], embeddings=[stored_emb.tolist()], metadatas=[meta_for_chroma], documents=[meta_for_chroma['label']])
                                    except Exception:
                                        pass
                                    self._register_aliases_local(cid, [phrase_norm])
                                mapping[phrase] = cid
                                continue
                    except Exception:
                        pass

            # fuzzy string fallback against alias_index
            best_alias = None
            best_score = 0
            for a, cid in self.alias_index.items():
                score = fuzz.token_sort_ratio(phrase_norm, a)
                if score > best_score:
                    best_score = score; best_alias = a
            if best_score >= self.fuzzy_thresh:
                matched_cid = self.alias_index[best_alias]
                try:
                    stored = self.col.get(ids=[matched_cid], include=['embeddings','metadatas'])
                    stored_embs = stored.get('embeddings', [[]])[0]
                    meta0 = stored.get('metadatas', [[]])[0][0] if stored.get('metadatas') else {}
                    if stored_embs:
                        stored_emb = np.array(stored_embs, dtype='float32')
                        existing_aliases_str = meta0.get('aliases','')
                        aliases_list = existing_aliases_str.split('|') if existing_aliases_str else []
                        if phrase_norm not in aliases_list:
                            aliases_list.append(phrase_norm)
                            meta_for_chroma = self._meta_to_chroma_safe(meta0.get('label', matched_cid), aliases_list)
                            try:
                                self.col.add(ids=[matched_cid], embeddings=[stored_emb.tolist()], metadatas=[meta_for_chroma], documents=[meta_for_chroma['label']])
                            except Exception:
                                pass
                            self._register_aliases_local(matched_cid, [phrase_norm])
                        mapping[phrase] = matched_cid
                        continue
                except Exception:
                    pass

            # create new canonical
            new_id = str(uuid.uuid4())
            label = phrase_norm
            meta_for_chroma = self._meta_to_chroma_safe(label, [phrase_norm])
            try:
                self.col.add(ids=[new_id], embeddings=[emb], metadatas=[meta_for_chroma], documents=[label])
            except Exception:
                pass
            mapping[phrase] = new_id
            self._register_aliases_local(new_id, [phrase_norm])

        if self.persist_flag:
            try:
                self.client.persist()
            except Exception:
                pass
        return mapping

    def map_phrase(self, phrase: str, generate_candidates_fn=generate_candidates) -> Tuple[str, float]:
        candidates = generate_candidates_fn(phrase)
        if not candidates:
            return None, 0.0
        cand = candidates[0]
        # exact alias
        if cand in self.alias_index:
            return self.alias_index[cand], 0.99

        emb = self._embed([cand])[0].tolist()
        try:
            col_count = self.col.count()
        except Exception:
            col_count = 0

        if col_count == 0:
            new_id = str(uuid.uuid4())
            meta = self._meta_to_chroma_safe(cand, [cand])
            try:
                self.col.add(ids=[new_id], embeddings=[emb], metadatas=[meta], documents=[cand])
            except Exception:
                pass
            self._register_aliases_local(new_id, [cand])
            if self.persist_flag:
                try: self.client.persist()
                except Exception: pass
            return new_id, 1.0

        try:
            res = self.col.query(query_embeddings=[emb], n_results=1, include=["metadatas","distances"])
            ids = res.get('ids', [[]])[0]; metas = res.get('metadatas', [[]])[0]
        except Exception:
            ids, metas = [], []
        if ids and metas and metas[0]:
            cid = ids[0]; meta0 = metas[0]
            try:
                stored = self.col.get(ids=[cid], include=['embeddings'])
                stored_embs = stored.get('embeddings', [[]])[0]
                if stored_embs:
                    stored_emb = np.array(stored_embs, dtype='float32')
                    sim = float(np.dot(stored_emb, np.array(emb, dtype='float32')))
                    if sim >= self.merge_thresh:
                        existing_aliases_str = meta0.get('aliases','')
                        aliases_list = existing_aliases_str.split('|') if existing_aliases_str else []
                        if cand not in aliases_list:
                            aliases_list.append(cand)
                            meta_for_chroma = self._meta_to_chroma_safe(meta0.get('label', cid), aliases_list)
                            try:
                                self.col.add(ids=[cid], embeddings=[stored_emb.tolist()], metadatas=[meta_for_chroma], documents=[meta_for_chroma['label']])
                            except Exception:
                                pass
                            self._register_aliases_local(cid, [cand])
                        return cid, sim
            except Exception:
                pass

        # fuzzy fallback
        best_alias = None; best_score = 0
        for a, ac in self.alias_index.items():
            score = fuzz.token_sort_ratio(cand, a)
            if score > best_score:
                best_score = score; best_alias = a
        if best_score >= self.fuzzy_thresh:
            matched_cid = self.alias_index[best_alias]
            return matched_cid, best_score/100.0

        # create new
        new_id = str(uuid.uuid4())
        meta_for_chroma = self._meta_to_chroma_safe(cand, [cand])
        try:
            self.col.add(ids=[new_id], embeddings=[emb], metadatas=[meta_for_chroma], documents=[cand])
        except Exception:
            pass
        self._register_aliases_local(new_id, [cand])
        if self.persist_flag:
            try: self.client.persist()
            except Exception: pass
        return new_id, 1.0

    def find_matches_between_lists(self, source_phrases: List[str], target_phrases: List[str], generate_candidates_fn=generate_candidates) -> List[Dict[str, Any]]:
        src_map = {s: self.map_phrase(s, generate_candidates_fn) for s in source_phrases}
        tgt_map = {t: self.map_phrase(t, generate_candidates_fn) for t in target_phrases}

        # cache canonical embeddings
        canon_cache: Dict[str, np.ndarray] = {}
        def _get_canon_emb(cid):
            if cid in canon_cache:
                return canon_cache[cid]
            try:
                resp = self.col.get(ids=[cid], include=['embeddings'])
                emb_list = resp.get('embeddings', [[]])[0]
                emb_arr = np.array(emb_list, dtype='float32') if emb_list else None
            except Exception:
                emb_arr = None
            canon_cache[cid] = emb_arr
            return emb_arr

        matches = []
        for s, (scid, ssim) in src_map.items():
            for t, (tcid, tsim) in tgt_map.items():
                if scid == tcid:
                    matches.append({'src_phrase': s, 'tgt_phrase': t, 'match_type': 'canonical_exact', 'canon_id': scid, 'confidence': 0.99})
                else:
                    emb_s = _get_canon_emb(scid)
                    emb_t = _get_canon_emb(tcid)
                    if emb_s is not None and emb_t is not None:
                        sim = float(np.dot(emb_s, emb_t))
                        if sim >= self.match_thresh:
                            matches.append({'src_phrase': s, 'tgt_phrase': t, 'match_type': 'embed_sim', 'canon_src': scid, 'canon_tgt': tcid, 'confidence': sim})
        return matches

    def export_registry(self) -> Dict[str, Any]:
        """
        Robust export of the registry. Try chroma.get(); if empty, build from alias_index.
        """
        try:
            resp = self.col.get(include=['ids', 'metadatas', 'documents'])
            ids = resp.get('ids', [])
            if ids:
                return resp
        except Exception:
            pass

        registry = {'ids': [], 'metadatas': [], 'documents': []}
        canon_map: Dict[str, Dict[str, Any]] = {}
        for alias, cid in self.alias_index.items():
            if cid not in canon_map:
                canon_map[cid] = {'label': None, 'aliases': set()}
            canon_map[cid]['aliases'].add(alias)

        for cid, info in canon_map.items():
            label = None
            try:
                resp = self.col.get(ids=[cid], include=['metadatas', 'documents'])
                metas = resp.get('metadatas', [[]])[0]
                docs = resp.get('documents', [[]])[0]
                if metas and metas[0]:
                    label = metas[0].get('label')
                elif docs and docs[0]:
                    label = docs[0]
            except Exception:
                pass
            if not label:
                label = next(iter(info['aliases']))
            aliases_str = "|".join(sorted(info['aliases']))
            meta = {'label': label, 'aliases': aliases_str}
            registry['ids'].append(cid)
            registry['metadatas'].append(meta)
            registry['documents'].append(label)
        return registry

    def dump_registry_json(self, path="skill_registry.json"):
        reg = self.export_registry()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)
        print("Wrote", path)


# ---------------------------------------------------------------------------------------
# Job description skill extractor (HTML-aware)
import re
from bs4 import BeautifulSoup
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# lazy singletons to avoid reloading model repeatedly
_SSBERT = None
def _get_sbert(model_name: str = MODEL_NAME):
    global _SSBERT
    if _SSBERT is None:
        _SSBERT = SentenceTransformer(model_name)
    return _SSBERT

# small set of heading hints to identify "skills" or "requirements" blocks
_HEADING_HINTS = [
    "requirements", "responsibilities", "skills", "qualifications",
    "you should have", "you will", "minimum requirements", "preferred",
    "what we're looking", "we are looking for", "what you'll do"
]

_BULLET_RE = re.compile(r'^\s*[-•\*\d\.\)]\s+')

def _strip_html_keep_lines(text: str) -> str:
    soup = BeautifulSoup(text or "", "lxml")
    for tag in soup(["script","style"]):
        tag.extract()
    return soup.get_text(separator="\n")

def _section_split_simple(text: str):
    lines = [ln.rstrip() for ln in text.splitlines()]
    indices = []
    for i, ln in enumerate(lines):
        lln = ln.strip().lower().rstrip(':')
        for hint in _HEADING_HINTS:
            if lln.startswith(hint) or hint in lln:
                indices.append((i, lln))
                break
    if not indices:
        return {"whole": text}
    sections = {}
    for idx, (i, head) in enumerate(indices):
        start = i + 1
        end = indices[idx+1][0] if idx+1 < len(indices) else len(lines)
        sec_text = "\n".join(lines[start:end]).strip()
        key = head.split()[0]
        sections[key] = sec_text
    return sections

def _extract_bullets_and_short_lines(text: str, max_words: int = 8) -> List[str]:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if _BULLET_RE.match(ln) or len(ln.split()) <= max_words:
            clean = re.sub(r'^[\-\•\*\d\.\)\s]+', '', ln).strip()
            if clean:
                out.append(clean)
    return out

def _heuristic_parenthetical_and_lists(text: str) -> List[str]:
    out = []
    for m in re.findall(r'\(([^)]+)\)', text):
        parts = re.split(r'[,\|/;]', m)
        for p in parts:
            p = p.strip()
            if p and len(p.split()) <= 6:
                out.append(p)
    for m in re.findall(r'([A-Za-z0-9\+\#\.\- ]{2,120}(?:[,/|][A-Za-z0-9\+\#\.\- ]+){1,10})', text):
        for frag in re.split(r'[,/|;]', m):
            frag = frag.strip()
            if frag and len(frag.split()) <= 6:
                out.append(frag)
    for token in re.findall(r'\b[A-Za-z][A-Za-z0-9\+\#\.\-]{1,30}\b', text):
        if re.search(r'[A-Z][a-z]', token) or re.search(r'\+|#|\.', token):
            out.append(token)
    seen = set(); res = []
    for x in out:
        n = normalize_text(x)
        if n and n not in seen:
            seen.add(n); res.append(x)
    return res

def _semantic_dedupe_topk(candidates: List[str], model_name: str = MODEL_NAME, thresh: float = 0.86, topk: int = 60) -> List[str]:
    if not candidates:
        return []
    limited = candidates[:topk]
    model = _get_sbert(model_name)
    embs = model.encode(limited, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    keep_idx = []
    keep_embs = []
    for i, e in enumerate(embs):
        if not keep_embs:
            keep_idx.append(i)
            keep_embs.append(e)
            continue
        sims = np.dot(np.vstack(keep_embs), e)
        if float(np.max(sims)) < thresh:
            keep_idx.append(i)
            keep_embs.append(e)
    return [limited[i] for i in keep_idx]

def extract_skills_from_jd_text(raw_text: str,
                                     model_name: str = MODEL_NAME,
                                     dedupe_threshold: float = MERGE_THRESHOLD,
                                     max_candidates: int = 60,
                                     prefer_sectioned: bool = True) -> List[str]:
    if not raw_text:
        return []

    text = _strip_html_keep_lines(raw_text)

    # 1) section split and collect candidates
    sections = _section_split_simple(text)
    candidates = []

    if prefer_sectioned:
        for name in ("skills", "requirements", "qualifications", "responsibilities"):
            if name in sections and sections[name]:
                candidates.extend(_extract_bullets_and_short_lines(sections[name]))
                candidates.extend(_heuristic_parenthetical_and_lists(sections[name]))

    if not candidates:
        candidates.extend(_extract_bullets_and_short_lines(text))
    else:
        candidates.extend(_heuristic_parenthetical_and_lists(text))

    if not candidates:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        short_lines = [ln for ln in lines if 1 <= len(ln.split()) <= 10]
        for ln in short_lines:
            for g in generate_candidates(ln):
                if g:
                    candidates.append(g)

    expanded = []
    for c in candidates:
        for g in generate_candidates(c):
            if g:
                expanded.append(g)

    filtered = []
    seen = set()
    for c in expanded:
        norm = normalize_text(c)
        if not norm:
            continue
        if not re.search(r'[a-zA-Z]', norm):
            continue
        if len(norm) < 2:
            continue
        if len(norm.split()) > 6:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        filtered.append(norm)

    if not filtered:
        return []

    if len(filtered) <= 16:
        return filtered[:max_candidates]

    deduped = _semantic_dedupe_topk(filtered, model_name=model_name, thresh=dedupe_threshold, topk=max_candidates)
    return deduped[:max_candidates]


# ---------- Demo ----------
if __name__ == "__main__":
    reg = ChromaCanonicalizer()
    raw = ["PyTorch", "pytorch", "Model Monitoring (MLOps)", "model monitoring"]
    print("add:", reg.add_phrases(raw, generate_candidates_fn=generate_candidates))
    print("col.count():", reg.col.count())
    print("alias_index keys:", list(reg.alias_index.keys()))
    print("export:", reg.export_registry())
    print("matches:", reg.find_matches_between_lists(
        ["PyTorch"], ["pyTorch", "model monitoring"],
        generate_candidates_fn=generate_candidates
    ))
    reg.dump_registry_json()
