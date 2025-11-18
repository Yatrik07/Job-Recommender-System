#!/usr/bin/env python3
"""
rank_resume_vs_jds.py

Usage examples:
python rank_resume_vs_jds.py --resume resume_skills.json --jds processed_jds.jsonl --sample 50 --out ranked_jds.jsonl

Requirements:
pip install sentence-transformers numpy pandas tqdm
"""

import argparse
import json
import random
import math
import os
from typing import List, Set, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import re

# ---------------- utils (same normalization used previously) ----------------
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

def avg_embedding(model, texts: List[str]) -> np.ndarray:
    if not texts:
        return None
    embs = model.encode(texts, convert_to_numpy=True)
    # handle single vector case
    if embs.ndim == 1:
        vec = embs
    else:
        vec = embs.mean(axis=0)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ---------------- I/O helpers ----------------
def load_resume_skills(path: str) -> Dict[str, Any]:
    """
    Accepts:
    - JSON file with { "canonical_ids": [...], "skills": [...] } OR
    - JSON array (["python", "pytorch", ...]) OR
    - plain text file, one skill per line
    Returns dict with:
      { "canonical_ids": set(...) or None, "skills": list(str) }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    skills = []
    canonical_ids = None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read().strip()
            # try JSON
            data = json.loads(txt)
            if isinstance(data, dict):
                canonical_ids = set(data.get('canonical_ids') or [])
                skills = list(data.get('skills') or data.get('candidates') or [])
            elif isinstance(data, list):
                # list of skills
                skills = [str(x) for x in data]
            else:
                # fallback to lines
                skills = txt.splitlines()
    except Exception:
        # fallback plain text lines
        with open(path, 'r', encoding='utf-8') as f:
            skills = [ln.strip() for ln in f.readlines() if ln.strip()]
    skills = [s for s in skills if s]
    return {"canonical_ids": set(canonical_ids) if canonical_ids else None, "skills": skills}

def load_jds_jsonl(path: str) -> List[Dict[str, Any]]:
    recs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                recs.append(rec)
            except Exception:
                continue
    return recs

# ---------------- scoring logic ----------------
def compute_overlap_score(resume_cids: Set[str], jd_cids: List[str]) -> float:
    # If resume_cids is None, this should not be called.
    if not resume_cids:
        return 0.0
    if not jd_cids:
        return 0.0
    jd_set = set([str(x) for x in jd_cids if x])
    if not jd_set:
        return 0.0
    inter = resume_cids.intersection(jd_set)
    # ratio w.r.t resume size (so small resume won't be dominated)
    return len(inter) / (len(resume_cids) + 1e-12)

def compute_token_jaccard(resume_tokens: Set[str], jd_tokens: Set[str]) -> float:
    if not resume_tokens or not jd_tokens:
        return 0.0
    inter = resume_tokens.intersection(jd_tokens)
    union = resume_tokens.union(jd_tokens)
    return len(inter) / (len(union) + 1e-12)

def tokens_from_labels(labels: List[str]) -> Set[str]:
    out = set()
    for lab in labels or []:
        n = normalize_text(str(lab))
        if n:
            out.add(n)
    return out

# ---------------- main ranking function ----------------
def rank_resume_vs_jds(resume_path: str, jds_path: str, sample_size: int = 50,
                       model_name: str = "all-MiniLM-L6-v2",
                       w_overlap: float = 0.6, w_embed: float = 0.4,
                       out_jsonl: str = "ranked_jds.jsonl", out_csv: str = "ranked_jds.csv",
                       random_seed: int = 42):
    # load resume
    resume = load_resume_skills(resume_path)
    resume_cids = resume.get("canonical_ids")
    resume_skills = resume.get("skills", [])
    print(f"Loaded resume: {len(resume_skills)} skills, canonical_ids: {len(resume_cids) if resume_cids else 0}")

    # load JDs
    jds = load_jds_jsonl(jds_path)
    if not jds:
        raise RuntimeError("No JDs loaded from " + jds_path)
    print(f"Loaded {len(jds)} JDs from {jds_path}")

    # sample
    random.seed(random_seed)
    if sample_size and sample_size < len(jds):
        sample_jds = random.sample(jds, sample_size)
    else:
        sample_jds = jds

    # prepare texts/candidates for embedding
    # prefer canonical_labels (already normalized) else fallback to 'candidates' list or 'candidates' text
    jd_texts = []
    for rec in sample_jds:
        labels = rec.get("canonical_labels") or []
        if isinstance(labels, list):
            if labels:
                jd_texts.append(" ".join(str(x) for x in labels))
                continue
        # fallback to candidates list
        cands = rec.get("candidates") or []
        jd_texts.append(" ".join(str(x) for x in cands))

    # build token sets for token-based overlap (if resume has no canonical ids)
    resume_tokens = tokens_from_labels(resume_skills)
    jd_token_sets = [tokens_from_labels(rec.get("canonical_labels") or rec.get("candidates") or []) for rec in sample_jds]

    # load SBERT model and compute embeddings
    model = SentenceTransformer(model_name)
    # compute resume embedding (avg of skill strings)
    resume_emb = avg_embedding(model, [str(s) for s in resume_skills])  # None if empty
    # compute JD embeddings in batch
    print("Computing JD embeddings (batch)...")
    jd_embs = model.encode(jd_texts, convert_to_numpy=True)
    # normalize jd_embs rows
    norms = np.linalg.norm(jd_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    jd_embs = (jd_embs / norms).astype('float32')

    results = []
    for idx, rec in enumerate(sample_jds):
        job_id = rec.get("job_id") or rec.get("id") or f"idx_{idx}"
        title = rec.get("title")
        company = rec.get("company")
        location = rec.get("location")
        candidates = rec.get("candidates") or []
        canonical_ids = rec.get("canonical_ids") or []
        canonical_labels = rec.get("canonical_labels") or []

        # overlap part
        overlap_score = 0.0
        if resume_cids:
            overlap_score = compute_overlap_score(resume_cids, canonical_ids)
        else:
            # token jaccard as fallback
            overlap_score = compute_token_jaccard(resume_tokens, jd_token_sets[idx])

        # embed similarity
        jd_emb = jd_embs[idx] if idx < len(jd_embs) else None
        embed_score = cosine_sim(resume_emb, jd_emb) if resume_emb is not None else 0.0
        # embed_score currently in [-1,1] due to cosine; clamp to [0,1]
        embed_score = max(0.0, min(1.0, (embed_score + 1.0) / 2.0)) if embed_score is not None else 0.0

        final_score = w_overlap * overlap_score + w_embed * embed_score

        results.append({
            "job_id": job_id,
            "title": title,
            "company": company,
            "location": location,
            "canonical_ids": canonical_ids,
            "canonical_labels": canonical_labels,
            "candidates": candidates,
            "overlap_score": overlap_score,
            "embed_score": embed_score,
            "final_score": final_score
        })

    # sort descending by final_score
    results_sorted = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # write outputs
    with open(out_jsonl, 'w', encoding='utf-8') as fo:
        for r in results_sorted:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    # csv
    df = pd.DataFrame(results_sorted)
    # pick columns in order
    cols = ["final_score", "overlap_score", "embed_score", "job_id", "title", "company", "location", "canonical_labels", "candidates"]
    df.to_csv(out_csv, index=False, columns=[c for c in cols if c in df.columns])

    # print top 10
    print("\nTop 10 JDs:")
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i}. score={r['final_score']:.4f} overlap={r['overlap_score']:.3f} embed={r['embed_score']:.3f} | {r.get('title')} @ {r.get('company')} ({r.get('location')})")

    print(f"\nWrote {len(results_sorted)} ranked records to {out_jsonl} and {out_csv}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank one resume against a sample of JDs (processed_jds.jsonl).")
    parser.add_argument("--resume", "-r", required=True, help="Path to resume skills file (JSON or txt).")
    parser.add_argument("--jds", "-j", required=True, help="Path to processed_jds.jsonl (output of process_promptcloud_ldjson).")
    parser.add_argument("--sample", "-n", type=int, default=50, help="Number of random JDs to sample (default 50).")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SBERT model name.")
    parser.add_argument("--out-jsonl", default="ranked_jds.jsonl", help="Output JSONL path.")
    parser.add_argument("--out-csv", default="ranked_jds.csv", help="Output CSV path.")
    parser.add_argument("--w-overlap", type=float, default=0.6, help="Weight for overlap score (0-1).")
    parser.add_argument("--w-embed", type=float, default=0.4, help="Weight for embedding similarity (0-1).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.w_overlap + args.w_embed <= 0:
        raise ValueError("At least one of w-overlap or w-embed must be > 0")
    # normalize weights
    s = args.w_overlap + args.w_embed
    w_ov = args.w_overlap / s
    w_em = args.w_embed / s

    rank_resume_vs_jds(args.resume, args.jds, sample_size=args.sample,
                       model_name=args.model, w_overlap=w_ov, w_embed=w_em,
                       out_jsonl=args.out_jsonl, out_csv=args.out_csv, random_seed=args.seed)
