#!/usr/bin/env python3
"""
rank_manual_jds.py

Usage:
1) Prepare a file manual_jds.txt where each JD is separated by a line with exactly:
   ===JD===
   Then run:
   python rank_manual_jds.py --resume resume_skills.json --jds-file manual_jds.txt --sample 50

2) Or pass a single JD string on the command line:
   python rank_manual_jds.py --resume resume_skills.json --jd-string "Paste whole job description here..."

Requires your project code:
 - domain_agnostic_normalize.py (ChromaCanonicalizer, extract_skills_from_jd_text, generate_candidates, normalize_text)
 - sentence-transformers, numpy, pandas

Outputs:
 - ranked_manual_jds.jsonl
 - ranked_manual_jds.csv
"""
import argparse, json, os, random, re
from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# import your project functions (assumes same directory)
from domain_agnostic_normalize import (
    ChromaCanonicalizer,
    extract_skills_from_jd_text,
    generate_candidates,
    normalize_text
)

def split_manual_jds_from_file(path: str, sep: str = "===JD==="):
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    parts = [p.strip() for p in txt.split(sep) if p.strip()]
    return parts

def embed_avg(model, texts):
    if not texts:
        return None
    embs = model.encode(texts, convert_to_numpy=True)
    if embs.ndim == 1:
        vec = embs
    else:
        vec = embs.mean(axis=0)
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm and norm != 0 else vec

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def rank_manual_jds(resume_path, jds_file=None, jd_string=None, use_chroma=True,
                   model_name="all-MiniLM-L6-v2", w_overlap=0.6, w_embed=0.4,
                   out_jsonl="ranked_manual_jds.jsonl", out_csv="ranked_manual_jds.csv"):
    # Load resume skills (accept simple json list or dict with 'skills' or 'canonical_ids')
    if not os.path.exists(resume_path):
        raise FileNotFoundError(resume_path)
    with open(resume_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            resume_skills = data.get("skills") or data.get("candidates") or []
            resume_cids = set(data.get("canonical_ids") or [])
        elif isinstance(data, list):
            resume_skills = data
            resume_cids = None
        else:
            resume_skills = raw.splitlines()
            resume_cids = None
    except Exception:
        resume_skills = raw.splitlines()
        resume_cids = None

    # Read JDs
    jds = []
    if jds_file:
        jds_texts = split_manual_jds_from_file(jds_file)
    elif jd_string:
        jds_texts = [jd_string]
    else:
        raise ValueError("Provide --jds-file or --jd-string")

    # Build canonicalizer if desired
    reg = None
    if use_chroma:
        try:
            reg = ChromaCanonicalizer()
        except Exception:
            print("Warning: could not initialize ChromaCanonicalizer, proceeding without canonical mapping.")
            reg = None

    # Extract candidates for each JD
    jd_records = []
    for i, txt in enumerate(jds_texts):
        cands = []
        try:
            cands = extract_skills_from_jd_text(txt, prefer_sectioned=True)
        except Exception:
            # fallback using generate_candidates on first 40 lines
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            for ln in lines[:40]:
                for g in generate_candidates(ln):
                    if g and g not in cands:
                        cands.append(g)
        # normalize display list
        cands_norm = [normalize_text(x) for x in cands if normalize_text(x)]
        jd_records.append({"text": txt, "candidates": cands_norm})

    # Map candidates to canonical ids if possible
    if reg is not None:
        all_cands = []
        for rec in jd_records:
            all_cands.extend(rec["candidates"])
        uniq = []
        seen = set()
        for c in all_cands:
            if c not in seen:
                seen.add(c); uniq.append(c)
        # add to chroma
        mapping = {}
        if uniq:
            try:
                mapping = reg.add_phrases(uniq, generate_candidates_fn=generate_candidates)
            except Exception:
                mapping = {}
        # attach canonical ids per JD
        for rec in jd_records:
            rec["canonical_ids"] = [mapping.get(c, None) for c in rec["candidates"]]
    else:
        for rec in jd_records:
            rec["canonical_ids"] = []

    # Prepare tokens and embeddings
    model = SentenceTransformer(model_name)
    resume_emb = embed_avg(model, resume_skills)
    jd_texts_for_embed = [" ".join(rec["candidates"]) for rec in jd_records]
    jd_embs = model.encode(jd_texts_for_embed, convert_to_numpy=True)
    # normalize jd_embs
    norms = np.linalg.norm(jd_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    jd_embs = (jd_embs / norms).astype('float32')

    results = []
    resume_token_set = set([normalize_text(s) for s in resume_skills if normalize_text(s)])
    for i, rec in enumerate(jd_records):
        overlap_score = 0.0
        if resume_cids:
            # overlap using canonical ids
            jd_cids = [c for c in rec.get("canonical_ids") if c]
            if jd_cids:
                overlap_score = len(set(jd_cids).intersection(resume_cids)) / (len(resume_cids) + 1e-12)
        else:
            jd_tokens = set(rec["candidates"])
            if resume_token_set and jd_tokens:
                inter = resume_token_set.intersection(jd_tokens)
                union = resume_token_set.union(jd_tokens)
                overlap_score = len(inter) / (len(union) + 1e-12)

        embed_score = cosine_sim(resume_emb, jd_embs[i]) if resume_emb is not None else 0.0
        # map to [0,1]
        embed_score = max(0.0, min(1.0, (embed_score + 1.0) / 2.0))
        final_score = w_overlap * overlap_score + w_embed * embed_score
        results.append({
            "index": i, "text": rec["text"], "candidates": rec["candidates"],
            "canonical_ids": rec.get("canonical_ids", []),
            "overlap_score": overlap_score, "embed_score": embed_score, "final_score": final_score
        })

    results_sorted = sorted(results, key=lambda x: x["final_score"], reverse=True)
    # write outputs
    with open(out_jsonl, 'w', encoding='utf-8') as fo:
        for r in results_sorted:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(results_sorted)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(results_sorted)} ranked JDs to {out_jsonl} and {out_csv}")
    # print top
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i}. score={r['final_score']:.4f} overlap={r['overlap_score']:.3f} embed={r['embed_score']:.3f}")
        print("   candidates:", r['candidates'][:12])
        print("")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True, help="resume skills file (json list or dict)")
    p.add_argument("--jds-file", help="manual jds file with ===JD=== separators")
    p.add_argument("--jd-string", help="single jd string")
    p.add_argument("--no-chroma", action="store_true", help="skip chroma canonical mapping")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--w-overlap", type=float, default=0.6)
    p.add_argument("--w-embed", type=float, default=0.4)
    p.add_argument("--out-jsonl", default="ranked_manual_jds.jsonl")
    p.add_argument("--out-csv", default="ranked_manual_jds.csv")
    args = p.parse_args()
    rank_manual_jds(args.resume, jds_file=args.jds_file, jd_string=args.jd_string, use_chroma=(not args.no_chroma),
                   model_name=args.model, w_overlap=args.w_overlap, w_embed=args.w_embed,
                   out_jsonl=args.out_jsonl, out_csv=args.out_csv)
