# process_promptcloud_ldjson.py
"""
PromptCloud LDJSON -> processed_jds.jsonl (batch canonicalization)
Usage:
python process_promptcloud_ldjson.py --input path/to/promptcloud.ldjson --out processed_jds.jsonl --sample 1000
"""

import argparse
import json
import os
import time
import re
import uuid
from collections import defaultdict
from tqdm import tqdm

# Import your canonicalizer + extractors
from domain_agnostic_normalize import (
    ChromaCanonicalizer,
    generate_candidates,
    extract_skills_from_jd_text,
    normalize_text,
)

# ---------------- Config / Stoplist ----------------
STOPWORDS = {
    "job", "information", "qualifications", "salary", "location", "experience",
    "preferred", "full-time", "part-time", "hour", "hours", "apply", "apply now",
    "job type", "benefits", "responsibilities", "requirements", "qualifications:",
    "about the role", "other", "if applicable", "read what", "equal opportunity employer"
}
_RE_SALARY = re.compile(r'^\$?\d+(\.\d+)?(\s?-\s?\$?\d+(\.\d+)?)?(\s?(\/|per)\s?(hour|week|month|year))?$', re.I)
_RE_PHONE = re.compile(r'(\+?\d{1,3}[\s\-\.])?(\(?\d{3}\)?[\s\-\.])?\d{3}[\s\-\.]\d{4}')
_RE_DATE = re.compile(r'\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t)?|oct(ober)?|nov(ember)?|dec(ember)?)[\s\d,]{0,6}\b', re.I)
_RE_LOC_CODE = re.compile(r'\b[A-Z]{2,3}\s?\d{3,6}\b')
_RE_NON_ALPHA = re.compile(r'^[^a-zA-Z]+$')
_MIN_TOKEN_LEN = 2

def filter_candidates(cands):
    out = []
    seen = set()
    for c in cands:
        if not c:
            continue
        s = c.strip()
        if not s:
            continue
        s_low = s.lower().strip()
        if s_low in STOPWORDS:
            continue
        if _RE_SALARY.match(s_low):
            continue
        if _RE_PHONE.search(s_low):
            continue
        if _RE_DATE.search(s_low):
            continue
        if _RE_LOC_CODE.search(s_low):
            continue
        if _RE_NON_ALPHA.match(s_low):
            continue
        if len(s_low) < _MIN_TOKEN_LEN:
            continue
        if any(sw in s_low for sw in ("salary", "location", "apply", "experience")) and len(s_low.split()) <= 3:
            continue
        norm = s_low
        if norm in seen:
            continue
        seen.add(norm)
        out.append(s)
    return out

CANDIDATE_TEXT_KEYS = [
    "job_description","description","full_description","jobdesc","job_description_text",
    "desc","jobDescription","job_description_html","raw_description","snippet","details","summary"
]

def get_job_text(record: dict) -> str:
    for k in CANDIDATE_TEXT_KEYS:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    parts = []
    for k, v in record.items():
        if isinstance(v, str) and len(v) > 50:
            parts.append(v)
    return "\n".join(parts)

def _quick_extract_fallback(raw_text, max_lines=40):
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    cand = []
    for ln in lines[:max_lines]:
        for g in generate_candidates(ln):
            if g and g not in cand:
                cand.append(g)
    return cand

def build_cid_label_map(reg: ChromaCanonicalizer):
    cid_label = {}
    try:
        data = reg.export_registry()
        ids = data.get('ids', [])
        metas = data.get('metadatas', [])
        for i, cid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            label = meta.get('label')
            aliases = meta.get('aliases')
            if not label and aliases:
                label = aliases.split('|')[0] if '|' in aliases else aliases
            cid_label[cid] = label
    except Exception:
        pass
    try:
        for alias, cid in getattr(reg, 'alias_index', {}).items():
            if cid not in cid_label or cid_label[cid] is None:
                cid_label[cid] = alias
    except Exception:
        pass
    return cid_label

def process_file(input_path: str, out_path: str, sample: int = None,
                 batch_size: int = 128, persist_every_batches: int = 2,
                 max_candidates_per_jd: int = 60, checkpoint_every: int = 500,
                 tmp_mode_append: bool = False):
    reg = ChromaCanonicalizer()
    print("Chroma registry loaded.")

    cid_label_map = build_cid_label_map(reg)
    print(f"Loaded {len(cid_label_map)} cid->label entries from registry (sample shown):")
    for k, v in list(cid_label_map.items())[:10]:
        print(" ", k, "->", v)

    cand_to_cid_cache = {}   # norm_candidate -> cid
    cid_to_label_cache = dict(cid_label_map)

    total_lines = 0
    written = 0
    batch_count = 0
    batch_jobs = []

    tmp_out = out_path + ".tmp"
    mode = 'a' if tmp_mode_append else 'w'
    fout = open(tmp_out, mode, encoding='utf-8')

    try:
        with open(input_path, 'r', encoding='utf-8') as fin:
            for raw_line in tqdm(fin, desc="Reading LDJSON"):
                if sample and total_lines >= sample:
                    break
                total_lines += 1
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                job_id = None
                for pid in ("id","job_id","_id","jobId","uuid"):
                    if pid in rec:
                        job_id = str(rec[pid]); break
                if not job_id:
                    job_id = f"file_{total_lines}"

                raw_text = get_job_text(rec)
                if not raw_text or len(raw_text.strip()) < 20:
                    continue

                try:
                    candidates_raw = extract_skills_from_jd_text(raw_text, prefer_sectioned=True, max_candidates=max_candidates_per_jd)
                except Exception:
                    candidates_raw = _quick_extract_fallback(raw_text)

                candidates_filt = filter_candidates(candidates_raw)
                if len(candidates_filt) > max_candidates_per_jd:
                    candidates_filt = candidates_filt[:max_candidates_per_jd]

                batch_jobs.append({
                    "job_id": job_id,
                    "title": rec.get("jobtitle") or rec.get("title") or rec.get("jobTitle"),
                    "company": rec.get("company"),
                    "location": rec.get("location"),
                    "raw_text_snippet": raw_text[:1000],
                    "candidates": candidates_filt,
                    "_raw_text": raw_text
                })

                if len(batch_jobs) >= batch_size:
                    batch_count += 1
                    t0 = time.time()

                    all_cands = []
                    for bj in batch_jobs:
                        all_cands.extend(bj["candidates"])
                    uniq = []
                    seen = set()
                    for c in all_cands:
                        norm = normalize_text(c)
                        if not norm:
                            continue
                        if norm not in seen:
                            seen.add(norm)
                            uniq.append((c, norm))

                    to_add = [orig for (orig, norm) in uniq if norm not in cand_to_cid_cache]
                    mapping_batch = {}
                    if to_add:
                        try:
                            add_map = reg.add_phrases(to_add, generate_candidates_fn=generate_candidates)
                            for cand, cid in add_map.items():
                                n = normalize_text(cand)
                                cand_to_cid_cache[n] = cid
                                mapping_batch[n] = cid
                        except Exception:
                            for cand in to_add:
                                n = normalize_text(cand)
                                cid = str(uuid.uuid4())
                                cand_to_cid_cache[n] = cid
                                mapping_batch[n] = cid

                    for orig, norm in uniq:
                        if norm in cand_to_cid_cache:
                            mapping_batch.setdefault(norm, cand_to_cid_cache[norm])

                    needed_cids = set(mapping_batch.values())
                    for cid in list(needed_cids):
                        if cid in cid_to_label_cache:
                            continue
                        try:
                            resp = reg.col.get(ids=[cid], include=['metadatas'])
                            metas = resp.get('metadatas', [[]])[0] if isinstance(resp.get('metadatas'), list) else []
                            meta0 = metas[0] if metas and metas[0] else {}
                            label = meta0.get('label') if meta0 else None
                        except Exception:
                            label = None
                        if not label:
                            label = cid_to_label_cache.get(cid) or None
                        cid_to_label_cache[cid] = label

                    for bj in batch_jobs:
                        mapped_ids = []
                        mapped_labels = []
                        for c in bj["candidates"]:
                            n = normalize_text(c)
                            cid = cand_to_cid_cache.get(n)
                            if cid:
                                mapped_ids.append(cid)
                                mapped_labels.append(cid_to_label_cache.get(cid))
                        seen_m = set(); uniq_mids = []; uniq_labels = []
                        for i, cid in enumerate(mapped_ids):
                            if cid not in seen_m:
                                seen_m.add(cid)
                                uniq_mids.append(cid)
                                uniq_labels.append(mapped_labels[i])
                        out_rec = {
                            "job_id": bj["job_id"],
                            "title": bj.get("title"),
                            "company": bj.get("company"),
                            "location": bj.get("location"),
                            "raw_text_snippet": bj.get("raw_text_snippet"),
                            "candidates": bj.get("candidates"),
                            "candidate_count": len(bj.get("candidates", [])),
                            "canonical_ids": uniq_mids,
                            "canonical_labels": uniq_labels
                        }
                        fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        written += 1

                    fout.flush()
                    os.fsync(fout.fileno())
                    if batch_count % persist_every_batches == 0:
                        try:
                            reg.client.persist()
                        except Exception:
                            pass

                    t1 = time.time()
                    print(f"[batch {batch_count}] processed {len(batch_jobs)} JDs in {t1-t0:.2f}s (avg {(t1-t0)/len(batch_jobs):.2f}s/JD). total written: {written}")

                    batch_jobs = []

                    if written > 0 and written % checkpoint_every == 0:
                        fout.flush()
                        os.fsync(fout.fileno())

        if batch_jobs:
            t0 = time.time()
            all_cands = []
            for bj in batch_jobs:
                all_cands.extend(bj["candidates"])
            seen = set(); uniq = []
            for c in all_cands:
                n = normalize_text(c)
                if not n:
                    continue
                if n not in seen:
                    seen.add(n)
                    uniq.append((c, n))

            to_add = [orig for (orig, norm) in uniq if norm not in cand_to_cid_cache]
            if to_add:
                try:
                    add_map = reg.add_phrases(to_add, generate_candidates_fn=generate_candidates)
                    for cand, cid in add_map.items():
                        n = normalize_text(cand)
                        cand_to_cid_cache[n] = cid
                except Exception:
                    for cand in to_add:
                        n = normalize_text(cand)
                        cid = str(uuid.uuid4())
                        cand_to_cid_cache[n] = cid

            needed_cids = set()
            for bj in batch_jobs:
                for c in bj["candidates"]:
                    cid = cand_to_cid_cache.get(normalize_text(c))
                    if cid:
                        needed_cids.add(cid)
            for cid in list(needed_cids):
                if cid in cid_to_label_cache:
                    continue
                try:
                    resp = reg.col.get(ids=[cid], include=['metadatas'])
                    metas = resp.get('metadatas', [[]])[0] if isinstance(resp.get('metadatas'), list) else []
                    meta0 = metas[0] if metas and metas[0] else {}
                    label = meta0.get('label') if meta0 else None
                except Exception:
                    label = None
                cid_to_label_cache[cid] = label

            for bj in batch_jobs:
                mapped_ids = []
                mapped_labels = []
                for c in bj["candidates"]:
                    cid = cand_to_cid_cache.get(normalize_text(c))
                    if cid:
                        mapped_ids.append(cid)
                        mapped_labels.append(cid_to_label_cache.get(cid))
                seen_m = set(); uniq_mids = []; uniq_labels = []
                for i, cid in enumerate(mapped_ids):
                    if cid not in seen_m:
                        seen_m.add(cid)
                        uniq_mids.append(cid)
                        uniq_labels.append(mapped_labels[i])
                out_rec = {
                    "job_id": bj["job_id"],
                    "title": bj.get("title"),
                    "company": bj.get("company"),
                    "location": bj.get("location"),
                    "raw_text_snippet": bj.get("raw_text_snippet"),
                    "candidates": bj.get("candidates"),
                    "candidate_count": len(bj.get("candidates", [])),
                    "canonical_ids": uniq_mids,
                    "canonical_labels": uniq_labels
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                written += 1

            try:
                reg.client.persist()
            except Exception:
                pass

            t1 = time.time()
            print(f"[final batch] processed {len(batch_jobs)} JDs in {t1-t0:.2f}s (avg {(t1-t0)/len(batch_jobs):.2f}s/JD). total written: {written}")

    finally:
        try:
            fout.flush()
            os.fsync(fout.fileno())
        except Exception:
            pass
        fout.close()
        try:
            os.replace(tmp_out, out_path)
        except Exception:
            with open(tmp_out, 'r', encoding='utf-8') as fr, open(out_path, 'w', encoding='utf-8') as fw:
                fw.write(fr.read())
        try:
            reg.client.persist()
        except Exception:
            pass

    print(f"Done. Total JDs processed/written: {written} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PromptCloud LDJSON -> processed_jds.jsonl (batch canonicalization).")
    parser.add_argument("--input", "-i", required=True, help="path to promptcloud ldjson file")
    parser.add_argument("--out", "-o", default="processed_jds.jsonl", help="output jsonl path")
    parser.add_argument("--sample", "-n", type=int, default=None, help="max JDs to process (for dev)")
    parser.add_argument("--batch-size", type=int, default=128, help="number of JDs to batch for canonicalization")
    parser.add_argument("--persist-every-batches", type=int, default=2, help="persist chroma every N batches")
    parser.add_argument("--max-candidates-per-jd", type=int, default=60, help="max candidates extracted per JD")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="checkpoint tmp file flush frequency")
    parser.add_argument("--tmp-append", action="store_true", help="open tmp output in append mode (resume). Default is fresh write.")
    args = parser.parse_args()

    process_file(args.input, args.out, sample=args.sample,
                 batch_size=args.batch_size,
                 persist_every_batches=args.persist_every_batches,
                 max_candidates_per_jd=args.max_candidates_per_jd,
                 checkpoint_every=args.checkpoint_every,
                 tmp_mode_append=args.tmp_append)
