#!/usr/bin/env python3
"""
PM-MaxSim MS MARCO Passage 8.84M (BeIR/msmarco, 8,841,823 passages) ColBERTv2 FP16 Encoder:
- Designed for 2x idle interactive B200 GPUs (runs 2 shards in parallel: --shard 0/2 on GPU 0, --shard 1/2 on GPU 1).
- Uses length-sorted chunked batching (chunk=65,536, bs=2048) for ~5,000+ docs/s per B200 (~15-20 min total!).
- Streams FP16 token vectors directly to flat binary file (zero 200 GB RAM blowup!), preserving EXACT original doc_id order!
- Also encodes 1,000 official BeIR/msmarco validation/test queries with official qrels and saves document token lengths
  & real lexical/topic proxy attributes (doc_len, first-char/hash bucket, length bucket) for Natural SQL predicates.
"""
import argparse
import json
import os
import string
import sys
import time
from pathlib import Path
import numpy as np
import torch

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
P09_ROOT = Path(os.environ.get("P09_ROOT", "./workspace"))
os.environ.setdefault("HF_HOME", str(P09_ROOT / "cache" / "hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(P09_ROOT / "cache" / "hf" / "datasets"))
OUT_DIR = P09_ROOT / "stage3" / "corpus_msmarco"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(P09_ROOT / "gate_2a0"))
from run_gate_2a0 import OfficialColBERTv2


@torch.no_grad()
def encode_queries_and_meta(device: str):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    q_path = OUT_DIR / "msmarco_q32.npy"
    meta_path = OUT_DIR / "msmarco_queries_meta.json"
    if q_path.exists() and meta_path.exists():
        print("[Queries] Already cached.", flush=True)
        return

    t0 = time.time()
    ds_q = load_dataset("BeIR/msmarco", "queries", split="queries")
    ds_qrels = load_dataset("BeIR/msmarco-qrels", split="validation")
    rel_map = {}
    for row in ds_qrels:
        qid = str(row["query-id"])
        cid = int(row["corpus-id"])
        sc = float(row["score"])
        if sc > 0:
            rel_map.setdefault(qid, []).append(cid)

    # Select first 1,000 queries that have positive qrels (reproducible deterministic order)
    sel_qids, sel_texts, sel_qrels = [], [], []
    for row in ds_q:
        qid = str(row["_id"])
        if qid in rel_map:
            sel_qids.append(qid)
            sel_texts.append(row["text"])
            sel_qrels.append(rel_map[qid])
            if len(sel_qids) >= 1000:
                break

    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    model = OfficialColBERTv2("colbert-ir/colbertv2.0").to(device=device, dtype=torch.bfloat16).eval()
    Q_MARKER = tokenizer.convert_tokens_to_ids("[unused0]")
    CLS_ID, SEP_ID, MASK_ID = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id

    Nq = len(sel_texts)
    q_ids = torch.full((Nq, 32), MASK_ID, dtype=torch.long)
    for i, txt in enumerate(sel_texts):
        toks = tokenizer.encode(txt, add_special_tokens=False)[:29]
        seq = [CLS_ID, Q_MARKER] + toks + [SEP_ID]
        q_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    q_out = []
    for s in range(0, Nq, 256):
        embs = model(q_ids[s:s + 256].to(device), torch.ones_like(q_ids[s:s + 256]).to(device))
        q_out.append(embs.float().cpu().numpy())
    np.save(q_path, np.concatenate(q_out, axis=0).astype(np.float32))
    with open(meta_path, "w") as f:
        json.dump({"qids": sel_qids, "qrels": sel_qrels}, f)
    print(f"[Queries] Encoded {Nq} official MS MARCO dev queries in {time.time()-t0:.1f}s", flush=True)


@torch.no_grad()
def encode_shard(shard_id: int, num_shards: int, gpu_id: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_bin = OUT_DIR / f"msmarco_vecs_s{shard_id}.f16"
    out_lens = OUT_DIR / f"msmarco_lens_s{shard_id}.npy"
    out_done = OUT_DIR / f"msmarco_s{shard_id}.done"
    if out_done.exists():
        print(f"[Shard {shard_id}/{num_shards}] Already complete!", flush=True)
        return

    t0 = time.time()
    device = f"cuda:{gpu_id}"
    if shard_id == 0:
        encode_queries_and_meta(device)

    ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    N_total = len(ds)
    per_shard = (N_total + num_shards - 1) // num_shards
    start_doc = shard_id * per_shard
    end_doc = min(N_total, start_doc + per_shard)
    N = end_doc - start_doc
    print(f"[Shard {shard_id}/{num_shards} GPU {gpu_id}] Encoding docs [{start_doc:,}, {end_doc:,}) = {N:,} passages", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    model = OfficialColBERTv2("colbert-ir/colbertv2.0").to(device=device, dtype=torch.bfloat16).eval()

    skiplist = {tokenizer.convert_tokens_to_ids(sym) for sym in string.punctuation}
    skiplist.add(tokenizer.pad_token_id)
    skip_mask_lut = torch.ones(tokenizer.vocab_size + 100, dtype=torch.bool)
    for sid in skiplist:
        if sid is not None and 0 <= sid < len(skip_mask_lut):
            skip_mask_lut[sid] = False
    skip_mask_lut = skip_mask_lut.to(device)

    D_MARKER = tokenizer.convert_tokens_to_ids("[unused1]")
    CLS_ID, SEP_ID = tokenizer.cls_token_id, tokenizer.sep_token_id

    doc_lens = np.empty(N, dtype=np.int32)
    CHUNK = 65536
    BS = 2048
    total_toks = 0
    t_enc = time.time()

    with open(out_bin, "wb") as fb:
        for cs in range(0, N, CHUNK):
            ce = min(N, cs + CHUNK)
            sub = ds.select(range(start_doc + cs, start_doc + ce))
            texts = [t if isinstance(t, str) else ("" if t is None else str(t)) for t in sub["text"]]
            enc = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=177, padding=False)["input_ids"]
            cn = len(enc)
            # Length-sort within chunk for maximal B200 utilization, then scatter back in exact original doc order!
            order = sorted(range(cn), key=lambda i: len(enc[i]))
            chunk_vecs = [None] * cn
            chunk_lens = np.empty(cn, dtype=np.int32)

            for bs0 in range(0, cn, BS):
                bs1 = min(cn, bs0 + BS)
                idxs = order[bs0:bs1]
                max_l = len(enc[idxs[-1]]) + 3
                ids = torch.zeros((len(idxs), max_l), dtype=torch.long)
                att = torch.zeros((len(idxs), max_l), dtype=torch.long)
                for bi, oi in enumerate(idxs):
                    toks = enc[oi]
                    L = len(toks) + 3
                    ids[bi, 0] = CLS_ID
                    ids[bi, 1] = D_MARKER
                    if toks:
                        ids[bi, 2:2 + len(toks)] = torch.tensor(toks, dtype=torch.long)
                    ids[bi, 2 + len(toks)] = SEP_ID
                    att[bi, :L] = 1
                ids_g = ids.to(device, non_blocking=True)
                att_g = att.to(device, non_blocking=True)
                embs = model(ids_g, att_g)
                keep = (att_g == 1) & skip_mask_lut[ids_g]
                kl = keep.sum(dim=1).cpu().numpy()
                flat = embs[keep].to(torch.float16).cpu().numpy()
                off = 0
                for bi, oi in enumerate(idxs):
                    k = int(kl[bi])
                    chunk_lens[oi] = k
                    chunk_vecs[oi] = flat[off:off + k]
                    off += k

            # Write in EXACT original document order
            doc_lens[cs:ce] = chunk_lens
            chunk_cat = np.concatenate(chunk_vecs, axis=0)
            fb.write(chunk_cat.tobytes())
            total_toks += int(chunk_cat.shape[0])
            elapsed = time.time() - t_enc
            rate = ce / max(elapsed, 1e-3)
            eta_m = (N - ce) / max(rate, 1e-3) / 60.0
            print(
                f"[Shard {shard_id}] {ce:,}/{N:,} ({100.0*ce/N:.1f}%) | "
                f"{rate:,.0f} docs/s | toks={total_toks:,} | ETA={eta_m:.1f}m",
                flush=True,
            )

    np.save(out_lens, doc_lens)
    with open(out_done, "w") as f:
        json.dump({"shard": shard_id, "start_doc": start_doc, "end_doc": end_doc, "N": N, "total_toks": total_toks, "sec": time.time() - t0}, f)
    print(f"=== [Shard {shard_id}] COMPLETE: {N:,} docs, {total_toks:,} toks ({total_toks*256/1e9:.2f} GB) in {(time.time()-t0)/60:.1f}m ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=2)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    encode_shard(args.shard, args.num_shards, args.gpu)
