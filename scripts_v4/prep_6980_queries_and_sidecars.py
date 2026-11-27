#!/usr/bin/env python3
"""
1. Encode ALL 6,980 official MS MARCO Dev queries (ColBERTv2, d=128) + load full LoTTE test queries
2. Generate deterministic 100,000-request Zipf(s=1.0) multi-tenant mixed SQL query streams
3. Build non-destructive read-only v4 sidecars (`frozen_postings.mmap` + exact `rho_max.bin`)
   for lotte_200k_matched, vidore_v3_adamerge, vidore_v3_raw, lotte_pooled, and msmarco_8m_lsm/{seg0,seg1,seg2}
"""
import json
import os
import pathlib
import subprocess
import time
import urllib.request
import tarfile

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

P09 = pathlib.Path(os.environ.get("P09_ROOT", "${P09_ROOT}"))
OUT_DIR = P09 / "stage3_v4" / "query_streams"
SIDECAR_ROOT = P09 / "stage3_v4" / "sidecars"
BUILDER = P09 / "stage3_v4" / "src" / "certvec_builder_v5"


def load_colbertv2_linear(model_name: str, device: str):
    from huggingface_hub import hf_hub_download
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()
    # ColBERTv2 checkpoint stores linear projection inside pytorch_model.bin (`linear.weight`)
    ckpt_path = hf_hub_download(model_name, "pytorch_model.bin")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    w = sd["linear.weight"].to(device=device, dtype=torch.bfloat16)  # [128, 768]
    return tok, enc, w


def encode_colbert_queries(queries: list[str], device: str = "cuda:0") -> np.ndarray:
    tok, enc, linear_w = load_colbertv2_linear("colbert-ir/colbertv2.0", device)
    # ColBERTv2 query marker "[Q] " prefix + 32-token fixed length with [MASK] augmentation
    q_marker_id = tok.convert_tokens_to_ids("[unused0]")
    mask_id = tok.mask_token_id
    N = len(queries)
    out = np.zeros((N, 32, 128), dtype=np.float32)
    bs = 256
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(N, s + bs)
            batch_ids = []
            batch_mask = []
            for qt in queries[s:e]:
                ids = tok.encode(qt, add_special_tokens=False)[:29]
                full = [tok.cls_token_id, q_marker_id] + ids + [tok.sep_token_id]
                while len(full) < 32:
                    full.append(mask_id)
                batch_ids.append(full[:32])
                batch_mask.append([1] * 32)
            t_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
            t_msk = torch.tensor(batch_mask, dtype=torch.long, device=device)
            h = enc(input_ids=t_ids, attention_mask=t_msk).last_hidden_state
            proj = F.linear(h, linear_w)
            proj = F.normalize(proj.float(), p=2, dim=-1)
            out[s:e] = proj.cpu().numpy()
    return out


def make_zipf_stream(n_queries: int, n_requests: int, n_tenants: int, s_param: float = 1.0, seed: int = 20261127):
    rng = np.random.default_rng(seed)
    # Zipf over queries AND Zipf over tenants (realistic multi-tenant SaaS skew!)
    q_ranks = np.arange(1, n_queries + 1, dtype=np.float64)
    q_probs = (1.0 / (q_ranks ** s_param))
    q_probs /= q_probs.sum()
    q_perm = rng.permutation(n_queries)
    q_ids = q_perm[rng.choice(n_queries, size=n_requests, p=q_probs)].astype(np.uint32)

    t_ranks = np.arange(1, n_tenants + 1, dtype=np.float64)
    t_probs = (1.0 / (t_ranks ** s_param))
    t_probs /= t_probs.sum()
    tenant_ids = rng.choice(n_tenants, size=n_requests, p=t_probs).astype(np.uint16)
    return q_ids, tenant_ids


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Load & encode ALL 6,980 official MS MARCO Passage Dev queries (directly from queries.tar.gz or BeIR/msmarco)!
    q_texts = []
    qdev_tar = OUT_DIR / "queries.tar.gz"
    if not qdev_tar.exists():
        url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"
        print(f"[Download] {url}", flush=True)
        urllib.request.urlretrieve(url, qdev_tar)
    with tarfile.open(qdev_tar, "r:gz") as tf:
        target = None
        for member in tf.getmembers():
            if member.name.endswith("queries.dev.tsv") or "dev" in member.name:
                target = member
                break
        if target is None:
            target = tf.getmembers()[0]
        print(f"[Tar] Reading {target.name} from queries.tar.gz", flush=True)
        with tf.extractfile(target) as f:
            for raw in f:
                parts = raw.decode("utf-8", errors="ignore").rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[1].strip():
                    q_texts.append(parts[1].strip())
                    if len(q_texts) >= 6980:
                        break
    print(f"[Encode] Encoding all {len(q_texts)} official MS MARCO Dev queries (ColBERTv2)...", flush=True)
    q_emb = encode_colbert_queries(q_texts, device="cuda:0")
    np.save(OUT_DIR / "msmarco_dev_6980_q32.npy", q_emb)

    # Generate 100,000-request Zipf(s=1.0) multi-tenant stream over 6,980 queries & 20 tenants
    rq_ids, rt_ids = make_zipf_stream(len(q_texts), 100000, 20, s_param=1.0, seed=20261127)
    np.savez_compressed(OUT_DIR / "zipf1_100k_stream_msmarco.npz", q_ids=rq_ids, tenant_ids=rt_ids)

    # Also generate 50,000-request Zipf(s=1.0) stream for lotte_200k (2000 query pool from existing core_index / LoTTE)
    rq_l, rt_l = make_zipf_stream(1000, 50000, 20, s_param=1.0, seed=20261128)
    np.savez_compressed(OUT_DIR / "zipf1_50k_stream_1000q.npz", q_ids=rq_l, tenant_ids=rt_l)

    # 2. Build read-only v4 sidecars (frozen_postings.mmap + exact rho_max.bin) for existing Stage 3 indexes!
    v4_targets = [
        ("lotte_200k_matched", P09 / "stage3" / "indexes" / "lotte_200k_matched"),
        ("vidore_v3_adamerge", P09 / "stage3" / "indexes" / "vidore_v3_adamerge"),
        ("vidore_v3_raw",      P09 / "stage3" / "indexes" / "vidore_v3_raw"),
        ("lotte_pooled",       P09 / "gate_2b" / "indexes_v3" / "lotte_pooled"),
        ("msmarco_seg0",       P09 / "stage3" / "indexes" / "msmarco_8m_lsm" / "seg0"),
        ("msmarco_seg1",       P09 / "stage3" / "indexes" / "msmarco_8m_lsm" / "seg1"),
        ("msmarco_seg2",       P09 / "stage3" / "indexes" / "msmarco_8m_lsm" / "seg2"),
    ]
    built = []
    for name, src_dir in v4_targets:
        if not (src_dir / "core_index.bin").exists():
            print(f"  [Skip missing] {name}: {src_dir}", flush=True)
            continue
        dst = SIDECAR_ROOT / name
        dst.mkdir(parents=True, exist_ok=True)
        if (dst / "frozen_postings.mmap").exists() and (dst / "rho_max.bin").exists():
            print(f"  [Exists] {name}", flush=True)
            built.append(name)
            continue
        subprocess.run([str(BUILDER), "--build-v4-sidecar", str(src_dir), str(dst)], check=True)
        built.append(name)

    with open(OUT_DIR / "PREP_DONE.json", "w") as f:
        json.dump({"elapsed_s": round(time.time() - t0, 1), "n_msmarco_dev_queries": len(q_texts), "sidecars_built": built}, f, indent=2)
    print(f"[ALL DONE] 6,980 queries + Zipf 100K streams + {len(built)} v4 sidecars done in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
