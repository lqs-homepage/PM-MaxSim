#!/usr/bin/env python3
"""
Layer 3 Corpus #6 Builder: BGE-M3 1024D Native Hybrid (150K Real Passages x 1024D ColBERT Tokens + 1024D Dense + Native Lexical Sparse)
Covers:
  - W2: High-d (d=1024) Late-Interaction MaxSim demonstrating 2.81x tighter spherical Beta concentration (Theorem 3)!
  - W3: Dynamic Runtime Hybrid Scoring `w0*Sparse_Lexical + w1*Dense_1024 + w2*ColBERT_1024` + SQL predicates!
  - W4: Certified Semantic Threshold Selection (`S(Q, r) >= tau`)!
  - Large-scale external DB comparison (`~25-35 GB` raw FP16, `60-100+ GB` in external DBs under 4/8/16 GB cgroup caps!).
"""
import json
import os
import pathlib
import subprocess
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

P09 = pathlib.Path(os.environ.get("P09_ROOT", "${P09_ROOT}"))
RAW_DIR = P09 / "stage3_v4" / "raw_bgem3"
IDX_DIR = P09 / "stage3_v4" / "indexes" / "bgem3_1024d_hybrid"
BUILDER = P09 / "stage3_v4" / "src" / "certvec_builder_v5"

N_DOCS = int(os.environ.get("BGEM3_N_DOCS", "150000"))
N_QUERIES = int(os.environ.get("BGEM3_NQ", "1000"))
DIM = 1024
K_TOK = 16384   # 33.6 MB C_bf16 — strictly L3-resident on Emerald Rapids (MAJOR-03 contract!)
K_DENSE = 4096
MODEL_NAME = "BAAI/bge-m3"


class BGEM3Wrapper(nn.Module):
    """Lightweight native BGE-M3 wrapper outputting Dense (1024D), ColBERT Multi-Vector (1024D), and Sparse Lexical weights!"""
    def __init__(self, model_name: str, device: str):
        super().__init__()
        from huggingface_hub import hf_hub_download
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()
        self.colbert_linear = nn.Linear(DIM, DIM, bias=True).to(device=device, dtype=torch.bfloat16).eval()
        self.sparse_linear = nn.Linear(DIM, 1, bias=True).to(device=device, dtype=torch.bfloat16).eval()
        cb_path = hf_hub_download(model_name, "colbert_linear.pt")
        sp_path = hf_hub_download(model_name, "sparse_linear.pt")
        self.colbert_linear.load_state_dict(torch.load(cb_path, map_location=device, weights_only=True))
        self.sparse_linear.load_state_dict(torch.load(sp_path, map_location=device, weights_only=True))

    @torch.no_grad()
    def encode_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dense = F.normalize(h[:, 0].float(), p=2, dim=1)
        colb = self.colbert_linear(h[:, 1:])  # drop CLS, keep token positions
        colb = F.normalize(colb.float(), p=2, dim=-1)
        # ReLU lexical weight per token id
        lex_w = torch.relu(self.sparse_linear(h).squeeze(-1)).float() * attention_mask.float()
        return dense, colb, lex_w


def random_haar_orthogonal_f64(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((d, d))
    q, r = np.linalg.qr(z)
    diag = np.sign(np.diag(r))
    diag[diag == 0] = 1.0
    return (q * diag).astype(np.float64)


def gpu_spherical_kmeans(vecs_f16: np.ndarray, K: int, device: str = "cuda:0", n_sample: int = 1_500_000, iters: int = 12) -> np.ndarray:
    rng = np.random.default_rng(20261127)
    n = len(vecs_f16)
    if n > n_sample:
        idx = rng.choice(n, size=n_sample, replace=False)
        sample = torch.from_numpy(vecs_f16[idx].astype(np.float32)).to(device)
    else:
        sample = torch.from_numpy(vecs_f16.astype(np.float32)).to(device)
    sample = F.normalize(sample, p=2, dim=1)
    C = sample[torch.randperm(len(sample), device=device)[:K]].clone()
    bs = 65536
    for _ in range(iters):
        sums = torch.zeros_like(C)
        cnts = torch.zeros(K, device=device, dtype=torch.float32)
        for s in range(0, len(sample), bs):
            x = sample[s:s + bs]
            a = torch.argmax(x @ C.T, dim=1)
            sums.index_add_(0, a, x)
            cnts.index_add_(0, a, torch.ones(len(x), device=device))
        empty = cnts < 1.0
        if empty.any():
            n_emp = int(empty.sum().item())
            sums[empty] = sample[torch.randint(0, len(sample), (n_emp,), device=device)]
            cnts[empty] = 1.0
        C = F.normalize(sums / cnts.unsqueeze(1), p=2, dim=1)
    return C.cpu().numpy().astype(np.float32)


def gpu_assign_centroids(vecs_f16: np.ndarray, C_f32: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    C = torch.from_numpy(C_f32).to(device).half()
    c_norm2 = (C.float().pow(2).sum(dim=1) * 0.5).half()
    out = np.empty(len(vecs_f16), dtype=np.uint16)
    bs = 65536
    for s in range(0, len(vecs_f16), bs):
        e = min(len(vecs_f16), s + bs)
        x = torch.from_numpy(vecs_f16[s:e]).to(device)
        out[s:e] = torch.argmax(x @ C.T - c_norm2.unsqueeze(0), dim=1).cpu().numpy().astype(np.uint16)
    return out


def main():
    t0 = time.time()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    (IDX_DIR / "field_0").mkdir(parents=True, exist_ok=True)
    (IDX_DIR / "field_1").mkdir(parents=True, exist_ok=True)

    print(f"[Stage 1] Loading {N_DOCS} real BeIR/msmarco passages + {N_QUERIES} Dev queries...", flush=True)
    ds_corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus", streaming=True)
    passages = []
    doc_char_lens = []
    for row in ds_corpus:
        txt = (row.get("title", "") + " " + row.get("text", "")).strip()
        if not txt:
            continue
        passages.append(txt)
        doc_char_lens.append(len(txt))
        if len(passages) >= N_DOCS:
            break

    ds_queries = load_dataset("BeIR/msmarco", "queries", split="queries", streaming=True)
    queries = []
    for row in ds_queries:
        txt = row.get("text", "").strip()
        if txt:
            queries.append(txt)
        if len(queries) >= N_QUERIES:
            break

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_ids = set(tok.all_special_ids)
    dev = "cuda:0"
    model = BGEM3Wrapper(MODEL_NAME, dev)

    # 2. Encode 1,000 queries (Dense 1024D, ColBERT up to 32 tokens x 1024D, Sparse lexical weights)
    print(f"[Stage 2] Encoding {len(queries)} queries with native {MODEL_NAME}...", flush=True)
    q_dense = np.zeros((len(queries), DIM), dtype=np.float32)
    q_colb = np.zeros((len(queries), 32, DIM), dtype=np.float32)
    q_lens = np.zeros(len(queries), dtype=np.uint32)
    q_sparse = []
    q_vocab = set()
    for s in range(0, len(queries), 128):
        e = min(len(queries), s + 128)
        enc = tok(queries[s:e], padding=True, truncation=True, max_length=33, return_tensors="pt").to(dev)
        dn, cb, lw = model.encode_batch(enc["input_ids"], enc["attention_mask"])
        q_dense[s:e] = dn.cpu().numpy()
        ids_np = enc["input_ids"].cpu().numpy()
        mask_np = enc["attention_mask"].cpu().numpy()
        cb_np = cb.cpu().numpy()
        lw_np = lw.cpu().numpy()
        for bi in range(e - s):
            qi = s + bi
            # Valid non-special colbert tokens (excluding CLS & EOS)
            vlen = int(mask_np[bi].sum()) - 2
            vlen = max(1, min(32, vlen))
            q_colb[qi, :vlen] = cb_np[bi, :vlen]
            q_lens[qi] = vlen
            sp = {}
            for pos in range(int(mask_np[bi].sum())):
                tid = int(ids_np[bi, pos])
                w = float(lw_np[bi, pos])
                if tid not in special_ids and w > 0.0:
                    if w > sp.get(tid, 0.0):
                        sp[tid] = w
                        q_vocab.add(tid)
            q_sparse.append(sp)

    # 3. Encode 150K documents streaming directly to disk (f0_vecs.f16 ~20-30 GB) + f1_dense (300 MB) + exact sparse postings
    print(f"[Stage 3] Encoding {N_DOCS} docs (1024D ColBERT tokens + 1024D Dense + Native Sparse)...", flush=True)
    f0_path = RAW_DIR / "f0_vecs.f16"
    doc_dense = np.empty((N_DOCS, DIM), dtype=np.float16)
    tok_lens = np.zeros(N_DOCS, dtype=np.uint32)
    sparse_postings = defaultdict(list)
    total_tokens = 0
    bs = 256
    with open(f0_path, "wb") as f0_out:
        for s in range(0, N_DOCS, bs):
            e = min(N_DOCS, s + bs)
            enc = tok(passages[s:e], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
            dn, cb, lw = model.encode_batch(enc["input_ids"], enc["attention_mask"])
            doc_dense[s:e] = dn.half().cpu().numpy()
            ids_np = enc["input_ids"].cpu().numpy()
            mask_np = enc["attention_mask"].cpu().numpy()
            cb_h = cb.half().cpu().numpy()
            lw_np = lw.cpu().numpy()
            for bi in range(e - s):
                rid = s + bi
                slen = int(mask_np[bi].sum())
                vlen = max(1, min(126, slen - 2))
                tok_lens[rid] = vlen
                f0_out.write(cb_h[bi, :vlen].tobytes())
                total_tokens += vlen
                # Record max lexical weight per token id in query vocab
                loc_sp = {}
                for pos in range(slen):
                    tid = int(ids_np[bi, pos])
                    if tid in q_vocab:
                        w = float(lw_np[bi, pos])
                        if w > loc_sp.get(tid, 0.0):
                            loc_sp[tid] = w
                for tid, w in loc_sp.items():
                    sparse_postings[tid].append((rid, w))
            if (s // bs) % 100 == 0:
                print(f"  encoded {e}/{N_DOCS} docs, total_1024d_tokens={total_tokens} ({total_tokens*2048/1e9:.2f} GB FP16)", flush=True)

    print(f"[Stage 3 Done] total_1024d_tokens={total_tokens} ({total_tokens*2048/1e9:.2f} GB FP16), elapsed={time.time()-t0:.1f}s", flush=True)

    # Compute exact native BGE-M3 sparse lexical scores per query
    sp_rids_list, sp_scores_list = [], []
    for qi, qsp in enumerate(q_sparse):
        acc = {}
        for tid, qw in qsp.items():
            for rid, dw in sparse_postings.get(tid, []):
                acc[rid] = acc.get(rid, 0.0) + qw * dw
        items = sorted(acc.items(), key=lambda x: -x[1])[:5000]
        sp_rids_list.append(np.array([r for r, _ in items], dtype=np.uint32))
        sp_scores_list.append(np.array([v for _, v in items], dtype=np.float32))

    # K-means topics (5 semantic topics on dense vectors for real topic predicates!)
    C_topic = gpu_spherical_kmeans(doc_dense, 5, device=dev, n_sample=N_DOCS, iters=10)
    doc_topics = gpu_assign_centroids(doc_dense, C_topic, device=dev).astype(np.uint8)

    # 4. Spherical K-means + CERTVEC5 Build for Field 0 (1024D tokens, K=16384) & Field 1 (1024D dense, K=4096)
    print("[Stage 4] Spherical K-means (K=16384 tokens, K=4096 dense) + CERTVEC5 build...", flush=True)
    tok_mmap = np.memmap(f0_path, dtype=np.float16, mode="r", shape=(total_tokens, DIM))
    C0 = gpu_spherical_kmeans(tok_mmap, K_TOK, device=dev, n_sample=1_200_000, iters=12)
    cids0 = gpu_assign_centroids(tok_mmap, C0, device=dev)
    P0 = random_haar_orthogonal_f64(DIM, seed=202611280)

    tok_lens.tofile(RAW_DIR / "f0_lens.u32")
    cids0.tofile(RAW_DIR / "f0_cids.u16")
    C0.tofile(RAW_DIR / "f0_C.f32")
    P0.tofile(RAW_DIR / "f0_P.f64")

    subprocess.run([
        str(BUILDER), "--build-field", str(IDX_DIR / "field_0"),
        str(N_DOCS), str(DIM), str(K_TOK), "1", "0",
        str(f0_path), str(RAW_DIR / "f0_lens.u32"),
        str(RAW_DIR / "f0_cids.u16"), str(RAW_DIR / "f0_C.f32"), str(RAW_DIR / "f0_P.f64"), "0"
    ], check=True)

    C1 = gpu_spherical_kmeans(doc_dense, K_DENSE, device=dev, n_sample=N_DOCS, iters=12)
    cids1 = gpu_assign_centroids(doc_dense, C1, device=dev)
    P1 = random_haar_orthogonal_f64(DIM, seed=202611281)
    ones_u32 = np.ones(N_DOCS, dtype=np.uint32)
    doc_dense.tofile(RAW_DIR / "f1_vecs.f16")
    ones_u32.tofile(RAW_DIR / "f1_lens.u32")
    cids1.tofile(RAW_DIR / "f1_cids.u16")
    C1.tofile(RAW_DIR / "f1_C.f32")
    P1.tofile(RAW_DIR / "f1_P.f64")

    subprocess.run([
        str(BUILDER), "--build-field", str(IDX_DIR / "field_1"),
        str(N_DOCS), str(DIM), str(K_DENSE), "1", "0",
        str(RAW_DIR / "f1_vecs.f16"), str(RAW_DIR / "f1_lens.u32"),
        str(RAW_DIR / "f1_cids.u16"), str(RAW_DIR / "f1_C.f32"), str(RAW_DIR / "f1_P.f64"), "1"
    ], check=True)

    subprocess.run([str(BUILDER), "--write-manifest", str(IDX_DIR), str(N_DOCS), "2"], check=True)

    # 5. Save queries, sparse lexical scores, metadata, and exact FP32 GPU Ground Truth for 100 eval queries!
    np.save(IDX_DIR / "q_dense_f32.npy", q_dense)
    np.save(IDX_DIR / "q_colb_f32.npy", q_colb)
    np.save(IDX_DIR / "q_lens.npy", q_lens)
    np.save(IDX_DIR / "doc_topics.npy", doc_topics)
    np.save(IDX_DIR / "doc_char_lens.npy", np.array(doc_char_lens, dtype=np.uint32))
    np.savez_compressed(
        IDX_DIR / "sparse_lexical_1000q.npz",
        offs=np.cumsum([0] + [len(x) for x in sp_rids_list]).astype(np.int64),
        rids=np.concatenate(sp_rids_list) if sp_rids_list else np.zeros(0, dtype=np.uint32),
        scores=np.concatenate(sp_scores_list) if sp_scores_list else np.zeros(0, dtype=np.float32),
    )

    print("[Stage 5] Computing 100% exact FP32 GPU ground truth (100q x 150K x 1024D MaxSim + Dense + Sparse)...", flush=True)
    Nq_gt = 100
    tok_offs = np.cumsum([0] + tok_lens.tolist())
    colb_scores = np.zeros((Nq_gt, N_DOCS), dtype=np.float32)
    dense_scores = np.zeros((Nq_gt, N_DOCS), dtype=np.float32)
    Qd = torch.from_numpy(q_dense[:Nq_gt]).to(dev)
    with torch.no_grad():
        for ds in range(0, N_DOCS, 32768):
            de = min(N_DOCS, ds + 32768)
            dense_scores[:, ds:de] = (Qd @ torch.from_numpy(doc_dense[ds:de]).to(dev).float().T).cpu().numpy()
        # Exact MaxSim over 1024D tokens in batches of 1024 docs
        Qc = torch.from_numpy(q_colb[:Nq_gt]).to(dev)  # [100, 32, 1024]
        Qc_flat = Qc.reshape(Nq_gt * 32, DIM)
        for ds in range(0, N_DOCS, 512):
            de = min(N_DOCS, ds + 512)
            cs, ce = int(tok_offs[ds]), int(tok_offs[de])
            X = torch.from_numpy(np.array(tok_mmap[cs:ce])).to(dev).float()  # [M_b, 1024]
            S = (Qc_flat @ X.T).reshape(Nq_gt, 32, -1).cpu().numpy()
            for d_i in range(ds, de):
                ls, le = int(tok_offs[d_i]) - cs, int(tok_offs[d_i + 1]) - cs
                sub = S[:, :, ls:le].max(axis=2)  # [Nq_gt, 32]
                for qi in range(Nq_gt):
                    colb_scores[qi, d_i] = sub[qi, :int(q_lens[qi])].sum()

    sp_mat = np.zeros((Nq_gt, N_DOCS), dtype=np.float32)
    for qi in range(Nq_gt):
        if len(sp_rids_list[qi]) > 0:
            sp_mat[qi, sp_rids_list[qi]] = sp_scores_list[qi]

    rng = np.random.default_rng(20261128)
    char_arr = np.array(doc_char_lens, dtype=np.uint32)
    masks = {
        "NoCorr_100pct": np.ones(N_DOCS, dtype=bool),
        "NoCorr_5pct": rng.random(N_DOCS) < 0.05,
        "Nat_Topic0_Eq": doc_topics == 0,
        "Nat_DocLen_Top10pct": char_arr >= np.percentile(char_arr, 90),
    }
    for mname, mask in masks.items():
        vidx = np.where(mask)[0]
        gt_colb = np.zeros((Nq_gt, 10), dtype=np.uint32)
        gt_hyb1 = np.zeros((Nq_gt, 10), dtype=np.uint32)  # BGE-M3 official ratio: 0.2*Sparse + 0.4*Dense + 0.4*ColBERT_avg
        gt_hyb2 = np.zeros((Nq_gt, 10), dtype=np.uint32)  # Lexical-heavy dynamic weight: 0.5*Sparse + 0.2*Dense + 0.3*ColBERT_avg
        w4_taus = np.zeros(Nq_gt, dtype=np.float64)
        for qi in range(Nq_gt):
            ql = float(q_lens[qi])
            sc_c = colb_scores[qi, vidx]
            sc_c_avg = sc_c / ql
            sc_d = dense_scores[qi, vidx]
            sc_s = sp_mat[qi, vidx]
            h1 = 0.2 * sc_s + 0.4 * sc_d + 0.4 * sc_c_avg
            h2 = 0.5 * sc_s + 0.2 * sc_d + 0.3 * sc_c_avg
            gt_colb[qi] = vidx[np.argsort(-sc_c)[:10]]
            gt_hyb1[qi] = vidx[np.argsort(-h1)[:10]]
            gt_hyb2[qi] = vidx[np.argsort(-h2)[:10]]
            kth = min(50, len(sc_c) - 1)
            w4_taus[qi] = float(np.partition(sc_c, -kth)[-kth])
        np.savez_compressed(
            IDX_DIR / f"gt_{mname}.npz",
            mask=mask, gt_colb=gt_colb, gt_hyb1=gt_hyb1, gt_hyb2=gt_hyb2, w4_taus=w4_taus,
        )

    del tok_mmap
    f0_path.unlink(missing_ok=True)
    (RAW_DIR / "f1_vecs.f16").unlink(missing_ok=True)
    with open(IDX_DIR / "BUILD_DONE.json", "w") as f:
        json.dump({
            "elapsed_s": round(time.time() - t0, 1), "N_docs": N_DOCS,
            "total_1024d_tokens": total_tokens, "fp16_gb": round(total_tokens * 2048 / 1e9, 2),
            "K_tok": K_TOK, "K_dense": K_DENSE,
        }, f, indent=2)
    print(f"[ALL DONE] bgem3_1024d_hybrid built cleanly in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
