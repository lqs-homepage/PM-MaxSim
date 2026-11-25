#!/usr/bin/env python3
"""
PM-MaxSim Complete Matched-Scale & Benchmark-Calibrated Builders:
  1. --lotte200k  : Build stage3/indexes/lotte_200k_matched/ (exact 200,000 docs matching VectorChord/Qdrant/Weaviate
                    with identical dom, n_toks, nc5 seed=202703, K=16,384, tier1_b1.ram, m_norm.bin,
                    and 100% EXACT IEEE-754 FP32 GPU Top-100 GT workloads!)
  2. --msmarco-gt : Upgrade stage3/indexes/msmarco_8m_lsm/global_workloads_gt.bin + global_gt100.bin to full 16 workloads
                    (adding NoCorr_0.10pct, NoCorr_30.00pct, ClusteredNeg_5.00pct, and AdvDeadPost_30.00pct with 0% hot-doc
                    survival across all 3 LSM segments!) + exact FP32 MRR@10 / Recall@100 anchor on official dev qrels!
  3. --vidore-ir  : Compute exact IEEE-754 FP32 downstream nDCG@5 & MRR@10 on ViDoRe v3 (16,868 pages, 140 queries)
                    comparing Raw ColQwen2.5 vs AdaMerge-258 on official vidore_v3_qrels.json!
"""
import argparse
import json
import math
import os
import struct
import subprocess
import time
from pathlib import Path
import numpy as np
import torch

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

P09 = Path(os.environ.get("P09_ROOT", "./workspace"))
RABITQ_V3 = P09 / "gate_2b" / "rabitq_nvme_builder_v3"
B1_V4 = P09 / "stage3" / "src" / "tier1_b1_builder_v4"


def pack_bitmap(mask: np.ndarray) -> np.ndarray:
    N = len(mask)
    nw = (N + 63) // 64
    out = np.zeros(nw, dtype=np.uint64)
    idx = np.nonzero(mask)[0]
    np.bitwise_or.at(out, idx >> 6, np.uint64(1) << (idx & 63).astype(np.uint64))
    return out


def make_haar_orthogonal(d=128, seed=202703):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return np.ascontiguousarray(Q * signs[None, :], dtype=np.float32)


def compute_ndcg_at_k(ranked_ids, rel_dict, k=5):
    dcg = 0.0
    for rank, did in enumerate(ranked_ids[:k]):
        rel = rel_dict.get(str(int(did)), rel_dict.get(int(did), 0.0))
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(rank + 2.0)
    ideal = sorted([float(v) for v in rel_dict.values() if float(v) > 0], reverse=True)[:k]
    idcg = sum((2.0 ** rel - 1.0) / math.log2(rank + 2.0) for rank, rel in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


def compute_mrr_at_k(ranked_ids, pos_set, k=10):
    for rank, did in enumerate(ranked_ids[:k]):
        if int(did) in pos_set:
            return 1.0 / (rank + 1.0)
    return 0.0


@torch.no_grad()
def build_lotte_200k_matched(device: str = "cuda:0"):
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    src_dir = P09 / "gate_2b" / "indexes_v3" / "lotte_pooled"
    out_dir = P09 / "stage3" / "indexes" / "lotte_200k_matched"
    out_dir.mkdir(parents=True, exist_ok=True)
    N_use = 200000
    K = 16384

    print(f"=== [Part A] Building lotte_200k_matched (N={N_use:,}, K={K}) ===", flush=True)
    with open(src_dir / "core_index.bin", "rb") as f:
        hdr = struct.unpack("<12Q", f.read(96))
        N_full, K_full, d, Nq_full, Lq = hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]
        f.seek(K_full * d * 4 + d * d * 4, os.SEEK_CUR)
        Q32 = np.frombuffer(f.read(Nq_full * Lq * d * 4), dtype=np.float32).reshape(Nq_full, Lq, d).copy()
        f.seek(Nq_full * Lq * d * 4, os.SEEK_CUR)
        doms = np.frombuffer(f.read(N_full), dtype=np.uint8)[:N_use].copy()

    # Exact same nc5 column as ingest_external_dbs.py (seed=202703)!
    rng_ext = np.random.default_rng(202703)
    nc5_mask = (rng_ext.random(N_use) < 0.05)

    meta_raw = np.fromfile(src_dir / "tier2_meta.bin", dtype=np.dtype([("off", "<u8"), ("ab", "<u4"), ("nt", "<u4")]))[:N_use]
    lens = meta_raw["nt"].astype(np.int64)
    offs = np.zeros(N_use + 1, dtype=np.uint64)
    np.cumsum(lens, out=offs[1:])
    tot_toks = int(offs[-1])

    # Extract exact FP16 vectors for [0, 200000)
    vecs = np.empty((tot_toks, 128), dtype=np.float16)
    with open(src_dir / "tier2_fp16.nvme", "rb") as f:
        for i in range(N_use):
            f.seek(int(meta_raw[i]["off"]))
            nt = int(lens[i])
            vecs[int(offs[i]):int(offs[i + 1])] = np.frombuffer(f.read(nt * 256), dtype=np.float16).reshape(nt, 128)

    # Train spherical K=16,384 codebook on GPU
    rng = np.random.default_rng(202703)
    samp = rng.choice(tot_toks, size=min(800000, tot_toks), replace=False)
    X = torch.from_numpy(vecs[samp].astype(np.float32)).to(device)
    X /= torch.clamp(X.norm(dim=1, keepdim=True), min=1e-12)
    C = X[rng.choice(len(X), size=K, replace=False)].clone()
    for _ in range(12):
        sums = torch.zeros_like(C)
        cnts = torch.zeros(K, device=device, dtype=torch.float32)
        for s in range(0, len(X), 65536):
            xb = X[s:s + 65536]
            a = torch.matmul(xb, C.T).argmax(dim=1)
            sums.index_add_(0, a, xb)
            cnts.index_add_(0, a, torch.ones_like(a, dtype=torch.float32))
        empty = cnts < 0.5
        if empty.any():
            r_idx = torch.randint(0, len(X), (int(empty.sum().item()),), device=device)
            sums[empty] = X[r_idx]
            cnts[empty] = 1.0
        C = sums / torch.clamp(sums.norm(dim=1, keepdim=True), min=1e-12)
    C_f32 = C.cpu().numpy().astype(np.float32)
    P_f32 = make_haar_orthogonal(128, seed=202703)
    Q_rot = np.ascontiguousarray(Q32 @ P_f32, dtype=np.float32)

    # Assign token cids on GPU
    C_g = torch.from_numpy(C_f32).to(device=device, dtype=torch.float16)
    tok_cids = np.empty(tot_toks, dtype=np.uint16)
    for s in range(0, tot_toks, 131072):
        e = min(tot_toks, s + 131072)
        xb = torch.from_numpy(vecs[s:e]).to(device=device, dtype=torch.float16)
        tok_cids[s:e] = torch.matmul(xb, C_g.T).argmax(dim=1).cpu().numpy().astype(np.uint16)

    # Deduplicated doc_cids + CSR inverted lists
    doc_offs = np.zeros(N_use + 1, dtype=np.uint64)
    dc_chunks = []
    inv_counts = np.zeros(K, dtype=np.uint64)
    for d_i in range(N_use):
        uc = np.unique(tok_cids[int(offs[d_i]):int(offs[d_i + 1])])
        dc_chunks.append(uc)
        doc_offs[d_i + 1] = doc_offs[d_i] + len(uc)
        inv_counts[uc] += 1
    doc_cids = np.concatenate(dc_chunks).astype(np.uint16)
    inv_offs = np.zeros(K + 1, dtype=np.uint64)
    np.cumsum(inv_counts, out=inv_offs[1:])
    dc_lens = np.diff(doc_offs).astype(np.int64)
    order = np.argsort(doc_cids, kind="mergesort")
    inv_docs = np.repeat(np.arange(N_use, dtype=np.uint32), dc_lens)[order].astype(np.uint32)

    S_CL, S_REP = 512, 128
    n_cl = (N_use + S_CL - 1) // S_CL
    b3_rep_lens = np.zeros(n_cl, dtype=np.uint16)
    b3_rep_cids = np.zeros((n_cl, S_REP), dtype=np.uint16)
    for cl in range(n_cl):
        ds, de = cl * S_CL, min(N_use, (cl + 1) * S_CL)
        u, cnt = np.unique(doc_cids[doc_offs[ds]:doc_offs[de]], return_counts=True)
        sel = u[np.argsort(-cnt)[:S_REP]]
        b3_rep_lens[cl] = len(sel)
        b3_rep_cids[cl, :len(sel)] = sel

    with open(out_dir / "core_index.bin", "wb") as f:
        f.write(struct.pack("<12Q", 0x504D5633434F5245, N_use, K, 128, Nq_full, 32, n_cl, S_REP, len(doc_cids), len(inv_docs), 1, 0))
        f.write(C_f32.tobytes())
        f.write(P_f32.tobytes())
        f.write(Q32.tobytes())
        f.write(Q_rot.tobytes())
        f.write(doms.tobytes())
        f.write(doc_offs.tobytes())
        f.write(doc_cids.tobytes())
        f.write(np.zeros(n_cl, dtype=np.uint16).tobytes())
        f.write(np.zeros((n_cl, S_REP), dtype=np.uint16).tobytes())
        f.write(np.zeros((n_cl, S_REP), dtype=np.uint64).tobytes())
        f.write(b3_rep_lens.tobytes())
        f.write(b3_rep_cids.tobytes())
        f.write(inv_offs.tobytes())
        f.write(inv_docs.tobytes())

    tmp_v, tmp_o, tmp_c, tmp_C, tmp_P = (out_dir / "_v.bin", out_dir / "_o.bin", out_dir / "_c.bin", out_dir / "_C.bin", out_dir / "_P.bin")
    vecs.tofile(tmp_v); offs.tofile(tmp_o); tok_cids.tofile(tmp_c); C_f32.tofile(tmp_C); P_f32.tofile(tmp_P)
    rab_env = {**os.environ, "OMP_NUM_THREADS": "24"}
    subprocess.run([str(RABITQ_V3), str(tmp_v), str(tmp_o), str(tmp_c), str(tmp_C), str(tmp_P), str(K), str(out_dir)], env=rab_env, check=True)
    for tf in (tmp_v, tmp_o, tmp_c, tmp_C, tmp_P):
        tf.unlink(missing_ok=True)
    subprocess.run([str(B1_V4), str(out_dir), "24"], check=True)

    # Exact IEEE-754 FP32 GPU scores for first 50 stratified queries (exact match with Gate 2B / external DBs!)
    Nq = 50
    Q_eval = Q32[:Nq]
    # Note: LoTTE queries in core_index.bin are strictly interleaved across 5 domains: q % 5!
    # Verify domains present in first 200,000 docs!
    dom_counts = np.bincount(doms, minlength=5).tolist()
    print(f"  lotte_200k dom_counts={dom_counts}", flush=True)
    # Note: In lotte_pooled, docs 0..200000 might only span early domains if sorted by domain!
    # Check actual domain distribution & build predicates that are valid & non-empty on 200K!
    scores = np.empty((Nq, N_use), dtype=np.float32)
    Q_g = torch.from_numpy(Q_eval).to(device=device, dtype=torch.float32)
    Q_valid = (Q_g.norm(dim=2) > 1e-4).to(dtype=torch.float32)
    Q_flat = Q_g.reshape(Nq * 32, 128)
    for ds in range(0, N_use, 32768):
        de = min(N_use, ds + 32768)
        ts, te = int(offs[ds]), int(offs[de])
        vg = torch.from_numpy(vecs[ts:te]).to(device=device, dtype=torch.float16).float()
        tdoc = torch.from_numpy(np.repeat(np.arange(de - ds, dtype=np.int64), lens[ds:de])).to(device)
        for qs in range(0, Nq, 25):
            qe = min(Nq, qs + 25)
            dots = torch.matmul(vg, Q_flat[qs * 32:qe * 32].T)
            dmax = torch.full((de - ds, (qe - qs) * 32), -1e9, device=device, dtype=torch.float32)
            dmax.scatter_reduce_(0, tdoc.unsqueeze(1).expand(-1, (qe - qs) * 32), dots, reduce="amax", include_self=True)
            scores[qs:qe, ds:de] = (dmax.view(de - ds, qe - qs, 32) * Q_valid[qs:qe].unsqueeze(0)).sum(dim=2).T.cpu().numpy()

    p95_len = int(np.percentile(lens, 95))
    p20_len = int(np.percentile(lens, 20))
    rng_w = np.random.default_rng(20270322)
    wl_list = [
        ("NoCorr_5.00pct_ExactNC5", "nocorr", [nc5_mask], [0] * Nq),
        ("NoCorr_100.00pct", "nocorr", [np.ones(N_use, dtype=bool)], [0] * Nq),
        ("Nat_DocLenRange_5pct", "nat_range", [lens >= p95_len], [0] * Nq),
        ("Nat_ShortDoc_20pct", "nat_short", [lens <= p20_len], [0] * Nq),
        ("NoCorr_1.00pct", "nocorr", [rng_w.random(N_use) < 0.01], [0] * Nq),
    ]
    # Add dom=0 predicate if multiple domains exist in 200K
    active_doms = [d_i for d_i, c in enumerate(dom_counts) if c >= 1000]
    if len(active_doms) >= 2:
        wl_list.append(("Nat_Dom0_Filter", "nat_dom", [doms == active_doms[0]], [0] * Nq))
        wl_list.append(("Nat_Dom1_Filter", "nat_dom", [doms == active_doms[1]], [0] * Nq))

    n_words = (N_use + 63) // 64
    gt100_dict = {}
    with open(out_dir / "workloads.bin", "wb") as f, open(out_dir / "workloads_gt100.bin", "wb") as f100:
        f.write(struct.pack("<4Q", 0x574C563342494E30, len(wl_list), N_use, n_words))
        f100.write(struct.pack("<4Q", 0x574C475431303030, len(wl_list), Nq, 100))
        for name, corr, mlist, q2b in wl_list:
            bms = np.stack([pack_bitmap(m) for m in mlist], axis=0)
            gt100 = np.zeros((Nq, 100), dtype=np.uint32)
            tot_v = 0
            for q in range(Nq):
                m = mlist[q2b[q]]
                tot_v += int(m.sum())
                sc = np.where(m, scores[q], -1e9)
                top = np.argsort(-sc, kind="stable")[:100]
                gt100[q] = top.astype(np.uint32)
            sig = tot_v / (Nq * N_use)
            f.write(name.encode("utf-8")[:63].ljust(64, b"\x00"))
            f.write(corr.encode("utf-8")[:31].ljust(32, b"\x00"))
            f.write(struct.pack("<d3Q", sig, Nq, len(mlist), int(tot_v // Nq)))
            f.write(bms.astype(np.uint64).tobytes())
            f.write(np.array(q2b, dtype=np.uint32).tobytes())
            f.write(gt100[:, :10].astype(np.uint32).tobytes())
            f100.write(name.encode("utf-8")[:63].ljust(64, b"\x00"))
            f100.write(gt100.astype(np.uint32).tobytes())
            gt100_dict[name] = gt100[:, :10].tolist()
            print(f"  [lotte_200k] WL {name:<26}: sigma={sig:.4f} (avg_valid={tot_v//Nq:,})", flush=True)

    with open(out_dir / "ext_db_eval_contract.json", "w") as f:
        json.dump({"n_docs": N_use, "nq": Nq, "p95_len": p95_len, "p20_len": p20_len, "dom_counts": dom_counts, "active_doms": active_doms, "nc5_seed": 202703}, f, indent=2)
    print(f"=== [Part A] lotte_200k_matched DONE in {time.time()-t0:.1f}s ===", flush=True)


@torch.no_grad()
def upgrade_msmarco_gt_and_ir(device: str = "cuda:0"):
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    corp_dir = P09 / "stage3" / "corpus_msmarco"
    idx_root = P09 / "stage3" / "indexes" / "msmarco_8m_lsm"
    lens0 = np.load(corp_dir / "msmarco_lens_s0.npy")
    lens1 = np.load(corp_dir / "msmarco_lens_s1.npy")
    lens = np.concatenate([lens0, lens1])
    N_all = len(lens)
    n0 = len(lens0)
    offs0 = np.zeros(n0 + 1, dtype=np.int64); np.cumsum(lens0, out=offs0[1:])
    offs1 = np.zeros(len(lens1) + 1, dtype=np.int64); np.cumsum(lens1, out=offs1[1:])
    mm0 = np.memmap(corp_dir / "msmarco_vecs_s0.f16", dtype=np.float16, mode="r", shape=(int(offs0[-1]), 128))
    mm1 = np.memmap(corp_dir / "msmarco_vecs_s1.f16", dtype=np.float16, mode="r", shape=(int(offs1[-1]), 128))

    def read_range(ds, de):
        sl = lens[ds:de]
        o = np.zeros(len(sl) + 1, dtype=np.int64); np.cumsum(sl, out=o[1:])
        if de <= n0:
            v = np.array(mm0[offs0[ds]:offs0[de]], dtype=np.float16)
        elif ds >= n0:
            v = np.array(mm1[offs1[ds - n0]:offs1[de - n0]], dtype=np.float16)
        else:
            v = np.concatenate([np.array(mm0[offs0[ds]:offs0[n0]], dtype=np.float16),
                                np.array(mm1[0:offs1[de - n0]], dtype=np.float16)], axis=0)
        return v, o

    Q32 = np.load(corp_dir / "msmarco_q32.npy")[:100]
    Nq = 100
    topics = np.load(idx_root / "doc_topics_5.npy")
    TC = np.load(idx_root / "topic_centers_5.npy")
    C_f32 = np.load(idx_root / "shared_C_f32.npy")
    Q_means = Q32.mean(axis=1); Q_means /= np.maximum(np.linalg.norm(Q_means, axis=1, keepdims=True), 1e-12)
    Q_topics = np.argmax(Q_means @ TC.T, axis=1).astype(np.uint8)

    # Build AdvDeadPost_30.00pct across all 3 base segments: exclude every doc whose posting appears in top-8 centroids of ANY eval query's top-256 hot centroids!
    print("=== [Part B] Constructing AdvDeadPost_30.00pct across 8.84M ===", flush=True)
    C_g = torch.from_numpy(C_f32).to(device=device, dtype=torch.float32)
    Q_g = torch.from_numpy(Q32).to(device=device, dtype=torch.float32)
    # Find union of top-8 centroids per query token across first 20 queries (or per-query top-8!) -> construct 5 masks (grouped by q % 5) with 0% hot survival!
    hot_masks = [np.zeros(N_all, dtype=bool) for _ in range(5)]
    per_grp_cids = [set() for _ in range(5)]
    for q in range(Nq):
        dots = torch.matmul(Q_g[q], C_g.T)  # [32, 65536]
        topc = torch.topk(dots, k=8, dim=1).indices.cpu().numpy().ravel()
        per_grp_cids[q % 5].update(int(c) for c in topc)

    # Mark hot docs from each segment's inverted lists
    base = 0
    for sname in ("seg0", "seg1", "seg2", "delta50k"):
        with open(idx_root / sname / "core_index.bin", "rb") as f:
            hdr = struct.unpack("<12Q", f.read(96))
            Ns, Ks, _, Nqs, _, n_cl, S_REP, len_dc, len_inv = hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7], hdr[8], hdr[9]
            skip = Ks * 128 * 4 + 128 * 128 * 4 + 2 * Nqs * 32 * 128 * 4 + Ns + (Ns + 1) * 8 + len_dc * 2 + n_cl * 2 + n_cl * S_REP * 2 + n_cl * S_REP * 8 + n_cl * 2 + n_cl * S_REP * 2
            f.seek(skip, os.SEEK_CUR)
            inv_offs = np.frombuffer(f.read((Ks + 1) * 8), dtype=np.uint64)
            inv_docs = np.frombuffer(f.read(len_inv * 4), dtype=np.uint32)
        for g in range(5):
            for cid in per_grp_cids[g]:
                docs_c = inv_docs[int(inv_offs[cid]):int(inv_offs[cid + 1])]
                if len(docs_c) > 0:
                    hot_masks[g][base + docs_c] = True
        base += Ns

    rng_adv = np.random.default_rng(20270399)
    adv_masks = []
    for g in range(5):
        cold = np.nonzero(~hot_masks[g])[0]
        target = min(len(cold), int(round(0.30 * N_all)))
        m = np.zeros(N_all, dtype=bool)
        if target > 0:
            pick = rng_adv.choice(cold, size=target, replace=False)
            m[pick] = True
        adv_masks.append(m)
    print(f"  AdvDeadPost_30pct cold pool sizes={[int((~h).sum()) for h in hot_masks]}, selected={[int(m.sum()) for m in adv_masks]}", flush=True)

    N_base = 8791823
    rng = np.random.default_rng(20270319)
    p25_len = int(np.percentile(lens, 25))
    p95_len = int(np.percentile(lens, 95))

    m_nc1 = rng.random(N_all) < 0.01
    m_nc5 = rng.random(N_all) < 0.05
    m_nc10 = rng.random(N_all) < 0.10
    m_tomb5 = rng.random(N_base) < 0.05
    m_tomb10 = rng.random(N_base) < 0.10

    m_dyn_base = m_nc5.copy(); m_dyn_base[N_base:] = False
    m_dyn_10k = m_nc5.copy(); m_dyn_10k[:N_base] &= ~m_tomb5; m_dyn_10k[N_base + 10000:] = False
    m_dyn_50k = m_nc5.copy(); m_dyn_50k[:N_base] &= ~m_tomb5
    m_dyn_t10 = m_nc5.copy(); m_dyn_t10[:N_base] &= ~m_tomb10

    rng_ext = np.random.default_rng(20270355)
    m_nc01 = rng_ext.random(N_all) < 0.001
    m_nc30 = rng_ext.random(N_all) < 0.30
    clust_neg5 = []
    for t in range(5):
        rem = np.nonzero(topics != t)[0]
        m = np.zeros(N_all, dtype=bool)
        m[rng_ext.choice(rem, size=int(round(0.05 * N_all)), replace=False)] = True
        clust_neg5.append(m)

    wl_specs = [
        ("NoCorr_0.10pct", "nocorr", [m_nc01], [0] * Nq),
        ("NoCorr_1.00pct", "nocorr", [m_nc1], [0] * Nq),
        ("NoCorr_5.00pct", "nocorr", [m_nc5], [0] * Nq),
        ("NoCorr_10.00pct", "nocorr", [m_nc10], [0] * Nq),
        ("NoCorr_30.00pct", "nocorr", [m_nc30], [0] * Nq),
        ("NoCorr_100.00pct", "nocorr", [np.ones(N_all, dtype=bool)], [0] * Nq),
        ("ClusteredNeg_5.00pct", "clust_neg", clust_neg5, [int(Q_topics[q]) for q in range(Nq)]),
        ("AdvDeadPost_30.00pct", "adv_dead", adv_masks, [q % 5 for q in range(Nq)]),
        ("Nat_DocLenRange_5pct", "nat_range", [lens >= p95_len], [0] * Nq),
        ("Nat_TopicEq_20pct", "nat_pos", [topics == t for t in range(5)], [int(Q_topics[q]) for q in range(Nq)]),
        ("Nat_CrossTopicNeg_20pct", "nat_neg", [topics == ((t + 1) % 5) for t in range(5)], [int(Q_topics[q]) for q in range(Nq)]),
        ("Nat_Compound_NegAND_5pct", "nat_neg_and", [(topics == ((t + 1) % 5)) & (lens <= p25_len) for t in range(5)], [int(Q_topics[q]) for q in range(Nq)]),
        ("LSM_Phase1_Base3Seg", "lsm_base", [m_dyn_base], [0] * Nq),
        ("LSM_Phase3_Tomb5_Delta10K", "lsm_d10k", [m_dyn_10k], [0] * Nq),
        ("LSM_Phase4_Tomb5_Delta50K", "lsm_d50k", [m_dyn_50k], [0] * Nq),
        ("LSM_Phase5_Tomb10_Delta50K", "lsm_t10", [m_dyn_t10], [0] * Nq),
    ]
    n_wl = len(wl_specs)
    K_GT = 100
    top_sc = np.full((n_wl, Nq, K_GT), -1e9, dtype=np.float32)
    top_id = np.zeros((n_wl, Nq, K_GT), dtype=np.uint32)

    Q_valid = (Q_g.norm(dim=2) > 1e-4).to(dtype=torch.float32)
    Q_flat = Q_g.reshape(Nq * 32, 128)
    BLK = 65536
    for ds in range(0, N_all, BLK):
        de = min(N_all, ds + BLK)
        v, o = read_range(ds, de)
        dl = np.diff(o)
        nd = de - ds
        blk_scores = np.empty((Nq, nd), dtype=np.float32)
        vg = torch.from_numpy(v).to(device=device, dtype=torch.float16).float()
        tdoc = torch.from_numpy(np.repeat(np.arange(nd, dtype=np.int64), dl)).to(device)
        for qs in range(0, Nq, 20):
            qe = min(Nq, qs + 20)
            dots = torch.matmul(vg, Q_flat[qs * 32:qe * 32].T)
            dmax = torch.full((nd, (qe - qs) * 32), -1e9, device=device, dtype=torch.float32)
            dmax.scatter_reduce_(0, tdoc.unsqueeze(1).expand(-1, (qe - qs) * 32), dots, reduce="amax", include_self=True)
            blk_scores[qs:qe] = (dmax.view(nd, qe - qs, 32) * Q_valid[qs:qe].unsqueeze(0)).sum(dim=2).T.cpu().numpy()

        for wi, (_, _, mlist, q2b) in enumerate(wl_specs):
            for q in range(Nq):
                m_sub = mlist[q2b[q]][ds:de]
                if not m_sub.any():
                    continue
                cand_sc = np.where(m_sub, blk_scores[q], -1e9)
                k_loc = min(K_GT, nd)
                loc_idx = np.argpartition(-cand_sc, k_loc - 1)[:k_loc]
                comb_sc = np.concatenate([top_sc[wi, q], cand_sc[loc_idx]])
                comb_id = np.concatenate([top_id[wi, q], (ds + loc_idx).astype(np.uint32)])
                ord_k = np.lexsort((comb_id, -comb_sc))[:K_GT]
                top_sc[wi, q] = comb_sc[ord_k]
                top_id[wi, q] = comb_id[ord_k]

        if (ds // BLK) % 25 == 0 or de == N_all:
            print(f"  [MSMARCO 16-WL Top-100 Exact FP32 GT] {de:,}/{N_all:,} ({100*de/N_all:.1f}%) in {time.time()-t0:.1f}s", flush=True)

    n_words = (N_all + 63) // 64
    with open(idx_root / "global_workloads_gt.bin", "wb") as f, open(idx_root / "global_gt100.bin", "wb") as f100:
        f.write(struct.pack("<4Q", 0x574C563342494E30, n_wl, N_all, n_words))
        f100.write(struct.pack("<4Q", 0x574C475431303030, n_wl, Nq, K_GT))
        for wi, (name, corr, mlist, q2b) in enumerate(wl_specs):
            bms = np.stack([pack_bitmap(m) for m in mlist], axis=0)
            avg_v = int(np.mean([int(mlist[q2b[q]].sum()) for q in range(Nq)]))
            sig = avg_v / float(N_all)
            f.write(name.encode("utf-8")[:63].ljust(64, b"\x00"))
            f.write(corr.encode("utf-8")[:31].ljust(32, b"\x00"))
            f.write(struct.pack("<d3Q", sig, Nq, len(mlist), avg_v))
            f.write(bms.astype(np.uint64).tobytes())
            f.write(np.array(q2b, dtype=np.uint32).tobytes())
            f.write(top_id[wi, :, :10].astype(np.uint32).tobytes())
            f100.write(name.encode("utf-8")[:63].ljust(64, b"\x00"))
            f100.write(top_id[wi].astype(np.uint32).tobytes())
            print(f"  Saved 16-WL GT {name:<28}: sigma={sig:.4f} (avg_valid={avg_v:,})", flush=True)

    # Exact official MS MARCO MRR@10 & Recall@100 anchor on Unfiltered 100% (wi=5)!
    qmeta = json.load(open(corp_dir / "msmarco_queries_meta.json"))
    qrels = qmeta["qrels"][:Nq]
    unf_wi = 5
    mrrs = [compute_mrr_at_k(top_id[unf_wi, q], set(qrels[q]), k=10) for q in range(Nq)]
    r100s = [1.0 if any(int(d) in set(qrels[q]) for d in top_id[unf_wi, q, :100]) else 0.0 for q in range(Nq)]
    ms_ir = {"msmarco_8m_exact_fp32_mrr10": round(float(np.mean(mrrs)), 4), "msmarco_8m_exact_fp32_hit100": round(float(np.mean(r100s)), 4), "nq": Nq}
    print(f"=== [MS MARCO 8.84M Official IR Anchor] {json.dumps(ms_ir)} (total {(time.time()-t0)/60:.1f}m) ===", flush=True)
    return ms_ir


@torch.no_grad()
def compute_vidore_v3_ir_anchor(device: str = "cuda:0"):
    torch.backends.cuda.matmul.allow_tf32 = False
    corp = P09 / "stage3" / "corpus_vidore_v3"
    Q32 = np.load(corp / "vidore_v3_q32.npy")
    qrels = json.load(open(corp / "vidore_v3_qrels.json"))
    Nq = len(Q32)
    Q_all = torch.from_numpy(Q32).to(device=device, dtype=torch.float32)
    q_valid = (Q_all.norm(dim=2) > 1e-4)

    out = {}
    for var in ("raw", "adamerge"):
        vecs = np.load(corp / f"vidore_v3_{var}_vecs.npy")
        offs = np.load(corp / f"vidore_v3_{var}_offs.npy")
        lens = np.diff(offs)
        N_docs = len(lens)
        scores = np.empty((Nq, N_docs), dtype=np.float32)
        for ds in range(0, N_docs, 4096):
            de = min(N_docs, ds + 4096)
            vb = torch.from_numpy(vecs[int(offs[ds]):int(offs[de])]).to(device=device, dtype=torch.float16).float()
            tdoc = torch.from_numpy(np.repeat(np.arange(de - ds, dtype=np.int64), lens[ds:de])).to(device)
            for qi in range(Nq):
                qg = Q_all[qi, q_valid[qi]]
                dots = torch.matmul(vb, qg.T)
                dmax = torch.full((de - ds, qg.shape[0]), -1e9, device=device, dtype=torch.float32)
                dmax.scatter_reduce_(0, tdoc.unsqueeze(1).expand(-1, qg.shape[0]), dots, reduce="amax", include_self=True)
                scores[qi, ds:de] = dmax.sum(dim=1).cpu().numpy()
        ndcg5, mrr10 = [], []
        for qi in range(Nq):
            ranked = np.argsort(-scores[qi], kind="stable")[:100]
            ndcg5.append(compute_ndcg_at_k(ranked, qrels[qi], k=5))
            pos = {int(k) for k, v in qrels[qi].items() if float(v) > 0}
            mrr10.append(compute_mrr_at_k(ranked, pos, k=10))
        out[f"vidore_v3_{var}_ndcg5"] = round(float(np.mean(ndcg5)), 4)
        out[f"vidore_v3_{var}_mrr10"] = round(float(np.mean(mrr10)), 4)
    out["vidore_v3_adamerge_ndcg5_delta"] = round(out["vidore_v3_adamerge_ndcg5"] - out["vidore_v3_raw_ndcg5"], 4)
    out["vidore_v3_nq"] = Nq
    print(f"=== [ViDoRe v3 Official IR Anchor] {json.dumps(out)} ===", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotte200k", action="store_true")
    ap.add_argument("--msmarco-gt", action="store_true")
    ap.add_argument("--vidore-ir", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}"
    anchors = {}
    if args.lotte200k:
        build_lotte_200k_matched(dev)
    if args.vidore_ir:
        anchors.update(compute_vidore_v3_ir_anchor(dev))
    if args.msmarco_gt:
        anchors.update(upgrade_msmarco_gt_and_ir(dev))
    if anchors:
        ap_path = P09 / "stage3" / "ir_quality_anchors.json"
        prev = json.load(open(ap_path)) if ap_path.exists() else {}
        prev.update(anchors)
        with open(ap_path, "w") as f:
            json.dump(prev, f, indent=2)
        print(f"Saved {ap_path}: {json.dumps(prev, indent=2)}", flush=True)
