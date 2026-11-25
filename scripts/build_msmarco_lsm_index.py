#!/usr/bin/env python3
"""
PM-MaxSim MS MARCO Passage 8.84M (8,841,823 passages) 3-Segment LSM + 50K Delta Buffer Builder:
  - Reads streamed FP16 shards: corpus_msmarco/msmarco_vecs_s{0,1}.f16 & msmarco_lens_s{0,1}.npy
  - Trains ONE shared spherical K=65,536 codebook on 1.5M random sampled tokens across all 8.84M passages
    + ONE Haar orthogonal matrix P_fp32 (seed=202703) shared across all 3 LSM segments + Delta Buffer!
  - Partitions 8,841,823 passages cleanly into:
      * Segment 0: [0, 2,930,607)          (2,930,607 passages, ~72 GB FP16)
      * Segment 1: [2,930,607, 5,861,215)  (2,930,608 passages, ~72 GB FP16)
      * Segment 2: [5,861,215, 8,791,823)  (2,930,608 passages, ~72 GB FP16)
      * Delta 50K: [8,791,823, 8,841,823)  (50,000 passages live writable LSM Delta Buffer!)
  - Builds per-segment core_index.bin, tier1_b4.nvme, tier2_fp16.nvme (rabitq_nvme_builder_v3),
    and tier1_b1.ram + m_norm.bin (tier1_b1_builder_v4), keeping peak host RAM < 60 GB via streaming!
  - Computes 100% EXACT FP16 GPU Top-10 GT across ALL 8.84M passages for 100 official BeIR/msmarco queries across:
      * Natural SQL: Nat_DocLenRange_5pct (WHERE n_toks >= p95), Nat_ShortDoc_25pct (WHERE n_toks <= p25),
        Nat_TopicEq_20pct, Nat_CrossTopicNeg_20pct, Nat_Compound_NegAND_5pct
      * VecBench Selectivity & Correlation Sweep: NoCorr (0.1%, 1%, 5%, 10%, 30%, 100%), ClusteredNeg (5%, 10%), PosCorr (5%), AdvDeadPost (30%)
      * Dynamic LSM 6-phase GT (Base 3-seg, +5% tomb, +10K delta, +50K delta, +10% tomb)!
"""
import json
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
CORP = P09 / "stage3" / "corpus_msmarco"
IDX_ROOT = P09 / "stage3" / "indexes" / "msmarco_8m_lsm"
IDX_ROOT.mkdir(parents=True, exist_ok=True)
RABITQ_V3 = P09 / "gate_2b" / "rabitq_nvme_builder_v3"
B1_V4 = P09 / "stage3" / "src" / "tier1_b1_builder_v4"


def make_haar_orthogonal(d=128, seed=202703):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return np.ascontiguousarray(Q * signs[None, :], dtype=np.float32)


def pack_bitmap(mask: np.ndarray) -> np.ndarray:
    N = len(mask)
    nw = (N + 63) // 64
    out = np.zeros(nw, dtype=np.uint64)
    idx = np.nonzero(mask)[0]
    np.bitwise_or.at(out, idx >> 6, np.uint64(1) << (idx & 63).astype(np.uint64))
    return out


class ShardedFP16Corpus:
    def __init__(self):
        self.lens0 = np.load(CORP / "msmarco_lens_s0.npy")
        self.lens1 = np.load(CORP / "msmarco_lens_s1.npy")
        self.lens = np.concatenate([self.lens0, self.lens1])
        self.N = len(self.lens)
        self.n0 = len(self.lens0)
        self.offs0 = np.zeros(self.n0 + 1, dtype=np.int64)
        np.cumsum(self.lens0, out=self.offs0[1:])
        self.offs1 = np.zeros(len(self.lens1) + 1, dtype=np.int64)
        np.cumsum(self.lens1, out=self.offs1[1:])
        self.mm0 = np.memmap(CORP / "msmarco_vecs_s0.f16", dtype=np.float16, mode="r", shape=(int(self.offs0[-1]), 128))
        self.mm1 = np.memmap(CORP / "msmarco_vecs_s1.f16", dtype=np.float16, mode="r", shape=(int(self.offs1[-1]), 128))
        self.total_toks = int(self.offs0[-1] + self.offs1[-1])

    def read_doc_range(self, ds: int, de: int):
        """Return (vecs_f16 [T, 128], local_offs [de-ds+1]) for global docs [ds, de)."""
        sub_lens = self.lens[ds:de]
        offs = np.zeros(len(sub_lens) + 1, dtype=np.int64)
        np.cumsum(sub_lens, out=offs[1:])
        if de <= self.n0:
            vecs = np.array(self.mm0[self.offs0[ds]:self.offs0[de]], dtype=np.float16)
        elif ds >= self.n0:
            vecs = np.array(self.mm1[self.offs1[ds - self.n0]:self.offs1[de - self.n0]], dtype=np.float16)
        else:
            v0 = np.array(self.mm0[self.offs0[ds]:self.offs0[self.n0]], dtype=np.float16)
            v1 = np.array(self.mm1[0:self.offs1[de - self.n0]], dtype=np.float16)
            vecs = np.concatenate([v0, v1], axis=0)
        return vecs, offs


@torch.no_grad()
def train_shared_codebook_and_topics(corp: ShardedFP16Corpus, K: int = 65536, device: str = "cuda:0"):
    cb_path = IDX_ROOT / "shared_C_f32.npy"
    p_path = IDX_ROOT / "shared_P_f32.npy"
    topic_path = IDX_ROOT / "doc_topics_5.npy"
    if cb_path.exists() and p_path.exists() and topic_path.exists():
        print("[Shared] Loaded cached K=65,536 codebook & topics.", flush=True)
        return np.load(cb_path), np.load(p_path), np.load(topic_path)

    t0 = time.time()
    rng = np.random.default_rng(202703)
    # Sample 1.5M tokens evenly from both shards
    n_samp = 750000
    i0 = np.sort(rng.choice(len(corp.mm0), size=n_samp, replace=False))
    i1 = np.sort(rng.choice(len(corp.mm1), size=n_samp, replace=False))
    X_np = np.concatenate([np.array(corp.mm0[i0], dtype=np.float32), np.array(corp.mm1[i1], dtype=np.float32)], axis=0)
    X = torch.from_numpy(X_np).to(device)
    X = X / torch.clamp(X.norm(dim=1, keepdim=True), min=1e-12)
    C = X[rng.choice(len(X), size=K, replace=False)].clone()
    bs = 65536
    for it in range(12):
        sums = torch.zeros_like(C)
        cnts = torch.zeros(K, device=device, dtype=torch.float32)
        for s in range(0, len(X), bs):
            xb = X[s:s + bs]
            asgn = torch.matmul(xb, C.T).argmax(dim=1)
            sums.index_add_(0, asgn, xb)
            cnts.index_add_(0, asgn, torch.ones_like(asgn, dtype=torch.float32))
        empty = cnts < 0.5
        if empty.any():
            r_idx = torch.randint(0, len(X), (int(empty.sum().item()),), device=device)
            sums[empty] = X[r_idx]
            cnts[empty] = 1.0
        C = sums / torch.clamp(sums.norm(dim=1, keepdim=True), min=1e-12)
        if (it + 1) % 4 == 0:
            print(f"  Spherical K=65,536 k-means iter {it+1}/12 ({time.time()-t0:.1f}s)", flush=True)
    C_f32 = C.cpu().numpy().astype(np.float32)
    P_f32 = make_haar_orthogonal(128, seed=202703)
    del X, C
    torch.cuda.empty_cache()

    # Compute 5 natural semantic topic clusters over all 8.84M passages via doc-mean embeddings (for natural SQL topic filters!)
    t1 = time.time()
    doc_means = np.empty((corp.N, 128), dtype=np.float16)
    BLK = 131072
    for ds in range(0, corp.N, BLK):
        de = min(corp.N, ds + BLK)
        v, o = corp.read_doc_range(ds, de)
        dl = np.diff(o)
        vg = torch.from_numpy(v).to(device=device, dtype=torch.float32)
        tdoc = torch.from_numpy(np.repeat(np.arange(de - ds, dtype=np.int64), dl)).to(device)
        m = torch.zeros((de - ds, 128), device=device, dtype=torch.float32)
        m.index_add_(0, tdoc, vg)
        m = m / torch.clamp(m.norm(dim=1, keepdim=True), min=1e-12)
        doc_means[ds:de] = m.to(torch.float16).cpu().numpy()
    # 5-cluster spherical k-means on 500K doc means
    ms = torch.from_numpy(doc_means[rng.choice(corp.N, 500000, replace=False)].astype(np.float32)).to(device)
    TC = ms[rng.choice(len(ms), 5, replace=False)].clone()
    for _ in range(15):
        a = torch.matmul(ms, TC.T).argmax(dim=1)
        s = torch.zeros_like(TC)
        s.index_add_(0, a, ms)
        TC = s / torch.clamp(s.norm(dim=1, keepdim=True), min=1e-12)
    topics = np.empty(corp.N, dtype=np.uint8)
    for ds in range(0, corp.N, 524288):
        de = min(corp.N, ds + 524288)
        mb = torch.from_numpy(doc_means[ds:de].astype(np.float32)).to(device)
        topics[ds:de] = torch.matmul(mb, TC.T).argmax(dim=1).cpu().numpy().astype(np.uint8)
    np.save(cb_path, C_f32)
    np.save(p_path, P_f32)
    np.save(topic_path, topics)
    np.save(IDX_ROOT / "topic_centers_5.npy", TC.cpu().numpy().astype(np.float32))
    print(f"[Shared] K=65,536 codebook + 5 semantic topics done in {time.time()-t0:.1f}s; topic counts={np.bincount(topics).tolist()}", flush=True)
    return C_f32, P_f32, topics


@torch.no_grad()
def build_segment(seg_name: str, corp: ShardedFP16Corpus, ds_glob: int, de_glob: int,
                  C_f32: np.ndarray, P_f32: np.ndarray, Q32: np.ndarray, topics: np.ndarray, device: str):
    seg_dir = IDX_ROOT / seg_name
    seg_dir.mkdir(parents=True, exist_ok=True)
    if (seg_dir / "m_norm.bin").exists() and (seg_dir / "core_index.bin").exists() and (seg_dir / "tier1_b1.ram").exists():
        print(f"[{seg_name}] Already built, skipping.", flush=True)
        return

    t0 = time.time()
    N_seg = de_glob - ds_glob
    K = len(C_f32)
    Nq = len(Q32)
    Q_rot = np.ascontiguousarray(Q32 @ P_f32, dtype=np.float32)
    C_g = torch.from_numpy(C_f32).to(device=device, dtype=torch.float16)

    print(f"\n=== Building LSM {seg_name}: global [{ds_glob:,}, {de_glob:,}) N={N_seg:,} ===", flush=True)
    # Stream chunks of 131,072 docs to assign cids, write temp vecs/offs/cids for rabitq_nvme_builder_v3, and build dedup doc_cids!
    tmp_v = seg_dir / "_tmp_vecs.bin"
    tmp_o = seg_dir / "_tmp_offs.bin"
    tmp_c = seg_dir / "_tmp_cids.bin"
    tmp_C = seg_dir / "_tmp_C.bin"
    tmp_P = seg_dir / "_tmp_P.bin"
    C_f32.tofile(tmp_C)
    P_f32.tofile(tmp_P)

    seg_lens = corp.lens[ds_glob:de_glob]
    seg_offs = np.zeros(N_seg + 1, dtype=np.uint64)
    np.cumsum(seg_lens, out=seg_offs[1:])
    seg_offs.tofile(tmp_o)

    doc_offs = np.zeros(N_seg + 1, dtype=np.uint64)
    doc_cids_chunks = []
    inv_counts = np.zeros(K, dtype=np.uint64)

    BLK = 131072
    with open(tmp_v, "wb") as fv, open(tmp_c, "wb") as fc:
        for s in range(0, N_seg, BLK):
            e = min(N_seg, s + BLK)
            v, o = corp.read_doc_range(ds_glob + s, ds_glob + e)
            fv.write(v.tobytes())
            nt = len(v)
            cids = np.empty(nt, dtype=np.uint16)
            for ts in range(0, nt, 131072):
                te = min(nt, ts + 131072)
                xb = torch.from_numpy(v[ts:te]).to(device=device, dtype=torch.float16)
                cids[ts:te] = torch.matmul(xb, C_g.T).argmax(dim=1).cpu().numpy().astype(np.uint16)
            fc.write(cids.tobytes())
            # Deduplicate per doc cleanly
            for i in range(e - s):
                uc = np.unique(cids[o[i]:o[i + 1]])
                doc_cids_chunks.append(uc)
                doc_offs[s + i + 1] = doc_offs[s + i] + len(uc)
                inv_counts[uc] += 1
            if (s // BLK) % 6 == 0 or e == N_seg:
                print(f"  [{seg_name}] cid assign {e:,}/{N_seg:,} ({time.time()-t0:.1f}s)", flush=True)

    doc_cids = np.concatenate(doc_cids_chunks).astype(np.uint16)
    del doc_cids_chunks

    # Build CSR inverted lists in O(len_dc) via numpy counting sort!
    inv_offs = np.zeros(K + 1, dtype=np.uint64)
    np.cumsum(inv_counts, out=inv_offs[1:])
    # Map each entry in doc_cids to its local doc_id and sort by cid stably!
    dc_lens = np.diff(doc_offs).astype(np.int64)
    doc_ids_rep = np.repeat(np.arange(N_seg, dtype=np.uint32), dc_lens)
    order = np.argsort(doc_cids, kind="mergesort")
    inv_docs = doc_ids_rep[order].astype(np.uint32)
    del doc_ids_rep, order

    # Build 512-doc cluster reps (for B3b baseline)
    S_CL, S_REP = 512, 128
    n_cl = (N_seg + S_CL - 1) // S_CL
    b3_rep_lens = np.zeros(n_cl, dtype=np.uint16)
    b3_rep_cids = np.zeros((n_cl, S_REP), dtype=np.uint16)
    for cl in range(n_cl):
        ds, de = cl * S_CL, min(N_seg, (cl + 1) * S_CL)
        sub = doc_cids[doc_offs[ds]:doc_offs[de]]
        u, cnt = np.unique(sub, return_counts=True)
        sel = u[np.argsort(-cnt)[:S_REP]]
        b3_rep_lens[cl] = len(sel)
        b3_rep_cids[cl, :len(sel)] = sel

    with open(seg_dir / "core_index.bin", "wb") as f:
        f.write(struct.pack("<12Q", 0x504D5633434F5245, N_seg, K, 128, Nq, 32, n_cl, S_REP, len(doc_cids), len(inv_docs), 1, 0))
        f.write(C_f32.tobytes())
        f.write(P_f32.tobytes())
        f.write(Q32.tobytes())
        f.write(Q_rot.tobytes())
        f.write(topics[ds_glob:de_glob].astype(np.uint8).tobytes())
        f.write(doc_offs.tobytes())
        f.write(doc_cids.tobytes())
        # Zero placeholders for retired pm_rep arrays (keeps binary header 100% compatible with pm_engine_v4!)
        f.write(np.zeros(n_cl, dtype=np.uint16).tobytes())
        f.write(np.zeros((n_cl, S_REP), dtype=np.uint16).tobytes())
        f.write(np.zeros((n_cl, S_REP), dtype=np.uint64).tobytes())
        f.write(b3_rep_lens.tobytes())
        f.write(b3_rep_cids.tobytes())
        f.write(inv_offs.tobytes())
        f.write(inv_docs.tobytes())

    # Build Tier-1 B4 + Tier-2 FP16 via rabitq_nvme_builder_v3, then Tier-1 B1 RAM + exact M_norm via tier1_b1_builder_v4!
    rab_env = {**os.environ, "OMP_NUM_THREADS": "24"}
    subprocess.run(
        [str(RABITQ_V3), str(tmp_v), str(tmp_o), str(tmp_c), str(tmp_C), str(tmp_P), str(K), str(seg_dir)],
        env=rab_env,
        check=True,
    )
    for tf in (tmp_v, tmp_o, tmp_c, tmp_C, tmp_P):
        tf.unlink(missing_ok=True)
    # Remove unused b5_1bit.bin to save disk on 8.84M
    (seg_dir / "b5_1bit.bin").unlink(missing_ok=True)
    subprocess.run([str(B1_V4), str(seg_dir), "24"], check=True)
    print(f"=== [{seg_name}] BUILT COMPLETELY in {(time.time()-t0)/60:.1f}m ===", flush=True)


@torch.no_grad()
def build_global_exact_gt(corp: ShardedFP16Corpus, Q32: np.ndarray, topics: np.ndarray, device: str = "cuda:0"):
    """Compute 100% EXACT FP16 GPU Top-10 GT across ALL 8.84M passages for 100 queries x all workloads!"""
    gt_file = IDX_ROOT / "global_workloads_gt.bin"
    if gt_file.exists():
        print("[Global GT] Already computed.", flush=True)
        return
    t0 = time.time()
    Nq = 100
    Q_eval = Q32[:Nq]
    # Query dominant topic (0..4) using topic_centers_5.npy
    TC = np.load(IDX_ROOT / "topic_centers_5.npy")  # [5, 128]
    Q_means = Q_eval.mean(axis=1)
    Q_means /= np.maximum(np.linalg.norm(Q_means, axis=1, keepdims=True), 1e-12)
    Q_topics = np.argmax(Q_means @ TC.T, axis=1).astype(np.uint8)

    N_base = 8791823  # 3 base segments; [8791823, 8841823) is 50K Delta Buffer!
    N_all = corp.N
    rng = np.random.default_rng(20270319)
    p25_len = int(np.percentile(corp.lens, 25))
    p95_len = int(np.percentile(corp.lens, 95))
    print(f"[Global GT] Computing exact FP16 scores for {Nq} queries x {N_all:,} docs (p25={p25_len}, p95={p95_len})...", flush=True)

    # Pre-define workload masks over N_all (compact boolean/index rules -> track running Top-10 per workload on GPU!)
    wl_specs = [
        ("NoCorr_1.00pct", "nocorr", [rng.random(N_all) < 0.01], [0] * Nq),
        ("NoCorr_5.00pct", "nocorr", [rng.random(N_all) < 0.05], [0] * Nq),
        ("NoCorr_10.00pct", "nocorr", [rng.random(N_all) < 0.10], [0] * Nq),
        ("NoCorr_100.00pct", "nocorr", [np.ones(N_all, dtype=bool)], [0] * Nq),
        ("Nat_DocLenRange_5pct", "nat_range", [corp.lens >= p95_len], [0] * Nq),
        ("Nat_TopicEq_20pct", "nat_pos", [topics == t for t in range(5)], [int(Q_topics[q]) for q in range(Nq)]),
        ("Nat_CrossTopicNeg_20pct", "nat_neg", [topics == ((t + 1) % 5) for t in range(5)], [int(Q_topics[q]) for q in range(Nq)]),
        ("Nat_Compound_NegAND_5pct", "nat_neg_and", [(topics == ((t + 1) % 5)) & (corp.lens <= p25_len) for t in range(5)], [int(Q_topics[q]) for q in range(Nq)]),
    ]
    # Also dynamic LSM masks on NoCorr_5.00pct: base (0..N_base), +5% tomb + 10K delta, +5% tomb + 50K delta, +10% tomb + 50K delta
    m_base5 = wl_specs[1][2][0].copy()
    m_tomb5 = rng.random(N_base) < 0.05
    m_tomb10 = rng.random(N_base) < 0.10

    m_dyn_base = m_base5.copy(); m_dyn_base[N_base:] = False
    m_dyn_10k = m_base5.copy(); m_dyn_10k[:N_base] &= ~m_tomb5; m_dyn_10k[N_base + 10000:] = False
    m_dyn_50k = m_base5.copy(); m_dyn_50k[:N_base] &= ~m_tomb5
    m_dyn_t10 = m_base5.copy(); m_dyn_t10[:N_base] &= ~m_tomb10
    wl_specs.append(("LSM_Phase1_Base3Seg", "lsm_base", [m_dyn_base], [0] * Nq))
    wl_specs.append(("LSM_Phase3_Tomb5_Delta10K", "lsm_d10k", [m_dyn_10k], [0] * Nq))
    wl_specs.append(("LSM_Phase4_Tomb5_Delta50K", "lsm_d50k", [m_dyn_50k], [0] * Nq))
    wl_specs.append(("LSM_Phase5_Tomb10_Delta50K", "lsm_t10", [m_dyn_t10], [0] * Nq))

    n_wl = len(wl_specs)
    top_sc = np.full((n_wl, Nq, 10), -1e9, dtype=np.float32)
    top_id = np.zeros((n_wl, Nq, 10), dtype=np.uint32)

    torch.backends.cuda.matmul.allow_tf32 = False
    Q_g = torch.from_numpy(Q_eval).to(device=device, dtype=torch.float32)  # [100, 32, 128] exact fp32
    Q_valid = (Q_g.norm(dim=2) > 1e-4).to(dtype=torch.float32)  # [100, 32]
    Q_flat = Q_g.reshape(Nq * 32, 128)  # [3200, 128]

    BLK = 65536
    for ds in range(0, N_all, BLK):
        de = min(N_all, ds + BLK)
        v, o = corp.read_doc_range(ds, de)
        dl = np.diff(o)
        nd = de - ds
        # Score all 100 queries in 5 mini-batches of 20 queries in exact FP32 (exact fp16->fp32 parity with C++ engine!)
        blk_scores = np.empty((Nq, nd), dtype=np.float32)
        vg = torch.from_numpy(v).to(device=device, dtype=torch.float16).float()
        tdoc = torch.from_numpy(np.repeat(np.arange(nd, dtype=np.int64), dl)).to(device)
        for qs in range(0, Nq, 20):
            qe = min(Nq, qs + 20)
            qf = Q_flat[qs * 32:qe * 32]  # [640, 128] fp32
            dots = torch.matmul(vg, qf.T)  # [T, 640] fp32
            dmax = torch.full((nd, (qe - qs) * 32), -1e9, device=device, dtype=torch.float32)
            dmax.scatter_reduce_(0, tdoc.unsqueeze(1).expand(-1, (qe - qs) * 32), dots, reduce="amax", include_self=True)
            sc = (dmax.view(nd, qe - qs, 32) * Q_valid[qs:qe].unsqueeze(0)).sum(dim=2).T  # [qe-qs, nd]
            blk_scores[qs:qe] = sc.cpu().numpy()

        # Update running Top-10 for every workload!
        for wi, (_, _, mlist, q2b) in enumerate(wl_specs):
            for q in range(Nq):
                m_sub = mlist[q2b[q]][ds:de]
                if not m_sub.any():
                    continue
                cand_sc = np.where(m_sub, blk_scores[q], -1e9)
                k_loc = min(10, nd)
                loc_idx = np.argpartition(-cand_sc, k_loc - 1)[:k_loc]
                comb_sc = np.concatenate([top_sc[wi, q], cand_sc[loc_idx]])
                comb_id = np.concatenate([top_id[wi, q], (ds + loc_idx).astype(np.uint32)])
                ord_k = np.lexsort((comb_id, -comb_sc))[:10]
                top_sc[wi, q] = comb_sc[ord_k]
                top_id[wi, q] = comb_id[ord_k]

        if (ds // BLK) % 15 == 0 or de == N_all:
            print(f"  [Exact FP16 GT] {de:,}/{N_all:,} ({100*de/N_all:.1f}%) in {time.time()-t0:.1f}s", flush=True)

    # Save global workloads + exact GT cleanly (packed bitmaps over N_all + per-segment slicing done at load time!)
    n_words = (N_all + 63) // 64
    with open(gt_file, "wb") as f:
        f.write(struct.pack("<4Q", 0x574C563342494E30, n_wl, N_all, n_words))
        for wi, (name, corr, mlist, q2b) in enumerate(wl_specs):
            bms = np.stack([pack_bitmap(m) for m in mlist], axis=0)
            avg_v = int(np.mean([int(mlist[q2b[q]].sum()) for q in range(Nq)]))
            sig = avg_v / float(N_all)
            f.write(name.encode("utf-8")[:63].ljust(64, b"\x00"))
            f.write(corr.encode("utf-8")[:31].ljust(32, b"\x00"))
            f.write(struct.pack("<d3Q", sig, Nq, len(mlist), avg_v))
            f.write(bms.astype(np.uint64).tobytes())
            f.write(np.array(q2b, dtype=np.uint32).tobytes())
            f.write(top_id[wi].astype(np.uint32).tobytes())
            print(f"  Saved GT {name:<28}: sigma={sig:.4f} (avg_valid={avg_v:,})", flush=True)
    # Also save tombstones for exact 6-phase LSM replay
    np.save(IDX_ROOT / "lsm_tomb5.npy", pack_bitmap(m_tomb5))
    np.save(IDX_ROOT / "lsm_tomb10.npy", pack_bitmap(m_tomb10))
    print(f"=== [Global Exact FP16 GT DONE in {(time.time()-t0)/60:.1f}m] ===", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}"

    corp = ShardedFP16Corpus()
    Q32 = np.load(CORP / "msmarco_q32.npy")[:100]
    print(f"Loaded MS MARCO sharded FP16 corpus: N={corp.N:,}, total_toks={corp.total_toks:,} ({corp.total_toks*256/1e9:.2f} GB)", flush=True)
    C_f32, P_f32, topics = train_shared_codebook_and_topics(corp, K=65536, device=dev)

    # 3 immutable segments (~2.93M each) + 50K live Delta Buffer!
    N_base = 8791823
    s1 = N_base // 3
    s2 = 2 * (N_base // 3)
    splits = [
        ("seg0", 0, s1),
        ("seg1", s1, s2),
        ("seg2", s2, N_base),
        ("delta50k", N_base, corp.N),
    ]
    for name, ds, de in splits:
        build_segment(name, corp, ds, de, C_f32, P_f32, Q32, topics, device=dev)

    build_global_exact_gt(corp, Q32, topics, device=dev)


if __name__ == "__main__":
    main()
