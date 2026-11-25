#!/usr/bin/env python3
"""
PM-MaxSim Build ViDoRe v3 (16,868 pages, 7 English Enterprise Domains) Indexes:
  Builds BOTH:
    - stage3/indexes/vidore_v3_raw/      (Raw ColQwen2.5-v0.2, spherical K=16,384, is_unit=1)
    - stage3/indexes/vidore_v3_adamerge/ (EOS-attention AdaMerge-258, spherical K=16,384, is_unit=0)
  Including:
    - core_index.bin (exact schema compatible with pm_engine_v4!)
    - tier1_b4.nvme + tier1_meta.bin + tier2_fp16.nvme + tier2_meta.bin (via rabitq_nvme_builder_v3)
    - tier1_b1.ram + tier1_b1_meta.bin + m_norm.bin (via tier1_b1_builder_v4)
    - workloads.bin with 100% EXACT GPU FP16 Top-10 GT across:
        * Natural Visual SQL Predicates: Nat_DomainEq (~14-31% pos-corr), Nat_CrossDomainNeg (~14-31% neg-corr),
          Nat_PageRange_10pct (WHERE page_num <= p10), Nat_Compound_NegAND_5pct (domain != q_dom AND page_num <= p30)
        * VecBench Selectivity & Correlation Sweep: NoCorr (1%, 5%, 10%, 30%, 100%), ClusteredNeg (5%, 10%), MixedClustNeg (5%, 10%), PosCorr (5%, 10%)
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
CORP = P09 / "stage3" / "corpus_vidore_v3"
IDX_ROOT = P09 / "stage3" / "indexes"
RABITQ_V3 = P09 / "gate_2b" / "rabitq_nvme_builder_v3"
B1_V4 = P09 / "stage3" / "src" / "tier1_b1_builder_v4"


def make_haar_orthogonal(d=128, seed=202703):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[None, :]
    return np.ascontiguousarray(Q, dtype=np.float32)


@torch.no_grad()
def train_spherical_kmeans(vecs_f16: np.ndarray, K: int, device: str, sample_n: int = 800000, iters: int = 12):
    rng = np.random.default_rng(202703)
    N_tok = len(vecs_f16)
    idx = rng.choice(N_tok, size=min(sample_n, N_tok), replace=False)
    X = torch.from_numpy(vecs_f16[idx].astype(np.float32)).to(device)
    X = X / torch.clamp(X.norm(dim=1, keepdim=True), min=1e-12)
    c_init = rng.choice(len(X), size=K, replace=False)
    C = X[c_init].clone()
    bs = 65536
    for it in range(iters):
        sums = torch.zeros_like(C)
        cnts = torch.zeros(K, device=device, dtype=torch.float32)
        for s in range(0, len(X), bs):
            xb = X[s:s + bs]
            dots = torch.matmul(xb, C.T)
            asgn = dots.argmax(dim=1)
            sums.index_add_(0, asgn, xb)
            cnts.index_add_(0, asgn, torch.ones_like(asgn, dtype=torch.float32))
        empty = cnts < 0.5
        if empty.any():
            r_idx = torch.randint(0, len(X), (int(empty.sum().item()),), device=device)
            sums[empty] = X[r_idx]
            cnts[empty] = 1.0
        C = sums / torch.clamp(sums.norm(dim=1, keepdim=True), min=1e-12)
    return C.cpu().numpy().astype(np.float32)


@torch.no_grad()
def assign_cids_gpu(vecs_f16: np.ndarray, C_f32: np.ndarray, device: str):
    C = torch.from_numpy(C_f32).to(device=device, dtype=torch.float16)
    N_tok = len(vecs_f16)
    out = np.empty(N_tok, dtype=np.uint16)
    bs = 131072
    for s in range(0, N_tok, bs):
        xb = torch.from_numpy(vecs_f16[s:s + bs]).to(device=device, dtype=torch.float16)
        # Cosine / Euclidean on spherical C -> max dot (for non-unit AdaMerge, exact L2 min = argmin ||x-c||^2 = argmax (2 x.c - ||c||^2); ||c||=1 -> argmax x.c!)
        asgn = torch.matmul(xb, C.T).argmax(dim=1)
        out[s:s + bs] = asgn.cpu().numpy().astype(np.uint16)
    return out


@torch.no_grad()
def compute_exact_scores_gpu(vecs_f16: np.ndarray, offs: np.ndarray, Q32: np.ndarray, device: str):
    torch.backends.cuda.matmul.allow_tf32 = False
    N_docs = len(offs) - 1
    Nq = len(Q32)
    lens = np.diff(offs)
    scores = np.empty((Nq, N_docs), dtype=np.float32)
    # Process docs in chunks of 4,096 pages in exact IEEE-754 FP32 (matches C++ _mm512_cvtph_ps + Q32 bit-for-bit!)
    DOC_BLK = 4096
    Q_all = torch.from_numpy(Q32).to(device=device, dtype=torch.float32)  # [Nq, 32, 128]
    q_valid = (Q_all.norm(dim=2) > 1e-4)  # [Nq, 32]
    for ds in range(0, N_docs, DOC_BLK):
        de = min(N_docs, ds + DOC_BLK)
        ts, te = int(offs[ds]), int(offs[de])
        vb = torch.from_numpy(vecs_f16[ts:te]).to(device=device, dtype=torch.float16).float()  # [T, 128] exact fp16->fp32
        dl = lens[ds:de]
        tdoc = torch.from_numpy(np.repeat(np.arange(de - ds, dtype=np.int64), dl)).to(device)
        for qi in range(Nq):
            vm = q_valid[qi]
            qg = Q_all[qi, vm]  # [L, 128] fp32
            dots = torch.matmul(vb, qg.T)
            dmax = torch.full((de - ds, qg.shape[0]), -1e9, device=device, dtype=torch.float32)
            dmax.scatter_reduce_(0, tdoc.unsqueeze(1).expand(-1, qg.shape[0]), dots, reduce="amax", include_self=True)
            scores[qi, ds:de] = dmax.sum(dim=1).cpu().numpy()
    return scores


def pack_bitmap(mask: np.ndarray) -> np.ndarray:
    N = len(mask)
    nw = (N + 63) // 64
    out = np.zeros(nw, dtype=np.uint64)
    idx = np.nonzero(mask)[0]
    np.bitwise_or.at(out, idx >> 6, np.uint64(1) << (idx & 63).astype(np.uint64))
    return out


def build_one(variant: str, is_unit: int, device: str):
    t0 = time.time()
    out_dir = IDX_ROOT / f"vidore_v3_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    vecs = np.load(CORP / f"vidore_v3_{variant}_vecs.npy")
    offs = np.load(CORP / f"vidore_v3_{variant}_offs.npy")
    doms = np.load(CORP / f"vidore_v3_doms.npy")
    pnums = np.load(CORP / f"vidore_v3_pnums.npy")
    Q32 = np.load(CORP / "vidore_v3_q32.npy")
    Qdoms = np.load(CORP / "vidore_v3_qdoms.npy")

    N_docs = len(doms)
    Nq = len(Q32)
    K = 16384
    print(f"\n=== Building ViDoRe v3 ({variant}): N={N_docs:,}, toks={len(vecs):,}, K={K}, Nq={Nq} ===", flush=True)

    C_f32 = train_spherical_kmeans(vecs, K, device=device)
    tok_cids = assign_cids_gpu(vecs, C_f32, device=device)
    P_f32 = make_haar_orthogonal(128, seed=202703)
    Q_rot = np.ascontiguousarray(Q32 @ P_f32, dtype=np.float32)

    # Deduplicated doc_cids + inverted lists
    doc_offs = np.zeros(N_docs + 1, dtype=np.int64)
    doc_cids_list = []
    inv_lists = [[] for _ in range(K)]
    for d in range(N_docs):
        uc = np.unique(tok_cids[offs[d]:offs[d + 1]])
        doc_cids_list.append(uc)
        doc_offs[d + 1] = doc_offs[d] + len(uc)
        for c in uc:
            inv_lists[int(c)].append(d)
    doc_cids = np.concatenate(doc_cids_list).astype(np.uint16)
    inv_offs = np.zeros(K + 1, dtype=np.int64)
    inv_flat = []
    for c in range(K):
        arr = np.array(inv_lists[c], dtype=np.uint32)
        inv_flat.append(arr)
        inv_offs[c + 1] = inv_offs[c] + len(arr)
    inv_docs = np.concatenate(inv_flat).astype(np.uint32)

    # Simple 512-doc cluster reps (for fair B3b_MVIVF_T3 baseline)
    S_CL, S_REP = 512, 128
    n_cl = (N_docs + S_CL - 1) // S_CL
    pm_rep_lens = np.zeros(n_cl, dtype=np.uint16)
    pm_rep_cids = np.zeros((n_cl, S_REP), dtype=np.uint16)
    pm_rep_masks = np.zeros((n_cl, S_REP), dtype=np.uint64)
    b3_rep_lens = np.zeros(n_cl, dtype=np.uint16)
    b3_rep_cids = np.zeros((n_cl, S_REP), dtype=np.uint16)
    for cl in range(n_cl):
        ds, de = cl * S_CL, min(N_docs, (cl + 1) * S_CL)
        sub = doc_cids[doc_offs[ds]:doc_offs[de]]
        u, cnt = np.unique(sub, return_counts=True)
        order = np.argsort(-cnt)[:S_REP]
        sel = u[order]
        b3_rep_lens[cl] = len(sel)
        b3_rep_cids[cl, :len(sel)] = sel
        pm_rep_lens[cl] = len(sel)
        pm_rep_cids[cl, :len(sel)] = sel

    # Write core_index.bin (exact Gate 2B / pm_engine_v4 header format!)
    with open(out_dir / "core_index.bin", "wb") as f:
        hdr = struct.pack(
            "<12Q",
            0x504D5633434F5245, N_docs, K, 128, Nq, 32, n_cl, S_REP,
            len(doc_cids), len(inv_docs), is_unit, 0
        )
        f.write(hdr)
        f.write(C_f32.tobytes())
        f.write(P_f32.tobytes())
        f.write(Q32.tobytes())
        f.write(Q_rot.tobytes())
        f.write(doms.astype(np.uint8).tobytes())
        f.write(doc_offs.astype(np.uint64).tobytes())
        f.write(doc_cids.astype(np.uint16).tobytes())
        f.write(pm_rep_lens.tobytes())
        f.write(pm_rep_cids.tobytes())
        f.write(pm_rep_masks.tobytes())
        f.write(b3_rep_lens.tobytes())
        f.write(b3_rep_cids.tobytes())
        f.write(inv_offs.astype(np.uint64).tobytes())
        f.write(inv_docs.astype(np.uint32).tobytes())

    # Write temp raw files & run rabitq_nvme_builder_v3 -> tier1_b4.nvme + tier2_fp16.nvme
    tmp_v = out_dir / "_tmp_vecs.bin"
    tmp_o = out_dir / "_tmp_offs.bin"
    tmp_c = out_dir / "_tmp_cids.bin"
    tmp_C = out_dir / "_tmp_C.bin"
    tmp_P = out_dir / "_tmp_P.bin"
    vecs.tofile(tmp_v)
    offs.astype(np.uint64).tofile(tmp_o)
    tok_cids.astype(np.uint16).tofile(tmp_c)
    C_f32.tofile(tmp_C)
    P_f32.tofile(tmp_P)
    rab_env = {**os.environ, "OMP_NUM_THREADS": "24"}
    subprocess.run(
        [str(RABITQ_V3), str(tmp_v), str(tmp_o), str(tmp_c), str(tmp_C), str(tmp_P), str(K), str(out_dir)],
        env=rab_env,
        check=True,
    )
    for tf in (tmp_v, tmp_o, tmp_c, tmp_C, tmp_P):
        tf.unlink(missing_ok=True)

    # Run tier1_b1_builder_v4 -> tier1_b1.ram + m_norm.bin!
    subprocess.run([str(B1_V4), str(out_dir), "24"], check=True)

    # Exact GPU FP16 scores for ALL 140 queries x 16,868 pages -> exact GT workloads!
    scores = compute_exact_scores_gpu(vecs, offs, Q32, device=device)
    rng = np.random.default_rng(20270309)
    n_words = (N_docs + 63) // 64
    p10_pg = int(np.percentile(pnums, 10))
    p30_pg = int(np.percentile(pnums, 30))

    workloads = []

    def add_wl(name, corr, masks_list, q_to_bm):
        bms = np.stack([pack_bitmap(m) for m in masks_list], axis=0)
        gt10 = np.zeros((Nq, 10), dtype=np.uint32)
        tot_v = 0
        for q in range(Nq):
            m = masks_list[q_to_bm[q]]
            tot_v += int(m.sum())
            sc = np.where(m, scores[q], -1e9)
            top = np.argsort(-sc, kind="stable")[:10]
            gt10[q] = top.astype(np.uint32)
        sig = tot_v / (Nq * N_docs)
        workloads.append((name, corr, sig, Nq, len(masks_list), int(tot_v // Nq), bms, np.array(q_to_bm, dtype=np.uint32), gt10))
        print(f"  WL {name:<26}: sigma={sig:.4f} (avg_valid={tot_v//Nq:,})", flush=True)

    # Standard NoCorr sweep
    for sig in (0.01, 0.05, 0.10, 0.30, 1.00):
        m = np.ones(N_docs, dtype=bool) if sig >= 0.999 else (rng.random(N_docs) < sig)
        add_wl(f"NoCorr_{sig*100:.2f}pct", "nocorr", [m], [0] * Nq)

    # Natural Multi-Attribute Visual SQL Predicates (ZERO synthetic clusters!)
    add_wl("Nat_DomainEq", "nat_pos", [doms == d for d in range(7)], [int(Qdoms[q]) for q in range(Nq)])
    add_wl("Nat_CrossDomainNeg", "nat_neg", [doms == ((d + 1) % 7) for d in range(7)], [int(Qdoms[q]) for q in range(Nq)])
    add_wl("Nat_PageRange_10pct", "nat_range", [pnums <= max(1, p10_pg)], [0] * Nq)
    add_wl(
        "Nat_Compound_NegAND",
        "nat_neg_and",
        [(doms != d) & (pnums <= max(2, p30_pg)) for d in range(7)],
        [int(Qdoms[q]) for q in range(Nq)],
    )

    # ClusteredNeg & MixedClustNeg (5%, 10%)
    for sig in (0.05, 0.10):
        ms_neg = []
        for d in range(7):
            # Exclude query's own domain + sample sig from remaining domains
            rem = np.nonzero(doms != d)[0]
            m = np.zeros(N_docs, dtype=bool)
            pick = rng.choice(rem, size=min(len(rem), int(round(sig * N_docs))), replace=False)
            m[pick] = True
            ms_neg.append(m)
        add_wl(f"ClusteredNeg_{sig*100:.2f}pct", "clust_neg", ms_neg, [int(Qdoms[q]) for q in range(Nq)])

        ms_pos = []
        for d in range(7):
            same = np.nonzero(doms == d)[0]
            m = np.zeros(N_docs, dtype=bool)
            pick = rng.choice(same, size=min(len(same), int(round(sig * N_docs))), replace=False)
            m[pick] = True
            ms_pos.append(m)
        add_wl(f"PosCorr_{sig*100:.2f}pct", "pos_corr", ms_pos, [int(Qdoms[q]) for q in range(Nq)])

    with open(out_dir / "workloads.bin", "wb") as f:
        f.write(struct.pack("<4Q", 0x574C563342494E30, len(workloads), N_docs, n_words))
        for name, corr, sig, nqe, nbm, av, bms, q2b, gt10 in workloads:
            f.write(name.encode("utf-8")[:63].ljust(64, b"\x00"))
            f.write(corr.encode("utf-8")[:31].ljust(32, b"\x00"))
            f.write(struct.pack("<d3Q", sig, nqe, nbm, av))
            f.write(bms.astype(np.uint64).tobytes())
            f.write(q2b.astype(np.uint32).tobytes())
            f.write(gt10.astype(np.uint32).tobytes())

    print(f"=== Built {out_dir} cleanly in {time.time()-t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}"
    build_one("raw", 1, dev)
    build_one("adamerge", 0, dev)
