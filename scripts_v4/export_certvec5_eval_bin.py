#!/usr/bin/env python3
"""Export flat C++-ready binary eval bundles for `certvec_bench_certvec5` and `certvec_bench_v4 --bench-zipf-cache`."""
import argparse
import pathlib
import struct
import numpy as np


def export_bgem3(idx_dir: pathlib.Path):
    q_colb = np.load(idx_dir / "q_colb_f32.npy")[:100].astype(np.float32)  # [100, 32, 1024]
    q_dense = np.load(idx_dir / "q_dense_f32.npy")[:100].astype(np.float32)  # [100, 1024]
    q_lens = np.load(idx_dir / "q_lens.npy")[:100].astype(np.uint32)
    N_docs = len(np.load(idx_dir / "doc_topics.npy"))
    sp = np.load(idx_dir / "sparse_lexical_1000q.npz")
    sp_offs = sp["offs"][:101].astype(np.uint64)
    n_sp = int(sp_offs[-1])
    sp_rids = sp["rids"][:n_sp].astype(np.uint32)
    sp_scores = sp["scores"][:n_sp].astype(np.float32)

    wl_names = ["NoCorr_100pct", "NoCorr_5pct", "Nat_Topic0_Eq", "Nat_DocLen_Top10pct"]
    out_p = idx_dir / "eval_bundle.bin"
    with open(out_p, "wb") as f:
        f.write(struct.pack("<8Q", 0x4556414C30303035, N_docs, 100, 1024, 1024, 32, len(wl_names), n_sp))
        f.write(q_colb.tobytes())
        f.write(q_dense.tobytes())
        f.write(q_lens.tobytes())
        f.write(sp_offs.tobytes())
        if n_sp > 0:
            f.write(sp_rids.tobytes())
            f.write(sp_scores.tobytes())
        for wn in wl_names:
            d = np.load(idx_dir / f"gt_{wn}.npz")
            mask = d["mask"].astype(bool)
            vidx = np.where(mask)[0].astype(np.uint32)
            sig = float(len(vidx)) / float(N_docs)
            nb = wn.encode("utf-8")[:63].ljust(64, b"\x00")
            f.write(nb)
            f.write(struct.pack("<dQ", sig, len(vidx)))
            f.write(vidx.tobytes())
            f.write(d["gt_colb"].astype(np.uint32).tobytes())
            f.write(d["gt_hyb1"].astype(np.uint32).tobytes())
            f.write(d["gt_hyb2"].astype(np.uint32).tobytes())
            f.write(d["w4_taus"].astype(np.float64).tobytes())
    print(f"[Exported] {out_p} ({out_p.stat().st_size / 1e6:.2f} MB)")


def export_docs768(idx_dir: pathlib.Path):
    q_vecs = np.load(idx_dir / "queries_f32.npy")[:200].astype(np.float32)  # [200, 768]
    q_lens = np.ones(200, dtype=np.uint32)
    N_docs = len(np.load(idx_dir / "dom_ids.npy"))
    sp = np.load(idx_dir / "bm25_sparse_1000q.npz")
    sp_offs = sp["offs"][:201].astype(np.uint64)
    n_sp = int(sp_offs[-1])
    sp_rids = sp["rids"][:n_sp].astype(np.uint32)
    sp_scores = sp["scores"][:n_sp].astype(np.float32)

    wl_names = ["NoCorr_100pct", "NoCorr_5pct", "Nat_Dom0_Eq", "Nat_DocLen_Top10pct", "Nat_Compound_DomOther_Long"]
    out_p = idx_dir / "eval_bundle.bin"
    with open(out_p, "wb") as f:
        f.write(struct.pack("<8Q", 0x4556414C30303035, N_docs, 200, 768, 768, 1, len(wl_names), n_sp))
        f.write(q_vecs.tobytes())
        f.write(q_vecs.tobytes())  # Same 768D query targets field 1 (title) too
        f.write(q_lens.tobytes())
        f.write(sp_offs.tobytes())
        if n_sp > 0:
            f.write(sp_rids.tobytes())
            f.write(sp_scores.tobytes())
        for wn in wl_names:
            d = np.load(idx_dir / f"gt_{wn}.npz")
            mask = d["mask"].astype(bool)
            vidx = np.where(mask)[0].astype(np.uint32)
            sig = float(len(vidx)) / float(N_docs)
            nb = wn.encode("utf-8")[:63].ljust(64, b"\x00")
            f.write(nb)
            f.write(struct.pack("<dQ", sig, len(vidx)))
            f.write(vidx.tobytes())
            f.write(d["w1_max_gt"].astype(np.uint32).tobytes())
            f.write(d["w1_top3_gt"].astype(np.uint32).tobytes())
            f.write(d["w3_hyb_gt"].astype(np.uint32).tobytes())
            f.write(d["w4_taus"].astype(np.float64).tobytes())
    print(f"[Exported] {out_p} ({out_p.stat().st_size / 1e6:.2f} MB)")


def export_zipf_streams(q_dir: pathlib.Path):
    q = np.load(q_dir / "msmarco_dev_6980_q32.npy").astype(np.float32)
    q.tofile(q_dir / "msmarco_dev_6980_q32.f32")
    s1 = np.load(q_dir / "zipf1_100k_stream_msmarco.npz")
    s1["q_ids"].astype(np.uint32).tofile(q_dir / "zipf1_100k_qids.u32")
    s1["tenant_ids"].astype(np.uint16).tofile(q_dir / "zipf1_100k_tids.u16")
    s2 = np.load(q_dir / "zipf1_50k_stream_1000q.npz")
    s2["q_ids"].astype(np.uint32).tofile(q_dir / "zipf1_50k_qids.u32")
    s2["tenant_ids"].astype(np.uint16).tofile(q_dir / "zipf1_50k_tids.u16")
    print(f"[Exported] Zipf streams & 6980 queries raw binaries in {q_dir} (shape={q.shape})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bgem3", "docs768", "zipf"], required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    p = pathlib.Path(args.dir)
    if args.mode == "bgem3": export_bgem3(p)
    elif args.mode == "docs768": export_docs768(p)
    elif args.mode == "zipf": export_zipf_streams(p)
