#!/usr/bin/env python3
"""
Measure exact 48-core hard-exclusive:
  1. Isolated OS Linux PSS / RSS / VmHWM for ALL 3 tiers of PM-MaxSim (PM_NVME_T1, Split_B1_B4_T3, PM_DRAM_T1)
     across all 5 corpora (lotte_200k_matched, vidore_v3_adamerge, vidore_v3_raw, lotte_pooled 2.82M, msmarco_8m_lsm).
  2. Exact 1:1 PM-MaxSim Index Build Time breakdown on lotte_200k_matched & vidore_v3_adamerge
     (K-means + assignment + CRoaring/core_index + Tier1-B4/Tier2-FP16 + Tier1-B1/M_norm) into a temporary directory
     cleaned up immediately afterward.
"""
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path
import numpy as np
import torch

P09 = Path(os.environ.get("P09_ROOT", "./workspace"))
SRC = P09 / "stage3" / "src"
OUT_DIR = P09 / "stage3" / "results_s33"
RABITQ_V3 = P09 / "gate_2b" / "rabitq_nvme_builder_v3"
B1_V4 = SRC / "tier1_b1_builder_v4"


def measure_pm_process_memory():
    rows = []
    single_targets = [
        ("lotte_200k_matched", P09 / "stage3" / "indexes" / "lotte_200k_matched", "NoCorr_5.00pct_ExactNC5"),
        ("vidore_v3_adamerge", P09 / "stage3" / "indexes" / "vidore_v3_adamerge", "Nat_DomainEq"),
        ("vidore_v3_raw", P09 / "stage3" / "indexes" / "vidore_v3_raw", "Nat_DomainEq"),
        ("lotte_2p8m", P09 / "gate_2b" / "indexes_v3" / "lotte_pooled", "NoCorr_5.00pct"),
    ]
    tmp_json = Path("/tmp/pm_mem_probe.json")
    for corp, idx_dir, wl in single_targets:
        for mode, m_id, m_lbl in [("nvme", "2", "PM_NVME_T1"), ("split", "0", "Split_B1_B4_T3"), ("dram", "1", "PM_DRAM_T1")]:
            cmd = [
                "taskset", "-c", "0-47", str(SRC / "pm_engine_v4"),
                "--bench-single", str(idx_dir), str(tmp_json),
                "--skip-unit-test", "--tier-mem-mode", mode,
                "--threads", "48", "--workloads", wl, "--methods", m_id,
            ]
            subprocess.run(cmd, check=True)
            d = json.load(open(tmp_json))
            pm = d["proc_mem_gb"]
            row = {"corpus": corp, "method": m_lbl, "tier_mem_mode": mode, "pss_gb": round(pm["pss_gb"], 4), "rss_gb": round(pm["rss_gb"], 4), "vm_hwm_gb": round(pm["vm_hwm_gb"], 4)}
            rows.append(row)
            print("  [PM True Mem]", json.dumps(row), flush=True)

    # MS MARCO 8.84M 3-seg LSM + 50K Delta
    ms_dir = P09 / "stage3" / "indexes" / "msmarco_8m_lsm"
    for mode, m_id, m_lbl in [("nvme", "2", "PM_NVME_T1"), ("split", "0", "Split_B1_B4_T3"), ("dram", "1", "PM_DRAM_T1")]:
        cmd = [
            "taskset", "-c", "0-47", str(SRC / "pm_multilsm_v4"),
            "--bench-lsm", str(ms_dir), str(tmp_json),
            "--skip-compact", "--tier-mem-mode", mode,
            "--threads", "48", "--workloads", "NoCorr_5.00pct", "--methods", m_id,
        ]
        subprocess.run(cmd, check=True)
        d = json.load(open(tmp_json))
        pm = d["proc_mem_gb"]
        row = {"corpus": "msmarco_8m_lsm", "method": m_lbl, "tier_mem_mode": mode, "pss_gb": round(pm["pss_gb"], 4), "rss_gb": round(pm["rss_gb"], 4), "vm_hwm_gb": round(pm["vm_hwm_gb"], 4)}
        rows.append(row)
        print("  [PM True Mem]", json.dumps(row), flush=True)

    if tmp_json.exists():
        tmp_json.unlink()
    out = OUT_DIR / "pm_true_memory.json"
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"Saved {out}", flush=True)


@torch.no_grad()
def time_pm_build_one(tag: str, src_dir: Path, K: int):
    torch.backends.cuda.matmul.allow_tf32 = False
    tmp_dir = P09 / "stage3" / f"tmp_build_timing_{tag}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        # Load raw FP16 embeddings in RAM first (excluded from index build time, exact match to external DB ingest input!)
        meta_raw = np.fromfile(src_dir / "tier2_meta.bin", dtype=np.dtype([("off", "<u8"), ("ab", "<u4"), ("nt", "<u4")]))
        N_docs = len(meta_raw)
        lens = meta_raw["nt"].astype(np.int64)
        offs = np.zeros(N_docs + 1, dtype=np.uint64)
        np.cumsum(lens, out=offs[1:])
        tot_toks = int(offs[-1])
        vecs = np.empty((tot_toks, 128), dtype=np.float16)
        with open(src_dir / "tier2_fp16.nvme", "rb") as f:
            for i in range(N_docs):
                f.seek(int(meta_raw[i]["off"]))
                nt = int(lens[i])
                vecs[int(offs[i]):int(offs[i + 1])] = np.frombuffer(f.read(nt * 256), dtype=np.float16).reshape(nt, 128)

        # Start timed PM-MaxSim full index construction!
        t_start = time.perf_counter()

        # Phase 1: Spherical K-means (GPU 0) + Haar rotation P
        rng = np.random.default_rng(202703)
        samp = rng.choice(tot_toks, size=min(800000, tot_toks), replace=False)
        X = torch.from_numpy(vecs[samp].astype(np.float32)).to("cuda:0")
        X /= torch.clamp(X.norm(dim=1, keepdim=True), min=1e-12)
        C = X[rng.choice(len(X), size=K, replace=False)].clone()
        for _ in range(12):
            sums = torch.zeros_like(C)
            cnts = torch.zeros(K, device="cuda:0", dtype=torch.float32)
            for s in range(0, len(X), 65536):
                xb = X[s:s + 65536]
                a = torch.matmul(xb, C.T).argmax(dim=1)
                sums.index_add_(0, a, xb)
                cnts.index_add_(0, a, torch.ones_like(a, dtype=torch.float32))
            empty = cnts < 0.5
            if empty.any():
                sums[empty] = X[rng.choice(len(X), size=int(empty.sum().item()), replace=False)]
                cnts[empty] = 1.0
            C = sums / cnts.unsqueeze(1)
            C /= torch.clamp(C.norm(dim=1, keepdim=True), min=1e-12)
        torch.cuda.synchronize()
        t_km = time.perf_counter()

        # Phase 2: Assign all tokens to centroids + build dedup doc_cids + inverted lists
        cids_all = np.empty(tot_toks, dtype=np.uint16)
        CT = C.T.contiguous()
        for s in range(0, tot_toks, 131072):
            e = min(tot_toks, s + 131072)
            xb = torch.from_numpy(vecs[s:e].astype(np.float32)).to("cuda:0")
            cids_all[s:e] = torch.matmul(xb, CT).argmax(dim=1).cpu().numpy().astype(np.uint16)
        torch.cuda.synchronize()

        C_np = C.cpu().numpy().astype(np.float32)
        A = rng.standard_normal((128, 128))
        Q_h, R_h = np.linalg.qr(A)
        sgn = np.sign(np.diag(R_h)); sgn[sgn == 0] = 1.0
        P = np.ascontiguousarray(Q_h * sgn[None, :], dtype=np.float32)

        # Write raw inputs for rabitq_nvme_builder_v3
        C_np.tofile(tmp_dir / "C_fp32.bin")
        P.tofile(tmp_dir / "P_fp32.bin")
        vecs.tofile(tmp_dir / "vecs_fp16.bin")
        offs.tofile(tmp_dir / "offs_u64.bin")
        cids_all.tofile(tmp_dir / "cids_u16.bin")
        t_assign = time.perf_counter()

        # Phase 3: 48-core C++20 Tier-1 B=4 + Tier-2 FP16 4KB-aligned NVMe builder
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "48"
        subprocess.run([
            "taskset", "-c", "0-47", str(RABITQ_V3),
            str(tmp_dir / "vecs_fp16.bin"), str(tmp_dir / "offs_u64.bin"), str(tmp_dir / "cids_u16.bin"),
            str(tmp_dir / "C_fp32.bin"), str(tmp_dir / "P_fp32.bin"),
            str(K), str(tmp_dir),
        ], env=env, check=True)
        t_b4 = time.perf_counter()

        # Copy core_index header for tier1_b1_builder_v4
        shutil.copy2(src_dir / "core_index.bin", tmp_dir / "core_index.bin")
        subprocess.run(["taskset", "-c", "0-47", str(B1_V4), str(tmp_dir), "48"], env=env, check=True)
        t_b1 = time.perf_counter()

        res = {
            "tag": tag, "n_docs": N_docs, "tot_toks": tot_toks, "K": K,
            "kmeans_gpu_s": round(t_km - t_start, 2),
            "assign_and_stage_s": round(t_assign - t_km, 2),
            "tier1_b4_and_tier2_nvme_48c_s": round(t_b4 - t_assign, 2),
            "tier1_b1_and_mnorm_48c_s": round(t_b1 - t_b4, 2),
            "total_build_s": round(t_b1 - t_start, 2),
        }
        print("  [PM Build Time]", json.dumps(res), flush=True)
        return res
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if "--build-only" not in sys.argv:
        measure_pm_process_memory()
    brows = [
        time_pm_build_one("vidore_v3_am", P09 / "stage3" / "indexes" / "vidore_v3_adamerge", 16384),
        time_pm_build_one("lotte_200k", P09 / "stage3" / "indexes" / "lotte_200k_matched", 16384),
    ]
    out = OUT_DIR / "pm_build_times.json"
    with open(out, "w") as f:
        json.dump({"rows": brows}, f, indent=2)
    print(f"Saved {out}", flush=True)
