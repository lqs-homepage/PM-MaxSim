#!/usr/bin/env python3
"""
Remediated Stage 3.3 Measurement Script (RES-P09-03 & RES-P09-04):
  1. Isolated OS Linux smaps_rollup Idle PSS, Active 48-Thread In-Flight Peak PSS/RSS, and VmHWM
     for ALL 3 PM-MaxSim deployment tiers (PM_NVME_T1, Split_B1_B4_T3, PM_DRAM_T1) across all 5 corpora
     (with compacted dynamic io_uring bounce buffers and --skip-unit-test).
  2. Complete From-Scratch PM-MaxSim Index Build Time on vidore_v3_am & lotte_200k (ZERO shutil.copy2!):
       - Builds spherical K-means, full token centroid assignment, Haar rotation P,
         deduplicated doc_cids + sorted inverted lists + CRoaring bitmaps + core_index.bin (via core_index_builder_v4),
         Tier-1 B4 + Tier-2 FP16 4KB-aligned NVMe (via rabitq_nvme_builder_v3),
         and Tier-1 B1 + M_norm scan (via tier1_b1_builder_v4) from scratch!
       - Measures BOTH:
           (a) 48-core CPU-Only (strict hardware parity against CPU-only external DBs!)
           (b) Optional GPU K-means/Assignment + 48-core CPU Tier-0/1/2 (explicitly labeled GPU-accelerated!)
       - Reports BOTH pure index build time (`idx_build_s`) AND full disk-read-to-index total (`total_with_disk_load_s`).
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import torch

P09 = Path(os.environ.get("P09_ROOT", "./workspace"))
SRC = P09 / "stage3" / "src"
OUT_DIR = P09 / "stage3" / "results_s33"
RABITQ_V3 = P09 / "gate_2b" / "rabitq_nvme_builder_v3"
CORE_V4 = SRC / "core_index_builder_v4"
B1_V4 = SRC / "tier1_b1_builder_v4"
NUMA_PREFIX = ["numactl", "--cpunodebind=0,1", "--membind=0,1", "taskset", "-c", "0-47"]


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
            cmd = NUMA_PREFIX + [
                str(SRC / "pm_engine_v4"),
                "--bench-single", str(idx_dir), str(tmp_json),
                "--skip-unit-test", "--tier-mem-mode", mode,
                "--threads", "48", "--workloads", wl, "--methods", m_id,
            ]
            subprocess.run(cmd, check=True)
            d = json.load(open(tmp_json))
            pm = d["proc_mem_gb"]
            row = {
                "corpus": corp, "method": m_lbl, "tier_mem_mode": mode,
                "idle_pss_gb": round(pm["idle_pss_gb"], 4),
                "active_pss_gb": round(pm["active_pss_gb"], 4),
                "active_rss_gb": round(pm["active_rss_gb"], 4),
                "pss_gb": round(pm["active_pss_gb"], 4),
                "rss_gb": round(pm["active_rss_gb"], 4),
                "vm_hwm_gb": round(pm["vm_hwm_gb"], 4),
            }
            rows.append(row)
            print("  [PM True Mem]", json.dumps(row), flush=True)

    ms_dir = P09 / "stage3" / "indexes" / "msmarco_8m_lsm"
    for mode, m_id, m_lbl in [("nvme", "2", "PM_NVME_T1"), ("split", "0", "Split_B1_B4_T3"), ("dram", "1", "PM_DRAM_T1")]:
        cmd = NUMA_PREFIX + [
            str(SRC / "pm_multilsm_v4"),
            "--bench-lsm", str(ms_dir), str(tmp_json),
            "--skip-compact", "--skip-unit-test", "--tier-mem-mode", mode,
            "--threads", "48", "--workloads", "NoCorr_5.00pct", "--methods", m_id,
        ]
        subprocess.run(cmd, check=True)
        d = json.load(open(tmp_json))
        pm = d["proc_mem_gb"]
        row = {
            "corpus": "msmarco_8m_lsm", "method": m_lbl, "tier_mem_mode": mode,
            "idle_pss_gb": round(pm["idle_pss_gb"], 4),
            "active_pss_gb": round(pm["active_pss_gb"], 4),
            "active_rss_gb": round(pm["active_rss_gb"], 4),
            "pss_gb": round(pm["active_pss_gb"], 4),
            "rss_gb": round(pm["active_rss_gb"], 4),
            "vm_hwm_gb": round(pm["vm_hwm_gb"], 4),
        }
        rows.append(row)
        print("  [PM True Mem]", json.dumps(row), flush=True)

    if tmp_json.exists():
        tmp_json.unlink()
    out = OUT_DIR / "pm_true_memory.json"
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"Saved {out}", flush=True)


@torch.no_grad()
def time_pm_build_one(tag: str, src_dir: Path, K: int, is_unit: int, mode: str):
    """
    mode in {'cpu48', 'gpu_kmeans_cpu48'}.
    Zero shutil.copy2! Constructs core_index.bin, CRoaring bitmaps, Tier-1 B4, Tier-2 FP16, Tier-1 B1, and m_norm.bin from scratch.
    """
    assert mode in ("cpu48", "gpu_kmeans_cpu48")
    use_gpu = (mode == "gpu_kmeans_cpu48")
    if use_gpu:
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.set_num_threads(48)

    tmp_dir = P09 / "stage3" / f"tmp_build_timing_{tag}_{mode}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        # Phase 0: Read raw FP16 embeddings from disk (timed separately so both idx_build_s and total_with_disk_load_s are transparent!)
        t_disk0 = time.perf_counter()
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
        t_start = time.perf_counter()
        disk_load_s = t_start - t_disk0

        # Phase 1: Spherical K-means (12 iterations)
        rng = np.random.default_rng(202703)
        samp = rng.choice(tot_toks, size=min(800000, tot_toks), replace=False)
        dev = "cuda:0" if use_gpu else "cpu"
        X = torch.from_numpy(vecs[samp].astype(np.float32)).to(dev)
        X /= torch.clamp(X.norm(dim=1, keepdim=True), min=1e-12)
        C = X[rng.choice(len(X), size=K, replace=False)].clone()
        bs_km = 65536 if use_gpu else 16384
        for _ in range(12):
            sums = torch.zeros_like(C)
            cnts = torch.zeros(K, device=dev, dtype=torch.float32)
            for s in range(0, len(X), bs_km):
                xb = X[s:s + bs_km]
                a = torch.matmul(xb, C.T).argmax(dim=1)
                sums.index_add_(0, a, xb)
                cnts.index_add_(0, a, torch.ones_like(a, dtype=torch.float32))
            empty = cnts < 0.5
            if empty.any():
                sums[empty] = X[rng.choice(len(X), size=int(empty.sum().item()), replace=False)]
                cnts[empty] = 1.0
            C = sums / cnts.unsqueeze(1)
            C /= torch.clamp(C.norm(dim=1, keepdim=True), min=1e-12)
        if use_gpu:
            torch.cuda.synchronize()
        t_km = time.perf_counter()

        # Phase 2: Full-corpus centroid assignment + Haar rotation P
        cids_all = np.empty(tot_toks, dtype=np.uint16)
        CT = C.T.contiguous()
        bs_asgn = 131072 if use_gpu else 32768
        for s in range(0, tot_toks, bs_asgn):
            e = min(tot_toks, s + bs_asgn)
            xb = torch.from_numpy(vecs[s:e].astype(np.float32)).to(dev)
            cids_all[s:e] = torch.matmul(xb, CT).argmax(dim=1).cpu().numpy().astype(np.uint16)
        if use_gpu:
            torch.cuda.synchronize()

        C_np = C.cpu().numpy().astype(np.float32)
        A = rng.standard_normal((128, 128))
        Q_h, R_h = np.linalg.qr(A)
        sgn = np.sign(np.diag(R_h)); sgn[sgn == 0] = 1.0
        P = np.ascontiguousarray(Q_h * sgn[None, :], dtype=np.float32)

        C_np.tofile(tmp_dir / "C_fp32.bin")
        P.tofile(tmp_dir / "P_fp32.bin")
        vecs.tofile(tmp_dir / "vecs_fp16.bin")
        offs.tofile(tmp_dir / "offs_u64.bin")
        cids_all.tofile(tmp_dir / "cids_u16.bin")
        t_assign = time.perf_counter()

        # Phase 3: Native 48-core C++20 From-Scratch Tier-0 core_index.bin + deduplicated doc_cids + sorted inverted lists + CRoaring bitmaps!
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "48"
        cp_core = subprocess.run(
            NUMA_PREFIX + [
                str(CORE_V4),
                str(tmp_dir / "offs_u64.bin"), str(tmp_dir / "cids_u16.bin"),
                str(tmp_dir / "C_fp32.bin"), str(tmp_dir / "P_fp32.bin"),
                str(K), str(is_unit), str(tmp_dir), "48",
            ],
            env=env, check=True, capture_output=True, text=True,
        )
        core_info = json.loads(cp_core.stdout.strip().splitlines()[-1])
        t_core = time.perf_counter()

        # Phase 4: 48-core C++20 Tier-1 B=4 + Tier-2 FP16 4KB-aligned NVMe builder
        subprocess.run(
            NUMA_PREFIX + [
                str(RABITQ_V3),
                str(tmp_dir / "vecs_fp16.bin"), str(tmp_dir / "offs_u64.bin"), str(tmp_dir / "cids_u16.bin"),
                str(tmp_dir / "C_fp32.bin"), str(tmp_dir / "P_fp32.bin"),
                str(K), str(tmp_dir),
            ],
            env=env, check=True,
        )
        t_b4 = time.perf_counter()

        # Phase 5: 48-core C++20 Tier-1 B=1 + exact M_norm scan (reads our freshly built core_index.bin, ZERO shutil.copy2!)
        subprocess.run(NUMA_PREFIX + [str(B1_V4), str(tmp_dir), "48"], env=env, check=True)
        t_b1 = time.perf_counter()

        # Verify freshly built temp index has all required files non-empty
        for req in ("core_index.bin", "tier1_b4.nvme", "tier2_fp16.nvme", "tier1_b1.ram", "m_norm.bin"):
            assert (tmp_dir / req).stat().st_size > 0, f"Missing or empty {req}"

        idx_build_s = t_b1 - t_start
        res = {
            "tag": tag, "mode": mode,
            "hardware": "48c CPU-Only (Socket-0)" if not use_gpu else "1x B200 GPU K-means/Assign + 48c CPU Tier-0/1/2",
            "n_docs": N_docs, "tot_toks": tot_toks, "K": K,
            "disk_load_s": round(disk_load_s, 2),
            "kmeans_s": round(t_km - t_start, 2),
            "assign_and_stage_s": round(t_assign - t_km, 2),
            "core_index_and_roaring_48c_s": round(t_core - t_assign, 2),
            "roaring_build_s": round(core_info["roaring_build_s"], 3),
            "tier1_b4_and_tier2_nvme_48c_s": round(t_b4 - t_core, 2),
            "tier1_b1_and_mnorm_48c_s": round(t_b1 - t_b4, 2),
            "idx_build_s": round(idx_build_s, 2),
            "total_with_disk_load_s": round(idx_build_s + disk_load_s, 2),
            "shutil_copy2_used": False,
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
        time_pm_build_one("vidore_v3_am", P09 / "stage3" / "indexes" / "vidore_v3_adamerge", 16384, 0, "cpu48"),
        time_pm_build_one("vidore_v3_am", P09 / "stage3" / "indexes" / "vidore_v3_adamerge", 16384, 0, "gpu_kmeans_cpu48"),
        time_pm_build_one("lotte_200k", P09 / "stage3" / "indexes" / "lotte_200k_matched", 16384, 1, "cpu48"),
        time_pm_build_one("lotte_200k", P09 / "stage3" / "indexes" / "lotte_200k_matched", 16384, 1, "gpu_kmeans_cpu48"),
    ]
    out = OUT_DIR / "pm_build_times.json"
    with open(out, "w") as f:
        json.dump({"rows": brows}, f, indent=2)
    print(f"Saved {out}", flush=True)
