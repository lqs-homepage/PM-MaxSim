#!/usr/bin/env python3
"""
Layer 4 Kernel-Enforced `cgroup v2 MemoryMax` Hard-RAM SLA Benchmark Harness:
  Runs inside a hard-exclusive Slurm allocation (`#SBATCH --exclusive`, Socket 0 CPUs 0-47),
  spawning each target process inside a kernel-enforced cgroup v2 step (`srun --ntasks=1 --cpus-per-task=48
  --mem=<cap>`, where Slurm `cgroup.conf` enforces `CgroupPlugin=cgroup/v2`, `ConstrainRAMSpace=yes`,
  `AllowedSwapSpace=0` -> exact kernel `memory.max` + `memory.swap.max=0` on Socket 0!).
  Falls back to `systemd-run --user --scope` ONLY when invoked outside Slurm (`--runner systemd`).

  Captures per cage (`2G / 4G / 8G / 16G`):
    - Exact kernel `/sys/fs/cgroup/.../memory.max`, `memory.swap.max`, `memory.peak`, `memory.events` (`oom`, `oom_kill`)
    - Exit code & `oom_killed` status (whether the engine survives gracefully or crashes under hard RAM cap!)
    - Full engine benchmark JSON (`QPS`, `P50/P99`, `Recall@10`, `B4/T2` S3-FIFO hit rates, NVMe bytes read)
"""
import argparse
import json
import os
import pathlib
import subprocess
import time

P09 = pathlib.Path(os.environ.get("P09_ROOT", "${P09_ROOT}"))


def evict_dir_page_cache(dir_path: pathlib.Path):
    """Evict Linux page cache for all files in dir_path so cgroup v2 charges cold mmap faults cleanly!"""
    if not dir_path or not dir_path.exists():
        return 0
    cnt = 0
    for root, _, files in os.walk(dir_path):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                fd = os.open(p, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                cnt += 1
            except OSError:
                pass
    return cnt


def parse_mem_cap_mb(cap: str) -> int:
    s = cap.strip().upper()
    if s.endswith("G"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("M"):
        return int(float(s[:-1]))
    return int(s)


def run_in_kernel_cgroup(cmd: list[str], mem_cap_str: str, runner: str, unit_name: str, env=None, timeout_s: int = 1800):
    """Run `cmd` inside a strict kernel-enforced cgroup v2 memory cage (`memory.max`, `memory.swap.max=0`)."""
    inner = (
        'CG=$(awk -F: \'$1==0{print $3}\' /proc/self/cgroup); '
        'CGP="/sys/fs/cgroup${CG}"; '
        'if [ "$(cat "$CGP/memory.max" 2>/dev/null)" = "max" ] && [ -f "$(dirname "$CGP")/memory.max" ]; then CGP="$(dirname "$CGP")"; fi; '
        'echo "===CGROUP_PRE==="; '
        'echo "cg_path=$CGP"; '
        'echo "memory_max=$(cat $CGP/memory.max 2>/dev/null || echo NA)"; '
        'echo "memory_swap_max=$(cat $CGP/memory.swap.max 2>/dev/null || echo NA)"; '
        'echo "cpus_allowed=$(grep Cpus_allowed_list /proc/self/status | awk \'{print $2}\')"; '
        + " ".join(subprocess.list2cmdline([c]) for c in cmd) + '; '
        'RC=$?; '
        'echo "===CGROUP_POST==="; '
        'echo "memory_peak=$(cat $CGP/memory.peak 2>/dev/null || echo 0)"; '
        'tr "\n" " " < "$CGP/memory.events" 2>/dev/null | sed "s/^/memory_events=/"; echo ""; '
        'exit $RC'
    )
    if runner == "slurm":
        mem_mb = parse_mem_cap_mb(mem_cap_str)
        full_cmd = [
            "srun", "--ntasks=1", "--cpus-per-task=96",
            f"--mem={mem_mb}M",
            "/bin/bash", "-c", inner,
        ]
    else:
        full_cmd = [
            "systemd-run", "--user", "--scope",
            f"--unit={unit_name}",
            "-p", f"MemoryMax={mem_cap_str}",
            "-p", "MemorySwapMax=0",
            "/bin/bash", "-c", inner,
        ]

    t0 = time.perf_counter()
    try:
        p = subprocess.run(full_cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
        rc, out_s, err_s = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as ex:
        rc = 124
        out_s = (ex.stdout or b"").decode("utf-8", errors="ignore") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
        err_s = (ex.stderr or b"").decode("utf-8", errors="ignore") if isinstance(ex.stderr, bytes) else (ex.stderr or "")
    elapsed = time.perf_counter() - t0
    oom = (rc in (137, -9)) or ("oom" in err_s.lower()) or ("out of memory" in err_s.lower())
    stats = {
        "runner": runner, "mem_cap": mem_cap_str,
        "returncode": rc, "elapsed_s": round(elapsed, 3),
        "oom_killed": oom, "memory_max_gb": 0.0, "memory_swap_max": "",
        "memory_peak_gb": 0.0, "memory_events": {}, "cpus_allowed": "",
    }
    for line in out_s.splitlines():
        if line.startswith("memory_max="):
            v = line.split("=", 1)[1].strip()
            if v.isdigit():
                stats["memory_max_gb"] = round(int(v) / (1024.0 ** 3), 4)
        elif line.startswith("memory_swap_max="):
            stats["memory_swap_max"] = line.split("=", 1)[1].strip()
        elif line.startswith("cpus_allowed="):
            stats["cpus_allowed"] = line.split("=", 1)[1].strip()
        elif line.startswith("memory_peak="):
            v = line.split("=", 1)[1].strip()
            if v.isdigit():
                stats["memory_peak_gb"] = round(int(v) / (1024.0 ** 3), 4)
        elif line.startswith("memory_events="):
            toks = line.split("=", 1)[1].strip().split()
            for i in range(0, len(toks) - 1, 2):
                try:
                    stats["memory_events"][toks[i]] = int(toks[i + 1])
                except Exception:
                    pass
    if stats["memory_events"].get("oom_kill", 0) > 0:
        stats["oom_killed"] = True
    stats["stdout_tail"] = out_s[-600:]
    stats["stderr_tail"] = err_s[-400:]
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cgroup_zipf_pm", "cgroup_certvec5_pm", "cgroup_qdrant", "cgroup_pg17_vchord"], required=True)
    ap.add_argument("--runner", choices=["slurm", "systemd"], default="slurm")
    ap.add_argument("--mem-cap", required=True, help="e.g. 2G, 4G, 8G, 16G")
    ap.add_argument("--ram-budget-mb", type=int, default=1024)
    ap.add_argument("--idx-dir", default="")
    ap.add_argument("--sidecar-dir", default="-")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--kind", default="bgem3")
    ap.add_argument("--corpus", default="bgem3_1024d_hybrid")
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--n-reqs", type=int, default=50000)
    ap.add_argument("--mutate-pct", type=int, default=2)
    args = ap.parse_args()

    out_p = pathlib.Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    bin_v4 = str(P09 / "stage3_v4" / "src" / "certvec_bench_v4")
    bin_c5 = str(P09 / "stage3_v4" / "src" / "certvec_bench_certvec5")
    q_dir = P09 / "stage3_v4" / "query_streams"
    inner_json = str(out_p.with_suffix(".inner.json"))
    uname = f"p09-cg-{args.mode}-{args.mem_cap}-{int(time.time())}"
    numa = ["numactl", "--cpunodebind=0,1", "--membind=0,1", "taskset", "-c", "0-47"] if args.runner == "slurm" else []

    if args.mode == "cgroup_zipf_pm":
        evict_dir_page_cache(pathlib.Path(args.idx_dir))
        if args.sidecar_dir and args.sidecar_dir != "-":
            evict_dir_page_cache(pathlib.Path(args.sidecar_dir))
        cmd = numa + [
            bin_v4, "--bench-zipf-cache",
            args.idx_dir, args.sidecar_dir,
            str(q_dir / "msmarco_dev_6980_q32.f32"), "6980", "32",
            str(q_dir / "zipf1_100k_qids.u32"), str(q_dir / "zipf1_100k_tids.u16"),
            str(args.n_reqs), inner_json,
            "--threads", str(args.threads), "--tier", "auto",
            "--ram-mb", str(args.ram_budget_mb), "--mutate-pct", str(args.mutate_pct),
        ]
    elif args.mode == "cgroup_certvec5_pm":
        evict_dir_page_cache(pathlib.Path(args.idx_dir))
        np_val = "128" if args.kind == "docs768" else "16"
        c1_val = "3000" if args.kind == "docs768" else "600"
        cmd = numa + [
            bin_c5, args.idx_dir, str(pathlib.Path(args.idx_dir) / "eval_bundle.bin"),
            args.kind, "auto", inner_json,
            "--threads", str(args.threads), "--ram-mb", str(args.ram_budget_mb),
            "--nprobe", np_val, "--c1", c1_val,
        ]
    elif args.mode == "cgroup_qdrant":
        dir_tag = "bgem3_1024d" if args.corpus == "bgem3_1024d_hybrid" else args.corpus
        evict_dir_page_cache(P09 / "stage3_v4" / "ext_dbs" / f"qdrant_{args.quant}_{dir_tag}")
        cmd = [
            "/home/user/.local/bin/uv", "run", "--with", "qdrant-client", "--with", "numpy",
            "python", str(P09 / "stage3_v4" / "scripts" / "bench_qdrant_new_corpora.py"),
            "--corpus", args.corpus, "--quant", args.quant,
            "--out-json", inner_json, "--threads", str(args.threads),
        ]
    elif args.mode == "cgroup_pg17_vchord":
        cmd = [
            "/home/user/.local/bin/uv", "run", "--with", "psycopg[binary]", "--with", "numpy",
            "python", str(P09 / "stage3_v4" / "scripts" / "bench_sql_and_cgroup_v4.py"),
            "--pg-data", args.idx_dir,
            "--idx-dir", args.sidecar_dir,
            "--sidecar-dir", "-",
            "--tag", args.corpus,
            "--out-json", inner_json,
            "--threads", str(args.threads),
            "--cache-mb", str(args.ram_budget_mb),
            "--shared-buf-mb", "512",
            "--no-numa",
        ]
    else:
        raise ValueError(args.mode)

    cg = run_in_kernel_cgroup(cmd, args.mem_cap, args.runner, uname)
    res = {"mode": args.mode, "mem_cap": args.mem_cap, "ram_budget_mb": args.ram_budget_mb, "cgroup": cg}
    if pathlib.Path(inner_json).exists():
        try:
            txt = pathlib.Path(inner_json).read_text().strip()
            if txt:
                res["inner"] = json.loads(txt)
        except Exception:
            pass
    out_p.write_text(json.dumps(res, indent=2) + "\n")
    print(
        f"[cgroup {args.mode} runner={args.runner} cap={args.mem_cap}] rc={cg['returncode']} "
        f"oom={cg['oom_killed']} max={cg['memory_max_gb']}GB swap={cg['memory_swap_max']} "
        f"peak={cg['memory_peak_gb']}GB events={cg['memory_events']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
