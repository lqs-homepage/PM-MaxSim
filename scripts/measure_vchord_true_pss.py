#!/usr/bin/env python3
"""Measure exact Linux PSS (Proportional Set Size, avoiding shared_buffers double-counting) and total VmRSS across all PostgreSQL processes under 48-client load."""
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import psycopg

P09 = Path(os.environ.get("P09_ROOT", "./workspace"))
TP = P09 / "third_party"
EXT_ROOT = P09 / "stage3" / "ext_dbs"
OUT = P09 / "stage3" / "results_s33" / "vchord_true_memory.json"


def sum_pg_pss_rss():
    pids = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            comm = Path(f"/proc/{p}/comm").read_text().strip()
            uid = os.stat(f"/proc/{p}").st_uid
            if comm == "postgres" and uid == os.getuid():
                pids.append(int(p))
        except OSError:
            continue
    tot_pss_kb, tot_rss_kb = 0, 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    tot_pss_kb += int(line.split()[1])
                elif line.startswith("Rss:"):
                    tot_rss_kb += int(line.split()[1])
        except OSError:
            continue
    return len(pids), round(tot_pss_kb / 1048576.0, 3), round(tot_rss_kb / 1048576.0, 3)


def probe_one(prefix: str, tag: str, port: int, col_type: str):
    pg_data = EXT_ROOT / f"{prefix}_{tag}"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{TP}/pg17_vchord/inst/lib:" + env.get("LD_LIBRARY_PATH", "")
    pg_bin = TP / "pg17_vchord" / "inst" / "bin"
    subprocess.run(["taskset", "-c", "0-47", str(pg_bin / "pg_ctl"), "-D", str(pg_data), "-l", str(pg_data / "pg_pss.log"), "start"], env=env, check=True)
    time.sleep(1.0)
    try:
        # Run 48 concurrent connections doing real MaxSim queries & sample PSS while active!
        rng = np.random.default_rng(7)
        q = rng.standard_normal((16, 128)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        q_str = "{" + ",".join('"[' + ",".join(f"{x:.4f}" for x in r) + ']"' for r in q) + "}"
        peak = (0, 0.0, 0.0)

        def worker(_):
            with psycopg.connect(f"host=/tmp port={port} user=postgres dbname=postgres", autocommit=True) as conn:
                conn.execute("SET vchordrq.probes = '16'")
                conn.execute("SET max_parallel_workers_per_gather = 0")
                for _ in range(3):
                    conn.execute(f"SELECT id FROM docs ORDER BY embs @# %s::{col_type} LIMIT 10", (q_str,)).fetchall()

        with concurrent.futures.ThreadPoolExecutor(max_workers=48) as ex:
            futs = [ex.submit(worker, i) for i in range(48)]
            while any(not f.done() for f in futs):
                s = sum_pg_pss_rss()
                if s[1] > peak[1]:
                    peak = s
                time.sleep(0.15)
            for f in futs:
                f.result()
        after = sum_pg_pss_rss()
        res = {"instance": f"{prefix}_{tag}", "n_pg_procs_peak": peak[0], "peak_pss_gb": peak[1], "peak_sum_rss_gb": peak[2], "post_pss_gb": after[1], "shared_buffers_cfg": "8GB"}
        print("  PG Memory:", json.dumps(res), flush=True)
        return res
    finally:
        subprocess.run([str(pg_bin / "pg_ctl"), "-D", str(pg_data), "stop"], env=env, check=False)
        time.sleep(1.0)


if __name__ == "__main__":
    rows = [
        probe_one("pg17_vchord_rq_halfvec", "lotte_200k", 55442, "halfvec(128)[]"),
        probe_one("pg17_vchord_rq_halfvec", "vidore_v3_am", 55442, "halfvec(128)[]"),
        probe_one("pg17_vchord", "lotte_200k", 55441, "vector(128)[]"),
        probe_one("pg17_vchord", "vidore_v3_am", 55441, "vector(128)[]"),
    ]
    with open(OUT, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"Saved {OUT}")
