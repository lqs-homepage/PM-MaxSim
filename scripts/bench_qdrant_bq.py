#!/usr/bin/env python3
"""
Hard-exclusive Slurm 48-core Socket-0 NUMA-bound benchmark for Qdrant v1.20.2 Binary Quantization (BQ + On-Disk):
  - Uses barrier-synchronized persistent clients + warmup BEFORE timing gate (RES-P09-02).
  - Logs Little's Law coherence ratio + idle & active Linux smaps_rollup PSS/VmRSS/VmHWM (RES-P09-03).
  - Performs genuine Cold-Start check by STOPPING Qdrant, evicting OS page cache with posix_fadvise(DONTNEED),
    restarting Qdrant fresh, and running 48 barrier-synchronized clients with ZERO query warmup!
"""
import json
import os
import subprocess
import time
from pathlib import Path

from bench_external_dbs import (
    TP, EXT_ROOT, OUT_DIR, NUMA_PREFIX,
    evict_dir_page_cache, get_pids_mem_and_io, load_corpus_gt,
    get_workload_specs, run_barrier_steady_state,
)


def start_qdrant_bq(q_data: Path):
    from qdrant_client import QdrantClient
    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(q_data / "storage")
    env["QDRANT__SERVICE__HTTP_PORT"] = "56345"
    env["QDRANT__SERVICE__GRPC_PORT"] = "56346"
    log_fh = open(q_data / "q_bench.log", "a")
    proc = subprocess.Popen(
        NUMA_PREFIX + [str(TP / "qdrant" / "qdrant")],
        env=env, stdout=log_fh, stderr=subprocess.STDOUT,
    )
    c0 = QdrantClient(host="127.0.0.1", port=56345, grpc_port=56346, prefer_grpc=True, timeout=120)
    for _ in range(60):
        try:
            c0.get_collections()
            break
        except Exception:
            time.sleep(0.5)
    c0.close()
    return proc, log_fh


def stop_qdrant_bq(proc, log_fh):
    proc.terminate()
    proc.wait(timeout=30)
    log_fh.close()
    time.sleep(1.0)


def bench_qdrant_bq_one(tag: str, rows_out: list):
    from qdrant_client import QdrantClient, models

    q_data = EXT_ROOT / f"qdrant_bq_{tag}"
    if not (q_data / "INGEST_DONE.json").exists():
        print(f"[SKIP] {q_data} not ingested", flush=True)
        return
    q_list, wls, contract, qdoms, qrels = load_corpus_gt(tag)
    wl_specs = get_workload_specs(tag, contract, qdoms)
    q_py_lists = [qmat.tolist() for qmat in q_list]

    def make_filter(spec_tuple):
        if spec_tuple is None:
            return None
        k, val = spec_tuple
        if k == "n_toks_ge":
            return models.Filter(must=[models.FieldCondition(key="n_toks", range=models.Range(gte=val))])
        return models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=val))])

    evict_dir_page_cache(q_data)
    proc, log_fh = start_qdrant_bq(q_data)
    try:
        _, idle_pss_b, _, _, _ = get_pids_mem_and_io("third_party/qdrant/qdrant")
        idle_pss_gb = round(idle_pss_b / (1024 ** 3), 3)

        configs = [
            (64, 2.0, [1, 48]),
            (128, 2.0, [1, 8, 24, 48]),
            (256, 2.0, [1, 48]),
            (128, 1.0, [48]),
        ]
        for ef, ovs, tgrid in configs:
            sp = models.SearchParams(
                hnsw_ef=ef,
                quantization=models.QuantizationSearchParams(rescore=True, oversampling=ovs),
            )
            for wl_name, pred_fn in wl_specs:
                if wl_name not in wls:
                    continue
                gt10 = wls[wl_name]["gt10"]
                nq = len(q_list)
                for nT in tgrid:
                    def make_cli(_tid):
                        return QdrantClient(host="127.0.0.1", port=56345, grpc_port=56346, prefer_grpc=True, timeout=120)

                    def run_q(cli, q):
                        flt = make_filter(pred_fn(q)[1])
                        res = cli.query_points(collection_name="docs", query=q_py_lists[q], query_filter=flt, search_params=sp, limit=10)
                        return [int(pt.id) for pt in res.points]

                    def close_cli(cli):
                        cli.close()

                    m = run_barrier_steady_state(
                        nT=nT, nq=nq, gt10=gt10, qrels=qrels, wl_name=wl_name,
                        make_client_fn=make_cli, run_query_fn=run_q, close_client_fn=close_cli,
                        mem_match_substr="third_party/qdrant/qdrant",
                        target_wall_s=6.0, n_warmup=2, fixed_total_q=(min(nq, 40) if nT == 1 else 0),
                    )
                    row = {
                        "tag": tag, "db": "Qdrant_1.20.2_BQ_OnDisk", "config": f"ef={ef},os={ovs:.0f}",
                        "workload": wl_name, "threads": nT, "idle_pss_gb": idle_pss_gb,
                        **m,
                    }
                    rows_out.append(row)
                    print(
                        f"  [{tag} Qdrant-BQ ef={ef:3d} os={ovs:.0f} {wl_name} T={nT:2d}] "
                        f"QPS={row['qps']:7.1f} P50={row['p50_ms']:7.1f}ms LL={row['littles_law']:.2f} "
                        f"n={row['n_eval']:4d} R@10={row['recall10']:.3f} All10={row['all10']:.2f} "
                        f"IO={row['proc_mb_q']:.2f}MB/q PSS={row['active_pss_gb']:.2f}G",
                        flush=True,
                    )
    finally:
        stop_qdrant_bq(proc, log_fh)

    # Genuine cold-start check on lotte_200k: stop server -> evict page cache -> restart fresh -> 0 warmup!
    if tag == "lotte_200k":
        sp_cold = models.SearchParams(
            hnsw_ef=128,
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0),
        )
        for wl_name, pred_fn in wl_specs:
            if wl_name not in ("NoCorr_5.00pct_ExactNC5", "NoCorr_100.00pct"):
                continue
            evict_dir_page_cache(q_data)
            cproc, clog = start_qdrant_bq(q_data)
            try:
                _, idle_pss_b, _, _, _ = get_pids_mem_and_io("third_party/qdrant/qdrant")
                gt10 = wls[wl_name]["gt10"]
                nq = len(q_list)

                def make_cli(_tid):
                    return QdrantClient(host="127.0.0.1", port=56345, grpc_port=56346, prefer_grpc=True, timeout=120)

                def run_q(cli, q):
                    flt = make_filter(pred_fn(q)[1])
                    res = cli.query_points(collection_name="docs", query=q_py_lists[q], query_filter=flt, search_params=sp_cold, limit=10)
                    return [int(pt.id) for pt in res.points]

                def close_cli(cli):
                    cli.close()

                m = run_barrier_steady_state(
                    nT=48, nq=nq, gt10=gt10, qrels=None, wl_name=wl_name,
                    make_client_fn=make_cli, run_query_fn=run_q, close_client_fn=close_cli,
                    mem_match_substr="third_party/qdrant/qdrant",
                    target_wall_s=0.0, n_warmup=0, fixed_total_q=192,
                )
                row = {
                    "tag": tag, "db": "Qdrant_1.20.2_BQ_OnDisk", "config": "ef=128,os=2,COLD_RESTART",
                    "workload": wl_name, "threads": 48, "idle_pss_gb": round(idle_pss_b / (1024 ** 3), 3),
                    **m,
                }
                rows_out.append(row)
                print(
                    f"  [COLD-RESTART {tag} Qdrant-BQ ef=128 os=2 {wl_name} T=48] "
                    f"QPS={row['qps']:7.1f} P50={row['p50_ms']:7.1f}ms LL={row['littles_law']:.2f} "
                    f"R@10={row['recall10']:.3f} IO={row['proc_mb_q']:.2f}MB/q PSS={row['active_pss_gb']:.2f}G",
                    flush=True,
                )
            finally:
                stop_qdrant_bq(cproc, clog)


def main():
    rows = []
    for tag in ("vidore_v3_am", "lotte_200k"):
        bench_qdrant_bq_one(tag, rows)
    out = OUT_DIR / "qdrant_bq_s33.json"
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"Saved {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
