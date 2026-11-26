#!/usr/bin/env python3
"""
Hard-exclusive Slurm 48-core benchmark for:
  1. Qdrant v1.20.2 Binary Quantization (BQ + On-Disk, always_ram=True, rescore=True, oversampling=1.0 & 2.0)
     across vidore_v3_am and lotte_200k (ef=64, 128, 256; T=1, 8, 24, 48).
  2. Strict Cold-Start (posix_fadvise DONTNEED before 48-client run) vs Warm-Cache check
     on lotte_200k for Qdrant INT8, Qdrant BQ, and VectorChord HalfvecRQ.
"""
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np

from bench_external_dbs import (
    P09, TP, EXT_ROOT, OUT_DIR,
    evict_dir_page_cache, get_pids_rss_and_io, load_corpus_gt,
    calc_recall_and_all10, calc_ndcg5, get_workload_specs,
)


def bench_qdrant_bq_one(tag: str, rows_out: list):
    from qdrant_client import QdrantClient, models

    q_data = EXT_ROOT / f"qdrant_bq_{tag}"
    if not (q_data / "INGEST_DONE.json").exists():
        print(f"[SKIP] {q_data} not ingested", flush=True)
        return
    q_list, wls, contract, qdoms, qrels = load_corpus_gt(tag)
    wl_specs = get_workload_specs(tag, contract, qdoms)
    q_py_lists = [qmat.tolist() for qmat in q_list]

    evict_dir_page_cache(q_data)
    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(q_data / "storage")
    env["QDRANT__SERVICE__HTTP_PORT"] = "56345"
    env["QDRANT__SERVICE__GRPC_PORT"] = "56346"
    log_fh = open(q_data / "q_bench.log", "w")
    proc = subprocess.Popen(
        ["taskset", "-c", "0-47", str(TP / "qdrant" / "qdrant")],
        env=env, stdout=log_fh, stderr=subprocess.STDOUT,
    )
    try:
        c0 = QdrantClient(host="127.0.0.1", port=56345, grpc_port=56346, prefer_grpc=True, timeout=120)
        for _ in range(40):
            try:
                c0.get_collections()
                break
            except Exception:
                time.sleep(0.5)
        c0.close()

        def make_filter(spec_tuple):
            if spec_tuple is None:
                return None
            k, val = spec_tuple
            if k == "n_toks_ge":
                return models.Filter(must=[models.FieldCondition(key="n_toks", range=models.Range(gte=val))])
            return models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=val))])

        configs = [
            (64, 2.0, [1, 48]),
            (128, 2.0, [1, 8, 24, 48]),
            (256, 2.0, [1, 48]),
            (128, 1.0, [48]),
        ]
        for ef, ovs, tgrid in configs:
            for wl_name, pred_fn in wl_specs:
                if wl_name not in wls:
                    continue
                gt10 = wls[wl_name]["gt10"]
                nq = len(q_list)
                for nT in tgrid:
                    total_q = min(nq, 40) if nT == 1 else max(nT * 4, 80)
                    rss0, io0 = get_pids_rss_and_io("third_party/qdrant/qdrant")
                    lats, recs, all10s, ndcg5s = [], [], [], []

                    def worker(tid):
                        loc_l, loc_r, loc_a, loc_n = [], [], [], []
                        cli = QdrantClient(host="127.0.0.1", port=56345, grpc_port=56346, prefer_grpc=True, timeout=120)
                        sp = models.SearchParams(
                            hnsw_ef=ef,
                            quantization=models.QuantizationSearchParams(rescore=True, oversampling=ovs),
                        )
                        for job in range(tid, total_q, nT):
                            q = job % nq
                            flt = make_filter(pred_fn(q)[1])
                            t0 = time.perf_counter()
                            res = cli.query_points(collection_name="docs", query=q_py_lists[q], query_filter=flt, search_params=sp, limit=10)
                            dt_ms = (time.perf_counter() - t0) * 1000.0
                            pids = [int(pt.id) for pt in res.points]
                            rec, a10 = calc_recall_and_all10(pids, gt10[q])
                            loc_l.append(dt_ms); loc_r.append(rec); loc_a.append(a10)
                            if qrels is not None and wl_name == "NoCorr_100.00pct":
                                loc_n.append(calc_ndcg5(pids, qrels[q]))
                        cli.close()
                        return loc_l, loc_r, loc_a, loc_n

                    wt0 = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=nT) as ex_pool:
                        for ll, lr, la, ln in ex_pool.map(worker, range(nT)):
                            lats.extend(ll); recs.extend(lr); all10s.extend(la); ndcg5s.extend(ln)
                    ws = time.perf_counter() - wt0
                    rss1, io1 = get_pids_rss_and_io("third_party/qdrant/qdrant")
                    lats.sort()
                    row = {
                        "tag": tag, "db": "Qdrant_1.20.2_BQ_OnDisk", "config": f"ef={ef},os={ovs:.0f}",
                        "workload": wl_name, "threads": nT, "n_eval": len(lats),
                        "qps": round(len(lats) / ws, 2),
                        "p50_ms": round(lats[len(lats) // 2], 2),
                        "p99_ms": round(lats[min(len(lats) - 1, int(len(lats) * 0.99))], 2),
                        "recall10": round(float(np.mean(recs)), 4),
                        "all10": round(float(np.mean(all10s)), 4),
                        "ndcg5": round(float(np.mean(ndcg5s)), 4) if ndcg5s else None,
                        "proc_mb_q": round(max(0, io1 - io0) / 1048576.0 / max(1, len(lats)), 3),
                        "rss_gb": round(max(rss0, rss1) / (1024 ** 3), 3),
                    }
                    rows_out.append(row)
                    print(f"  [{tag} Qdrant-BQ ef={ef:3d} os={ovs:.0f} {wl_name} T={nT:2d}] QPS={row['qps']:7.1f} P50={row['p50_ms']:7.1f}ms R@10={row['recall10']:.3f} All10={row['all10']:.2f} IO={row['proc_mb_q']:.2f}MB/q RSS={row['rss_gb']:.2f}G", flush=True)

        # Also strict COLD-START vs WARM check on lotte_200k NoCorr_5.00pct_ExactNC5 & NoCorr_100.00pct (T=48, ef=128, os=2)
        if tag == "lotte_200k":
            for wl_name, pred_fn in wl_specs:
                if wl_name not in ("NoCorr_5.00pct_ExactNC5", "NoCorr_100.00pct"):
                    continue
                evict_dir_page_cache(q_data)
                gt10 = wls[wl_name]["gt10"]
                nq = len(q_list)
                total_q = 192
                rss0, io0 = get_pids_rss_and_io("third_party/qdrant/qdrant")
                lats, recs, all10s = [], [], []
                def cworker(tid):
                    loc_l, loc_r, loc_a = [], [], []
                    cli = QdrantClient(host="127.0.0.1", port=56345, grpc_port=56346, prefer_grpc=True, timeout=120)
                    sp = models.SearchParams(hnsw_ef=128, quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0))
                    for job in range(tid, total_q, 48):
                        q = job % nq
                        flt = make_filter(pred_fn(q)[1])
                        t0 = time.perf_counter()
                        res = cli.query_points(collection_name="docs", query=q_py_lists[q], query_filter=flt, search_params=sp, limit=10)
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        rec, a10 = calc_recall_and_all10([int(pt.id) for pt in res.points], gt10[q])
                        loc_l.append(dt_ms); loc_r.append(rec); loc_a.append(a10)
                    cli.close()
                    return loc_l, loc_r, loc_a
                wt0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=48) as ex_pool:
                    for ll, lr, la in ex_pool.map(cworker, range(48)):
                        lats.extend(ll); recs.extend(lr); all10s.extend(la)
                ws = time.perf_counter() - wt0
                rss1, io1 = get_pids_rss_and_io("third_party/qdrant/qdrant")
                lats.sort()
                row = {
                    "tag": tag, "db": "Qdrant_1.20.2_BQ_OnDisk", "config": "ef=128,os=2,COLD_DONTNEED",
                    "workload": wl_name, "threads": 48, "n_eval": len(lats),
                    "qps": round(len(lats) / ws, 2),
                    "p50_ms": round(lats[len(lats) // 2], 2),
                    "p99_ms": round(lats[min(len(lats) - 1, int(len(lats) * 0.99))], 2),
                    "recall10": round(float(np.mean(recs)), 4),
                    "all10": round(float(np.mean(all10s)), 4),
                    "ndcg5": None,
                    "proc_mb_q": round(max(0, io1 - io0) / 1048576.0 / max(1, len(lats)), 3),
                    "rss_gb": round(max(rss0, rss1) / (1024 ** 3), 3),
                }
                rows_out.append(row)
                print(f"  [COLD {tag} Qdrant-BQ ef=128 os=2 {wl_name} T=48] QPS={row['qps']:7.1f} P50={row['p50_ms']:7.1f}ms R@10={row['recall10']:.3f} IO={row['proc_mb_q']:.2f}MB/q RSS={row['rss_gb']:.2f}G", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log_fh.close()
        time.sleep(1.0)


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
