#!/usr/bin/env python3
"""
PM-MaxSim Hard-Exclusive 1:1 Apples-to-Apples External Production Vector DB Benchmark:
  Runs INSIDE Slurm 48-core hard-exclusive Socket-0 allocation (taskset -c 0-47, swap.max=0).
  Evaluates ALL 5 unmodified external production DB configurations across BOTH corpora:
    - lotte_200k   (exact 200,000 docs matching stage3/indexes/lotte_200k_matched/workloads.bin FP32 GT!)
    - vidore_v3_am (exact 16,868 pages matching stage3/indexes/vidore_v3_adamerge/workloads.bin FP32 GT + official nDCG@5!)
  Systems & Fair Tuned Knobs:
    1. VectorChord 1.5.0 flagship (halfvec(128)[] + residual_quantization=true): probes in [16, 64, 256]
    2. VectorChord 1.5.0 FP32     (vector(128)[]  + residual_quantization=false): probes in [64]
    3. Qdrant v1.20.2             (on-disk MaxSim + INT8 RAM + HNSW, rescore=True): hnsw_ef in [64, 128, 256]
    4. Weaviate v1.41.0 RQ        (MUVERA + HNSW + RQ): ef in [-1 (default), 256]
    5. Weaviate v1.41.0           (MUVERA + HNSW): ef in [-1 (default), 256]
  Metrics:
    - 1-client sequential cold & steady-state P50/P99 latency + Recall@10 + All-10 + official nDCG@5 (ViDoRe v3)
    - Concurrent scaling (1, 8, 24, 48 client workers) QPS + P50/P99 + Recall@10 + server /proc/<pid>/io MB/q + server RSS (GB)!
"""
import argparse
import concurrent.futures
import json
import math
import os
import struct
import subprocess
import time
from pathlib import Path
import numpy as np

P09 = Path(os.environ.get("P09_ROOT", "./workspace"))
TP = P09 / "third_party"
EXT_ROOT = P09 / "stage3" / "ext_dbs"
OUT_DIR = P09 / "stage3" / "results_s33"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evict_dir_page_cache(dir_path: Path):
    """Non-root cold NVMe cache flush across all files in a DB data directory."""
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


def get_pids_rss_and_io(match_substr: str):
    """Sum VmRSS (bytes) and /proc/<pid>/io read_bytes across all matching server processes."""
    rss_b, io_b = 0, 0
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid_str}/cmdline").read_bytes().decode("utf-8", errors="ignore")
            if match_substr not in cmd:
                continue
            for line in Path(f"/proc/{pid_str}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_b += int(line.split()[1]) * 1024
                    break
            for line in Path(f"/proc/{pid_str}/io").read_text().splitlines():
                if line.startswith("read_bytes:"):
                    io_b += int(line.split()[1])
                    break
        except OSError:
            continue
    return rss_b, io_b


def load_corpus_gt(tag: str):
    if tag == "lotte_200k":
        idx_dir = P09 / "stage3" / "indexes" / "lotte_200k_matched"
        contract = json.load(open(idx_dir / "ext_db_eval_contract.json"))
        qrels = None
        qdoms = None
    elif tag == "vidore_v3_am":
        idx_dir = P09 / "stage3" / "indexes" / "vidore_v3_adamerge"
        contract = {"p95_len": 0, "p20_len": 0}
        qrels = json.load(open(P09 / "stage3" / "corpus_vidore_v3" / "vidore_v3_qrels.json"))
        qdoms = np.load(P09 / "stage3" / "corpus_vidore_v3" / "vidore_v3_qdoms.npy").tolist()
    else:
        raise ValueError(tag)

    with open(idx_dir / "core_index.bin", "rb") as f:
        hdr = struct.unpack("<12Q", f.read(96))
        N_docs, K, d, Nq_tot, Lq = hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]
        f.seek(K * d * 4 + d * d * 4, os.SEEK_CUR)
        Q32 = np.frombuffer(f.read(Nq_tot * Lq * d * 4), dtype=np.float32).reshape(Nq_tot, Lq, d).copy()

    wls = {}
    with open(idx_dir / "workloads.bin", "rb") as f:
        _, nwl, nd, nw = struct.unpack("<4Q", f.read(32))
        for _ in range(nwl):
            nm = f.read(64).split(b"\x00", 1)[0].decode()
            _corr = f.read(32)
            sig, nq, nbm, av = struct.unpack("<d3Q", f.read(32))
            f.seek(nbm * nw * 8 + nq * 4, 1)
            gt10 = np.frombuffer(f.read(nq * 10 * 4), dtype=np.uint32).reshape(nq, 10).copy()
            wls[nm] = {"sigma": sig, "nq": nq, "gt10": gt10}

    # Pre-filter valid query tokens once
    q_valid_list = []
    nq_use = 50 if tag == "lotte_200k" else 140
    for q in range(nq_use):
        v = Q32[q]
        q_valid_list.append(v[np.linalg.norm(v, axis=1) > 1e-4])
    return q_valid_list, wls, contract, qdoms, qrels


def calc_recall_and_all10(pred_ids, gt10_row):
    gt_set = set(int(x) for x in gt10_row)
    hits = sum(1 for x in pred_ids[:10] if int(x) in gt_set)
    return hits / 10.0, (1.0 if hits == 10 else 0.0)


def calc_ndcg5(pred_ids, rel_dict):
    if not rel_dict:
        return 0.0
    dcg = 0.0
    for r, did in enumerate(pred_ids[:5]):
        rel = rel_dict.get(str(int(did)), rel_dict.get(int(did), 0.0))
        if rel > 0:
            dcg += (2.0 ** float(rel) - 1.0) / math.log2(r + 2.0)
    ideal = sorted([float(v) for v in rel_dict.values() if float(v) > 0], reverse=True)[:5]
    idcg = sum((2.0 ** rel - 1.0) / math.log2(r + 2.0) for r, rel in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


def get_workload_specs(tag: str, contract, qdoms):
    if tag == "lotte_200k":
        p95 = contract["p95_len"]
        return [
            ("NoCorr_100.00pct", lambda q: ("TRUE", None, None)),
            ("NoCorr_5.00pct_ExactNC5", lambda q: ("nc5 = TRUE", ("nc5", True), ("nc5", "eq", True))),
            ("Nat_DocLenRange_5pct", lambda q: (f"n_toks >= {p95}", ("n_toks_ge", p95), ("n_toks", "gte", p95))),
            ("Nat_Dom0_Filter", lambda q: ("dom = 0", ("dom", 0), ("dom", "eq", 0))),
            ("Nat_Dom1_Filter", lambda q: ("dom = 1", ("dom", 1), ("dom", "eq", 1))),
        ]
    else:
        return [
            ("NoCorr_100.00pct", lambda q: ("TRUE", None, None)),
            ("Nat_DomainEq", lambda q: (f"dom = {qdoms[q]}", ("dom", qdoms[q]), ("dom", "eq", qdoms[q]))),
            ("Nat_CrossDomainNeg", lambda q: (f"dom = {(qdoms[q]+1)%7}", ("dom", (qdoms[q]+1)%7), ("dom", "eq", (qdoms[q]+1)%7))),
        ]


def bench_vectorchord(tag: str, use_rq_halfvec: bool, rows_out: list):
    import psycopg

    prefix = "pg17_vchord_rq_halfvec" if use_rq_halfvec else "pg17_vchord"
    db_label = "VectorChord_1.5_HalfvecRQ" if use_rq_halfvec else "VectorChord_1.5_FP32"
    pg_data = EXT_ROOT / f"{prefix}_{tag}"
    if not (pg_data / "INGEST_DONE.json").exists():
        print(f"Skipping missing {pg_data}", flush=True)
        return

    q_list, wls, contract, qdoms, qrels = load_corpus_gt(tag)
    wl_specs = get_workload_specs(tag, contract, qdoms)
    col_type = "halfvec(128)[]" if use_rq_halfvec else "vector(128)[]"
    q_pg_strs = [
        "{" + ",".join('"[' + ",".join(f"{x:.4f}" for x in row) + ']"' for row in qmat) + "}"
        for qmat in q_list
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{TP}/pg17_vchord/inst/lib:" + env.get("LD_LIBRARY_PATH", "")
    pg_bin = TP / "pg17_vchord" / "inst" / "bin"
    port = 55442 if use_rq_halfvec else 55441
    # Ensure max_connections >= 64 cleanly for 48 concurrent clients!
    conf_p = pg_data / "postgresql.conf"
    if "max_connections = 128" not in conf_p.read_text():
        with open(conf_p, "a") as f:
            f.write("\nmax_connections = 128\n")

    evict_dir_page_cache(pg_data)
    subprocess.run(["taskset", "-c", "0-47", str(pg_bin / "pg_ctl"), "-D", str(pg_data), "-l", str(pg_data / "pg_bench.log"), "start"], env=env, check=True)
    time.sleep(1.0)
    try:
        # Test both prefilter=off (default) and prefilter=on (VectorChord's built-in bitmap prefilter!) + tuned probes!
        cfg_list = (
            [(16, "off"), (64, "off"), (64, "on"), (256, "on")] if use_rq_halfvec else [(64, "off"), (64, "on")]
        )
        for probes, pref in cfg_list:
            thread_grid = [1, 8, 24, 48] if (probes == 64 and pref == "on") else [1, 48]
            for wl_name, pred_fn in wl_specs:
                if wl_name not in wls:
                    continue
                gt10 = wls[wl_name]["gt10"]
                nq = len(q_list)

                # Capture EXPLAIN plan once on q=0
                plan_str = ""
                with psycopg.connect(f"host=/tmp port={port} user=postgres dbname=postgres", autocommit=True) as c0:
                    c0.execute(f"SET vchordrq.probes = '{probes}'")
                    c0.execute(f"SET vchordrq.prefilter = '{pref}'")
                    sql_pred = pred_fn(0)[0]
                    ex = c0.execute(
                        f"EXPLAIN SELECT id, embs @# %s::{col_type} AS sc FROM docs WHERE {sql_pred} ORDER BY sc LIMIT 10",
                        (q_pg_strs[0],),
                    ).fetchall()
                    plan_str = " | ".join(r[0].strip() for r in ex[:4])

                for nT in thread_grid:
                    total_q = min(nq, 30) if nT == 1 else max(nT * 3, 60)
                    if probes == 256 and nT == 1:
                        total_q = min(nq, 20)
                    rss0, io0 = get_pids_rss_and_io(str(pg_data))
                    lats, recs, all10s, ndcg5s = [], [], [], []

                    def worker(tid):
                        loc_l, loc_r, loc_a, loc_n = [], [], [], []
                        with psycopg.connect(f"host=/tmp port={port} user=postgres dbname=postgres", autocommit=True) as conn:
                            conn.execute(f"SET vchordrq.probes = '{probes}'")
                            conn.execute(f"SET vchordrq.prefilter = '{pref}'")
                            conn.execute("SET max_parallel_workers_per_gather = 0")
                            for job in range(tid, total_q, nT):
                                q = job % nq
                                sql_pred = pred_fn(q)[0]
                                t0 = time.perf_counter()
                                res = conn.execute(
                                    f"SELECT id FROM docs WHERE {sql_pred} ORDER BY embs @# %s::{col_type} LIMIT 10",
                                    (q_pg_strs[q],),
                                ).fetchall()
                                dt_ms = (time.perf_counter() - t0) * 1000.0
                                pids = [int(r[0]) for r in res]
                                rec, a10 = calc_recall_and_all10(pids, gt10[q])
                                loc_l.append(dt_ms); loc_r.append(rec); loc_a.append(a10)
                                if qrels is not None and wl_name == "NoCorr_100.00pct":
                                    loc_n.append(calc_ndcg5(pids, qrels[q]))
                        return loc_l, loc_r, loc_a, loc_n

                    wt0 = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=nT) as ex_pool:
                        for ll, lr, la, ln in ex_pool.map(worker, range(nT)):
                            lats.extend(ll); recs.extend(lr); all10s.extend(la); ndcg5s.extend(ln)
                    ws = time.perf_counter() - wt0
                    rss1, io1 = get_pids_rss_and_io(str(pg_data))
                    lats.sort()
                    row = {
                        "tag": tag, "db": db_label, "config": f"probes={probes},prefilter={pref}",
                        "workload": wl_name, "threads": nT, "n_eval": len(lats),
                        "qps": round(len(lats) / ws, 2),
                        "p50_ms": round(lats[len(lats) // 2], 2),
                        "p99_ms": round(lats[min(len(lats) - 1, int(len(lats) * 0.99))], 2),
                        "recall10": round(float(np.mean(recs)), 4),
                        "all10": round(float(np.mean(all10s)), 4),
                        "ndcg5": round(float(np.mean(ndcg5s)), 4) if ndcg5s else None,
                        "proc_mb_q": round(max(0, io1 - io0) / 1048576.0 / max(1, len(lats)), 3),
                        "rss_gb": round(max(rss0, rss1) / (1024 ** 3), 3),
                        "plan": plan_str if nT == 1 else "",
                    }
                    rows_out.append(row)
                    print(f"  [{tag} {db_label} p={probes},pf={pref} {wl_name} T={nT:2d}] QPS={row['qps']:7.1f} P50={row['p50_ms']:7.1f}ms P99={row['p99_ms']:7.1f}ms R@10={row['recall10']:.3f} All10={row['all10']:.2f} IO={row['proc_mb_q']:.2f}MB/q RSS={row['rss_gb']:.2f}G", flush=True)
    finally:
        subprocess.run([str(pg_bin / "pg_ctl"), "-D", str(pg_data), "stop"], env=env, check=False)
        time.sleep(1.0)


def bench_qdrant(tag: str, rows_out: list):
    from qdrant_client import QdrantClient, models

    q_data = EXT_ROOT / f"qdrant_{tag}"
    if not (q_data / "INGEST_DONE.json").exists():
        return
    q_list, wls, contract, qdoms, qrels = load_corpus_gt(tag)
    wl_specs = get_workload_specs(tag, contract, qdoms)
    q_py_lists = [qmat.tolist() for qmat in q_list]

    evict_dir_page_cache(q_data)
    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(q_data / "storage")
    env["QDRANT__SERVICE__HTTP_PORT"] = "56343"
    env["QDRANT__SERVICE__GRPC_PORT"] = "56344"
    log_fh = open(q_data / "q_bench.log", "w")
    proc = subprocess.Popen(["taskset", "-c", "0-47", str(TP / "qdrant" / "qdrant")], env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        c0 = QdrantClient(host="127.0.0.1", port=56343, grpc_port=56344, prefer_grpc=True, timeout=120)
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

        for ef in (64, 128, 256):
            thread_grid = [1, 8, 24, 48] if (ef == 128) else [1, 48]
            for wl_name, pred_fn in wl_specs:
                if wl_name not in wls:
                    continue
                gt10 = wls[wl_name]["gt10"]
                nq = len(q_list)
                for nT in thread_grid:
                    total_q = min(nq, 40) if nT == 1 else max(nT * 4, 80)
                    rss0, io0 = get_pids_rss_and_io("third_party/qdrant/qdrant")
                    lats, recs, all10s, ndcg5s = [], [], [], []

                    def worker(tid):
                        loc_l, loc_r, loc_a, loc_n = [], [], [], []
                        cli = QdrantClient(host="127.0.0.1", port=56343, grpc_port=56344, prefer_grpc=True, timeout=120)
                        sp = models.SearchParams(hnsw_ef=ef, quantization=models.QuantizationSearchParams(rescore=True))
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
                        "tag": tag, "db": "Qdrant_1.20.2_INT8_OnDisk", "config": f"ef={ef}",
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
                    print(f"  [{tag} Qdrant ef={ef:3d} {wl_name} T={nT:2d}] QPS={row['qps']:7.1f} P50={row['p50_ms']:7.1f}ms P99={row['p99_ms']:7.1f}ms R@10={row['recall10']:.3f} All10={row['all10']:.2f} IO={row['proc_mb_q']:.2f}MB/q RSS={row['rss_gb']:.2f}G", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log_fh.close()
        time.sleep(1.0)


def bench_weaviate(tag: str, use_rq: bool, rows_out: list):
    import weaviate
    from weaviate.classes.config import Reconfigure
    from weaviate.classes.query import Filter

    prefix = "weaviate_rq" if use_rq else "weaviate"
    db_label = "Weaviate_1.41_MUVERA_RQ" if use_rq else "Weaviate_1.41_MUVERA"
    w_data = EXT_ROOT / f"{prefix}_{tag}"
    if not (w_data / "INGEST_DONE.json").exists():
        return
    q_list, wls, contract, qdoms, qrels = load_corpus_gt(tag)
    wl_specs = get_workload_specs(tag, contract, qdoms)
    q_py_lists = [qmat.tolist() for qmat in q_list]

    evict_dir_page_cache(w_data)
    http_port = 58090 if use_rq else 58089
    grpc_port = 50060 if use_rq else 50059
    env = os.environ.copy()
    env["PERSISTENCE_DATA_PATH"] = str(w_data / "data")
    env["AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED"] = "true"
    env["DEFAULT_VECTORIZER_MODULE"] = "none"
    env["CLUSTER_HOSTNAME"] = "node1"
    env["GRPC_PORT"] = str(grpc_port)
    log_fh = open(w_data / "w_bench.log", "w")
    proc = subprocess.Popen(
        ["taskset", "-c", "0-47", str(TP / "weaviate" / "weaviate"), "--host", "127.0.0.1", "--port", str(http_port), "--scheme", "http"],
        env=env, stdout=log_fh, stderr=subprocess.STDOUT,
    )
    try:
        c0 = None
        for _ in range(40):
            try:
                c0 = weaviate.connect_to_local(host="127.0.0.1", port=http_port, grpc_port=grpc_port)
                if c0.is_ready():
                    break
            except Exception:
                time.sleep(0.5)

        def make_wfilter(spec_tuple):
            if spec_tuple is None:
                return None
            k, op, val = spec_tuple
            if op == "gte":
                return Filter.by_property(k).greater_or_equal(val)
            return Filter.by_property(k).equal(val)

        ef_list = [-1, 256] if use_rq else [-1]
        for ef in ef_list:
            try:
                coll0 = c0.collections.get("Docs")
                coll0.config.update(
                    vector_config=Reconfigure.Vectors.update(
                        name="mv", vector_index_config=Reconfigure.VectorIndex.hnsw(ef=ef)
                    )
                )
            except Exception as e:
                print(f"  [Weaviate ef={ef} reconfigure note: {e}]", flush=True)

            thread_grid = [1, 8, 24, 48] if (ef == -1) else [1, 48]
            for wl_name, pred_fn in wl_specs:
                if wl_name not in wls:
                    continue
                gt10 = wls[wl_name]["gt10"]
                nq = len(q_list)
                for nT in thread_grid:
                    total_q = min(nq, 40) if nT == 1 else max(nT * 4, 80)
                    rss0, io0 = get_pids_rss_and_io("third_party/weaviate/weaviate")
                    lats, recs, all10s, ndcg5s = [], [], [], []

                    def worker(tid):
                        loc_l, loc_r, loc_a, loc_n = [], [], [], []
                        cli = weaviate.connect_to_local(host="127.0.0.1", port=http_port, grpc_port=grpc_port)
                        coll = cli.collections.get("Docs")
                        for job in range(tid, total_q, nT):
                            q = job % nq
                            flt = make_wfilter(pred_fn(q)[2])
                            t0 = time.perf_counter()
                            res = coll.query.near_vector(near_vector=q_py_lists[q], target_vector="mv", filters=flt, limit=10, return_properties=["doc_id"])
                            dt_ms = (time.perf_counter() - t0) * 1000.0
                            pids = [int(o.properties["doc_id"]) for o in res.objects]
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
                    rss1, io1 = get_pids_rss_and_io("third_party/weaviate/weaviate")
                    lats.sort()
                    row = {
                        "tag": tag, "db": db_label, "config": f"ef={ef}",
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
                    print(f"  [{tag} {db_label} ef={ef:3d} {wl_name} T={nT:2d}] QPS={row['qps']:7.1f} P50={row['p50_ms']:7.1f}ms P99={row['p99_ms']:7.1f}ms R@10={row['recall10']:.3f} All10={row['all10']:.2f} IO={row['proc_mb_q']:.2f}MB/q RSS={row['rss_gb']:.2f}G", flush=True)
        c0.close()
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log_fh.close()
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "external_dbs_s33.json"))
    args = ap.parse_args()
    rows = []
    for tag in ("vidore_v3_am", "lotte_200k"):
        print(f"\n================ BENCHMARKING EXTERNAL DBs ON {tag} ================", flush=True)
        bench_vectorchord(tag, use_rq_halfvec=True, rows_out=rows)
        bench_vectorchord(tag, use_rq_halfvec=False, rows_out=rows)
        bench_qdrant(tag, rows_out=rows)
        bench_weaviate(tag, use_rq=True, rows_out=rows)
        bench_weaviate(tag, use_rq=False, rows_out=rows)
        # Checkpoint after each corpus
        with open(args.out, "w") as f:
            json.dump({"rows": rows}, f, indent=2)
    print(f"\n=== ALL EXTERNAL DB BENCHMARKS SAVED TO {args.out} ({len(rows)} rows) ===", flush=True)


if __name__ == "__main__":
    main()
