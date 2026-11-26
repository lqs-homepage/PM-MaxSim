#!/usr/bin/env python3
"""
Stage 3.3 Hard-Exclusive 1:1 Apples-to-Apples External Production Vector DB Benchmark (Remediated RES-P09-02/03/07):
  Runs INSIDE Slurm 48-core hard-exclusive Socket-0 allocation (SNC-2 NUMA nodes 0,1:
  numactl --cpunodebind=0,1 --membind=0,1 taskset -c 0-47, swap.max=0).

  Fixes RES-P09-02 completely:
    - Creates persistent per-worker clients (psycopg.connect + SET params, QdrantClient, weaviate.connect_to_local)
      and runs 2 warmup queries per worker strictly BEFORE threading.Barrier(nT) start gate!
    - Runs steady-state timed window (>= 6.0 s at T>1 or >= 30 queries/worker on fast configs; >= 2 q/worker floor on slow)
      using lock-free itertools.count() queue, synchronizing all nT workers on an end threading.Barrier(nT) BEFORE closing clients!
    - Logs Little's Law coherence ratio `littles_law = p50_ms * qps / (nT * 1000.0)` on EVERY row (target >= 0.70 on steady-state runs).

  Fixes RES-P09-03 completely:
    - Samples BOTH idle server PSS (before concurrent client load) AND active in-flight server PSS/VmRSS/VmHWM via
      /proc/<pid>/smaps_rollup & /proc/<pid>/status WHILE all nT persistent clients remain connected!
"""
import argparse
import concurrent.futures
import itertools
import json
import math
import os
import struct
import subprocess
import threading
import time
from pathlib import Path
import numpy as np

P09 = Path(os.environ.get("P09_ROOT", "./workspace"))
TP = P09 / "third_party"
EXT_ROOT = P09 / "stage3" / "ext_dbs"
OUT_DIR = P09 / "stage3" / "results_s33"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMA_PREFIX = ["numactl", "--cpunodebind=0,1", "--membind=0,1", "taskset", "-c", "0-47"]


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


def get_pids_mem_and_io(match_substr: str, match_pg_uid: bool = False):
    """Sum smaps_rollup PSS, VmRSS, VmHWM (bytes), and /proc/<pid>/io read_bytes across matching server processes."""
    pss_b, rss_b, hwm_b, io_b, n_procs = 0, 0, 0, 0, 0
    my_uid = os.getuid()
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        try:
            if match_pg_uid:
                comm = Path(f"/proc/{pid_str}/comm").read_text().strip()
                if comm != "postgres" or os.stat(f"/proc/{pid_str}").st_uid != my_uid:
                    continue
            else:
                cmd = Path(f"/proc/{pid_str}/cmdline").read_bytes().decode("utf-8", errors="ignore")
                if match_substr not in cmd:
                    continue
            n_procs += 1
            for line in Path(f"/proc/{pid_str}/smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    pss_b += int(line.split()[1]) * 1024
                elif line.startswith("Rss:"):
                    rss_b += int(line.split()[1]) * 1024
            for line in Path(f"/proc/{pid_str}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    hwm_b += int(line.split()[1]) * 1024
                    break
            for line in Path(f"/proc/{pid_str}/io").read_text().splitlines():
                if line.startswith("read_bytes:"):
                    io_b += int(line.split()[1])
                    break
        except OSError:
            continue
    return n_procs, pss_b, rss_b, hwm_b, io_b


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


def run_barrier_steady_state(
    nT: int,
    nq: int,
    gt10: np.ndarray,
    qrels,
    wl_name: str,
    make_client_fn,
    run_query_fn,
    close_client_fn,
    mem_match_substr: str,
    match_pg_uid: bool = False,
    target_wall_s: float = 6.0,
    n_warmup: int = 2,
    fixed_total_q: int = 0,
):
    """
    Exact barrier-synchronized steady-state multi-client benchmark engine:
      1. Each worker opens persistent client + runs `n_warmup` warmup queries BEFORE start_bar.
      2. start_bar action snapshots server IO/memory and starts wall clock `wt0` cleanly.
      3. All nT workers pull jobs from C-atomic itertools.count() until steady-state deadline/count.
      4. end_bar action stops wall clock `wt1` and snapshots active server IO + PSS/RSS/VmHWM
         WHILE ALL nT CLIENTS REMAIN OPEN AND ACTIVE.
      5. Only AFTER end_bar do workers close clients!
    """
    state = {"wt0": 0.0, "wt1": 0.0, "io0": 0, "io1": 0, "pss_b": 0, "rss_b": 0, "hwm_b": 0, "n_procs": 0}
    counter = itertools.count(0)
    min_total_q = fixed_total_q if fixed_total_q > 0 else (min(nq, 40) if nT == 1 else nT * 2)
    max_total_q = fixed_total_q if fixed_total_q > 0 else (min(nq, 40) if nT == 1 else nT * 800)

    def on_start():
        _, p0, r0, h0, io0 = get_pids_mem_and_io(mem_match_substr, match_pg_uid=match_pg_uid)
        state["io0"] = io0
        state["pss_b"] = p0
        state["rss_b"] = r0
        state["hwm_b"] = h0
        state["wt0"] = time.perf_counter()

    def on_end():
        state["wt1"] = time.perf_counter()
        np1, p1, r1, h1, io1 = get_pids_mem_and_io(mem_match_substr, match_pg_uid=match_pg_uid)
        state["io1"] = io1
        state["n_procs"] = np1
        state["pss_b"] = max(state["pss_b"], p1)
        state["rss_b"] = max(state["rss_b"], r1)
        state["hwm_b"] = max(state["hwm_b"], h1)

    start_bar = threading.Barrier(nT, action=on_start)
    end_bar = threading.Barrier(nT, action=on_end)

    def worker(tid):
        loc_l, loc_r, loc_a, loc_n = [], [], [], []
        cli = make_client_fn(tid)
        try:
            # Warmup strictly OUTSIDE timed window!
            for w in range(n_warmup):
                run_query_fn(cli, (tid + w) % nq)
            start_bar.wait()
            deadline = state["wt0"] + target_wall_s
            while True:
                if fixed_total_q > 0:
                    job = next(counter)
                    if job >= fixed_total_q:
                        break
                else:
                    if nT == 1:
                        job = next(counter)
                        if job >= min_total_q:
                            break
                    else:
                        now = time.perf_counter()
                        # Check stop condition before pulling next job once min_total_q reached
                        if now >= deadline and len(loc_l) >= 2:
                            break
                        job = next(counter)
                        if job >= max_total_q:
                            break
                q = job % nq
                t0 = time.perf_counter()
                pids = run_query_fn(cli, q)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                rec, a10 = calc_recall_and_all10(pids, gt10[q])
                loc_l.append(dt_ms)
                loc_r.append(rec)
                loc_a.append(a10)
                if qrels is not None and wl_name == "NoCorr_100.00pct":
                    loc_n.append(calc_ndcg5(pids, qrels[q]))
            end_bar.wait()
        finally:
            close_client_fn(cli)
        return loc_l, loc_r, loc_a, loc_n

    lats, recs, all10s, ndcg5s = [], [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=nT) as ex_pool:
        for ll, lr, la, ln in ex_pool.map(worker, range(nT)):
            lats.extend(ll)
            recs.extend(lr)
            all10s.extend(la)
            ndcg5s.extend(ln)

    ws = max(state["wt1"] - state["wt0"], 1e-6)
    lats.sort()
    n_eval = len(lats)
    qps = n_eval / ws
    p50 = lats[n_eval // 2]
    p99 = lats[min(n_eval - 1, int(n_eval * 0.99))]
    ll_ratio = (p50 * qps) / (nT * 1000.0)
    return {
        "n_eval": n_eval,
        "wall_s": round(ws, 3),
        "qps": round(qps, 2),
        "p50_ms": round(p50, 2),
        "p99_ms": round(p99, 2),
        "littles_law": round(ll_ratio, 3),
        "recall10": round(float(np.mean(recs)), 4),
        "all10": round(float(np.mean(all10s)), 4),
        "ndcg5": round(float(np.mean(ndcg5s)), 4) if ndcg5s else None,
        "proc_mb_q": round(max(0, state["io1"] - state["io0"]) / 1048576.0 / max(1, n_eval), 3),
        "active_pss_gb": round(state["pss_b"] / (1024 ** 3), 3),
        "rss_gb": round(state["rss_b"] / (1024 ** 3), 3),
        "active_hwm_gb": round(state["hwm_b"] / (1024 ** 3), 3),
        "n_server_procs": state["n_procs"],
    }


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
    conf_p = pg_data / "postgresql.conf"
    if "max_connections = 128" not in conf_p.read_text():
        with open(conf_p, "a") as f:
            f.write("\nmax_connections = 128\n")

    evict_dir_page_cache(pg_data)
    subprocess.run(
        NUMA_PREFIX + [str(pg_bin / "pg_ctl"), "-D", str(pg_data), "-l", str(pg_data / "pg_bench.log"), "start"],
        env=env, check=True,
    )
    time.sleep(1.0)
    try:
        _, idle_pss_b, idle_rss_b, _, _ = get_pids_mem_and_io(str(pg_data), match_pg_uid=True)
        idle_pss_gb = round(idle_pss_b / (1024 ** 3), 3)

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
                    def make_cli(_tid):
                        conn = psycopg.connect(f"host=/tmp port={port} user=postgres dbname=postgres", autocommit=True)
                        conn.execute(f"SET vchordrq.probes = '{probes}'")
                        conn.execute(f"SET vchordrq.prefilter = '{pref}'")
                        conn.execute("SET max_parallel_workers_per_gather = 0")
                        return conn

                    def run_q(conn, q):
                        sql_pred = pred_fn(q)[0]
                        res = conn.execute(
                            f"SELECT id FROM docs WHERE {sql_pred} ORDER BY embs @# %s::{col_type} LIMIT 10",
                            (q_pg_strs[q],),
                        ).fetchall()
                        return [int(r[0]) for r in res]

                    def close_cli(conn):
                        conn.close()

                    fixed_q = (min(nq, 20) if probes == 256 else min(nq, 30)) if nT == 1 else 0
                    wall_s = 8.0 if (probes == 256 and tag == "lotte_200k") else 6.0
                    m = run_barrier_steady_state(
                        nT=nT, nq=nq, gt10=gt10, qrels=qrels, wl_name=wl_name,
                        make_client_fn=make_cli, run_query_fn=run_q, close_client_fn=close_cli,
                        mem_match_substr=str(pg_data), match_pg_uid=True,
                        target_wall_s=wall_s, n_warmup=2, fixed_total_q=fixed_q,
                    )
                    row = {
                        "tag": tag, "db": db_label, "config": f"probes={probes},prefilter={pref}",
                        "workload": wl_name, "threads": nT,
                        "idle_pss_gb": idle_pss_gb, "plan": plan_str if nT == 1 else "",
                        **m,
                    }
                    rows_out.append(row)
                    print(
                        f"  [{tag} {db_label} p={probes},pf={pref} {wl_name} T={nT:2d}] "
                        f"QPS={row['qps']:7.1f} P50={row['p50_ms']:7.1f}ms P99={row['p99_ms']:7.1f}ms "
                        f"LL={row['littles_law']:.2f} n={row['n_eval']:4d} R@10={row['recall10']:.3f} "
                        f"All10={row['all10']:.2f} IO={row['proc_mb_q']:.2f}MB/q PSS={row['active_pss_gb']:.2f}G",
                        flush=True,
                    )
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
    proc = subprocess.Popen(
        NUMA_PREFIX + [str(TP / "qdrant" / "qdrant")],
        env=env, stdout=log_fh, stderr=subprocess.STDOUT,
    )
    try:
        c0 = QdrantClient(host="127.0.0.1", port=56343, grpc_port=56344, prefer_grpc=True, timeout=120)
        for _ in range(60):
            try:
                c0.get_collections()
                break
            except Exception:
                time.sleep(0.5)
        c0.close()
        _, idle_pss_b, _, _, _ = get_pids_mem_and_io("third_party/qdrant/qdrant")
        idle_pss_gb = round(idle_pss_b / (1024 ** 3), 3)

        def make_filter(spec_tuple):
            if spec_tuple is None:
                return None
            k, val = spec_tuple
            if k == "n_toks_ge":
                return models.Filter(must=[models.FieldCondition(key="n_toks", range=models.Range(gte=val))])
            return models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=val))])

        for ef in (64, 128, 256):
            thread_grid = [1, 8, 24, 48] if (ef == 128) else [1, 48]
            sp = models.SearchParams(hnsw_ef=ef, quantization=models.QuantizationSearchParams(rescore=True))
            for wl_name, pred_fn in wl_specs:
                if wl_name not in wls:
                    continue
                gt10 = wls[wl_name]["gt10"]
                nq = len(q_list)
                for nT in thread_grid:
                    def make_cli(_tid):
                        return QdrantClient(host="127.0.0.1", port=56343, grpc_port=56344, prefer_grpc=True, timeout=120)

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
                        "tag": tag, "db": "Qdrant_1.20.2_INT8_OnDisk", "config": f"ef={ef}",
                        "workload": wl_name, "threads": nT, "idle_pss_gb": idle_pss_gb,
                        **m,
                    }
                    rows_out.append(row)
                    print(
                        f"  [{tag} Qdrant-INT8 ef={ef:3d} {wl_name} T={nT:2d}] "
                        f"QPS={row['qps']:7.1f} P50={row['p50_ms']:7.1f}ms P99={row['p99_ms']:7.1f}ms "
                        f"LL={row['littles_law']:.2f} n={row['n_eval']:4d} R@10={row['recall10']:.3f} "
                        f"All10={row['all10']:.2f} IO={row['proc_mb_q']:.2f}MB/q PSS={row['active_pss_gb']:.2f}G",
                        flush=True,
                    )
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
        NUMA_PREFIX + [str(TP / "weaviate" / "weaviate"), "--host", "127.0.0.1", "--port", str(http_port), "--scheme", "http"],
        env=env, stdout=log_fh, stderr=subprocess.STDOUT,
    )
    try:
        c0 = None
        for _ in range(60):
            try:
                c0 = weaviate.connect_to_local(host="127.0.0.1", port=http_port, grpc_port=grpc_port)
                if c0.is_ready():
                    break
            except Exception:
                time.sleep(0.5)
        _, idle_pss_b, _, _, _ = get_pids_mem_and_io("third_party/weaviate/weaviate")
        idle_pss_gb = round(idle_pss_b / (1024 ** 3), 3)

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
                    def make_cli(_tid):
                        cli = weaviate.connect_to_local(host="127.0.0.1", port=http_port, grpc_port=grpc_port)
                        return (cli, cli.collections.get("Docs"))

                    def run_q(pair, q):
                        _cli, coll = pair
                        flt = make_wfilter(pred_fn(q)[2])
                        res = coll.query.near_vector(near_vector=q_py_lists[q], target_vector="mv", filters=flt, limit=10, return_properties=["doc_id"])
                        return [int(o.properties["doc_id"]) for o in res.objects]

                    def close_cli(pair):
                        pair[0].close()

                    m = run_barrier_steady_state(
                        nT=nT, nq=nq, gt10=gt10, qrels=qrels, wl_name=wl_name,
                        make_client_fn=make_cli, run_query_fn=run_q, close_client_fn=close_cli,
                        mem_match_substr="third_party/weaviate/weaviate",
                        target_wall_s=6.0, n_warmup=2, fixed_total_q=(min(nq, 40) if nT == 1 else 0),
                    )
                    row = {
                        "tag": tag, "db": db_label, "config": f"ef={ef}",
                        "workload": wl_name, "threads": nT, "idle_pss_gb": idle_pss_gb,
                        **m,
                    }
                    rows_out.append(row)
                    print(
                        f"  [{tag} {db_label} ef={ef:3d} {wl_name} T={nT:2d}] "
                        f"QPS={row['qps']:7.1f} P50={row['p50_ms']:7.1f}ms P99={row['p99_ms']:7.1f}ms "
                        f"LL={row['littles_law']:.2f} n={row['n_eval']:4d} R@10={row['recall10']:.3f} "
                        f"All10={row['all10']:.2f} IO={row['proc_mb_q']:.2f}MB/q PSS={row['active_pss_gb']:.2f}G",
                        flush=True,
                    )
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
        print(f"\n================ BARRIER-SYNCED EXTERNAL DBs ON {tag} ================", flush=True)
        bench_vectorchord(tag, use_rq_halfvec=True, rows_out=rows)
        bench_vectorchord(tag, use_rq_halfvec=False, rows_out=rows)
        bench_qdrant(tag, rows_out=rows)
        bench_weaviate(tag, use_rq=True, rows_out=rows)
        bench_weaviate(tag, use_rq=False, rows_out=rows)
        with open(args.out, "w") as f:
            json.dump({"rows": rows}, f, indent=2)
    print(f"\n=== ALL BARRIER-SYNCED EXTERNAL DB BENCHMARKS SAVED TO {args.out} ({len(rows)} rows) ===", flush=True)


if __name__ == "__main__":
    main()
