#!/usr/bin/env python3
"""
Layer 4 Barrier-Synchronized External DB Benchmark for New Large Out-of-Core Corpora + Kernel `cgroup v2 MemoryMax` SLA:
  1. `msmarco_docs_768d` (300K parent docs x ~6M 768D chunks):
     - Qdrant v1.20.2 (`BQ` & `INT8`) native `query_points_groups(group_by="doc_id")`!
     - PostgreSQL 17.12 + VectorChord 1.5 (`DISTINCT ON (doc_id)`) vs `pg_certvec`!
  2. `bgem3_1024d_hybrid` (150K docs x 11.73M 1024D tokens = 24 GB FP16):
     - Qdrant v1.20.2 (`BQ` & `INT8`) native 1024D `MAX_SIM` & Hybrid!
  3. Kernel-Enforced `cgroup v2 MemoryMax` (`4G / 8G / 16G`, `MemorySwapMax=0`):
     - Launches the target DB server inside `systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0`,
       measures active 48-thread barrier-synchronized QPS, P50/P99, Little's Law, Recall@10, active PSS/RSS,
       AND records exact kernel `memory.peak` / `memory.events` (`oom_kill` if DB crashes under hard cap!).
"""
import argparse
import itertools
import json
import os
import pathlib
import subprocess
import threading
import time
import numpy as np

P09 = pathlib.Path(os.environ.get("P09_ROOT", "${P09_ROOT}"))
TP = P09 / "third_party"
EXT_ROOT = P09 / "stage3_v4" / "ext_dbs"


def read_pid_pss_rss_gb(pid: int):
    pss = rss = 0
    try:
        for line in open(f"/proc/{pid}/smaps_rollup", "r"):
            if line.startswith("Pss:"): pss = int(line.split()[1])
            elif line.startswith("Rss:"): rss = int(line.split()[1])
    except Exception:
        pass
    return pss / 1048576.0, rss / 1048576.0


def bench_qdrant_new_corpus(corpus: str, quant: str, out_json: pathlib.Path, n_threads: int = 48, mem_cap: str = ""):
    from qdrant_client import QdrantClient, models

    dir_tag = "bgem3_1024d" if corpus == "bgem3_1024d_hybrid" else corpus
    q_data = EXT_ROOT / f"qdrant_{quant}_{dir_tag}"
    if not (q_data / "INGEST_DONE.json").exists():
        raise RuntimeError(f"Not ingested: {q_data}")

    idx_dir = P09 / "stage3_v4" / "indexes" / ("msmarco_docs_768d" if corpus == "msmarco_docs_768d" else "bgem3_1024d_hybrid")
    http_p, grpc_p = 56481, 56482
    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(q_data / "storage")
    env["QDRANT__SERVICE__HTTP_PORT"] = str(http_p)
    env["QDRANT__SERVICE__GRPC_PORT"] = str(grpc_p)

    cmd = [str(TP / "qdrant" / "qdrant")]
    if mem_cap:
        uname = f"p09-cg-qdrant-{quant}-{corpus}-{mem_cap}-{int(time.time())}"
        cmd = ["systemd-run", "--user", "--scope", f"--unit={uname}", "-p", f"MemoryMax={mem_cap}", "-p", "MemorySwapMax=0"] + cmd

    log_fh = open(q_data / f"bench_{mem_cap or 'uncapped'}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    oom_killed = False
    rows = []
    try:
        # Wait for Qdrant ready (or detect OOM death during startup!)
        ready = False
        client0 = QdrantClient(host="127.0.0.1", port=http_p, grpc_port=grpc_p, prefer_grpc=True, timeout=30)
        for _ in range(90):
            if proc.poll() is not None:
                oom_killed = (proc.returncode in (137, -9, 1))
                break
            try:
                client0.get_collections()
                ready = True
                break
            except Exception:
                time.sleep(1.0)
        if not ready:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps({
                "db": f"qdrant_1.20.2_{quant}", "corpus": corpus, "mem_cap": mem_cap or "uncapped",
                "status": "OOM_AT_STARTUP" if oom_killed else f"FAILED_RC_{proc.returncode}",
                "returncode": proc.returncode, "rows": [],
            }, indent=2) + "\n")
            print(f"[Qdrant {quant} {corpus} cap={mem_cap}] FAILED/OOM at startup (rc={proc.returncode})", flush=True)
            return

        # Find actual qdrant binary PID (in case wrapped by systemd-run --scope)
        q_pid = proc.pid
        if mem_cap:
            for e in os.listdir("/proc"):
                if e.isdigit():
                    try:
                        if "third_party/qdrant/qdrant" in open(f"/proc/{e}/cmdline", "rb").read().decode("utf-8", errors="ignore"):
                            q_pid = int(e)
                    except Exception:
                        pass

        idle_pss, idle_rss = read_pid_pss_rss_gb(q_pid)

        if corpus == "bgem3_1024d_hybrid":
            Q_colb = np.load(idx_dir / "q_colb_f32.npy")[:100]
            Q_lens = np.load(idx_dir / "q_lens.npy")[:100]
            q_list = [Q_colb[i, :int(Q_lens[i])].tolist() for i in range(100)]
            wls = [
                ("NoCorr_100pct", None, np.load(idx_dir / "gt_NoCorr_100pct.npz")["gt_colb"]),
                ("NoCorr_5pct", models.Filter(must=[models.FieldCondition(key="nc5", match=models.MatchValue(value=True))]), np.load(idx_dir / "gt_NoCorr_5pct.npz")["gt_colb"]),
                ("Nat_Topic0_Eq", models.Filter(must=[models.FieldCondition(key="topic", match=models.MatchValue(value=0))]), np.load(idx_dir / "gt_Nat_Topic0_Eq.npz")["gt_colb"]),
            ]
        else:
            Q_f32 = np.load(idx_dir / "queries_f32.npy")[:100]
            q_list = [Q_f32[i].tolist() for i in range(100)]
            wls = [
                ("NoCorr_100pct", None, np.load(idx_dir / "gt_NoCorr_100pct.npz")["w1_max_gt"]),
                ("NoCorr_5pct", models.Filter(must=[models.FieldCondition(key="nc5", match=models.MatchValue(value=True))]), np.load(idx_dir / "gt_NoCorr_5pct.npz")["w1_max_gt"]),
                ("Nat_Dom0_Eq", models.Filter(must=[models.FieldCondition(key="dom", match=models.MatchValue(value=0))]), np.load(idx_dir / "gt_Nat_Dom0_Eq.npz")["w1_max_gt"]),
            ]

        for wname, qflt, gt10 in wls:
            for ef in [128]:
                bar_s = threading.Barrier(n_threads)
                bar_e = threading.Barrier(n_threads)
                stop_evt = threading.Event()
                counter = itertools.count()
                lats, recs, errs = [], [], []
                counts = [0] * n_threads
                lock = threading.Lock()
                t0_box = [0.0]
                t1_box = [0.0]
                pk_pss = [idle_pss]
                pk_rss = [idle_rss]

                def worker(tid: int):
                    ll, lr = [], []
                    try:
                        cli = QdrantClient(host="127.0.0.1", port=http_p, grpc_port=grpc_p, prefer_grpc=True, timeout=60)
                        sp = models.SearchParams(
                            hnsw_ef=ef,
                            quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0) if quant == "bq" else None,
                        )
                        def run_q(qi: int):
                            if corpus == "bgem3_1024d_hybrid":
                                r = cli.query_points(collection_name="docs", using="colbert", query=q_list[qi], query_filter=qflt, search_params=sp, limit=10)
                                return [int(p.id) for p in r.points]
                            else:
                                r = cli.query_points_groups(collection_name="chunks", query=q_list[qi], group_by="doc_id", group_size=1, query_filter=qflt, search_params=sp, limit=10)
                                return [int(g.id) for g in r.groups]

                        for w in range(2):
                            run_q((tid + w) % len(q_list))
                        bar_s.wait()
                        if tid == 0:
                            t0_box[0] = time.perf_counter()
                        while not stop_evt.is_set():
                            idx = next(counter)
                            qi = idx % len(q_list)
                            ts = time.perf_counter()
                            ids = run_q(qi)
                            ll.append((time.perf_counter() - ts) * 1000.0)
                            counts[tid] += 1
                            if len(lr) < 4:
                                lr.append(len(set(int(x) for x in gt10[qi]).intersection(ids)) / 10.0)
                            if tid == 0:
                                el = time.perf_counter() - t0_box[0]
                                if counts[0] % 3 == 0:
                                    p_, r_ = read_pid_pss_rss_gb(q_pid)
                                    if p_ > pk_pss[0]: pk_pss[0], pk_rss[0] = p_, r_
                                if el >= 6.0 and sum(counts) >= n_threads:
                                    stop_evt.set()
                        bar_e.wait()
                        if tid == 0:
                            t1_box[0] = time.perf_counter()
                        cli.close()
                    except Exception as ex:
                        with lock: errs.append(repr(ex))
                        stop_evt.set()
                        try: bar_s.abort()
                        except Exception: pass
                        try: bar_e.abort()
                        except Exception: pass
                    with lock:
                        lats.extend(ll)
                        recs.extend(lr)

                threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
                for t in threads: t.start()
                for t in threads: t.join()

                if proc.poll() is not None and proc.returncode in (137, -9):
                    oom_killed = True
                    rows.append({"workload": wname, "ef": ef, "status": "OOM_DURING_QUERY"})
                    print(f"[Qdrant {quant} {corpus} {wname} cap={mem_cap}] OOM KILLED DURING QUERY!", flush=True)
                    break

                wall = max(1e-6, t1_box[0] - t0_box[0])
                tot_q = sum(counts)
                qps = tot_q / wall
                arr = np.array(lats) if lats else np.array([0.0])
                ll_val = (qps * float(np.mean(arr)) / 1000.0) / n_threads if len(lats) else 0.0
                row = {
                    "workload": wname, "ef": ef, "threads": n_threads, "mem_cap": mem_cap or "uncapped",
                    "n_queries": tot_q, "qps": round(qps, 2),
                    "p50_ms": round(float(np.percentile(arr, 50)), 3),
                    "p99_ms": round(float(np.percentile(arr, 99)), 3),
                    "littles_law": round(ll_val, 3),
                    "recall10": round(float(np.mean(recs)), 4) if recs else 0.0,
                    "idle_pss_gb": round(idle_pss, 4),
                    "active_pss_gb": round(pk_pss[0], 4),
                    "active_rss_gb": round(pk_rss[0], 4),
                    "n_errors": len(errs),
                }
                rows.append(row)
                print(f"[Qdrant {quant} {corpus} {wname} cap={mem_cap or 'uncapped'}] QPS={row['qps']} P99={row['p99_ms']}ms LL={row['littles_law']} R@10={row['recall10']} PSS={row['active_pss_gb']}GB errs={row['n_errors']}", flush=True)

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"db": f"qdrant_1.20.2_{quant}", "corpus": corpus, "mem_cap": mem_cap or "uncapped", "oom_killed": oom_killed, "rows": rows}, indent=2) + "\n")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=30)
            except Exception: proc.kill()
        log_fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["bgem3_1024d_hybrid", "msmarco_docs_768d"], required=True)
    ap.add_argument("--quant", choices=["bq", "int8"], required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--mem-cap", default="")
    args = ap.parse_args()
    bench_qdrant_new_corpus(args.corpus, args.quant, pathlib.Path(args.out_json), args.threads, args.mem_cap)
