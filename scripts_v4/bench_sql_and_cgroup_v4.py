#!/usr/bin/env python3
"""
Layer 4 Unified PostgreSQL 17.12 Same-Server Head-to-Head SQL Benchmark Harness:
  Boots ONE PostgreSQL 17.12 server with BOTH `vchord.so` (VectorChord 1.5.0) AND `pg_certvec.so`
  co-preloaded (`shared_preload_libraries = 'vchord.so,pg_certvec.so'`, pointer-free PG Shmem S3-FIFO Cache),
  and runs 48-client barrier-synchronized head-to-head real SQL across THREE corpora:
    1. `vidore_v3_am`      (16,868 multimodal visual pages, ColQwen2.5 AdaMerge 128D)
    2. `lotte_200k`        (200,000 ColBERTv2 passages, 128D)
    3. `msmarco_docs_768d` (300,000 parent long docs x 3.61M 768D chunks — W1 Parent-Entity SQL `GROUP BY doc_id`!)
  Records: QPS, P50/P99, Little's Law coherence, exact Recall@10, Q-Thresh 0-FP precision,
  exact full `postmaster.pid` process-tree PSS/RSS via `/proc/<pid>/smaps_rollup`, and PG `EXPLAIN` plans.
  Strictly operates ONLY on `stage3_v4/ext_dbs/` (zero writes to frozen `stage3/ext_dbs/`).
"""
import argparse
import itertools
import json
import os
import pathlib
import struct
import subprocess
import threading
import time
import numpy as np

P09 = pathlib.Path(os.environ.get("P09_ROOT", "${P09_ROOT}"))
TP = P09 / "third_party"
NUMA_PREFIX = ["numactl", "--cpunodebind=0,1", "--membind=0,1", "taskset", "-c", "0-47"]


def evict_dir_page_cache(dir_path: pathlib.Path):
    for root, _, files in os.walk(dir_path):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                fd = os.open(p, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass


def get_process_tree_pids(root_pid: int):
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            st = open(f"/proc/{entry}/stat", "r").read()
            rp = st.rfind(")")
            ppid = int(st[rp + 2:].split()[1])
            children.setdefault(ppid, []).append(int(entry))
        except Exception:
            continue
    res, stack, seen = [], [root_pid], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        res.append(p)
        stack.extend(children.get(p, []))
    return res


def measure_tree_pss_rss_gb(root_pid: int):
    pids = get_process_tree_pids(root_pid)
    pss_kb = rss_kb = 0
    for p in pids:
        try:
            for line in open(f"/proc/{p}/smaps_rollup", "r"):
                if line.startswith("Pss:"):
                    pss_kb += int(line.split()[1])
                elif line.startswith("Rss:"):
                    rss_kb += int(line.split()[1])
        except Exception:
            continue
    return len(pids), pss_kb / 1048576.0, rss_kb / 1048576.0


def load_legacy_queries_and_gt(tag: str, idx_dir: pathlib.Path):
    with open(idx_dir / "core_index.bin", "rb") as cf:
        hdr = struct.unpack("<12Q", cf.read(96))
        N_docs, K, d, Nq, Lq = hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]
        cf.seek(K * d * 4 + d * d * 4, os.SEEK_CUR)
        Q32 = np.frombuffer(cf.read(Nq * Lq * d * 4), dtype=np.float32).reshape(Nq, Lq, d).copy()

    gt_map = {}
    with open(idx_dir / "workloads.bin", "rb") as wf:
        _, nwl, nd, nw = struct.unpack("<4Q", wf.read(32))
        for _ in range(nwl):
            wname = wf.read(64).split(b"\x00", 1)[0].decode()
            _corr = wf.read(32)
            _sig, nq_w, nbm_w, _av = struct.unpack("<d3Q", wf.read(32))
            wf.seek(nbm_w * nw * 8 + nq_w * 4, os.SEEK_CUR)
            gt10 = np.frombuffer(wf.read(nq_w * 10 * 4), dtype=np.uint32).reshape(nq_w, 10).copy()
            gt_map[wname] = gt10

    if tag == "vidore_v3_am":
        qdoms = np.load(P09 / "stage3" / "corpus_vidore_v3" / "vidore_v3_qdoms.npy").tolist()
        nq_use = min(Nq, 140)
        wl_specs = [
            ("NoCorr_100pct", lambda q: "TRUE", "NoCorr_100.00pct"),
            ("Nat_DomainEq", lambda q: f"dom = {qdoms[q]}", "Nat_DomainEq"),
            ("Nat_CrossDomainNeg", lambda q: f"dom = {(qdoms[q] + 1) % 7}", "Nat_CrossDomainNeg"),
        ]
    else:
        contract = json.load(open(idx_dir / "ext_db_eval_contract.json"))
        p95 = int(contract["p95_len"])
        nq_use = min(Nq, 50)
        wl_specs = [
            ("NoCorr_100pct", lambda q: "TRUE", "NoCorr_100.00pct"),
            ("NoCorr_5pct", lambda q: "nc5 = TRUE", "NoCorr_5.00pct_ExactNC5"),
            ("Nat_Dom0_Filter", lambda q: "dom = 0", "Nat_Dom0_Filter"),
            ("Nat_DocLenRange_5pct", lambda q: f"n_toks >= {p95}", "Nat_DocLenRange_5pct"),
        ]
    return Q32[:nq_use], Lq, wl_specs, gt_map


def bench_pg17_same_server_head_to_head(
    pg_data: pathlib.Path,
    idx_dir: pathlib.Path,
    sidecar_dir: pathlib.Path,
    tag: str,
    out_json: pathlib.Path,
    n_threads: int = 48,
    cache_mb: int = 256,
    shared_buf_mb: int = 1024,
    tier_mode: int = 2,
    use_numa: bool = True,
):
    import psycopg

    # Safety guard: NEVER touch frozen stage3/ext_dbs/!
    if "/stage3/ext_dbs/" in str(pg_data.resolve()):
        raise RuntimeError(f"Refusing to mutate frozen stage3 dir: {pg_data}")

    pg_bin = TP / "pg17_vchord" / "inst" / "bin"
    pg_lib = TP / "pg17_vchord" / "inst" / "lib"
    port = 55461
    sock_dir = P09 / "stage3_v4" / f"sock_h2h_{tag}"
    sock_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{pg_lib}:{P09 / 'stage3_v4' / 'src'}:" + env.get("LD_LIBRARY_PATH", "")

    conf_p = pg_data / "postgresql.conf"
    orig_conf = conf_p.read_text()
    evict_dir_page_cache(pg_data)
    try:
        marker = "# === PM_H2H_OVERRIDE ==="
        base_conf = orig_conf.split(marker)[0]
        conf_p.write_text(
            base_conf + f"\n{marker}\n"
            f"dynamic_library_path = '$libdir:{TP}/pg17_vchord/inst/lib/postgresql:{TP}/pg17_vchord/inst/lib'\n"
            f"port = {port}\n"
            f"unix_socket_directories = '{sock_dir}'\n"
            f"max_connections = 96\n"
            f"shared_buffers = '{shared_buf_mb}MB'\n"
            f"shared_preload_libraries = 'vchord.so,pg_certvec.so'\n"
            f"pg_certvec.cache_mb = {cache_mb}\n"
            f"pg_certvec.tier_mode = {tier_mode}\n"
            f"pg_certvec.use_o_direct = 1\n"
        )
        start_cmd = (NUMA_PREFIX if use_numa else []) + [str(pg_bin / "pg_ctl"), "-D", str(pg_data), "-l", str(pg_data / "pg_h2h.log"), "start"]
        subprocess.run(start_cmd, env=env, check=True)
        time.sleep(0.5)
        pm_pid = int((pg_data / "postmaster.pid").read_text().splitlines()[0].strip())
        dsn = f"host={sock_dir} port={port} user=postgres dbname=postgres"

        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_certvec;")
            info_json = json.loads(conn.execute("SELECT pm_certvec_info(%s, %s);", (str(idx_dir), str(sidecar_dir))).fetchone()[0])
            print(f"[PG17 H2H {tag}] pg_certvec info: {info_json}", flush=True)

        idle_procs, idle_pss, idle_rss = measure_tree_pss_rss_gb(pm_pid)

        is_docs768 = (tag == "msmarco_docs_768d")
        if is_docs768:
            Q_f32 = np.load(idx_dir / "queries_f32.npy")[:100].astype(np.float32)
            Lq = 1
            q_bytes_list = [Q_f32[i].tobytes() for i in range(len(Q_f32))]
            q_vchord_list = ["[" + ",".join(f"{x:.4f}" for x in Q_f32[i]) + "]" for i in range(len(Q_f32))]
            gt_map = {
                "NoCorr_100pct": np.load(idx_dir / "gt_NoCorr_100pct.npz")["w1_max_gt"],
                "NoCorr_5pct": np.load(idx_dir / "gt_NoCorr_5pct.npz")["w1_max_gt"],
                "Nat_Dom0_Eq": np.load(idx_dir / "gt_Nat_Dom0_Eq.npz")["w1_max_gt"],
            }
            wl_specs = [
                ("NoCorr_100pct", lambda q: "TRUE", "NoCorr_100pct"),
                ("NoCorr_5pct", lambda q: "nc5 = TRUE", "NoCorr_5pct"),
                ("Nat_Dom0_Eq", lambda q: "dom = 0", "Nat_Dom0_Eq"),
            ]
            id_col = "doc_id"
        else:
            Q32, Lq, wl_specs, gt_map = load_legacy_queries_and_gt(tag, idx_dir)
            q_bytes_list = [Q32[i].tobytes() for i in range(len(Q32))]
            q_vchord_list = []
            for i in range(len(Q32)):
                qv = Q32[i][np.linalg.norm(Q32[i], axis=1) > 1e-4]
                q_vchord_list.append("{" + ",".join('"[' + ",".join(f"{x:.4f}" for x in row) + ']"' for row in qv) + "}")
            id_col = "id"

        id_col = "doc_id" if is_docs768 else "id"
        pm_nprobe = 128 if is_docs768 else 16
        rows = []
        engines = ["pg_certvec", "vchord_rq_p64_pf"]
        for wlabel, pred_fn, gt_key in wl_specs:
            gt10 = gt_map.get(gt_key)
            n_eval = min(len(q_bytes_list), len(gt10) if gt10 is not None else 50)
            for engine in engines:
                # Capture EXPLAIN plan on query 0
                plan_str = ""
                with psycopg.connect(dsn, autocommit=True) as c0:
                    w0 = pred_fn(0)
                    if engine == "pg_certvec":
                        if w0 == "TRUE":
                            sql0 = f"EXPLAIN SELECT doc_id FROM pm_certvec_topk(%s, %s, NULL, %s, %s, 10, 0.05, {pm_nprobe}, 600);"
                            ex = c0.execute(sql0, (str(idx_dir), str(sidecar_dir), q_bytes_list[0], Lq)).fetchall()
                        else:
                            sql0 = f"EXPLAIN SELECT r.doc_id FROM pm_certvec_topk(%s, %s, (SELECT pm_roaring_agg({id_col}) FROM docs WHERE {w0}), %s, %s, 10, 0.05, {pm_nprobe}, 600) r JOIN docs d ON d.{id_col} = r.doc_id ORDER BY r.score DESC;"
                            ex = c0.execute(sql0, (str(idx_dir), str(sidecar_dir), q_bytes_list[0], Lq)).fetchall()
                    else:
                        c0.execute("SET vchordrq.probes = '64'; SET vchordrq.prefilter = on; SET max_parallel_workers_per_gather = 0;")
                        if is_docs768:
                            sql0 = f"EXPLAIN SELECT doc_id, MAX(-(emb <#> %s::halfvec(768))) AS sc FROM (SELECT doc_id, emb FROM chunks WHERE {w0} ORDER BY emb <#> %s::halfvec(768) LIMIT 120) sub GROUP BY doc_id ORDER BY sc DESC LIMIT 10;"
                            ex = c0.execute(sql0, (q_vchord_list[0], q_vchord_list[0])).fetchall()
                        else:
                            sql0 = f"EXPLAIN SELECT id FROM docs WHERE {w0} ORDER BY embs @# %s::halfvec(128)[] LIMIT 10;"
                            ex = c0.execute(sql0, (q_vchord_list[0],)).fetchall()
                    plan_str = " | ".join(r[0].strip() for r in ex[:4])

                bar_start = threading.Barrier(n_threads)
                bar_end = threading.Barrier(n_threads)
                stop_evt = threading.Event()
                counter = itertools.count(0)
                lats, recalls, all10s = [], [], []
                counts = [0] * n_threads
                lock = threading.Lock()
                t_start, t_end = [0.0], [0.0]
                peak_pss, peak_rss, peak_procs = [idle_pss], [idle_rss], [idle_procs]

                def worker(tid: int):
                    local_lats, local_rec, local_a10 = [], [], []
                    with psycopg.connect(dsn, autocommit=True) as c:
                        if engine.startswith("vchord"):
                            c.execute("SET vchordrq.probes = '64'; SET vchordrq.prefilter = on; SET max_parallel_workers_per_gather = 0;")

                        def run_one(qi: int):
                            wsql = pred_fn(qi)
                            if engine == "pg_certvec":
                                if wsql == "TRUE":
                                    res = c.execute(
                                        f"SELECT doc_id FROM pm_certvec_topk(%s, %s, NULL, %s, %s, 10, 0.05, {pm_nprobe}, 600);",
                                        (str(idx_dir), str(sidecar_dir), q_bytes_list[qi], Lq),
                                    ).fetchall()
                                else:
                                    res = c.execute(
                                        f"SELECT r.doc_id FROM pm_certvec_topk(%s, %s, (SELECT pm_roaring_agg({id_col}) FROM docs WHERE {wsql}), %s, %s, 10, 0.05, {pm_nprobe}, 600) r JOIN docs d ON d.{id_col} = r.doc_id ORDER BY r.score DESC;",
                                        (str(idx_dir), str(sidecar_dir), q_bytes_list[qi], Lq),
                                    ).fetchall()
                            elif is_docs768:
                                res = c.execute(
                                    f"SELECT doc_id, MAX(-(emb <#> %s::halfvec(768))) AS sc FROM (SELECT doc_id, emb FROM chunks WHERE {wsql} ORDER BY emb <#> %s::halfvec(768) LIMIT 120) sub GROUP BY doc_id ORDER BY sc DESC LIMIT 10;",
                                    (q_vchord_list[qi], q_vchord_list[qi]),
                                ).fetchall()
                            else:
                                res = c.execute(
                                    f"SELECT id FROM docs WHERE {wsql} ORDER BY embs @# %s::halfvec(128)[] LIMIT 10;",
                                    (q_vchord_list[qi],),
                                ).fetchall()
                            return [int(r[0]) for r in res]

                        for wq in range(2):
                            run_one((tid + wq) % n_eval)

                        bar_start.wait()
                        if tid == 0:
                            t_start[0] = time.perf_counter()
                        while not stop_evt.is_set():
                            idx_q = next(counter)
                            qi = idx_q % n_eval
                            t0 = time.perf_counter()
                            ids = run_one(qi)
                            local_lats.append((time.perf_counter() - t0) * 1000.0)
                            counts[tid] += 1
                            if gt10 is not None and len(local_rec) < 6:
                                gset = set(int(x) for x in gt10[qi])
                                hits = len(gset.intersection(ids[:10]))
                                local_rec.append(hits / 10.0)
                                local_a10.append(1.0 if hits == 10 else 0.0)
                            if tid == 0:
                                el = time.perf_counter() - t_start[0]
                                if counts[0] % 3 == 0:
                                    np_, pss_, rss_ = measure_tree_pss_rss_gb(pm_pid)
                                    if pss_ > peak_pss[0]:
                                        peak_procs[0], peak_pss[0], peak_rss[0] = np_, pss_, rss_
                                if el >= 6.0 and min(counts) >= 2:
                                    stop_evt.set()
                        bar_end.wait()
                        if tid == 0:
                            t_end[0] = time.perf_counter()
                    with lock:
                        lats.extend(local_lats)
                        recalls.extend(local_rec)
                        all10s.extend(local_a10)

                ts = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
                for t in ts: t.start()
                for t in ts: t.join()

                wall = max(1e-6, t_end[0] - t_start[0])
                tot_q = sum(counts)
                qps = tot_q / wall
                arr = np.array(lats)
                mean_ms = float(np.mean(arr))
                p50 = float(np.percentile(arr, 50))
                p99 = float(np.percentile(arr, 99))
                ll = (qps * mean_ms / 1000.0) / n_threads if n_threads > 0 else 0.0
                row = {
                    "tag": tag, "workload": wlabel, "engine": engine, "threads": n_threads,
                    "tier_mode": tier_mode, "cache_mb": cache_mb, "shared_buf_mb": shared_buf_mb,
                    "n_queries": tot_q, "wall_s": round(wall, 3), "qps": round(qps, 2),
                    "mean_ms": round(mean_ms, 3), "p50_ms": round(p50, 3), "p99_ms": round(p99, 3),
                    "littles_law": round(ll, 3),
                    "recall10": round(float(np.mean(recalls)), 4) if recalls else None,
                    "all10": round(float(np.mean(all10s)), 4) if all10s else None,
                    "idle_pss_gb": round(idle_pss, 4),
                    "pg_procs": peak_procs[0],
                    "active_pss_gb": round(peak_pss[0], 4),
                    "active_rss_gb": round(peak_rss[0], 4),
                    "plan": plan_str,
                }
                rows.append(row)
                print(
                    f"  [PG17 H2H {tag} {wlabel} {engine}] QPS={row['qps']} mean={row['mean_ms']}ms P50={row['p50_ms']}ms P99={row['p99_ms']}ms "
                    f"LL={row['littles_law']} R@10={row['recall10']} All10={row['all10']} procs={row['pg_procs']} PSS={row['active_pss_gb']}GB",
                    flush=True,
                )

        # Verify PG 17 Q-Thresh + GROUP BY dom across 20 queries (10 unfiltered + 10 filtered) for exact 0-FP precision!
        with psycopg.connect(dsn, autocommit=True) as c0:
            n_chk = min(10, len(q_bytes_list))
            tot_fp = 0
            tot_emitted = 0
            tot_groups = 0
            t_ms_list = []
            flt_wlabel, flt_pred_fn, _ = wl_specs[1] if len(wl_specs) > 1 else wl_specs[0]
            for is_flt in (False, True):
                for qi in range(n_chk):
                    wsql = flt_pred_fn(qi) if is_flt else "TRUE"
                    if wsql == "TRUE":
                        top10_q = c0.execute(
                            f"SELECT doc_id, score FROM pm_certvec_topk(%s, %s, NULL, %s, %s, 10, 0.05, {pm_nprobe}, 600);",
                            (str(idx_dir), str(sidecar_dir), q_bytes_list[qi], Lq),
                        ).fetchall()
                    else:
                        top10_q = c0.execute(
                            f"SELECT doc_id, score FROM pm_certvec_topk(%s, %s, (SELECT pm_roaring_agg({id_col}) FROM docs WHERE {wsql}), %s, %s, 10, 0.05, {pm_nprobe}, 600);",
                            (str(idx_dir), str(sidecar_dir), q_bytes_list[qi], Lq),
                        ).fetchall()
                    tau_q = float(top10_q[4][1]) if len(top10_q) >= 5 else (float(top10_q[-1][1]) if top10_q else 0.5)
                    t0 = time.perf_counter()
                    if wsql == "TRUE":
                        grps = c0.execute(
                            f"SELECT d.dom, COUNT(*), AVG(r.score) FROM pm_certvec_thresh(%s, %s, NULL, %s, %s, %s, 0.05, {pm_nprobe}) r "
                            f"JOIN docs d ON d.{id_col} = r.doc_id GROUP BY d.dom ORDER BY COUNT(*) DESC;",
                            (str(idx_dir), str(sidecar_dir), q_bytes_list[qi], Lq, tau_q),
                        ).fetchall()
                        fp_q = c0.execute(
                            f"SELECT COUNT(*) FROM pm_certvec_thresh(%s, %s, NULL, %s, %s, %s, 0.05, {pm_nprobe}) WHERE score < %s;",
                            (str(idx_dir), str(sidecar_dir), q_bytes_list[qi], Lq, tau_q, tau_q - 1e-9),
                        ).fetchone()[0]
                    else:
                        grps = c0.execute(
                            f"SELECT d.dom, COUNT(*), AVG(r.score) FROM pm_certvec_thresh(%s, %s, (SELECT pm_roaring_agg({id_col}) FROM docs WHERE {wsql}), %s, %s, %s, 0.05, {pm_nprobe}) r "
                            f"JOIN docs d ON d.{id_col} = r.doc_id GROUP BY d.dom ORDER BY COUNT(*) DESC;",
                            (str(idx_dir), str(sidecar_dir), q_bytes_list[qi], Lq, tau_q),
                        ).fetchall()
                        fp_q = c0.execute(
                            f"SELECT COUNT(*) FROM pm_certvec_thresh(%s, %s, (SELECT pm_roaring_agg({id_col}) FROM docs WHERE {wsql}), %s, %s, %s, 0.05, {pm_nprobe}) WHERE score < %s;",
                            (str(idx_dir), str(sidecar_dir), q_bytes_list[qi], Lq, tau_q, tau_q - 1e-9),
                        ).fetchone()[0]
                    t_ms_list.append((time.perf_counter() - t0) * 1000.0)
                    tot_groups += len(grps)
                    tot_emitted += sum(int(g[1]) for g in grps)
                    tot_fp += int(fp_q)
            thresh_summary = {
                "verified_queries": n_chk * 2,
                "filtered_workload": flt_wlabel,
                "total_emitted": tot_emitted,
                "avg_groups": round(tot_groups / max(1, n_chk * 2), 2),
                "p50_ms": round(float(np.median(t_ms_list)), 3),
                "false_positives": tot_fp,
            }
            print(f"  [PG17 Q-Thresh GROUP BY {tag}] {thresh_summary}", flush=True)

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"tag": tag, "info": info_json, "thresh_groupby_check": thresh_summary, "rows": rows}, indent=2) + "\n")
    finally:
        subprocess.run([str(pg_bin / "pg_ctl"), "-D", str(pg_data), "-m", "fast", "stop"], env=env, check=False)
        conf_p.write_text(orig_conf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pg17_h2h"], default="pg17_h2h")
    ap.add_argument("--pg-data", required=True)
    ap.add_argument("--idx-dir", required=True)
    ap.add_argument("--sidecar-dir", default="-")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--cache-mb", type=int, default=256)
    ap.add_argument("--shared-buf-mb", type=int, default=1024)
    ap.add_argument("--tier-mode", type=int, default=2)
    ap.add_argument("--no-numa", action="store_true")
    args = ap.parse_args()
    bench_pg17_same_server_head_to_head(
        pathlib.Path(args.pg_data), pathlib.Path(args.idx_dir), pathlib.Path(args.sidecar_dir),
        args.tag, pathlib.Path(args.out_json), args.threads, args.cache_mb, args.shared_buf_mb,
        args.tier_mode, not args.no_numa,
    )
