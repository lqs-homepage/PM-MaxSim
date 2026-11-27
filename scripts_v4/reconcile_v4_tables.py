#!/usr/bin/env python3
"""
Layer 4 Machine-Readable Results Reconciler & `tables_v4/` Exporter (Audit-Hardened Rev. 2).
Strictly isolated from frozen `r3-3.3/tables/T1..T8.csv`.
"""
import csv
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent.parent
RES_DIR = BASE / "results_v4"
OUT_DIR = BASE / "tables_v4"


def load_json(path: pathlib.Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[Exported] {path.name}: {len(rows)} rows")


def export_t1(summary: dict):
    rows = []
    for corpus, prefix, tiers in [
        ("bgem3_1024d_hybrid", "bgem3_1024d", ["nvme", "split", "dram"]),
        ("bgem3_1024d_hybrid_np32", "bgem3_1024d_np32", ["split", "dram"]),
        ("bgem3_1024d_hybrid_np64_c2000", "bgem3_1024d_np64_c2000", ["split", "dram"]),
        ("bgem3_1024d_hybrid_np128_c4000", "bgem3_1024d_np128_c4000", ["split", "dram"]),
        ("msmarco_docs_768d", "msmarco_docs_768d", ["nvme", "split", "dram"]),
        ("msmarco_docs_768d_c3000", "msmarco_docs_768d_c3000", ["split", "dram"]),
    ]:
        for tier in tiers:
            d = load_json(RES_DIR / f"{prefix}_{tier}.json")
            if not d:
                continue
            full_ram = d.get("full_index_ram_mb", round(float(d.get("tier0_mb", 0)) + float(d.get("b1_mb", 0)) + float(d.get("b4_mb", 0)), 2))
            for r in d.get("rows", []):
                rows.append({
                    "corpus": corpus, "engine": r["method"], "tier": tier,
                    "workload": r["workload"], "task": r["task"], "threads": d["threads"],
                    "qps": r["qps"], "mean_ms": r.get("mean_ms", ""), "p50_ms": r["p50_ms"], "p99_ms": r["p99_ms"],
                    "littles_law": r.get("littles_law", ""),
                    "recall10": r["recall10"], "all10": r["all10"],
                    "thresh_prec": r.get("precision", ""), "thresh_fp": r.get("false_positives", ""),
                    "t2_prune_rate": r.get("t2_prune_rate", ""), "app_mb_q": r.get("app_mb_q", ""),
                    "proc_mb_q": r.get("proc_mb_q", ""), "active_pss_gb": r.get("active_pss_gb", ""),
                    "ram_tier0_mb": d.get("tier0_mb", ""), "ram_b1_mb": d.get("b1_mb", ""),
                    "ram_b4_mb": d.get("b4_mb", ""), "full_index_ram_mb": full_ram,
                })
    for corpus, fn_tpl in [
        ("bgem3_1024d_hybrid", "qdrant_{q}_bgem3_uncapped_48c.json"),
        ("msmarco_docs_768d", "qdrant_{q}_docs768_48c.json"),
    ]:
        for q in ["bq", "int8"]:
            d = load_json(RES_DIR / fn_tpl.format(q=q))
            if not d:
                continue
            for r in d.get("rows", []):
                if r.get("status"):
                    continue
                rows.append({
                    "corpus": corpus, "engine": f"Qdrant_1.20.2_{q.upper()}", "tier": "on_disk_hnsw",
                    "workload": r["workload"],
                    "task": "W2_ColBERT_1024D" if corpus == "bgem3_1024d_hybrid" else "W1_ParentDoc_Max_group_by",
                    "threads": r["threads"], "qps": r["qps"], "mean_ms": "", "p50_ms": r["p50_ms"], "p99_ms": r["p99_ms"],
                    "littles_law": r.get("littles_law", ""), "recall10": r["recall10"], "all10": "",
                    "thresh_prec": "", "thresh_fp": "", "t2_prune_rate": "", "app_mb_q": "", "proc_mb_q": "",
                    "active_pss_gb": r["active_pss_gb"], "ram_tier0_mb": "", "ram_b1_mb": "", "ram_b4_mb": "", "full_index_ram_mb": "",
                })
    fields = [
        "corpus", "engine", "tier", "workload", "task", "threads", "qps", "mean_ms", "p50_ms", "p99_ms",
        "littles_law", "recall10", "all10", "thresh_prec", "thresh_fp", "t2_prune_rate", "app_mb_q",
        "proc_mb_q", "active_pss_gb", "ram_tier0_mb", "ram_b1_mb", "ram_b4_mb", "full_index_ram_mb",
    ]
    write_csv(OUT_DIR / "T1_v4_generalized_corpora.csv", fields, rows)
    summary["T1_rows"] = len(rows)


def export_t2(summary: dict):
    rows = []
    specs = [
        ("lotte_200k_np16_c600", "lotte200k_s0a_dram.json", "dram", 600),
        ("lotte_200k_np32_c4000", "lotte200k_s0a_hi_dram.json", "dram", 4000),
        ("lotte_2.82m_split", "lotte28m_s0a_split.json", "split", 600),
        ("lotte_2.82m_dram", "lotte28m_s0a_dram.json", "dram", 600),
        ("lotte_2.82m_c0_4096_split", "lotte28m_s0a_c0_4096_split.json", "split", 600),
        ("lotte_2.82m_c0_4096_dram", "lotte28m_s0a_c0_4096_dram.json", "dram", 600),
        ("lotte_2.82m_c0_8192_split", "lotte28m_s0a_c0_8192_split.json", "split", 600),
        ("lotte_2.82m_c0_8192_dram", "lotte28m_s0a_c0_8192_dram.json", "dram", 600),
        ("lotte_2.82m_c0_16384_split", "lotte28m_s0a_c0_16384_split.json", "split", 600),
        ("lotte_2.82m_c0_16384_dram", "lotte28m_s0a_c0_16384_dram.json", "dram", 600),
        ("lotte_2.82m_c0_32768_split", "lotte28m_s0a_c0_32768_split.json", "split", 600),
        ("lotte_2.82m_c0_32768_dram", "lotte28m_s0a_c0_32768_dram.json", "dram", 600),
        ("msmarco_8.84m_split", "msmarco8m_s0a_split.json", "split", 900),
        ("msmarco_8.84m_dram", "msmarco8m_s0a_dram.json", "dram", 900),
    ]
    for label, fn, tier, c1_eff in specs:
        d = load_json(RES_DIR / fn)
        if not d:
            continue
        for r in d.get("rows", []):
            rows.append({
                "suite": label, "tier": tier, "threads": d.get("threads", 48),
                "nprobe": d.get("nprobe", 16), "c1_pool_effective": c1_eff,
                "workload": r["workload"], "corr": r.get("corr", ""), "sigma": r.get("sigma", ""),
                "stage0a": r["stage0a"], "qps": r["qps"], "mean_ms": r.get("mean_ms", ""),
                "p50_ms": r["p50_ms"], "p99_ms": r["p99_ms"], "littles_law": r.get("littles_law", ""),
                "t_tier0_ms": r["t_tier0_ms"], "recall10": r["recall10"], "all10": r["all10"],
                "app_mb_q": r.get("app_mb_q", ""), "proc_mb_q": r.get("proc_mb_q", ""),
                "s0a_fired": r.get("s0a_fired", ""), "cb_trips": r.get("cb_trips", ""),
                "active_pss_gb": r.get("active_pss_gb", ""),
            })
    fields = [
        "suite", "tier", "threads", "nprobe", "c1_pool_effective", "workload", "corr", "sigma",
        "stage0a", "qps", "mean_ms", "p50_ms", "p99_ms", "littles_law", "t_tier0_ms",
        "recall10", "all10", "app_mb_q", "proc_mb_q", "s0a_fired", "cb_trips", "active_pss_gb",
    ]
    write_csv(OUT_DIR / "T2_v4_stage0a_theorem4c.csv", fields, rows)
    summary["T2_rows"] = len(rows)


def export_t3(summary: dict):
    rows = []
    for tag, fn in [
        ("vidore_v3_am", "pg17_h2h_vidore_am_48c.json"),
        ("lotte_200k", "pg17_h2h_lotte_200k_48c.json"),
        ("msmarco_docs_768d", "pg17_h2h_docs768_48c.json"),
    ]:
        d = load_json(RES_DIR / fn)
        if not d:
            continue
        tchk = d.get("thresh_groupby_check", {})
        for r in d.get("rows", []):
            rows.append({
                "host": "PostgreSQL_17.12", "corpus": tag, "engine": r["engine"],
                "workload": r["workload"], "threads": r["threads"],
                "qps": r["qps"], "mean_ms": r.get("mean_ms", ""), "p50_ms": r["p50_ms"], "p99_ms": r["p99_ms"],
                "littles_law": r["littles_law"], "recall10": r["recall10"], "all10": r.get("all10", ""),
                "pg_procs": r["pg_procs"], "active_pss_gb": r["active_pss_gb"],
                "core_p50_ms": "", "host_overhead_ms": "",
                "integration_loc": 635 if r["engine"].startswith("pg_certvec") else "",
                "thresh_fp": tchk.get("false_positives", "") if r["engine"].startswith("pg_certvec") else "",
                "thresh_verified_q": tchk.get("verified_queries", "") if r["engine"].startswith("pg_certvec") else "",
            })
    for tag, fn in [
        ("vidore_v3_am", "duckdb_decomp_vidore_am.json"),
        ("lotte_200k", "duckdb_decomp_lotte_200k.json"),
        ("msmarco_docs_768d", "duckdb_decomp_docs768.json"),
    ]:
        d = load_json(RES_DIR / fn)
        if not d:
            continue
        for mode_k, wlabel in [("topk", "Q_TopK_JOIN_WHERE"), ("thresh_groupby", "Q_Thresh_GROUP_BY")]:
            m = d.get(mode_k, {})
            rows.append({
                "host": "DuckDB_1.2.1", "corpus": tag, "engine": "duckdb_certvec_TableInOut",
                "workload": wlabel, "threads": 1,
                "qps": round(1000.0 / max(1e-6, m.get("duckdb_e2e_p50_ms", 1.0)), 2),
                "mean_ms": "", "p50_ms": m.get("duckdb_e2e_p50_ms", ""), "p99_ms": "",
                "littles_law": "", "recall10": "", "all10": "",
                "pg_procs": 1, "active_pss_gb": "",
                "core_p50_ms": m.get("core_p50_ms", ""), "host_overhead_ms": m.get("host_overhead_ms", ""),
                "integration_loc": d.get("integration_loc", ""),
                "thresh_fp": m.get("false_positives", ""),
                "thresh_verified_q": m.get("verified_queries", ""),
            })
    fields = [
        "host", "corpus", "engine", "workload", "threads", "qps", "mean_ms", "p50_ms", "p99_ms",
        "littles_law", "recall10", "all10", "pg_procs", "active_pss_gb",
        "core_p50_ms", "host_overhead_ms", "integration_loc", "thresh_fp", "thresh_verified_q",
    ]
    write_csv(OUT_DIR / "T3_v4_dual_sql_hosts.csv", fields, rows)
    summary["T3_rows"] = len(rows)


def export_t4(summary: dict):
    rows = []
    for p in sorted(RES_DIR.glob("cg_*.json")):
        if p.name.endswith(".inner.json"):
            continue
        d = load_json(p)
        if not d:
            continue
        cg = d.get("cgroup", {})
        inn = d.get("inner", {})
        ev = cg.get("memory_events", {})
        qps = mean_ms = p50 = p99 = ll = r10 = hit_rate = hits_b4 = hits_t2 = io_mb = active_tier = pss_gb = ""
        if "mode" in inn and inn["mode"] == "zipf_mixed_stream":
            qps, mean_ms, p50, p99 = inn.get("qps", ""), inn.get("mean_ms", ""), inn.get("p50_ms", ""), inn.get("p99_ms", "")
            ll = inn.get("littles_law", "")
            hit_rate, hits_b4, hits_t2 = inn.get("cache_hit_rate", ""), inn.get("cache_hits_b4", ""), inn.get("cache_hits_t2", "")
            io_mb, active_tier, pss_gb = inn.get("proc_mb_q", ""), inn.get("active_tier", ""), inn.get("active_pss_gb", "")
        elif "rows" in inn and inn["rows"]:
            r0 = inn["rows"][0]
            qps, mean_ms, p50, p99 = r0.get("qps", ""), r0.get("mean_ms", ""), r0.get("p50_ms", ""), r0.get("p99_ms", "")
            ll, r10 = r0.get("littles_law", ""), r0.get("recall10", "")
            io_mb = r0.get("proc_mb_q", "")
            active_tier = inn.get("active_tier_mode", inn.get("tier", ""))
            pss_gb = max((float(x.get("active_pss_gb", 0.0)) for x in inn["rows"] if x.get("active_pss_gb") not in ("", None)), default="")
        rows.append({
            "case": p.stem, "mode": d.get("mode", ""), "mem_cap": d.get("mem_cap", ""),
            "ram_budget_mb": d.get("ram_budget_mb", ""), "active_tier": active_tier,
            "cpus_allowed": cg.get("cpus_allowed", ""),
            "returncode": cg.get("returncode", ""), "oom_killed": cg.get("oom_killed", ""),
            "memory_max_gb": cg.get("memory_max_gb", ""), "memory_swap_max": cg.get("memory_swap_max", ""),
            "memory_peak_gb": cg.get("memory_peak_gb", ""), "active_pss_gb": pss_gb,
            "high_events": ev.get("high", 0), "max_reclaim_events": ev.get("max", 0),
            "oom_events": ev.get("oom", 0), "oom_kill_events": ev.get("oom_kill", 0),
            "qps": qps, "mean_ms": mean_ms, "p50_ms": p50, "p99_ms": p99, "littles_law": ll,
            "recall10": r10, "cache_hit_rate": hit_rate, "cache_hits_b4": hits_b4,
            "cache_hits_t2": hits_t2, "proc_mb_q": io_mb,
        })
    fields = [
        "case", "mode", "mem_cap", "ram_budget_mb", "active_tier", "cpus_allowed",
        "returncode", "oom_killed", "memory_max_gb", "memory_swap_max", "memory_peak_gb",
        "active_pss_gb", "high_events", "max_reclaim_events", "oom_events", "oom_kill_events",
        "qps", "mean_ms", "p50_ms", "p99_ms", "littles_law", "recall10", "cache_hit_rate",
        "cache_hits_b4", "cache_hits_t2", "proc_mb_q",
    ]
    write_csv(OUT_DIR / "T4_v4_cgroup_kkt_zipf.csv", fields, rows)
    summary["T4_rows"] = len(rows)


def export_t5(summary: dict):
    rows = []
    for p in sorted(RES_DIR.glob("icelake16c_*.json")):
        d = load_json(p)
        if not d:
            continue
        for r in d.get("rows", []):
            if r.get("stage0a") and "vidore" in p.name:
                continue
            rows.append({
                "machine": "icelake_noamx_linux5.4_pread",
                "file": p.stem, "threads": d["threads"], "tier": d["tier"],
                "io_backend": d["io_backend"], "amx": d["amx"],
                "calibrated_alpha_k0_ns": d.get("calibrated_alpha_k0_ns", ""),
                "calibrated_tau1": d.get("calibrated_tau1", ""),
                "workload": r["workload"], "stage0a": r["stage0a"],
                "qps": r["qps"], "mean_ms": r.get("mean_ms", ""), "p50_ms": r["p50_ms"], "p99_ms": r["p99_ms"],
                "littles_law": r.get("littles_law", ""),
                "recall10": r["recall10"], "all10": r["all10"],
                "proc_mb_q": r.get("proc_mb_q", ""), "active_pss_gb": r.get("active_pss_gb", ""),
            })
    fields = [
        "machine", "file", "threads", "tier", "io_backend", "amx",
        "calibrated_alpha_k0_ns", "calibrated_tau1", "workload", "stage0a",
        "qps", "mean_ms", "p50_ms", "p99_ms", "littles_law",
        "recall10", "all10", "proc_mb_q", "active_pss_gb",
    ]
    write_csv(OUT_DIR / "T5_v4_cross_gen_portability.csv", fields, rows)
    summary["T5_rows"] = len(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"results_dir": str(RES_DIR), "tables_dir": str(OUT_DIR)}
    export_t1(summary)
    export_t2(summary)
    export_t3(summary)
    export_t4(summary)
    export_t5(summary)
    (OUT_DIR / "reconciled_v4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[Reconcile Summary]", json.dumps(summary))


if __name__ == "__main__":
    main()
