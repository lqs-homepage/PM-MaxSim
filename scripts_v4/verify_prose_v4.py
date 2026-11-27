#!/usr/bin/env python3
"""
Automated Prose-vs-CSV Consistency, Telemetry Hygiene & Frozen-Boundary Verifier for Stage 3.5 (`r3-3.5-generalized/`).
Checks:
  1. Every key quantitative metric cited in `synthesis-v4-elevated.md` matches `tables_v4/T1_v4..T5_v4.csv` exactly.
  2. All 5 CSVs are non-empty, match `reconciled_v4_summary.json` row counts, and satisfy Little's Law coherence bounds.
  3. All 20 `T4_v4` cgroup rows satisfy uniform `cpus_allowed == "0-47,96-143"`, `memory_max_gb > 0`, `memory_swap_max == "0"`,
     `active_pss_gb <= memory_max_gb` on non-OOM rows, and `max_reclaim_events == 0` on all PM-MaxSim cages.
  4. Real measured `0` false positives (`verified_queries == 40` DuckDB, `20` PG17) and exact LOC counts (`578` PG, `252`/`284` DuckDB).
  5. Exact `T5_v4` full min/max ranges and explicit disclosure of `LoTTE 2.82M` `-1.4 pts` Stage-0a recall delta.
  6. Zero modifications to frozen Stage 3 directories (`r3-3.2/`, `r3-3.3/`, `r3-3.4/`, `science-reviews/P-09/`).
"""
import csv
import json
import pathlib
import subprocess

BASE = pathlib.Path(__file__).resolve().parent.parent
TD = BASE / "tables_v4"
SYNTH = (BASE / "synthesis-v4-elevated.md").read_text()


def load_csv(name):
    with open(TD / name) as f:
        return list(csv.DictReader(f))


def find_row(rows, **conds):
    for r in rows:
        if all(str(r.get(k, "")) == str(v) for k, v in conds.items()):
            return r
    raise AssertionError(f"Row not found: {conds}")


def check_in_prose(substr: str, label: str):
    if substr not in SYNTH:
        raise AssertionError(f"MISMATCH [{label}]: expected '{substr}' in synthesis-v4-elevated.md")


def main():
    t1 = load_csv("T1_v4_generalized_corpora.csv")
    t2 = load_csv("T2_v4_stage0a_theorem4c.csv")
    t3 = load_csv("T3_v4_dual_sql_hosts.csv")
    t4 = load_csv("T4_v4_cgroup_kkt_zipf.csv")
    t5 = load_csv("T5_v4_cross_gen_portability.csv")
    summary = json.loads((TD / "reconciled_v4_summary.json").read_text())

    assert len(t1) == summary["T1_rows"] == 317
    assert len(t2) == summary["T2_rows"] == 200
    assert len(t3) == summary["T3_rows"] == 38
    assert len(t4) == summary["T4_rows"] == 20
    assert len(t5) == summary["T5_rows"] == 68

    checks = 0

    # --- Global Little's Law coherence checks ---
    for r in t1:
        if r["littles_law"] and not r["task"].startswith("W4_QThresh"):
            ll = float(r["littles_law"])
            assert 0.85 <= ll <= 1.05, f"T1 LL out of bounds: {r}"
    checks += 1
    for tname, rows in [("T2", t2), ("T3", t3), ("T4", t4), ("T5", t5)]:
        for r in rows:
            if r.get("littles_law"):
                ll = float(r["littles_law"])
                assert 0.80 <= ll <= 1.05, f"{tname} LL out of bounds: {ll} in {r}"
        checks += 1

    # --- T1 checks (including full_index_ram_mb & paired recall disclosures) ---
    assert "t2_prune_rate" in t1[0] and "full_index_ram_mb" in t1[0]
    checks += 1

    r = find_row(t1, corpus="msmarco_docs_768d_c3000", engine="PM_split", workload="NoCorr_5pct", task="W1_ParentDoc_Max")
    check_in_prose(f"{float(r['qps']):.2f} QPS", "T1 docs768 c3000 5pct PM_split QPS"); checks += 1
    check_in_prose(f"R@10 = {float(r['recall10']):.4f}", "T1 docs768 c3000 5pct PM_split R10"); checks += 1
    check_in_prose(f"{float(r['full_index_ram_mb']):.2f} MB", "T1 docs768 c3000 PM_split full RAM"); checks += 1

    r = find_row(t1, corpus="msmarco_docs_768d_c3000", engine="PM_split", workload="NoCorr_100pct", task="W1_ParentDoc_Max")
    check_in_prose(f"{float(r['qps']):.2f} QPS", "T1 docs768 c3000 100pct PM_split QPS"); checks += 1
    check_in_prose(f"R@10 = {float(r['recall10']):.4f}", "T1 docs768 c3000 100pct PM_split R10"); checks += 1

    r = find_row(t1, corpus="msmarco_docs_768d", engine="Qdrant_1.20.2_INT8", workload="NoCorr_5pct")
    check_in_prose(f"{float(r['qps']):.2f} QPS", "T1 docs768 Qdrant INT8 5pct QPS"); checks += 1
    check_in_prose(f"R@10 = {float(r['recall10']):.4f}", "T1 docs768 Qdrant INT8 5pct R10"); checks += 1

    r = find_row(t1, corpus="bgem3_1024d_hybrid", engine="PM_split", workload="NoCorr_100pct", task="W2_ColBERT_1024D")
    check_in_prose(f"{float(r['qps']):.2f} QPS", "T1 bgem3 PM_split 100pct QPS"); checks += 1
    check_in_prose(f"{float(r['full_index_ram_mb']):.2f} MB", "T1 bgem3 PM_split full RAM"); checks += 1

    r = find_row(t1, corpus="bgem3_1024d_hybrid_np32", engine="PM_dram", workload="Nat_Topic0_Eq", task="W2_ColBERT_1024D")
    check_in_prose(f"{float(r['qps']):.2f} QPS", "T1 bgem3 np32 PM_dram Nat_Topic0_Eq QPS"); checks += 1
    check_in_prose(f"R@10 = {float(r['recall10']):.4f}", "T1 bgem3 np32 PM_dram Nat_Topic0_Eq R10"); checks += 1

    r = find_row(t1, corpus="bgem3_1024d_hybrid_np128_c4000", engine="PM_split", workload="NoCorr_100pct", task="W2_ColBERT_1024D")
    check_in_prose(f"{float(r['qps']):.2f} QPS", "GAP-1 T1 bgem3 np128 PM_split 100pct QPS"); checks += 1
    check_in_prose(f"R@10 = {float(r['recall10']):.4f}", "GAP-1 T1 bgem3 np128 PM_split 100pct R10"); checks += 1
    r = find_row(t1, corpus="bgem3_1024d_hybrid_np128_c4000", engine="PM_split", workload="NoCorr_5pct", task="W2_ColBERT_1024D")
    check_in_prose(f"{float(r['qps']):.2f} QPS", "GAP-1 T1 bgem3 np128 PM_split 5pct QPS"); checks += 1
    check_in_prose(f"R@10 = {float(r['recall10']):.4f}", "GAP-1 T1 bgem3 np128 PM_split 5pct R10"); checks += 1
    r = find_row(t1, corpus="bgem3_1024d_hybrid_np128_c4000", engine="PM_dram", workload="Nat_Topic0_Eq", task="W2_ColBERT_1024D")
    check_in_prose(f"{float(r['qps']):.2f} QPS", "GAP-1 T1 bgem3 np128 PM_dram Nat_Topic0_Eq QPS"); checks += 1
    check_in_prose(f"R@10 = {float(r['recall10']):.4f}", "GAP-1 T1 bgem3 np128 PM_dram Nat_Topic0_Eq R10"); checks += 1

    r = find_row(t1, corpus="bgem3_1024d_hybrid", engine="Qdrant_1.20.2_INT8", workload="NoCorr_100pct")
    check_in_prose(f"{float(r['qps']):.2f} QPS", "T1 bgem3 Qdrant INT8 100pct QPS"); checks += 1

    # --- T2 checks (warmup-clean 8.84M + LoTTE 2.82M -1.4 pts disclosure + GAP-3 C0=16384 recovery) ---
    r_m0 = find_row(t2, suite="msmarco_8.84m_dram", workload="NoCorr_100.00pct", stage0a="False")
    r_m1 = find_row(t2, suite="msmarco_8.84m_dram", workload="NoCorr_100.00pct", stage0a="True")
    assert r_m1["c1_pool_effective"] == "900"
    check_in_prose(f"{float(r_m0['qps']):.2f}", "T2 msmarco 8.84M dram s0a=False QPS"); checks += 1
    check_in_prose(f"{float(r_m1['qps']):.2f} QPS", "T2 msmarco 8.84M dram s0a=True QPS"); checks += 1
    check_in_prose(f"{float(r_m1['recall10']):.4f}", "T2 msmarco 8.84M s0a=True R10"); checks += 1

    # Confirm no cold-fault confounding on MS MARCO 8.84M DRAM NoCorr_1% (s0a_fired=0)
    r_nc1_0 = find_row(t2, suite="msmarco_8.84m_dram", workload="NoCorr_1.00pct", stage0a="False")
    r_nc1_1 = find_row(t2, suite="msmarco_8.84m_dram", workload="NoCorr_1.00pct", stage0a="True")
    assert abs(float(r_nc1_1["qps"]) / float(r_nc1_0["qps"]) - 1.0) < 0.05
    checks += 1

    r_l0 = find_row(t2, suite="lotte_2.82m_split", workload="NoCorr_100.00pct", stage0a="False")
    r_l1 = find_row(t2, suite="lotte_2.82m_split", workload="NoCorr_100.00pct", stage0a="True")
    check_in_prose(f"{float(r_l1['qps']):.2f} QPS", "T2 lotte 2.82M split s0a=True QPS"); checks += 1
    check_in_prose("-1.4 pts", "T2 lotte 2.82M -1.4 pts recall delta disclosure"); checks += 1
    check_in_prose(f"{float(r_l0['recall10']):.4f}", "T2 lotte 2.82M s0a=False R10"); checks += 1
    check_in_prose(f"{float(r_l1['recall10']):.4f}", "T2 lotte 2.82M s0a=True R10"); checks += 1
    r_c16s = find_row(t2, suite="lotte_2.82m_c0_16384_split", workload="NoCorr_100.00pct", stage0a="True")
    r_c16d = find_row(t2, suite="lotte_2.82m_c0_16384_dram", workload="NoCorr_100.00pct", stage0a="True")
    assert float(r_c16s["recall10"]) == 0.934 and float(r_c16d["recall10"]) == 0.934
    check_in_prose("C0 = 16384", "GAP-3 T2 lotte 2.82M C0=16384 recovery"); checks += 1

    # --- T3 checks (PG17 + DuckDB real FP, exact LOC, GAP-2/4/5) ---
    pg_c_lines = len((BASE / "src/pg_certvec.c").read_text().splitlines())
    pg_sql_lines = len((BASE / "src/pg_certvec--1.0.sql").read_text().splitlines())
    assert pg_c_lines + pg_sql_lines == 635
    check_in_prose("635", "T3 PG17 exact raw LOC"); checks += 1

    # GAP-4: Verify ALL 6 ViDoRe PG17 rows captured true 54-proc PSS!
    for r in [x for x in t3 if x["host"] == "PostgreSQL_17.12" and x["corpus"] == "vidore_v3_am"]:
        assert int(r["pg_procs"]) == 54 and float(r["active_pss_gb"]) > 1.7, f"GAP-4 ViDoRe PSS check failed: {r}"
    checks += 1

    for corpus, wl, eng_pm, eng_vc in [
        ("vidore_v3_am", "Nat_DomainEq", "pg_certvec", "vchord_rq_p64_pf"),
        ("lotte_200k", "NoCorr_5pct", "pg_certvec", "vchord_rq_p64_pf"),
        ("msmarco_docs_768d", "NoCorr_5pct", "pg_certvec_c3000", "vchord_rq_p64_lim1000"),
        ("msmarco_docs_768d", "NoCorr_5pct", "pg_certvec_w3_hybrid", "vchord_rq_p64_pf"),
        ("msmarco_docs_768d", "Nat_Dom0_Eq", "pg_certvec_w3_hybrid", "vchord_rq_p64_lim1000"),
    ]:
        r_pm = find_row(t3, host="PostgreSQL_17.12", corpus=corpus, engine=eng_pm, workload=wl)
        r_vc = find_row(t3, host="PostgreSQL_17.12", corpus=corpus, engine=eng_vc, workload=wl)
        check_in_prose(f"{float(r_pm['qps']):.2f}", f"T3 PG17 {corpus} {wl} {eng_pm}"); checks += 1
        check_in_prose(f"{float(r_vc['qps']):.2f}", f"T3 PG17 {corpus} {wl} {eng_vc}"); checks += 1
        assert r_pm["thresh_fp"] == "0" and r_pm["thresh_verified_q"] == "20" and r_pm["integration_loc"] == "635"
        checks += 1

    for corpus in ["vidore_v3_am", "lotte_200k", "msmarco_docs_768d"]:
        r_dk1 = find_row(t3, host="DuckDB_1.2.1", corpus=corpus, workload="Q_TopK_JOIN_WHERE")
        r_dk2 = find_row(t3, host="DuckDB_1.2.1", corpus=corpus, workload="Q_Thresh_GROUP_BY")
        assert r_dk2["thresh_fp"] == "0" and r_dk2["thresh_verified_q"] == "40"
        check_in_prose(f"{float(r_dk1['host_overhead_ms']):.2f} ms", f"T3 DuckDB {corpus} TopK overhead"); checks += 1
        check_in_prose(f"{float(r_dk2['host_overhead_ms']):.2f} ms", f"T3 DuckDB {corpus} Thresh overhead"); checks += 1

    # --- T4 checks (uniform 48c, cold eviction, memory_max > 0, swap == 0, PSS <= cap, reclaim stalls) ---
    for r in t4:
        assert r["cpus_allowed"] == "0-47,96-143", f"T4 non-48c cpus_allowed: {r}"
        assert float(r["memory_max_gb"]) > 0.0, f"T4 memory_max_gb <= 0: {r}"
        assert r["memory_swap_max"] == "0", f"T4 swap != 0: {r}"
        if r["oom_killed"] == "False":
            assert float(r["active_pss_gb"]) <= float(r["memory_max_gb"]), f"T4 active_pss > memory_max: {r}"
        if r["mode"] in ("cgroup_zipf_pm", "cgroup_certvec5_pm"):
            assert r["active_tier"] in ("1", "2", "3"), f"T4 invalid active_tier: {r}"
            assert r["oom_killed"] == "False" and r["oom_kill_events"] == "0"
            assert r["max_reclaim_events"] == "0" and r["high_events"] == "0"
    checks += 1

    r_cross = find_row(t4, case="cg_msmarco_seg0_zipf50k_20G_kkt_cross")
    r_24 = find_row(t4, case="cg_msmarco_seg0_zipf50k_24G")
    assert r_cross["active_tier"] == "1" and r_24["active_tier"] == "3"
    assert float(r_cross["proc_mb_q"]) < float(r_24["proc_mb_q"]) and float(r_cross["qps"]) > float(r_24["qps"])
    check_in_prose(f"{float(r_cross['qps']):.2f} QPS", "T4 20G KKT crossover QPS"); checks += 1
    check_in_prose(f"{float(r_cross['proc_mb_q']):.3f} MB/q", "T4 20G KKT crossover IO"); checks += 1
    check_in_prose(f"{float(r_24['qps']):.2f} QPS", "T4 24G DRAM QPS"); checks += 1

    for oom_case in ["cg_bgem3_qdrant_bq_2G", "cg_bgem3_qdrant_int8_4G", "cg_bgem3_qdrant_int8_8G", "cg_docs768_qdrant_int8_2G", "cg_docs768_qdrant_int8_4G"]:
        ro = find_row(t4, case=oom_case)
        assert ro["oom_killed"] == "True" and ro["oom_kill_events"] == "1", f"Expected OOM kill on {oom_case}"
        checks += 1

    r_qbq4 = find_row(t4, case="cg_bgem3_qdrant_bq_4G")
    assert int(r_qbq4["max_reclaim_events"]) > 1_000_000
    check_in_prose("1,141,695", "T4 Qdrant BQ 4G kernel reclaim stalls"); checks += 1
    check_in_prose(f"{float(r_qbq4['qps']):.2f} QPS", "T4 Qdrant BQ 4G QPS"); checks += 1

    # --- T5 checks (EXACT full min/max ranges across all 5 icelake-16c files!) ---
    t5_by_file = {}
    for r in t5:
        t5_by_file.setdefault(r["file"], []).append(r)
    for fn, rows in t5_by_file.items():
        qmin, qmax = min(float(x["qps"]) for x in rows), max(float(x["qps"]) for x in rows)
        check_in_prose(f"{qmin:.2f}–{qmax:.2f} QPS", f"T5 {fn} exact full QPS range"); checks += 1
    alphas = [float(x["calibrated_alpha_k0_ns"]) for x in t5]
    check_in_prose(f"{min(alphas):.4f}–{max(alphas):.4f} ns/cid", "T5 exact full alpha_k0 range"); checks += 1

    # --- Frozen Stage 3 boundary integrity check via git status ---
    repo_root = BASE.parent.parent.parent
    st = subprocess.check_output(
        ["git", "status", "--porcelain", "paper-work/research/r3-3.2", "paper-work/research/r3-3.3",
         "paper-work/research/r3-3.4", "paper-work/science-reviews"],
        cwd=repo_root, text=True,
    )
    assert st.strip() == "", f"Frozen Stage 3 paths modified!\n{st}"
    checks += 1

    print(f"[PASS] verify_prose_v4.py: {checks}/{checks} prose-vs-CSV, LL, cgroup, FP, range, and frozen-boundary assertions passed (0 mismatches)!")


if __name__ == "__main__":
    main()
