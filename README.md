# PM-MaxSim: Certified Out-of-Core Execution for Predicate-Filtered Multi-Vector Retrieval

This repository contains the C++20 source code, native index builders, dataset encoders, barrier-synchronized external vector database comparison harnesses, and machine-readable evaluation tables for **PM-MaxSim**, a predicate-aware multi-vector late-interaction (`MaxSim`) execution engine over out-of-core NVMe storage.

## Overview

PM-MaxSim combines three co-designed mechanisms:
1. **Adaptive Dual-Path Tier-0 Execution & KL Circuit Breaker**: Dynamically routes between predicate-first 4-bit L2-resident centroid-LUT scanning (`Path I`) and decoupled CRoaring-filtered inverted probing (`Path III`) using a closed-form crossover threshold $\tau_1^*$, guarded by a finite-sample KL-divergence circuit breaker with pure Path-I fallback against dead-posting starvation.
2. **Certified Out-of-Core Reranking Cascade & `Split_B1_B4_T3`**: Uses centroid-centered rotated residual quantization with exact $\mathbb{S}^{d-2}$ Beta-quantile upper bounds (`delta in {0.001, 0.01, 0.05, 0.10}`, supporting both unit-norm text tokens and non-unit merged visual patches such as `AdaMerge`), including a hybrid 1-bit-RAM / 4-bit-NVMe / FP16-NVMe tier driven by a 32 KB L1d-resident transposed nibble kernel and compacted dynamic `io_uring` bounce buffers (`256 KB` initial per thread).
3. **Zero-Remap Multi-Segment Dynamic LSM**: Supports online inserts via a live Delta Buffer, tombstone deletes, and sub-second background `CRoaring` compaction (`0.154 s` on MS MARCO 8.84M) without global RID remapping.

## Repository Structure

```text
src/
  Makefile                       # Zero-warning C++20 build (4 native targets: AVX-512 / AMX-BF16 / OpenMP / CRoaring / liburing)
  beta_quantiles_d128.hpp        # All 16 exact S^126 Beta quantile tables (delta in {0.001..0.10} x {Part1, Part2 k=10/50/100})
  core_index_builder_v4.cpp      # Native 48-core C++20 from-scratch core_index.bin + deduplicated doc_cids + CRoaring builder
  tier1_b1_builder_v4.cpp        # 24-thread 1-bit Tier-1 RAM builder + exact corpus M_norm_max scanner
  pm_engine_v4.cpp               # Production single-segment engine (8 methods, authentic Prop-2 WARP, --topk, --no-amx, active PSS)
  pm_multilsm_v4.cpp             # Production multi-segment LSM engine (3 base segs + 50K Delta + Compaction + --unit-test + --cache-cliff)
scripts/
  encode_msmarco_8m.py           # MS MARCO Passage (8.84M passages, ColBERTv2) FP16 encoder
  encode_vidore_v3.py            # ViDoRe v3 (7 enterprise domains, 16,868 PDF pages, ColQwen2.5 + AdaMerge-258) encoder
  build_vidore_v3_index.py       # ViDoRe v3 Raw & AdaMerge index + exact FP32 GPU ground truth builder
  build_msmarco_lsm_index.py     # MS MARCO 8.84M 3-segment LSM + 50K Delta + exact FP32 GPU GT builder
  build_lotte200k_and_gt_upgrade.py # 1:1 lotte_200k matched index, 16-WL MS MARCO GT, and official IR anchor verifier
  ingest_external_dbs.py         # Non-root ingestion for unmodified VectorChord 1.5.0, Qdrant v1.20.2, Weaviate v1.41.0
  ingest_qdrant_bq.py            # Non-root ingestion for unmodified Qdrant v1.20.2 Binary Quantization (BQ + On-Disk)
  bench_external_dbs.py          # Barrier-synchronized steady-state 48T benchmark + active smaps_rollup PSS for all external DBs
  bench_qdrant_bq.py             # Barrier-synchronized steady-state + genuine stop/flush/restart cold-start harness for Qdrant BQ
  measure_vchord_true_pss.py     # Linux /proc/<pid>/smaps_rollup full process-group PSS profiler for PostgreSQL/VectorChord
  measure_pm_build_and_mem.py    # Symmetric active 48T PSS/VmHWM & from-scratch 48c CPU-only / GPU K-means build harness
results/
  T1_e2e_48c_main.csv            # 48-core cold NVMe main results across all corpora & 8 methods (1,656 rows, max |proc-app| = 0.00 KB)
  T2_ablation_drilldown.csv      # AMX vs AVX-512, k=10/50/100, real MS MARCO LSM phases + Compaction, Cache-Cliff (90 rows)
  T3_rerank_fidelity_ir.csv      # Official 100/100 MS MARCO MRR@10 & 140/140 ViDoRe v3 nDCG@5 vs exact FP32 anchors (23 rows)
  T4_external_dbs_1to1.csv       # Barrier-clean 1:1 external DB comparison (median LL=0.920) & active 48T peak PSS (402 rows)
  T5_pareto_curves.csv           # 48-core QPS-vs-Recall@10 Pareto sweeps across lotte_200k, LoTTE 2.82M, MS MARCO 8.84M (254 rows)
  T6_latency_breakdown_prune.csv # All-10 exactness, Wave-1/Tier-2 prune rates & 5-stage ms latency breakdown (126 rows)
  T7_sensitivity_crossover.csv   # Complete 5-point 3-line Tau1* crossover, delta sensitivity, Theorem-4B semi-poisoning & 3-rep CV (258 rows)
  T8_build_memory_footprint.csv  # Symmetric active 48T peak PSS/VmHWM, disk size & CPU-only vs GPU K-means build times (29 rows)
```

## Quick Start

### 1. Build C++20 Production Binaries

Prerequisites: Linux (x86-64 with AVX-512; optional AMX-BF16 with runtime `--no-amx` fallback), `g++ >= 12`, `CRoaring`, and `liburing`.

```bash
export P09_ROOT=/path/to/workspace
cd src
make all
```

### 2. Run Theorem-3 Adversarial Unit Tests

Verifies exact hardware upper-bound soundness across `B=4` and `B=1` (unit and non-unit vectors, Part 1 and Part 2 $k \in \{10, 50, 100\}$, $\delta \in \{0.001, 0.01, 0.05, 0.10\}$, AMX and `--no-amx`; **36,954,264 combined adversarial pairs, 0 violations**):

```bash
./src/pm_engine_v4 --unit-test "$P09_ROOT/stage3/indexes/vidore_v3_raw"
./src/pm_engine_v4 --unit-test "$P09_ROOT/stage3/indexes/lotte_200k_matched" --no-amx
./src/pm_multilsm_v4 --unit-test "$P09_ROOT/stage3/indexes/msmarco_8m_lsm"
```

### 3. Benchmark Execution

```bash
# Single-segment cold NVMe benchmark (8 methods, 1/8/24/48 threads)
./src/pm_engine_v4 --bench-single "$P09_ROOT/stage3/indexes/vidore_v3_adamerge" out.json --threads 1,8,24,48 --topk 10

# Multi-segment 8.84M dynamic LSM + Compaction benchmark
./src/pm_multilsm_v4 --bench-lsm "$P09_ROOT/stage3/indexes/msmarco_8m_lsm" lsm_out.json --load-b4-ram --threads 1,8,24,48

# L1/L2/L3 4-bit vs 8-bit cache-cliff microbenchmark
./src/pm_multilsm_v4 --cache-cliff cache_cliff.json
```

## License

MIT License.
