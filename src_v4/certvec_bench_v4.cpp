// certvec_bench_v4.cpp — Complete Layer 4 Exclusive Benchmark Suite for Generalized libpm_certvec.so
// Implements 4 benchmark modes (0 warnings, -Wall -Wextra):
//   1. --bench-stage0a <v4_idx_dir> <sidecar_dir> <out_json> [--threads 48] [--tier split|dram|nvme|auto] [--ram-mb 0] [--pread] [--no-amx] [--nprobe 16] [--c1 600]
//   2. --bench-lsm-stage0a <msmarco_lsm_dir> <sidecar_root> <out_json> [--threads 48] [--tier dram|split|nvme] [--nprobe 16] [--c1 600]
//   3. --bench-certvec5 <idx_dir> <kind:docs768|bgem3> <out_json> [--threads 48] [--tier split|dram|nvme|auto] [--ram-mb 0] [--nprobe 16] [--c1 600]
//   4. --bench-zipf-cache <v4_idx_dir> <sidecar_dir> <queries_f32_bin> <Nq> <Lq> <stream_qids_u32> <stream_tids_u16> <n_reqs> <out_json> [--threads 48] [--tier auto|split|nvme|dram] [--ram-mb 1024] [--mutate-pct 2]

#include "pm_certvec.h"
#include "pm_kernels.hpp"
#include "roaring.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <time.h>
#include <unordered_set>
#include <vector>

static inline double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<double>(ts.tv_sec) * 1e9 + static_cast<double>(ts.tv_nsec);
}

static uint64_t proc_io_read_bytes() {
    std::ifstream f("/proc/self/io");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("read_bytes:", 0) == 0) return std::stoull(line.substr(11));
    }
    return 0;
}

static void proc_mem_gb(double& pss, double& rss, double& hwm) {
    pss = rss = hwm = 0.0;
    {
        std::ifstream f("/proc/self/smaps_rollup");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("Pss:", 0) == 0) pss = std::stod(line.substr(4)) / 1048576.0;
            else if (line.rfind("Rss:", 0) == 0) rss = std::stod(line.substr(4)) / 1048576.0;
        }
    }
    {
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("VmHWM:", 0) == 0) { hwm = std::stod(line.substr(6)) / 1048576.0; break; }
        }
    }
}

static pm_tier_mode_t parse_tier(const std::string& s) {
    if (s == "nvme") return PM_TIER_NVME_T1;
    if (s == "dram") return PM_TIER_DRAM_T1;
    if (s == "auto") return PM_TIER_AUTO_KKT;
    return PM_TIER_SPLIT_B1B4;
}

struct V4Workload {
    std::string name, corr;
    double sigma = 0.0;
    uint64_t n_q = 0, n_bm = 0, actual_valid = 0;
    std::vector<uint64_t> bitmaps;
    std::vector<uint32_t> q_to_bm;
    std::vector<uint32_t> gt10;
};

static bool load_v4_queries_and_workloads(const std::string& dir, uint64_t& N_docs, uint64_t& d, uint64_t& Nq, uint64_t& Lq,
                                          std::vector<float>& Q32, std::vector<V4Workload>& wls,
                                          const std::string& wl_file = "") {
    {
        std::ifstream f(dir + "/core_index.bin", std::ios::binary);
        if (!f) return false;
        uint64_t hdr[12];
        f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        N_docs = hdr[1]; uint64_t K = hdr[2]; d = hdr[3]; Nq = hdr[4]; Lq = hdr[5];
        f.seekg(static_cast<std::streamoff>(K * d * 4ULL + d * d * 4ULL), std::ios::cur);
        Q32.resize(Nq * Lq * d);
        f.read(reinterpret_cast<char*>(Q32.data()), Q32.size() * 4ULL);
    }
    std::string wpath = wl_file.empty() ? (dir + "/workloads.bin") : wl_file;
    std::ifstream f(wpath, std::ios::binary);
    if (!f) return false;
    uint64_t h[4]; f.read(reinterpret_cast<char*>(h), sizeof(h));
    uint64_t nw = (N_docs + 63ULL) / 64ULL;
    wls.resize(h[1]);
    for (uint64_t i = 0; i < h[1]; ++i) {
        char nb[64], cb[32];
        f.read(nb, 64); f.read(cb, 32);
        wls[i].name = nb; wls[i].corr = cb;
        f.read(reinterpret_cast<char*>(&wls[i].sigma), 8);
        uint64_t qh[3]; f.read(reinterpret_cast<char*>(qh), sizeof(qh));
        wls[i].n_q = qh[0]; wls[i].n_bm = qh[1]; wls[i].actual_valid = qh[2];
        wls[i].bitmaps.resize(qh[1] * nw);
        f.read(reinterpret_cast<char*>(wls[i].bitmaps.data()), wls[i].bitmaps.size() * 8ULL);
        wls[i].q_to_bm.resize(qh[0]);
        f.read(reinterpret_cast<char*>(wls[i].q_to_bm.data()), qh[0] * 4ULL);
        wls[i].gt10.resize(qh[0] * 10ULL);
        f.read(reinterpret_cast<char*>(wls[i].gt10.data()), qh[0] * 10ULL * 4ULL);
    }
    return true;
}

// ============================================================================
// Mode 1: Single-Segment Stage-0a (Theorem 4C) + Q-Thresh + Portability Suite
// ============================================================================
static int run_bench_stage0a(int argc, char** argv) {
    if (argc < 5) { std::cerr << "Usage: --bench-stage0a <v4_dir> <sidecar_dir> <out_json> [opts]\n"; return 1; }
    std::string dir = argv[2], sc_dir = argv[3], out_json = argv[4];
    int nT = 48;
    pm_tier_mode_t tier = PM_TIER_DRAM_T1;
    pm_io_backend_t io_b = PM_IO_URING;
    uint8_t no_amx = 0;
    size_t ram_mb = 0;
    uint32_t nprobe = 16, c1_pool = 600;
    bool run_thresh = false;
    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) nT = std::stoi(argv[++i]);
        else if (a == "--tier" && i + 1 < argc) tier = parse_tier(argv[++i]);
        else if (a == "--ram-mb" && i + 1 < argc) ram_mb = std::stoull(argv[++i]);
        else if (a == "--nprobe" && i + 1 < argc) nprobe = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--c1" && i + 1 < argc) c1_pool = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--pread") io_b = PM_IO_PREAD_POOL;
        else if (a == "--no-amx") no_amx = 1;
        else if (a == "--thresh") run_thresh = true;
    }

    uint64_t N_docs = 0, d = 0, Nq = 0, Lq = 0;
    std::vector<float> Q32;
    std::vector<V4Workload> wls;
    if (!load_v4_queries_and_workloads(dir, N_docs, d, Nq, Lq, Q32, wls)) { std::cerr << "Failed load v4\n"; return 1; }
    uint64_t nw = (N_docs + 63ULL) / 64ULL;

    pm_open_opts_t opts{};
    opts.tier_mode = tier;
    opts.ram_budget_bytes = ram_mb * 1024ULL * 1024ULL;
    opts.io_backend = io_b;
    opts.self_calibrate = 1;
    opts.disable_amx = no_amx;
    opts.use_o_direct = 1;
    opts.sidecar_dir = (sc_dir == "-" || sc_dir.empty()) ? nullptr : sc_dir.c_str();

    pm_handle_t* h = pm_open(dir.c_str(), &opts);
    if (!h) { std::cerr << "pm_open failed\n"; return 1; }
    pm_index_info_t info{};
    pm_get_info(h, &info);

    std::ofstream out(out_json);
    out << "{\n  \"index_dir\": \"" << dir << "\", \"threads\": " << nT
        << ", \"tier\": " << info.active_tier_mode
        << ", \"io_backend\": " << info.io_backend_active
        << ", \"amx\": " << info.amx_enabled
        << ", \"nprobe\": " << nprobe << ", \"c1_pool\": " << c1_pool
        << ", \"calibrated_tau1\": " << info.calibrated_tau1
        << ", \"calibrated_alpha_k0_ns\": " << std::fixed << std::setprecision(4) << info.calibrated_alpha_k0_ns
        << ",\n  \"rows\": [\n";

    bool first_row = true;
    for (const auto& wl : wls) {
        std::vector<roaring_bitmap_t*> rbms(wl.n_bm, nullptr);
        for (uint64_t b = 0; b < wl.n_bm; ++b) {
            rbms[b] = roaring_bitmap_create();
            const uint64_t* wp = wl.bitmaps.data() + b * nw;
            for (uint64_t w = 0; w < nw; ++w) {
                uint64_t bits = wp[w];
                uint32_t base = static_cast<uint32_t>(w * 64ULL);
                while (bits) {
                    uint32_t rid = base + static_cast<uint32_t>(__builtin_ctzll(bits));
                    if (rid < N_docs) roaring_bitmap_add(rbms[b], rid);
                    bits &= bits - 1ULL;
                }
            }
            roaring_bitmap_run_optimize(rbms[b]);
        }

        for (int s0a_mode = 0; s0a_mode < 2; ++s0a_mode) {
            bool disable_s0a = (s0a_mode == 0);
            uint64_t nq_eval = wl.n_q;
            std::vector<double> rec10(nq_eval, 0.0), t0_ms_v(nq_eval, 0.0);
            std::vector<int> all10(nq_eval, 0);
            std::atomic<uint64_t> tot_nvme_b{0}, tot_t2{0}, tot_s0a_fired{0}, tot_cb{0};

            double peak_pss = 0.0, peak_rss = 0.0, peak_hwm = 0.0;
            std::atomic<uint64_t> total_done{0};
            uint64_t io_before = 0;
            double t0_w = 0.0, wall_s = 0.0;
            std::vector<std::vector<double>> thr_lats(nT);

            omp_set_num_threads(nT);
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                // 1 full untimed warmup pass strictly BEFORE timing barrier (eliminates lazy-mmap cold-fault bias!)
                for (uint64_t qi = static_cast<uint64_t>(tid); qi < nq_eval; qi += static_cast<uint64_t>(nT)) {
                    pm_bitmap_view_t bmv{};
                    if (wl.sigma >= 0.9999) bmv.kind = PM_BM_NONE;
                    else { bmv.kind = PM_BM_ROARING_PTR; bmv.data = rbms[wl.q_to_bm[qi]]; }
                    pm_query_spec_t qs{};
                    qs.L = static_cast<uint32_t>(Lq); qs.q_vecs = Q32.data() + qi * Lq * d;
                    qs.agg_kind = PM_AGG_WSUM; qs.mode = PM_MODE_TOPK; qs.k = 10;
                    qs.delta = 0.05; qs.split_lambda = 0.2; qs.nprobe = nprobe; qs.c1_pool = c1_pool;
                    qs.disable_stage0a = disable_s0a ? 1 : 0;
                    pm_hit_t hits[10]; uint32_t out_n = 10; pm_query_stats_t qst{};
                    pm_execute_query(h, &bmv, nullptr, &qs, hits, &out_n, nullptr, nullptr, &qst);
                }
                #pragma omp barrier
                #pragma omp single
                {
                    io_before = proc_io_read_bytes();
                    t0_w = now_ns();
                }
                #pragma omp barrier

                uint64_t local_done = 0;
                auto& ll = thr_lats[tid];
                ll.reserve(128);
                for (uint64_t rep = 0; ; ++rep) {
                    for (uint64_t qi = static_cast<uint64_t>(tid); qi < nq_eval; qi += static_cast<uint64_t>(nT)) {
                        pm_bitmap_view_t bmv{};
                        if (wl.sigma >= 0.9999) bmv.kind = PM_BM_NONE;
                        else { bmv.kind = PM_BM_ROARING_PTR; bmv.data = rbms[wl.q_to_bm[qi]]; }

                        pm_query_spec_t qs{};
                        qs.L = static_cast<uint32_t>(Lq);
                        qs.q_vecs = Q32.data() + qi * Lq * d;
                        qs.agg_kind = PM_AGG_WSUM;
                        qs.mode = PM_MODE_TOPK;
                        qs.k = 10;
                        qs.delta = 0.05;
                        qs.split_lambda = 0.2;
                        qs.nprobe = nprobe;
                        qs.c1_pool = c1_pool;
                        qs.disable_stage0a = disable_s0a ? 1 : 0;

                        pm_hit_t hits[10]; uint32_t out_n = 10;
                        pm_query_stats_t qst{};
                        pm_execute_query(h, &bmv, nullptr, &qs, hits, &out_n, nullptr, nullptr, &qst);
                        local_done++;
                        ll.push_back(qst.t_total_ms);

                        if (rep == 0) {
                            t0_ms_v[qi] = qst.t_tier0_ms;
                            tot_nvme_b.fetch_add(qst.nvme_bytes_read, std::memory_order_relaxed);
                            tot_t2.fetch_add(qst.tier2_fetched, std::memory_order_relaxed);
                            tot_s0a_fired.fetch_add(qst.stage0a_used, std::memory_order_relaxed);
                            tot_cb.fetch_add(qst.cb_tripped, std::memory_order_relaxed);
                            int m = 0;
                            const uint32_t* gt = wl.gt10.data() + qi * 10ULL;
                            for (uint32_t i = 0; i < out_n; ++i)
                                for (int j = 0; j < 10; ++j) if (hits[i].rid == gt[j]) { m++; break; }
                            rec10[qi] = static_cast<double>(m) / 10.0;
                            all10[qi] = (m == 10) ? 1 : 0;
                        }
                    }
                    if (tid == 0) {
                        double p, r, hm; proc_mem_gb(p, r, hm);
                        if (p > peak_pss) { peak_pss = p; peak_rss = r; peak_hwm = hm; }
                    }
                    double el_s = (now_ns() - t0_w) * 1e-9;
                    if (el_s >= 2.5 && rep >= 1) break;
                }
                total_done.fetch_add(local_done, std::memory_order_relaxed);
                #pragma omp barrier
                #pragma omp single
                { wall_s = (now_ns() - t0_w) * 1e-9; }
            }
            uint64_t io_after = proc_io_read_bytes();
            double qps = static_cast<double>(total_done.load()) / std::max(1e-6, wall_s);
            std::vector<double> sl;
            for (const auto& v : thr_lats) sl.insert(sl.end(), v.begin(), v.end());
            std::sort(sl.begin(), sl.end());
            double p50 = sl.empty() ? 0.0 : sl[sl.size() / 2];
            double p99 = sl.empty() ? 0.0 : sl[std::min(sl.size() - 1, (sl.size() * 99) / 100)];
            double mean_ms = sl.empty() ? 0.0 : (std::accumulate(sl.begin(), sl.end(), 0.0) / static_cast<double>(sl.size()));
            double ll_ratio = (qps * mean_ms / 1000.0) / static_cast<double>(std::max(1, nT));
            double avg_t0 = std::accumulate(t0_ms_v.begin(), t0_ms_v.end(), 0.0) / nq_eval;
            double avg_r10 = std::accumulate(rec10.begin(), rec10.end(), 0.0) / nq_eval;
            double avg_a10 = std::accumulate(all10.begin(), all10.end(), 0.0) / nq_eval;
            double app_mb = (static_cast<double>(tot_nvme_b.load()) / nq_eval) / 1048576.0;
            double proc_mb = (static_cast<double>(io_after - io_before) / std::max<uint64_t>(1, total_done.load())) / 1048576.0;

            if (!first_row) out << ",\n";
            first_row = false;
            out << std::fixed
                << "    {\"workload\": \"" << wl.name << "\", \"corr\": \"" << wl.corr
                << "\", \"sigma\": " << std::setprecision(4) << wl.sigma
                << ", \"stage0a\": " << (disable_s0a ? "false" : "true")
                << ", \"qps\": " << std::setprecision(2) << qps
                << ", \"mean_ms\": " << std::setprecision(3) << mean_ms
                << ", \"p50_ms\": " << p50
                << ", \"p99_ms\": " << p99
                << ", \"littles_law\": " << ll_ratio
                << ", \"t_tier0_ms\": " << avg_t0
                << ", \"recall10\": " << std::setprecision(4) << avg_r10
                << ", \"all10\": " << avg_a10
                << ", \"app_mb_q\": " << std::setprecision(3) << app_mb
                << ", \"proc_mb_q\": " << proc_mb
                << ", \"t2_avg\": " << std::setprecision(2) << (static_cast<double>(tot_t2.load()) / nq_eval)
                << ", \"s0a_fired\": " << tot_s0a_fired.load()
                << ", \"cb_trips\": " << tot_cb.load()
                << ", \"active_pss_gb\": " << std::setprecision(4) << peak_pss
                << ", \"active_rss_gb\": " << peak_rss
                << ", \"vm_hwm_gb\": " << peak_hwm << "}";
            std::cout << "[" << wl.name << " s0a=" << (!disable_s0a) << "] QPS=" << std::fixed << std::setprecision(1) << qps
                      << " LL=" << std::setprecision(3) << ll_ratio
                      << " t_t0=" << std::setprecision(2) << avg_t0 << "ms R@10=" << std::setprecision(4) << avg_r10
                      << " All10=" << avg_a10 << " IO=" << std::setprecision(2) << app_mb << "MB/q s0a=" << tot_s0a_fired.load()
                      << " PSS=" << std::setprecision(3) << peak_pss << "GB\n" << std::flush;
        }
        (void)run_thresh;
        for (auto* r : rbms) roaring_bitmap_free(r);
    }
    out << "\n  ]\n}\n";
    pm_close(h);
    return 0;
}

// ============================================================================
// Mode 2: MS MARCO 8.84M Multi-Segment (seg0+seg1+seg2) Stage-0a ON vs OFF!
// ============================================================================
static int run_bench_lsm_stage0a(int argc, char** argv) {
    if (argc < 5) { std::cerr << "Usage: --bench-lsm-stage0a <msmarco_lsm_dir> <sidecar_root> <out_json> [--threads 48] [--tier dram|split|nvme]\n"; return 1; }
    std::string lsm_dir = argv[2], sc_root = argv[3], out_json = argv[4];
    int nT = 48;
    pm_tier_mode_t tier = PM_TIER_DRAM_T1;
    uint32_t nprobe = 16, c1_pool = 600;
    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) nT = std::stoi(argv[++i]);
        else if (a == "--tier" && i + 1 < argc) tier = parse_tier(argv[++i]);
        else if (a == "--nprobe" && i + 1 < argc) nprobe = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--c1" && i + 1 < argc) c1_pool = static_cast<uint32_t>(std::stoul(argv[++i]));
    }

    // Open seg0, seg1, seg2 via libpm_certvec.so (shared mmap frozen postings!)
    pm_handle_t* segs[3] = {nullptr, nullptr, nullptr};
    uint64_t seg_N[3] = {0, 0, 0}, seg_off[4] = {0, 0, 0, 0};
    for (int s = 0; s < 3; ++s) {
        pm_open_opts_t opts{};
        opts.tier_mode = tier;
        opts.io_backend = PM_IO_URING;
        opts.self_calibrate = 1;
        opts.use_o_direct = 1;
        std::string sdir = lsm_dir + "/seg" + std::to_string(s);
        std::string sc = sc_root + "/msmarco_seg" + std::to_string(s);
        opts.sidecar_dir = sc.c_str();
        segs[s] = pm_open(sdir.c_str(), &opts);
        if (!segs[s]) { std::cerr << "Failed open " << sdir << "\n"; return 1; }
        pm_index_info_t inf{}; pm_get_info(segs[s], &inf);
        seg_N[s] = inf.n_docs;
        seg_off[s + 1] = seg_off[s] + seg_N[s];
    }
    uint64_t N_global = seg_off[3];
    uint64_t nw_global = (N_global + 63ULL) / 64ULL;

    // Load queries from seg0/core_index.bin & global workloads from global_workloads_gt.bin
    uint64_t tmpN = 0, d = 128, Nq = 0, Lq = 32;
    std::vector<float> Q32;
    {
        std::ifstream f(lsm_dir + "/seg0/core_index.bin", std::ios::binary);
        uint64_t hdr[12]; f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        tmpN = hdr[1]; uint64_t K = hdr[2]; d = hdr[3]; Nq = hdr[4]; Lq = hdr[5]; (void)tmpN;
        f.seekg(static_cast<std::streamoff>(K * d * 4ULL + d * d * 4ULL), std::ios::cur);
        Q32.resize(Nq * Lq * d); f.read(reinterpret_cast<char*>(Q32.data()), Q32.size() * 4ULL);
    }
    std::vector<V4Workload> wls;
    {
        std::ifstream f(lsm_dir + "/global_workloads_gt.bin", std::ios::binary);
        if (!f) { std::cerr << "Missing global_workloads_gt.bin\n"; return 1; }
        uint64_t h[4]; f.read(reinterpret_cast<char*>(h), sizeof(h));
        uint64_t n_wl = h[1];
        uint64_t nw = (h[2] > 0) ? ((h[2] + 63ULL) / 64ULL) : nw_global;
        wls.resize(n_wl);
        for (uint64_t i = 0; i < n_wl; ++i) {
            char nb[64], cb[32]; f.read(nb, 64); f.read(cb, 32);
            wls[i].name = nb; wls[i].corr = cb;
            f.read(reinterpret_cast<char*>(&wls[i].sigma), 8);
            uint64_t qh[3]; f.read(reinterpret_cast<char*>(qh), sizeof(qh));
            wls[i].n_q = qh[0]; wls[i].n_bm = qh[1]; wls[i].actual_valid = qh[2];
            wls[i].bitmaps.resize(qh[1] * nw); f.read(reinterpret_cast<char*>(wls[i].bitmaps.data()), wls[i].bitmaps.size() * 8ULL);
            wls[i].q_to_bm.resize(qh[0]); f.read(reinterpret_cast<char*>(wls[i].q_to_bm.data()), qh[0] * 4ULL);
            wls[i].gt10.resize(qh[0] * 10ULL); f.read(reinterpret_cast<char*>(wls[i].gt10.data()), qh[0] * 10ULL * 4ULL);
        }
    }

    // Target key workloads spanning full selectivity 0.1%..100% + NegCorr + AdvDeadPost!
    std::unordered_set<std::string> keep_wls = {
        "NoCorr_0.10pct", "NoCorr_1.00pct", "NoCorr_5.00pct", "NoCorr_10.00pct",
        "NoCorr_30.00pct", "NoCorr_100.00pct", "Nat_TopicEq_20pct",
        "Nat_CrossTopicNeg_20pct", "Nat_Compound_NegAND_5pct", "AdvDeadPost_30.00pct"
    };

    std::ofstream out(out_json);
    out << "{\n  \"corpus\": \"msmarco_8m_lsm\", \"N_global\": " << N_global
        << ", \"threads\": " << nT << ", \"tier\": " << static_cast<int>(tier) << ",\n  \"rows\": [\n";
    bool first_row = true;

    for (const auto& wl : wls) {
        if (!keep_wls.count(wl.name)) continue;
        uint64_t nw_wl = wl.bitmaps.size() / std::max<uint64_t>(1, wl.n_bm);
        // Split global bitmap per segment into seg-local CRoaring bitmaps
        std::vector<std::array<roaring_bitmap_t*, 3>> seg_rbms(wl.n_bm);
        for (uint64_t b = 0; b < wl.n_bm; ++b) {
            for (int s = 0; s < 3; ++s) seg_rbms[b][s] = roaring_bitmap_create();
            if (wl.sigma < 0.9999) {
                const uint64_t* wp = wl.bitmaps.data() + b * nw_wl;
                for (uint64_t w = 0; w < nw_wl; ++w) {
                    uint64_t bits = wp[w];
                    uint64_t base = w * 64ULL;
                    while (bits) {
                        uint64_t gid = base + static_cast<uint64_t>(__builtin_ctzll(bits));
                        if (gid < N_global) {
                            int s = (gid < seg_off[1]) ? 0 : ((gid < seg_off[2]) ? 1 : 2);
                            roaring_bitmap_add(seg_rbms[b][s], static_cast<uint32_t>(gid - seg_off[s]));
                        }
                        bits &= bits - 1ULL;
                    }
                }
                for (int s = 0; s < 3; ++s) roaring_bitmap_run_optimize(seg_rbms[b][s]);
            }
        }

        for (int s0a_mode = 0; s0a_mode < 2; ++s0a_mode) {
            bool disable_s0a = (s0a_mode == 0);
            uint64_t nq_eval = wl.n_q;
            std::vector<double> rec10(nq_eval, 0.0), t0_ms_v(nq_eval, 0.0);
            std::vector<int> all10(nq_eval, 0);
            std::atomic<uint64_t> tot_nvme_b{0}, tot_s0a{0}, tot_cb{0}, total_done{0};
            uint64_t io_before = 0;
            double peak_pss = 0.0, peak_rss = 0.0, peak_hwm = 0.0, t0_w = 0.0, wall_s = 0.0;
            std::vector<std::vector<double>> thr_lats(nT);

            omp_set_num_threads(nT);
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                // 1 full untimed warmup pass strictly BEFORE timing barrier (eliminates DRAM cold-fault bias!)
                for (uint64_t qi = static_cast<uint64_t>(tid); qi < nq_eval; qi += static_cast<uint64_t>(nT)) {
                    for (int s = 0; s < 3; ++s) {
                        pm_bitmap_view_t bmv{};
                        if (wl.sigma >= 0.9999) bmv.kind = PM_BM_NONE;
                        else { bmv.kind = PM_BM_ROARING_PTR; bmv.data = seg_rbms[wl.q_to_bm[qi]][s]; }
                        pm_query_spec_t qs{};
                        qs.L = static_cast<uint32_t>(Lq); qs.q_vecs = Q32.data() + qi * Lq * d;
                        qs.agg_kind = PM_AGG_WSUM; qs.mode = PM_MODE_TOPK; qs.k = 10;
                        qs.delta = 0.05; qs.split_lambda = 0.2; qs.nprobe = nprobe;
                        qs.c1_pool = std::max<uint32_t>(200u, c1_pool / 2u);
                        qs.disable_stage0a = disable_s0a ? 1 : 0;
                        pm_hit_t sh[10]; uint32_t sn = 10; pm_query_stats_t qst{};
                        pm_execute_query(segs[s], &bmv, nullptr, &qs, sh, &sn, nullptr, nullptr, &qst);
                    }
                }
                #pragma omp barrier
                #pragma omp single
                {
                    io_before = proc_io_read_bytes();
                    t0_w = now_ns();
                }
                #pragma omp barrier
                uint64_t ldone = 0;
                auto& ll = thr_lats[tid];
                ll.reserve(128);
                for (uint64_t rep = 0; ; ++rep) {
                    for (uint64_t qi = static_cast<uint64_t>(tid); qi < nq_eval; qi += static_cast<uint64_t>(nT)) {
                        double qt0 = now_ns();
                        std::vector<pm_hit_t> merged; merged.reserve(30);
                        uint64_t q_io = 0, q_s0a = 0, q_cb = 0;
                        double q_t0 = 0.0;
                        for (int s = 0; s < 3; ++s) {
                            pm_bitmap_view_t bmv{};
                            if (wl.sigma >= 0.9999) bmv.kind = PM_BM_NONE;
                            else { bmv.kind = PM_BM_ROARING_PTR; bmv.data = seg_rbms[wl.q_to_bm[qi]][s]; }
                            pm_query_spec_t qs{};
                            qs.L = static_cast<uint32_t>(Lq);
                            qs.q_vecs = Q32.data() + qi * Lq * d;
                            qs.agg_kind = PM_AGG_WSUM;
                            qs.mode = PM_MODE_TOPK;
                            qs.k = 10;
                            qs.delta = 0.05;
                            qs.split_lambda = 0.2;
                            qs.nprobe = nprobe;
                            qs.c1_pool = std::max<uint32_t>(200u, c1_pool / 2u);
                            qs.disable_stage0a = disable_s0a ? 1 : 0;
                            pm_hit_t sh[10]; uint32_t sn = 10; pm_query_stats_t qst{};
                            pm_execute_query(segs[s], &bmv, nullptr, &qs, sh, &sn, nullptr, nullptr, &qst);
                            for (uint32_t i = 0; i < sn; ++i) merged.push_back({static_cast<uint32_t>(sh[i].rid + seg_off[s]), sh[i].score});
                            q_io += qst.nvme_bytes_read; q_s0a += qst.stage0a_used; q_cb += qst.cb_tripped; q_t0 += qst.t_tier0_ms;
                        }
                        std::sort(merged.begin(), merged.end(), [](const pm_hit_t& a, const pm_hit_t& b) {
                            if (a.score != b.score) return a.score > b.score;
                            return a.rid < b.rid;
                        });
                        double qms = (now_ns() - qt0) * 1e-6;
                        ldone++;
                        ll.push_back(qms);
                        if (rep == 0) {
                            t0_ms_v[qi] = q_t0;
                            tot_nvme_b.fetch_add(q_io, std::memory_order_relaxed);
                            tot_s0a.fetch_add(q_s0a, std::memory_order_relaxed);
                            tot_cb.fetch_add(q_cb, std::memory_order_relaxed);
                            int m = 0; const uint32_t* gt = wl.gt10.data() + qi * 10ULL;
                            size_t lim = std::min<size_t>(10, merged.size());
                            for (size_t i = 0; i < lim; ++i) for (int j = 0; j < 10; ++j) if (merged[i].rid == gt[j]) { m++; break; }
                            rec10[qi] = static_cast<double>(m) / 10.0;
                            all10[qi] = (m == 10) ? 1 : 0;
                        }
                    }
                    if (tid == 0) {
                        double p, r, hm; proc_mem_gb(p, r, hm);
                        if (p > peak_pss) { peak_pss = p; peak_rss = r; peak_hwm = hm; }
                    }
                    if ((now_ns() - t0_w) * 1e-9 >= 3.0 && rep >= 1) break;
                }
                total_done.fetch_add(ldone, std::memory_order_relaxed);
                #pragma omp barrier
                #pragma omp single
                { wall_s = (now_ns() - t0_w) * 1e-9; }
            }
            uint64_t io_after = proc_io_read_bytes();
            double qps = static_cast<double>(total_done.load()) / std::max(1e-6, wall_s);
            std::vector<double> sl;
            for (const auto& v : thr_lats) sl.insert(sl.end(), v.begin(), v.end());
            std::sort(sl.begin(), sl.end());
            double p50 = sl.empty() ? 0.0 : sl[sl.size() / 2];
            double p99 = sl.empty() ? 0.0 : sl[std::min(sl.size() - 1, (sl.size() * 99) / 100)];
            double mean_ms = sl.empty() ? 0.0 : (std::accumulate(sl.begin(), sl.end(), 0.0) / static_cast<double>(sl.size()));
            double ll_ratio = (qps * mean_ms / 1000.0) / static_cast<double>(std::max(1, nT));
            double avg_t0 = std::accumulate(t0_ms_v.begin(), t0_ms_v.end(), 0.0) / nq_eval;
            double avg_r10 = std::accumulate(rec10.begin(), rec10.end(), 0.0) / nq_eval;
            double avg_a10 = std::accumulate(all10.begin(), all10.end(), 0.0) / nq_eval;
            double app_mb = (static_cast<double>(tot_nvme_b.load()) / nq_eval) / 1048576.0;
            double proc_mb = (static_cast<double>(io_after - io_before) / std::max<uint64_t>(1, total_done.load())) / 1048576.0;

            if (!first_row) out << ",\n";
            first_row = false;
            out << std::fixed
                << "    {\"workload\": \"" << wl.name << "\", \"sigma\": " << std::setprecision(4) << wl.sigma
                << ", \"stage0a\": " << (disable_s0a ? "false" : "true")
                << ", \"qps\": " << std::setprecision(2) << qps
                << ", \"mean_ms\": " << std::setprecision(3) << mean_ms
                << ", \"p50_ms\": " << p50 << ", \"p99_ms\": " << p99
                << ", \"littles_law\": " << ll_ratio
                << ", \"t_tier0_ms\": " << avg_t0
                << ", \"recall10\": " << std::setprecision(4) << avg_r10 << ", \"all10\": " << avg_a10
                << ", \"app_mb_q\": " << std::setprecision(3) << app_mb << ", \"proc_mb_q\": " << proc_mb
                << ", \"s0a_fired\": " << tot_s0a.load() << ", \"cb_trips\": " << tot_cb.load()
                << ", \"active_pss_gb\": " << std::setprecision(4) << peak_pss << ", \"active_rss_gb\": " << peak_rss << ", \"vm_hwm_gb\": " << peak_hwm << "}";
            std::cout << "[MSMARCO-8.84M " << wl.name << " s0a=" << (!disable_s0a) << "] QPS=" << std::fixed << std::setprecision(2) << qps
                      << " LL=" << std::setprecision(3) << ll_ratio
                      << " t_t0=" << std::setprecision(2) << avg_t0 << "ms R@10=" << std::setprecision(4) << avg_r10
                      << " IO=" << std::setprecision(2) << app_mb << "MB/q s0a=" << tot_s0a.load()
                      << " PSS=" << std::setprecision(2) << peak_pss << "GB\n" << std::flush;
        }
        for (uint64_t b = 0; b < wl.n_bm; ++b) for (int s = 0; s < 3; ++s) roaring_bitmap_free(seg_rbms[b][s]);
    }
    out << "\n  ]\n}\n";
    for (int s = 0; s < 3; ++s) pm_close(segs[s]);
    return 0;
}

// ============================================================================
// Mode 4: Zipf(s=1.0) Multi-Tenant 50K–100K Mixed Read+Write Stream (Theorem 5 Unified S3-FIFO Cache!)
// ============================================================================
static int run_bench_zipf_cache(int argc, char** argv) {
    if (argc < 10) {
        std::cerr << "Usage: --bench-zipf-cache <v4_dir> <sidecar_dir> <queries_raw_f32> <Nq> <Lq> <stream_qids_u32> <stream_tids_u16> <n_reqs> <out_json> [--threads 48] [--tier auto|split|nvme|dram] [--ram-mb 1024] [--mutate-pct 2]\n";
        return 1;
    }
    std::string dir = argv[2], sc_dir = argv[3], q_path = argv[4];
    uint64_t Nq = std::stoull(argv[5]), Lq = std::stoull(argv[6]);
    std::string qids_path = argv[7], tids_path = argv[8];
    uint64_t n_reqs = std::stoull(argv[9]);
    std::string out_json = argv[10];
    int nT = 48;
    pm_tier_mode_t tier = PM_TIER_AUTO_KKT;
    size_t ram_mb = 1024;
    double mutate_pct = 2.0;
    for (int i = 11; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) nT = std::stoi(argv[++i]);
        else if (a == "--tier" && i + 1 < argc) tier = parse_tier(argv[++i]);
        else if (a == "--ram-mb" && i + 1 < argc) ram_mb = std::stoull(argv[++i]);
        else if (a == "--mutate-pct" && i + 1 < argc) mutate_pct = std::stod(argv[++i]);
    }

    pm_open_opts_t opts{};
    opts.tier_mode = tier;
    opts.ram_budget_bytes = ram_mb * 1024ULL * 1024ULL;
    opts.io_backend = PM_IO_URING;
    opts.self_calibrate = 1;
    opts.use_o_direct = 1;
    opts.sidecar_dir = (sc_dir == "-" || sc_dir.empty()) ? nullptr : sc_dir.c_str();
    pm_handle_t* h = pm_open(dir.c_str(), &opts);
    if (!h) { std::cerr << "pm_open failed\n"; return 1; }
    pm_index_info_t info{}; pm_get_info(h, &info);
    uint64_t N_docs = info.n_docs, d = info.dims[0];

    std::vector<float> Q(Nq * Lq * d);
    { std::ifstream f(q_path, std::ios::binary); f.read(reinterpret_cast<char*>(Q.data()), Q.size() * 4ULL); }
    std::vector<uint32_t> req_q(n_reqs);
    { std::ifstream f(qids_path, std::ios::binary); f.read(reinterpret_cast<char*>(req_q.data()), n_reqs * 4ULL); }
    std::vector<uint16_t> req_t(n_reqs);
    { std::ifstream f(tids_path, std::ios::binary); f.read(reinterpret_cast<char*>(req_t.data()), n_reqs * 2ULL); }

    // 20 disjoint tenants (each ~5% selectivity; include [N_docs, N_docs+5000) so inserted LSM delta entities are genuinely scored!)
    std::vector<roaring_bitmap_t*> tenant_bm(20, nullptr);
    for (int t = 0; t < 20; ++t) tenant_bm[t] = roaring_bitmap_create();
    for (uint64_t rid = 0; rid < N_docs + 5000ULL; ++rid) roaring_bitmap_add(tenant_bm[rid % 20], static_cast<uint32_t>(rid));
    for (int t = 0; t < 20; ++t) roaring_bitmap_run_optimize(tenant_bm[t]);

    std::atomic<uint64_t> tot_nvme_b{0}, tot_hb4{0}, tot_ht2{0}, tot_miss{0}, tot_ins{0}, tot_del{0};
    std::vector<double> lats(n_reqs, 0.0);
    uint64_t io_before = proc_io_read_bytes();
    double peak_pss = 0.0, peak_rss = 0.0, peak_hwm = 0.0;
    double t0_w = 0.0, wall_s = 0.0;

    std::vector<float> synth_vec(32ULL * d, 0.05f);
    omp_set_num_threads(nT);
    #pragma omp parallel
    {
        #pragma omp barrier
        #pragma omp single
        { t0_w = now_ns(); }
        #pragma omp barrier

        #pragma omp for schedule(dynamic, 32)
        for (uint64_t ri = 0; ri < n_reqs; ++ri) {
            uint64_t mod_m = (mutate_pct > 0.0) ? static_cast<uint64_t>(100.0 / mutate_pct) : 0ULL;
            if (mod_m > 0 && (ri % mod_m) == 0) {
                uint32_t ext_id = static_cast<uint32_t>(N_docs + (ri % 5000));
                if ((ri / mod_m) % 2 == 0) {
                    pm_field_vecs_t fv{0, 32, synth_vec.data()};
                    pm_entity_vecs_t ev{1, &fv};
                    pm_insert_entity(h, ext_id, &ev);
                    tot_ins.fetch_add(1, std::memory_order_relaxed);
                } else {
                    pm_delete_entity(h, ext_id);
                    tot_del.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                uint32_t qi = req_q[ri] % static_cast<uint32_t>(Nq);
                uint16_t ten = req_t[ri] % 20u;
                pm_bitmap_view_t bmv{PM_BM_ROARING_PTR, tenant_bm[ten], 0};
                pm_query_spec_t qs{};
                qs.L = static_cast<uint32_t>(Lq);
                qs.q_vecs = Q.data() + static_cast<size_t>(qi) * Lq * d;
                qs.agg_kind = PM_AGG_WSUM;
                qs.mode = PM_MODE_TOPK;
                qs.k = 10;
                qs.delta = 0.05;
                qs.split_lambda = 0.2;
                qs.nprobe = 16;
                qs.c1_pool = 600;
                pm_hit_t hits[10]; uint32_t out_n = 10; pm_query_stats_t qst{};
                pm_execute_query(h, &bmv, nullptr, &qs, hits, &out_n, nullptr, nullptr, &qst);
                lats[ri] = qst.t_total_ms;
                tot_nvme_b.fetch_add(qst.nvme_bytes_read, std::memory_order_relaxed);
                tot_hb4.fetch_add(qst.cache_hits_b4, std::memory_order_relaxed);
                tot_ht2.fetch_add(qst.cache_hits_t2, std::memory_order_relaxed);
                tot_miss.fetch_add(qst.cache_misses, std::memory_order_relaxed);
            }
            if ((ri & 1023ULL) == 0) {
                double p, r, hm; proc_mem_gb(p, r, hm);
                #pragma omp critical
                { if (p > peak_pss) { peak_pss = p; peak_rss = r; peak_hwm = hm; } }
            }
        }
        #pragma omp barrier
        #pragma omp single
        { wall_s = (now_ns() - t0_w) * 1e-9; }
    }
    uint64_t io_after = proc_io_read_bytes();
    std::vector<double> valid_lats; valid_lats.reserve(n_reqs);
    for (double v : lats) if (v > 0.0) valid_lats.push_back(v);
    std::sort(valid_lats.begin(), valid_lats.end());
    double p50 = valid_lats.empty() ? 0.0 : valid_lats[valid_lats.size() / 2];
    double p99 = valid_lats.empty() ? 0.0 : valid_lats[(valid_lats.size() * 99) / 100];
    double mean_ms = valid_lats.empty() ? 0.0 : (std::accumulate(valid_lats.begin(), valid_lats.end(), 0.0) / static_cast<double>(valid_lats.size()));
    double qps = static_cast<double>(n_reqs) / std::max(1e-6, wall_s);
    double ll_ratio = (qps * mean_ms / 1000.0) / static_cast<double>(std::max(1, nT));
    uint64_t h_tot = tot_hb4.load() + tot_ht2.load();
    double hit_rate = (h_tot + tot_miss.load() > 0) ? (static_cast<double>(h_tot) / static_cast<double>(h_tot + tot_miss.load())) : 0.0;
    double proc_mb_q = (static_cast<double>(io_after - io_before) / std::max<uint64_t>(1, valid_lats.size())) / 1048576.0;

    std::ofstream out(out_json);
    out << "{\n  \"mode\": \"zipf_mixed_stream\", \"n_reqs\": " << n_reqs
        << ", \"threads\": " << nT << ", \"ram_budget_mb\": " << ram_mb
        << ", \"active_tier\": " << info.active_tier_mode
        << ", \"cache_ram_mb\": " << std::fixed << std::setprecision(2) << (info.cache_ram_bytes / 1048576.0)
        << ", \"qps\": " << qps << ", \"mean_ms\": " << std::setprecision(3) << mean_ms
        << ", \"p50_ms\": " << p50 << ", \"p99_ms\": " << p99 << ", \"littles_law\": " << ll_ratio
        << ", \"cache_hit_rate\": " << std::setprecision(4) << hit_rate
        << ", \"cache_hits_b4\": " << tot_hb4.load() << ", \"cache_hits_t2\": " << tot_ht2.load()
        << ", \"cache_misses\": " << tot_miss.load()
        << ", \"proc_mb_q\": " << std::setprecision(3) << proc_mb_q
        << ", \"inserts\": " << tot_ins.load() << ", \"deletes\": " << tot_del.load()
        << ", \"active_pss_gb\": " << std::setprecision(4) << peak_pss
        << ", \"active_rss_gb\": " << peak_rss << ", \"vm_hwm_gb\": " << peak_hwm << "\n}\n";
    std::cout << "[Zipf " << n_reqs << " reqs ram=" << ram_mb << "MB] QPS=" << std::fixed << std::setprecision(1) << qps
              << " P99=" << std::setprecision(2) << p99 << "ms hit_rate=" << std::setprecision(4) << hit_rate
              << " (B4=" << tot_hb4.load() << ", T2=" << tot_ht2.load() << ") IO=" << std::setprecision(2) << proc_mb_q
              << "MB/q PSS=" << std::setprecision(3) << peak_pss << "GB\n";
    for (auto* r : tenant_bm) roaring_bitmap_free(r);
    pm_close(h);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: certvec_bench_v4 [--bench-stage0a | --bench-lsm-stage0a | --bench-zipf-cache] ...\n"; return 1; }
    std::string mode = argv[1];
    if (mode == "--bench-stage0a") return run_bench_stage0a(argc, argv);
    if (mode == "--bench-lsm-stage0a") return run_bench_lsm_stage0a(argc, argv);
    if (mode == "--bench-zipf-cache") return run_bench_zipf_cache(argc, argv);
    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}
