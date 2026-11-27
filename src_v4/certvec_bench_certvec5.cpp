// certvec_bench_certvec5.cpp — 48-Core Exclusive Benchmark Binary for Native Multi-Field CERTVEC5 Corpora:
//   - Corpus #5: `msmarco_docs_768d` (W1 Parent-Doc Max & Top-3 Mean, W3 Multi-Field Hybrid BM25+Top3+Title, W4 Q-Thresh)
//   - Corpus #6: `bgem3_1024d_hybrid` (W2 High-d 1024D MaxSim, W3 Dynamic Hybrid Weights 1 & 2, W4 Q-Thresh)
// Reads simple binary eval bundle exported by `export_certvec5_eval_bin.py`, runs full 48-core OpenMP barrier-to-barrier
// steady-state benchmarks across {PM_NVME_T1, Split_B1_B4_T3, PM_DRAM_T1, Approx_B4} with exact /proc/self/io & PSS!

#include "pm_certvec.h"
#include "pm_kernels.hpp"
#include "roaring.h"

#include <algorithm>
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
    while (std::getline(f, line)) if (line.rfind("read_bytes:", 0) == 0) return std::stoull(line.substr(11));
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

struct EvalWorkload {
    std::string name;
    double sigma = 1.0;
    roaring_bitmap_t* bm = nullptr; // NULL if sigma == 1.0
    uint32_t nq = 0;
    // Per query: 3 GT10 arrays (task0, task1, task2) + w4_taus[nq] + w4_true_counts[nq]
    std::vector<uint32_t> gt0, gt1, gt2;
    std::vector<double> w4_taus;
    std::vector<uint32_t> w4_true_pos_offs;
    std::vector<uint32_t> w4_true_pos_rids;
    ~EvalWorkload() { if (bm) roaring_bitmap_free(bm); }
};

static int thresh_cb_collect(void* ctx, uint32_t rid, double sc) {
    static_cast<std::vector<pm_hit_t>*>(ctx)->push_back({rid, sc});
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: certvec_bench_certvec5 <idx_dir> <eval_bin> <kind:docs768|bgem3> <tier:nvme|split|dram|auto> <out_json> [--threads 48] [--ram-mb 0] [--nprobe 16] [--c1 600]\n";
        return 1;
    }
    std::string idx_dir = argv[1], eval_bin = argv[2], kind = argv[3], tier_s = argv[4], out_json = argv[5];
    int nT = 48;
    size_t ram_mb = 0;
    uint32_t nprobe = 16, c1_pool = 600;
    for (int i = 6; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) nT = std::stoi(argv[++i]);
        else if (a == "--ram-mb" && i + 1 < argc) ram_mb = std::stoull(argv[++i]);
        else if (a == "--nprobe" && i + 1 < argc) nprobe = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--c1" && i + 1 < argc) c1_pool = static_cast<uint32_t>(std::stoul(argv[++i]));
    }
    pm_tier_mode_t tier = PM_TIER_SPLIT_B1B4;
    if (tier_s == "nvme") tier = PM_TIER_NVME_T1;
    else if (tier_s == "dram") tier = PM_TIER_DRAM_T1;
    else if (tier_s == "auto") tier = PM_TIER_AUTO_KKT;

    pm_open_opts_t opts{};
    opts.tier_mode = tier;
    opts.ram_budget_bytes = ram_mb * 1024ULL * 1024ULL;
    opts.io_backend = PM_IO_URING;
    opts.self_calibrate = 1;
    opts.use_o_direct = 1;
    pm_handle_t* h = pm_open(idx_dir.c_str(), &opts);
    if (!h) { std::cerr << "pm_open failed on " << idx_dir << "\n"; return 1; }
    pm_index_info_t info{}; pm_get_info(h, &info);

    // Load flat binary eval bundle exported by export_certvec5_eval_bin.py
    std::ifstream ef(eval_bin, std::ios::binary);
    if (!ef) { std::cerr << "Cannot open eval bundle " << eval_bin << "\n"; return 1; }
    uint64_t hdr[8]; ef.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    // hdr: [magic, N_docs, Nq, d0, d1, L_max_f0, n_workloads, total_sparse_pairs]
    uint64_t N_docs = hdr[1], Nq = hdr[2], d0 = hdr[3], d1 = hdr[4], L0 = hdr[5], n_wls = hdr[6], n_sp = hdr[7];
    std::vector<float> q_f0(Nq * L0 * d0), q_f1(Nq * d1);
    std::vector<uint32_t> q_f0_lens(Nq);
    ef.read(reinterpret_cast<char*>(q_f0.data()), q_f0.size() * 4ULL);
    ef.read(reinterpret_cast<char*>(q_f1.data()), q_f1.size() * 4ULL);
    ef.read(reinterpret_cast<char*>(q_f0_lens.data()), Nq * 4ULL);

    std::vector<uint64_t> sp_offs(Nq + 1);
    std::vector<uint32_t> sp_rids(n_sp);
    std::vector<float>    sp_scores(n_sp);
    ef.read(reinterpret_cast<char*>(sp_offs.data()), (Nq + 1) * 8ULL);
    if (n_sp > 0) {
        ef.read(reinterpret_cast<char*>(sp_rids.data()), n_sp * 4ULL);
        ef.read(reinterpret_cast<char*>(sp_scores.data()), n_sp * 4ULL);
    }

    std::vector<EvalWorkload> wls(n_wls);
    for (uint64_t w = 0; w < n_wls; ++w) {
        char nb[64]; ef.read(nb, 64); wls[w].name = nb;
        ef.read(reinterpret_cast<char*>(&wls[w].sigma), 8);
        uint64_t m_sz = 0; ef.read(reinterpret_cast<char*>(&m_sz), 8);
        std::vector<uint32_t> valid_rids(m_sz);
        if (m_sz > 0) ef.read(reinterpret_cast<char*>(valid_rids.data()), m_sz * 4ULL);
        if (wls[w].sigma < 0.9999) {
            wls[w].bm = roaring_bitmap_of_ptr(static_cast<size_t>(m_sz), valid_rids.data());
            roaring_bitmap_run_optimize(wls[w].bm);
        }
        wls[w].nq = static_cast<uint32_t>(Nq);
        wls[w].gt0.resize(Nq * 10ULL); wls[w].gt1.resize(Nq * 10ULL); wls[w].gt2.resize(Nq * 10ULL);
        wls[w].w4_taus.resize(Nq);
        ef.read(reinterpret_cast<char*>(wls[w].gt0.data()), Nq * 10ULL * 4ULL);
        ef.read(reinterpret_cast<char*>(wls[w].gt1.data()), Nq * 10ULL * 4ULL);
        ef.read(reinterpret_cast<char*>(wls[w].gt2.data()), Nq * 10ULL * 4ULL);
        ef.read(reinterpret_cast<char*>(wls[w].w4_taus.data()), Nq * 8ULL);
    }

    // Define the 4 tasks per corpus!
    // docs768: 0=W1_ParentDoc_Max, 1=W1_ParentDoc_Top3Mean, 2=W3_Hybrid_BM25_Top3_Title, 3=W4_QThresh_Max
    // bgem3:   0=W2_ColBERT_1024D, 1=W3_Hybrid_0.2S_0.4D_0.4C, 2=W3_Hybrid_0.5S_0.2D_0.3C, 3=W4_QThresh_ColBERT1024
    const char* task_names_docs[4]  = {"W1_ParentDoc_Max", "W1_ParentDoc_Top3Mean", "W3_Hybrid_BM25_Top3_Title", "W4_QThresh_Max"};
    const char* task_names_bgem3[4] = {"W2_ColBERT_1024D", "W3_Hybrid_0.2S_0.4D_0.4C", "W3_Hybrid_0.5S_0.2D_0.3C", "W4_QThresh_ColBERT1024"};
    const char** task_names = (kind == "docs768") ? task_names_docs : task_names_bgem3;

    std::ofstream out(out_json);
    out << "{\n  \"corpus\": \"" << kind << "\", \"N_docs\": " << N_docs
        << ", \"d0\": " << d0 << ", \"d1\": " << d1
        << ", \"threads\": " << nT << ", \"tier\": \"" << tier_s << "\""
        << ", \"active_tier_mode\": " << info.active_tier_mode
        << ", \"ram_mb\": " << ram_mb << ", \"tier0_mb\": " << std::fixed << std::setprecision(2) << (info.tier0_ram_bytes / 1048576.0)
        << ", \"b1_mb\": " << (info.b1_ram_bytes / 1048576.0)
        << ", \"b4_mb\": " << (info.b4_ram_bytes / 1048576.0)
        << ", \"full_index_ram_mb\": " << ((info.tier0_ram_bytes + info.b1_ram_bytes + info.b4_ram_bytes) / 1048576.0)
        << ", \"cache_mb\": " << (info.cache_ram_bytes / 1048576.0)
        << ",\n  \"rows\": [\n";
    bool first_row = true;

    for (const auto& wl : wls) {
        for (int task = 0; task < 4; ++task) {
            // Also run uncertified approx ablation on task 0!
            int n_variants = (task == 0) ? 2 : 1;
            for (int var = 0; var < n_variants; ++var) {
                bool approx = (var == 1);
                std::string mlabel = approx ? "Approx_B4_Uncert" : ("PM_" + tier_s);
                bool is_thresh = (task == 3);
                uint32_t nq_eval = wl.nq;
                std::vector<double> rec(nq_eval, 0.0), prec(nq_eval, 1.0);
                std::vector<int> all10(nq_eval, 0);
                std::atomic<uint64_t> tot_io{0}, tot_c1{0}, tot_t2{0}, tot_emitted{0}, tot_fp{0}, total_done{0};
                uint64_t io_before = 0;
                double peak_pss = 0.0, peak_rss = 0.0, peak_hwm = 0.0, t0_w = 0.0, wall_s = 0.0;
                std::vector<std::vector<double>> thr_lats(nT);

                auto run_one_qi = [&](uint32_t qi, bool record_metrics, std::vector<double>* lat_out) {
                    pm_bitmap_view_t bmv{};
                    if (!wl.bm) bmv.kind = PM_BM_NONE;
                    else { bmv.kind = PM_BM_ROARING_PTR; bmv.data = wl.bm; }

                    pm_aux_scores_t aux{};
                    aux.kind = PM_AUX_NONE;
                    std::vector<float> q_buf;
                    std::vector<uint32_t> fids, rvals;
                    std::vector<pm_set_op_t> sops;
                    std::vector<float> wts;

                    if (kind == "docs768") {
                        const float* qv = q_f0.data() + static_cast<size_t>(qi) * d0;
                        if (task == 0 || task == 3) {
                            q_buf.resize(d0); std::memcpy(q_buf.data(), qv, d0 * sizeof(float));
                            fids = {0}; sops = {PM_OP_MAX}; rvals = {1}; wts = {1.0f};
                        } else if (task == 1) {
                            q_buf.resize(d0); std::memcpy(q_buf.data(), qv, d0 * sizeof(float));
                            fids = {0}; sops = {PM_OP_TOP_R_MEAN}; rvals = {3}; wts = {1.0f};
                        } else {
                            q_buf.resize(d0 + d1);
                            std::memcpy(q_buf.data(), qv, d0 * 4ULL);
                            std::memcpy(q_buf.data() + d0, q_f1.data() + static_cast<size_t>(qi) * d1, d1 * 4ULL);
                            fids = {0, 1}; sops = {PM_OP_TOP_R_MEAN, PM_OP_MAX}; rvals = {3, 1}; wts = {0.6f, 0.4f};
                            aux.kind = PM_AUX_SPARSE_PAIRS;
                            aux.rids = sp_rids.data() + sp_offs[qi];
                            aux.scores = sp_scores.data() + sp_offs[qi];
                            aux.count = static_cast<size_t>(sp_offs[qi + 1] - sp_offs[qi]);
                            aux.w0 = 0.02f;
                        }
                    } else {
                        uint32_t ql = std::max<uint32_t>(1u, q_f0_lens[qi]);
                        const float* qc = q_f0.data() + static_cast<size_t>(qi) * L0 * d0;
                        const float* qd = q_f1.data() + static_cast<size_t>(qi) * d1;
                        if (task == 0 || task == 3) {
                            q_buf.resize(static_cast<size_t>(ql) * d0);
                            std::memcpy(q_buf.data(), qc, q_buf.size() * sizeof(float));
                            fids.assign(ql, 0u); sops.assign(ql, PM_OP_MAX); rvals.assign(ql, 1u); wts.assign(ql, 1.0f);
                        } else {
                            float w_sp = (task == 1) ? 0.2f : 0.5f;
                            float w_dn = (task == 1) ? 0.4f : 0.2f;
                            float w_cb = ((task == 1) ? 0.4f : 0.3f) / static_cast<float>(ql);
                            q_buf.resize(static_cast<size_t>(ql) * d0 + d1);
                            std::memcpy(q_buf.data(), qc, static_cast<size_t>(ql) * d0 * 4ULL);
                            std::memcpy(q_buf.data() + static_cast<size_t>(ql) * d0, qd, d1 * 4ULL);
                            fids.assign(ql + 1, 0u); fids[ql] = 1u;
                            sops.assign(ql + 1, PM_OP_MAX);
                            rvals.assign(ql + 1, 1u);
                            wts.assign(ql + 1, w_cb); wts[ql] = w_dn;
                            aux.kind = PM_AUX_SPARSE_PAIRS;
                            aux.rids = sp_rids.data() + sp_offs[qi];
                            aux.scores = sp_scores.data() + sp_offs[qi];
                            aux.count = static_cast<size_t>(sp_offs[qi + 1] - sp_offs[qi]);
                            aux.w0 = w_sp;
                        }
                    }

                    pm_query_spec_t qs{};
                    qs.L = static_cast<uint32_t>(fids.size());
                    qs.q_vecs = q_buf.data();
                    qs.field_ids = fids.data();
                    qs.set_ops = sops.data();
                    qs.r_vals = rvals.data();
                    qs.weights = wts.data();
                    qs.agg_kind = PM_AGG_WSUM;
                    qs.mode = is_thresh ? PM_MODE_THRESHOLD : PM_MODE_TOPK;
                    qs.k = 10;
                    qs.tau = wl.w4_taus[qi];
                    qs.delta = 0.05;
                    qs.split_lambda = 0.2;
                    qs.nprobe = nprobe;
                    qs.c1_pool = c1_pool;
                    qs.uncertified_approx = approx ? 1 : 0;
                    if (is_thresh && wl.sigma <= 0.15) qs.force_path1 = 1;

                    pm_query_stats_t qst{};
                    if (is_thresh) {
                        std::vector<pm_hit_t> th;
                        uint32_t out_n = 0;
                        pm_execute_query(h, &bmv, &aux, &qs, nullptr, &out_n, thresh_cb_collect, &th, &qst);
                        if (lat_out) lat_out->push_back(qst.t_total_ms);
                        if (record_metrics) {
                            tot_io.fetch_add(qst.nvme_bytes_read, std::memory_order_relaxed);
                            tot_c1.fetch_add(qst.c1_evaluated, std::memory_order_relaxed);
                            tot_t2.fetch_add(qst.tier2_fetched, std::memory_order_relaxed);
                            tot_emitted.fetch_add(th.size(), std::memory_order_relaxed);
                            uint64_t fp = 0;
                            for (const auto& hit : th) if (hit.score < qs.tau) fp++;
                            tot_fp.fetch_add(fp, std::memory_order_relaxed);
                            const uint32_t* gt = wl.gt0.data() + static_cast<size_t>(qi) * 10ULL;
                            int m = 0;
                            for (int j = 0; j < 10; ++j)
                                for (const auto& hit : th) if (hit.rid == gt[j]) { m++; break; }
                            rec[qi] = static_cast<double>(m) / 10.0;
                            prec[qi] = th.empty() ? 1.0 : (1.0 - static_cast<double>(fp) / th.size());
                        }
                    } else {
                        pm_hit_t hits[10]; uint32_t out_n = 10;
                        pm_execute_query(h, &bmv, &aux, &qs, hits, &out_n, nullptr, nullptr, &qst);
                        if (lat_out) lat_out->push_back(qst.t_total_ms);
                        if (record_metrics) {
                            tot_io.fetch_add(qst.nvme_bytes_read, std::memory_order_relaxed);
                            tot_c1.fetch_add(qst.c1_evaluated, std::memory_order_relaxed);
                            tot_t2.fetch_add(qst.tier2_fetched, std::memory_order_relaxed);
                            const uint32_t* gt = (task == 0 ? wl.gt0.data() : (task == 1 ? wl.gt1.data() : wl.gt2.data()))
                                               + static_cast<size_t>(qi) * 10ULL;
                            int m = 0;
                            for (uint32_t i = 0; i < out_n; ++i)
                                for (int j = 0; j < 10; ++j) if (hits[i].rid == gt[j]) { m++; break; }
                            rec[qi] = static_cast<double>(m) / 10.0;
                            all10[qi] = (m == 10) ? 1 : 0;
                        }
                    }
                };

                omp_set_num_threads(nT);
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    // 1 full untimed warmup pass BEFORE timing barrier (eliminates lazy-mmap cold-fault bias!)
                    for (uint32_t qi = static_cast<uint32_t>(tid); qi < nq_eval; qi += static_cast<uint32_t>(nT)) {
                        run_one_qi(qi, false, nullptr);
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
                    ll.reserve(256);
                    for (uint64_t rep = 0; ; ++rep) {
                        for (uint32_t qi = static_cast<uint32_t>(tid); qi < nq_eval; qi += static_cast<uint32_t>(nT)) {
                            run_one_qi(qi, rep == 0, &ll);
                            ldone++;
                        }
                        if (tid == 0) {
                            double p, r, hm; proc_mem_gb(p, r, hm);
                            if (p > peak_pss) { peak_pss = p; peak_rss = r; peak_hwm = hm; }
                        }
                        if ((now_ns() - t0_w) * 1e-9 >= 2.5 && rep >= 1) break;
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
                double avg_rec = std::accumulate(rec.begin(), rec.end(), 0.0) / nq_eval;
                double avg_a10 = std::accumulate(all10.begin(), all10.end(), 0.0) / nq_eval;
                double avg_prec = std::accumulate(prec.begin(), prec.end(), 0.0) / nq_eval;
                double app_mb = (static_cast<double>(tot_io.load()) / nq_eval) / 1048576.0;
                double proc_mb = (static_cast<double>(io_after - io_before) / std::max<uint64_t>(1, total_done.load())) / 1048576.0;
                double avg_c1 = static_cast<double>(tot_c1.load()) / nq_eval;
                double avg_t2 = static_cast<double>(tot_t2.load()) / nq_eval;
                double t2_prune = (avg_c1 > 0.0) ? std::max(0.0, 1.0 - avg_t2 / avg_c1) : 0.0;

                if (!first_row) out << ",\n";
                first_row = false;
                out << std::fixed
                    << "    {\"workload\": \"" << wl.name << "\", \"sigma\": " << std::setprecision(4) << wl.sigma
                    << ", \"task\": \"" << task_names[task] << "\", \"method\": \"" << mlabel << "\""
                    << ", \"qps\": " << std::setprecision(2) << qps
                    << ", \"mean_ms\": " << std::setprecision(3) << mean_ms
                    << ", \"p50_ms\": " << p50 << ", \"p99_ms\": " << p99
                    << ", \"littles_law\": " << ll_ratio
                    << ", \"recall10\": " << std::setprecision(4) << avg_rec
                    << ", \"all10\": " << avg_a10
                    << ", \"precision\": " << avg_prec
                    << ", \"false_positives\": " << tot_fp.load()
                    << ", \"avg_emitted\": " << std::setprecision(2) << (static_cast<double>(tot_emitted.load()) / nq_eval)
                    << ", \"t2_prune_rate\": " << std::setprecision(4) << t2_prune
                    << ", \"app_mb_q\": " << std::setprecision(3) << app_mb
                    << ", \"proc_mb_q\": " << proc_mb
                    << ", \"active_pss_gb\": " << std::setprecision(4) << peak_pss
                    << ", \"active_rss_gb\": " << peak_rss << ", \"vm_hwm_gb\": " << peak_hwm << "}";
                std::cout << "[" << kind << " " << wl.name << " " << task_names[task] << " " << mlabel << "] "
                          << "QPS=" << std::fixed << std::setprecision(1) << qps << " LL=" << std::setprecision(3) << ll_ratio
                          << " R=" << std::setprecision(4) << avg_rec << " All10=" << avg_a10
                          << " T2pr=" << std::setprecision(3) << t2_prune << " IO=" << std::setprecision(2) << app_mb << "MB/q"
                          << " FP=" << tot_fp.load() << " PSS=" << std::setprecision(3) << peak_pss << "GB\n" << std::flush;
            }
        }
    }
    out << "\n  ]\n}\n";
    pm_close(h);
    return 0;
}
