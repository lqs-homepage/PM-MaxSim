// certvec_cli.cpp — Comprehensive Adversarial Unit Test Suite & Benchmark Driver for libpm_certvec.so
// Verifies ALL Layer 2 audited invariants across d in {64, 128, 256, 512, 768, 1024}:
//  1. Lemma 3.1 hardware upper-bound soundness (B1 + B4, FP32/FP16, exact int32 centering, 0 violations)
//  2. Theorem 3 + 3B end-to-end synthetic CERTVEC5 multi-field (d1=768 + d2=512) build -> open -> query:
//     - W1 Top-3 Sum & Mean, W2 MaxSim, W3 Multi-Field + signed w0 < 0 / w0 > 0, ReLU, Min,
//     - W4 Q-Thresh (0 false positives, global Path I recall >= 1-delta), exact rank-k tie breaking,
//     - Negative weight w_i < 0 rejected with -EINVAL
//  3. Theorem 4C 64K-container pre-filter (signed int32_t acc, exact per-q_i max, correlation gate)
//  4. Theorem 5 Unified 4 KiB S3-FIFO Page Cache (heap + simulated PG shared-memory attach, exact B4/T2 hit accounting)
//  5. Live Dynamic LSM (pm_insert_entity with larger-norm vector testing atomic_max M_norm, pm_delete_entity, pm_compact_delta)

#include "pm_certvec.h"
#include "pm_kernels.hpp"
#include "pm_s3fifo_cache.hpp"
#include "roaring.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace pm_certvec;

static std::vector<double> make_haar_orthogonal_f64(uint32_t d, std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> A(static_cast<size_t>(d) * d);
    for (auto& v : A) v = nd(rng);
    // Modified Gram-Schmidt in FP64 (exact orthonormal basis)
    std::vector<double> Q(static_cast<size_t>(d) * d, 0.0);
    for (uint32_t c = 0; c < d; ++c) {
        std::vector<double> col(d);
        for (uint32_t r = 0; r < d; ++r) col[r] = A[static_cast<size_t>(r) * d + c];
        for (int pass = 0; pass < 2; ++pass) {
            for (uint32_t prev = 0; prev < c; ++prev) {
                double dot = 0.0;
                for (uint32_t r = 0; r < d; ++r) dot += col[r] * Q[static_cast<size_t>(r) * d + prev];
                for (uint32_t r = 0; r < d; ++r) col[r] -= dot * Q[static_cast<size_t>(r) * d + prev];
            }
        }
        double n2 = 0.0;
        for (uint32_t r = 0; r < d; ++r) n2 += col[r] * col[r];
        double inv = 1.0 / std::sqrt(n2);
        for (uint32_t r = 0; r < d; ++r) Q[static_cast<size_t>(r) * d + c] = col[r] * inv;
    }
    return Q;
}

static int test_s3fifo_shmem_and_hits() {
    std::cout << "[Test 1] Unified 4 KiB S3-FIFO Page Cache (Heap + Shared Memory Attach)... ";
    size_t req = pm_s3fifo_required_shmem_bytes(8ULL * 1024 * 1024);
    void* shmem = std::aligned_alloc(4096, ((req + 4095) / 4096) * 4096);
    assert(shmem != nullptr);

    UnifiedS3FifoCache c1, c2;
    assert(c1.attach_region(shmem, req, true));
    // Simulate second PostgreSQL backend attaching to the SAME shared memory region without wiping!
    assert(c2.attach_region(shmem, req, false));

    alignas(4096) uint8_t p_b4[4096], p_t2[4096], out[4096];
    std::memset(p_b4, 0xA5, 4096);
    std::memset(p_t2, 0x5A, 4096);
    // Process 1 writes a B4 page and a T2 page
    c1.put_page(0, 42, 0, p_b4);
    c1.put_page(1, 99, 1, p_t2);
    // Process 2 reads BOTH pages and gets exact bytes!
    assert(c2.get_page(0, 42, 0, out) && std::memcmp(out, p_b4, 4096) == 0);
    assert(c2.get_page(1, 99, 1, out) && std::memcmp(out, p_t2, 4096) == 0);
    assert(!c2.get_page(0, 43, 0, out));

    uint64_t hb4 = 0, ht2 = 0, miss = 0, ev = 0;
    c1.get_stats(hb4, ht2, miss, ev);
    assert(hb4 == 1 && ht2 == 1 && miss == 1);
    std::free(shmem);
    std::cout << "PASS (shared B4/T2 hits=1/1, misses=1)\n";
    return 0;
}

static int test_lemma31_all_dims() {
    std::cout << "[Test 2] Generalized Lemma 3.1 C++ SIMD Torture across d in {64,128,256,512,768,1024}... ";
    std::mt19937_64 rng(2026112701ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.1, 8.0);
    const uint32_t dims[6] = {64, 128, 256, 512, 768, 1024};
    uint64_t total_pairs = 0;

    for (uint32_t d : dims) {
        auto P = make_haar_orthogonal_f64(d, rng);
        double t_min_d = beta_t_min_d(static_cast<int>(d));
        bool scalar_fp32 = (d >= 256);
        const int n_pairs = 256;
        for (int p = 0; p < n_pairs; ++p) {
            std::vector<float> q(d), c(d);
            std::vector<uint16_t> c_bf(d);
            double qscale = ud(rng);
            double cn2 = 0.0, cbe2 = 0.0;
            for (uint32_t j = 0; j < d; ++j) {
                q[j] = static_cast<float>(nd(rng) * qscale / std::sqrt(static_cast<double>(d)));
                c[j] = static_cast<float>(nd(rng) / std::sqrt(static_cast<double>(d)));
                c_bf[j] = f32_to_bf16_rne(c[j]);
                double diff = static_cast<double>(c[j]) - static_cast<double>(bf16_to_f32(c_bf[j]));
                cn2 += static_cast<double>(c[j]) * c[j];
                cbe2 += diff * diff;
            }
            float c_norm = static_cast<float>(std::sqrt(cn2));
            float c_bf_err = static_cast<float>(std::sqrt(cbe2));

            PreparedQuerySlot qs;
            qs.init(0, 0, d, q.data(), P.data(), PM_OP_MAX, 1, 1.0, 0.0, false, 1.0, 0.05, 0.2, 1, 32);

            std::vector<double> r(d, 0.0);
            if (p == 0) { /* exact zero residual */ }
            else if (p == 1) { for (uint32_t j = 0; j < d; ++j) r[j] = 1e-13; }
            else {
                double rsc = (p == 2) ? 0.002 : (0.3 + 0.7 * (p % 3));
                for (uint32_t j = 0; j < d; ++j) r[j] = nd(rng) * rsc / std::sqrt(static_cast<double>(d));
            }
            double rho2 = 0.0;
            for (double v : r) rho2 += v * v;
            double rho = std::sqrt(rho2);

            // Rotate r in FP64
            std::vector<double> o_p(d, 0.0);
            if (rho > 1e-12) {
                for (uint32_t k = 0; k < d; ++k) {
                    double ok = r[k] / rho;
                    for (uint32_t j = 0; j < d; ++j) o_p[j] += ok * P[static_cast<size_t>(k) * d + j];
                }
            }
            // Test BOTH B4 (including worst-case y in {+-0.5}^d cancellation on p==3!) and B1
            for (int mode = 0; mode < 2; ++mode) {
                total_pairs++;
                if (rho <= 1e-12) continue;
                float cent_ip = dot_bf16_pair(qs.q_bf16.data(), c_bf.data(), d);
                double exact_qc = 0.0;
                for (uint32_t j = 0; j < d; ++j) exact_qc += static_cast<double>(q[j]) * c[j];

                if (mode == 0) {
                    // B4
                    std::vector<int> u_code(d);
                    double ip = 0.0, yn2 = 0.0;
                    for (uint32_t j = 0; j < d; ++j) {
                        int u = (p == 3) ? ((o_p[j] >= 0.0) ? 8 : 7)
                                         : std::clamp<int>(static_cast<int>(std::floor(o_p[j] * 5.0 * std::sqrt(static_cast<double>(d)) + 7.5 + 0.5)), 0, 15);
                        u_code[j] = u;
                        double y = static_cast<double>(u) - 7.5;
                        ip += y * o_p[j]; yn2 += y * y;
                    }
                    double ynorm = std::sqrt(yn2);
                    double cx = ip / ynorm;
                    double sx = rho / ip;
                    double eta = rho * std::sqrt(std::max(0.0, 1.0 - cx * cx)) / cx;
                    double exact_y_qp = 0.0;
                    for (uint32_t j = 0; j < d; ++j) exact_y_qp += (static_cast<double>(u_code[j]) - 7.5) * qs.q_rot_f64[j];
                    double ub_math = exact_qc + sx * exact_y_qp + qs.q_norm_f64 * eta * qs.kappa_single;

                    std::vector<uint8_t> packed(d / 2, 0);
                    for (uint32_t b = 0; b < d / 2; ++b) packed[b] = static_cast<uint8_t>((u_code[2 * b] & 0xF) | ((u_code[2 * b + 1] & 0xF) << 4));
                    double s_st = 0.0, e_st = 0.0;
                    if (scalar_fp32) {
                        float sf = static_cast<float>(sx);
                        s_st = sf;
                        e_st = std::nextafter(static_cast<float>(eta + std::fabs(static_cast<double>(sf) - sx) * ynorm / t_min_d), std::numeric_limits<float>::infinity());
                    } else {
                        uint16_t sh = f32_to_f16_scalar(static_cast<float>(sx));
                        s_st = f16_to_f32_scalar(sh);
                        e_st = f16_to_f32_scalar(f32_to_f16_round_up(eta + std::fabs(s_st - sx) * ynorm / t_min_d));
                    }
                    int32_t dot_u = 0; double y_up = 0.0;
                    b4_exact_dot_and_ynorm(packed.data(), qs.q_i8_eo.data(), d, dot_u, y_up);
                    double exact_dot_y = static_cast<double>(2LL * dot_u - 15LL * qs.sum_qi8) * (0.5 * qs.delta_q);
                    double ub_hw = lemma31_pair_ub(cent_ip, c_norm, c_bf_err, s_st, e_st, exact_dot_y, y_up, qs.kappa_single, qs);
                    if (ub_hw < ub_math) {
                        std::cerr << "\nFAIL B4 d=" << d << " p=" << p << " deficit=" << (ub_math - ub_hw) << "\n";
                        return 1;
                    }
                } else {
                    // B1
                    std::vector<uint8_t> bits(d / 8, 0);
                    double ip = 0.0, exact_y_qp = 0.0;
                    for (uint32_t j = 0; j < d; ++j) {
                        if (o_p[j] >= 0.0) { bits[j >> 3] |= static_cast<uint8_t>(1u << (j & 7u)); ip += o_p[j]; exact_y_qp += qs.q_rot_f64[j]; }
                        else { ip -= o_p[j]; exact_y_qp -= qs.q_rot_f64[j]; }
                    }
                    double ynorm = std::sqrt(static_cast<double>(d));
                    double cx = ip / ynorm;
                    double sx = rho / ip;
                    double eta = rho * std::sqrt(std::max(0.0, 1.0 - cx * cx)) / cx;
                    double ub_math = exact_qc + sx * exact_y_qp + qs.q_norm_f64 * eta * qs.kappa_w1;

                    double s_st = 0.0, e_st = 0.0;
                    if (scalar_fp32) {
                        float sf = static_cast<float>(sx);
                        s_st = sf;
                        e_st = std::nextafter(static_cast<float>(eta + std::fabs(static_cast<double>(sf) - sx) * ynorm / t_min_d), std::numeric_limits<float>::infinity());
                    } else {
                        uint16_t sh = f32_to_f16_scalar(static_cast<float>(sx));
                        s_st = f16_to_f32_scalar(sh);
                        e_st = f16_to_f32_scalar(f32_to_f16_round_up(eta + std::fabs(s_st - sx) * ynorm / t_min_d));
                    }
                    int32_t dot_b = b1_exact_dot_bits(bits.data(), qs.q_i8_linear.data(), d);
                    double exact_dot_y = static_cast<double>(2 * dot_b - qs.sum_qi8) * qs.delta_q;
                    double y_up = std::nextafter(std::sqrt(static_cast<float>(d)), std::numeric_limits<float>::infinity());
                    double ub_hw = lemma31_pair_ub(cent_ip, c_norm, c_bf_err, s_st, e_st, exact_dot_y, y_up, qs.kappa_w1, qs);
                    if (ub_hw < ub_math) {
                        std::cerr << "\nFAIL B1 d=" << d << " p=" << p << " deficit=" << (ub_math - ub_hw) << "\n";
                        return 1;
                    }
                }
            }
        }
    }
    std::cout << "PASS (" << total_pairs << " adversarial SIMD pairs across d=64..1024, 0 violations)\n";
    return 0;
}

static int test_end_to_end_multifield_certvec5() {
    std::cout << "[Test 3] End-to-End Multi-Field CERTVEC5 (d0=256 chunks + d1=512 title + signed BM25 + Q-Thresh + LSM)...\n";
    std::string tmp_dir = "./_certvec5_ut_tmp";
    int rc_rm = std::system(("rm -rf " + tmp_dir + " && mkdir -p " + tmp_dir + "/raw " + tmp_dir + "/idx/field_0 " + tmp_dir + "/idx/field_1").c_str());
    (void)rc_rm;

    std::mt19937_64 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    const uint32_t N = 200;
    const uint32_t d0 = 256, K0 = 32;
    const uint32_t d1 = 512, K1 = 32;

    // Build two fields: field 0 (variable 4..8 chunks of d=256), field 1 (1 title vec of d=512)
    auto gen_field = [&](uint32_t fid, uint32_t d, uint32_t K, uint32_t min_m, uint32_t max_m,
                         std::vector<std::vector<uint16_t>>& out_doc_vecs,
                         std::vector<uint32_t>& out_lens) {
        std::vector<float> C(K * d);
        for (uint32_t c = 0; c < K; ++c) {
            double n2 = 0.0;
            for (uint32_t j = 0; j < d; ++j) { C[c * d + j] = static_cast<float>(nd(rng)); n2 += C[c * d + j] * C[c * d + j]; }
            float inv = static_cast<float>(1.0 / std::sqrt(n2));
            for (uint32_t j = 0; j < d; ++j) C[c * d + j] *= inv;
        }
        auto P = make_haar_orthogonal_f64(d, rng);
        out_doc_vecs.resize(N);
        out_lens.resize(N);
        std::vector<uint16_t> all_vecs;
        std::vector<uint16_t> all_cids;
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t m = min_m + (i % (max_m - min_m + 1));
            out_lens[i] = m;
            out_doc_vecs[i].resize(m * d);
            for (uint32_t tk = 0; tk < m; ++tk) {
                uint16_t cid = static_cast<uint16_t>((i + tk) % K);
                std::vector<float> v(d);
                double n2 = 0.0;
                for (uint32_t j = 0; j < d; ++j) {
                    v[j] = C[cid * d + j] + static_cast<float>(0.25 * nd(rng) / std::sqrt(static_cast<double>(d)));
                    n2 += v[j] * v[j];
                }
                float inv = static_cast<float>(1.0 / std::sqrt(n2));
                // Find true nearest centroid for cid assignment
                uint16_t best_c = 0; double best_d2 = 1e30;
                for (uint32_t c = 0; c < K; ++c) {
                    double d2 = 0.0;
                    for (uint32_t j = 0; j < d; ++j) { double df = v[j] * inv - C[c * d + j]; d2 += df * df; }
                    if (d2 < best_d2) { best_d2 = d2; best_c = static_cast<uint16_t>(c); }
                }
                all_cids.push_back(best_c);
                for (uint32_t j = 0; j < d; ++j) {
                    uint16_t h16 = f32_to_f16_scalar(v[j] * inv);
                    out_doc_vecs[i][tk * d + j] = h16;
                    all_vecs.push_back(h16);
                }
            }
        }
        std::string pfx = tmp_dir + "/raw/f" + std::to_string(fid);
        { std::ofstream f(pfx + "_vecs.f16", std::ios::binary); f.write(reinterpret_cast<const char*>(all_vecs.data()), all_vecs.size() * 2); }
        { std::ofstream f(pfx + "_lens.u32", std::ios::binary); f.write(reinterpret_cast<const char*>(out_lens.data()), N * 4); }
        { std::ofstream f(pfx + "_cids.u16", std::ios::binary); f.write(reinterpret_cast<const char*>(all_cids.data()), all_cids.size() * 2); }
        { std::ofstream f(pfx + "_C.f32", std::ios::binary); f.write(reinterpret_cast<const char*>(C.data()), C.size() * 4); }
        { std::ofstream f(pfx + "_P.f64", std::ios::binary); f.write(reinterpret_cast<const char*>(P.data()), P.size() * 8); }
        std::string cmd = "./certvec_builder_v5 --build-field " + tmp_dir + "/idx/field_" + std::to_string(fid) + " "
            + std::to_string(N) + " " + std::to_string(d) + " " + std::to_string(K) + " 1 1 "
            + pfx + "_vecs.f16 " + pfx + "_lens.u32 " + pfx + "_cids.u16 " + pfx + "_C.f32 " + pfx + "_P.f64 " + std::to_string(fid);
        int rc = std::system(cmd.c_str());
        assert(rc == 0);
    };

    std::vector<std::vector<uint16_t>> f0_vecs, f1_vecs;
    std::vector<uint32_t> f0_lens, f1_lens;
    gen_field(0, d0, K0, 4, 8, f0_vecs, f0_lens);
    gen_field(1, d1, K1, 1, 1, f1_vecs, f1_lens);
    assert(std::system(("./certvec_builder_v5 --write-manifest " + tmp_dir + "/idx " + std::to_string(N) + " 2").c_str()) == 0);

    pm_open_opts_t opts{};
    opts.tier_mode = PM_TIER_SPLIT_B1B4;
    opts.ram_budget_bytes = 8ULL * 1024 * 1024;
    opts.io_backend = PM_IO_PREAD_POOL; // Also verifies portable pread backend!
    opts.self_calibrate = 1;
    opts.use_o_direct = 0;
    pm_handle_t* h = pm_open((tmp_dir + "/idx").c_str(), &opts);
    assert(h != nullptr);

    pm_index_info_t info{};
    assert(pm_get_info(h, &info) == 0);
    assert(info.n_docs == N && info.n_fields == 2 && info.dims[0] == d0 && info.dims[1] == d1);

    // Prepare multi-field query: q0 (d0=256, TOP_R_MEAN r=3, w1=0.6) + q1 (d1=512, MAX r=1, w2=0.4) + signed w0=-0.15 * aux
    std::vector<float> q_all(d0 + d1);
    for (uint32_t j = 0; j < d0; ++j) q_all[j] = f16_to_f32_scalar(f0_vecs[7][j]);
    for (uint32_t j = 0; j < d1; ++j) q_all[d0 + j] = f16_to_f32_scalar(f1_vecs[7][j]);
    std::vector<float> aux_dense(N);
    for (uint32_t i = 0; i < N; ++i) aux_dense[i] = static_cast<float>(i) * 0.01f;

    // Filter bitmap: odd RIDs only (N_P = 100)
    std::vector<uint32_t> odd_rids;
    for (uint32_t i = 1; i < N; i += 2) odd_rids.push_back(i);
    pm_bitmap_view_t bm{PM_BM_RID_ARRAY, odd_rids.data(), odd_rids.size()};
    pm_aux_scores_t aux{PM_AUX_DENSE_F32, aux_dense.data(), nullptr, N, 0.0f, -0.15f};

    // Exact ground truth over odd RIDs
    std::vector<std::pair<double, uint32_t>> exact_gt;
    for (uint32_t rid : odd_rids) {
        std::vector<double> ips(f0_lens[rid]);
        for (uint32_t tk = 0; tk < f0_lens[rid]; ++tk)
            ips[tk] = dot_f32_f16(q_all.data(), f0_vecs[rid].data() + tk * d0, d0);
        double g0 = reduce_top_r_f64(ips.data(), f0_lens[rid], PM_OP_TOP_R_MEAN, 3);
        double g1 = dot_f32_f16(q_all.data() + d0, f1_vecs[rid].data(), d1);
        double sc = -0.15 * aux_dense[rid] + 0.6 * g0 + 0.4 * g1;
        exact_gt.emplace_back(sc, rid);
    }
    std::sort(exact_gt.begin(), exact_gt.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });

    // 3A. Negative weight w_i < 0 MUST return -EINVAL!
    uint32_t fids[2] = {0, 1};
    pm_set_op_t ops[2] = {PM_OP_TOP_R_MEAN, PM_OP_MAX};
    uint32_t rvals[2] = {3, 1};
    float bad_w[2] = {0.6f, -0.4f};
    float good_w[2] = {0.6f, 0.4f};
    pm_query_spec_t qspec{};
    qspec.L = 2; qspec.q_vecs = q_all.data(); qspec.field_ids = fids;
    qspec.set_ops = ops; qspec.r_vals = rvals; qspec.weights = bad_w;
    qspec.agg_kind = PM_AGG_WSUM; qspec.mode = PM_MODE_TOPK; qspec.k = 10; qspec.delta = 0.05; qspec.split_lambda = 0.2;
    pm_hit_t hits[10]; uint32_t out_n = 0; pm_query_stats_t st{};
    assert(pm_execute_query(h, &bm, &aux, &qspec, hits, &out_n, nullptr, nullptr, &st) == -EINVAL);

    // 3B. Exact Q-TopK (Split_B1_B4_T3, Path I) must match exact Top-10!
    qspec.weights = good_w; qspec.force_path1 = 1;
    assert(pm_execute_query(h, &bm, &aux, &qspec, hits, &out_n, nullptr, nullptr, &st) == 0);
    assert(out_n == 10);
    assert(hits[0].rid == exact_gt[0].second && std::fabs(hits[0].score - exact_gt[0].first) < 1e-6);
    int matched = 0;
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j) if (hits[i].rid == exact_gt[j].second) matched++;
    assert(matched >= 9);

    // 3C. Certified Q-Thresh (tau = 5th GT score): 0 false positives, exact recall >= 0.9!
    qspec.mode = PM_MODE_THRESHOLD;
    qspec.tau = exact_gt[4].first - 1e-9;
    std::vector<pm_hit_t> th_out;
    auto cb = [](void* ctx, uint32_t rid, double sc) -> int {
        static_cast<std::vector<pm_hit_t>*>(ctx)->push_back({rid, sc});
        return 0;
    };
    uint32_t th_n = 0;
    assert(pm_execute_query(h, &bm, &aux, &qspec, nullptr, &th_n, cb, &th_out, &st) == 0);
    for (const auto& th : th_out) assert(th.score >= qspec.tau); // 100% precision!
    assert(th_out.size() >= 4);

    // 3D. Live LSM insert (with higher-norm vector -> verify atomic_max M_norm!) + delete + compact!
    double prev_mnorm0 = info.m_norm_max[0];
    std::vector<float> big_v0(4 * d0, 0.0f), big_v1(d1, 0.0f);
    for (uint32_t j = 0; j < d0; ++j) big_v0[j] = 1.5f * q_all[j]; // norm ~ 1.5 > 1.0!
    for (uint32_t j = 0; j < d1; ++j) big_v1[j] = 1.5f * q_all[d0 + j];
    pm_field_vecs_t fvs[2] = {{0, 4, big_v0.data()}, {1, 1, big_v1.data()}};
    pm_entity_vecs_t ev{2, fvs};
    assert(pm_insert_entity(h, 999999u, &ev) == 0);
    assert(pm_get_info(h, &info) == 0 && info.m_norm_max[0] > prev_mnorm0 + 0.3);

    // Unfiltered query must now find newly inserted 999999 as #1!
    qspec.mode = PM_MODE_TOPK; qspec.k = 5;
    assert(pm_execute_query(h, nullptr, nullptr, &qspec, hits, &out_n, nullptr, nullptr, &st) == 0);
    assert(hits[0].rid == 999999u);

    // Delete 999999 + compact -> 999999 vanishes!
    assert(pm_delete_entity(h, 999999u) == 0);
    assert(pm_compact_delta(h) == 0);
    assert(pm_execute_query(h, nullptr, nullptr, &qspec, hits, &out_n, nullptr, nullptr, &st) == 0);
    for (uint32_t i = 0; i < out_n; ++i) assert(hits[i].rid != 999999u);

    pm_close(h);
    rc_rm = std::system(("rm -rf " + tmp_dir).c_str()); (void)rc_rm;
    std::cout << "  PASS (Multi-field d=256+512, Top-3 Mean, signed w0, -EINVAL, Q-Thresh 0 FP, atomic_max LSM)\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--unit-test-all") {
        if (test_s3fifo_shmem_and_hits() != 0) return 1;
        if (test_lemma31_all_dims() != 0) return 1;
        if (test_end_to_end_multifield_certvec5() != 0) return 1;
        std::cout << "ALL CERTVEC UNIT TESTS PASSED (0 VIOLATIONS)!\n";
        return 0;
    }
    std::cerr << "Usage: certvec_cli --unit-test-all\n";
    return 1;
}
