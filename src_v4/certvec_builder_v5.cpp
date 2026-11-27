// certvec_builder_v5.cpp — Multi-Threaded C++20 Builder for CERTVEC5 Multi-Field Indexes & V4 Sidecars
// Supports arbitrary d in {64, 128, 256, 512, 768, 1024}:
//  - Flat 64B-aligned tier0_shared.mmap (C_fp32, C_bf16, C_norms, C_bf_err, FP64 P_f64, rho_max,
//    doc_cids, doc_cids_full, and 32B-aligned roaring_bitmap_frozen_view for 48-process PG sharing!)
//  - FP32 (s_x, eta) for d >= 256; FP16 with hard <= 65,504 abort for d <= 128
//  - 4 KiB-aligned tier1_b4.nvme + tier2_fp16.nvme + mmap tier1_b1.mmap + avg_fp16.mmap
//  - Read-only sidecar builder (`--build-v4-sidecar`) for frozen Stage 3 v4 indexes!

#include "beta_quantiles_gen.hpp"
#include "pm_certvec.h"
#include "pm_io_backend.hpp"
#include "pm_kernels.hpp"
#include "roaring.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace pm_certvec;

constexpr uint64_t kCertVec5Magic = 0x4345525456454335ULL;
constexpr uint64_t kTier0MapMagic = 0x54304D4D41503035ULL;
constexpr uint64_t kFrozenSidecarMagic = 0x46524F5A454E3035ULL;

struct CertVec5Manifest {
    uint64_t magic;
    uint64_t n_docs;
    uint32_t n_fields;
    uint32_t reserved;
};

struct CertVec5FieldMeta {
    uint32_t field_id;
    uint32_t d;
    uint32_t K;
    uint32_t is_unit;
    uint32_t scalar_fp32;
    uint32_t has_doc_cids_full;
    uint32_t has_avg_fp16;
    uint32_t b1_y_unit_pm1;
    uint64_t u_max;
    double   u_doc_avg;
    double   m_norm_max;
};

struct Tier0MmapHeader {
    uint64_t magic;
    uint64_t n_docs;
    uint32_t K;
    uint32_t d;
    uint64_t off_C_fp32;
    uint64_t off_C_bf16;
    uint64_t off_C_norms;
    uint64_t off_C_bf_err;
    uint64_t off_P_f64;
    uint64_t off_rho_max;
    uint64_t off_doc_offs;
    uint64_t off_doc_cids;
    uint64_t len_doc_cids;
    uint64_t off_doc_full_offs;
    uint64_t off_doc_cids_full;
    uint64_t len_doc_cids_full;
    uint64_t off_frozen_offs;
    uint64_t off_frozen_blob;
};

static inline void pad_file_to(std::ofstream& out, size_t& pos, size_t align) {
    size_t target = align_up_sz(pos, align);
    if (target > pos) {
        std::vector<char> z(target - pos, 0);
        out.write(z.data(), static_cast<std::streamsize>(z.size()));
        pos = target;
    }
}

// Fast Extended RaBitQ 4-bit optimal scale search on rotated unit residual o_p (matches verify_theory_v4.py!)
static void encode_rabitq_b4_vec(const double* r_vec, const double* P_f64, uint32_t d,
                                 bool scalar_fp32, double t_min_d,
                                 double& out_s, double& out_eta, uint8_t* out_packed) {
    double rho2 = 0.0;
    for (uint32_t j = 0; j < d; ++j) rho2 += r_vec[j] * r_vec[j];
    double rho = std::sqrt(rho2);
    if (rho <= 1e-12) {
        out_s = 0.0; out_eta = 0.0;
        std::memset(out_packed, 0, d / 2);
        return;
    }
    double inv_rho = 1.0 / rho;
    std::vector<double> o_p(d, 0.0);
    for (uint32_t k = 0; k < d; ++k) {
        double ok = r_vec[k] * inv_rho;
        const double* prow = P_f64 + static_cast<size_t>(k) * d;
        for (uint32_t j = 0; j < d; ++j) o_p[j] += ok * prow[j];
    }
    double max_abs = 1e-12;
    for (uint32_t j = 0; j < d; ++j) max_abs = std::max(max_abs, std::fabs(o_p[j]));

    double best_c = -1.0;
    std::vector<int> best_u(d, 7);
    std::vector<int> cur_u(d);
    for (int s_idx = 0; s_idx < 14; ++s_idx) {
        double frac = 3.5 + (4.5 * s_idx) / 13.0;
        double sc = frac / max_abs;
        double yn2 = 0.0, ip = 0.0;
        for (uint32_t j = 0; j < d; ++j) {
            int u = std::clamp<int>(static_cast<int>(std::floor(o_p[j] * sc + 7.5 + 0.5)), 0, 15);
            cur_u[j] = u;
            double y = static_cast<double>(u) - 7.5;
            yn2 += y * y;
            ip += y * o_p[j];
        }
        if (yn2 > 0.0) {
            double c = ip / std::sqrt(yn2);
            if (c > best_c) { best_c = c; best_u.swap(cur_u); }
        }
    }
    double ip = 0.0, yn2 = 0.0;
    for (uint32_t j = 0; j < d; ++j) {
        double y = static_cast<double>(best_u[j]) - 7.5;
        ip += y * o_p[j];
        yn2 += y * y;
    }
    double ynorm = std::sqrt(yn2);
    double c_x = ip / ynorm;
    double sx = rho / ip;
    double eta = rho * std::sqrt(std::max(0.0, 1.0 - c_x * c_x)) / c_x;

    if (!scalar_fp32) {
        // FP16 storage (d <= 128): absorb |s_f16 - s_x|*||y||_2 / T_min(d) into eta_f16 with hard <= 65504 check!
        uint16_t sh = f32_to_f16_scalar(static_cast<float>(sx));
        double sf = static_cast<double>(f16_to_f32_scalar(sh));
        double eta_req = eta + (std::fabs(sf - sx) * ynorm) / t_min_d + 1e-12 * rho / t_min_d;
        if (std::fabs(sx) > 65504.0 || eta_req > 65504.0) {
            std::cerr << "FATAL Lemma 3.1 FP16 overflow: sx=" << sx << " eta_req=" << eta_req << "\n";
            std::exit(2);
        }
        out_s = sf;
        out_eta = static_cast<double>(f16_to_f32_scalar(f32_to_f16_round_up(eta_req)));
    } else {
        float sf = static_cast<float>(sx);
        double eta_req = eta + (std::fabs(static_cast<double>(sf) - sx) * ynorm) / t_min_d + 1e-12 * rho / t_min_d;
        out_s = static_cast<double>(sf);
        out_eta = static_cast<double>(std::nextafter(static_cast<float>(eta_req), std::numeric_limits<float>::infinity()));
    }

    // Pack even in low nibble, odd in high nibble (matches b4_exact_dot_and_ynorm!)
    for (uint32_t b = 0; b < d / 2; ++b) {
        out_packed[b] = static_cast<uint8_t>((best_u[2 * b] & 0x0F) | ((best_u[2 * b + 1] & 0x0F) << 4));
    }
}

static void encode_rabitq_b1_vec(const double* r_vec, const double* P_f64, uint32_t d,
                                 bool scalar_fp32, double t_min_d,
                                 double& out_s, double& out_eta, uint8_t* out_bits) {
    double rho2 = 0.0;
    for (uint32_t j = 0; j < d; ++j) rho2 += r_vec[j] * r_vec[j];
    double rho = std::sqrt(rho2);
    std::memset(out_bits, 0, d / 8);
    if (rho <= 1e-12) { out_s = 0.0; out_eta = 0.0; return; }
    double inv_rho = 1.0 / rho;
    std::vector<double> o_p(d, 0.0);
    for (uint32_t k = 0; k < d; ++k) {
        double ok = r_vec[k] * inv_rho;
        const double* prow = P_f64 + static_cast<size_t>(k) * d;
        for (uint32_t j = 0; j < d; ++j) o_p[j] += ok * prow[j];
    }
    double ip = 0.0;
    for (uint32_t j = 0; j < d; ++j) {
        if (o_p[j] >= 0.0) {
            out_bits[j >> 3] |= static_cast<uint8_t>(1u << (j & 7u));
            ip += o_p[j];
        } else {
            ip -= o_p[j];
        }
    }
    // y in {-1, +1}^d => ||y||_2 = sqrt(d)
    double ynorm = std::sqrt(static_cast<double>(d));
    double c_x = ip / ynorm;
    if (c_x < 1e-12) c_x = 1e-12;
    double sx = rho / (c_x * ynorm);
    double eta = rho * std::sqrt(std::max(0.0, 1.0 - c_x * c_x)) / c_x;

    if (!scalar_fp32) {
        uint16_t sh = f32_to_f16_scalar(static_cast<float>(sx));
        double sf = static_cast<double>(f16_to_f32_scalar(sh));
        double eta_req = eta + (std::fabs(sf - sx) * ynorm) / t_min_d + 1e-12 * rho / t_min_d;
        if (std::fabs(sx) > 65504.0 || eta_req > 65504.0) {
            std::cerr << "FATAL Lemma 3.1 B1 FP16 overflow: sx=" << sx << " eta_req=" << eta_req << "\n";
            std::exit(2);
        }
        out_s = sf;
        out_eta = static_cast<double>(f16_to_f32_scalar(f32_to_f16_round_up(eta_req)));
    } else {
        float sf = static_cast<float>(sx);
        double eta_req = eta + (std::fabs(static_cast<double>(sf) - sx) * ynorm) / t_min_d + 1e-12 * rho / t_min_d;
        out_s = static_cast<double>(sf);
        out_eta = static_cast<double>(std::nextafter(static_cast<float>(eta_req), std::numeric_limits<float>::infinity()));
    }
}

// Mode 1: Build a complete CERTVEC5 field directory from raw binary inputs
static int build_field_cmd(int argc, char** argv) {
    // Usage: certvec_builder_v5 --build-field <out_field_dir> <N_docs> <d> <K> <is_unit> <keep_full_cids>
    //        <vecs_f16.bin> <lens_u32.bin> <cids_full_u16.bin> <C_f32.bin> <P_f64.bin> <field_id>
    if (argc < 14) {
        std::cerr << "Usage: --build-field <out_dir> <N> <d> <K> <is_unit> <keep_full> <vecs.f16> <lens.u32> <cids.u16> <C.f32> <P.f64> <field_id>\n";
        return 1;
    }
    std::string out_dir = argv[2];
    uint64_t N = std::stoull(argv[3]);
    uint32_t d = static_cast<uint32_t>(std::stoul(argv[4]));
    uint32_t K = static_cast<uint32_t>(std::stoul(argv[5]));
    uint32_t is_unit = static_cast<uint32_t>(std::stoul(argv[6]));
    uint32_t keep_full = static_cast<uint32_t>(std::stoul(argv[7]));
    std::string vecs_path = argv[8], lens_path = argv[9], cids_path = argv[10], C_path = argv[11], P_path = argv[12];
    uint32_t field_id = static_cast<uint32_t>(std::stoul(argv[13]));

    mkdir(out_dir.c_str(), 0755);
    const bool scalar_fp32 = (d >= 256);
    const double t_min_d = beta_t_min_d(static_cast<int>(d));

    // Load centroids C_f32, Haar P_f64, doc lengths, and token centroid assignments
    std::vector<float> C_fp32(static_cast<size_t>(K) * d);
    { std::ifstream f(C_path, std::ios::binary); f.read(reinterpret_cast<char*>(C_fp32.data()), C_fp32.size() * 4); }
    std::vector<double> P_f64(static_cast<size_t>(d) * d);
    { std::ifstream f(P_path, std::ios::binary); f.read(reinterpret_cast<char*>(P_f64.data()), P_f64.size() * 8); }
    std::vector<uint32_t> lens(N);
    { std::ifstream f(lens_path, std::ios::binary); f.read(reinterpret_cast<char*>(lens.data()), N * 4); }

    std::vector<uint64_t> full_offs(N + 1, 0);
    for (uint64_t i = 0; i < N; ++i) full_offs[i + 1] = full_offs[i] + lens[i];
    uint64_t total_toks = full_offs[N];

    std::vector<uint16_t> cids_full(total_toks);
    { std::ifstream f(cids_path, std::ios::binary); f.read(reinterpret_cast<char*>(cids_full.data()), total_toks * 2); }

    // 1. Precompute C_bf16, C_norms, C_bf_err
    std::vector<uint16_t> C_bf16(static_cast<size_t>(K) * d);
    std::vector<float> C_norms(K), C_bf_err(K);
    #pragma omp parallel for schedule(static)
    for (uint32_t c = 0; c < K; ++c) {
        double n2 = 0.0, e2 = 0.0;
        for (uint32_t j = 0; j < d; ++j) {
            float cf = C_fp32[static_cast<size_t>(c) * d + j];
            uint16_t bf = f32_to_bf16_rne(cf);
            C_bf16[static_cast<size_t>(c) * d + j] = bf;
            double diff = static_cast<double>(cf) - static_cast<double>(bf16_to_f32(bf));
            n2 += static_cast<double>(cf) * cf;
            e2 += diff * diff;
        }
        C_norms[c] = static_cast<float>(std::sqrt(n2));
        C_bf_err[c] = static_cast<float>(std::sqrt(e2));
    }

    // 2. Build deduplicated doc_cids + exact 32-byte-aligned frozen CRoaring inverted postings!
    std::vector<uint64_t> doc_offs(N + 1, 0);
    std::vector<std::vector<uint16_t>> per_doc_dedup(N);
    uint64_t u_max = 1;
    #pragma omp parallel for schedule(dynamic, 1024) reduction(max:u_max)
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t s = full_offs[i], e = full_offs[i + 1];
        auto& v = per_doc_dedup[i];
        v.assign(cids_full.begin() + s, cids_full.begin() + e);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        u_max = std::max<uint64_t>(u_max, v.size());
    }
    for (uint64_t i = 0; i < N; ++i) doc_offs[i + 1] = doc_offs[i] + per_doc_dedup[i].size();
    std::vector<uint16_t> doc_cids(doc_offs[N]);
    std::vector<uint64_t> inv_counts(K, 0);
    for (uint64_t i = 0; i < N; ++i) {
        std::memcpy(doc_cids.data() + doc_offs[i], per_doc_dedup[i].data(), per_doc_dedup[i].size() * 2);
        for (uint16_t c : per_doc_dedup[i]) inv_counts[c]++;
    }
    std::vector<uint64_t> inv_offs(K + 1, 0);
    for (uint32_t c = 0; c < K; ++c) inv_offs[c + 1] = inv_offs[c] + inv_counts[c];
    std::vector<uint32_t> inv_docs(inv_offs[K]);
    std::vector<uint64_t> cur_pos = inv_offs;
    for (uint64_t i = 0; i < N; ++i) {
        for (uint16_t c : per_doc_dedup[i]) inv_docs[cur_pos[c]++] = static_cast<uint32_t>(i);
    }
    per_doc_dedup.clear(); per_doc_dedup.shrink_to_fit();

    std::vector<std::vector<char>> frozen_bufs(K);
    #pragma omp parallel for schedule(dynamic, 64)
    for (uint32_t c = 0; c < K; ++c) {
        roaring_bitmap_t* r = roaring_bitmap_of_ptr(static_cast<size_t>(inv_offs[c + 1] - inv_offs[c]), inv_docs.data() + inv_offs[c]);
        roaring_bitmap_run_optimize(r);
        roaring_bitmap_shrink_to_fit(r);
        size_t fsz = roaring_bitmap_frozen_size_in_bytes(r);
        frozen_bufs[c].resize(fsz);
        roaring_bitmap_frozen_serialize(r, frozen_bufs[c].data());
        roaring_bitmap_free(r);
    }
    inv_docs.clear(); inv_docs.shrink_to_fit();

    // 3. Stream raw vecs_f16.bin in chunks to build tier2_fp16.nvme, tier1_b4.nvme, tier1_b1.mmap,
    //    avg_fp16.mmap, exact m_norm_max, and exact per-centroid rho_max[K]!
    MappedReadFile vecs_map;
    if (!vecs_map.open_mmap(vecs_path)) { std::cerr << "Cannot mmap " << vecs_path << "\n"; return 1; }
    const uint16_t* all_vecs_f16 = reinterpret_cast<const uint16_t*>(vecs_map.ptr);

    std::ofstream ft2(out_dir + "/tier2_fp16.nvme", std::ios::binary);
    std::ofstream fb4(out_dir + "/tier1_b4.nvme", std::ios::binary);
    std::ofstream fb1(out_dir + "/tier1_b1.mmap", std::ios::binary);
    std::ofstream favg(out_dir + "/avg_fp16.mmap", std::ios::binary);

    std::vector<DocDiskEntry> t2_meta(N), b4_meta(N), b1_meta(N);
    std::vector<float> rho_max(K, 0.0f);
    double m_norm_max_global = 0.0;
    uint64_t t2_off = 0, b4_off = 0, b1_off = 0;

    const uint64_t batch_docs = 4096;
    for (uint64_t bs = 0; bs < N; bs += batch_docs) {
        uint64_t be = std::min(N, bs + batch_docs);
        uint64_t cnt = be - bs;
        std::vector<std::vector<uint8_t>> b_t2(cnt), b_b4(cnt), b_b1(cnt);
        std::vector<std::vector<uint16_t>> b_avg(cnt);
        std::vector<double> local_mnorm(cnt, 0.0);
        // Per-thread rho_max reduction
        int nT = omp_get_max_threads();
        std::vector<std::vector<float>> thr_rho(nT, std::vector<float>(K, 0.0f));

        #pragma omp parallel for schedule(dynamic, 32)
        for (uint64_t bi = 0; bi < cnt; ++bi) {
            int tid = omp_get_thread_num();
            uint64_t doc = bs + bi;
            uint32_t m = lens[doc];
            uint64_t s = full_offs[doc];
            b_avg[bi].assign(d, 0);
            if (m == 0) {
                b_t2[bi].assign(kPageSize, 0);
                b_b4[bi].assign(kPageSize, 0);
                b_b1[bi].assign(16, 0);
                continue;
            }

            // Tier-2 FP16 (4 KiB aligned)
            size_t t2_raw = static_cast<size_t>(m) * d * 2ULL;
            size_t t2_al = align_up_sz(t2_raw, kPageSize);
            b_t2[bi].assign(t2_al, 0);
            std::memcpy(b_t2[bi].data(), all_vecs_f16 + s * d, t2_raw);

            // Tier-1 B4 (4 KiB aligned) & Tier-1 B1 (16-byte aligned)
            size_t b4_sc_off = scalar_fp32 ? align_up_sz(2ULL * m, 4) : (2ULL * m);
            size_t b4_pk_off = scalar_fp32 ? align_up_sz(b4_sc_off + 8ULL * m, 64) : align_up_sz(6ULL * m, 64);
            size_t b4_al = align_up_sz(b4_pk_off + static_cast<size_t>(m) * (d / 2), kPageSize);
            b_b4[bi].assign(b4_al, 0);

            size_t b1_sc_off = scalar_fp32 ? align_up_sz(2ULL * m, 4) : (2ULL * m);
            size_t b1_bt_off = scalar_fp32 ? align_up_sz(b1_sc_off + 8ULL * m, 16) : (6ULL * m);
            size_t b1_al = align_up_sz(b1_bt_off + static_cast<size_t>(m) * (d / 8), 16);
            b_b1[bi].assign(b1_al, 0);

            uint16_t* b4_cids = reinterpret_cast<uint16_t*>(b_b4[bi].data());
            uint16_t* b1_cids = reinterpret_cast<uint16_t*>(b_b1[bi].data());
            std::vector<double> mean_v(d, 0.0), r_vec(d);
            double doc_mnorm = 0.0;

            for (uint32_t tk = 0; tk < m; ++tk) {
                uint16_t cid = cids_full[s + tk];
                b4_cids[tk] = cid;
                b1_cids[tk] = cid;
                const uint16_t* vf16 = all_vecs_f16 + (s + tk) * d;
                const float* cvec = C_fp32.data() + static_cast<size_t>(cid) * d;
                double vn2 = 0.0, rn2 = 0.0;
                for (uint32_t j = 0; j < d; ++j) {
                    double fv = static_cast<double>(f16_to_f32_scalar(vf16[j]));
                    vn2 += fv * fv;
                    mean_v[j] += fv;
                    double rv = fv - static_cast<double>(cvec[j]);
                    r_vec[j] = rv;
                    rn2 += rv * rv;
                }
                doc_mnorm = std::max(doc_mnorm, std::sqrt(vn2));
                float r_norm = std::nextafter(static_cast<float>(std::sqrt(rn2)), std::numeric_limits<float>::infinity());
                if (r_norm > thr_rho[tid][cid]) thr_rho[tid][cid] = r_norm;

                double s4 = 0.0, e4 = 0.0, s1 = 0.0, e1 = 0.0;
                encode_rabitq_b4_vec(r_vec.data(), P_f64.data(), d, scalar_fp32, t_min_d, s4, e4,
                                     b_b4[bi].data() + b4_pk_off + static_cast<size_t>(tk) * (d / 2));
                encode_rabitq_b1_vec(r_vec.data(), P_f64.data(), d, scalar_fp32, t_min_d, s1, e1,
                                     b_b1[bi].data() + b1_bt_off + static_cast<size_t>(tk) * (d / 8));

                if (scalar_fp32) {
                    reinterpret_cast<float*>(b_b4[bi].data() + b4_sc_off)[tk] = static_cast<float>(s4);
                    reinterpret_cast<float*>(b_b4[bi].data() + b4_sc_off + 4ULL * m)[tk] = static_cast<float>(e4);
                    reinterpret_cast<float*>(b_b1[bi].data() + b1_sc_off)[tk] = static_cast<float>(s1);
                    reinterpret_cast<float*>(b_b1[bi].data() + b1_sc_off + 4ULL * m)[tk] = static_cast<float>(e1);
                } else {
                    reinterpret_cast<uint16_t*>(b_b4[bi].data() + 2ULL * m)[tk] = f32_to_f16_scalar(static_cast<float>(s4));
                    reinterpret_cast<uint16_t*>(b_b4[bi].data() + 4ULL * m)[tk] = f32_to_f16_round_up(e4);
                    reinterpret_cast<uint16_t*>(b_b1[bi].data() + 2ULL * m)[tk] = f32_to_f16_scalar(static_cast<float>(s1));
                    reinterpret_cast<uint16_t*>(b_b1[bi].data() + 4ULL * m)[tk] = f32_to_f16_round_up(e1);
                }
            }
            local_mnorm[bi] = doc_mnorm;
            double inv_m = 1.0 / static_cast<double>(m);
            for (uint32_t j = 0; j < d; ++j) b_avg[bi][j] = f32_to_f16_scalar(static_cast<float>(mean_v[j] * inv_m));
        }

        for (int t = 0; t < nT; ++t) {
            for (uint32_t c = 0; c < K; ++c) rho_max[c] = std::max(rho_max[c], thr_rho[t][c]);
        }
        for (uint64_t bi = 0; bi < cnt; ++bi) {
            uint64_t doc = bs + bi;
            m_norm_max_global = std::max(m_norm_max_global, local_mnorm[bi]);
            t2_meta[doc] = {t2_off, static_cast<uint32_t>(b_t2[bi].size()), lens[doc]};
            ft2.write(reinterpret_cast<const char*>(b_t2[bi].data()), static_cast<std::streamsize>(b_t2[bi].size()));
            t2_off += b_t2[bi].size();

            b4_meta[doc] = {b4_off, static_cast<uint32_t>(b_b4[bi].size()), lens[doc]};
            fb4.write(reinterpret_cast<const char*>(b_b4[bi].data()), static_cast<std::streamsize>(b_b4[bi].size()));
            b4_off += b_b4[bi].size();

            b1_meta[doc] = {b1_off, static_cast<uint32_t>(b_b1[bi].size()), lens[doc]};
            fb1.write(reinterpret_cast<const char*>(b_b1[bi].data()), static_cast<std::streamsize>(b_b1[bi].size()));
            b1_off += b_b1[bi].size();

            favg.write(reinterpret_cast<const char*>(b_avg[bi].data()), static_cast<std::streamsize>(d * 2));
        }
    }
    { std::ofstream fm(out_dir + "/tier2_meta.bin", std::ios::binary); fm.write(reinterpret_cast<const char*>(t2_meta.data()), N * sizeof(DocDiskEntry)); }
    { std::ofstream fm(out_dir + "/tier1_b4_meta.bin", std::ios::binary); fm.write(reinterpret_cast<const char*>(b4_meta.data()), N * sizeof(DocDiskEntry)); }
    { std::ofstream fm(out_dir + "/tier1_b1_meta.bin", std::ios::binary); fm.write(reinterpret_cast<const char*>(b1_meta.data()), N * sizeof(DocDiskEntry)); }

    // 4. Write single flat 64B-aligned tier0_shared.mmap (with 32B-aligned frozen CRoaring blob!)
    {
        std::ofstream f0(out_dir + "/tier0_shared.mmap", std::ios::binary);
        Tier0MmapHeader h{};
        h.magic = kTier0MapMagic; h.n_docs = N; h.K = K; h.d = d;
        std::vector<char> dummy(sizeof(h), 0);
        f0.write(dummy.data(), sizeof(h));
        size_t pos = sizeof(h);

        auto write_sec = [&](const void* ptr, size_t bytes, size_t align) -> uint64_t {
            pad_file_to(f0, pos, align);
            uint64_t off = pos;
            if (bytes > 0) f0.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(bytes));
            pos += bytes;
            return off;
        };

        h.off_C_fp32   = write_sec(C_fp32.data(), C_fp32.size() * 4, 64);
        h.off_C_bf16   = write_sec(C_bf16.data(), C_bf16.size() * 2, 64);
        h.off_C_norms  = write_sec(C_norms.data(), C_norms.size() * 4, 64);
        h.off_C_bf_err = write_sec(C_bf_err.data(), C_bf_err.size() * 4, 64);
        h.off_P_f64    = write_sec(P_f64.data(), P_f64.size() * 8, 64);
        h.off_rho_max  = write_sec(rho_max.data(), rho_max.size() * 4, 64);
        h.off_doc_offs = write_sec(doc_offs.data(), doc_offs.size() * 8, 64);
        h.off_doc_cids = write_sec(doc_cids.data(), doc_cids.size() * 2, 64);
        h.len_doc_cids = doc_cids.size();
        if (keep_full) {
            h.off_doc_full_offs = write_sec(full_offs.data(), full_offs.size() * 8, 64);
            h.off_doc_cids_full = write_sec(cids_full.data(), cids_full.size() * 2, 64);
            h.len_doc_cids_full = cids_full.size();
        }
        // Build 32-byte aligned offsets within frozen blob
        std::vector<uint64_t> foffs(K + 1, 0);
        for (uint32_t c = 0; c < K; ++c) foffs[c + 1] = align_up_sz(foffs[c] + frozen_bufs[c].size(), 32);
        h.off_frozen_offs = write_sec(foffs.data(), foffs.size() * 8, 64);
        pad_file_to(f0, pos, 64);
        h.off_frozen_blob = pos;
        for (uint32_t c = 0; c < K; ++c) {
            f0.write(frozen_bufs[c].data(), static_cast<std::streamsize>(frozen_bufs[c].size()));
            size_t pad = static_cast<size_t>(foffs[c + 1] - foffs[c]) - frozen_bufs[c].size();
            if (pad > 0) { char z[32] = {0}; f0.write(z, static_cast<std::streamsize>(pad)); }
            pos += static_cast<size_t>(foffs[c + 1] - foffs[c]);
        }
        f0.seekp(0);
        f0.write(reinterpret_cast<const char*>(&h), sizeof(h));
    }

    // Write per-field metadata binary snippet for `--write-manifest`
    CertVec5FieldMeta fm{};
    fm.field_id = field_id; fm.d = d; fm.K = K; fm.is_unit = is_unit;
    fm.scalar_fp32 = scalar_fp32 ? 1u : 0u;
    fm.has_doc_cids_full = keep_full ? 1u : 0u;
    fm.has_avg_fp16 = 1u;
    fm.b1_y_unit_pm1 = 1u;
    fm.u_max = u_max;
    fm.u_doc_avg = (N > 0) ? (static_cast<double>(doc_cids.size()) / static_cast<double>(N)) : 0.0;
    fm.m_norm_max = m_norm_max_global;
    { std::ofstream f(out_dir + "/field_meta.bin", std::ios::binary); f.write(reinterpret_cast<const char*>(&fm), sizeof(fm)); }

    std::cout << "[CERTVEC5 field_" << field_id << "] N=" << N << " d=" << d << " K=" << K
              << " toks=" << total_toks << " u_avg=" << std::fixed << std::setprecision(2) << fm.u_doc_avg
              << " M_norm=" << std::setprecision(5) << m_norm_max_global
              << " B1=" << std::setprecision(2) << (b1_off / 1048576.0) << "MB"
              << " B4=" << (b4_off / 1048576.0) << "MB"
              << " T2=" << (t2_off / 1048576.0) << "MB\n";
    return 0;
}

static int write_manifest_cmd(int argc, char** argv) {
    if (argc < 5) { std::cerr << "Usage: --write-manifest <index_dir> <N_docs> <n_fields>\n"; return 1; }
    std::string dir = argv[2];
    uint64_t N = std::stoull(argv[3]);
    uint32_t F = static_cast<uint32_t>(std::stoul(argv[4]));
    CertVec5Manifest man{kCertVec5Magic, N, F, 0};
    std::vector<CertVec5FieldMeta> fms(F);
    for (uint32_t f = 0; f < F; ++f) {
        std::ifstream in(dir + "/field_" + std::to_string(f) + "/field_meta.bin", std::ios::binary);
        if (!in) { std::cerr << "Missing field_meta.bin for field_" << f << "\n"; return 1; }
        in.read(reinterpret_cast<char*>(&fms[f]), sizeof(CertVec5FieldMeta));
    }
    std::ofstream out(dir + "/manifest.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(&man), sizeof(man));
    out.write(reinterpret_cast<const char*>(fms.data()), F * sizeof(CertVec5FieldMeta));
    std::cout << "[CERTVEC5 manifest] dir=" << dir << " N=" << N << " F=" << F << "\n";
    return 0;
}

// Mode 3: Build non-destructive read-only sidecar (frozen_postings.mmap + exact rho_max.bin) for frozen v4 indexes!
static int build_v4_sidecar_cmd(int argc, char** argv) {
    if (argc < 4) { std::cerr << "Usage: --build-v4-sidecar <v4_index_dir> <sidecar_out_dir>\n"; return 1; }
    std::string dir = argv[2], out_dir = argv[3];
    mkdir(out_dir.c_str(), 0755);

    std::ifstream f(dir + "/core_index.bin", std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << dir << "/core_index.bin\n"; return 1; }
    uint64_t hdr[12];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    uint64_t N = hdr[1], K = hdr[2], d = hdr[3], Nq = hdr[4], Lq = hdr[5], n_cl = hdr[6], S_REP = hdr[7];
    uint64_t len_dc = hdr[8], len_inv = hdr[9];
    std::vector<float> C_fp32(K * d);
    f.read(reinterpret_cast<char*>(C_fp32.data()), K * d * 4);
    f.seekg(static_cast<std::streamoff>(d * d * 4ULL + Nq * Lq * d * 4ULL * 2ULL + N + (N + 1) * 8ULL + len_dc * 2ULL
                                        + n_cl * 2 + n_cl * S_REP * 2 + n_cl * S_REP * 8 + n_cl * 2 + n_cl * S_REP * 2), std::ios::cur);
    std::vector<uint64_t> inv_offs(K + 1);
    f.read(reinterpret_cast<char*>(inv_offs.data()), (K + 1) * 8);
    std::vector<uint32_t> inv_docs(len_inv);
    f.read(reinterpret_cast<char*>(inv_docs.data()), len_inv * 4);

    // 1. Serialize 32B-aligned frozen CRoaring postings
    std::vector<std::vector<char>> frozen_bufs(K);
    #pragma omp parallel for schedule(dynamic, 64)
    for (uint64_t c = 0; c < K; ++c) {
        roaring_bitmap_t* r = roaring_bitmap_of_ptr(static_cast<size_t>(inv_offs[c + 1] - inv_offs[c]), inv_docs.data() + inv_offs[c]);
        roaring_bitmap_run_optimize(r);
        roaring_bitmap_shrink_to_fit(r);
        frozen_bufs[c].resize(roaring_bitmap_frozen_size_in_bytes(r));
        roaring_bitmap_frozen_serialize(r, frozen_bufs[c].data());
        roaring_bitmap_free(r);
    }
    std::vector<uint64_t> foffs(K + 1, 0);
    for (uint64_t c = 0; c < K; ++c) foffs[c + 1] = align_up_sz(foffs[c] + frozen_bufs[c].size(), 32);
    size_t hdr_bytes = align_up_sz((2 + K + 2) * 8ULL, 64);
    {
        std::ofstream fo(out_dir + "/frozen_postings.mmap", std::ios::binary);
        uint64_t magic = kFrozenSidecarMagic;
        uint64_t blob_off = hdr_bytes;
        fo.write(reinterpret_cast<const char*>(&magic), 8);
        fo.write(reinterpret_cast<const char*>(&K), 8);
        fo.write(reinterpret_cast<const char*>(foffs.data()), (K + 1) * 8);
        fo.write(reinterpret_cast<const char*>(&blob_off), 8);
        size_t pos = (2 + K + 2) * 8ULL;
        pad_file_to(fo, pos, 64);
        for (uint64_t c = 0; c < K; ++c) {
            fo.write(frozen_bufs[c].data(), static_cast<std::streamsize>(frozen_bufs[c].size()));
            size_t pad = static_cast<size_t>(foffs[c + 1] - foffs[c]) - frozen_bufs[c].size();
            if (pad > 0) { char z[32] = {0}; fo.write(z, static_cast<std::streamsize>(pad)); }
        }
    }

    // 2. Scan tier1_b4.nvme cids + tier2_fp16.nvme to compute exact per-centroid rho_max[K]!
    MappedReadFile t1_meta_m, t2_meta_m;
    t1_meta_m.open_mmap(dir + "/tier1_meta.bin");
    t2_meta_m.open_mmap(dir + "/tier2_meta.bin");
    const auto* t1m = reinterpret_cast<const DocDiskEntry*>(t1_meta_m.ptr);
    const auto* t2m = reinterpret_cast<const DocDiskEntry*>(t2_meta_m.ptr);
    int fd1 = ::open((dir + "/tier1_b4.nvme").c_str(), O_RDONLY);
    int fd2 = ::open((dir + "/tier2_fp16.nvme").c_str(), O_RDONLY);
    std::vector<float> rho_max(K, 0.0f);
    int nT = omp_get_max_threads();
    std::vector<std::vector<float>> thr_rho(nT, std::vector<float>(K, 0.0f));
    #pragma omp parallel for schedule(dynamic, 512)
    for (uint64_t doc = 0; doc < N; ++doc) {
        int tid = omp_get_thread_num();
        uint32_t m = t1m[doc].n_toks;
        if (m == 0) continue;
        std::vector<uint16_t> cids(m);
        std::vector<uint16_t> vecs(static_cast<size_t>(m) * d);
        if (::pread(fd1, cids.data(), m * 2ULL, static_cast<off_t>(t1m[doc].offset)) != static_cast<ssize_t>(m * 2ULL)) continue;
        if (::pread(fd2, vecs.data(), static_cast<size_t>(m) * d * 2ULL, static_cast<off_t>(t2m[doc].offset)) != static_cast<ssize_t>(static_cast<size_t>(m) * d * 2ULL)) continue;
        for (uint32_t tk = 0; tk < m; ++tk) {
            uint16_t cid = cids[tk];
            const float* cvec = C_fp32.data() + static_cast<size_t>(cid) * d;
            double rn2 = 0.0;
            for (uint32_t j = 0; j < d; ++j) {
                double diff = static_cast<double>(f16_to_f32_scalar(vecs[static_cast<size_t>(tk) * d + j])) - static_cast<double>(cvec[j]);
                rn2 += diff * diff;
            }
            float rn = std::nextafter(static_cast<float>(std::sqrt(rn2)), std::numeric_limits<float>::infinity());
            if (rn > thr_rho[tid][cid]) thr_rho[tid][cid] = rn;
        }
    }
    ::close(fd1); ::close(fd2);
    for (int t = 0; t < nT; ++t) for (uint64_t c = 0; c < K; ++c) rho_max[c] = std::max(rho_max[c], thr_rho[t][c]);
    { std::ofstream fr(out_dir + "/rho_max.bin", std::ios::binary); fr.write(reinterpret_cast<const char*>(rho_max.data()), K * 4); }
    float max_r = *std::max_element(rho_max.begin(), rho_max.end());
    std::cout << "[V4 Sidecar] dir=" << dir << " -> " << out_dir << " K=" << K
              << " frozen_MB=" << std::fixed << std::setprecision(2) << (foffs[K] / 1048576.0)
              << " max_rho=" << std::setprecision(4) << max_r << "\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: certvec_builder_v5 [--build-field | --write-manifest | --build-v4-sidecar] ...\n";
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "--build-field") return build_field_cmd(argc, argv);
    if (mode == "--write-manifest") return write_manifest_cmd(argc, argv);
    if (mode == "--build-v4-sidecar") return build_v4_sidecar_cmd(argc, argv);
    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}
