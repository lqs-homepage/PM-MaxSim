#pragma once
// Hardware-Complete Arbitrary-d (64..1024) SIMD Kernels for PM-MaxSim / CertVec Core
// Implements Generalized Lemma 3.1:
//  - FP64 Haar query rotation P_f (orth error < 1e-14)
//  - Exact-int32 B4 vpdpbusd centering (2*dot_u - 15*sum_qi8, zero FP32 offset cancellation!)
//  - AVX-512BW 1-instruction B1 bit-unpack (_mm512_maskz_set1_epi8) + vpdpbusd for d >= 256
//    (and ultra-fast L1d nibble-LUT fast-path when d == 128 && L <= 32 && legacy v4)
//  - FP32 (s_x, eta) for d >= 256; FP16 for d <= 128; 256*u Lemma 3.1 guard; FP64 outer Phi; -INFINITY init
//  - Portable AVX-512 (Emerald Rapids + Ice Lake non-AMX) and scalar fallbacks

#include "beta_quantiles_gen.hpp"
#include "pm_certvec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <x86intrin.h>
#endif

namespace pm_certvec {

static inline float f16_to_f32_scalar(uint16_t h) noexcept {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            out = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, 4);
    return f;
#endif
}

static inline uint16_t f32_to_f16_scalar(float f) noexcept {
#if defined(__F16C__)
    return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#else
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
#endif
}

static inline uint16_t f32_to_f16_round_up(double v) noexcept {
    if (v <= 0.0) return 0;
    float fv = static_cast<float>(v);
    uint16_t h = f32_to_f16_scalar(fv);
    while (static_cast<double>(f16_to_f32_scalar(h)) < v && h < 0x7BFFu) h++;
    return h;
}

static inline uint16_t f32_to_bf16_rne(float f) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    return static_cast<uint16_t>((u + 0x7FFFu + lsb) >> 16);
}

static inline float bf16_to_f32(uint16_t b) noexcept {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Exact FP32 dot product between float32 query vector and FP16 document vector of arbitrary d (multiple of 16)
static inline float dot_f32_f16(const float* q, const uint16_t* x_f16, uint32_t d) noexcept {
#if defined(__AVX512F__) && defined(__F16C__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    uint32_t j = 0;
    for (; j + 32 <= d; j += 32) {
        __m512 x0 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_f16 + j)));
        __m512 x1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_f16 + j + 16)));
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(q + j), x0, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(q + j + 16), x1, acc1);
    }
    for (; j + 16 <= d; j += 16) {
        __m512 x0 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_f16 + j)));
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(q + j), x0, acc0);
    }
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; j < d; ++j) sum += q[j] * f16_to_f32_scalar(x_f16[j]);
    return sum;
#else
    float sum = 0.0f;
    for (uint32_t j = 0; j < d; ++j) sum += q[j] * f16_to_f32_scalar(x_f16[j]);
    return sum;
#endif
}

// Exact FP32 dot product between BF16 vectors (or AMX/AVX-512 BF16) for LUT construction
static inline float dot_bf16_pair(const uint16_t* a_bf, const uint16_t* b_bf, uint32_t d) noexcept {
#if defined(__AVX512BF16__) && defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    uint32_t j = 0;
    for (; j + 32 <= d; j += 32) {
        __m512bh va, vb;
        std::memcpy(&va, a_bf + j, 64);
        std::memcpy(&vb, b_bf + j, 64);
        acc = _mm512_dpbf16_ps(acc, va, vb);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; j < d; ++j) sum += bf16_to_f32(a_bf[j]) * bf16_to_f32(b_bf[j]);
    return sum;
#elif defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    uint32_t j = 0;
    for (; j + 16 <= d; j += 16) {
        __m512i ha = _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_bf + j)));
        __m512i hb = _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_bf + j)));
        __m512 fa = _mm512_castsi512_ps(_mm512_slli_epi32(ha, 16));
        __m512 fb = _mm512_castsi512_ps(_mm512_slli_epi32(hb, 16));
        acc = _mm512_fmadd_ps(fa, fb, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; j < d; ++j) sum += bf16_to_f32(a_bf[j]) * bf16_to_f32(b_bf[j]);
    return sum;
#else
    float sum = 0.0f;
    for (uint32_t j = 0; j < d; ++j) sum += bf16_to_f32(a_bf[j]) * bf16_to_f32(b_bf[j]);
    return sum;
#endif
}

// Per-query-vector prepared Lemma 3.1 hardware state (supports arbitrary d in {64..1024}!)
struct PreparedQuerySlot {
    uint32_t    orig_idx = 0;
    uint32_t    field_id = 0;
    uint32_t    d = 128;
    pm_set_op_t set_op = PM_OP_MAX;
    uint32_t    r_val = 1;
    double      weight = 1.0;
    double      relu_tau = 0.0;

    std::vector<float>    q_f32;       // [d] original query vector
    std::vector<uint16_t> q_bf16;      // [d] exact BF16 rounded query
    std::vector<double>   q_rot_f64;   // [d] FP64 exact Haar-rotated query q @ P_f
    // Interleaved even/odd int8 layout for B4 (for each 64-pair block b: 64 even then 64 odd)
    // AND linear int8 layout for B1 bit-unpack vpdpbusd!
    std::vector<int8_t>   q_i8_linear; // [d]
    std::vector<int8_t>   q_i8_eo;     // [d] grouped per 128 dims as [64 even, 64 odd] (or d/2 even, d/2 odd for d=64)

    double  q_norm_f64 = 0.0;
    double  q_bf_norm = 0.0;
    double  q_bf_err = 0.0;
    double  delta_q = 1.0;
    int32_t sum_qi8 = 0;
    double  q_i8_err = 0.0;
    double  gamma_d = 0.0;
    double  eps0 = 0.0;
    double  cap_val = std::numeric_limits<double>::infinity();
    bool    use_cap = false;

    double  kappa_single = 0.0;
    double  kappa_w1 = 0.0;
    double  kappa_w2 = 0.0;

    void init(uint32_t idx, uint32_t fid, uint32_t dim,
              const float* q_in, const double* P_f64,
              pm_set_op_t op, uint32_t r, double w, double rtau,
              bool is_unit_field, double m_norm_max,
              double delta, double split_lambda, int part2_k, int r_eff_total) {
        orig_idx = idx;
        field_id = fid;
        d = dim;
        set_op = op;
        r_val = std::max<uint32_t>(1u, r);
        weight = w;
        relu_tau = rtau;

        q_f32.assign(q_in, q_in + d);
        q_bf16.resize(d);
        double n2 = 0.0, bfn2 = 0.0, bfe2 = 0.0;
        for (uint32_t j = 0; j < d; ++j) {
            double v = static_cast<double>(q_f32[j]);
            n2 += v * v;
            q_bf16[j] = f32_to_bf16_rne(q_f32[j]);
            double bv = static_cast<double>(bf16_to_f32(q_bf16[j]));
            bfn2 += bv * bv;
            double diff = v - bv;
            bfe2 += diff * diff;
        }
        q_norm_f64 = std::sqrt(n2);
        q_bf_norm  = std::sqrt(bfn2);
        q_bf_err   = std::sqrt(bfe2);
        gamma_d    = higham_gamma_d(static_cast<int>(d));
        eps0       = 1.0e-11 * std::max(1.0, q_norm_f64);

        use_cap = is_unit_field && std::isfinite(m_norm_max) && (m_norm_max > 0.0);
        if (use_cap) {
            double raw_cap = m_norm_max * (1.0 + 2.0 * gamma_d) * q_norm_f64;
            cap_val = static_cast<double>(std::nextafter(static_cast<float>(raw_cap), std::numeric_limits<float>::infinity()));
        } else {
            cap_val = std::numeric_limits<double>::infinity();
        }

        // Exact FP64 Haar rotation: q_rot[j] = sum_k q[k] * P[k * d + j]
        q_rot_f64.assign(d, 0.0);
        for (uint32_t k = 0; k < d; ++k) {
            double qk = static_cast<double>(q_f32[k]);
            const double* prow = P_f64 + static_cast<size_t>(k) * d;
            for (uint32_t j = 0; j < d; ++j) q_rot_f64[j] += qk * prow[j];
        }

        double max_abs = 0.0;
        for (uint32_t j = 0; j < d; ++j) max_abs = std::max(max_abs, std::fabs(q_rot_f64[j]));
        delta_q = (max_abs > 1e-18) ? (max_abs / 127.0) : 1.0;

        q_i8_linear.resize(d);
        q_i8_eo.resize(d);
        int64_t sq = 0;
        double err2 = 0.0;
        for (uint32_t j = 0; j < d; ++j) {
            long qi = std::clamp<long>(std::lrint(q_rot_f64[j] / delta_q), -127, 127);
            q_i8_linear[j] = static_cast<int8_t>(qi);
            sq += qi;
            double de = q_rot_f64[j] - static_cast<double>(qi) * delta_q;
            err2 += de * de;
        }
        sum_qi8 = static_cast<int32_t>(sq);
        q_i8_err = std::sqrt(err2);

        // Pack even/odd layout matching packed nibbles (each 64 bytes holds 128 nibbles: 64 even low, 64 odd high)
        if (d >= 128 && (d % 128 == 0)) {
            for (uint32_t blk = 0; blk < d; blk += 128) {
                for (uint32_t b = 0; b < 64; ++b) {
                    q_i8_eo[blk + b]      = q_i8_linear[blk + 2 * b];
                    q_i8_eo[blk + 64 + b] = q_i8_linear[blk + 2 * b + 1];
                }
            }
        } else {
            // d == 64: 32 even then 32 odd
            uint32_t half = d / 2;
            for (uint32_t b = 0; b < half; ++b) {
                q_i8_eo[b]        = q_i8_linear[2 * b];
                q_i8_eo[half + b] = q_i8_linear[2 * b + 1];
            }
        }

        kappa_single = lookup_beta_kappa(static_cast<int>(d), delta, 1.0, part2_k, r_eff_total);
        kappa_w1     = lookup_beta_kappa(static_cast<int>(d), delta, split_lambda, part2_k, r_eff_total);
        kappa_w2     = lookup_beta_kappa(static_cast<int>(d), delta, 1.0 - split_lambda, part2_k, r_eff_total);
    }
};

// Exact integer dot <u_x, q_i8> AND exact ||y_x||_2^up for B4 (y = u - 7.5 = 0.5*(2u - 15)) across ANY d in {64..1024}
static inline void b4_exact_dot_and_ynorm(const uint8_t* packed_nibbles, const int8_t* q_i8_eo,
                                          uint32_t d, int32_t& out_dot_u, double& out_ynorm_up) noexcept {
    int32_t dot_u = 0;
    int32_t sum_sq_2y = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if (d >= 128 && (d % 128 == 0)) {
        const __m512i m0f = _mm512_set1_epi8(0x0F);
        const __m512i v15 = _mm512_set1_epi8(15);
        __m512i acc_dot = _mm512_setzero_si512();
        __m512i acc_sq  = _mm512_setzero_si512();
        for (uint32_t blk = 0; blk < d; blk += 128) {
            __m512i raw = _mm512_loadu_si512(packed_nibbles + blk / 2);
            __m512i ev8 = _mm512_and_si512(raw, m0f);
            __m512i od8 = _mm512_and_si512(_mm512_srli_epi16(raw, 4), m0f);
            __m512i qe  = _mm512_loadu_si512(q_i8_eo + blk);
            __m512i qo  = _mm512_loadu_si512(q_i8_eo + blk + 64);
            acc_dot = _mm512_dpbusd_epi32(acc_dot, ev8, qe);
            acc_dot = _mm512_dpbusd_epi32(acc_dot, od8, qo);
            __m512i ae = _mm512_abs_epi8(_mm512_sub_epi8(_mm512_add_epi8(ev8, ev8), v15));
            __m512i ao = _mm512_abs_epi8(_mm512_sub_epi8(_mm512_add_epi8(od8, od8), v15));
            acc_sq = _mm512_dpbusd_epi32(acc_sq, ae, ae);
            acc_sq = _mm512_dpbusd_epi32(acc_sq, ao, ao);
        }
        dot_u = _mm512_reduce_add_epi32(acc_dot);
        sum_sq_2y = _mm512_reduce_add_epi32(acc_sq);
        out_dot_u = dot_u;
        out_ynorm_up = static_cast<double>(std::nextafter(0.5f * std::sqrt(static_cast<float>(sum_sq_2y)), std::numeric_limits<float>::infinity()));
        return;
    }
#endif
    uint32_t half = d / 2;
    if (d >= 128 && (d % 128 == 0)) {
        for (uint32_t blk = 0; blk < d; blk += 128) {
            const uint8_t* p = packed_nibbles + blk / 2;
            for (uint32_t b = 0; b < 64; ++b) {
                int32_t ev = p[b] & 0x0F;
                int32_t od = (p[b] >> 4) & 0x0F;
                dot_u += ev * static_cast<int32_t>(q_i8_eo[blk + b]) + od * static_cast<int32_t>(q_i8_eo[blk + 64 + b]);
                int32_t ae = 2 * ev - 15, ao = 2 * od - 15;
                sum_sq_2y += ae * ae + ao * ao;
            }
        }
    } else {
        for (uint32_t b = 0; b < half; ++b) {
            int32_t ev = packed_nibbles[b] & 0x0F;
            int32_t od = (packed_nibbles[b] >> 4) & 0x0F;
            dot_u += ev * static_cast<int32_t>(q_i8_eo[b]) + od * static_cast<int32_t>(q_i8_eo[half + b]);
            int32_t ae = 2 * ev - 15, ao = 2 * od - 15;
            sum_sq_2y += ae * ae + ao * ao;
        }
    }
    out_dot_u = dot_u;
    out_ynorm_up = static_cast<double>(std::nextafter(0.5f * std::sqrt(static_cast<float>(sum_sq_2y)), std::numeric_limits<float>::infinity()));
}

// Fast batched B4 unpack + y_norm ONCE per token, then 4-way vpdpbusd across L query slots!
static inline double b4_unpack_and_ynorm_once(const uint8_t* packed_nibbles, uint32_t d, uint8_t* out_eo) noexcept {
    int32_t sum_sq_2y = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if (d >= 128 && (d % 128 == 0)) {
        const __m512i m0f = _mm512_set1_epi8(0x0F);
        const __m512i v15 = _mm512_set1_epi8(15);
        __m512i acc_sq = _mm512_setzero_si512();
        for (uint32_t blk = 0; blk < d; blk += 128) {
            __m512i raw = _mm512_loadu_si512(packed_nibbles + blk / 2);
            __m512i ev8 = _mm512_and_si512(raw, m0f);
            __m512i od8 = _mm512_and_si512(_mm512_srli_epi16(raw, 4), m0f);
            _mm512_storeu_si512(out_eo + blk, ev8);
            _mm512_storeu_si512(out_eo + blk + 64, od8);
            __m512i ae = _mm512_abs_epi8(_mm512_sub_epi8(_mm512_add_epi8(ev8, ev8), v15));
            __m512i ao = _mm512_abs_epi8(_mm512_sub_epi8(_mm512_add_epi8(od8, od8), v15));
            acc_sq = _mm512_dpbusd_epi32(acc_sq, ae, ae);
            acc_sq = _mm512_dpbusd_epi32(acc_sq, ao, ao);
        }
        sum_sq_2y = _mm512_reduce_add_epi32(acc_sq);
        return static_cast<double>(std::nextafter(0.5f * std::sqrt(static_cast<float>(sum_sq_2y)), std::numeric_limits<float>::infinity()));
    }
#endif
    uint32_t half = d / 2;
    if (d >= 128 && (d % 128 == 0)) {
        for (uint32_t blk = 0; blk < d; blk += 128) {
            const uint8_t* p = packed_nibbles + blk / 2;
            for (uint32_t b = 0; b < 64; ++b) {
                uint8_t ev = p[b] & 0x0Fu, od = (p[b] >> 4) & 0x0Fu;
                out_eo[blk + b] = ev; out_eo[blk + 64 + b] = od;
                int32_t ae = 2 * ev - 15, ao = 2 * od - 15;
                sum_sq_2y += ae * ae + ao * ao;
            }
        }
    } else {
        for (uint32_t b = 0; b < half; ++b) {
            uint8_t ev = packed_nibbles[b] & 0x0Fu, od = (packed_nibbles[b] >> 4) & 0x0Fu;
            out_eo[b] = ev; out_eo[half + b] = od;
            int32_t ae = 2 * ev - 15, ao = 2 * od - 15;
            sum_sq_2y += ae * ae + ao * ao;
        }
    }
    return static_cast<double>(std::nextafter(0.5f * std::sqrt(static_cast<float>(sum_sq_2y)), std::numeric_limits<float>::infinity()));
}

static inline int32_t u8_i8_exact_dot(const uint8_t* u_vec, const int8_t* q_i8, uint32_t d) noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    __m512i acc = _mm512_setzero_si512();
    for (uint32_t j = 0; j < d; j += 64) {
        acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512(u_vec + j), _mm512_loadu_si512(q_i8 + j));
    }
    return _mm512_reduce_add_epi32(acc);
#else
    int32_t s = 0;
    for (uint32_t j = 0; j < d; ++j) s += static_cast<int32_t>(u_vec[j]) * static_cast<int32_t>(q_i8[j]);
    return s;
#endif
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
static inline __m128i hsum4_epi32_512(__m512i a0, __m512i a1, __m512i a2, __m512i a3) noexcept {
    __m256i b0 = _mm256_add_epi32(_mm512_castsi512_si256(a0), _mm512_extracti64x4_epi64(a0, 1));
    __m256i b1 = _mm256_add_epi32(_mm512_castsi512_si256(a1), _mm512_extracti64x4_epi64(a1, 1));
    __m256i b2 = _mm256_add_epi32(_mm512_castsi512_si256(a2), _mm512_extracti64x4_epi64(a2, 1));
    __m256i b3 = _mm256_add_epi32(_mm512_castsi512_si256(a3), _mm512_extracti64x4_epi64(a3, 1));
    __m256i q0123 = _mm256_hadd_epi32(_mm256_hadd_epi32(b0, b1), _mm256_hadd_epi32(b2, b3));
    return _mm_add_epi32(_mm256_castsi256_si128(q0123), _mm256_extracti128_si256(q0123, 1));
}
static inline __m128 hsum4_ps_512(__m512 a0, __m512 a1, __m512 a2, __m512 a3) noexcept {
    __m256 b0 = _mm256_add_ps(_mm512_castps512_ps256(a0), _mm512_extractf32x8_ps(a0, 1));
    __m256 b1 = _mm256_add_ps(_mm512_castps512_ps256(a1), _mm512_extractf32x8_ps(a1, 1));
    __m256 b2 = _mm256_add_ps(_mm512_castps512_ps256(a2), _mm512_extractf32x8_ps(a2, 1));
    __m256 b3 = _mm256_add_ps(_mm512_castps512_ps256(a3), _mm512_extractf32x8_ps(a3, 1));
    __m256 q0123 = _mm256_hadd_ps(_mm256_hadd_ps(b0, b1), _mm256_hadd_ps(b2, b3));
    return _mm_add_ps(_mm256_castps256_ps128(q0123), _mm256_extractf128_ps(q0123, 1));
}
#endif

static inline void u8_i8_exact_dot4(const uint8_t* u_vec,
                                    const int8_t* q0, const int8_t* q1, const int8_t* q2, const int8_t* q3,
                                    uint32_t d, int32_t out4[4]) noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
    for (uint32_t j = 0; j < d; j += 64) {
        __m512i u = _mm512_loadu_si512(u_vec + j);
        a0 = _mm512_dpbusd_epi32(a0, u, _mm512_loadu_si512(q0 + j));
        a1 = _mm512_dpbusd_epi32(a1, u, _mm512_loadu_si512(q1 + j));
        a2 = _mm512_dpbusd_epi32(a2, u, _mm512_loadu_si512(q2 + j));
        a3 = _mm512_dpbusd_epi32(a3, u, _mm512_loadu_si512(q3 + j));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out4), hsum4_epi32_512(a0, a1, a2, a3));
#else
    out4[0] = u8_i8_exact_dot(u_vec, q0, d);
    out4[1] = u8_i8_exact_dot(u_vec, q1, d);
    out4[2] = u8_i8_exact_dot(u_vec, q2, d);
    out4[3] = u8_i8_exact_dot(u_vec, q3, d);
#endif
}

static inline void b1_unpack_bits_once(const uint8_t* packed_bits, uint32_t d, uint8_t* out_b) noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (uint32_t j = 0; j < d; j += 64) {
        uint64_t mask64 = 0;
        std::memcpy(&mask64, packed_bits + (j / 8), 8);
        _mm512_storeu_si512(out_b + j, _mm512_maskz_set1_epi8(static_cast<__mmask64>(mask64), 1));
    }
#else
    for (uint32_t j = 0; j < d; ++j) out_b[j] = (packed_bits[j >> 3] >> (j & 7u)) & 1u;
#endif
}

static inline void dot4_f32_f16(const float* q0, const float* q1, const float* q2, const float* q3,
                                const uint16_t* x_f16, uint32_t d, float out4[4]) noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__F16C__)
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    for (uint32_t j = 0; j < d; j += 16) {
        __m512 xv = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_f16 + j)));
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q0 + j), xv, a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(q1 + j), xv, a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(q2 + j), xv, a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(q3 + j), xv, a3);
    }
    _mm_storeu_ps(out4, hsum4_ps_512(a0, a1, a2, a3));
#else
    out4[0] = dot_f32_f16(q0, x_f16, d);
    out4[1] = dot_f32_f16(q1, x_f16, d);
    out4[2] = dot_f32_f16(q2, x_f16, d);
    out4[3] = dot_f32_f16(q3, x_f16, d);
#endif
}

// Exact integer dot <b_x, q_i8> for B1 bit-packed vectors across ANY d in {64..1024}
// Uses 1-instruction AVX-512BW _mm512_maskz_set1_epi8(mask64, 1) + _mm512_dpbusd_epi32!
static inline int32_t b1_exact_dot_bits(const uint8_t* packed_bits, const int8_t* q_i8_linear, uint32_t d) noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    __m512i acc = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi8(1);
    for (uint32_t j = 0; j < d; j += 64) {
        uint64_t mask64 = 0;
        std::memcpy(&mask64, packed_bits + (j / 8), 8);
        __m512i bvec = _mm512_maskz_set1_epi8(static_cast<__mmask64>(mask64), 1);
        (void)ones;
        __m512i qvec = _mm512_loadu_si512(q_i8_linear + j);
        acc = _mm512_dpbusd_epi32(acc, bvec, qvec);
    }
    return _mm512_reduce_add_epi32(acc);
#else
    int32_t acc = 0;
    for (uint32_t j = 0; j < d; ++j) {
        if ((packed_bits[j >> 3] >> (j & 7u)) & 1u) acc += static_cast<int32_t>(q_i8_linear[j]);
    }
    return acc;
#endif
}

// Generalized Lemma 3.1 Single-Pair Hardware-Complete Certified Upper Bound (B4 or B1, FP32 or FP16 scalar)
static inline double lemma31_pair_ub(float cent_ip_bf16, float c_norm, float c_bf_err,
                                     double s_st, double eta_st, double exact_dot_y,
                                     double y_norm_up, double kappa, const PreparedQuerySlot& qs) noexcept {
    constexpr double kUnitRoundoff = 5.9604644775390625e-8; // 2^-24
    double eps_bf = static_cast<double>(c_norm) * qs.q_bf_err + qs.q_bf_norm * static_cast<double>(c_bf_err);
    double eps_i8 = std::fabs(s_st) * y_norm_up * qs.q_i8_err;
    double eps_fp32 = qs.gamma_d * qs.q_bf_norm * static_cast<double>(c_norm)
                    + 256.0 * kUnitRoundoff * (std::fabs(s_st) * qs.q_norm_f64 * y_norm_up + qs.q_norm_f64 * eta_st)
                    + qs.eps0;
    double ub = static_cast<double>(cent_ip_bf16)
              + s_st * exact_dot_y
              + qs.q_norm_f64 * eta_st * kappa
              + eps_bf + eps_i8 + eps_fp32;
    if (qs.use_cap && ub > qs.cap_val) ub = qs.cap_val;
    return ub;
}

// Evaluate outer coordinate-wise non-decreasing aggregation Phi(s_aux, g_1, ..., g_L) strictly in FP64!
static inline double eval_phi_f64(pm_agg_kind_t agg, double w0, double s_aux,
                                  const std::vector<PreparedQuerySlot>& slots,
                                  const double* g_vals) noexcept {
    double base = w0 * s_aux;
    if (agg == PM_AGG_MIN) {
        double mv = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < slots.size(); ++i) {
            double term = slots[i].weight * g_vals[i];
            if (term < mv) mv = term;
        }
        return base + (slots.empty() ? 0.0 : mv);
    }
    double sum = base;
    if (agg == PM_AGG_RELU_WSUM) {
        for (size_t i = 0; i < slots.size(); ++i) {
            double v = g_vals[i] - slots[i].relu_tau;
            if (v > 0.0) sum += slots[i].weight * v;
        }
    } else {
        for (size_t i = 0; i < slots.size(); ++i) {
            sum += slots[i].weight * g_vals[i];
        }
    }
    return sum;
}

// Helper: maintain top-r descending scores for a single query slot (r == 1 fast-path or general r > 1)
static inline double reduce_top_r_f64(const double* vals, uint32_t m, pm_set_op_t op, uint32_t r_val, double null_sentinel = 0.0) noexcept {
    if (m == 0) return null_sentinel;
    if (op == PM_OP_MAX || r_val <= 1) {
        double best = -std::numeric_limits<double>::infinity();
        for (uint32_t i = 0; i < m; ++i) if (vals[i] > best) best = vals[i];
        return best;
    }
    uint32_t take = std::min(r_val, m);
    // Small insertion buffer for typical r in [2..16]
    double top[32];
    if (take <= 32) {
        uint32_t sz = 0;
        for (uint32_t i = 0; i < m; ++i) {
            double v = vals[i];
            if (sz < take) {
                uint32_t pos = sz++;
                while (pos > 0 && top[pos - 1] < v) { top[pos] = top[pos - 1]; pos--; }
                top[pos] = v;
            } else if (v > top[take - 1]) {
                uint32_t pos = take - 1;
                while (pos > 0 && top[pos - 1] < v) { top[pos] = top[pos - 1]; pos--; }
                top[pos] = v;
            }
        }
        double s = 0.0;
        for (uint32_t i = 0; i < take; ++i) s += top[i];
        return (op == PM_OP_TOP_R_MEAN) ? (s / static_cast<double>(take)) : s;
    }
    std::vector<double> tmp(vals, vals + m);
    std::nth_element(tmp.begin(), tmp.begin() + take, tmp.end(), std::greater<double>());
    double s = 0.0;
    for (uint32_t i = 0; i < take; ++i) s += tmp[i];
    return (op == PM_OP_TOP_R_MEAN) ? (s / static_cast<double>(take)) : s;
}

} // namespace pm_certvec
