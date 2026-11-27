// pm_certvec_core.cpp — Generalized Certified Out-of-Core Vector-Collection & Composite SQL Core Library
// Implements the audited Layer 2 `pm_certvec.h` C ABI, Theorems 1–5, and Hardware-Complete Lemma 3.1.
// Zero-duplication mmap(MAP_SHARED) + roaring_bitmap_frozen_view + shared Unified 4 KiB S3-FIFO Cache.

#include "pm_certvec.h"
#include "beta_quantiles_gen.hpp"
#include "pm_io_backend.hpp"
#include "pm_kernels.hpp"
#include "pm_s3fifo_cache.hpp"

#include "roaring.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <queue>
#include <shared_mutex>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

namespace pm_certvec {

static inline double monotonic_ns() noexcept {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<double>(ts.tv_sec) * 1e9 + static_cast<double>(ts.tv_nsec);
}

struct HeapItem {
    double   score;
    uint32_t rid;
};
// Min-heap comparator: returns true if 'a' is STRICTLY BETTER than 'b', so worst item sits at top!
struct HeapMinCmp {
    bool operator()(const HeapItem& a, const HeapItem& b) const noexcept {
        if (a.score != b.score) return a.score > b.score;
        return a.rid < b.rid;
    }
};
static inline bool is_better_hit(const HeapItem& cand, const HeapItem& top) noexcept {
    if (cand.score != top.score) return cand.score > top.score;
    return cand.rid < top.rid;
}

static inline const roaring_bitmap_t* open_frozen_view_padded(const char* ptr, size_t slot_len) noexcept {
    // Builder 32B-aligns each frozen slot with trailing zero padding (0..31 bytes).
    // CRoaring stores FROZEN_COOKIE at the exact end of the serialized payload and validates
    // exact container-zone byte sums, so trimming trailing zero pad [0..31] recovers the exact view!
    if (!ptr || slot_len < 4) return nullptr;
    for (size_t trim = 0; trim < 32 && slot_len >= 4 + trim; ++trim) {
        const roaring_bitmap_t* rv = roaring_bitmap_frozen_view(ptr, slot_len - trim);
        if (rv != nullptr) return rv;
    }
    return nullptr;
}

static bool file_exists(const std::string& p) noexcept {
    struct stat st{};
    return stat(p.c_str(), &st) == 0;
}

static bool init_amx_if_available(bool disable_amx) noexcept {
#if defined(__AMX_BF16__) && defined(__linux__)
    if (disable_amx) return false;
    return (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0);
#else
    (void)disable_amx;
    return false;
#endif
}

struct TileCfg {
    uint8_t  palette_id, start_row, reserved_0[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

static inline void ensure_amx_thread(bool amx_enabled) noexcept {
#if defined(__AMX_BF16__) && defined(__linux__)
    thread_local bool configured = false;
    if (!amx_enabled || configured) return;
    TileCfg cfg{};
    cfg.palette_id = 1;
    for (int i = 0; i < 8; ++i) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
    _tile_loadconfig(&cfg);
    configured = true;
#else
    (void)amx_enabled;
#endif
}

// Atomic max for double (used by pm_insert_entity to maintain monotonic M_norm_max per Lemma 3.1!)
static inline void atomic_max_double(std::atomic<double>& target, double val) noexcept {
    double cur = target.load(std::memory_order_acquire);
    while (cur < val && !target.compare_exchange_weak(cur, val, std::memory_order_release, std::memory_order_acquire)) {}
}

// ============================================================================
// Binary Manifest & Shared Field Index Layout (CERTVEC5 + Legacy v4 Adapter)
// ============================================================================
constexpr uint64_t kCertVec5Magic = 0x4345525456454335ULL; // "CERTVEC5"
constexpr uint64_t kTier0MapMagic = 0x54304D4D41503035ULL; // "T0MMAP05"
constexpr uint64_t kMNormMagicV4  = 0x4D4E4F524D303034ULL;

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
    uint32_t scalar_fp32;     // 1 for d >= 256, 0 for d <= 128
    uint32_t has_doc_cids_full;
    uint32_t has_avg_fp16;
    uint32_t b1_y_unit_pm1;   // 1 => y in {-1,+1}, 0 => legacy v4 y in {-0.5,+0.5}
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
    uint64_t off_doc_full_offs; // 0 if absent
    uint64_t off_doc_cids_full; // 0 if absent
    uint64_t len_doc_cids_full;
    uint64_t off_frozen_offs;   // [K+1] uint64 byte offsets from off_frozen_blob
    uint64_t off_frozen_blob;   // 32-byte aligned frozen CRoaring views!
};

struct FieldIndex {
    uint32_t field_id = 0;
    uint64_t n_docs = 0;
    uint32_t d = 128;
    uint32_t K = 0;
    bool     is_unit = true;
    bool     scalar_fp32 = false;
    bool     b1_y_unit_pm1 = true;
    bool     has_doc_cids_full = false;
    bool     has_avg_fp16 = false;
    uint64_t u_max = 1;
    double   u_doc_avg = 73.0;
    std::atomic<double> m_norm_max{1.005};

    // Either zero-copy mmap-backed (CERTVEC5 / PG shared) OR heap-owned (legacy v4 fallback)
    MappedReadFile tier0_mmap;
    MappedReadFile avg_fp16_mmap;
    MappedReadFile sidecar_rho_mmap;
    MappedReadFile sidecar_frozen_mmap;

    const float*    C_fp32 = nullptr;
    const uint16_t* C_bf16 = nullptr;
    const float*    C_norms = nullptr;
    const float*    C_bf_err = nullptr;
    const double*   P_f64 = nullptr;
    const float*    rho_max = nullptr; // [K], deterministic max residual norm per centroid!
    const uint64_t* doc_offs = nullptr;
    const uint16_t* doc_cids = nullptr;
    const uint64_t* doc_full_offs = nullptr;
    const uint16_t* doc_cids_full = nullptr;
    const uint16_t* avg_fp16 = nullptr; // [n_docs * d]

    // Frozen CRoaring views (pointing into mmap — zero duplication across 48 PG processes!)
    std::vector<const roaring_bitmap_t*> frozen_postings;
    // Heap-owned buffers only used when opening legacy v4 without sidecars
    AlignedBuf<float>    owned_C_fp32;
    AlignedBuf<uint16_t> owned_C_bf16;
    std::vector<float>   owned_C_norms, owned_C_bf_err, owned_rho_max;
    std::vector<double>  owned_P_f64;
    std::vector<uint64_t> owned_doc_offs;
    std::vector<uint16_t> owned_doc_cids;
    std::vector<roaring_bitmap_t*> owned_postings;

    SharedDiskStore tier1_b1;
    SharedDiskStore tier1_b4;
    SharedDiskStore tier2_fp16;
    bool has_b1 = false;
    uint64_t tier0_ram_bytes = 0;

    ~FieldIndex() {
        for (auto* r : owned_postings) if (r) roaring_bitmap_free(r);
        // Note: frozen_postings created via roaring_bitmap_frozen_view ARE lightweight heap wrappers
        // pointing into mmap buffer; CRoaring requires roaring_bitmap_free() on the view struct
        // (which frees the small wrapper without touching the mmap buffer).
        for (auto* r : frozen_postings) if (r) roaring_bitmap_free(r);
    }

    const roaring_bitmap_t* posting(uint32_t cid) const noexcept {
        if (!frozen_postings.empty()) return frozen_postings[cid];
        return owned_postings[cid];
    }

    bool load_certvec5_field(const std::string& fdir, const CertVec5FieldMeta& fm, uint64_t N,
                             uint16_t fid_base, bool map_b1, bool map_b4, bool use_o_direct) {
        field_id = fm.field_id;
        n_docs = N;
        d = fm.d;
        K = fm.K;
        is_unit = (fm.is_unit != 0);
        scalar_fp32 = (fm.scalar_fp32 != 0);
        b1_y_unit_pm1 = (fm.b1_y_unit_pm1 != 0);
        has_doc_cids_full = (fm.has_doc_cids_full != 0);
        has_avg_fp16 = (fm.has_avg_fp16 != 0);
        u_max = fm.u_max;
        u_doc_avg = fm.u_doc_avg;
        m_norm_max.store(fm.m_norm_max, std::memory_order_relaxed);

        if (!tier0_mmap.open_mmap(fdir + "/tier0_shared.mmap")) return false;
        tier0_ram_bytes = tier0_mmap.len;
        const auto* h = reinterpret_cast<const Tier0MmapHeader*>(tier0_mmap.ptr);
        if (h->magic != kTier0MapMagic || h->K != K || h->d != d || h->n_docs != N) return false;

        C_fp32   = reinterpret_cast<const float*>(tier0_mmap.ptr + h->off_C_fp32);
        C_bf16   = reinterpret_cast<const uint16_t*>(tier0_mmap.ptr + h->off_C_bf16);
        C_norms  = reinterpret_cast<const float*>(tier0_mmap.ptr + h->off_C_norms);
        C_bf_err = reinterpret_cast<const float*>(tier0_mmap.ptr + h->off_C_bf_err);
        P_f64    = reinterpret_cast<const double*>(tier0_mmap.ptr + h->off_P_f64);
        rho_max  = reinterpret_cast<const float*>(tier0_mmap.ptr + h->off_rho_max);
        doc_offs = reinterpret_cast<const uint64_t*>(tier0_mmap.ptr + h->off_doc_offs);
        doc_cids = reinterpret_cast<const uint16_t*>(tier0_mmap.ptr + h->off_doc_cids);
        if (has_doc_cids_full && h->off_doc_full_offs > 0) {
            doc_full_offs = reinterpret_cast<const uint64_t*>(tier0_mmap.ptr + h->off_doc_full_offs);
            doc_cids_full = reinterpret_cast<const uint16_t*>(tier0_mmap.ptr + h->off_doc_cids_full);
        }

        const uint64_t* foffs = reinterpret_cast<const uint64_t*>(tier0_mmap.ptr + h->off_frozen_offs);
        const char* fblob = reinterpret_cast<const char*>(tier0_mmap.ptr + h->off_frozen_blob);
        frozen_postings.resize(K, nullptr);
        for (uint32_t c = 0; c < K; ++c) {
            size_t len = static_cast<size_t>(foffs[c + 1] - foffs[c]);
            frozen_postings[c] = open_frozen_view_padded(fblob + foffs[c], len);
            if (!frozen_postings[c]) {
                std::cerr << "[CERTVEC5 ERROR] Failed to open frozen_view for field=" << field_id << " cid=" << c << " slot_len=" << len << "\n";
                return false;
            }
        }

        if (has_avg_fp16 && file_exists(fdir + "/avg_fp16.mmap")) {
            if (avg_fp16_mmap.open_mmap(fdir + "/avg_fp16.mmap")) {
                avg_fp16 = reinterpret_cast<const uint16_t*>(avg_fp16_mmap.ptr);
                tier0_ram_bytes += avg_fp16_mmap.len;
            }
        }

        if (!tier2_fp16.open_store(fdir + "/tier2_fp16.nvme", fdir + "/tier2_meta.bin",
                                   static_cast<uint16_t>(fid_base + 1), 1, false, use_o_direct)) return false;
        if (!tier1_b4.open_store(fdir + "/tier1_b4.nvme", fdir + "/tier1_b4_meta.bin",
                                 static_cast<uint16_t>(fid_base + 0), 0, map_b4, use_o_direct)) return false;
        if (map_b1 && file_exists(fdir + "/tier1_b1.mmap")) {
            has_b1 = tier1_b1.open_store(fdir + "/tier1_b1.mmap", fdir + "/tier1_b1_meta.bin",
                                         static_cast<uint16_t>(fid_base + 2), 0, true, false);
        }
        return true;
    }

    // Zero-copy backward-compatible loader for frozen Stage 3 v4 single-segment indexes!
    bool load_legacy_v4(const std::string& dir, const std::string& sidecar_dir,
                        uint16_t fid_base, bool map_b1, bool map_b4, bool use_o_direct) {
        std::ifstream f(dir + "/core_index.bin", std::ios::binary);
        if (!f) return false;
        uint64_t hdr[12];
        f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        n_docs = hdr[1]; K = static_cast<uint32_t>(hdr[2]); d = static_cast<uint32_t>(hdr[3]);
        uint64_t Nq = hdr[4], Lq = hdr[5], n_cl = hdr[6], S_REP = hdr[7];
        uint64_t len_dc = hdr[8], len_inv = hdr[9];
        is_unit = (hdr[10] != 0);
        scalar_fp32 = false;
        b1_y_unit_pm1 = false; // Legacy v4 tier1_b1.ram stores y in {-0.5, +0.5}

        owned_C_fp32.resize(static_cast<size_t>(K) * d);
        f.read(reinterpret_cast<char*>(owned_C_fp32.data()), owned_C_fp32.size() * 4);
        std::vector<float> P_f32(static_cast<size_t>(d) * d);
        f.read(reinterpret_cast<char*>(P_f32.data()), P_f32.size() * 4);
        owned_P_f64.resize(static_cast<size_t>(d) * d);
        for (size_t i = 0; i < P_f32.size(); ++i) owned_P_f64[i] = static_cast<double>(P_f32[i]);

        // Skip Q_32, Q_rot, doms
        f.seekg(static_cast<std::streamoff>(Nq * Lq * d * 4ULL * 2ULL + n_docs), std::ios::cur);
        owned_doc_offs.resize(n_docs + 1);
        f.read(reinterpret_cast<char*>(owned_doc_offs.data()), owned_doc_offs.size() * 8);
        u_max = 1;
        for (uint64_t di = 0; di < n_docs; ++di) u_max = std::max(u_max, owned_doc_offs[di + 1] - owned_doc_offs[di]);
        u_doc_avg = (n_docs > 0) ? (static_cast<double>(len_dc) / static_cast<double>(n_docs)) : 73.0;
        owned_doc_cids.resize(len_dc);
        f.read(reinterpret_cast<char*>(owned_doc_cids.data()), len_dc * 2);

        // Skip retired pm_rep + b3_rep
        f.seekg(static_cast<std::streamoff>(n_cl * 2 + n_cl * S_REP * 2 + n_cl * S_REP * 8 + n_cl * 2 + n_cl * S_REP * 2), std::ios::cur);

        // Load sidecar frozen postings FIRST (zero-duplication mmap; avoids allocating 700MB+ inv_docs!)
        bool loaded_frozen = false;
        if (!sidecar_dir.empty() && file_exists(sidecar_dir + "/frozen_postings.mmap")) {
            if (sidecar_frozen_mmap.open_mmap(sidecar_dir + "/frozen_postings.mmap")) {
                const uint64_t* fhdr = reinterpret_cast<const uint64_t*>(sidecar_frozen_mmap.ptr);
                if (fhdr[0] == 0x46524F5A454E3035ULL && fhdr[1] == K) {
                    const uint64_t* foffs = fhdr + 2;
                    const char* fblob = reinterpret_cast<const char*>(sidecar_frozen_mmap.ptr + fhdr[2 + K + 1]);
                    frozen_postings.resize(K, nullptr);
                    bool all_ok = true;
                    for (uint32_t c = 0; c < K; ++c) {
                        frozen_postings[c] = open_frozen_view_padded(fblob + foffs[c], static_cast<size_t>(foffs[c + 1] - foffs[c]));
                        if (!frozen_postings[c]) { all_ok = false; break; }
                    }
                    if (all_ok) {
                        loaded_frozen = true;
                        tier0_ram_bytes += sidecar_frozen_mmap.len;
                    } else {
                        for (auto* r : frozen_postings) if (r) roaring_bitmap_free(r);
                        frozen_postings.clear();
                    }
                }
            }
        }
        if (!loaded_frozen) {
            std::vector<uint64_t> inv_offs(K + 1);
            f.read(reinterpret_cast<char*>(inv_offs.data()), (K + 1) * 8);
            std::vector<uint32_t> inv_docs(len_inv);
            f.read(reinterpret_cast<char*>(inv_docs.data()), len_inv * 4);
            owned_postings.resize(K, nullptr);
            uint64_t rbytes = 0;
            #pragma omp parallel for schedule(dynamic, 64) reduction(+:rbytes)
            for (uint32_t c = 0; c < K; ++c) {
                uint64_t s = inv_offs[c], e = inv_offs[c + 1];
                roaring_bitmap_t* r = roaring_bitmap_of_ptr(static_cast<size_t>(e - s), inv_docs.data() + s);
                roaring_bitmap_run_optimize(r);
                owned_postings[c] = r;
                rbytes += static_cast<uint64_t>(roaring_bitmap_portable_size_in_bytes(r));
            }
            tier0_ram_bytes += rbytes;
        }

        owned_C_norms.resize(K);
        owned_C_bf_err.resize(K);
        owned_C_bf16.resize(static_cast<size_t>(K) * d);
        for (uint32_t c = 0; c < K; ++c) {
            double n2 = 0.0, e2 = 0.0;
            for (uint32_t j = 0; j < d; ++j) {
                float cf = owned_C_fp32[static_cast<size_t>(c) * d + j];
                uint16_t bf = f32_to_bf16_rne(cf);
                owned_C_bf16[static_cast<size_t>(c) * d + j] = bf;
                double diff = static_cast<double>(cf) - static_cast<double>(bf16_to_f32(bf));
                n2 += static_cast<double>(cf) * cf;
                e2 += diff * diff;
            }
            owned_C_norms[c] = static_cast<float>(std::sqrt(n2));
            owned_C_bf_err[c] = static_cast<float>(std::sqrt(e2));
        }

        // Exact M_norm from m_norm.bin
        {
            std::ifstream fn(dir + "/m_norm.bin", std::ios::binary);
            if (fn) {
                struct { uint64_t magic, n, tot; double mx, p99, med; } mh{};
                fn.read(reinterpret_cast<char*>(&mh), sizeof(mh));
                if (mh.magic == kMNormMagicV4 && mh.mx > 0.0) m_norm_max.store(mh.mx, std::memory_order_relaxed);
            }
        }

        // Load sidecar rho_max if present; else conservative deterministic upper bound (m_norm_max + C_norms[c])
        if (!sidecar_dir.empty() && file_exists(sidecar_dir + "/rho_max.bin") && sidecar_rho_mmap.open_mmap(sidecar_dir + "/rho_max.bin")) {
            rho_max = reinterpret_cast<const float*>(sidecar_rho_mmap.ptr);
        } else {
            owned_rho_max.resize(K);
            double mn = m_norm_max.load(std::memory_order_relaxed);
            for (uint32_t c = 0; c < K; ++c) owned_rho_max[c] = static_cast<float>(mn + owned_C_norms[c]);
            rho_max = owned_rho_max.data();
        }

        C_fp32 = owned_C_fp32.data();
        C_bf16 = owned_C_bf16.data();
        C_norms = owned_C_norms.data();
        C_bf_err = owned_C_bf_err.data();
        P_f64 = owned_P_f64.data();
        doc_offs = owned_doc_offs.data();
        doc_cids = owned_doc_cids.data();
        tier0_ram_bytes += owned_C_fp32.size() * 4 + owned_C_bf16.size() * 2 + owned_doc_offs.size() * 8 + owned_doc_cids.size() * 2;

        if (!tier2_fp16.open_store(dir + "/tier2_fp16.nvme", dir + "/tier2_meta.bin",
                                   static_cast<uint16_t>(fid_base + 1), 1, false, use_o_direct)) return false;
        if (!tier1_b4.open_store(dir + "/tier1_b4.nvme", dir + "/tier1_meta.bin",
                                 static_cast<uint16_t>(fid_base + 0), 0, map_b4, use_o_direct)) return false;
        if (map_b1 && file_exists(dir + "/tier1_b1.ram")) {
            has_b1 = tier1_b1.open_store(dir + "/tier1_b1.ram", dir + "/tier1_b1_meta.bin",
                                         static_cast<uint16_t>(fid_base + 2), 0, true, false);
        }
        return true;
    }
};

// ============================================================================
// Live Zero-Remap Dynamic LSM Delta Buffer & Tombstones
// ============================================================================
struct DeltaEntity {
    uint32_t ext_id;
    std::vector<std::vector<uint16_t>> field_vecs_f16; // [F][n_vecs * d_f]
    std::vector<uint32_t>              field_n_vecs;
    std::vector<std::vector<uint16_t>> field_avg_f16;  // [F][d_f]
};

// ============================================================================
// Field LUT Builder (Exact FP32 for L=1; 4-bit L2 Nibble LUT + AMX/AVX-512 for L>1)
// ============================================================================
struct FieldLUT {
    uint32_t K = 0;
    uint32_t L = 0; // active slots targeting this field
    bool     exact_f32_only = false; // L == 1 fast path: 64 KB exact FP32 LUT!
    AlignedBuf<float>   S_bf16; // [K * L_pad] (exact BF16 dot products)
    AlignedBuf<uint8_t> S4;     // [(K+1) * 16 * n_strips] (4-bit quantized per strip of 32)
    std::vector<float>  strip_min, strip_scale;
    std::vector<uint32_t> slot_indices; // indices into PreparedQuerySlot[] targeting this field
};

static void build_field_lut(const FieldIndex& fi, const std::vector<PreparedQuerySlot>& all_slots,
                            const std::vector<uint32_t>& slot_ids, bool amx_enabled, FieldLUT& out) {
    out.K = fi.K;
    out.L = static_cast<uint32_t>(slot_ids.size());
    out.slot_indices = slot_ids;
    if (out.L == 0) return;

    out.exact_f32_only = (out.L == 1);
    uint32_t n_strips = (out.L + 31u) / 32u;
    uint32_t L_pad = out.exact_f32_only ? 1u : (n_strips * 32u);
    out.S_bf16.resize(static_cast<size_t>(fi.K) * L_pad);
    float* S = out.S_bf16.data();

    if (out.exact_f32_only) {
        const auto& qs = all_slots[slot_ids[0]];
        const uint16_t* qbf = qs.q_bf16.data();
        for (uint32_t c = 0; c < fi.K; ++c) {
            S[c] = dot_bf16_pair(fi.C_bf16 + static_cast<size_t>(c) * fi.d, qbf, fi.d);
        }
        return;
    }

    ensure_amx_thread(amx_enabled);
    out.S4.resize(static_cast<size_t>(n_strips) * (fi.K + 1u) * 16u);
    out.strip_min.assign(n_strips, 0.0f);
    out.strip_scale.assign(n_strips, 1.0f);

    for (uint32_t st = 0; st < n_strips; ++st) {
        uint32_t base_t = st * 32u;
        uint32_t cnt = std::min<uint32_t>(32u, out.L - base_t);
        alignas(64) std::vector<uint16_t> Q_bf(32ULL * fi.d, 0);
        for (uint32_t t = 0; t < cnt; ++t) {
            std::memcpy(Q_bf.data() + static_cast<size_t>(t) * fi.d,
                        all_slots[slot_ids[base_t + t]].q_bf16.data(),
                        fi.d * sizeof(uint16_t));
        }

        float vmin = std::numeric_limits<float>::infinity();
        float vmax = -std::numeric_limits<float>::infinity();

#if defined(__AMX_BF16__) && defined(__linux__)
        if (amx_enabled && (fi.K % 32 == 0) && (fi.d % 32 == 0)) {
            uint32_t n_kb = fi.d / 32u;
            std::vector<uint32_t> Q_pack(static_cast<size_t>(n_kb) * 2 * 16 * 16, 0);
            for (uint32_t kb = 0; kb < n_kb; ++kb) {
                for (uint32_t k = 0; k < 16; ++k) {
                    uint32_t d0 = kb * 32 + 2 * k, d1 = d0 + 1;
                    for (uint32_t j = 0; j < 16; ++j) {
                        uint32_t v0 = static_cast<uint32_t>(Q_bf[static_cast<size_t>(j) * fi.d + d0])
                                    | (static_cast<uint32_t>(Q_bf[static_cast<size_t>(j) * fi.d + d1]) << 16);
                        uint32_t v1 = static_cast<uint32_t>(Q_bf[static_cast<size_t>(16 + j) * fi.d + d0])
                                    | (static_cast<uint32_t>(Q_bf[static_cast<size_t>(16 + j) * fi.d + d1]) << 16);
                        Q_pack[((kb * 2 + 0) * 16 + k) * 16 + j] = v0;
                        Q_pack[((kb * 2 + 1) * 16 + k) * 16 + j] = v1;
                    }
                }
            }
            for (uint32_t c = 0; c < fi.K; c += 32) {
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                const uint8_t* cp0 = reinterpret_cast<const uint8_t*>(fi.C_bf16 + static_cast<size_t>(c) * fi.d);
                const uint8_t* cp1 = reinterpret_cast<const uint8_t*>(fi.C_bf16 + static_cast<size_t>(c + 16) * fi.d);
                int stride_c = static_cast<int>(fi.d * 2);
                for (uint32_t kb = 0; kb < n_kb; ++kb) {
                    _tile_loadd(4, cp0 + kb * 64, stride_c);
                    _tile_loadd(5, cp1 + kb * 64, stride_c);
                    _tile_loadd(6, &Q_pack[(kb * 2 + 0) * 256], 64);
                    _tile_loadd(7, &Q_pack[(kb * 2 + 1) * 256], 64);
                    _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(1, 4, 7);
                    _tile_dpbf16ps(2, 5, 6); _tile_dpbf16ps(3, 5, 7);
                }
                alignas(64) float blk0[256], blk1[256], blk2[256], blk3[256];
                _tile_stored(0, blk0, 64); _tile_stored(1, blk1, 64);
                _tile_stored(2, blk2, 64); _tile_stored(3, blk3, 64);
                for (uint32_t r = 0; r < 16; ++r) {
                    float* dst0 = S + static_cast<size_t>(c + r) * L_pad + base_t;
                    float* dst1 = S + static_cast<size_t>(c + 16 + r) * L_pad + base_t;
                    std::memcpy(dst0,      blk0 + r * 16, 64);
                    std::memcpy(dst0 + 16, blk1 + r * 16, 64);
                    std::memcpy(dst1,      blk2 + r * 16, 64);
                    std::memcpy(dst1 + 16, blk3 + r * 16, 64);
                    for (uint32_t t = 0; t < cnt; ++t) {
                        vmin = std::min(vmin, std::min(dst0[t], dst1[t]));
                        vmax = std::max(vmax, std::max(dst0[t], dst1[t]));
                    }
                }
            }
        } else
#endif
        {
            for (uint32_t c = 0; c < fi.K; ++c) {
                const uint16_t* cbf = fi.C_bf16 + static_cast<size_t>(c) * fi.d;
                float* dst = S + static_cast<size_t>(c) * L_pad + base_t;
                for (uint32_t t = 0; t < cnt; ++t) {
                    float v = dot_bf16_pair(cbf, Q_bf.data() + static_cast<size_t>(t) * fi.d, fi.d);
                    dst[t] = v;
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
                for (uint32_t t = cnt; t < 32u; ++t) dst[t] = 0.0f;
            }
        }

        float range = vmax - vmin;
        if (range < 1e-6f) range = 1.0f;
        float inv_scale = 15.0f / range;
        out.strip_min[st] = vmin;
        out.strip_scale[st] = range / 15.0f;

        uint8_t* s4_strip = out.S4.data() + static_cast<size_t>(st) * (fi.K + 1u) * 16u;
        for (uint32_t c = 0; c < fi.K; ++c) {
            const float* src = S + static_cast<size_t>(c) * L_pad + base_t;
            uint8_t* dst = s4_strip + static_cast<size_t>(c) * 16u;
            for (uint32_t b = 0; b < 16u; ++b) {
                int q0 = (2 * b < cnt)     ? std::clamp<int>(static_cast<int>(std::lrintf((src[2 * b]     - vmin) * inv_scale)), 0, 15) : 0;
                int q1 = (2 * b + 1 < cnt) ? std::clamp<int>(static_cast<int>(std::lrintf((src[2 * b + 1] - vmin) * inv_scale)), 0, 15) : 0;
                dst[b] = static_cast<uint8_t>(q0 | (q1 << 4));
            }
        }
    }
}

// Fast K0_4b / Exact-FP32 Tier-0 Entity Scoring (arbitrary L, multi-field, Top-r / Max / Avg, exact s_aux)
static double score_entity_tier0(const std::vector<std::unique_ptr<FieldIndex>>& fields,
                                 const std::vector<FieldLUT>& fluts,
                                 const std::vector<PreparedQuerySlot>& slots,
                                 pm_agg_kind_t agg_kind, double w0, double s_aux,
                                 uint32_t rid) noexcept {
    double g_vals[256];
    const size_t L = slots.size();
    for (size_t i = 0; i < L; ++i) g_vals[i] = 0.0;

    for (size_t f = 0; f < fields.size(); ++f) {
        const auto& flut = fluts[f];
        if (flut.L == 0) continue;
        const auto& fi = *fields[f];

        // Check whether any slot on this field requires r > 1 (non-deduplicated doc_cids_full!)
        bool need_full = false;
        for (uint32_t sid : flut.slot_indices) {
            if (slots[sid].r_val > 1) { need_full = true; break; }
        }
        const uint64_t* offs = (need_full && fi.has_doc_cids_full) ? fi.doc_full_offs : fi.doc_offs;
        const uint16_t* cids = (need_full && fi.has_doc_cids_full) ? fi.doc_cids_full : fi.doc_cids;
        uint64_t cs = offs[rid], ce = offs[rid + 1];
        uint32_t m = static_cast<uint32_t>(ce - cs);

        if (m == 0) {
            for (uint32_t sid : flut.slot_indices) g_vals[sid] = (agg_kind == PM_AGG_MIN) ? -std::numeric_limits<double>::infinity() : 0.0;
            continue;
        }

        if (flut.exact_f32_only) {
            uint32_t sid = flut.slot_indices[0];
            const auto& qs = slots[sid];
            const float* S = flut.S_bf16.data();
            if (qs.set_op == PM_OP_MAX || qs.r_val <= 1) {
                float best = -std::numeric_limits<float>::infinity();
                for (uint64_t p = cs; p < ce; ++p) {
                    float v = S[cids[p]];
                    if (v > best) best = v;
                }
                g_vals[sid] = static_cast<double>(best);
            } else {
                std::vector<double> tmp(m);
                for (uint32_t i = 0; i < m; ++i) tmp[i] = static_cast<double>(S[cids[cs + i]]);
                g_vals[sid] = reduce_top_r_f64(tmp.data(), m, qs.set_op, qs.r_val);
            }
            continue;
        }

        // L > 1 fast 4-bit L2 nibble LUT path (when all slots on field have r_val == 1)
        if (!need_full) {
            uint32_t n_strips = (flut.L + 31u) / 32u;
            for (uint32_t st = 0; st < n_strips; ++st) {
                uint32_t base_t = st * 32u;
                uint32_t cnt = std::min<uint32_t>(32u, flut.L - base_t);
                const uint8_t* s4 = flut.S4.data() + static_cast<size_t>(st) * (fi.K + 1u) * 16u;
                uint8_t lo_max[16], hi_max[16];
                std::memset(lo_max, 0, 16);
                std::memset(hi_max, 0, 16);
#if defined(__AVX512F__) && defined(__AVX512BW__)
                __m128i m0f = _mm_set1_epi8(0x0F);
                __m128i vlo = _mm_setzero_si128(), vhi = _mm_setzero_si128();
                for (uint64_t p = cs; p < ce; ++p) {
                    __m128i row = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s4 + static_cast<size_t>(cids[p]) * 16u));
                    vlo = _mm_max_epu8(vlo, _mm_and_si128(row, m0f));
                    vhi = _mm_max_epu8(vhi, _mm_and_si128(_mm_srli_epi16(row, 4), m0f));
                }
                _mm_storeu_si128(reinterpret_cast<__m128i*>(lo_max), vlo);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(hi_max), vhi);
#else
                for (uint64_t p = cs; p < ce; ++p) {
                    const uint8_t* row = s4 + static_cast<size_t>(cids[p]) * 16u;
                    for (int b = 0; b < 16; ++b) {
                        uint8_t l = row[b] & 0x0Fu, h = row[b] >> 4;
                        if (l > lo_max[b]) lo_max[b] = l;
                        if (h > hi_max[b]) hi_max[b] = h;
                    }
                }
#endif
                float smin = flut.strip_min[st], ssc = flut.strip_scale[st];
                for (uint32_t t = 0; t < cnt; ++t) {
                    uint8_t qv = (t & 1u) ? hi_max[t >> 1] : lo_max[t >> 1];
                    g_vals[flut.slot_indices[base_t + t]] = static_cast<double>(smin + ssc * static_cast<float>(qv));
                }
            }
        } else {
            uint32_t L_pad = ((flut.L + 31u) / 32u) * 32u;
            const float* S = flut.S_bf16.data();
            std::vector<double> tmp(m);
            for (uint32_t local_t = 0; local_t < flut.L; ++local_t) {
                uint32_t sid = flut.slot_indices[local_t];
                for (uint32_t i = 0; i < m; ++i) tmp[i] = static_cast<double>(S[static_cast<size_t>(cids[cs + i]) * L_pad + local_t]);
                g_vals[sid] = reduce_top_r_f64(tmp.data(), m, slots[sid].set_op, slots[sid].r_val);
            }
        }
    }
    return eval_phi_f64(agg_kind, w0, s_aux, slots, g_vals);
}

// Path I Q-Thresh Deterministic Tier-0 Upper Bound using per-centroid rho_max_f[c] (Major-01 closure!)
// Inverted outer-centroid / vectorized-inner loop with ZERO per-doc heap allocation for PM_OP_MAX!
static double score_entity_tier0_det_ub(const std::vector<std::unique_ptr<FieldIndex>>& fields,
                                        const std::vector<FieldLUT>& fluts,
                                        const std::vector<PreparedQuerySlot>& slots,
                                        pm_agg_kind_t agg_kind, double w0, double s_aux,
                                        uint32_t rid) noexcept {
    double g_vals[256];
    for (size_t i = 0; i < slots.size(); ++i) g_vals[i] = 0.0;

    for (size_t f = 0; f < fields.size(); ++f) {
        const auto& flut = fluts[f];
        if (flut.L == 0) continue;
        const auto& fi = *fields[f];
        bool need_full = false;
        for (uint32_t sid : flut.slot_indices) if (slots[sid].r_val > 1) { need_full = true; break; }
        const uint64_t* offs = (need_full && fi.has_doc_cids_full) ? fi.doc_full_offs : fi.doc_offs;
        const uint16_t* cids = (need_full && fi.has_doc_cids_full) ? fi.doc_cids_full : fi.doc_cids;
        uint64_t cs = offs[rid], ce = offs[rid + 1];
        uint32_t m = static_cast<uint32_t>(ce - cs);
        if (m == 0) {
            for (uint32_t sid : flut.slot_indices) g_vals[sid] = (agg_kind == PM_AGG_MIN) ? -std::numeric_limits<double>::infinity() : 0.0;
            continue;
        }
        uint32_t L_pad = flut.exact_f32_only ? 1u : (((flut.L + 31u) / 32u) * 32u);
        const float* S = flut.S_bf16.data();
        if (!need_full) {
            // Fast stack-only MAX path (zero heap allocation, sequential S[cid] row scan!)
            double dmax[256];
            for (uint32_t lt = 0; lt < flut.L; ++lt) dmax[lt] = -std::numeric_limits<double>::infinity();
            for (uint64_t p = cs; p < ce; ++p) {
                uint32_t cid = cids[p];
                const float* row = S + static_cast<size_t>(cid) * L_pad;
                double cn = static_cast<double>(fi.C_norms[cid]);
                double cbe = static_cast<double>(fi.C_bf_err[cid]);
                double rmax = static_cast<double>(fi.rho_max[cid]);
                for (uint32_t lt = 0; lt < flut.L; ++lt) {
                    const auto& qs = slots[flut.slot_indices[lt]];
                    double ub = static_cast<double>(row[lt]) + cn * qs.q_bf_err + qs.q_bf_norm * (cbe + qs.gamma_d * cn) + qs.eps0 + qs.q_norm_f64 * rmax;
                    if (qs.use_cap && ub > qs.cap_val) ub = qs.cap_val;
                    if (ub > dmax[lt]) dmax[lt] = ub;
                }
            }
            for (uint32_t lt = 0; lt < flut.L; ++lt) g_vals[flut.slot_indices[lt]] = dmax[lt];
        } else {
            std::vector<double> tmp(m);
            for (uint32_t local_t = 0; local_t < flut.L; ++local_t) {
                uint32_t sid = flut.slot_indices[local_t];
                const auto& qs = slots[sid];
                for (uint32_t i = 0; i < m; ++i) {
                    uint32_t cid = cids[cs + i];
                    double cip = static_cast<double>(S[static_cast<size_t>(cid) * L_pad + local_t]);
                    double cn = static_cast<double>(fi.C_norms[cid]);
                    double eps_bf = cn * qs.q_bf_err + qs.q_bf_norm * (static_cast<double>(fi.C_bf_err[cid]) + qs.gamma_d * cn) + qs.eps0;
                    double ub = cip + eps_bf + qs.q_norm_f64 * static_cast<double>(fi.rho_max[cid]);
                    if (qs.use_cap && ub > qs.cap_val) ub = qs.cap_val;
                    tmp[i] = ub;
                }
                g_vals[sid] = reduce_top_r_f64(tmp.data(), m, qs.set_op, qs.r_val);
            }
        }
    }
    return eval_phi_f64(agg_kind, w0, s_aux, slots, g_vals);
}

// ============================================================================
// Stage B: Per-Query Packed Field Constants + Hoisted 32 KB L1d B1 Nibble LUT
// Built ONCE per query per active field — eliminates all per-doc/per-token setup!
// ============================================================================
struct FieldQueryPack {
    uint32_t L = 0;
    uint32_t L_pad = 0;
    bool     any_topr = false;
    bool     any_cap = false;
    bool     has_b1_nib = false;

    // Contiguous 64B-aligned per-slot FP64 arrays (for 4-wide __m256d exact-FP64 Lemma 3.1!)
    AlignedBuf<double>  half_dq;          // 0.5 * delta_q (for B4)
    AlignedBuf<double>  b1_dq;            // scale_mul * delta_q (for B1)
    AlignedBuf<int32_t> sum_qi8;          // exact int32 sum(q_i8)
    AlignedBuf<double>  qbfe_plus_g;      // q_bf_err + gamma_d * q_bf_norm
    AlignedBuf<double>  qbfn;             // q_bf_norm
    AlignedBuf<double>  qi8_plus_256u;    // q_i8_err + 256 * u * q_norm
    AlignedBuf<double>  kw1_plus_256u;    // (kappa_w1 + 256 * u) * q_norm
    AlignedBuf<double>  kw2_plus_256u;    // (kappa_w2 + 256 * u) * q_norm
    AlignedBuf<double>  ksg_plus_256u;    // (kappa_single + 256 * u) * q_norm
    AlignedBuf<double>  eps0;             // qs.eps0
    AlignedBuf<double>  cap_val;          // qs.cap_val (or +inf)

    // Hoisted 32 KB L1d-resident 1-bit nibble LUT for d == 128 && L <= 32 && !scalar_fp32 && !any_topr
    alignas(64) int16_t b1_nib[32][16][32];
    alignas(64) float   qb1s[32], qb1e[32], qbfe_f[32], qbfn_f[32], qetm_w1[32], qetm_sg[32], qeps_f[32], qcap_f[32];

    void init(const FieldIndex& fi, const FieldLUT& flut, const std::vector<PreparedQuerySlot>& slots, bool build_b1_lut) {
        L = flut.L;
        if (L == 0) return;
        L_pad = ((L + 3u) / 4u) * 4u;
        any_topr = false;
        any_cap = false;
        for (uint32_t lt = 0; lt < L; ++lt) {
            const auto& qs = slots[flut.slot_indices[lt]];
            if (qs.r_val > 1) any_topr = true;
            if (qs.use_cap) any_cap = true;
        }

        constexpr double kUnitRoundoff = 5.9604644775390625e-8; // 2^-24
        const double scale_mul = fi.b1_y_unit_pm1 ? 1.0 : 0.5;

        half_dq.resize(L_pad);       b1_dq.resize(L_pad);
        sum_qi8.resize(L_pad);       qbfe_plus_g.resize(L_pad);
        qbfn.resize(L_pad);          qi8_plus_256u.resize(L_pad);
        kw1_plus_256u.resize(L_pad); kw2_plus_256u.resize(L_pad);
        ksg_plus_256u.resize(L_pad); eps0.resize(L_pad);
        cap_val.resize(L_pad);

        for (uint32_t lt = 0; lt < L_pad; ++lt) {
            if (lt < L) {
                const auto& qs = slots[flut.slot_indices[lt]];
                half_dq[lt]       = 0.5 * qs.delta_q;
                b1_dq[lt]         = scale_mul * qs.delta_q;
                sum_qi8[lt]       = qs.sum_qi8;
                qbfe_plus_g[lt]   = qs.q_bf_err + qs.gamma_d * qs.q_bf_norm;
                qbfn[lt]          = qs.q_bf_norm;
                qi8_plus_256u[lt] = qs.q_i8_err + 256.0 * kUnitRoundoff * qs.q_norm_f64;
                kw1_plus_256u[lt] = (qs.kappa_w1 + 256.0 * kUnitRoundoff) * qs.q_norm_f64;
                kw2_plus_256u[lt] = (qs.kappa_w2 + 256.0 * kUnitRoundoff) * qs.q_norm_f64;
                ksg_plus_256u[lt] = (qs.kappa_single + 256.0 * kUnitRoundoff) * qs.q_norm_f64;
                eps0[lt]          = qs.eps0;
                cap_val[lt]       = qs.use_cap ? qs.cap_val : std::numeric_limits<double>::infinity();
            } else {
                half_dq[lt] = b1_dq[lt] = qbfe_plus_g[lt] = qbfn[lt] = qi8_plus_256u[lt] = 0.0;
                kw1_plus_256u[lt] = kw2_plus_256u[lt] = ksg_plus_256u[lt] = eps0[lt] = 0.0;
                sum_qi8[lt] = 0;
                cap_val[lt] = std::numeric_limits<double>::infinity();
            }
        }

        has_b1_nib = false;
#if defined(__AVX512F__) && defined(__AVX512BW__)
        if (build_b1_lut && fi.d == 128 && L <= 32 && !fi.scalar_fp32 && !any_topr) {
            has_b1_nib = true;
            const double y_val = fi.b1_y_unit_pm1 ? 1.0 : 0.5;
            const double y_norm_up = fi.b1_y_unit_pm1
                ? static_cast<double>(std::nextafter(std::sqrt(128.0f), std::numeric_limits<float>::infinity()))
                : static_cast<double>(std::nextafter(0.5f * std::sqrt(128.0f), std::numeric_limits<float>::infinity()));
            const double sc_mult = fi.b1_y_unit_pm1 ? 12.0 : 6.0;
            auto fup = [](double v) noexcept -> float {
                return std::nextafter(static_cast<float>(v), std::numeric_limits<float>::infinity());
            };
            for (uint32_t t = 0; t < 32; ++t) {
                qb1s[t] = qb1e[t] = qbfe_f[t] = qbfn_f[t] = qetm_w1[t] = qetm_sg[t] = qeps_f[t] = 0.0f;
                qcap_f[t] = std::numeric_limits<float>::infinity();
            }
            for (uint32_t lt = 0; lt < L; ++lt) {
                const auto& qs = slots[flut.slot_indices[lt]];
                double sc = std::max(1e-12, (sc_mult * qs.q_norm_f64) / 30000.0);
                qb1s[lt]    = static_cast<float>(sc);
                qb1e[lt]    = fup(16.0 * sc + 256.0 * kUnitRoundoff * qs.q_norm_f64 * y_norm_up + 4.0e-5 * qs.q_norm_f64);
                qbfe_f[lt]  = fup(qs.q_bf_err + qs.gamma_d * qs.q_bf_norm);
                qbfn_f[lt]  = fup(qs.q_bf_norm);
                qetm_w1[lt] = fup((qs.kappa_w1 + 256.0 * kUnitRoundoff) * qs.q_norm_f64);
                qetm_sg[lt] = fup((qs.kappa_single + 256.0 * kUnitRoundoff) * qs.q_norm_f64);
                qeps_f[lt]  = fup(qs.eps0 + 4.0e-6 * std::max(1.0, qs.q_norm_f64));
                if (qs.use_cap) qcap_f[lt] = fup(qs.cap_val);
            }
            for (int g = 0; g < 32; ++g) {
                for (int mask = 0; mask < 16; ++mask) {
                    int16_t* dst = b1_nib[g][mask];
                    for (uint32_t lt = 0; lt < L; ++lt) {
                        const double* qr = slots[flut.slot_indices[lt]].q_rot_f64.data() + g * 4;
                        double s = ((mask & 1) ? y_val : -y_val) * qr[0] + ((mask & 2) ? y_val : -y_val) * qr[1]
                                 + ((mask & 4) ? y_val : -y_val) * qr[2] + ((mask & 8) ? y_val : -y_val) * qr[3];
                        dst[lt] = static_cast<int16_t>(std::clamp<long>(std::lrint(s / static_cast<double>(qb1s[lt])), -30000, 30000));
                    }
                    for (uint32_t lt = L; lt < 32; ++lt) dst[lt] = 0;
                }
            }
        }
#else
        (void)build_b1_lut;
#endif
    }
};

// ============================================================================
// Stage B: Hardware-Complete Tier-1 (B1 / B4) Certified UB & Tier-2 Exact FP16 Verification
// Zero per-doc heap allocation for PM_OP_MAX; hoisted B1 L1d LUT + 4-wide __m256d FP64 SIMD!
// ============================================================================
static double compute_field_tier1_b1_ub(const FieldIndex& fi, const FieldLUT& flut,
                                        const FieldQueryPack& pack,
                                        const std::vector<PreparedQuerySlot>& slots,
                                        const uint8_t* buf, uint32_t m, bool split_wave1,
                                        double* out_g_vals) noexcept {
    if (m == 0) {
        for (uint32_t sid : flut.slot_indices) out_g_vals[sid] = 0.0;
        return 0.0;
    }
    const uint16_t* cids = reinterpret_cast<const uint16_t*>(buf);
    const float*    sc_f32 = nullptr;
    const float*    et_f32 = nullptr;
    const uint16_t* sc_f16 = nullptr;
    const uint16_t* et_f16 = nullptr;
    const uint8_t*  bits = nullptr;
    if (fi.scalar_fp32) {
        size_t off_sc = align_up_sz(2ULL * m, 4);
        sc_f32 = reinterpret_cast<const float*>(buf + off_sc);
        et_f32 = reinterpret_cast<const float*>(buf + off_sc + 4ULL * m);
        size_t off_bits = align_up_sz(off_sc + 8ULL * m, 16);
        bits = buf + off_bits;
    } else {
        sc_f16 = reinterpret_cast<const uint16_t*>(buf + 2ULL * m);
        et_f16 = reinterpret_cast<const uint16_t*>(buf + 4ULL * m);
        bits = buf + 6ULL * m;
    }

    const uint32_t bit_stride = fi.d / 8u;
    const uint32_t L_pad = flut.exact_f32_only ? 1u : (((flut.L + 31u) / 32u) * 32u);

    // Ultra-fast HOISTED 32 KB L1d-resident 1-bit nibble-LUT path for d == 128 && L <= 32!
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (pack.has_b1_nib) {
        const float* qetm = split_wave1 ? pack.qetm_w1 : pack.qetm_sg;
        __m512 dmax0 = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        __m512 dmax1 = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        const __m512 vs0 = _mm512_load_ps(pack.qb1s + 0),   vs1 = _mm512_load_ps(pack.qb1s + 16);
        const __m512 ve0 = _mm512_load_ps(pack.qb1e + 0),   ve1 = _mm512_load_ps(pack.qb1e + 16);
        const __m512 vb0 = _mm512_load_ps(pack.qbfe_f + 0), vb1 = _mm512_load_ps(pack.qbfe_f + 16);
        const __m512 vn0 = _mm512_load_ps(pack.qbfn_f + 0), vn1 = _mm512_load_ps(pack.qbfn_f + 16);
        const __m512 vt0 = _mm512_load_ps(qetm + 0),        vt1 = _mm512_load_ps(qetm + 16);
        const __m512 vp0 = _mm512_load_ps(pack.qeps_f + 0), vp1 = _mm512_load_ps(pack.qeps_f + 16);
        const __m512 vc0 = _mm512_load_ps(pack.qcap_f + 0), vc1 = _mm512_load_ps(pack.qcap_f + 16);
        const __m512 vabs_mask = _mm512_set1_ps(-0.0f);
        for (uint32_t tk = 0; tk < m; ++tk) {
            const uint8_t* s = bits + static_cast<size_t>(tk) * 16u;
            __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
            for (int b = 0; b < 16; ++b) {
                uint8_t byte = s[b];
                a0 = _mm512_add_epi16(a0, _mm512_load_si512(pack.b1_nib[2 * b + 0][byte & 0x0F]));
                a1 = _mm512_add_epi16(a1, _mm512_load_si512(pack.b1_nib[2 * b + 1][byte >> 4]));
            }
            __m512i sum_i16 = _mm512_add_epi16(a0, a1);
            __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_i16))), vs0);
            __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_i16, 1))), vs1);
            uint32_t cid = cids[tk];
            const float* cs = flut.S_bf16.data() + static_cast<size_t>(cid) * L_pad;
            __m512 vsc = _mm512_set1_ps(f16_to_f32_scalar(sc_f16[tk]));
            __m512 vabs_sc = _mm512_andnot_ps(vabs_mask, vsc);
            __m512 vet = _mm512_set1_ps(f16_to_f32_scalar(et_f16[tk]));
            __m512 vcn = _mm512_set1_ps(fi.C_norms[cid]);
            __m512 vcbe = _mm512_set1_ps(fi.C_bf_err[cid]);
            __m512 bf0 = _mm512_fmadd_ps(vcn, vb0, _mm512_fmadd_ps(vcbe, vn0, vp0));
            __m512 bf1 = _mm512_fmadd_ps(vcn, vb1, _mm512_fmadd_ps(vcbe, vn1, vp1));
            __m512 r0 = _mm512_fmadd_ps(vet, vt0, _mm512_fmadd_ps(vabs_sc, ve0, bf0));
            __m512 r1 = _mm512_fmadd_ps(vet, vt1, _mm512_fmadd_ps(vabs_sc, ve1, bf1));
            __m512 u0 = _mm512_add_ps(_mm512_loadu_ps(cs + 0), _mm512_fmadd_ps(vsc, f0, r0));
            __m512 u1 = _mm512_add_ps(_mm512_loadu_ps(cs + 16), _mm512_fmadd_ps(vsc, f1, r1));
            if (pack.any_cap) { u0 = _mm512_min_ps(u0, vc0); u1 = _mm512_min_ps(u1, vc1); }
            dmax0 = _mm512_max_ps(dmax0, u0);
            dmax1 = _mm512_max_ps(dmax1, u1);
        }
        alignas(64) float dm[32];
        _mm512_store_ps(dm + 0, dmax0);
        _mm512_store_ps(dm + 16, dmax1);
        for (uint32_t lt = 0; lt < flut.L; ++lt) out_g_vals[flut.slot_indices[lt]] = static_cast<double>(dm[lt]);
        return 0.0;
    }
#endif
    const double y_norm_up = fi.b1_y_unit_pm1
        ? static_cast<double>(std::nextafter(std::sqrt(static_cast<float>(fi.d)), std::numeric_limits<float>::infinity()))
        : static_cast<double>(std::nextafter(0.5f * std::sqrt(static_cast<float>(fi.d)), std::numeric_limits<float>::infinity()));
    const double scale_mul = fi.b1_y_unit_pm1 ? 1.0 : 0.5;
    const double* k_qn_arr = split_wave1 ? pack.kw1_plus_256u.data() : pack.ksg_plus_256u.data();

    alignas(64) uint8_t b_unpacked[1024];
    alignas(64) double dmax[256];
    std::vector<double> tok_ubs;
    if (!pack.any_topr) {
        for (uint32_t lt = 0; lt < pack.L_pad; ++lt) dmax[lt] = -std::numeric_limits<double>::infinity();
    } else {
        tok_ubs.resize(static_cast<size_t>(flut.L) * m);
    }

    for (uint32_t tk = 0; tk < m; ++tk) {
        uint32_t cid = cids[tk];
        double s_st = fi.scalar_fp32 ? static_cast<double>(sc_f32[tk]) : static_cast<double>(f16_to_f32_scalar(sc_f16[tk]));
        double e_st = fi.scalar_fp32 ? static_cast<double>(et_f32[tk]) : static_cast<double>(f16_to_f32_scalar(et_f16[tk]));
        b1_unpack_bits_once(bits + static_cast<size_t>(tk) * bit_stride, fi.d, b_unpacked);
        const float* cs = flut.S_bf16.data() + static_cast<size_t>(cid) * L_pad;
        double cn = static_cast<double>(fi.C_norms[cid]);
        double cbe = static_cast<double>(fi.C_bf_err[cid]);
        double abs_s_yn = std::fabs(s_st) * y_norm_up;

        uint32_t lt = 0;
#if defined(__AVX2__)
        const __m256d vs_st = _mm256_set1_pd(s_st);
        const __m256d veta = _mm256_set1_pd(e_st);
        const __m256d vabs_syn = _mm256_set1_pd(abs_s_yn);
        const __m256d vcn = _mm256_set1_pd(cn);
        const __m256d vcbe = _mm256_set1_pd(cbe);
        for (; lt + 4 <= flut.L; lt += 4) {
            const auto& q0 = slots[flut.slot_indices[lt + 0]];
            const auto& q1 = slots[flut.slot_indices[lt + 1]];
            const auto& q2 = slots[flut.slot_indices[lt + 2]];
            const auto& q3 = slots[flut.slot_indices[lt + 3]];
            int32_t d4[4];
            u8_i8_exact_dot4(b_unpacked, q0.q_i8_linear.data(), q1.q_i8_linear.data(), q2.q_i8_linear.data(), q3.q_i8_linear.data(), fi.d, d4);
            alignas(32) double cent4[4] = {
                static_cast<double>(2LL * d4[0] - pack.sum_qi8[lt + 0]),
                static_cast<double>(2LL * d4[1] - pack.sum_qi8[lt + 1]),
                static_cast<double>(2LL * d4[2] - pack.sum_qi8[lt + 2]),
                static_cast<double>(2LL * d4[3] - pack.sum_qi8[lt + 3])
            };
            __m256d vdot_y = _mm256_mul_pd(_mm256_load_pd(cent4), _mm256_load_pd(pack.b1_dq.data() + lt));
            __m256d vcs    = _mm256_cvtps_pd(_mm_loadu_ps(cs + lt));
            __m256d ub = _mm256_add_pd(vcs,
                         _mm256_fmadd_pd(vs_st, vdot_y,
                         _mm256_fmadd_pd(veta, _mm256_load_pd(k_qn_arr + lt),
                         _mm256_fmadd_pd(vabs_syn, _mm256_load_pd(pack.qi8_plus_256u.data() + lt),
                         _mm256_fmadd_pd(vcn, _mm256_load_pd(pack.qbfe_plus_g.data() + lt),
                         _mm256_fmadd_pd(vcbe, _mm256_load_pd(pack.qbfn.data() + lt),
                                         _mm256_load_pd(pack.eps0.data() + lt)))))));
            if (pack.any_cap) ub = _mm256_min_pd(ub, _mm256_load_pd(pack.cap_val.data() + lt));
            if (!pack.any_topr) {
                _mm256_store_pd(dmax + lt, _mm256_max_pd(_mm256_load_pd(dmax + lt), ub));
            } else {
                alignas(32) double u4[4];
                _mm256_store_pd(u4, ub);
                for (int k = 0; k < 4; ++k) tok_ubs[static_cast<size_t>(lt + k) * m + tk] = u4[k];
            }
        }
#endif
        for (; lt < flut.L; ++lt) {
            const auto& qs = slots[flut.slot_indices[lt]];
            int32_t dot_b = u8_i8_exact_dot(b_unpacked, qs.q_i8_linear.data(), fi.d);
            double exact_dot_y = static_cast<double>(2LL * dot_b - qs.sum_qi8) * (scale_mul * qs.delta_q);
            double ub = lemma31_pair_ub(cs[lt], fi.C_norms[cid], fi.C_bf_err[cid], s_st, e_st, exact_dot_y, y_norm_up, split_wave1 ? qs.kappa_w1 : qs.kappa_single, qs);
            if (!pack.any_topr) { if (ub > dmax[lt]) dmax[lt] = ub; }
            else tok_ubs[static_cast<size_t>(lt) * m + tk] = ub;
        }
    }
    if (!pack.any_topr) {
        for (uint32_t lt = 0; lt < flut.L; ++lt) out_g_vals[flut.slot_indices[lt]] = dmax[lt];
    } else {
        for (uint32_t lt = 0; lt < flut.L; ++lt) {
            uint32_t sid = flut.slot_indices[lt];
            out_g_vals[sid] = reduce_top_r_f64(tok_ubs.data() + static_cast<size_t>(lt) * m, m, slots[sid].set_op, slots[sid].r_val);
        }
    }
    return 0.0;
}

static double compute_field_tier1_b4_ub(const FieldIndex& fi, const FieldLUT& flut,
                                        const FieldQueryPack& pack,
                                        const std::vector<PreparedQuerySlot>& slots,
                                        const uint8_t* buf, uint32_t m, bool split_wave2,
                                        bool uncertified_approx, double* out_g_vals) noexcept {
    if (m == 0) {
        for (uint32_t sid : flut.slot_indices) out_g_vals[sid] = 0.0;
        return 0.0;
    }
    const uint16_t* cids = reinterpret_cast<const uint16_t*>(buf);
    const float*    sc_f32 = nullptr;
    const float*    et_f32 = nullptr;
    const uint16_t* sc_f16 = nullptr;
    const uint16_t* et_f16 = nullptr;
    const uint8_t*  packed = nullptr;
    if (fi.scalar_fp32) {
        size_t off_sc = align_up_sz(2ULL * m, 4);
        sc_f32 = reinterpret_cast<const float*>(buf + off_sc);
        et_f32 = reinterpret_cast<const float*>(buf + off_sc + 4ULL * m);
        size_t off_pk = align_up_sz(off_sc + 8ULL * m, 64);
        packed = buf + off_pk;
    } else {
        sc_f16 = reinterpret_cast<const uint16_t*>(buf + 2ULL * m);
        et_f16 = reinterpret_cast<const uint16_t*>(buf + 4ULL * m);
        size_t off_pk = align_up_sz(6ULL * m, 64);
        packed = buf + off_pk;
    }

    const uint32_t nib_stride = fi.d / 2u;
    const uint32_t L_pad = flut.exact_f32_only ? 1u : (((flut.L + 31u) / 32u) * 32u);
    const double* k_qn_arr = split_wave2 ? pack.kw2_plus_256u.data() : pack.ksg_plus_256u.data();

    alignas(64) uint8_t u_eo[1024];
    alignas(64) double dmax[256];
    std::vector<double> tok_ubs;
    if (!pack.any_topr) {
        for (uint32_t lt = 0; lt < pack.L_pad; ++lt) dmax[lt] = -std::numeric_limits<double>::infinity();
    } else {
        tok_ubs.resize(static_cast<size_t>(flut.L) * m);
    }

    for (uint32_t tk = 0; tk < m; ++tk) {
        uint32_t cid = cids[tk];
        double s_st = fi.scalar_fp32 ? static_cast<double>(sc_f32[tk]) : static_cast<double>(f16_to_f32_scalar(sc_f16[tk]));
        double e_st = fi.scalar_fp32 ? static_cast<double>(et_f32[tk]) : static_cast<double>(f16_to_f32_scalar(et_f16[tk]));
        double y_norm_up = b4_unpack_and_ynorm_once(packed + static_cast<size_t>(tk) * nib_stride, fi.d, u_eo);
        const float* cs = flut.S_bf16.data() + static_cast<size_t>(cid) * L_pad;
        double cn = static_cast<double>(fi.C_norms[cid]);
        double cbe = static_cast<double>(fi.C_bf_err[cid]);
        double abs_s_yn = std::fabs(s_st) * y_norm_up;

        uint32_t lt = 0;
#if defined(__AVX2__)
        const __m256d vs_st = _mm256_set1_pd(s_st);
        const __m256d veta = _mm256_set1_pd(e_st);
        const __m256d vabs_syn = _mm256_set1_pd(abs_s_yn);
        const __m256d vcn = _mm256_set1_pd(cn);
        const __m256d vcbe = _mm256_set1_pd(cbe);
        for (; lt + 4 <= flut.L; lt += 4) {
            const auto& q0 = slots[flut.slot_indices[lt + 0]];
            const auto& q1 = slots[flut.slot_indices[lt + 1]];
            const auto& q2 = slots[flut.slot_indices[lt + 2]];
            const auto& q3 = slots[flut.slot_indices[lt + 3]];
            int32_t d4[4];
            u8_i8_exact_dot4(u_eo, q0.q_i8_eo.data(), q1.q_i8_eo.data(), q2.q_i8_eo.data(), q3.q_i8_eo.data(), fi.d, d4);
            alignas(32) double cent4[4] = {
                static_cast<double>(2LL * d4[0] - 15LL * pack.sum_qi8[lt + 0]),
                static_cast<double>(2LL * d4[1] - 15LL * pack.sum_qi8[lt + 1]),
                static_cast<double>(2LL * d4[2] - 15LL * pack.sum_qi8[lt + 2]),
                static_cast<double>(2LL * d4[3] - 15LL * pack.sum_qi8[lt + 3])
            };
            __m256d vdot_y = _mm256_mul_pd(_mm256_load_pd(cent4), _mm256_load_pd(pack.half_dq.data() + lt));
            __m256d vcs    = _mm256_cvtps_pd(_mm_loadu_ps(cs + lt));
            __m256d ub;
            if (uncertified_approx) {
                ub = _mm256_fmadd_pd(vs_st, vdot_y, vcs);
            } else {
                ub = _mm256_add_pd(vcs,
                     _mm256_fmadd_pd(vs_st, vdot_y,
                     _mm256_fmadd_pd(veta, _mm256_load_pd(k_qn_arr + lt),
                     _mm256_fmadd_pd(vabs_syn, _mm256_load_pd(pack.qi8_plus_256u.data() + lt),
                     _mm256_fmadd_pd(vcn, _mm256_load_pd(pack.qbfe_plus_g.data() + lt),
                     _mm256_fmadd_pd(vcbe, _mm256_load_pd(pack.qbfn.data() + lt),
                                     _mm256_load_pd(pack.eps0.data() + lt)))))));
                if (pack.any_cap) ub = _mm256_min_pd(ub, _mm256_load_pd(pack.cap_val.data() + lt));
            }
            if (!pack.any_topr) {
                _mm256_store_pd(dmax + lt, _mm256_max_pd(_mm256_load_pd(dmax + lt), ub));
            } else {
                alignas(32) double u4[4];
                _mm256_store_pd(u4, ub);
                for (int k = 0; k < 4; ++k) tok_ubs[static_cast<size_t>(lt + k) * m + tk] = u4[k];
            }
        }
#endif
        for (; lt < flut.L; ++lt) {
            const auto& qs = slots[flut.slot_indices[lt]];
            int32_t dot_u = u8_i8_exact_dot(u_eo, qs.q_i8_eo.data(), fi.d);
            double exact_dot_y = static_cast<double>(2LL * dot_u - 15LL * qs.sum_qi8) * (0.5 * qs.delta_q);
            double ub = uncertified_approx
                ? (static_cast<double>(cs[lt]) + s_st * exact_dot_y)
                : lemma31_pair_ub(cs[lt], fi.C_norms[cid], fi.C_bf_err[cid], s_st, e_st, exact_dot_y, y_norm_up, split_wave2 ? qs.kappa_w2 : qs.kappa_single, qs);
            if (!pack.any_topr) { if (ub > dmax[lt]) dmax[lt] = ub; }
            else tok_ubs[static_cast<size_t>(lt) * m + tk] = ub;
        }
    }
    if (!pack.any_topr) {
        for (uint32_t lt = 0; lt < flut.L; ++lt) out_g_vals[flut.slot_indices[lt]] = dmax[lt];
    } else {
        for (uint32_t lt = 0; lt < flut.L; ++lt) {
            uint32_t sid = flut.slot_indices[lt];
            out_g_vals[sid] = reduce_top_r_f64(tok_ubs.data() + static_cast<size_t>(lt) * m, m, slots[sid].set_op, slots[sid].r_val);
        }
    }
    return 0.0;
}

static void compute_field_tier2_exact(const FieldIndex& fi, const FieldLUT& flut,
                                      const std::vector<PreparedQuerySlot>& slots,
                                      const uint8_t* buf, uint32_t m, uint32_t rid,
                                      double* out_g_vals) noexcept {
    const uint16_t* x_f16 = reinterpret_cast<const uint16_t*>(buf);
    bool all_max = true;
    for (uint32_t sid : flut.slot_indices) if (slots[sid].set_op != PM_OP_MAX || slots[sid].r_val > 1) { all_max = false; break; }
    if (all_max && m > 0) {
        double dmax[256];
        for (uint32_t lt = 0; lt < flut.L; ++lt) dmax[lt] = -std::numeric_limits<double>::infinity();
        for (uint32_t tk = 0; tk < m; ++tk) {
            const uint16_t* xp = x_f16 + static_cast<size_t>(tk) * fi.d;
            uint32_t lt = 0;
            for (; lt + 4 <= flut.L; lt += 4) {
                float d4[4];
                dot4_f32_f16(slots[flut.slot_indices[lt + 0]].q_f32.data(),
                             slots[flut.slot_indices[lt + 1]].q_f32.data(),
                             slots[flut.slot_indices[lt + 2]].q_f32.data(),
                             slots[flut.slot_indices[lt + 3]].q_f32.data(), xp, fi.d, d4);
                for (int k = 0; k < 4; ++k) if (static_cast<double>(d4[k]) > dmax[lt + k]) dmax[lt + k] = static_cast<double>(d4[k]);
            }
            for (; lt < flut.L; ++lt) {
                double v = static_cast<double>(dot_f32_f16(slots[flut.slot_indices[lt]].q_f32.data(), xp, fi.d));
                if (v > dmax[lt]) dmax[lt] = v;
            }
        }
        for (uint32_t lt = 0; lt < flut.L; ++lt) out_g_vals[flut.slot_indices[lt]] = dmax[lt];
        return;
    }

    for (uint32_t lt = 0; lt < flut.L; ++lt) {
        uint32_t sid = flut.slot_indices[lt];
        const auto& qs = slots[sid];
        if (qs.set_op == PM_OP_AVG && fi.has_avg_fp16 && fi.avg_fp16 != nullptr) {
            out_g_vals[sid] = static_cast<double>(dot_f32_f16(qs.q_f32.data(), fi.avg_fp16 + static_cast<size_t>(rid) * fi.d, fi.d));
            continue;
        }
        if (m == 0) { out_g_vals[sid] = 0.0; continue; }
        std::vector<double> tmp(m);
        for (uint32_t tk = 0; tk < m; ++tk) {
            tmp[tk] = static_cast<double>(dot_f32_f16(qs.q_f32.data(), x_f16 + static_cast<size_t>(tk) * fi.d, fi.d));
        }
        out_g_vals[sid] = reduce_top_r_f64(tmp.data(), m, qs.set_op, qs.r_val);
    }
}

// ============================================================================
// Auxiliary Score Lookup Helper (Dense / Sparse / None)
// ============================================================================
struct AuxLookup {
    pm_aux_kind_t kind = PM_AUX_NONE;
    const float*  dense = nullptr;
    double        w0 = 0.0;
    double        def_val = 0.0;
    std::unordered_map<uint32_t, float> sparse_map;
    std::vector<std::pair<uint32_t, float>> sparse_vec;

    void init(const pm_aux_scores_t* aux) {
        if (!aux || aux->kind == PM_AUX_NONE) { kind = PM_AUX_NONE; w0 = 0.0; return; }
        kind = aux->kind;
        w0 = static_cast<double>(aux->w0);
        def_val = static_cast<double>(aux->default_val);
        if (kind == PM_AUX_DENSE_F32) {
            dense = aux->scores;
        } else if (kind == PM_AUX_SPARSE_PAIRS && aux->rids && aux->scores) {
            sparse_map.reserve(aux->count * 2 + 16);
            sparse_vec.reserve(aux->count);
            for (size_t i = 0; i < aux->count; ++i) {
                sparse_map[aux->rids[i]] = aux->scores[i];
                sparse_vec.emplace_back(aux->rids[i], aux->scores[i]);
            }
        }
    }
    inline double get(uint32_t rid) const noexcept {
        if (kind == PM_AUX_NONE) return 0.0;
        if (kind == PM_AUX_DENSE_F32) return static_cast<double>(dense[rid]);
        auto it = sparse_map.find(rid);
        return (it != sparse_map.end()) ? static_cast<double>(it->second) : def_val;
    }
};

// ============================================================================
// Filter Bitmap Abstraction (CRoaring / Frozen / RID Array / Dense Bitset / None)
// ============================================================================
struct FilterGuard {
    const roaring_bitmap_t* bm = nullptr;
    roaring_bitmap_t*       owned = nullptr;
    bool                    unfiltered = false;
    uint64_t                n_docs = 0;
    uint64_t                n_valid = 0;

    bool init(const pm_bitmap_view_t* v, uint64_t N) {
        n_docs = N;
        if (!v || v->kind == PM_BM_NONE || v->data == nullptr) {
            unfiltered = true;
            n_valid = N;
            return true;
        }
        if (v->kind == PM_BM_ROARING_PTR) {
            bm = static_cast<const roaring_bitmap_t*>(v->data);
            n_valid = roaring_bitmap_get_cardinality(bm);
            return true;
        }
        if (v->kind == PM_BM_FROZEN_ROARING) {
            owned = const_cast<roaring_bitmap_t*>(roaring_bitmap_frozen_view(static_cast<const char*>(v->data), v->len));
            bm = owned;
            n_valid = roaring_bitmap_get_cardinality(bm);
            return true;
        }
        if (v->kind == PM_BM_RID_ARRAY) {
            owned = roaring_bitmap_of_ptr(v->len, static_cast<const uint32_t*>(v->data));
            roaring_bitmap_run_optimize(owned);
            bm = owned;
            n_valid = roaring_bitmap_get_cardinality(bm);
            return true;
        }
        if (v->kind == PM_BM_BITSET64) {
            owned = roaring_bitmap_create();
            const uint64_t* words = static_cast<const uint64_t*>(v->data);
            for (size_t w = 0; w < v->len; ++w) {
                uint64_t bits = words[w];
                uint32_t base = static_cast<uint32_t>(w * 64ULL);
                while (bits) {
                    uint32_t tz = static_cast<uint32_t>(__builtin_ctzll(bits));
                    uint32_t rid = base + tz;
                    if (rid < N) roaring_bitmap_add(owned, rid);
                    bits &= bits - 1ULL;
                }
            }
            roaring_bitmap_run_optimize(owned);
            bm = owned;
            n_valid = roaring_bitmap_get_cardinality(bm);
            return true;
        }
        return false;
    }

    inline bool contains(uint32_t rid) const noexcept {
        if (unfiltered) return rid < n_docs;
        return roaring_bitmap_contains(bm, rid);
    }

    ~FilterGuard() { if (owned) roaring_bitmap_free(owned); }
};

} // namespace pm_certvec

// ============================================================================
// Main Handle Definition (`struct pm_handle`)
// ============================================================================
struct pm_handle {
    std::string index_dir;
    uint64_t    n_docs = 0;
    uint32_t    n_fields = 0;
    std::vector<std::unique_ptr<pm_certvec::FieldIndex>> fields;

    pm_certvec::UnifiedS3FifoCache unified_cache;
    pm_tier_mode_t  active_tier = PM_TIER_SPLIT_B1B4;
    pm_io_backend_t io_backend = PM_IO_URING;
    bool            amx_enabled = false;
    bool            use_o_direct = true;

    // Startup hardware self-calibration parameters (Theorem 4A / 5)
    double   calibrated_alpha_k0_ns = 0.65;
    double   calibrated_ov_ms = 0.90;
    uint64_t calibrated_tau1_base = 25000;
    uint32_t calibrated_qd = 64;

    // Live Zero-Remap Dynamic LSM state (protected by shared_mutex: concurrent reads, exclusive writes)
    mutable std::shared_mutex lsm_mtx;
    std::vector<pm_certvec::DeltaEntity> delta_buf;
    std::unordered_set<uint32_t> tombstones;
    std::unordered_map<uint32_t, size_t> delta_ext_to_idx;

    void run_self_calibration() {
        if (fields.empty() || n_docs == 0) return;
        const auto& f0 = *fields[0];
        uint32_t sample_docs = static_cast<uint32_t>(std::min<uint64_t>(n_docs, 8192ULL));
        // Synthetic 4-bit LUT scan probe to measure exact host c_K0 (ns/cid)
        std::vector<uint8_t> dummy_lut((static_cast<size_t>(f0.K) + 1u) * 16u, 7u);
        uint64_t total_cids = 0;
        double t0 = pm_certvec::monotonic_ns();
        volatile uint32_t sink = 0;
        for (int rep = 0; rep < 4; ++rep) {
            for (uint32_t d = 0; d < sample_docs; ++d) {
                uint64_t cs = f0.doc_offs[d], ce = f0.doc_offs[d + 1];
                total_cids += (ce - cs);
                uint8_t mx = 0;
                for (uint64_t p = cs; p < ce; ++p) {
                    uint8_t v = dummy_lut[static_cast<size_t>(f0.doc_cids[p]) * 16u];
                    if (v > mx) mx = v;
                }
                sink += mx;
            }
        }
        double elapsed_ns = pm_certvec::monotonic_ns() - t0;
        (void)sink;
        if (total_cids > 0 && elapsed_ns > 1e3) {
            calibrated_alpha_k0_ns = std::clamp(elapsed_ns / static_cast<double>(total_cids), 0.20, 4.00);
        }
        calibrated_tau1_base = compute_dynamic_tau1(16, 32);
    }

    uint64_t compute_dynamic_tau1(uint32_t nprobe, uint32_t n_active_q) const noexcept {
        if (fields.empty()) return 16384;
        const auto& f0 = *fields[0];
        double mp = std::min<double>(static_cast<double>(f0.K), static_cast<double>(n_active_q) * static_cast<double>(nprobe) * 0.85);
        double lam = (mp * f0.u_doc_avg) / static_cast<double>(std::max<uint32_t>(1u, f0.K));
        double kappa_u = 1.0 - std::exp(-lam);
        if (kappa_u >= 0.92) return n_docs + 1;
        double tau = (calibrated_ov_ms * 1e6) / (std::max(1.0, f0.u_doc_avg) * calibrated_alpha_k0_ns * (1.0 - kappa_u));
        return static_cast<uint64_t>(std::clamp(tau, 8192.0, static_cast<double>(n_docs) + 1.0));
    }
};

// Thread-local I/O reader pool (reuses io_uring ring & dynamic 256 KB bounce buffer per thread!)
static pm_certvec::ThreadPageReader& get_thread_reader(pm_io_backend_t backend, uint32_t qd) {
    thread_local pm_certvec::ThreadPageReader reader;
    thread_local bool initialized = false;
    thread_local pm_io_backend_t last_backend = PM_IO_PREAD_POOL;
    if (!initialized || last_backend != backend) {
        reader.init(backend, qd, 256);
        initialized = true;
        last_backend = backend;
    }
    return reader;
}

extern "C" {

size_t pm_s3fifo_required_shmem_bytes(size_t cache_data_bytes) {
    pm_certvec::S3GlobalHeader hdr{};
    return pm_certvec::compute_s3fifo_layout(cache_data_bytes, hdr);
}

pm_handle_t* pm_open(const char* index_dir, const pm_open_opts_t* opts_in) {
    if (!index_dir) return nullptr;
    pm_open_opts_t opts{};
    if (opts_in) opts = *opts_in;
    else {
        opts.tier_mode = PM_TIER_SPLIT_B1B4;
        opts.io_backend = PM_IO_URING;
        opts.self_calibrate = 1;
        opts.use_o_direct = 1;
    }

    auto h = std::make_unique<pm_handle>();
    h->index_dir = index_dir;
    h->io_backend = opts.io_backend;
    h->use_o_direct = (opts.use_o_direct != 0);
    h->amx_enabled = pm_certvec::init_amx_if_available(opts.disable_amx != 0);

    std::string dir(index_dir);
    std::string sc_dir = opts.sidecar_dir ? std::string(opts.sidecar_dir) : std::string();

    bool is_certvec5 = pm_certvec::file_exists(dir + "/manifest.bin");
    bool is_legacy_v4 = (!is_certvec5 && pm_certvec::file_exists(dir + "/core_index.bin"));
    if (!is_certvec5 && !is_legacy_v4) return nullptr;

    // Inspect total B1/B4 sizes first for Theorem 5 KKT tier selection!
    uint64_t total_b1_bytes = 0, total_b4_bytes = 0;
    std::vector<pm_certvec::CertVec5FieldMeta> fmetas;
    if (is_certvec5) {
        std::ifstream fm(dir + "/manifest.bin", std::ios::binary);
        pm_certvec::CertVec5Manifest man{};
        fm.read(reinterpret_cast<char*>(&man), sizeof(man));
        if (man.magic != pm_certvec::kCertVec5Magic || man.n_fields == 0) return nullptr;
        h->n_docs = man.n_docs;
        h->n_fields = man.n_fields;
        fmetas.resize(man.n_fields);
        fm.read(reinterpret_cast<char*>(fmetas.data()), man.n_fields * sizeof(pm_certvec::CertVec5FieldMeta));
        for (uint32_t f = 0; f < man.n_fields; ++f) {
            struct stat st{};
            std::string fdir = dir + "/field_" + std::to_string(f);
            if (stat((fdir + "/tier1_b1.mmap").c_str(), &st) == 0) total_b1_bytes += static_cast<uint64_t>(st.st_size);
            if (stat((fdir + "/tier1_b4.nvme").c_str(), &st) == 0) total_b4_bytes += static_cast<uint64_t>(st.st_size);
        }
    } else {
        h->n_fields = 1;
        struct stat st{};
        if (stat((dir + "/tier1_b1.ram").c_str(), &st) == 0) total_b1_bytes = static_cast<uint64_t>(st.st_size);
        if (stat((dir + "/tier1_b4.nvme").c_str(), &st) == 0) total_b4_bytes = static_cast<uint64_t>(st.st_size);
    }

    // Theorem 5 Continuous KKT Tier & Unified S3-FIFO Cache Selection!
    // Exact continuous crossover M_rem* = M_B4 - (Cw1 / (C1 - Cw1)) * M_B1 (~0.917 * M_B4)
    bool map_b1 = false, map_b4 = false;
    size_t cache_budget = 0;
    if (opts.tier_mode == PM_TIER_AUTO_KKT && opts.ram_budget_bytes > 0) {
        double m_rem_star = static_cast<double>(total_b4_bytes) - (160.0 / 440.0) * static_cast<double>(total_b1_bytes);
        if (opts.ram_budget_bytes >= total_b4_bytes && total_b4_bytes > 0) {
            h->active_tier = PM_TIER_DRAM_T1;
            map_b4 = true;
            cache_budget = opts.ram_budget_bytes - total_b4_bytes;
        } else if (static_cast<double>(opts.ram_budget_bytes) >= m_rem_star && total_b4_bytes > 0) {
            // Above continuous crossover M_rem*: drop B1, give full budget to unified 4 KiB S3-FIFO (caches B4 + hot T2)!
            h->active_tier = PM_TIER_NVME_T1;
            cache_budget = opts.ram_budget_bytes;
        } else if (total_b1_bytes > 0 && opts.ram_budget_bytes >= total_b1_bytes) {
            h->active_tier = PM_TIER_SPLIT_B1B4;
            map_b1 = true;
            cache_budget = opts.ram_budget_bytes - total_b1_bytes;
        } else {
            h->active_tier = PM_TIER_NVME_T1;
            cache_budget = opts.ram_budget_bytes;
        }
    } else {
        h->active_tier = (opts.tier_mode == PM_TIER_AUTO_KKT) ? PM_TIER_SPLIT_B1B4 : opts.tier_mode;
        map_b1 = (h->active_tier == PM_TIER_SPLIT_B1B4);
        map_b4 = (h->active_tier == PM_TIER_DRAM_T1);
        cache_budget = opts.ram_budget_bytes;
    }

    h->fields.resize(h->n_fields);
    if (is_certvec5) {
        for (uint32_t f = 0; f < h->n_fields; ++f) {
            h->fields[f] = std::make_unique<pm_certvec::FieldIndex>();
            if (!h->fields[f]->load_certvec5_field(dir + "/field_" + std::to_string(f), fmetas[f], h->n_docs,
                                                   static_cast<uint16_t>(f * 4u), map_b1, map_b4, h->use_o_direct)) {
                return nullptr;
            }
        }
    } else {
        h->fields[0] = std::make_unique<pm_certvec::FieldIndex>();
        if (!h->fields[0]->load_legacy_v4(dir, sc_dir, 0, map_b1, map_b4, h->use_o_direct)) return nullptr;
        h->n_docs = h->fields[0]->n_docs;
    }

    if (h->active_tier == PM_TIER_SPLIT_B1B4 && !h->fields[0]->has_b1) {
        h->active_tier = PM_TIER_NVME_T1;
    }

    // Initialize Unified 4 KiB S3-FIFO Page Cache (either caller-provided PG Shared Memory OR local heap)
    if (opts.shared_cache_mem && opts.shared_cache_bytes >= 64 * 16 * 4096ULL) {
        h->unified_cache.attach_region(opts.shared_cache_mem, opts.shared_cache_bytes, false);
    } else if (cache_budget >= 64 * 16 * 4096ULL) {
        h->unified_cache.init_heap(cache_budget);
    }

    if (opts.self_calibrate) h->run_self_calibration();
    return h.release();
}

int pm_execute_query(
    pm_handle_t*            h,
    const pm_bitmap_view_t* filter_bm,
    const pm_aux_scores_t*  aux_scores,
    const pm_query_spec_t*  qspec,
    pm_hit_t*               topk_out,
    uint32_t*               out_count,
    pm_emit_cb_t            thresh_cb,
    void*                   thresh_ctx,
    pm_query_stats_t*       stats_out) {
    if (!h || !qspec || !qspec->q_vecs || qspec->L == 0) return -EINVAL;
    if (qspec->mode == PM_MODE_TOPK && (!topk_out || !out_count || qspec->k == 0)) return -EINVAL;
    if (qspec->mode == PM_MODE_THRESHOLD && !thresh_cb && (!topk_out || !out_count)) return -EINVAL;

    // Strict input validation: reject negative vector-component weights w_i < 0 with -EINVAL!
    if (qspec->weights) {
        for (uint32_t i = 0; i < qspec->L; ++i) {
            if (qspec->weights[i] < 0.0f || !std::isfinite(qspec->weights[i])) return -EINVAL;
        }
    }

    double t_start = pm_certvec::monotonic_ns();
    pm_query_stats_t st{};

    pm_certvec::FilterGuard filt;
    if (!filt.init(filter_bm, h->n_docs)) return -EINVAL;
    st.valid_filtered_docs = filt.n_valid;

    pm_certvec::AuxLookup aux;
    aux.init(aux_scores);

    // Build active query slots (excluding zero-norm and zero-weight slots from L and R_eff!)
    double delta = (qspec->delta > 0.0 && qspec->delta < 0.5) ? qspec->delta : 0.05;
    double split_lam = (qspec->split_lambda > 0.01 && qspec->split_lambda < 0.99) ? qspec->split_lambda : 0.2;
    int part2_k = (qspec->part2_allk && qspec->mode == PM_MODE_TOPK) ? static_cast<int>(qspec->k) : 1;

    // First pass: count R_eff(Q) across active non-zero slots
    struct RawSlot {
        uint32_t    orig_i;
        uint32_t    fid;
        const float* ptr;
        pm_set_op_t op;
        uint32_t    r;
        double      w;
        double      rtau;
    };
    std::vector<RawSlot> raw_slots;
    raw_slots.reserve(qspec->L);
    int r_eff_total = 0;
    size_t q_off = 0;
    bool any_r_gt_1 = false;
    for (uint32_t i = 0; i < qspec->L; ++i) {
        uint32_t fid = qspec->field_ids ? qspec->field_ids[i] : 0u;
        if (fid >= h->n_fields) return -EINVAL;
        uint32_t d = h->fields[fid]->d;
        const float* qp = qspec->q_vecs + q_off;
        q_off += d;
        double w = qspec->weights ? static_cast<double>(qspec->weights[i]) : 1.0;
        if (w == 0.0) continue; // Zero-weight slot excluded from L and R_eff!
        double n2 = 0.0;
        for (uint32_t j = 0; j < d; ++j) n2 += static_cast<double>(qp[j]) * qp[j];
        if (n2 <= 1e-8) continue; // Zero-padding slot excluded!
        pm_set_op_t op = qspec->set_ops ? qspec->set_ops[i] : PM_OP_MAX;
        uint32_t rv = (op == PM_OP_TOP_R_SUM || op == PM_OP_TOP_R_MEAN)
            ? std::max<uint32_t>(1u, qspec->r_vals ? qspec->r_vals[i] : 1u) : 1u;
        if (rv > 1) any_r_gt_1 = true;
        double rtau = qspec->relu_taus ? static_cast<double>(qspec->relu_taus[i]) : 0.0;
        raw_slots.push_back({i, fid, qp, op, rv, w, rtau});
        r_eff_total += static_cast<int>((op == PM_OP_AVG) ? 1u : rv);
    }
    if (raw_slots.empty()) {
        if (out_count) *out_count = 0;
        if (stats_out) *stats_out = st;
        return 0;
    }

    // Prepare FP64-rotated Lemma 3.1 hardware state per active query slot
    std::vector<pm_certvec::PreparedQuerySlot> slots(raw_slots.size());
    std::vector<std::vector<uint32_t>> field_slot_ids(h->n_fields);
    for (size_t i = 0; i < raw_slots.size(); ++i) {
        const auto& rs = raw_slots[i];
        const auto& fi = *h->fields[rs.fid];
        slots[i].init(static_cast<uint32_t>(i), rs.fid, fi.d, rs.ptr, fi.P_f64,
                      rs.op, rs.r, rs.w, rs.rtau, fi.is_unit,
                      fi.m_norm_max.load(std::memory_order_acquire),
                      delta, split_lam, part2_k, r_eff_total);
        field_slot_ids[rs.fid].push_back(static_cast<uint32_t>(i));
    }

    // Snapshot live LSM tombstones & evaluate delta buffer under a <20us shared_lock,
    // then RELEASE lsm_mtx completely before Stage A + Stage B NVMe I/O (zero writer-convoy stalls!)
    std::priority_queue<pm_certvec::HeapItem, std::vector<pm_certvec::HeapItem>, pm_certvec::HeapMinCmp> exact_heap;
    std::vector<pm_hit_t> thresh_hits;
    std::unordered_set<uint32_t> local_tombs;
    {
        std::shared_lock<std::shared_mutex> lsm_rlock(h->lsm_mtx);
        if (!h->tombstones.empty()) local_tombs = h->tombstones;
        if (!h->delta_buf.empty()) {
            std::vector<double> g_vals(slots.size(), 0.0);
            for (const auto& de : h->delta_buf) {
                if (!local_tombs.empty() && local_tombs.count(de.ext_id)) continue;
                if (!filt.unfiltered && !roaring_bitmap_contains(filt.bm, de.ext_id)) continue;
                for (size_t sid = 0; sid < slots.size(); ++sid) {
                    const auto& qs = slots[sid];
                    uint32_t f = qs.field_id;
                    if (qs.set_op == PM_OP_AVG && !de.field_avg_f16[f].empty()) {
                        g_vals[sid] = static_cast<double>(pm_certvec::dot_f32_f16(qs.q_f32.data(), de.field_avg_f16[f].data(), qs.d));
                    } else {
                        uint32_t m = de.field_n_vecs[f];
                        if (m == 0) { g_vals[sid] = 0.0; continue; }
                        std::vector<double> tmp(m);
                        const uint16_t* vptr = de.field_vecs_f16[f].data();
                        for (uint32_t tk = 0; tk < m; ++tk) {
                            tmp[tk] = static_cast<double>(pm_certvec::dot_f32_f16(qs.q_f32.data(), vptr + static_cast<size_t>(tk) * qs.d, qs.d));
                        }
                        g_vals[sid] = pm_certvec::reduce_top_r_f64(tmp.data(), m, qs.set_op, qs.r_val);
                    }
                }
                double ex = pm_certvec::eval_phi_f64(qspec->agg_kind, aux.w0, aux.get(de.ext_id), slots, g_vals.data());
                if (qspec->mode == PM_MODE_THRESHOLD) {
                    if (ex >= qspec->tau) thresh_hits.push_back({de.ext_id, ex});
                } else {
                    pm_certvec::HeapItem item{ex, de.ext_id};
                    if (exact_heap.size() < qspec->k) exact_heap.push(item);
                    else if (pm_certvec::is_better_hit(item, exact_heap.top())) { exact_heap.pop(); exact_heap.push(item); }
                }
            }
        }
    }

    // Build per-field centroid LUTs (exact FP32 for L_f==1; 4-bit L2 LUT + AMX/AVX-512 for L_f>1)
    double t_lut0 = pm_certvec::monotonic_ns();
    std::vector<pm_certvec::FieldLUT> fluts(h->n_fields);
    for (uint32_t f = 0; f < h->n_fields; ++f) {
        if (!field_slot_ids[f].empty()) {
            pm_certvec::build_field_lut(*h->fields[f], slots, field_slot_ids[f], h->amx_enabled, fluts[f]);
        }
    }
    st.t_lut_ms = (pm_certvec::monotonic_ns() - t_lut0) * 1e-6;

    // ========================================================================
    // STAGE A: Rotation-Independent Adaptive Dual-Path Filtering (C_1 \perp P)
    // ========================================================================
    double t_t0 = pm_certvec::monotonic_ns();
    uint32_t nprobe = (qspec->nprobe > 0) ? qspec->nprobe : 16u;
    uint32_t c1_cap = (qspec->c1_pool > 0) ? qspec->c1_pool : 600u;
    if (qspec->mode == PM_MODE_TOPK && c1_cap < qspec->k) c1_cap = qspec->k;

    uint64_t tau1 = h->compute_dynamic_tau1(nprobe, static_cast<uint32_t>(slots.size()));
    bool use_path1 = (filt.n_valid <= tau1);
    if (qspec->force_path1) use_path1 = true;
    else if (qspec->force_path3) use_path1 = false;

    std::vector<uint32_t> c1_rids;

    // Helper lambda: run exact Path I scan over all valid filtered entities in B_P!
    auto run_path1 = [&]() {
        c1_rids.clear();
        if (qspec->mode == PM_MODE_THRESHOLD) {
            // Path I Q-Thresh: Deterministic Tier-0 rho_max_f[c] residual-radius pruning!
            // Guarantees global unconditional recall >= 1 - delta and 100% precision!
            auto eval_det = [&](uint32_t rid) {
                if (!local_tombs.empty() && local_tombs.count(rid)) return;
                double ub_t0 = pm_certvec::score_entity_tier0_det_ub(h->fields, fluts, slots,
                                                                     qspec->agg_kind, aux.w0, aux.get(rid), rid);
                if (ub_t0 >= qspec->tau) c1_rids.push_back(rid);
            };
            if (filt.unfiltered) {
                for (uint32_t rid = 0; rid < static_cast<uint32_t>(h->n_docs); ++rid) eval_det(rid);
            } else {
                roaring_uint32_iterator_t it;
                roaring_iterator_init(filt.bm, &it);
                while (it.has_value) { eval_det(it.current_value); roaring_uint32_iterator_advance(&it); }
            }
            return;
        }
        // Path I Q-TopK: exact zero-imputation K0_4b / FP32-LUT scan keeping top-C1
        std::priority_queue<pm_certvec::HeapItem, std::vector<pm_certvec::HeapItem>, pm_certvec::HeapMinCmp> pq;
        auto eval_doc = [&](uint32_t rid) {
            if (!local_tombs.empty() && local_tombs.count(rid)) return;
            double sc = pm_certvec::score_entity_tier0(h->fields, fluts, slots,
                                                       qspec->agg_kind, aux.w0, aux.get(rid), rid);
            pm_certvec::HeapItem item{sc, rid};
            if (pq.size() < c1_cap) pq.push(item);
            else if (pm_certvec::is_better_hit(item, pq.top())) { pq.pop(); pq.push(item); }
        };
        if (filt.unfiltered) {
            for (uint32_t rid = 0; rid < static_cast<uint32_t>(h->n_docs); ++rid) eval_doc(rid);
        } else {
            roaring_uint32_iterator_t it;
            roaring_iterator_init(filt.bm, &it);
            while (it.has_value) { eval_doc(it.current_value); roaring_uint32_iterator_advance(&it); }
        }
        c1_rids.resize(pq.size());
        for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) { c1_rids[i] = pq.top().rid; pq.pop(); }
    };

    if (use_path1) {
        st.path_used = 1;
        run_path1();
    } else {
        st.path_used = 3;
        // Path III: Select top-nprobe centroids per active query vector across all fields
        struct ProbedCent {
            uint32_t field_id;
            uint32_t slot_id;
            uint32_t cid;
            float    score;
            float    gain; // score - min_probed_score for this slot
        };
        std::vector<ProbedCent> probed;
        probed.reserve(slots.size() * nprobe);

        // Deduplicated vector postings per field for Theorem 4B multi-field KL circuit breaker
        std::vector<std::unordered_set<uint32_t>> uniq_cids_per_field(h->n_fields);
        std::vector<std::vector<uint32_t>> uniq_list_per_field(h->n_fields);

        for (uint32_t f = 0; f < h->n_fields; ++f) {
            const auto& flut = fluts[f];
            if (flut.L == 0) continue;
            const auto& fi = *h->fields[f];
            uint32_t np = std::min(nprobe, fi.K);
            uint32_t L_pad = flut.exact_f32_only ? 1u : (((flut.L + 31u) / 32u) * 32u);
            const float* S = flut.S_bf16.data();
            std::vector<std::pair<float, uint32_t>> topc(np);

            for (uint32_t lt = 0; lt < flut.L; ++lt) {
                uint32_t sid = flut.slot_indices[lt];
                // Single-pass top-np min-heap over K centroids
                uint32_t hsz = 0;
                float min_v = -std::numeric_limits<float>::infinity();
                auto cmp = [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
                    return a.first > b.first;
                };
                for (uint32_t c = 0; c < fi.K; ++c) {
                    float v = S[static_cast<size_t>(c) * L_pad + lt];
                    if (hsz < np) {
                        topc[hsz++] = {v, c};
                        if (hsz == np) { std::make_heap(topc.begin(), topc.end(), cmp); min_v = topc[0].first; }
                    } else if (v > min_v) {
                        std::pop_heap(topc.begin(), topc.end(), cmp);
                        topc[np - 1] = {v, c};
                        std::push_heap(topc.begin(), topc.end(), cmp);
                        min_v = topc[0].first;
                    }
                }
                float boundary = (hsz > 0) ? min_v : 0.0f;
                for (uint32_t p = 0; p < hsz; ++p) {
                    uint32_t cid = topc[p].second;
                    probed.push_back({f, sid, cid, topc[p].first, std::max(0.0f, topc[p].first - boundary)});
                    if (uniq_cids_per_field[f].insert(cid).second) uniq_list_per_field[f].push_back(cid);
                }
            }
        }

        // Theorem 4B Multi-Field Zero-Heuristic KL Circuit Breaker (vector postings ONLY, excluding lexical!)
        // AND cheap per-posting filtered cardinality check BEFORE building any giant CRoaring union!
        double sigma = (h->n_docs > 0) ? (static_cast<double>(filt.n_valid) / static_cast<double>(h->n_docs)) : 1.0;
        uint64_t R_tot = 0, L_tot = 0;
        uint64_t W_max_multi = 0;
        uint64_t M_p_total = 0;
        for (uint32_t f = 0; f < h->n_fields; ++f) M_p_total += uniq_list_per_field[f].size();
        const double cb_gamma = 0.08;
        const double cb_thresh = std::log(std::max(1.0, static_cast<double>(M_p_total)) / 1e-6);
        bool cb_tripped = false;

        if (!filt.unfiltered && sigma >= 0.10 && M_p_total > 0) {
            std::vector<uint64_t> m_f(h->n_fields, 0);
            uint64_t max_len = 0;
            for (uint32_t f = 0; f < h->n_fields; ++f) max_len = std::max<uint64_t>(max_len, uniq_list_per_field[f].size());
            for (uint64_t step = 0; step < max_len && !cb_tripped; ++step) {
                for (uint32_t f = 0; f < h->n_fields; ++f) {
                    if (step >= uniq_list_per_field[f].size()) continue;
                    const auto* rbm = h->fields[f]->posting(uniq_list_per_field[f][step]);
                    R_tot += roaring_bitmap_get_cardinality(rbm);
                    L_tot += roaring_bitmap_and_cardinality(rbm, filt.bm);
                    m_f[f]++;
                    W_max_multi = 0;
                    for (uint32_t ff = 0; ff < h->n_fields; ++ff) {
                        W_max_multi += std::min<uint64_t>(m_f[ff], h->fields[ff]->u_max);
                    }
                    if (R_tot > 0 && W_max_multi > 0) {
                        double p_hat = static_cast<double>(L_tot) / static_cast<double>(R_tot);
                        if (p_hat < cb_gamma * sigma) {
                            double n_eff = static_cast<double>(R_tot) / static_cast<double>(W_max_multi);
                            double kl = (p_hat > 0.0 ? p_hat * std::log(p_hat / sigma) : 0.0)
                                      + (p_hat < 1.0 ? (1.0 - p_hat) * std::log((1.0 - p_hat) / (1.0 - sigma)) : 0.0);
                            if (n_eff * kl > cb_thresh) {
                                cb_tripped = true;
                                break;
                            }
                        }
                    }
                }
            }
        } else {
            // Unfiltered or sigma < 0.10: still sum exact posting cardinalities to estimate union size for Stage-0a!
            for (uint32_t f = 0; f < h->n_fields; ++f) {
                for (uint32_t cid : uniq_list_per_field[f]) {
                    const auto* rbm = h->fields[f]->posting(cid);
                    uint64_t rc = roaring_bitmap_get_cardinality(rbm);
                    R_tot += rc;
                    L_tot += filt.unfiltered ? rc : roaring_bitmap_and_cardinality(rbm, filt.bm);
                }
            }
        }

        if (cb_tripped && !qspec->force_path3) {
            st.cb_tripped = 1;
            st.path_used = 31;
            run_path1();
        } else {
            // Estimate filtered union size |U| BEFORE building giant CRoaring union (Theorem 4C Single-Pass Trigger!)
            double rho_surv = (R_tot > 0 && sigma > 0.0) ? ((static_cast<double>(L_tot) / static_cast<double>(R_tot)) / sigma) : 1.0;
            double lam_obs = (filt.n_valid > 0) ? (static_cast<double>(L_tot) / static_cast<double>(filt.n_valid)) : 0.0;
            double est_union = static_cast<double>(filt.n_valid) * (1.0 - std::exp(-lam_obs));

            const uint64_t tau_short = 32768ULL;
            bool trigger_stage0a = (!qspec->disable_stage0a)
                                && (!any_r_gt_1)
                                && (qspec->mode == PM_MODE_TOPK)
                                && (est_union > static_cast<double>(tau_short));

            std::vector<uint32_t> cand_list;
            if (trigger_stage0a) {
                // Theorem 4C: Correlation-Gated Single-Pass 64K-Container Pre-Filter!
                // Signed int32_t acc[65536] (256 KB) + uint16_t q_max[65536] (128 KB) = 384 KB (< 1.25 MiB L2)!
                st.stage0a_used = 1;
                uint64_t base_c0 = std::max<uint64_t>(8ULL * c1_cap, 8192ULL);
                if (const char* env_c0 = std::getenv("PM_S0A_C0")) {
                    uint64_t ov = std::strtoull(env_c0, nullptr, 10);
                    if (ov > 0) base_c0 = ov;
                }
                uint64_t c0_cap = (rho_surv >= 0.5)
                    ? base_c0
                    : std::min<uint64_t>(static_cast<uint64_t>(est_union),
                                         static_cast<uint64_t>(std::ceil(static_cast<double>(base_c0) / std::max(rho_surv, 0.1))));

                // Group probed centroids by slot_id so we compute EXACT per-q_i max (zero multi-centroid double counting!)
                std::vector<std::vector<std::pair<uint32_t, uint16_t>>> slot_postings(slots.size());
                float max_gain_all = 1e-6f;
                for (const auto& pc : probed) max_gain_all = std::max(max_gain_all, pc.gain);
                float gain_scale = 4000.0f / max_gain_all; // Fits uint16_t with huge headroom; scaled by weight fits int32_t!
                for (const auto& pc : probed) {
                    uint16_t qg = static_cast<uint16_t>(std::clamp<int>(static_cast<int>(std::lrintf(pc.gain * gain_scale)) + 1, 1, 60000));
                    slot_postings[pc.slot_id].emplace_back(pc.cid, qg);
                }

                // Single-pass 64K container sweep with persistent streaming iterators + 8 KB L1d block filter bitset!
                // ZERO per-block range_cardinality or roaring_bitmap_contains calls!
                struct SlotPostIter {
                    roaring_uint32_iterator_t it;
                    uint16_t qg;
                };
                std::vector<std::vector<SlotPostIter>> slot_iters(slots.size());
                for (size_t sid = 0; sid < slots.size(); ++sid) {
                    const auto& plist = slot_postings[sid];
                    slot_iters[sid].resize(plist.size());
                    const auto& fi = *h->fields[slots[sid].field_id];
                    for (size_t p = 0; p < plist.size(); ++p) {
                        roaring_iterator_init(fi.posting(plist[p].first), &slot_iters[sid][p].it);
                        slot_iters[sid][p].qg = plist[p].second;
                    }
                }
                roaring_uint32_iterator_t filt_it{};
                if (!filt.unfiltered) roaring_iterator_init(filt.bm, &filt_it);
                alignas(64) uint64_t filt_blk[1024];

                uint32_t n_blocks = static_cast<uint32_t>((h->n_docs + 65535ULL) / 65536ULL);
                std::vector<int32_t>  acc(65536, 0);
                std::vector<uint16_t> q_max(65536, 0);
                std::vector<uint16_t> touched_q; touched_q.reserve(4096);
                std::vector<uint16_t> touched_b; touched_b.reserve(16384);
                std::vector<uint8_t>  in_b(65536, 0);

                std::priority_queue<std::pair<int32_t, uint32_t>,
                                    std::vector<std::pair<int32_t, uint32_t>>,
                                    std::greater<std::pair<int32_t, uint32_t>>> top_c0;

                for (uint32_t blk = 0; blk < n_blocks; ++blk) {
                    uint32_t lo_start = blk << 16;
                    uint32_t hi_end = static_cast<uint32_t>(std::min<uint64_t>(h->n_docs, static_cast<uint64_t>(blk + 1u) << 16));
                    if (!filt.unfiltered) {
                        if (!filt_it.has_value) break;
                        if (filt_it.current_value >= hi_end) continue;
                        std::memset(filt_blk, 0, sizeof(filt_blk));
                        uint32_t fcnt = 0;
                        while (filt_it.has_value && filt_it.current_value < hi_end) {
                            uint16_t lo = static_cast<uint16_t>(filt_it.current_value & 0xFFFFu);
                            filt_blk[lo >> 6] |= (1ULL << (lo & 63u));
                            fcnt++;
                            roaring_uint32_iterator_advance(&filt_it);
                        }
                        if (fcnt == 0) continue;
                    }

                    // For each active query slot i: stream monotonically across [lo_start, hi_end) for exact per-slot max
                    for (size_t sid = 0; sid < slots.size(); ++sid) {
                        auto& siters = slot_iters[sid];
                        if (siters.empty()) continue;
                        int32_t w_scale = std::max<int32_t>(1, static_cast<int32_t>(std::lrint(slots[sid].weight * 16.0)));

                        for (auto& spi : siters) {
                            auto& it = spi.it;
                            if (!it.has_value) continue;
                            if (it.current_value < lo_start) roaring_uint32_iterator_move_equalorlarger(&it, lo_start);
                            uint16_t qg = spi.qg;
                            if (filt.unfiltered) {
                                while (it.has_value && it.current_value < hi_end) {
                                    uint16_t lo = static_cast<uint16_t>(it.current_value & 0xFFFFu);
                                    if (q_max[lo] == 0) touched_q.push_back(lo);
                                    if (qg > q_max[lo]) q_max[lo] = qg;
                                    roaring_uint32_iterator_advance(&it);
                                }
                            } else {
                                while (it.has_value && it.current_value < hi_end) {
                                    uint16_t lo = static_cast<uint16_t>(it.current_value & 0xFFFFu);
                                    if ((filt_blk[lo >> 6] >> (lo & 63u)) & 1ULL) {
                                        if (q_max[lo] == 0) touched_q.push_back(lo);
                                        if (qg > q_max[lo]) q_max[lo] = qg;
                                    }
                                    roaring_uint32_iterator_advance(&it);
                                }
                            }
                        }
                        // Fold per-slot exact max into signed int32_t acc[lo] & reset q_max cleanly
                        for (uint16_t lo : touched_q) {
                            if (!in_b[lo]) { in_b[lo] = 1; touched_b.push_back(lo); }
                            acc[lo] += w_scale * static_cast<int32_t>(q_max[lo]);
                            q_max[lo] = 0;
                        }
                        touched_q.clear();
                    }

                    // In-container signed w0 * s_aux[lo] fusion (captures mid-rank hybrid synergy!)
                    if (aux.kind != PM_AUX_NONE && aux.w0 != 0.0) {
                        double aux_sc = aux.w0 * static_cast<double>(gain_scale) * 16.0;
                        for (uint16_t lo : touched_b) {
                            acc[lo] += static_cast<int32_t>(std::lrint(aux_sc * aux.get(lo_start + lo)));
                        }
                    }

                    // Extract top-C0 survivors from this container and reset sparse touched entries
                    for (uint16_t lo : touched_b) {
                        uint32_t rid = lo_start + lo;
                        int32_t sc = acc[lo];
                        acc[lo] = 0;
                        in_b[lo] = 0;
                        if (!local_tombs.empty() && local_tombs.count(rid)) continue;
                        if (top_c0.size() < c0_cap) top_c0.emplace(sc, rid);
                        else if (sc > top_c0.top().first) { top_c0.pop(); top_c0.emplace(sc, rid); }
                    }
                    touched_b.clear();
                }

                // Also inject top-C1 sparse s_aux entities so purely lexical-dominant hybrid docs never vanish
                std::unordered_set<uint32_t> seen_c0;
                cand_list.reserve(top_c0.size() + c1_cap);
                while (!top_c0.empty()) {
                    uint32_t r = top_c0.top().second; top_c0.pop();
                    if (seen_c0.insert(r).second) cand_list.push_back(r);
                }
                if (aux.kind == PM_AUX_SPARSE_PAIRS && aux.w0 > 0.0) {
                    auto sv = aux.sparse_vec;
                    size_t take = std::min<size_t>(sv.size(), c1_cap);
                    std::partial_sort(sv.begin(), sv.begin() + take, sv.end(),
                                      [](const auto& a, const auto& b) { return a.second > b.second; });
                    for (size_t i = 0; i < take; ++i) {
                        uint32_t r = sv[i].first;
                        if (filt.contains(r) && seen_c0.insert(r).second) cand_list.push_back(r);
                    }
                }
                st.stage0a_shortlist = cand_list.size();
            } else {
                // Direct filtered CRoaring union (small/medium |U| or Q-Thresh or r > 1 — zero Prop. 2 risk!)
                roaring_bitmap_t* u_bm = roaring_bitmap_create();
                for (uint32_t f = 0; f < h->n_fields; ++f) {
                    for (uint32_t cid : uniq_list_per_field[f]) {
                        const auto* rbm = h->fields[f]->posting(cid);
                        if (filt.unfiltered) {
                            roaring_bitmap_or_inplace(u_bm, rbm);
                        } else {
                            roaring_bitmap_t* tmp = roaring_bitmap_and(rbm, filt.bm);
                            roaring_bitmap_or_inplace(u_bm, tmp);
                            roaring_bitmap_free(tmp);
                        }
                    }
                }
                if (aux.kind == PM_AUX_SPARSE_PAIRS && aux.w0 > 0.0) {
                    for (const auto& [r, sc] : aux.sparse_vec) {
                        (void)sc;
                        if (filt.contains(r)) roaring_bitmap_add(u_bm, r);
                    }
                }
                uint64_t u_card = roaring_bitmap_get_cardinality(u_bm);
                cand_list.resize(static_cast<size_t>(u_card));
                roaring_bitmap_to_uint32_array(u_bm, cand_list.data());
                roaring_bitmap_free(u_bm);
            }

            // Stage-0b: Zero-imputation full forward K0_4b / FP32-LUT scoring on cand_list -> C_1
            if (qspec->mode == PM_MODE_THRESHOLD) {
                for (uint32_t rid : cand_list) {
                    if (!local_tombs.empty() && local_tombs.count(rid)) continue;
                    double ub_t0 = pm_certvec::score_entity_tier0_det_ub(h->fields, fluts, slots,
                                                                         qspec->agg_kind, aux.w0, aux.get(rid), rid);
                    if (ub_t0 >= qspec->tau) c1_rids.push_back(rid);
                }
            } else {
                std::priority_queue<pm_certvec::HeapItem, std::vector<pm_certvec::HeapItem>, pm_certvec::HeapMinCmp> pq;
                for (uint32_t rid : cand_list) {
                    if (!local_tombs.empty() && local_tombs.count(rid)) continue;
                    double sc = pm_certvec::score_entity_tier0(h->fields, fluts, slots,
                                                               qspec->agg_kind, aux.w0, aux.get(rid), rid);
                    pm_certvec::HeapItem item{sc, rid};
                    if (pq.size() < c1_cap) pq.push(item);
                    else if (pm_certvec::is_better_hit(item, pq.top())) { pq.pop(); pq.push(item); }
                }
                c1_rids.resize(pq.size());
                for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) { c1_rids[i] = pq.top().rid; pq.pop(); }
            }
        }
    }
    st.t_tier0_ms = (pm_certvec::monotonic_ns() - t_t0) * 1e-6;
    st.c1_evaluated = c1_rids.size();

    // ========================================================================
    // STAGE B: Distribution-Free Certified Out-of-Core Reranking + Unified S3-FIFO Cache
    // ========================================================================
    auto& reader = get_thread_reader(h->io_backend, h->calibrated_qd);
    auto* cache_ptr = h->unified_cache.enabled() ? &h->unified_cache : nullptr;

    // (Live LSM delta buffer already evaluated into exact_heap / thresh_hits under <20us snapshot above!)

    // 2. Helper to batch-fetch Tier-2 exact FP16 across all active fields and update exact_heap / thresh_hits
    std::vector<uint32_t> active_fields;
    for (uint32_t f = 0; f < h->n_fields; ++f) if (fluts[f].L > 0) active_fields.push_back(f);

    bool use_split = (h->active_tier == PM_TIER_SPLIT_B1B4 && !qspec->uncertified_approx);
    for (uint32_t f : active_fields) if (!h->fields[f]->has_b1) use_split = false;

    // Build per-field packed FP64 SIMD arrays & hoisted 32 KB L1d B1 Nibble LUT ONCE per query!
    std::vector<pm_certvec::FieldQueryPack> fpacks(h->n_fields);
    for (uint32_t f : active_fields) {
        fpacks[f].init(*h->fields[f], fluts[f], slots, use_split);
    }

    const size_t wave_sz = 64;
    const size_t n_slots = slots.size();
    std::vector<double> doc_g_flat(std::max<size_t>(wave_sz, qspec->k) * n_slots, 0.0);

    auto fetch_and_verify_tier2 = [&](const std::vector<uint32_t>& batch_rids) {
        if (batch_rids.empty()) return;
        double tt2 = pm_certvec::monotonic_ns();
        int nb = static_cast<int>(batch_rids.size());
        if (static_cast<size_t>(nb) * n_slots > doc_g_flat.size()) doc_g_flat.resize(static_cast<size_t>(nb) * n_slots);
        std::fill(doc_g_flat.begin(), doc_g_flat.begin() + static_cast<size_t>(nb) * n_slots, 0.0);
        std::vector<const uint8_t*> ptrs;
        for (uint32_t f : active_fields) {
            const auto& fi = *h->fields[f];
            // If ALL slots on field f are PM_OP_AVG and avg_fp16 is mapped in RAM, zero NVMe read needed!
            bool all_avg_ram = fi.has_avg_fp16 && (fi.avg_fp16 != nullptr);
            if (all_avg_ram) {
                for (uint32_t sid : fluts[f].slot_indices) if (slots[sid].set_op != PM_OP_AVG) { all_avg_ram = false; break; }
            }
            if (!all_avg_ram) {
                pm_certvec::ThreadPageReader::BatchStats bst{};
                reader.fetch_entities(fi.tier2_fp16, cache_ptr, batch_rids.data(), nb, ptrs, bst);
                st.nvme_bytes_read += bst.nvme_bytes;
                st.cache_hits_t2 += bst.cache_hits;
                st.cache_misses += bst.cache_misses;
            }
            for (int i = 0; i < nb; ++i) {
                uint32_t rid = batch_rids[i];
                uint32_t m = fi.tier2_fp16.meta[rid].n_toks;
                const uint8_t* p = all_avg_ram ? nullptr : ptrs[i];
                pm_certvec::compute_field_tier2_exact(fi, fluts[f], slots, p, m, rid, doc_g_flat.data() + static_cast<size_t>(i) * n_slots);
            }
        }
        for (int i = 0; i < nb; ++i) {
            uint32_t rid = batch_rids[i];
            double ex = pm_certvec::eval_phi_f64(qspec->agg_kind, aux.w0, aux.get(rid), slots, doc_g_flat.data() + static_cast<size_t>(i) * n_slots);
            if (qspec->mode == PM_MODE_THRESHOLD) {
                if (ex >= qspec->tau) thresh_hits.push_back({rid, ex});
            } else {
                pm_certvec::HeapItem item{ex, rid};
                if (exact_heap.size() < qspec->k) exact_heap.push(item);
                else if (pm_certvec::is_better_hit(item, exact_heap.top())) { exact_heap.pop(); exact_heap.push(item); }
            }
        }
        st.tier2_fetched += static_cast<uint64_t>(nb);
        st.t_t2_ms += (pm_certvec::monotonic_ns() - tt2) * 1e-6;
    };

    // 3. Seed exact_heap with first k candidates in Q-TopK (unless uncertified approx ablation)
    size_t start_idx = 0;
    if (qspec->mode == PM_MODE_TOPK && !qspec->uncertified_approx && !c1_rids.empty()) {
        size_t seed_n = std::min<size_t>(c1_rids.size(), qspec->k);
        std::vector<uint32_t> seed_batch(c1_rids.begin(), c1_rids.begin() + seed_n);
        fetch_and_verify_tier2(seed_batch);
        start_idx = seed_n;
    }

    // 4. Interleaved Certified Wave-1 (B1 RAM) -> Wave-2 (B4 RAM/NVMe+S3-FIFO) -> Wave-3 (Tier-2 FP16)
    // Interleaving in waves of 64 immediately tightens theta_k after wave 0 while remaining 100% sound
    // under Theorem 3B (strict delta_1 + delta_2 = delta split, ZERO fixed C_w1 cap!).
    std::vector<uint32_t> w1_surv;
    std::vector<const uint8_t*> b4_ptrs;
    std::vector<uint32_t> t2_batch;
    w1_surv.reserve(wave_sz);
    t2_batch.reserve(wave_sz);
    std::vector<double> b1_g(n_slots, 0.0);

    for (size_t off = start_idx; off < c1_rids.size(); off += wave_sz) {
        size_t wn = std::min(wave_sz, c1_rids.size() - off);
        double gate = (qspec->mode == PM_MODE_THRESHOLD)
            ? qspec->tau
            : ((exact_heap.size() >= qspec->k) ? exact_heap.top().score : -std::numeric_limits<double>::infinity());

        const uint32_t* rptr = nullptr;
        size_t nb = 0;
        if (use_split) {
            double tb1 = pm_certvec::monotonic_ns();
            w1_surv.clear();
            for (size_t i = 0; i < wn; ++i) {
                uint32_t rid = c1_rids[off + i];
                std::fill(b1_g.begin(), b1_g.end(), 0.0);
                for (uint32_t f : active_fields) {
                    const auto& fi = *h->fields[f];
                    const auto& e = fi.tier1_b1.meta[rid];
                    const uint8_t* ptr = fi.tier1_b1.ram_map.ptr + e.offset;
                    pm_certvec::compute_field_tier1_b1_ub(fi, fluts[f], fpacks[f], slots, ptr, e.n_toks, true, b1_g.data());
                }
                double ub1 = pm_certvec::eval_phi_f64(qspec->agg_kind, aux.w0, aux.get(rid), slots, b1_g.data());
                if (ub1 >= gate) w1_surv.push_back(rid);
            }
            st.t_b1_ms += (pm_certvec::monotonic_ns() - tb1) * 1e-6;
            st.wave1_survivors += w1_surv.size();
            if (w1_surv.empty()) continue;
            rptr = w1_surv.data();
            nb = w1_surv.size();
        } else {
            rptr = c1_rids.data() + off;
            nb = wn;
            st.wave1_survivors += nb;
        }

        double tb4 = pm_certvec::monotonic_ns();
        std::fill(doc_g_flat.begin(), doc_g_flat.begin() + nb * n_slots, 0.0);
        for (uint32_t f : active_fields) {
            const auto& fi = *h->fields[f];
            pm_certvec::ThreadPageReader::BatchStats bst{};
            reader.fetch_entities(fi.tier1_b4, cache_ptr, rptr, static_cast<int>(nb), b4_ptrs, bst);
            st.nvme_bytes_read += bst.nvme_bytes;
            st.cache_hits_b4 += bst.cache_hits;
            st.cache_misses += bst.cache_misses;
            for (size_t i = 0; i < nb; ++i) {
                uint32_t rid = rptr[i];
                uint32_t m = fi.tier1_b4.meta[rid].n_toks;
                pm_certvec::compute_field_tier1_b4_ub(fi, fluts[f], fpacks[f], slots, b4_ptrs[i], m,
                                                      use_split, qspec->uncertified_approx != 0,
                                                      doc_g_flat.data() + i * n_slots);
            }
        }
        st.t_b4_ms += (pm_certvec::monotonic_ns() - tb4) * 1e-6;

        t2_batch.clear();
        for (size_t i = 0; i < nb; ++i) {
            uint32_t rid = rptr[i];
            double ub2 = pm_certvec::eval_phi_f64(qspec->agg_kind, aux.w0, aux.get(rid), slots, doc_g_flat.data() + i * n_slots);
            if (qspec->uncertified_approx && qspec->mode == PM_MODE_TOPK) {
                pm_certvec::HeapItem item{ub2, rid};
                if (exact_heap.size() < qspec->k) exact_heap.push(item);
                else if (pm_certvec::is_better_hit(item, exact_heap.top())) { exact_heap.pop(); exact_heap.push(item); }
            } else if (ub2 >= gate) {
                t2_batch.push_back(rid);
            }
        }
        if (!qspec->uncertified_approx) fetch_and_verify_tier2(t2_batch);
    }

    // 5. Emit deterministic final outputs!
    if (qspec->mode == PM_MODE_THRESHOLD) {
        std::sort(thresh_hits.begin(), thresh_hits.end(), [](const pm_hit_t& a, const pm_hit_t& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.rid < b.rid;
        });
        uint32_t emitted = 0;
        if (thresh_cb) {
            for (const auto& hit : thresh_hits) {
                if (thresh_cb(thresh_ctx, hit.rid, hit.score) != 0) break;
                emitted++;
            }
        } else if (topk_out && out_count) {
            uint32_t cap = *out_count;
            for (size_t i = 0; i < thresh_hits.size() && i < cap; ++i) {
                topk_out[i] = thresh_hits[i];
                emitted++;
            }
        }
        if (out_count) *out_count = emitted;
    } else {
        uint32_t cnt = static_cast<uint32_t>(exact_heap.size());
        for (int i = static_cast<int>(cnt) - 1; i >= 0; --i) {
            topk_out[i] = {exact_heap.top().rid, exact_heap.top().score};
            exact_heap.pop();
        }
        *out_count = cnt;
    }

    st.t_total_ms = (pm_certvec::monotonic_ns() - t_start) * 1e-6;
    if (stats_out) *stats_out = st;
    return 0;
}

int pm_insert_entity(pm_handle_t* h, uint32_t ext_id, const pm_entity_vecs_t* vecs) {
    if (!h || !vecs) return -EINVAL;
    std::unique_lock<std::shared_mutex> wlock(h->lsm_mtx);
    pm_certvec::DeltaEntity de;
    de.ext_id = ext_id;
    de.field_vecs_f16.resize(h->n_fields);
    de.field_n_vecs.assign(h->n_fields, 0);
    de.field_avg_f16.resize(h->n_fields);

    for (uint32_t i = 0; i < vecs->n_fields; ++i) {
        const auto& fv = vecs->fields[i];
        if (fv.field_id >= h->n_fields || !fv.vecs_f32) return -EINVAL;
        uint32_t f = fv.field_id;
        uint32_t d = h->fields[f]->d;
        de.field_n_vecs[f] = fv.n_vecs;
        de.field_vecs_f16[f].resize(static_cast<size_t>(fv.n_vecs) * d);
        std::vector<double> mean_v(d, 0.0);
        double max_norm_ins = 0.0;
        for (uint32_t tk = 0; tk < fv.n_vecs; ++tk) {
            double n2 = 0.0;
            for (uint32_t j = 0; j < d; ++j) {
                float val = fv.vecs_f32[static_cast<size_t>(tk) * d + j];
                uint16_t h16 = pm_certvec::f32_to_f16_scalar(val);
                de.field_vecs_f16[f][static_cast<size_t>(tk) * d + j] = h16;
                double f16v = static_cast<double>(pm_certvec::f16_to_f32_scalar(h16));
                n2 += f16v * f16v;
                mean_v[j] += f16v;
            }
            max_norm_ins = std::max(max_norm_ins, std::sqrt(n2));
        }
        if (fv.n_vecs > 0) {
            de.field_avg_f16[f].resize(d);
            double inv_m = 1.0 / static_cast<double>(fv.n_vecs);
            for (uint32_t j = 0; j < d; ++j) de.field_avg_f16[f][j] = pm_certvec::f32_to_f16_scalar(static_cast<float>(mean_v[j] * inv_m));
            // Monotonic atomic_max preserves Lemma 3.1 cap_f soundness across dynamic inserts!
            pm_certvec::atomic_max_double(h->fields[f]->m_norm_max, max_norm_ins);
        }
    }
    h->tombstones.erase(ext_id);
    auto it = h->delta_ext_to_idx.find(ext_id);
    if (it != h->delta_ext_to_idx.end()) {
        h->delta_buf[it->second] = std::move(de);
    } else {
        h->delta_ext_to_idx[ext_id] = h->delta_buf.size();
        h->delta_buf.push_back(std::move(de));
    }
    return 0;
}

int pm_delete_entity(pm_handle_t* h, uint32_t ext_id) {
    if (!h) return -EINVAL;
    std::unique_lock<std::shared_mutex> wlock(h->lsm_mtx);
    h->tombstones.insert(ext_id);
    return 0;
}

int pm_compact_delta(pm_handle_t* h) {
    if (!h) return -EINVAL;
    std::unique_lock<std::shared_mutex> wlock(h->lsm_mtx);
    // Purges tombstoned entries from delta_buf in-place (zero global RID remap)
    std::vector<pm_certvec::DeltaEntity> kept;
    kept.reserve(h->delta_buf.size());
    h->delta_ext_to_idx.clear();
    for (auto& de : h->delta_buf) {
        if (!h->tombstones.count(de.ext_id)) {
            h->delta_ext_to_idx[de.ext_id] = kept.size();
            kept.push_back(std::move(de));
        }
    }
    h->delta_buf = std::move(kept);
    return 0;
}

int pm_get_info(const pm_handle_t* h, pm_index_info_t* info) {
    if (!h || !info) return -EINVAL;
    std::memset(info, 0, sizeof(*info));
    info->n_docs = h->n_docs;
    info->n_fields = h->n_fields;
    uint64_t t0 = 0, b1 = 0, b4 = 0;
    for (uint32_t f = 0; f < h->n_fields && f < 8; ++f) {
        info->dims[f] = h->fields[f]->d;
        info->codebooks[f] = h->fields[f]->K;
        info->m_norm_max[f] = h->fields[f]->m_norm_max.load(std::memory_order_acquire);
        t0 += h->fields[f]->tier0_ram_bytes;
        if (h->fields[f]->has_b1) b1 += h->fields[f]->tier1_b1.file_bytes;
        if (h->fields[f]->tier1_b4.in_ram) b4 += h->fields[f]->tier1_b4.file_bytes;
    }
    info->tier0_ram_bytes = t0;
    info->b1_ram_bytes = b1;
    info->b4_ram_bytes = b4;
    info->cache_ram_bytes = h->unified_cache.total_ram_bytes();
    info->calibrated_tau1 = h->calibrated_tau1_base;
    info->calibrated_alpha_k0_ns = h->calibrated_alpha_k0_ns;
    info->active_tier_mode = static_cast<int>(h->active_tier);
    info->amx_enabled = h->amx_enabled ? 1 : 0;
    info->io_backend_active = static_cast<int>(h->io_backend);
    return 0;
}

void pm_close(pm_handle_t* h) {
    delete h;
}

} // extern "C"
