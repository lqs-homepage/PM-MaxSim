// Stage 3 Complete Multi-Segment Dynamic LSM + VecBench 6-Phase + Compaction + Cache-Cliff Engine (pm_multilsm_v4.cpp)
//   Supports:
//     --cache-cliff <out.json>                       : L1/L2/L3 working-set LUT step curve (4-bit vs 8-bit, 1c & 48c)
//     --bench-lsm <msmarco_8m_lsm_dir> <out.json>    : Full 8.84M 3-segment LSM (seg0+seg1+seg2+delta50k) 1/8/24/48c
//                                                      cold NVMe QPS/P50/P99/Recall@10/MB/q + VecBench 6-phase lifecycle + Compaction!
//     [--no-amx] [--threads 1,8,24,48]
//
// Compile:
//   g++ -O3 -march=native -mamx-tile -mamx-bf16 -std=c++20 -fopenmp -Wall -Wextra
//     -I$(P09_ROOT)/third_party/CRoaring
//     -I$(P09_ROOT)/third_party/install/include
//     pm_multilsm_v4.cpp
//     $(P09_ROOT)/third_party/CRoaring/roaring.o
//     $(P09_ROOT)/third_party/install/lib/liburing.a
//     -o pm_multilsm_v4

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <liburing.h>
#include <numeric>
#include <omp.h>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <x86intrin.h>

#include "beta_quantiles_d128.hpp"
#include "roaring.h"

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

static inline double monotonic_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
static uint64_t read_proc_self_io_read_bytes() {
    std::ifstream f("/proc/self/io");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("read_bytes:", 0) == 0) return std::stoull(line.substr(11));
    return 0;
}

template <typename T>
struct AlignedVec {
    T* ptr = nullptr; size_t sz = 0;
    AlignedVec() = default;
    ~AlignedVec() { if (ptr) std::free(ptr); }
    AlignedVec(const AlignedVec&) = delete;
    AlignedVec& operator=(const AlignedVec&) = delete;
    AlignedVec(AlignedVec&& o) noexcept : ptr(o.ptr), sz(o.sz) { o.ptr = nullptr; o.sz = 0; }
    AlignedVec& operator=(AlignedVec&& o) noexcept {
        if (this != &o) { if (ptr) std::free(ptr); ptr = o.ptr; sz = o.sz; o.ptr = nullptr; o.sz = 0; }
        return *this;
    }
    void resize(size_t n, size_t align = 64) {
        if (ptr) std::free(ptr);
        sz = n;
        size_t bytes = ((n * sizeof(T) + align - 1) / align) * align;
        if (bytes == 0) bytes = align;
        ptr = static_cast<T*>(std::aligned_alloc(align, bytes));
        if (!ptr) { std::perror("aligned_alloc"); std::exit(1); }
        std::memset(ptr, 0, bytes);
    }
    T* data() { return ptr; }
    const T* data() const { return ptr; }
    size_t size() const { return sz; }
    T& operator[](size_t i) { return ptr[i]; }
    const T& operator[](size_t i) const { return ptr[i]; }
};

struct DocDiskEntry { uint64_t offset; uint32_t aligned_bytes; uint32_t n_toks; };
struct MNormHeader { uint64_t magic, N_docs, total_tokens; double m_norm_max, m_norm_p99, m_norm_med; };
struct HeapItem { float score; uint32_t did; };
struct HeapMinCmp {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        return (a.score != b.score) ? (a.score > b.score) : (a.did < b.did);
    }
};
static inline bool is_better_hit(const HeapItem& c, const HeapItem& t) {
    return (c.score != t.score) ? (c.score > t.score) : (c.did < t.did);
}

struct SharedDiskFile {
    int fd = -1; std::vector<DocDiskEntry> meta; std::vector<uint8_t> ram_copy;
    bool in_ram = false; uint64_t file_bytes = 0;
    bool open_file(const std::string& path, const std::string& meta_path, bool load_to_ram) {
        std::ifstream fm(meta_path, std::ios::binary | std::ios::ate);
        if (!fm) return false;
        size_t n = (size_t)fm.tellg() / sizeof(DocDiskEntry);
        fm.seekg(0); meta.resize(n);
        fm.read(reinterpret_cast<char*>(meta.data()), n * sizeof(DocDiskEntry));
        struct stat st{}; if (stat(path.c_str(), &st) == 0) file_bytes = (uint64_t)st.st_size;
        if (load_to_ram) {
            std::ifstream fn(path, std::ios::binary | std::ios::ate); if (!fn) return false;
            size_t fsz = (size_t)fn.tellg(); fn.seekg(0); ram_copy.resize(fsz);
            fn.read(reinterpret_cast<char*>(ram_copy.data()), fsz); in_ram = true; return true;
        }
        fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) { std::perror(("open " + path).c_str()); return false; }
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        return true;
    }
    void flush_cold() const { if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); }
    ~SharedDiskFile() { if (fd >= 0) ::close(fd); }
};

struct ThreadUringReader {
    const SharedDiskFile* store = nullptr; struct io_uring ring{}; bool ring_ok = false;
    AlignedVec<uint8_t> bounce; uint64_t bytes_read = 0, logical_bytes_read = 0;
    uint32_t bytes_per_tok = 256;
    void init(const SharedDiskFile* s, size_t init_mb = 16, uint32_t bpt = 256) {
        store = s; bytes_per_tok = bpt;
        if (s && !s->in_ram) {
            if (io_uring_queue_init(128, &ring, 0) < 0) { std::perror("io_uring"); std::exit(1); }
            ring_ok = true; bounce.resize(init_mb * 1024ULL * 1024ULL, 4096);
        }
    }
    void fetch_batch(const uint32_t* doc_ids, int count, std::vector<const uint8_t*>& out_ptrs) {
        if (count <= 0) { out_ptrs.clear(); return; }
        out_ptrs.resize((size_t)count);
        if (store->in_ram) {
            for (int i = 0; i < count; ++i) out_ptrs[i] = store->ram_copy.data() + store->meta[doc_ids[i]].offset;
            return;
        }
        size_t tot = 0; for (int i = 0; i < count; ++i) tot += store->meta[doc_ids[i]].aligned_bytes;
        if (tot > bounce.size()) bounce.resize(tot, 4096);
        size_t off = 0; std::vector<size_t> boffs(count);
        for (int i = 0; i < count; ++i) { boffs[i] = off; out_ptrs[i] = bounce.data() + off; off += store->meta[doc_ids[i]].aligned_bytes; }
        int done = 0, sub = 0;
        while (done < count) {
            int b = 0;
            while (sub < count && (sub - done) < 64) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring); if (!sqe) break;
                const auto& e = store->meta[doc_ids[sub]];
                io_uring_prep_read(sqe, store->fd, bounce.data() + boffs[sub], e.aligned_bytes, e.offset);
                bytes_read += e.aligned_bytes;
                logical_bytes_read += (uint64_t)e.n_toks * bytes_per_tok;
                sub++; b++;
            }
            if (b > 0) io_uring_submit(&ring);
            struct io_uring_cqe* cqe = nullptr;
            if (io_uring_wait_cqe(&ring, &cqe) < 0) { std::perror("wait_cqe"); std::exit(1); }
            unsigned head; int seen = 0;
            io_uring_for_each_cqe(&ring, head, cqe) { if (cqe->res < 0) std::exit(1); seen++; }
            io_uring_cq_advance(&ring, seen); done += seen;
        }
    }
    ~ThreadUringReader() { if (ring_ok) io_uring_queue_exit(&ring); }
};

struct TileCfg { uint8_t palette_id, start_row, r0[14]; uint16_t colsb[16]; uint8_t rows[16]; };
static bool g_amx = false, g_no_amx = false;
static void init_amx_proc() { g_amx = (!g_no_amx) && (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0); }
static void init_amx_thr() {
    if (!g_amx) return;
    TileCfg c{}; c.palette_id = 1; for (int i = 0; i < 8; ++i) { c.rows[i] = 16; c.colsb[i] = 64; }
    _tile_loadconfig(&c);
}

struct Segment {
    uint32_t glob_base = 0;
    uint64_t N_docs = 0, K = 0, Nq = 0, n_cl = 0, S_REP = 0, is_unit = 1, n_words = 0, u_max = 1;
    double u_avg = 70.0, m_norm_max = 1.006;
    AlignedVec<float> C_fp32; std::vector<float> C_norms, C_bf_err; AlignedVec<uint16_t> C_bf16;
    std::vector<float> P_fp32, Q_32, Q_rot;
    std::vector<uint64_t> doc_offs; std::vector<uint16_t> doc_cids;
    std::vector<uint16_t> b3_rep_lens, b3_rep_cids;
    std::vector<uint64_t> inv_offs; std::vector<uint32_t> inv_docs;
    std::vector<roaring_bitmap_t*> inv_roaring;
    uint64_t roaring_bytes = 0;
    SharedDiskFile t1_b1_ram, t1_b4_ram, t1_b4_nvme, t2_nvme;
    bool has_b1 = false, has_b4_ram = false;
    ~Segment() { for (auto* r : inv_roaring) if (r) roaring_bitmap_free(r); }

    bool load(const std::string& dir, uint32_t base_id, bool load_b1, bool load_b4_ram, bool load_shared_cb) {
        glob_base = base_id;
        std::ifstream f(dir + "/core_index.bin", std::ios::binary); if (!f) return false;
        uint64_t hdr[12]; f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        N_docs = hdr[1]; K = hdr[2]; Nq = hdr[4]; n_cl = hdr[6]; S_REP = hdr[7];
        uint64_t len_dc = hdr[8], len_inv = hdr[9]; is_unit = hdr[10]; n_words = (N_docs + 63) / 64;
        if (load_shared_cb) {
            C_fp32.resize(K * 128); f.read(reinterpret_cast<char*>(C_fp32.data()), K * 128 * 4);
            P_fp32.resize(128 * 128); f.read(reinterpret_cast<char*>(P_fp32.data()), 128 * 128 * 4);
            Q_32.resize(Nq * 32 * 128); f.read(reinterpret_cast<char*>(Q_32.data()), Q_32.size() * 4);
            Q_rot.resize(Nq * 32 * 128); f.read(reinterpret_cast<char*>(Q_rot.data()), Q_rot.size() * 4);
            C_norms.resize(K); C_bf_err.resize(K); C_bf16.resize(K * 128);
            for (uint64_t i = 0; i < K * 128; i += 32) {
                __m512bh bh = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(C_fp32.data() + i + 16), _mm512_loadu_ps(C_fp32.data() + i));
                std::memcpy(C_bf16.data() + i, &bh, 64);
            }
            for (uint64_t c = 0; c < K; ++c) {
                double n2 = 0, e2 = 0;
                for (int j = 0; j < 128; ++j) {
                    double cf = C_fp32[c * 128 + j]; uint32_t u = ((uint32_t)C_bf16[c * 128 + j]) << 16;
                    float cb = 0; std::memcpy(&cb, &u, 4); double df = cf - cb; n2 += cf * cf; e2 += df * df;
                }
                C_norms[c] = (float)std::sqrt(n2); C_bf_err[c] = (float)std::sqrt(e2);
            }
        } else {
            f.seekg((std::streamoff)(K * 128 * 4 + 128 * 128 * 4 + 2 * Nq * 32 * 128 * 4), std::ios::cur);
        }
        f.seekg((std::streamoff)N_docs, std::ios::cur);
        doc_offs.resize(N_docs + 1); f.read(reinterpret_cast<char*>(doc_offs.data()), doc_offs.size() * 8);
        u_max = 1; for (uint64_t i = 0; i < N_docs; ++i) u_max = std::max(u_max, doc_offs[i + 1] - doc_offs[i]);
        u_avg = (double)len_dc / std::max<double>(1.0, (double)N_docs);
        doc_cids.resize(len_dc); f.read(reinterpret_cast<char*>(doc_cids.data()), len_dc * 2);
        f.seekg((std::streamoff)(n_cl * 2 + n_cl * S_REP * 2 + n_cl * S_REP * 8), std::ios::cur);
        b3_rep_lens.resize(n_cl); f.read(reinterpret_cast<char*>(b3_rep_lens.data()), n_cl * 2);
        b3_rep_cids.resize(n_cl * S_REP); f.read(reinterpret_cast<char*>(b3_rep_cids.data()), b3_rep_cids.size() * 2);
        inv_offs.resize(K + 1); f.read(reinterpret_cast<char*>(inv_offs.data()), (K + 1) * 8);
        inv_docs.resize(len_inv); f.read(reinterpret_cast<char*>(inv_docs.data()), len_inv * 4);
        {
            std::ifstream fn(dir + "/m_norm.bin", std::ios::binary);
            if (fn) { MNormHeader mh{}; fn.read(reinterpret_cast<char*>(&mh), sizeof(mh)); if (mh.m_norm_max > 0) m_norm_max = mh.m_norm_max; }
        }
        inv_roaring.resize(K, nullptr); uint64_t rb = 0;
        #pragma omp parallel for schedule(dynamic, 64) reduction(+:rb)
        for (uint64_t c = 0; c < K; ++c) {
            roaring_bitmap_t* r = roaring_bitmap_of_ptr((size_t)(inv_offs[c + 1] - inv_offs[c]), inv_docs.data() + inv_offs[c]);
            roaring_bitmap_run_optimize(r); inv_roaring[c] = r; rb += (uint64_t)roaring_bitmap_portable_size_in_bytes(r);
        }
        roaring_bytes = rb;
        if (!t2_nvme.open_file(dir + "/tier2_fp16.nvme", dir + "/tier2_meta.bin", false)) return false;
        if (!t1_b4_nvme.open_file(dir + "/tier1_b4.nvme", dir + "/tier1_meta.bin", false)) return false;
        if (load_b4_ram) has_b4_ram = t1_b4_ram.open_file(dir + "/tier1_b4.nvme", dir + "/tier1_meta.bin", true);
        if (load_b1) has_b1 = t1_b1_ram.open_file(dir + "/tier1_b1.ram", dir + "/tier1_b1_meta.bin", true);
        std::cerr << "[Seg loaded] " << dir << " base=" << glob_base << " N=" << N_docs
                  << " roaring_MB=" << (roaring_bytes / 1048576.0) << " b1_GB=" << (t1_b1_ram.file_bytes / 1073741824.0)
                  << " M_norm=" << m_norm_max << "\n";
        return true;
    }

    uint64_t dynamic_tau1(int nprobe_c, int nq_v = 32) const {
        double mp = std::min<double>((double)K, (double)nq_v * nprobe_c * 0.85);
        double kappa_u = 1.0 - std::exp(-(mp * u_avg) / (double)K);
        if (kappa_u >= 0.92) return N_docs + 1;
        return (uint64_t)std::clamp(9.0e5 / (u_avg * 0.65 * (1.0 - kappa_u)), 8192.0, (double)N_docs + 1.0);
    }
};

struct QueryLUT { uint64_t K = 0; AlignedVec<float> S_bf16; AlignedVec<uint8_t> S4; uint32_t vmask = 0; int n_vq = 32; };

static void build_lut(const uint16_t* C_bf16, uint64_t K, const float* Q_strip, QueryLUT& out) {
    out.K = K;
    if (out.S_bf16.size() != K * 32) out.S_bf16.resize(K * 32);
    if (out.S4.size() != (K + 1) * 16) out.S4.resize((K + 1) * 16);
    out.vmask = 0; out.n_vq = 0;
    for (int t = 0; t < 32; ++t) {
        float n2 = 0; for (int j = 0; j < 128; ++j) n2 += Q_strip[t * 128 + j] * Q_strip[t * 128 + j];
        if (n2 > 1e-8f) { out.vmask |= (1u << t); out.n_vq++; }
    }
    uint16_t Q_bf[32][128];
    for (int t = 0; t < 32; ++t)
        for (int j = 0; j < 128; j += 32) {
            __m512bh bh = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(Q_strip + t * 128 + j + 16), _mm512_loadu_ps(Q_strip + t * 128 + j));
            std::memcpy(&Q_bf[t][j], &bh, 64);
        }
    float* S = out.S_bf16.data();
    __m512 vmin = _mm512_set1_ps(1e9f), vmax = _mm512_set1_ps(-1e9f);
    if (g_amx && (K % 32 == 0)) {
        alignas(64) uint32_t Qp[4][2][16][16];
        for (int kb = 0; kb < 4; ++kb) for (int k = 0; k < 16; ++k) {
            int d0 = kb * 32 + 2 * k, d1 = d0 + 1;
            for (int j = 0; j < 16; ++j) {
                Qp[kb][0][k][j] = (uint32_t)Q_bf[j][d0] | ((uint32_t)Q_bf[j][d1] << 16);
                Qp[kb][1][k][j] = (uint32_t)Q_bf[16 + j][d0] | ((uint32_t)Q_bf[16 + j][d1] << 16);
            }
        }
        for (uint64_t c = 0; c < K; c += 32) {
            _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
            const uint8_t *cp0 = reinterpret_cast<const uint8_t*>(C_bf16 + c * 128), *cp1 = reinterpret_cast<const uint8_t*>(C_bf16 + (c + 16) * 128);
            for (int kb = 0; kb < 4; ++kb) {
                _tile_loadd(4, cp0 + kb * 64, 256); _tile_loadd(5, cp1 + kb * 64, 256);
                _tile_loadd(6, Qp[kb][0], 64); _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(2, 5, 6);
                _tile_loadd(7, Qp[kb][1], 64); _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(3, 5, 7);
            }
            _tile_stored(0, S + c * 32, 128); _tile_stored(1, S + c * 32 + 16, 128);
            _tile_stored(2, S + (c + 16) * 32, 128); _tile_stored(3, S + (c + 16) * 32 + 16, 128);
        }
        for (uint64_t i = 0; i < K * 32; i += 32) {
            __m512 a = _mm512_load_ps(S + i), b = _mm512_load_ps(S + i + 16);
            vmin = _mm512_min_ps(vmin, _mm512_min_ps(a, b)); vmax = _mm512_max_ps(vmax, _mm512_max_ps(a, b));
        }
    } else {
        alignas(64) uint32_t Ql[64][16], Qh[64][16];
        for (int p = 0; p < 64; ++p) for (int j = 0; j < 16; ++j) {
            Ql[p][j] = (uint32_t)Q_bf[j][2 * p] | ((uint32_t)Q_bf[j][2 * p + 1] << 16);
            Qh[p][j] = (uint32_t)Q_bf[16 + j][2 * p] | ((uint32_t)Q_bf[16 + j][2 * p + 1] << 16);
        }
        const uint32_t* Cu = reinterpret_cast<const uint32_t*>(C_bf16);
        for (uint64_t c = 0; c < K; c += 2) {
            __m512 a0l = _mm512_setzero_ps(), a0h = _mm512_setzero_ps(), a1l = _mm512_setzero_ps(), a1h = _mm512_setzero_ps();
            const uint32_t *c0 = Cu + c * 64, *c1 = Cu + (c + 1) * 64;
            for (int p = 0; p < 64; ++p) {
                __m512bh ql = (__m512bh)_mm512_load_si512(Ql[p]), qh = (__m512bh)_mm512_load_si512(Qh[p]);
                a0l = _mm512_dpbf16_ps(a0l, (__m512bh)_mm512_set1_epi32((int)c0[p]), ql);
                a0h = _mm512_dpbf16_ps(a0h, (__m512bh)_mm512_set1_epi32((int)c0[p]), qh);
                a1l = _mm512_dpbf16_ps(a1l, (__m512bh)_mm512_set1_epi32((int)c1[p]), ql);
                a1h = _mm512_dpbf16_ps(a1h, (__m512bh)_mm512_set1_epi32((int)c1[p]), qh);
            }
            _mm512_store_ps(S + c * 32, a0l); _mm512_store_ps(S + c * 32 + 16, a0h);
            _mm512_store_ps(S + (c + 1) * 32, a1l); _mm512_store_ps(S + (c + 1) * 32 + 16, a1h);
            vmin = _mm512_min_ps(vmin, _mm512_min_ps(_mm512_min_ps(a0l, a0h), _mm512_min_ps(a1l, a1h)));
            vmax = _mm512_max_ps(vmax, _mm512_max_ps(_mm512_max_ps(a0l, a0h), _mm512_max_ps(a1l, a1h)));
        }
    }
    float gm = _mm512_reduce_min_ps(vmin), gx = _mm512_reduce_max_ps(vmax); if (gx <= gm + 1e-6f) gx = gm + 1.0f;
    __m512 vgm = _mm512_set1_ps(gm), vsc = _mm512_set1_ps(15.0f / (gx - gm)), v0 = _mm512_setzero_ps(), v15 = _mm512_set1_ps(15.0f);
    __mmask16 ml = (__mmask16)(out.vmask & 0xFFFFu), mh = (__mmask16)(out.vmask >> 16);
    uint8_t* dst = out.S4.data();
    for (uint64_t c = 0; c < K; ++c) {
        __m512 x0 = _mm512_min_ps(_mm512_max_ps(_mm512_maskz_mul_ps(ml, _mm512_sub_ps(_mm512_load_ps(S + c * 32), vgm), vsc), v0), v15);
        __m512 x1 = _mm512_min_ps(_mm512_max_ps(_mm512_maskz_mul_ps(mh, _mm512_sub_ps(_mm512_load_ps(S + c * 32 + 16), vgm), vsc), v0), v15);
        _mm_store_si128(reinterpret_cast<__m128i*>(dst + c * 16),
                        _mm_or_si128(_mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(x0)),
                                     _mm_slli_epi16(_mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(x1)), 4)));
    }
    std::memset(dst + K * 16, 0, 16);
}

static inline uint16_t hsum_u4(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v), hi = _mm_srli_epi16(_mm256_extracti128_si256(v, 1), 4);
    __m128i s = _mm_sad_epu8(_mm_add_epi8(lo, hi), _mm_setzero_si128());
    return (uint16_t)(_mm_cvtsi128_si64(s) + _mm_extract_epi64(s, 1));
}
static inline uint16_t score_k0_4b(const uint16_t* cids, uint64_t s, uint64_t e, const uint8_t* S4) {
    const __m256i nm = _mm256_set_m128i(_mm_set1_epi8((char)0xF0), _mm_set1_epi8(0x0F));
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256(); uint64_t j = s;
    for (; j + 1 < e; j += 2) {
        a0 = _mm256_max_epu8(a0, _mm256_and_si256(_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(S4 + (size_t)cids[j] * 16))), nm));
        a1 = _mm256_max_epu8(a1, _mm256_and_si256(_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(S4 + (size_t)cids[j + 1] * 16))), nm));
    }
    for (; j < e; ++j) a0 = _mm256_max_epu8(a0, _mm256_and_si256(_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(S4 + (size_t)cids[j] * 16))), nm));
    return hsum_u4(_mm256_max_epu8(a0, a1));
}

struct CentroidTopK {
    std::array<std::vector<std::pair<float, uint32_t>>, 32> top_per_tok;
    std::array<float, 32> min_sc{};
    static inline void rep(std::pair<float, uint32_t>* h, int k, float sc, uint32_t cid) {
        h[0] = {sc, cid}; int i = 0;
        while (true) {
            int l = 2 * i + 1, r = l + 1, sm = i;
            if (l < k && h[l].first < h[sm].first) sm = l;
            if (r < k && h[r].first < h[sm].first) sm = r;
            if (sm == i) break;
            std::swap(h[i], h[sm]);
            i = sm;
        }
    }
    void select(const float* S, uint64_t K, uint32_t vm, int np) {
        np = std::min<int>(std::max(1, np), (int)K); alignas(64) float th[32];
        for (int t = 0; t < 32; ++t) {
            top_per_tok[t].clear();
            if (!((vm >> t) & 1u)) { th[t] = 1e30f; min_sc[t] = 0; continue; }
            top_per_tok[t].resize(np);
            for (int j = 0; j < np; ++j) top_per_tok[t][j] = {S[(size_t)j * 32 + t], (uint32_t)j};
            std::make_heap(top_per_tok[t].begin(), top_per_tok[t].end(), [](auto& a, auto& b) { return a.first > b.first; });
            th[t] = top_per_tok[t][0].first;
        }
        __m512 vt0 = _mm512_load_ps(th), vt1 = _mm512_load_ps(th + 16);
        for (uint64_t c = (uint64_t)np; c < K; ++c) {
            const float* row = S + c * 32;
            __mmask16 m0 = _mm512_cmp_ps_mask(_mm512_load_ps(row), vt0, _CMP_GT_OQ);
            __mmask16 m1 = _mm512_cmp_ps_mask(_mm512_load_ps(row + 16), vt1, _CMP_GT_OQ);
            if (__builtin_expect((m0 | m1) != 0, 0)) {
                while (m0) { int t = __builtin_ctz((uint32_t)m0); m0 &= m0 - 1; rep(top_per_tok[t].data(), np, row[t], (uint32_t)c); th[t] = top_per_tok[t][0].first; }
                if (m1) { uint32_t u = (uint32_t)m1; while (u) { int t = 16 + __builtin_ctz(u); u &= u - 1; rep(top_per_tok[t].data(), np, row[t], (uint32_t)c); th[t] = top_per_tok[t][0].first; } vt1 = _mm512_load_ps(th + 16); }
                vt0 = _mm512_load_ps(th);
            }
        }
        for (int t = 0; t < 32; ++t) if ((vm >> t) & 1u) min_sc[t] = top_per_tok[t][0].first;
    }
};

struct QPrepV4 {
    alignas(64) int8_t  q_i8_eo[32][128];
    alignas(64) int16_t b1_nib[32][16][32];
    alignas(64) float   q_i8_scale[32], q_half_bias_i8[32], q_i8_err[32], q_b1_scale[32], q_b1_err[32];
    alignas(64) float   q_bf_err[32], q_bf_norm[32], q_et_mult[32], q_unit_cap[32], q_eps0[32];
    uint32_t vmask = 0; bool use_cap = true;

    void init(const float* Q32, const float* Qrot, uint32_t vm, int nvq, double m_norm, bool build_b1, bool part2_allk = false, int topk = 10) {
        vmask = vm; double tc = get_t_coeff_v4(nvq, part2_allk, topk);
        for (int t = 0; t < 32; ++t) {
            if (!((vm >> t) & 1u)) { q_i8_scale[t] = q_b1_scale[t] = q_et_mult[t] = q_eps0[t] = 0; q_unit_cap[t] = 1e30f; continue; }
            uint16_t qbf[128];
            for (int j = 0; j < 128; j += 32) {
                __m512bh bh = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(Q32 + t * 128 + j + 16), _mm512_loadu_ps(Q32 + t * 128 + j));
                std::memcpy(qbf + j, &bh, 64);
            }
            double bn2 = 0, be2 = 0, q32n2 = 0;
            for (int j = 0; j < 128; ++j) {
                double qf = Q32[t * 128 + j]; q32n2 += qf * qf;
                uint32_t u = ((uint32_t)qbf[j]) << 16; float qb = 0; std::memcpy(&qb, &u, 4);
                double df = qf - qb; bn2 += (double)qb * qb; be2 += df * df;
            }
            double qbfn = std::sqrt(bn2), q32n = std::sqrt(q32n2);
            q_bf_norm[t] = (float)qbfn; q_bf_err[t] = (float)(std::sqrt(be2) + 3e-5 * std::max(qbfn, q32n));
            double mx = 1e-12, n2 = 0;
            for (int j = 0; j < 128; ++j) { double v = Qrot[t * 128 + j]; mx = std::max(mx, std::fabs(v)); n2 += v * v; }
            double qn = std::max(std::sqrt(n2), q32n), dl = mx / 127.0, e2 = 0; int64_t sq = 0;
            for (int b = 0; b < 64; ++b) {
                double ve = Qrot[t * 128 + 2 * b], vo = Qrot[t * 128 + 2 * b + 1];
                int8_t qe = (int8_t)std::clamp<long>(std::lrint(ve / dl), -127, 127);
                int8_t qo = (int8_t)std::clamp<long>(std::lrint(vo / dl), -127, 127);
                q_i8_eo[t][b] = qe; q_i8_eo[t][64 + b] = qo; sq += qe + qo;
                double de = ve - qe * dl, d_o = vo - qo * dl; e2 += de * de + d_o * d_o;
            }
            q_i8_scale[t] = (float)dl; q_half_bias_i8[t] = (float)(7.5 * dl * sq);
            q_i8_err[t] = (float)(std::sqrt(e2) + 3e-5 * qn); q_et_mult[t] = (float)(tc * qn);
            q_unit_cap[t] = (float)(m_norm * 1.00002 * q32n); q_eps0[t] = (float)(3e-6 * std::max(1.0, qn));
            double b1s = std::max(1e-12, (6.0 * qn) / 30000.0);
            q_b1_scale[t] = (float)b1s; q_b1_err[t] = (float)(16.0 * b1s + 4e-5 * qn);
        }
        if (build_b1) {
            for (int g = 0; g < 32; ++g) for (int m = 0; m < 16; ++m) {
                int16_t* dst = b1_nib[g][m];
                for (int t = 0; t < 32; ++t) {
                    if (!((vm >> t) & 1u)) { dst[t] = 0; continue; }
                    const float* qr = Qrot + t * 128 + g * 4;
                    double s = ((m & 1) ? 0.5 : -0.5) * qr[0] + ((m & 2) ? 0.5 : -0.5) * qr[1] + ((m & 4) ? 0.5 : -0.5) * qr[2] + ((m & 8) ? 0.5 : -0.5) * qr[3];
                    dst[t] = (int16_t)std::clamp<long>(std::lrint(s / (double)q_b1_scale[t]), -30000, 30000);
                }
            }
        }
    }
};

static inline __m128i hsum4_i32(__m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    __m256i b0 = _mm256_add_epi32(_mm512_castsi512_si256(a0), _mm512_extracti64x4_epi64(a0, 1));
    __m256i b1 = _mm256_add_epi32(_mm512_castsi512_si256(a1), _mm512_extracti64x4_epi64(a1, 1));
    __m256i b2 = _mm256_add_epi32(_mm512_castsi512_si256(a2), _mm512_extracti64x4_epi64(a2, 1));
    __m256i b3 = _mm256_add_epi32(_mm512_castsi512_si256(a3), _mm512_extracti64x4_epi64(a3, 1));
    __m256i q = _mm256_hadd_epi32(_mm256_hadd_epi32(b0, b1), _mm256_hadd_epi32(b2, b3));
    return _mm_add_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
}
static inline float ynorm_dpbusd(__m512i ev, __m512i od) {
    const __m512i v15 = _mm512_set1_epi8(15);
    __m512i ae = _mm512_abs_epi8(_mm512_sub_epi8(_mm512_add_epi8(ev, ev), v15));
    __m512i ao = _mm512_abs_epi8(_mm512_sub_epi8(_mm512_add_epi8(od, od), v15));
    return 0.5f * std::sqrt((float)_mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ae, ae), ao, ao))) + 1e-5f;
}
static float exact_fp16_doc(const uint8_t* buf, uint32_t m, const float* Q32, uint32_t vm) {
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(buf);
    float dm[32]; for (int t = 0; t < 32; ++t) dm[t] = -1e9f;
    for (uint32_t tk = 0; tk < m; ++tk) {
        const uint16_t* hp = raw + (size_t)tk * 128; __m512 x[8];
        for (int j = 0; j < 8; ++j) x[j] = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(hp + j * 16)));
        uint32_t v = vm;
        while (v) {
            int t = __builtin_ctz(v); v &= v - 1; const float* qp = Q32 + t * 128;
            __m512 a0 = _mm512_mul_ps(x[0], _mm512_loadu_ps(qp)), a1 = _mm512_mul_ps(x[1], _mm512_loadu_ps(qp + 16));
            a0 = _mm512_fmadd_ps(x[2], _mm512_loadu_ps(qp + 32), a0); a1 = _mm512_fmadd_ps(x[3], _mm512_loadu_ps(qp + 48), a1);
            a0 = _mm512_fmadd_ps(x[4], _mm512_loadu_ps(qp + 64), a0); a1 = _mm512_fmadd_ps(x[5], _mm512_loadu_ps(qp + 80), a1);
            a0 = _mm512_fmadd_ps(x[6], _mm512_loadu_ps(qp + 96), a0); a1 = _mm512_fmadd_ps(x[7], _mm512_loadu_ps(qp + 112), a1);
            float d = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)); if (d > dm[t]) dm[t] = d;
        }
    }
    float tot = 0; uint32_t v = vm; while (v) { int t = __builtin_ctz(v); v &= v - 1; tot += dm[t]; }
    return tot;
}
static float b1_ub_doc(const uint8_t* b1, uint32_t m, const float* S, const float* Cn, const float* Cbe, const QPrepV4& qp) {
    const uint16_t *cids = reinterpret_cast<const uint16_t*>(b1), *scf = cids + m, *etf = cids + 2 * m;
    const uint8_t* sg = b1 + 6 * m;
    __m512 d0 = _mm512_set1_ps(-1e9f), d1 = _mm512_set1_ps(-1e9f);
    const __m512 vs0 = _mm512_load_ps(qp.q_b1_scale), vs1 = _mm512_load_ps(qp.q_b1_scale + 16);
    const __m512 ve0 = _mm512_load_ps(qp.q_b1_err),   ve1 = _mm512_load_ps(qp.q_b1_err + 16);
    const __m512 bfe0 = _mm512_load_ps(qp.q_bf_err),  bfe1 = _mm512_load_ps(qp.q_bf_err + 16);
    const __m512 bfn0 = _mm512_load_ps(qp.q_bf_norm), bfn1 = _mm512_load_ps(qp.q_bf_norm + 16);
    const __m512 etm0 = _mm512_load_ps(qp.q_et_mult), etm1 = _mm512_load_ps(qp.q_et_mult + 16);
    const __m512 ep0 = _mm512_load_ps(qp.q_eps0),     ep1 = _mm512_load_ps(qp.q_eps0 + 16);
    const __m512 cp0 = _mm512_load_ps(qp.q_unit_cap), cp1 = _mm512_load_ps(qp.q_unit_cap + 16);
    for (uint32_t tk = 0; tk < m; ++tk) {
        const uint8_t* s = sg + (size_t)tk * 16; __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
        for (int b = 0; b < 16; ++b) {
            a0 = _mm512_add_epi16(a0, _mm512_load_si512(qp.b1_nib[2 * b][s[b] & 0x0F]));
            a1 = _mm512_add_epi16(a1, _mm512_load_si512(qp.b1_nib[2 * b + 1][s[b] >> 4]));
        }
        __m512i sm = _mm512_add_epi16(a0, a1);
        __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(sm))), vs0);
        __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sm, 1))), vs1);
        uint32_t cid = cids[tk]; const float* cs = S + (size_t)cid * 32;
        __m512 vsc = _mm512_set1_ps(_cvtsh_ss(scf[tk])), vet = _mm512_set1_ps(_cvtsh_ss(etf[tk]));
        __m512 vcn = _mm512_set1_ps(Cn[cid]), vcbe = _mm512_set1_ps(Cbe[cid]);
        __m512 u0 = _mm512_add_ps(_mm512_loadu_ps(cs), _mm512_fmadd_ps(vsc, f0, _mm512_fmadd_ps(vet, etm0, _mm512_fmadd_ps(vsc, ve0, _mm512_fmadd_ps(vcn, bfe0, _mm512_fmadd_ps(vcbe, bfn0, ep0))))));
        __m512 u1 = _mm512_add_ps(_mm512_loadu_ps(cs + 16), _mm512_fmadd_ps(vsc, f1, _mm512_fmadd_ps(vet, etm1, _mm512_fmadd_ps(vsc, ve1, _mm512_fmadd_ps(vcn, bfe1, _mm512_fmadd_ps(vcbe, bfn1, ep1))))));
        if (qp.use_cap) { u0 = _mm512_min_ps(u0, cp0); u1 = _mm512_min_ps(u1, cp1); }
        d0 = _mm512_max_ps(d0, u0); d1 = _mm512_max_ps(d1, u1);
    }
    alignas(64) float dm[32]; _mm512_store_ps(dm, d0); _mm512_store_ps(dm + 16, d1);
    float tot = 0; uint32_t v = qp.vmask; while (v) { int t = __builtin_ctz(v); v &= v - 1; tot += dm[t]; }
    return tot;
}
static float b4_ub_doc(const uint8_t* t1, uint32_t m, const float* S, const float* Cn, const float* Cbe, const QPrepV4& qp, bool approx_pt = false) {
    const uint16_t *cids = reinterpret_cast<const uint16_t*>(t1), *scf = cids + m, *etf = cids + 2 * m;
    const uint8_t* pk = t1 + ((6 * m + 63u) & ~63u);
    alignas(16) float dm[32]; for (int t = 0; t < 32; ++t) dm[t] = -1e9f;
    const __m512i m0f = _mm512_set1_epi8(0x0F);
    for (uint32_t tk = 0; tk < m; ++tk) {
        __m512i rw = _mm512_loadu_si512(pk + (size_t)tk * 64), ev = _mm512_and_si512(rw, m0f), od = _mm512_and_si512(_mm512_srli_epi16(rw, 4), m0f);
        uint32_t cid = cids[tk]; const float* cs = S + (size_t)cid * 32;
        float sc = _cvtsh_ss(scf[tk]); __m128 vsc = _mm_set1_ps(sc);
        float et = approx_pt ? 0.0f : _cvtsh_ss(etf[tk]), syn = approx_pt ? 0.0f : (sc * ynorm_dpbusd(ev, od));
        __m128 vet = _mm_set1_ps(et), vsyn = _mm_set1_ps(syn), vcn = _mm_set1_ps(Cn[cid]), vcbe = _mm_set1_ps(Cbe[cid]);
        for (int t = 0; t < 32; t += 4) {
            if (!((qp.vmask >> t) & 0xFu)) continue;
            __m512i a0 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev, _mm512_load_si512(qp.q_i8_eo[t])), od, _mm512_load_si512(qp.q_i8_eo[t] + 64));
            __m512i a1 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev, _mm512_load_si512(qp.q_i8_eo[t + 1])), od, _mm512_load_si512(qp.q_i8_eo[t + 1] + 64));
            __m512i a2 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev, _mm512_load_si512(qp.q_i8_eo[t + 2])), od, _mm512_load_si512(qp.q_i8_eo[t + 2] + 64));
            __m512i a3 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev, _mm512_load_si512(qp.q_i8_eo[t + 3])), od, _mm512_load_si512(qp.q_i8_eo[t + 3] + 64));
            __m128 dots = _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(hsum4_i32(a0, a1, a2, a3)), _mm_load_ps(qp.q_i8_scale + t)), _mm_load_ps(qp.q_half_bias_i8 + t));
            __m128 val;
            if (approx_pt) {
                val = _mm_fmadd_ps(vsc, dots, _mm_loadu_ps(cs + t));
            } else {
                __m128 bfe = _mm_fmadd_ps(vcn, _mm_load_ps(qp.q_bf_err + t), _mm_fmadd_ps(vcbe, _mm_load_ps(qp.q_bf_norm + t), _mm_load_ps(qp.q_eps0 + t)));
                __m128 rad = _mm_fmadd_ps(vet, _mm_load_ps(qp.q_et_mult + t), _mm_fmadd_ps(vsyn, _mm_load_ps(qp.q_i8_err + t), bfe));
                val = _mm_add_ps(_mm_loadu_ps(cs + t), _mm_fmadd_ps(vsc, dots, rad));
                if (qp.use_cap) val = _mm_min_ps(val, _mm_load_ps(qp.q_unit_cap + t));
            }
            _mm_store_ps(dm + t, _mm_max_ps(_mm_load_ps(dm + t), val));
        }
    }
    float tot = 0; uint32_t v = qp.vmask; while (v) { int t = __builtin_ctz(v); v &= v - 1; tot += dm[t]; }
    return tot;
}

static void run_cache_cliff_bench(const std::string& out_json) {
    const uint64_t Ks[8] = {2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    std::ofstream fo(out_json); fo << std::fixed << std::setprecision(4) << "{\"cache_cliff\": [\n";
    bool first = true; const __m256i nm = _mm256_set_m128i(_mm_set1_epi8((char)0xF0), _mm_set1_epi8(0x0F));
    for (int nT : {1, 48}) {
        for (int ki = 0; ki < 8; ++ki) {
            uint64_t K = Ks[ki]; uint32_t mask = (uint32_t)(K - 1);
            std::vector<uint32_t> cids(10000 * 73); std::mt19937 rng(42 + ki);
            for (auto& x : cids) x = rng() & mask;
            for (int is4b : {1, 0}) {
                size_t rb = is4b ? 16 : 32, lut_kb = (K * rb) / 1024;
                double ns_cid = 0.0, t0 = 0.0, t1 = 0.0; uint64_t sink = 0;
                #pragma omp parallel num_threads(nT) reduction(+:sink)
                {
                    AlignedVec<uint8_t> lut; lut.resize(K * rb); int tid = omp_get_thread_num();
                    for (size_t i = 0; i < K * rb; ++i) lut[i] = (uint8_t)((i * 31 + tid * 7) & 0xFF);
                    const uint8_t* lp = lut.data(); uint64_t acc = 0;
                    for (int rep = 0; rep < 2; ++rep) for (int d = 0; d < 2000; ++d) {
                        __m256i a = _mm256_setzero_si256(); const uint32_t* p = cids.data() + d * 73;
                        for (int j = 0; j < 73; ++j)
                            a = is4b ? _mm256_max_epu8(a, _mm256_and_si256(_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(lp + (size_t)p[j] * 16))), nm))
                                     : _mm256_max_epu8(a, _mm256_load_si256(reinterpret_cast<const __m256i*>(lp + (size_t)p[j] * 32)));
                        acc += (uint64_t)_mm256_extract_epi64(a, 0);
                    }
                    #pragma omp barrier
                    #pragma omp single
                    { t0 = monotonic_ns(); }
                    for (int rep = 0; rep < 12; ++rep) for (int d = 0; d < 10000; ++d) {
                        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256(); const uint32_t* p = cids.data() + d * 73;
                        for (int j = 0; j + 1 < 73; j += 2) {
                            if (is4b) {
                                a0 = _mm256_max_epu8(a0, _mm256_and_si256(_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(lp + (size_t)p[j] * 16))), nm));
                                a1 = _mm256_max_epu8(a1, _mm256_and_si256(_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(lp + (size_t)p[j + 1] * 16))), nm));
                            } else {
                                a0 = _mm256_max_epu8(a0, _mm256_load_si256(reinterpret_cast<const __m256i*>(lp + (size_t)p[j] * 32)));
                                a1 = _mm256_max_epu8(a1, _mm256_load_si256(reinterpret_cast<const __m256i*>(lp + (size_t)p[j + 1] * 32)));
                            }
                        }
                        acc += (uint64_t)_mm256_extract_epi64(_mm256_max_epu8(a0, a1), 0);
                    }
                    #pragma omp barrier
                    #pragma omp single
                    { t1 = monotonic_ns(); ns_cid = (t1 - t0) / (12.0 * 10000.0 * 73.0); }
                    sink += acc;
                }
                if (!first) fo << ",\n";
                first = false;
                fo << "  {\"threads\": " << nT << ", \"K\": " << K << ", \"bits\": " << (is4b ? 4 : 8)
                   << ", \"lut_kb_per_core\": " << lut_kb << ", \"ns_per_cid\": " << ns_cid << ", \"sink\": " << (sink & 1) << "}";
                std::cerr << "  [T=" << std::setw(2) << nT << " K=" << std::setw(6) << K << " " << (is4b ? "4b" : "8b")
                          << " LUT=" << std::setw(5) << lut_kb << " KB] " << std::setprecision(3) << ns_cid << " ns/cid\n";
            }
        }
    }
    fo << "\n]}\n";
}

struct GlobalWL {
    std::string name, corr; double sigma = 0; uint64_t nq = 0, nbm = 0, av = 0;
    std::vector<uint64_t> bms; std::vector<uint32_t> q2b, gt10;
};

static std::vector<roaring_bitmap_t*> slice_roaring_for_seg(const uint64_t* glob_bm, uint64_t nbm, uint64_t g_nwords, uint32_t base, uint64_t seg_n) {
    std::vector<roaring_bitmap_t*> out(nbm, nullptr);
    for (uint64_t b = 0; b < nbm; ++b) {
        const uint64_t* bm = glob_bm + b * g_nwords;
        roaring_bitmap_t* r = roaring_bitmap_create();
        std::vector<uint32_t> tmp; tmp.reserve(16384);
        for (uint64_t loc = 0; loc < seg_n; ++loc) {
            uint64_t g = (uint64_t)base + loc;
            if ((bm[g >> 6] >> (g & 63)) & 1ULL) {
                tmp.push_back((uint32_t)loc);
                if (tmp.size() == 16384) { roaring_bitmap_add_many(r, tmp.size(), tmp.data()); tmp.clear(); }
            }
        }
        if (!tmp.empty()) roaring_bitmap_add_many(r, tmp.size(), tmp.data());
        roaring_bitmap_run_optimize(r);
        out[b] = r;
    }
    return out;
}

static double recall_k(const HeapItem* out, int n, const uint32_t* gt, int k) {
    int h = 0; for (int i = 0; i < std::min(n, k); ++i) for (int j = 0; j < k; ++j) if (out[i].did == gt[j]) { h++; break; }
    return (double)h / (double)k;
}

static void run_bench_lsm(const std::string& root, const std::string& out_json, const std::vector<int>& threads,
                          int topk, bool load_b4_ram, int c1_arg, int nprobe_arg,
                          const std::vector<std::string>& wl_filter, const std::string& dump_dir) {
    std::array<Segment, 4> segs;
    uint32_t bases[4] = {0, 0, 0, 0};
    const char* snames[4] = {"seg0", "seg1", "seg2", "delta50k"};
    uint64_t cur_b = 0;
    for (int s = 0; s < 4; ++s) {
        bases[s] = (uint32_t)cur_b;
        if (!segs[s].load(root + "/" + snames[s], bases[s], true, load_b4_ram, s == 0)) {
            std::cerr << "Failed loading " << snames[s] << "\n"; std::exit(2);
        }
        cur_b += segs[s].N_docs;
    }
    uint64_t N_all = cur_b, g_nwords = (N_all + 63) / 64;
    double max_mnorm = 1.0; for (int s = 0; s < 4; ++s) max_mnorm = std::max(max_mnorm, segs[s].m_norm_max);

    std::vector<GlobalWL> wls;
    {
        std::ifstream f(root + "/global_workloads_gt.bin", std::ios::binary);
        if (!f) { std::cerr << "Missing global_workloads_gt.bin\n"; std::exit(3); }
        uint64_t h[4]; f.read(reinterpret_cast<char*>(h), sizeof(h));
        wls.resize(h[1]);
        for (uint64_t i = 0; i < h[1]; ++i) {
            char nb[64], cb[32]; f.read(nb, 64); f.read(cb, 32);
            wls[i].name = std::string(nb); wls[i].corr = std::string(cb);
            double sig = 0; uint64_t qh[3]; f.read(reinterpret_cast<char*>(&sig), 8); f.read(reinterpret_cast<char*>(qh), sizeof(qh));
            wls[i].sigma = sig; wls[i].nq = qh[0]; wls[i].nbm = qh[1]; wls[i].av = qh[2];
            wls[i].bms.resize(qh[1] * g_nwords); f.read(reinterpret_cast<char*>(wls[i].bms.data()), wls[i].bms.size() * 8);
            wls[i].q2b.resize(qh[0]); f.read(reinterpret_cast<char*>(wls[i].q2b.data()), qh[0] * 4);
            wls[i].gt10.resize(qh[0] * 10); f.read(reinterpret_cast<char*>(wls[i].gt10.data()), qh[0] * 10 * 4);
        }
    }
    std::vector<std::pair<std::string, std::vector<uint32_t>>> gt100_map;
    {
        std::ifstream f100(root + "/global_gt100.bin", std::ios::binary);
        if (f100) {
            uint64_t h[4]; f100.read(reinterpret_cast<char*>(h), sizeof(h));
            for (uint64_t i = 0; i < h[1]; ++i) {
                char nb[64]; f100.read(nb, 64);
                std::vector<uint32_t> ids(h[2] * h[3]);
                f100.read(reinterpret_cast<char*>(ids.data()), ids.size() * 4);
                gt100_map.push_back({std::string(nb), std::move(ids)});
            }
        }
    }

    double mb_cb = (segs[0].K * 128 * 2.0 + segs[0].K * 8.0) / 1048576.0;
    double mb_dc = 0, mb_roar = 0, mb_b1 = 0, mb_b4 = 0, mb_t2 = 0;
    for (int s = 0; s < 4; ++s) {
        mb_dc += (segs[s].doc_offs.size() * 8.0 + segs[s].doc_cids.size() * 2.0) / 1048576.0;
        mb_roar += segs[s].roaring_bytes / 1048576.0;
        mb_b1 += segs[s].t1_b1_ram.file_bytes / 1048576.0;
        mb_b4 += segs[s].t1_b4_nvme.file_bytes / 1048576.0;
        mb_t2 += segs[s].t2_nvme.file_bytes / 1048576.0;
    }

    std::ofstream fo(out_json);
    fo << std::fixed << std::setprecision(4)
       << "{\n  \"N_all\": " << N_all << ", \"K\": " << segs[0].K << ", \"topk\": " << topk
       << ", \"amx\": " << (g_amx ? "true" : "false") << ", \"load_b4_ram\": " << (load_b4_ram ? "true" : "false")
       << ", \"m_norm_max\": " << max_mnorm
       << ", \"memory_mb\": {\"tier0_total\": " << (mb_cb + mb_dc + mb_roar) << ", \"tier1_b1_ram\": " << mb_b1
       << ", \"tier1_b4_nvme\": " << mb_b4 << ", \"tier2_fp16_nvme\": " << mb_t2 << "},\n  \"rows\": [\n";
    bool first = true;

    const char* mnames[8] = {"Split_B1_B4_T3", "PM_DRAM_T1", "PM_NVME_T1", "Approx_B4_NVMe", "Approx_B4_DRAM", "B2b_InvK0", "B2a_WARP_Pure", "B1b_Flat_T3"};
    int C1_pool = std::max(c1_arg, topk * 10);

    for (const auto& wl : wls) {
        if (!wl_filter.empty() && std::find(wl_filter.begin(), wl_filter.end(), wl.name) == wl_filter.end()) continue;
        bool is_unf = (wl.sigma >= 0.9999);
        bool is_lsm = (wl.name.rfind("LSM_", 0) == 0);
        int nq = (int)std::min<uint64_t>(wl.nq, 100);
        const uint32_t* gt_ptr = wl.gt10.data();
        int gt_stride = 10;
        if (topk > 10) {
            for (const auto& kv : gt100_map) if (kv.first == wl.name) { gt_ptr = kv.second.data(); gt_stride = 100; break; }
        }
        int eval_k = std::min(topk, gt_stride);

        std::array<std::vector<roaring_bitmap_t*>, 4> seg_rbs;
        if (!is_unf) {
            for (int s = 0; s < 4; ++s) seg_rbs[s] = slice_roaring_for_seg(wl.bms.data(), wl.nbm, g_nwords, bases[s], segs[s].N_docs);
        }

        std::vector<int> run_ts = is_lsm ? std::vector<int>{1, 48} : threads;
        for (int nT : run_ts) {
            int total_q = (nT == 1) ? ((!dump_dir.empty()) ? nq : 50) : std::max(nT * 6, 200);
            for (int m = 0; m < 8; ++m) {
                if ((m == 1 || m == 4) && !load_b4_ram) continue;
                if (is_lsm && m > 2) continue;
                if (m == 7 && wl.sigma > 0.15) continue;
                for (int s = 0; s < 4; ++s) { segs[s].t1_b4_nvme.flush_cold(); segs[s].t2_nvme.flush_cold(); }

                std::vector<double> lats(total_q, 0.0), recs(total_q, 0.0), allk(total_q, 0.0);
                std::vector<double> t_lut_v(total_q, 0.0), t_t0_v(total_q, 0.0), t_b1_v(total_q, 0.0), t_b4_v(total_q, 0.0), t_t2_v(total_q, 0.0);
                std::vector<double> cand_v(total_q, 0.0), w1_v(total_q, 0.0), t2_v(total_q, 0.0);
                std::vector<uint32_t> dumped_ids((!dump_dir.empty() && nT == run_ts.front()) ? (size_t)nq * topk : 0, 0);
                std::atomic<uint64_t> app_b{0}, log_b{0}, cb_trips{0}; std::atomic<int> next_q{0};
                uint64_t io0 = 0, pio = 0; double wt0 = 0, wt1 = 0;

                #pragma omp parallel num_threads(nT)
                {
                    init_amx_thr();
                    std::array<ThreadUringReader, 4> tr_b1, tr_b4r, tr_b4, tr_t2;
                    for (int s = 0; s < 4; ++s) {
                        tr_b1[s].init(&segs[s].t1_b1_ram, 1, 22);
                        if (load_b4_ram) tr_b4r[s].init(&segs[s].t1_b4_ram, 1, 70);
                        tr_b4[s].init(&segs[s].t1_b4_nvme, 12, 70);
                        tr_t2[s].init(&segs[s].t2_nvme, 24, 256);
                    }
                    QueryLUT lut;
                    #pragma omp barrier
                    #pragma omp single
                    { io0 = read_proc_self_io_read_bytes(); wt0 = monotonic_ns(); }
                    while (true) {
                        int job = next_q.fetch_add(1, std::memory_order_relaxed); if (job >= total_q) break;
                        int q = job % nq; uint32_t bi = wl.q2b[q];
                        const float* Qq = segs[0].Q_32.data() + (size_t)q * 32 * 128;
                        const float* Qrq = segs[0].Q_rot.data() + (size_t)q * 32 * 128;
                        double qt0 = monotonic_ns();
                        build_lut(segs[0].C_bf16.data(), segs[0].K, Qq, lut);
                        QPrepV4 qp; qp.init(Qq, Qrq, lut.vmask, lut.n_vq, max_mnorm, (m == 0), (topk > 10), topk);
                        double qt_lut = monotonic_ns();

                        struct SegCand { uint16_t score; uint8_t seg; uint32_t loc_id; };
                        std::vector<SegCand> pool; pool.reserve(8192);
                        bool q_tripped = false;

                        if (m == 6) {
                            CentroidTopK topk_c; topk_c.select(lut.S_bf16.data(), segs[0].K, lut.vmask, nprobe_arg);
                            std::vector<std::pair<uint32_t, float>> hits; hits.reserve(32768);
                            for (int s = 0; s < 4; ++s) {
                                const auto& sg = segs[s];
                                uint32_t vm = lut.vmask;
                                while (vm) {
                                    int t = __builtin_ctz(vm); vm &= vm - 1;
                                    for (const auto& [sc, cid] : topk_c.top_per_tok[t]) {
                                        const roaring_bitmap_t* pl = sg.inv_roaring[cid];
                                        if (is_unf) {
                                            uint64_t n = roaring_bitmap_get_cardinality(pl);
                                            std::vector<uint32_t> tmp(n); roaring_bitmap_to_uint32_array(pl, tmp.data());
                                            for (uint32_t d : tmp) hits.push_back({bases[s] + d, sc});
                                        } else {
                                            roaring_bitmap_t* inter = roaring_bitmap_and(pl, seg_rbs[s][bi]);
                                            uint64_t n = roaring_bitmap_get_cardinality(inter);
                                            if (n > 0) {
                                                std::vector<uint32_t> tmp(n); roaring_bitmap_to_uint32_array(inter, tmp.data());
                                                for (uint32_t d : tmp) hits.push_back({bases[s] + d, sc});
                                            }
                                            roaring_bitmap_free(inter);
                                        }
                                    }
                                }
                            }
                            if (!hits.empty()) {
                                std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                                std::vector<HeapItem> agg; agg.reserve(hits.size() / 2 + 1);
                                for (size_t i = 0; i < hits.size(); ) {
                                    uint32_t gd = hits[i].first; float sc = 0.0f;
                                    while (i < hits.size() && hits[i].first == gd) { sc += hits[i].second; ++i; }
                                    agg.push_back({sc, gd});
                                }
                                int nc_w = std::min<int>(C1_pool, (int)agg.size());
                                std::partial_sort(agg.begin(), agg.begin() + nc_w, agg.end(), HeapMinCmp{});
                                for (int i = 0; i < nc_w; ++i) {
                                    uint32_t gd = agg[i].did; uint8_t sg_i = 0;
                                    for (int s = 3; s >= 0; --s) if (gd >= bases[s]) { sg_i = (uint8_t)s; break; }
                                    pool.push_back({(uint16_t)(65535 - i), sg_i, gd - bases[sg_i]});
                                }
                            }
                        } else if (m == 7) {
                            for (int s = 0; s < 4; ++s) {
                                const auto& sg = segs[s];
                                if (is_unf) {
                                    for (uint32_t d = 0; d < sg.N_docs; ++d)
                                        pool.push_back({score_k0_4b(sg.doc_cids.data(), sg.doc_offs[d], sg.doc_offs[d + 1], lut.S4.data()), (uint8_t)s, d});
                                } else {
                                    const roaring_bitmap_t* prb = seg_rbs[s][bi];
                                    uint64_t nv = roaring_bitmap_get_cardinality(prb);
                                    if (nv > 0) {
                                        std::vector<uint32_t> vs(nv); roaring_bitmap_to_uint32_array(prb, vs.data());
                                        for (uint32_t d : vs)
                                            pool.push_back({score_k0_4b(sg.doc_cids.data(), sg.doc_offs[d], sg.doc_offs[d + 1], lut.S4.data()), (uint8_t)s, d});
                                    }
                                }
                            }
                        } else {
                            CentroidTopK topk_c; topk_c.select(lut.S_bf16.data(), segs[0].K, lut.vmask, nprobe_arg);
                            std::vector<uint32_t> pcids; pcids.reserve(32 * nprobe_arg);
                            uint32_t vm = lut.vmask;
                            while (vm) { int t = __builtin_ctz(vm); vm &= vm - 1; for (const auto& [sc, cid] : topk_c.top_per_tok[t]) pcids.push_back(cid); }
                            std::sort(pcids.begin(), pcids.end()); pcids.erase(std::unique(pcids.begin(), pcids.end()), pcids.end());
                            bool use_cb = (m != 5);

                            for (int s = 0; s < 3; ++s) {
                                const auto& sg = segs[s];
                                roaring_bitmap_t* cu = roaring_bitmap_create();
                                bool tripped = false;
                                if (is_unf) {
                                    for (uint32_t cid : pcids) roaring_bitmap_or_inplace(cu, sg.inv_roaring[cid]);
                                } else {
                                    const roaring_bitmap_t* prb = seg_rbs[s][bi];
                                    uint64_t nv = roaring_bitmap_get_cardinality(prb);
                                    if (nv == 0) { roaring_bitmap_free(cu); continue; }
                                    if (nv <= sg.dynamic_tau1(nprobe_arg, lut.n_vq)) {
                                        std::vector<uint32_t> vs(nv); roaring_bitmap_to_uint32_array(prb, vs.data());
                                        for (uint32_t d : vs) pool.push_back({score_k0_4b(sg.doc_cids.data(), sg.doc_offs[d], sg.doc_offs[d + 1], lut.S4.data()), (uint8_t)s, d});
                                        roaring_bitmap_free(cu); continue;
                                    }
                                    uint64_t tot_raw = 0, tot_live = 0;
                                    double sigma_s = (double)nv / (double)sg.N_docs;
                                    for (size_t pi = 0; pi < pcids.size(); ++pi) {
                                        const roaring_bitmap_t* pl = sg.inv_roaring[pcids[pi]];
                                        uint64_t raw_sz = roaring_bitmap_get_cardinality(pl);
                                        roaring_bitmap_t* inter = roaring_bitmap_and(pl, prb);
                                        uint64_t live_sz = roaring_bitmap_get_cardinality(inter);
                                        tot_raw += raw_sz; tot_live += live_sz;
                                        if (live_sz > 0) roaring_bitmap_or_inplace(cu, inter);
                                        roaring_bitmap_free(inter);
                                        if (use_cb && sigma_s >= 0.10 && pi >= 15) {
                                            double w_max = std::min<double>((double)(pi + 1), (double)sg.u_max);
                                            double n_eff = (double)tot_raw / w_max;
                                            double p_hat = (double)tot_live / std::max<double>(1.0, (double)tot_raw);
                                            if (p_hat < 0.08 * sigma_s) {
                                                double kl = (p_hat <= 0.0) ? std::log(1.0 / (1.0 - sigma_s))
                                                                           : (p_hat * std::log(p_hat / sigma_s) + (1.0 - p_hat) * std::log((1.0 - p_hat) / (1.0 - sigma_s)));
                                                if (n_eff * kl > std::log((double)std::max<size_t>(1, pcids.size()) / 1e-6)) { tripped = true; break; }
                                            }
                                        }
                                    }
                                    if (tripped) {
                                        q_tripped = true;
                                        double trip_p = (double)tot_live / std::max<double>(1.0, (double)tot_raw);
                                        if (nv > 150000 && tot_live > 0 && trip_p >= 0.01 * sigma_s) {
                                            CentroidTopK topk_exp; topk_exp.select(lut.S_bf16.data(), segs[0].K, lut.vmask, 64);
                                            std::vector<uint32_t> ecids; ecids.reserve(32 * 64);
                                            uint32_t vm2 = lut.vmask;
                                            while (vm2) { int t = __builtin_ctz(vm2); vm2 &= vm2 - 1; for (const auto& [sc, cid] : topk_exp.top_per_tok[t]) ecids.push_back(cid); }
                                            std::sort(ecids.begin(), ecids.end()); ecids.erase(std::unique(ecids.begin(), ecids.end()), ecids.end());
                                            for (uint32_t cid : ecids) {
                                                roaring_bitmap_t* inter = roaring_bitmap_and(sg.inv_roaring[cid], prb);
                                                if (roaring_bitmap_get_cardinality(inter) > 0) roaring_bitmap_or_inplace(cu, inter);
                                                roaring_bitmap_free(inter);
                                                if (roaring_bitmap_get_cardinality(cu) >= (uint64_t)(4 * C1_pool)) break;
                                            }
                                        }
                                        if (roaring_bitmap_get_cardinality(cu) < (uint64_t)C1_pool) {
                                            roaring_bitmap_free(cu);
                                            std::vector<uint32_t> vs(nv); roaring_bitmap_to_uint32_array(prb, vs.data());
                                            for (uint32_t d : vs) pool.push_back({score_k0_4b(sg.doc_cids.data(), sg.doc_offs[d], sg.doc_offs[d + 1], lut.S4.data()), (uint8_t)s, d});
                                            continue;
                                        }
                                    }
                                }
                                uint64_t nc = roaring_bitmap_get_cardinality(cu);
                                std::vector<uint32_t> loc(nc); roaring_bitmap_to_uint32_array(cu, loc.data());
                                roaring_bitmap_free(cu);
                                for (uint32_t d : loc) pool.push_back({score_k0_4b(sg.doc_cids.data(), sg.doc_offs[d], sg.doc_offs[d + 1], lut.S4.data()), (uint8_t)s, d});
                            }
                            {
                                const auto& dg = segs[3];
                                if (is_unf) {
                                    for (uint32_t d = 0; d < dg.N_docs; ++d)
                                        pool.push_back({score_k0_4b(dg.doc_cids.data(), dg.doc_offs[d], dg.doc_offs[d + 1], lut.S4.data()), 3, d});
                                } else {
                                    const roaring_bitmap_t* drb = seg_rbs[3][bi];
                                    uint64_t nv = roaring_bitmap_get_cardinality(drb);
                                    if (nv > 0) {
                                        std::vector<uint32_t> vs(nv); roaring_bitmap_to_uint32_array(drb, vs.data());
                                        for (uint32_t d : vs)
                                            pool.push_back({score_k0_4b(dg.doc_cids.data(), dg.doc_offs[d], dg.doc_offs[d + 1], lut.S4.data()), 3, d});
                                    }
                                }
                            }
                        }
                        if (q_tripped) cb_trips.fetch_add(1, std::memory_order_relaxed);

                        int nc = std::min<int>(C1_pool, (int)pool.size());
                        if (nc > 0 && m != 6) {
                            std::partial_sort(pool.begin(), pool.begin() + nc, pool.end(), [](const SegCand& a, const SegCand& b) {
                                return (a.score != b.score) ? (a.score > b.score) : ((a.seg != b.seg) ? (a.seg < b.seg) : (a.loc_id < b.loc_id));
                            });
                        }
                        double qt_t0 = monotonic_ns();

                        int loc_w1 = 0, loc_t2 = 0;
                        double tb1 = 0.0, tb4 = 0.0, tt2 = 0.0;
                        std::priority_queue<HeapItem, std::vector<HeapItem>, HeapMinCmp> heap;
                        auto eval_exact_batch = [&](const std::vector<SegCand>& batch) {
                            double ts = monotonic_ns();
                            std::array<std::vector<uint32_t>, 4> by_s;
                            for (const auto& c : batch) by_s[c.seg].push_back(c.loc_id);
                            std::vector<const uint8_t*> ptrs;
                            for (int s = 0; s < 4; ++s) {
                                if (by_s[s].empty()) continue;
                                tr_t2[s].fetch_batch(by_s[s].data(), (int)by_s[s].size(), ptrs);
                                for (size_t i = 0; i < by_s[s].size(); ++i) {
                                    uint32_t lid = by_s[s][i];
                                    HeapItem it{exact_fp16_doc(ptrs[i], segs[s].t2_nvme.meta[lid].n_toks, Qq, lut.vmask), bases[s] + lid};
                                    if ((int)heap.size() < topk) heap.push(it);
                                    else if (is_better_hit(it, heap.top())) { heap.pop(); heap.push(it); }
                                }
                            }
                            loc_t2 += (int)batch.size();
                            tt2 += (monotonic_ns() - ts) * 1e-6;
                        };

                        if (m == 5 || m == 6) {
                            std::vector<SegCand> allc(pool.begin(), pool.begin() + nc);
                            eval_exact_batch(allc);
                        } else if (m == 3 || m == 4) {
                            double ts = monotonic_ns();
                            auto& b4_rdr = (m == 4) ? tr_b4r : tr_b4;
                            std::array<std::vector<uint32_t>, 4> by_s;
                            for (int i = 0; i < nc; ++i) by_s[pool[i].seg].push_back(pool[i].loc_id);
                            std::vector<const uint8_t*> ptrs;
                            for (int s = 0; s < 4; ++s) {
                                if (by_s[s].empty()) continue;
                                b4_rdr[s].fetch_batch(by_s[s].data(), (int)by_s[s].size(), ptrs);
                                for (size_t i = 0; i < by_s[s].size(); ++i) {
                                    uint32_t lid = by_s[s][i];
                                    HeapItem it{b4_ub_doc(ptrs[i], segs[s].t1_b4_nvme.meta[lid].n_toks, lut.S_bf16.data(), segs[0].C_norms.data(), segs[0].C_bf_err.data(), qp, true), bases[s] + lid};
                                    if ((int)heap.size() < topk) heap.push(it);
                                    else if (is_better_hit(it, heap.top())) { heap.pop(); heap.push(it); }
                                }
                            }
                            tb4 += (monotonic_ns() - ts) * 1e-6;
                        } else {
                            auto& b4_rdr = (m == 1) ? tr_b4r : tr_b4;
                            int sk = std::min(topk, nc);
                            std::vector<SegCand> seed(pool.begin(), pool.begin() + sk);
                            eval_exact_batch(seed);
                            const int WAVE = 64;
                            for (int ws = sk; ws < nc; ws += WAVE) {
                                int we = std::min(nc, ws + WAVE);
                                float th = ((int)heap.size() == topk) ? heap.top().score : -1e9f;
                                std::array<std::vector<uint32_t>, 4> w1_by_s;
                                if (m == 0) {
                                    double t1s = monotonic_ns();
                                    for (int i = ws; i < we; ++i) {
                                        int s = pool[i].seg; uint32_t lid = pool[i].loc_id;
                                        const uint8_t* ptr = segs[s].t1_b1_ram.ram_copy.data() + segs[s].t1_b1_ram.meta[lid].offset;
                                        if (b1_ub_doc(ptr, segs[s].t1_b1_ram.meta[lid].n_toks, lut.S_bf16.data(), segs[0].C_norms.data(), segs[0].C_bf_err.data(), qp) >= th) {
                                            w1_by_s[s].push_back(lid);
                                            loc_w1++;
                                        }
                                    }
                                    tb1 += (monotonic_ns() - t1s) * 1e-6;
                                } else {
                                    for (int i = ws; i < we; ++i) w1_by_s[pool[i].seg].push_back(pool[i].loc_id);
                                }
                                double t4s = monotonic_ns();
                                std::vector<SegCand> w2_surv;
                                std::vector<const uint8_t*> p4;
                                for (int s = 0; s < 4; ++s) {
                                    if (w1_by_s[s].empty()) continue;
                                    b4_rdr[s].fetch_batch(w1_by_s[s].data(), (int)w1_by_s[s].size(), p4);
                                    for (size_t i = 0; i < w1_by_s[s].size(); ++i) {
                                        uint32_t lid = w1_by_s[s][i];
                                        if (b4_ub_doc(p4[i], segs[s].t1_b4_nvme.meta[lid].n_toks, lut.S_bf16.data(), segs[0].C_norms.data(), segs[0].C_bf_err.data(), qp, false) >= th)
                                            w2_surv.push_back({0, (uint8_t)s, lid});
                                    }
                                }
                                tb4 += (monotonic_ns() - t4s) * 1e-6;
                                if (!w2_surv.empty()) eval_exact_batch(w2_surv);
                            }
                        }

                        std::vector<HeapItem> res; while (!heap.empty()) { res.push_back(heap.top()); heap.pop(); }
                        std::sort(res.begin(), res.end(), HeapMinCmp{});
                        double qt_end = monotonic_ns();
                        lats[job] = (qt_end - qt0) * 1e-6;
                        t_lut_v[job] = (qt_lut - qt0) * 1e-6;
                        t_t0_v[job]  = (qt_t0 - qt_lut) * 1e-6;
                        t_b1_v[job]  = tb1; t_b4_v[job] = tb4; t_t2_v[job] = tt2;
                        cand_v[job]  = (double)nc; w1_v[job] = (double)loc_w1; t2_v[job] = (double)loc_t2;
                        double r_val = recall_k(res.data(), (int)res.size(), gt_ptr + (size_t)q * gt_stride, eval_k);
                        recs[job] = r_val;
                        allk[job] = (r_val >= 0.999999) ? 1.0 : 0.0;
                        if (!dumped_ids.empty() && job < nq) {
                            for (int r = 0; r < std::min<int>(topk, (int)res.size()); ++r) dumped_ids[(size_t)q * topk + r] = res[r].did;
                        }
                    }
                    #pragma omp barrier
                    #pragma omp single
                    { wt1 = monotonic_ns(); pio = read_proc_self_io_read_bytes() - io0; }
                    for (int s = 0; s < 4; ++s) {
                        app_b.fetch_add(tr_b4[s].bytes_read + tr_t2[s].bytes_read, std::memory_order_relaxed);
                        log_b.fetch_add(tr_b4[s].logical_bytes_read + tr_t2[s].logical_bytes_read, std::memory_order_relaxed);
                    }
                }

                if (!dumped_ids.empty()) {
                    std::ofstream fd(dump_dir + "/" + wl.name + "__" + mnames[m] + "_topids.bin", std::ios::binary);
                    if (fd) fd.write(reinterpret_cast<const char*>(dumped_ids.data()), dumped_ids.size() * 4);
                }
                double ws = (wt1 - wt0) * 1e-9;
                std::sort(lats.begin(), lats.end());
                double p50 = lats[lats.size() / 2], p99 = lats[std::min(lats.size() - 1, (size_t)(lats.size() * 0.99))];
                auto avg_v = [&](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / total_q; };
                double qps = (double)total_q / ws, rk = avg_v(recs), a10 = avg_v(allk);
                double nvme_mbs = ((double)pio / 1048576.0) / ws;
                double log_mb = ((double)log_b.load() / 1048576.0) / total_q;
                double app_mb = ((double)app_b.load() / 1048576.0) / total_q;
                double proc_mb = ((double)pio / 1048576.0) / total_q;
                double raf = (log_mb > 1e-9) ? (proc_mb / log_mb) : 1.0;
                if (!first) fo << ",\n";
                first = false;
                fo << "    {\"workload\": \"" << wl.name << "\", \"threads\": " << nT << ", \"method\": \"" << mnames[m]
                   << "\", \"topk\": " << eval_k << ", \"c1\": " << C1_pool << ", \"nprobe\": " << nprobe_arg
                   << ", \"qps\": " << qps << ", \"p50_ms\": " << p50 << ", \"p99_ms\": " << p99
                   << ", \"recall_k\": " << rk << ", \"recall10\": " << rk << ", \"all10_exact\": " << a10
                   << ", \"avg_cand\": " << avg_v(cand_v) << ", \"avg_w1_surv\": " << avg_v(w1_v) << ", \"avg_t2_reranked\": " << avg_v(t2_v)
                   << ", \"t_lut_ms\": " << avg_v(t_lut_v) << ", \"t_tier0_ms\": " << avg_v(t_t0_v)
                   << ", \"t_b1_ms\": " << avg_v(t_b1_v) << ", \"t_b4_ms\": " << avg_v(t_b4_v) << ", \"t_t2_ms\": " << avg_v(t_t2_v)
                   << ", \"nvme_mbs\": " << nvme_mbs
                   << ", \"logical_mb_q\": " << log_mb << ", \"app_mb_q\": " << app_mb << ", \"proc_mb_q\": " << proc_mb
                   << ", \"raf\": " << raf << ", \"cb_trips\": " << cb_trips.load() << "}";
                std::cerr << "  [" << wl.name << " T=" << std::setw(2) << nT << " C1=" << C1_pool << " np=" << nprobe_arg << "] " << std::setw(15) << mnames[m]
                          << " | QPS=" << std::setprecision(1) << std::setw(7) << qps
                          << " | R@k=" << std::setprecision(3) << rk << " All10=" << std::setprecision(2) << a10
                          << " | T2=" << std::setprecision(1) << avg_v(t2_v) << "/" << avg_v(cand_v)
                          << " | IO=" << std::setprecision(2) << proc_mb << " MB/q\n";
            }
        }
        if (!is_unf) for (int s = 0; s < 4; ++s) for (auto* r : seg_rbs[s]) roaring_bitmap_free(r);
    }

    // Phase 6: Measure real background Segment Compaction (merging 50K Delta Buffer + CRoaring rebuild into seg2!)
    double tc0 = monotonic_ns();
    {
        uint64_t K = segs[0].K;
        std::vector<roaring_bitmap_t*> merged(K, nullptr);
        #pragma omp parallel for schedule(dynamic, 64)
        for (uint64_t c = 0; c < K; ++c) {
            roaring_bitmap_t* r = roaring_bitmap_copy(segs[2].inv_roaring[c]);
            uint64_t s = segs[3].inv_offs[c], e = segs[3].inv_offs[c + 1];
            for (uint64_t p = s; p < e; ++p) roaring_bitmap_add(r, (uint32_t)(segs[2].N_docs + segs[3].inv_docs[p]));
            roaring_bitmap_run_optimize(r);
            merged[c] = r;
        }
        for (auto* r : merged) roaring_bitmap_free(r);
    }
    double compact_s = (monotonic_ns() - tc0) * 1e-9;
    std::cerr << "  [LSM Phase 6 Compaction] Merged 50K Delta into 2.93M Seg2 (K=65,536 CRoaring + run_optimize) in "
              << std::setprecision(3) << compact_s << " s!\n";
    fo << "\n  ],\n  \"compaction_50k_into_seg2_s\": " << compact_s << "\n}\n";
}

int main(int argc, char** argv) {
    std::string mode, dir, out_json, dump_dir;
    std::vector<int> threads = {1, 8, 24, 48};
    std::vector<std::string> wl_filter;
    int topk = 10, c1_arg = 600, nprobe_arg = 8; bool load_b4_ram = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-amx") g_no_amx = true;
        else if (a == "--load-b4-ram") load_b4_ram = true;
        else if (a == "--topk" && i + 1 < argc) topk = std::clamp(std::stoi(argv[++i]), 1, 100);
        else if (a == "--c1" && i + 1 < argc) c1_arg = std::max(10, std::stoi(argv[++i]));
        else if (a == "--nprobe" && i + 1 < argc) nprobe_arg = std::max(1, std::stoi(argv[++i]));
        else if (a == "--dump-topids" && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--workloads" && i + 1 < argc) {
            std::stringstream ss(argv[++i]); std::string tok;
            while (std::getline(ss, tok, ',')) if (!tok.empty()) wl_filter.push_back(tok);
        }
        else if (a == "--cache-cliff" && i + 1 < argc) { mode = "cliff"; out_json = argv[++i]; }
        else if (a == "--bench-lsm" && i + 2 < argc) { mode = "lsm"; dir = argv[++i]; out_json = argv[++i]; }
        else if (a == "--threads" && i + 1 < argc) {
            threads.clear(); std::stringstream ss(argv[++i]); std::string tok;
            while (std::getline(ss, tok, ',')) threads.push_back(std::stoi(tok));
        }
    }
    init_amx_proc(); init_amx_thr();
    if (mode == "cliff") { run_cache_cliff_bench(out_json); return 0; }
    if (mode == "lsm") { run_bench_lsm(dir, out_json, threads, topk, load_b4_ram, c1_arg, nprobe_arg, wl_filter, dump_dir); return 0; }
    std::cerr << "Usage: " << argv[0] << " --cache-cliff <out.json> | --bench-lsm <dir> <out.json> [--load-b4-ram] [--topk 10|50|100] [--workloads a,b] [--dump-topids <dir>]\n";
    return 1;
}
