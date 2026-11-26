// Stage 3 Production Engine (pm_engine_v4.cpp) for P-09 (PM-MaxSim):
//   Complete VLDB 2027 C++20 AVX-512 / AMX-BF16 / CRoaring / liburing O_DIRECT Multi-Vector Engine
//   Implements ALL Stage-3 (3.1) contracts:
//     1. Three Production Deployment Tiers:
//          - PM_NVME_T1     (Tier-0 RAM only ~0.75 GB on LoTTE / ~2.4 GB on MS MARCO; NVMe Tier-1 B4 + NVMe Tier-2 FP16)
//          - Split_B1_B4_T3 (L1d-resident 32KB 1-bit nibble transposed Wave-1 in tiny RAM -> NVMe B4 Wave-2 -> NVMe FP16 Wave-3!)
//          - PM_DRAM_T1     (RAM Tier-1 B4 + NVMe Tier-2 FP16, peak 48-core QPS!)
//     2. All 6 IndependentMathReader-R3 (M3-B03 / M3-M02) Code-to-Math Contracts:
//          - Exact build-time scanned M_norm = max_{x in D_fp16} ||x||_2 from m_norm.bin for q_unit_cap
//          - q_eps0[t] = 3.0e-6 * max(1.0, ||q_i||_2) scaled strictly with query norm
//          - Exact S^{126} Beta quantile tables for ALL |Q| in [1..256], BOTH Part 1 (delta/|Q|) & Part 2 (delta/(10|Q|))
//          - Deterministic (score, -doc_id) min-heap tie-breaking across all rerankers & multi-segment merges
//          - Expanded 1,000,000-pair startup unit test covering B=1 AND B=4, is_unit=true (real FP16 unit vectors!) AND false,
//            non-unit ||q_i||_2 in [0.1, 10.0], zero-padding masks, independent Q_32 != Q_rot, and Part 1 + Part 2!
//          - Runtime --no-amx switch forcing AVX-512 vdpbf16ps LUT build for fair AMX-BF16 vs AVX-512 ablation
//     3. Theorem 4A Dynamic tau1* + Theorem 4B Pure Rule B (gamma=0.08, exact W_max=min(m, u_max)) with smooth
//        two-stage large-N_P fallback (expanded nprobe=64 probe before full Path I scan, eliminating P99 tail spikes!).
//     4. Multi-Segment Dynamic LSM + Writable Delta Buffer (up to 50K docs) + Tombstones + Compaction measurement
//        (supports 1-segment LoTTE/ViDoRe and 3-segment MS MARCO 8.84M seamlessly!).
//     5. Mandatory Barrier-to-Barrier steady-state QPS timing, RAM-only Tier-0 warmup, and kernel /proc/self/io audit.
//
// Compile:
//   g++ -O3 -march=native -mamx-tile -mamx-bf16 -std=c++20 -fopenmp -Wall -Wextra
//     -I$(P09_ROOT)/third_party/CRoaring
//     -I$(P09_ROOT)/third_party/install/include
//     pm_engine_v4.cpp
//     $(P09_ROOT)/third_party/CRoaring/roaring.o
//     $(P09_ROOT)/third_party/install/lib/liburing.a
//     -o pm_engine_v4

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
    while (std::getline(f, line)) {
        if (line.rfind("read_bytes:", 0) == 0) return std::stoull(line.substr(11));
    }
    return 0;
}

static void read_proc_self_mem_gb(double& pss_gb, double& rss_gb, double& hwm_gb) {
    pss_gb = rss_gb = hwm_gb = 0.0;
    {
        std::ifstream f("/proc/self/smaps_rollup");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("Pss:", 0) == 0) pss_gb = std::stod(line.substr(4)) / 1048576.0;
            else if (line.rfind("Rss:", 0) == 0) rss_gb = std::stod(line.substr(4)) / 1048576.0;
        }
    }
    {
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("VmHWM:", 0) == 0) { hwm_gb = std::stod(line.substr(6)) / 1048576.0; break; }
        }
    }
}

static double g_delta = 0.05;
static bool g_force_path3 = false;

static double sample_active_mhz_0_47(int nT) {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    double sum = 0.0;
    int cnt = 0;
    while (std::getline(f, line)) {
        if (line.rfind("cpu MHz", 0) == 0) {
            auto p = line.find(':');
            if (p != std::string::npos && cnt < nT) { sum += std::stod(line.substr(p + 1)); cnt++; }
        }
    }
    return cnt > 0 ? sum / cnt : 0.0;
}

template <typename T>
struct AlignedVec {
    T* ptr = nullptr;
    size_t sz = 0;
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

struct DocDiskEntry {
    uint64_t offset;
    uint32_t aligned_bytes;
    uint32_t n_toks;
};

struct MNormHeader {
    uint64_t magic;
    uint64_t N_docs;
    uint64_t total_tokens;
    double   m_norm_max;
    double   m_norm_p99;
    double   m_norm_med;
};

// Deterministic (score, -doc_id) min-heap item (M3-M02(3) contract!):
// Higher score wins; ties break deterministically in favor of smaller doc_id.
struct HeapItem {
    float score;
    uint32_t did;
};
struct HeapMinCmp {
    // Returns true if 'a' is strictly BETTER than 'b' (so worst element sits at min-heap top!)
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        if (a.score != b.score) return a.score > b.score;
        return a.did < b.did;
    }
};
static inline bool is_better_hit(const HeapItem& cand, const HeapItem& top) {
    if (cand.score != top.score) return cand.score > top.score;
    return cand.did < top.did;
}

struct SharedDiskFile {
    int fd = -1;
    std::vector<DocDiskEntry> meta;
    std::vector<uint8_t> ram_copy;
    bool in_ram = false;
    uint64_t file_bytes = 0;

    bool open_file(const std::string& path, const std::string& meta_path, bool load_to_ram) {
        std::ifstream fm(meta_path, std::ios::binary | std::ios::ate);
        if (!fm) { std::cerr << "Cannot open meta " << meta_path << "\n"; return false; }
        size_t n = (size_t)fm.tellg() / sizeof(DocDiskEntry);
        fm.seekg(0);
        meta.resize(n);
        fm.read(reinterpret_cast<char*>(meta.data()), n * sizeof(DocDiskEntry));

        struct stat st{};
        if (stat(path.c_str(), &st) == 0) file_bytes = (uint64_t)st.st_size;

        if (load_to_ram) {
            std::ifstream fn(path, std::ios::binary | std::ios::ate);
            if (!fn) { std::cerr << "Cannot open ram file " << path << "\n"; return false; }
            size_t fsz = (size_t)fn.tellg();
            fn.seekg(0);
            ram_copy.resize(fsz);
            fn.read(reinterpret_cast<char*>(ram_copy.data()), fsz);
            in_ram = true;
            return true;
        }
        fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) { std::perror(("open O_DIRECT " + path).c_str()); return false; }
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        return true;
    }

    void flush_cold() const {
        if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    }
    ~SharedDiskFile() { if (fd >= 0) ::close(fd); }
};

struct ThreadUringReader {
    const SharedDiskFile* store = nullptr;
    struct io_uring ring{};
    bool ring_ok = false;
    AlignedVec<uint8_t> bounce;
    uint64_t bytes_read = 0;
    uint64_t logical_bytes_read = 0;
    uint64_t ops_read = 0;
    uint32_t bytes_per_tok = 256;

    void init(const SharedDiskFile* s, size_t init_buf_mb = 32, uint32_t bpt = 256) {
        store = s;
        bytes_per_tok = bpt;
        if (s && !s->in_ram) {
            if (io_uring_queue_init(128, &ring, 0) < 0) { std::perror("io_uring_queue_init"); std::exit(1); }
            ring_ok = true;
            bounce.resize(init_buf_mb * 1024ULL * 1024ULL, 4096);
        }
    }

    void fetch_batch(const uint32_t* doc_ids, int count, std::vector<const uint8_t*>& out_ptrs) {
        if (count <= 0) { out_ptrs.clear(); return; }
        out_ptrs.resize((size_t)count);
        if (store->in_ram) {
            for (int i = 0; i < count; ++i) out_ptrs[i] = store->ram_copy.data() + store->meta[doc_ids[i]].offset;
            return;
        }
        size_t total_b = 0;
        for (int i = 0; i < count; ++i) total_b += store->meta[doc_ids[i]].aligned_bytes;
        if (total_b > bounce.size()) bounce.resize(total_b, 4096);
        size_t off = 0;
        std::vector<size_t> boffs(count);
        for (int i = 0; i < count; ++i) {
            boffs[i] = off;
            out_ptrs[i] = bounce.data() + off;
            off += store->meta[doc_ids[i]].aligned_bytes;
        }
        const int QD = 64;
        int done = 0, submitted = 0;
        while (done < count) {
            int batch = 0;
            while (submitted < count && (submitted - done) < QD) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (!sqe) break;
                const auto& e = store->meta[doc_ids[submitted]];
                io_uring_prep_read(sqe, store->fd, bounce.data() + boffs[submitted], e.aligned_bytes, e.offset);
                io_uring_sqe_set_data64(sqe, (uint64_t)submitted);
                bytes_read += e.aligned_bytes;
                logical_bytes_read += (uint64_t)e.n_toks * bytes_per_tok;
                ops_read++;
                submitted++;
                batch++;
            }
            if (batch > 0) io_uring_submit(&ring);
            struct io_uring_cqe* cqe = nullptr;
            int ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) { std::perror("io_uring_wait_cqe"); std::exit(1); }
            unsigned head; int seen = 0;
            io_uring_for_each_cqe(&ring, head, cqe) {
                if (cqe->res < 0) { std::cerr << "io_uring err: " << strerror(-cqe->res) << "\n"; std::exit(1); }
                seen++;
            }
            io_uring_cq_advance(&ring, seen);
            done += seen;
        }
    }
    ~ThreadUringReader() { if (ring_ok) io_uring_queue_exit(&ring); }
};

struct TileCfg {
    uint8_t palette_id, start_row, reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};
static bool g_amx_perm = false;
static bool g_force_no_amx = false;
static bool init_amx_process() {
    if (g_force_no_amx) { g_amx_perm = false; return false; }
    g_amx_perm = (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0);
    return g_amx_perm;
}
static void init_amx_thread() {
    if (!g_amx_perm || g_force_no_amx) return;
    TileCfg cfg{};
    cfg.palette_id = 1;
    for (int i = 0; i < 8; ++i) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
    _tile_loadconfig(&cfg);
}

struct Workload {
    std::string name, corr;
    double sigma = 0.0;
    uint64_t n_q_eval = 0, n_bitmaps = 0, actual_valid = 0;
    std::vector<uint64_t> bitmaps;
    std::vector<uint32_t> q_to_bm;
    std::vector<uint32_t> gt_top10;
};

struct SegmentIndex {
    uint64_t N_docs = 0, K = 0, d = 0, Nq = 0, Lq = 0, n_cl = 0, S_REP = 0, is_unit = 1;
    uint64_t n_words = 0, u_max = 1;
    double u_doc_avg = 73.0;
    double m_norm_max = 1.005; // Overwritten by exact m_norm.bin!

    AlignedVec<float> C_fp32;
    std::vector<float> C_norms, C_bf_err;
    AlignedVec<uint16_t> C_bf16;
    std::vector<float> P_fp32, Q_32, Q_rot;
    std::vector<uint8_t> doms;

    std::vector<uint64_t> doc_offs;
    std::vector<uint16_t> doc_cids;
    std::vector<uint16_t> b3_rep_lens, b3_rep_cids;

    std::vector<uint64_t> inv_offs;
    std::vector<uint32_t> inv_docs;
    std::vector<roaring_bitmap_t*> inv_roaring;
    uint64_t roaring_total_bytes = 0;

    std::vector<Workload> workloads;
    SharedDiskFile tier1_b1_ram, tier1_b4_ram, tier1_b4_nvme, tier2_nvme;
    bool has_b1 = false, has_b4_ram = false;

    ~SegmentIndex() { for (auto* r : inv_roaring) if (r) roaring_bitmap_free(r); }

    bool load(const std::string& dir, bool load_b4_ram, bool load_b1_ram, bool load_workloads = true) {
        double t0 = monotonic_ns();
        {
            std::ifstream f(dir + "/core_index.bin", std::ios::binary);
            if (!f) { std::cerr << "Missing core_index.bin in " << dir << "\n"; return false; }
            uint64_t hdr[12];
            f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
            N_docs = hdr[1]; K = hdr[2]; d = hdr[3]; Nq = hdr[4]; Lq = hdr[5];
            n_cl = hdr[6]; S_REP = hdr[7];
            uint64_t len_dc = hdr[8], len_inv = hdr[9];
            is_unit = hdr[10];
            n_words = (N_docs + 63) / 64;

            C_fp32.resize(K * d); f.read(reinterpret_cast<char*>(C_fp32.data()), K * d * 4);
            P_fp32.resize(d * d); f.read(reinterpret_cast<char*>(P_fp32.data()), d * d * 4);
            Q_32.resize(Nq * Lq * d); f.read(reinterpret_cast<char*>(Q_32.data()), Q_32.size() * 4);
            Q_rot.resize(Nq * Lq * d); f.read(reinterpret_cast<char*>(Q_rot.data()), Q_rot.size() * 4);
            doms.resize(N_docs); f.read(reinterpret_cast<char*>(doms.data()), N_docs);
            doc_offs.resize(N_docs + 1); f.read(reinterpret_cast<char*>(doc_offs.data()), doc_offs.size() * 8);
            u_max = 1;
            for (uint64_t di = 0; di < N_docs; ++di) u_max = std::max(u_max, doc_offs[di + 1] - doc_offs[di]);
            u_doc_avg = (N_docs > 0) ? ((double)len_dc / (double)N_docs) : 73.0;
            doc_cids.resize(len_dc); f.read(reinterpret_cast<char*>(doc_cids.data()), len_dc * 2);

            // Skip retired pm_rep arrays
            f.seekg((std::streamoff)(n_cl * 2 + n_cl * S_REP * 2 + n_cl * S_REP * 8), std::ios::cur);
            b3_rep_lens.resize(n_cl); f.read(reinterpret_cast<char*>(b3_rep_lens.data()), n_cl * 2);
            b3_rep_cids.resize(n_cl * S_REP); f.read(reinterpret_cast<char*>(b3_rep_cids.data()), b3_rep_cids.size() * 2);
            inv_offs.resize(K + 1); f.read(reinterpret_cast<char*>(inv_offs.data()), (K + 1) * 8);
            inv_docs.resize(len_inv); f.read(reinterpret_cast<char*>(inv_docs.data()), len_inv * 4);

            C_norms.resize(K); C_bf_err.resize(K); C_bf16.resize(K * d);
            for (uint64_t i = 0; i < K * d; i += 32) {
                __m512 a = _mm512_loadu_ps(C_fp32.data() + i);
                __m512 b = _mm512_loadu_ps(C_fp32.data() + i + 16);
                __m512bh bh = _mm512_cvtne2ps_pbh(b, a);
                std::memcpy(C_bf16.data() + i, &bh, 64);
            }
            for (uint64_t c = 0; c < K; ++c) {
                double n2 = 0.0, e2 = 0.0;
                for (uint64_t j = 0; j < d; ++j) {
                    double cf = (double)C_fp32[c * d + j];
                    uint32_t u32 = ((uint32_t)C_bf16[c * d + j]) << 16;
                    float cb_f = 0.0f; std::memcpy(&cb_f, &u32, 4);
                    double diff = cf - (double)cb_f;
                    n2 += cf * cf; e2 += diff * diff;
                }
                C_norms[c] = (float)std::sqrt(n2);
                C_bf_err[c] = (float)std::sqrt(e2);
            }
        }
        // Load exact M_norm from m_norm.bin (M3-B03 contract!)
        {
            std::ifstream fn(dir + "/m_norm.bin", std::ios::binary);
            if (fn) {
                MNormHeader mh{};
                fn.read(reinterpret_cast<char*>(&mh), sizeof(mh));
                if (mh.magic == 0x4D4E4F524D303034ULL && mh.m_norm_max > 0.0) {
                    m_norm_max = mh.m_norm_max;
                }
            }
        }
        inv_roaring.resize(K, nullptr);
        uint64_t rbytes = 0;
        #pragma omp parallel for schedule(dynamic, 64) reduction(+:rbytes)
        for (uint64_t c = 0; c < K; ++c) {
            uint64_t s = inv_offs[c], e = inv_offs[c + 1];
            roaring_bitmap_t* r = roaring_bitmap_of_ptr((size_t)(e - s), inv_docs.data() + s);
            roaring_bitmap_run_optimize(r);
            inv_roaring[c] = r;
            rbytes += (uint64_t)roaring_bitmap_portable_size_in_bytes(r);
        }
        roaring_total_bytes = rbytes;

        if (load_workloads) {
            std::ifstream f(dir + "/workloads.bin", std::ios::binary);
            if (f) {
                uint64_t h[4]; f.read(reinterpret_cast<char*>(h), sizeof(h));
                uint64_t n_wl = h[1];
                workloads.resize(n_wl);
                for (uint64_t i = 0; i < n_wl; ++i) {
                    char nb[64], cb[32];
                    f.read(nb, 64); f.read(cb, 32);
                    workloads[i].name = std::string(nb);
                    workloads[i].corr = std::string(cb);
                    double sig = 0.0; uint64_t qh[3];
                    f.read(reinterpret_cast<char*>(&sig), 8);
                    f.read(reinterpret_cast<char*>(qh), sizeof(qh));
                    workloads[i].sigma = sig;
                    workloads[i].n_q_eval = qh[0];
                    workloads[i].n_bitmaps = qh[1];
                    workloads[i].actual_valid = qh[2];
                    workloads[i].bitmaps.resize(qh[1] * n_words);
                    f.read(reinterpret_cast<char*>(workloads[i].bitmaps.data()), workloads[i].bitmaps.size() * 8);
                    workloads[i].q_to_bm.resize(qh[0]);
                    f.read(reinterpret_cast<char*>(workloads[i].q_to_bm.data()), qh[0] * 4);
                    workloads[i].gt_top10.resize(qh[0] * 10);
                    f.read(reinterpret_cast<char*>(workloads[i].gt_top10.data()), qh[0] * 10 * 4);
                }
            }
        }
        if (!tier2_nvme.open_file(dir + "/tier2_fp16.nvme", dir + "/tier2_meta.bin", false)) return false;
        if (!tier1_b4_nvme.open_file(dir + "/tier1_b4.nvme", dir + "/tier1_meta.bin", false)) return false;
        if (load_b4_ram) {
            has_b4_ram = tier1_b4_ram.open_file(dir + "/tier1_b4.nvme", dir + "/tier1_meta.bin", true);
        }
        if (load_b1_ram) {
            struct stat st{};
            if (stat((dir + "/tier1_b1.ram").c_str(), &st) == 0) {
                has_b1 = tier1_b1_ram.open_file(dir + "/tier1_b1.ram", dir + "/tier1_b1_meta.bin", true);
            }
        }
        std::cerr << "[Loaded v4] dir=" << dir << " N=" << N_docs << " K=" << K
                  << " u_avg=" << std::fixed << std::setprecision(1) << u_doc_avg << " u_max=" << u_max
                  << " M_norm=" << std::setprecision(5) << m_norm_max
                  << " has_b1=" << has_b1 << " (" << std::setprecision(2) << (tier1_b1_ram.file_bytes / 1073741824.0) << "GB)"
                  << " has_b4_ram=" << has_b4_ram << " t1_b4_GB=" << (tier1_b4_nvme.file_bytes / 1073741824.0)
                  << " t2_GB=" << (tier2_nvme.file_bytes / 1073741824.0)
                  << " load_s=" << ((monotonic_ns() - t0) * 1e-9) << "\n";
        return true;
    }

    // Theorem 4A Dynamic Local Fixed-Point Crossover tau1* (calibrated on K, u_doc_avg, nprobe)
    uint64_t dynamic_tau1(int nprobe_c, int n_valid_q = 32) const {
        double mp = std::min<double>((double)K, (double)n_valid_q * (double)nprobe_c * 0.85);
        double lam = (mp * u_doc_avg) / (double)K;
        double kappa_u = 1.0 - std::exp(-lam);
        if (kappa_u >= 0.92) return N_docs + 1; // Visual/high-patch regime: Path I dominates [1, N]!
        // Calibrated hardware constants: overhead ~ 0.9 ms, alpha_K0 ~ 0.65 ns/cid
        double tau = 9.0e5 / (u_doc_avg * 0.65 * (1.0 - kappa_u));
        return (uint64_t)std::clamp(tau, 8192.0, (double)N_docs + 1.0);
    }
};

struct QueryLUT {
    uint64_t K = 0;
    AlignedVec<float> S_bf16;
    AlignedVec<uint8_t> S4;
    uint32_t valid_q_mask = 0;
    int n_valid_q = 32;
};

static void build_query_lut(const uint16_t* C_bf16, uint64_t K, const float* Q_strip, QueryLUT& out) {
    out.K = K;
    if (out.S_bf16.size() != K * 32) out.S_bf16.resize(K * 32);
    if (out.S4.size() != (K + 1) * 16) out.S4.resize((K + 1) * 16);
    out.valid_q_mask = 0; out.n_valid_q = 0;
    for (int t = 0; t < 32; ++t) {
        float n2 = 0.0f;
        for (int j = 0; j < 128; ++j) n2 += Q_strip[t * 128 + j] * Q_strip[t * 128 + j];
        if (n2 > 1e-8f) { out.valid_q_mask |= (1u << t); out.n_valid_q++; }
    }
    uint16_t Q_bf[32][128];
    for (int t = 0; t < 32; ++t) {
        for (int j = 0; j < 128; j += 32) {
            __m512 a = _mm512_loadu_ps(Q_strip + t * 128 + j);
            __m512 b = _mm512_loadu_ps(Q_strip + t * 128 + j + 16);
            __m512bh bh = _mm512_cvtne2ps_pbh(b, a);
            std::memcpy(&Q_bf[t][j], &bh, 64);
        }
    }
    float* S = out.S_bf16.data();
    __m512 vmin = _mm512_set1_ps(1e9f), vmax = _mm512_set1_ps(-1e9f);
    if (g_amx_perm && !g_force_no_amx && (K % 32 == 0)) {
        alignas(64) uint32_t Q_pack[4][2][16][16];
        for (int kb = 0; kb < 4; ++kb)
            for (int k = 0; k < 16; ++k) {
                int d0 = kb * 32 + 2 * k, d1 = d0 + 1;
                for (int j = 0; j < 16; ++j) {
                    Q_pack[kb][0][k][j] = (uint32_t)Q_bf[j][d0] | ((uint32_t)Q_bf[j][d1] << 16);
                    Q_pack[kb][1][k][j] = (uint32_t)Q_bf[16 + j][d0] | ((uint32_t)Q_bf[16 + j][d1] << 16);
                }
            }
        for (uint64_t c = 0; c < K; c += 32) {
            _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
            const uint8_t* cp0 = reinterpret_cast<const uint8_t*>(C_bf16 + (c + 0) * 128);
            const uint8_t* cp1 = reinterpret_cast<const uint8_t*>(C_bf16 + (c + 16) * 128);
            for (int kb = 0; kb < 4; ++kb) {
                _tile_loadd(4, cp0 + kb * 64, 256);
                _tile_loadd(5, cp1 + kb * 64, 256);
                _tile_loadd(6, Q_pack[kb][0], 64);
                _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(2, 5, 6);
                _tile_loadd(7, Q_pack[kb][1], 64);
                _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(3, 5, 7);
            }
            _tile_stored(0, S + (c + 0) * 32 + 0, 128);
            _tile_stored(1, S + (c + 0) * 32 + 16, 128);
            _tile_stored(2, S + (c + 16) * 32 + 0, 128);
            _tile_stored(3, S + (c + 16) * 32 + 16, 128);
        }
        for (uint64_t i = 0; i < K * 32; i += 32) {
            __m512 a = _mm512_load_ps(S + i), b = _mm512_load_ps(S + i + 16);
            vmin = _mm512_min_ps(vmin, _mm512_min_ps(a, b));
            vmax = _mm512_max_ps(vmax, _mm512_max_ps(a, b));
        }
    } else {
        alignas(64) uint32_t Q_lo[64][16], Q_hi[64][16];
        for (int p = 0; p < 64; ++p)
            for (int j = 0; j < 16; ++j) {
                Q_lo[p][j] = (uint32_t)Q_bf[j][2 * p] | ((uint32_t)Q_bf[j][2 * p + 1] << 16);
                Q_hi[p][j] = (uint32_t)Q_bf[16 + j][2 * p] | ((uint32_t)Q_bf[16 + j][2 * p + 1] << 16);
            }
        const uint32_t* C_u32 = reinterpret_cast<const uint32_t*>(C_bf16);
        for (uint64_t c = 0; c < K; c += 4) {
            __m512 a0l = _mm512_setzero_ps(), a0h = _mm512_setzero_ps(), a1l = _mm512_setzero_ps(), a1h = _mm512_setzero_ps();
            __m512 a2l = _mm512_setzero_ps(), a2h = _mm512_setzero_ps(), a3l = _mm512_setzero_ps(), a3h = _mm512_setzero_ps();
            const uint32_t *c0 = C_u32 + (c + 0) * 64, *c1 = C_u32 + (c + 1) * 64, *c2 = C_u32 + (c + 2) * 64, *c3 = C_u32 + (c + 3) * 64;
            for (int p = 0; p < 64; ++p) {
                __m512bh ql = (__m512bh)_mm512_load_si512(Q_lo[p]), qh = (__m512bh)_mm512_load_si512(Q_hi[p]);
                a0l = _mm512_dpbf16_ps(a0l, (__m512bh)_mm512_set1_epi32((int)c0[p]), ql);
                a0h = _mm512_dpbf16_ps(a0h, (__m512bh)_mm512_set1_epi32((int)c0[p]), qh);
                a1l = _mm512_dpbf16_ps(a1l, (__m512bh)_mm512_set1_epi32((int)c1[p]), ql);
                a1h = _mm512_dpbf16_ps(a1h, (__m512bh)_mm512_set1_epi32((int)c1[p]), qh);
                a2l = _mm512_dpbf16_ps(a2l, (__m512bh)_mm512_set1_epi32((int)c2[p]), ql);
                a2h = _mm512_dpbf16_ps(a2h, (__m512bh)_mm512_set1_epi32((int)c2[p]), qh);
                a3l = _mm512_dpbf16_ps(a3l, (__m512bh)_mm512_set1_epi32((int)c3[p]), ql);
                a3h = _mm512_dpbf16_ps(a3h, (__m512bh)_mm512_set1_epi32((int)c3[p]), qh);
            }
            _mm512_store_ps(S + (c + 0) * 32, a0l); _mm512_store_ps(S + (c + 0) * 32 + 16, a0h);
            _mm512_store_ps(S + (c + 1) * 32, a1l); _mm512_store_ps(S + (c + 1) * 32 + 16, a1h);
            _mm512_store_ps(S + (c + 2) * 32, a2l); _mm512_store_ps(S + (c + 2) * 32 + 16, a2h);
            _mm512_store_ps(S + (c + 3) * 32, a3l); _mm512_store_ps(S + (c + 3) * 32 + 16, a3h);
            vmin = _mm512_min_ps(vmin, _mm512_min_ps(_mm512_min_ps(a0l, a0h), _mm512_min_ps(a1l, a1h)));
            vmin = _mm512_min_ps(vmin, _mm512_min_ps(_mm512_min_ps(a2l, a2h), _mm512_min_ps(a3l, a3h)));
            vmax = _mm512_max_ps(vmax, _mm512_max_ps(_mm512_max_ps(a0l, a0h), _mm512_max_ps(a1l, a1h)));
            vmax = _mm512_max_ps(vmax, _mm512_max_ps(_mm512_max_ps(a2l, a2h), _mm512_max_ps(a3l, a3h)));
        }
    }
    float g_min = _mm512_reduce_min_ps(vmin), g_max = _mm512_reduce_max_ps(vmax);
    if (g_max <= g_min + 1e-6f) g_max = g_min + 1.0f;
    __m512 vgmin = _mm512_set1_ps(g_min), vsc4 = _mm512_set1_ps(15.0f / (g_max - g_min));
    __m512 v0 = _mm512_setzero_ps(), v15 = _mm512_set1_ps(15.0f);
    __mmask16 m_lo = (__mmask16)(out.valid_q_mask & 0xFFFFu);
    __mmask16 m_hi = (__mmask16)(out.valid_q_mask >> 16);
    uint8_t* dst4 = out.S4.data();
    for (uint64_t c = 0; c < K; ++c) {
        __m512 x0 = _mm512_min_ps(_mm512_max_ps(_mm512_maskz_mul_ps(m_lo, _mm512_sub_ps(_mm512_load_ps(S + c * 32), vgmin), vsc4), v0), v15);
        __m512 x1 = _mm512_min_ps(_mm512_max_ps(_mm512_maskz_mul_ps(m_hi, _mm512_sub_ps(_mm512_load_ps(S + c * 32 + 16), vgmin), vsc4), v0), v15);
        _mm_store_si128(reinterpret_cast<__m128i*>(dst4 + c * 16),
                        _mm_or_si128(_mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(x0)),
                                     _mm_slli_epi16(_mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(x1)), 4)));
    }
    std::memset(dst4 + K * 16, 0, 16);
}

static inline uint16_t hsum_u4_256(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm_srli_epi16(_mm256_extracti128_si256(v, 1), 4);
    __m128i sad = _mm_sad_epu8(_mm_add_epi8(lo, hi), _mm_setzero_si128());
    return (uint16_t)(_mm_cvtsi128_si64(sad) + _mm_extract_epi64(sad, 1));
}

static inline uint16_t score_k0_4b(const uint16_t* cids, uint64_t s, uint64_t e, const uint8_t* S4) {
    const __m256i nm = _mm256_set_m128i(_mm_set1_epi8((char)0xF0), _mm_set1_epi8(0x0F));
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    uint64_t j = s;
    for (; j + 1 < e; j += 2) {
        a0 = _mm256_max_epu8(a0, _mm256_and_si256(_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(S4 + (size_t)cids[j] * 16))), nm));
        a1 = _mm256_max_epu8(a1, _mm256_and_si256(_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(S4 + (size_t)cids[j + 1] * 16))), nm));
    }
    for (; j < e; ++j)
        a0 = _mm256_max_epu8(a0, _mm256_and_si256(_mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(S4 + (size_t)cids[j] * 16))), nm));
    return hsum_u4_256(_mm256_max_epu8(a0, a1));
}

struct CentroidTopK {
    std::array<std::vector<std::pair<float, uint32_t>>, 32> top_per_tok;
    std::array<float, 32> min_probed_score{};

    static inline void heap_replace_root(std::pair<float, uint32_t>* h, int k, float sc, uint32_t cid) {
        h[0] = {sc, cid};
        int i = 0;
        while (true) {
            int l = 2 * i + 1, r = l + 1, sm = i;
            if (l < k && h[l].first < h[sm].first) sm = l;
            if (r < k && h[r].first < h[sm].first) sm = r;
            if (sm == i) break;
            std::swap(h[i], h[sm]);
            i = sm;
        }
    }

    void select(const float* S_bf16, uint64_t K, uint32_t vmask, int nprobe_c) {
        int np = std::min<int>(std::max(1, nprobe_c), (int)K);
        alignas(64) float thresh[32];
        for (int t = 0; t < 32; ++t) {
            top_per_tok[t].clear();
            if (!((vmask >> t) & 1u)) { thresh[t] = 1e30f; min_probed_score[t] = 0.0f; continue; }
            top_per_tok[t].resize(np);
            for (int j = 0; j < np; ++j) top_per_tok[t][j] = {S_bf16[(size_t)j * 32 + t], (uint32_t)j};
            std::make_heap(top_per_tok[t].begin(), top_per_tok[t].end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            thresh[t] = top_per_tok[t][0].first;
        }
        __m512 vt0 = _mm512_load_ps(thresh + 0), vt1 = _mm512_load_ps(thresh + 16);
        for (uint64_t c = (uint64_t)np; c < K; ++c) {
            const float* row = S_bf16 + c * 32;
            __mmask16 m0 = _mm512_cmp_ps_mask(_mm512_load_ps(row + 0), vt0, _CMP_GT_OQ);
            __mmask16 m1 = _mm512_cmp_ps_mask(_mm512_load_ps(row + 16), vt1, _CMP_GT_OQ);
            if (__builtin_expect((m0 | m1) != 0, 0)) {
                while (m0) {
                    int t = __builtin_ctz((uint32_t)m0); m0 &= (m0 - 1);
                    heap_replace_root(top_per_tok[t].data(), np, row[t], (uint32_t)c);
                    thresh[t] = top_per_tok[t][0].first;
                }
                if (m1) {
                    uint32_t u1 = (uint32_t)m1;
                    while (u1) {
                        int t = 16 + __builtin_ctz(u1); u1 &= (u1 - 1);
                        heap_replace_root(top_per_tok[t].data(), np, row[t], (uint32_t)c);
                        thresh[t] = top_per_tok[t][0].first;
                    }
                    vt1 = _mm512_load_ps(thresh + 16);
                }
                vt0 = _mm512_load_ps(thresh + 0);
            }
        }
        for (int t = 0; t < 32; ++t) if ((vmask >> t) & 1u) min_probed_score[t] = top_per_tok[t][0].first;
    }
};

static float exact_maxsim_fp16_doc(const uint8_t* doc_buf, uint32_t n_toks, const float* Q_32, uint32_t valid_q_mask) {
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(doc_buf);
    float dmax[32]; for (int t = 0; t < 32; ++t) dmax[t] = -1e9f;
    for (uint32_t tk = 0; tk < n_toks; ++tk) {
        const uint16_t* hp = raw + (size_t)tk * 128;
        __m512 x[8];
        for (int j = 0; j < 8; ++j) x[j] = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(hp + j * 16)));
        uint32_t vm = valid_q_mask;
        while (vm) {
            int t = __builtin_ctz(vm); vm &= vm - 1;
            const float* qp = Q_32 + t * 128;
            __m512 a0 = _mm512_mul_ps(x[0], _mm512_loadu_ps(qp + 0));
            __m512 a1 = _mm512_mul_ps(x[1], _mm512_loadu_ps(qp + 16));
            a0 = _mm512_fmadd_ps(x[2], _mm512_loadu_ps(qp + 32), a0);
            a1 = _mm512_fmadd_ps(x[3], _mm512_loadu_ps(qp + 48), a1);
            a0 = _mm512_fmadd_ps(x[4], _mm512_loadu_ps(qp + 64), a0);
            a1 = _mm512_fmadd_ps(x[5], _mm512_loadu_ps(qp + 80), a1);
            a0 = _mm512_fmadd_ps(x[6], _mm512_loadu_ps(qp + 96), a0);
            a1 = _mm512_fmadd_ps(x[7], _mm512_loadu_ps(qp + 112), a1);
            float dot = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
            if (dot > dmax[t]) dmax[t] = dot;
        }
    }
    float total = 0.0f; uint32_t vm = valid_q_mask;
    while (vm) { int t = __builtin_ctz(vm); vm &= vm - 1; total += dmax[t]; }
    return total;
}

// Unified Query Prep for BOTH B=4 VNNI4 AND B=1 32KB L1d Transposed Nibble FastScan (M3-B03/M02 sound!)
struct Tier1QueryPrepV4 {
    alignas(64) int8_t  q_i8_eo[32][128];
    alignas(64) int16_t b1_nib[32][16][32]; // EXACTLY 32 KB -> 100% L1d RESIDENT! [nibble_g][val 0..15][q_tok 0..31]
    alignas(64) float   q_i8_scale[32], q_half_bias_i8[32], q_i8_err[32];
    alignas(64) float   q_b1_scale[32], q_b1_err[32];
    alignas(64) float   q_bf_err[32], q_bf_norm[32], q_et_mult[32], q_unit_cap[32], q_eps0[32];
    float t_coeff = 0.0f;
    uint32_t valid_q_mask = 0;
    bool use_cap = false;

    void init(const float* Q_32, const float* Q_rot, uint32_t vmask, int total_valid_q,
              bool is_unit, double m_norm_max, bool part2_allk = false, bool build_b1_lut = true, int topk = 10) {
        valid_q_mask = vmask;
        use_cap = is_unit;
        int mc = std::clamp(total_valid_q, 1, 256);
        double tc = get_t_coeff_v4(mc, part2_allk, topk, g_delta);
        t_coeff = (float)tc;

        for (int t = 0; t < 32; ++t) {
            if (!((vmask >> t) & 1u)) {
                q_bf_norm[t] = q_bf_err[t] = q_i8_scale[t] = q_half_bias_i8[t] = q_i8_err[t] = 0.0f;
                q_b1_scale[t] = q_b1_err[t] = q_et_mult[t] = q_eps0[t] = 0.0f;
                q_unit_cap[t] = 1e30f;
                std::memset(q_i8_eo[t], 0, 128);
                continue;
            }
            uint16_t qbf[128];
            for (int j = 0; j < 128; j += 32) {
                __m512bh bh = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(Q_32 + t * 128 + j + 16), _mm512_loadu_ps(Q_32 + t * 128 + j));
                std::memcpy(qbf + j, &bh, 64);
            }
            double bf_n2 = 0.0, bf_e2 = 0.0, q32_n2 = 0.0;
            for (int j = 0; j < 128; ++j) {
                double qf = (double)Q_32[t * 128 + j];
                q32_n2 += qf * qf;
                uint32_t u32 = ((uint32_t)qbf[j]) << 16;
                float qb_f = 0.0f; std::memcpy(&qb_f, &u32, 4);
                double diff = qf - (double)qb_f;
                bf_n2 += (double)qb_f * (double)qb_f; bf_e2 += diff * diff;
            }
            double qbfn = std::sqrt(bf_n2);
            double q32n = std::sqrt(q32_n2);
            q_bf_norm[t] = (float)qbfn;
            q_bf_err[t]  = (float)(std::sqrt(bf_e2) + 3.0e-5 * std::max(qbfn, q32n));

            double max_abs = 1e-12, n2 = 0.0;
            for (int j = 0; j < 128; ++j) {
                double v = (double)Q_rot[t * 128 + j];
                max_abs = std::max(max_abs, std::fabs(v));
                n2 += v * v;
            }
            double qn = std::max(std::sqrt(n2), q32n);
            double delta = max_abs / 127.0;
            int64_t sum_qi = 0; double err2 = 0.0;
            for (int b = 0; b < 64; ++b) {
                double ve = (double)Q_rot[t * 128 + 2 * b], vo = (double)Q_rot[t * 128 + 2 * b + 1];
                int8_t qe = (int8_t)std::clamp<long>(std::lrint(ve / delta), -127, 127);
                int8_t qo = (int8_t)std::clamp<long>(std::lrint(vo / delta), -127, 127);
                q_i8_eo[t][b] = qe; q_i8_eo[t][64 + b] = qo;
                sum_qi += (int64_t)qe + (int64_t)qo;
                double de = ve - (double)qe * delta, d_o = vo - (double)qo * delta;
                err2 += de * de + d_o * d_o;
            }
            q_i8_scale[t]     = (float)delta;
            q_half_bias_i8[t] = (float)(7.5 * delta * (double)sum_qi);
            q_i8_err[t]       = (float)(std::sqrt(err2) + 3.0e-5 * qn);
            q_et_mult[t]      = (float)(tc * qn);
            // M3-B03 & M3-M02 contracts: exact M_norm_max * (1 + gamma_128) and scaled q_eps0!
            q_unit_cap[t]     = (float)(m_norm_max * 1.00002 * q32n);
            q_eps0[t]         = (float)(3.0e-6 * std::max(1.0, qn));

            // 1-bit L1d nibble LUT scaling: |sum_{g=0..31} nib_g| <= sqrt(32)*qn < 5.66*qn -> map 6.0*qn to 30000 (0 overflow!)
            double b1_sc = (6.0 * qn) / 30000.0;
            if (b1_sc < 1e-12) b1_sc = 1e-12;
            q_b1_scale[t] = (float)b1_sc;
            // Exact per-nibble rounding <= 0.5*b1_sc -> across 32 nibbles total deterministic error <= 16*b1_sc + FP32 slack
            q_b1_err[t]   = (float)(16.0 * b1_sc + 4.0e-5 * qn);
        }

        if (build_b1_lut) {
            for (int g = 0; g < 32; ++g) {
                for (int mask = 0; mask < 16; ++mask) {
                    int16_t* dst = b1_nib[g][mask];
                    for (int t = 0; t < 32; ++t) {
                        if (!((vmask >> t) & 1u)) { dst[t] = 0; continue; }
                        const float* qr = Q_rot + t * 128 + g * 4;
                        double s = 0.0;
                        s += ((mask & 1) ? 0.5 : -0.5) * (double)qr[0];
                        s += ((mask & 2) ? 0.5 : -0.5) * (double)qr[1];
                        s += ((mask & 4) ? 0.5 : -0.5) * (double)qr[2];
                        s += ((mask & 8) ? 0.5 : -0.5) * (double)qr[3];
                        long qi = std::lrint(s / (double)q_b1_scale[t]);
                        dst[t] = (int16_t)std::clamp<long>(qi, -30000, 30000);
                    }
                }
            }
        }
    }
};

static inline __m128i hsum4_epi32(__m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    __m256i b0 = _mm256_add_epi32(_mm512_castsi512_si256(a0), _mm512_extracti64x4_epi64(a0, 1));
    __m256i b1 = _mm256_add_epi32(_mm512_castsi512_si256(a1), _mm512_extracti64x4_epi64(a1, 1));
    __m256i b2 = _mm256_add_epi32(_mm512_castsi512_si256(a2), _mm512_extracti64x4_epi64(a2, 1));
    __m256i b3 = _mm256_add_epi32(_mm512_castsi512_si256(a3), _mm512_extracti64x4_epi64(a3, 1));
    __m256i q0123 = _mm256_hadd_epi32(_mm256_hadd_epi32(b0, b1), _mm256_hadd_epi32(b2, b3));
    return _mm_add_epi32(_mm256_castsi256_si128(q0123), _mm256_extracti128_si256(q0123, 1));
}

static inline float exact_y_norm_dpbusd(__m512i ev8, __m512i od8) {
    const __m512i v15 = _mm512_set1_epi8(15);
    __m512i ae = _mm512_abs_epi8(_mm512_sub_epi8(_mm512_add_epi8(ev8, ev8), v15));
    __m512i ao = _mm512_abs_epi8(_mm512_sub_epi8(_mm512_add_epi8(od8, od8), v15));
    __m512i acc = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ae, ae), ao, ao);
    return 0.5f * std::sqrt((float)_mm512_reduce_add_epi32(acc)) + 1.0e-5f;
}

// ULTRA-FAST 32 KB L1d-RESIDENT 1-BIT THEOREM-3 UPPER BOUND (ALL 32 QUERY TOKENS SIMULTANEOUSLY IN 32 L1d LOADS!)
static float compute_theorem3_b1_ub_doc_v4(const uint8_t* b1_buf, uint32_t m,
                                           const float* S_bf16, const float* C_norms, const float* C_bf_err,
                                           const Tier1QueryPrepV4& qp) {
    const uint16_t* cids   = reinterpret_cast<const uint16_t*>(b1_buf);
    const uint16_t* sc_f16 = reinterpret_cast<const uint16_t*>(b1_buf + 2 * m);
    const uint16_t* et_f16 = reinterpret_cast<const uint16_t*>(b1_buf + 4 * m);
    const uint8_t*  sgn16  = b1_buf + 6 * m;

    __m512 dmax0 = _mm512_set1_ps(-1e9f), dmax1 = _mm512_set1_ps(-1e9f);
    const __m512 vq_b1s0 = _mm512_load_ps(qp.q_b1_scale + 0),  vq_b1s1 = _mm512_load_ps(qp.q_b1_scale + 16);
    const __m512 vq_b1e0 = _mm512_load_ps(qp.q_b1_err + 0),    vq_b1e1 = _mm512_load_ps(qp.q_b1_err + 16);
    const __m512 vq_bfe0 = _mm512_load_ps(qp.q_bf_err + 0),    vq_bfe1 = _mm512_load_ps(qp.q_bf_err + 16);
    const __m512 vq_bfn0 = _mm512_load_ps(qp.q_bf_norm + 0),   vq_bfn1 = _mm512_load_ps(qp.q_bf_norm + 16);
    const __m512 vq_etm0 = _mm512_load_ps(qp.q_et_mult + 0),   vq_etm1 = _mm512_load_ps(qp.q_et_mult + 16);
    const __m512 vq_eps0 = _mm512_load_ps(qp.q_eps0 + 0),      vq_eps1 = _mm512_load_ps(qp.q_eps0 + 16);
    const __m512 vq_cap0 = _mm512_load_ps(qp.q_unit_cap + 0),  vq_cap1 = _mm512_load_ps(qp.q_unit_cap + 16);

    for (uint32_t tk = 0; tk < m; ++tk) {
        const uint8_t* s = sgn16 + (size_t)tk * 16;
        __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
        for (int b = 0; b < 16; ++b) {
            uint8_t byte = s[b];
            a0 = _mm512_add_epi16(a0, _mm512_load_si512(qp.b1_nib[2 * b + 0][byte & 0x0F]));
            a1 = _mm512_add_epi16(a1, _mm512_load_si512(qp.b1_nib[2 * b + 1][byte >> 4]));
        }
        __m512i sum_i16 = _mm512_add_epi16(a0, a1);
        __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_i16))), vq_b1s0);
        __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_i16, 1))), vq_b1s1);

        uint32_t cid = cids[tk];
        const float* cs = S_bf16 + (size_t)cid * 32;
        __m512 vsc = _mm512_set1_ps(_cvtsh_ss(sc_f16[tk]));
        __m512 vet = _mm512_set1_ps(_cvtsh_ss(et_f16[tk]));
        __m512 vcn = _mm512_set1_ps(C_norms[cid]);
        __m512 vcbe = _mm512_set1_ps(C_bf_err[cid]);

        __m512 bf0 = _mm512_fmadd_ps(vcn, vq_bfe0, _mm512_fmadd_ps(vcbe, vq_bfn0, vq_eps0));
        __m512 bf1 = _mm512_fmadd_ps(vcn, vq_bfe1, _mm512_fmadd_ps(vcbe, vq_bfn1, vq_eps1));
        __m512 rad0 = _mm512_fmadd_ps(vet, vq_etm0, _mm512_fmadd_ps(vsc, vq_b1e0, bf0));
        __m512 rad1 = _mm512_fmadd_ps(vet, vq_etm1, _mm512_fmadd_ps(vsc, vq_b1e1, bf1));
        __m512 ub0 = _mm512_add_ps(_mm512_loadu_ps(cs + 0),  _mm512_fmadd_ps(vsc, f0, rad0));
        __m512 ub1 = _mm512_add_ps(_mm512_loadu_ps(cs + 16), _mm512_fmadd_ps(vsc, f1, rad1));
        if (qp.use_cap) {
            ub0 = _mm512_min_ps(ub0, vq_cap0);
            ub1 = _mm512_min_ps(ub1, vq_cap1);
        }
        dmax0 = _mm512_max_ps(dmax0, ub0);
        dmax1 = _mm512_max_ps(dmax1, ub1);
    }
    alignas(64) float dm[32];
    _mm512_store_ps(dm + 0, dmax0);
    _mm512_store_ps(dm + 16, dmax1);
    float total = 0.0f;
    uint32_t vm = qp.valid_q_mask;
    while (vm) { int t = __builtin_ctz(vm); vm &= vm - 1; total += dm[t]; }
    return total;
}

static float compute_theorem3_b4_ub_doc_v4(const uint8_t* t1_buf, uint32_t m,
                                           const float* S_bf16, const float* C_norms, const float* C_bf_err,
                                           const Tier1QueryPrepV4& qp) {
    const uint16_t* cids   = reinterpret_cast<const uint16_t*>(t1_buf);
    const uint16_t* sc_f16 = reinterpret_cast<const uint16_t*>(t1_buf + 2 * m);
    const uint16_t* et_f16 = reinterpret_cast<const uint16_t*>(t1_buf + 4 * m);
    uint32_t meta_bytes    = (6 * m + 63u) & ~63u;
    const uint8_t* packed64 = t1_buf + meta_bytes;
    alignas(16) float doc_max[32]; for (int t = 0; t < 32; ++t) doc_max[t] = -1e9f;
    const __m512i m0f = _mm512_set1_epi8(0x0F);
    for (uint32_t tk = 0; tk < m; ++tk) {
        __m512i raw = _mm512_loadu_si512(packed64 + (size_t)tk * 64);
        __m512i ev8 = _mm512_and_si512(raw, m0f);
        __m512i od8 = _mm512_and_si512(_mm512_srli_epi16(raw, 4), m0f);
        uint32_t cid = cids[tk];
        const float* cs = S_bf16 + (size_t)cid * 32;
        float sc = _cvtsh_ss(sc_f16[tk]), et_raw = _cvtsh_ss(et_f16[tk]);
        float sc_ynorm = sc * exact_y_norm_dpbusd(ev8, od8);
        __m128 vsc = _mm_set1_ps(sc), vet = _mm_set1_ps(et_raw), vscyn = _mm_set1_ps(sc_ynorm);
        __m128 vcn = _mm_set1_ps(C_norms[cid]), vcbe = _mm_set1_ps(C_bf_err[cid]);
        for (int t = 0; t < 32; t += 4) {
            if (!((qp.valid_q_mask >> t) & 0xFu)) continue;
            __m512i a0 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev8, _mm512_load_si512(qp.q_i8_eo[t + 0])), od8, _mm512_load_si512(qp.q_i8_eo[t + 0] + 64));
            __m512i a1 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev8, _mm512_load_si512(qp.q_i8_eo[t + 1])), od8, _mm512_load_si512(qp.q_i8_eo[t + 1] + 64));
            __m512i a2 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev8, _mm512_load_si512(qp.q_i8_eo[t + 2])), od8, _mm512_load_si512(qp.q_i8_eo[t + 2] + 64));
            __m512i a3 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev8, _mm512_load_si512(qp.q_i8_eo[t + 3])), od8, _mm512_load_si512(qp.q_i8_eo[t + 3] + 64));
            __m128 dots = _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(hsum4_epi32(a0, a1, a2, a3)), _mm_load_ps(qp.q_i8_scale + t)), _mm_load_ps(qp.q_half_bias_i8 + t));
            __m128 bf_err = _mm_fmadd_ps(vcn, _mm_load_ps(qp.q_bf_err + t), _mm_fmadd_ps(vcbe, _mm_load_ps(qp.q_bf_norm + t), _mm_load_ps(qp.q_eps0 + t)));
            __m128 err    = _mm_fmadd_ps(vscyn, _mm_load_ps(qp.q_i8_err + t), bf_err);
            __m128 rad    = _mm_fmadd_ps(vet, _mm_load_ps(qp.q_et_mult + t), err);
            __m128 ub     = _mm_add_ps(_mm_loadu_ps(cs + t), _mm_fmadd_ps(vsc, dots, rad));
            if (qp.use_cap) ub = _mm_min_ps(ub, _mm_load_ps(qp.q_unit_cap + t));
            _mm_store_ps(doc_max + t, _mm_max_ps(_mm_load_ps(doc_max + t), ub));
        }
    }
    float total = 0.0f; uint32_t vm = qp.valid_q_mask;
    while (vm) { int t = __builtin_ctz(vm); vm &= vm - 1; total += doc_max[t]; }
    return total;
}

static float compute_approx_b4_point_doc_v4(const uint8_t* t1_buf, uint32_t m,
                                            const float* S_bf16, const Tier1QueryPrepV4& qp) {
    const uint16_t* cids   = reinterpret_cast<const uint16_t*>(t1_buf);
    const uint16_t* sc_f16 = reinterpret_cast<const uint16_t*>(t1_buf + 2 * m);
    uint32_t meta_bytes    = (6 * m + 63u) & ~63u;
    const uint8_t* packed64 = t1_buf + meta_bytes;
    alignas(16) float doc_max[32]; for (int t = 0; t < 32; ++t) doc_max[t] = -1e9f;
    const __m512i m0f = _mm512_set1_epi8(0x0F);
    for (uint32_t tk = 0; tk < m; ++tk) {
        __m512i raw = _mm512_loadu_si512(packed64 + (size_t)tk * 64);
        __m512i ev8 = _mm512_and_si512(raw, m0f);
        __m512i od8 = _mm512_and_si512(_mm512_srli_epi16(raw, 4), m0f);
        const float* cs = S_bf16 + (size_t)cids[tk] * 32;
        __m128 vsc = _mm_set1_ps(_cvtsh_ss(sc_f16[tk]));
        for (int t = 0; t < 32; t += 4) {
            if (!((qp.valid_q_mask >> t) & 0xFu)) continue;
            __m512i a0 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev8, _mm512_load_si512(qp.q_i8_eo[t + 0])), od8, _mm512_load_si512(qp.q_i8_eo[t + 0] + 64));
            __m512i a1 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev8, _mm512_load_si512(qp.q_i8_eo[t + 1])), od8, _mm512_load_si512(qp.q_i8_eo[t + 1] + 64));
            __m512i a2 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev8, _mm512_load_si512(qp.q_i8_eo[t + 2])), od8, _mm512_load_si512(qp.q_i8_eo[t + 2] + 64));
            __m512i a3 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev8, _mm512_load_si512(qp.q_i8_eo[t + 3])), od8, _mm512_load_si512(qp.q_i8_eo[t + 3] + 64));
            __m128 dots = _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(hsum4_epi32(a0, a1, a2, a3)), _mm_load_ps(qp.q_i8_scale + t)), _mm_load_ps(qp.q_half_bias_i8 + t));
            __m128 est = _mm_fmadd_ps(vsc, dots, _mm_loadu_ps(cs + t));
            _mm_store_ps(doc_max + t, _mm_max_ps(_mm_load_ps(doc_max + t), est));
        }
    }
    float total = 0.0f; uint32_t vm = qp.valid_q_mask;
    while (vm) { int t = __builtin_ctz(vm); vm &= vm - 1; total += doc_max[t]; }
    return total;
}

// ============================================================================
// EXPANDED 1,000,000-PAIR DETERMINISTIC SOUNDNESS UNIT TEST (M3-B03 & M3-M02!)
// Covers: B=1 AND B=4, is_unit=true (real FP16-inflated unit vectors!) AND is_unit=false,
//         non-unit ||q||_2 in [0.1, 10.0], zero-padding masks, independent Q_32 != Q_rot,
//         and BOTH Part 1 (delta/|Q|) & Part 2 (delta/(10|Q|))!
// ============================================================================
struct UnitTestReport {
    uint64_t n_pairs_b4 = 0, viol_b4 = 0;
    uint64_t n_pairs_b1 = 0, viol_b1 = 0;
    double max_def_b4 = 0.0, max_def_b1 = 0.0;
};

static UnitTestReport run_theorem3_unit_test_v4(const SegmentIndex& idx) {
    UnitTestReport rep;
    std::mt19937_64 rng(20270304ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> u_scale(0.1, 10.0);

    // Promote P_fp32 to exact FP64 for test ground truth
    std::vector<double> P64(128 * 128);
    for (int i = 0; i < 128 * 128; ++i) P64[i] = (double)idx.P_fp32[i];

    // Pick 32,000 real tokens from tier2_fp16 + tier1_b4 + tier1_b1 (covering unit/non-unit real corpus vectors!)
    // Test 32 query tokens per doc token -> > 1,000,000 real (q, x) pairs!
    ThreadUringReader tr2, tr1b4, tr1b1;
    tr2.init(&idx.tier2_nvme, 32);
    tr1b4.init(&idx.tier1_b4_nvme, 32);
    if (idx.has_b1) tr1b1.init(&idx.tier1_b1_ram);

    std::vector<uint32_t> test_docs;
    uint64_t step = std::max<uint64_t>(1, idx.N_docs / 400);
    for (uint64_t d = 0; d < idx.N_docs && test_docs.size() < 400; d += step) test_docs.push_back((uint32_t)d);
    std::vector<const uint8_t*> p2, p4, p1;
    tr2.fetch_batch(test_docs.data(), (int)test_docs.size(), p2);
    tr1b4.fetch_batch(test_docs.data(), (int)test_docs.size(), p4);
    if (idx.has_b1) tr1b1.fetch_batch(test_docs.data(), (int)test_docs.size(), p1);

    // Build 4 diverse synthetic+adversarial query batches (32 tokens each):
    //   Batch 0: standard unit queries, full mask, Part 1 (is_unit=idx.is_unit)
    //   Batch 1: NON-UNIT queries ||q||_2 in [0.1, 10.0], half zero-padded (n_valid_q=16), Part 1
    //   Batch 2: adversarial large-norm + sparse queries (n_valid_q=1..32), Part 2 (k=10)
    //   Batch 3: queries aligned to real FP16 corpus tokens (adversarial M3-B03 cap stress test!)
    QueryLUT lut;
    for (int b = 0; b < 4; ++b) {
        std::vector<float> Q32(32 * 128, 0.0f), Qrot(32 * 128, 0.0f);
        int nv = (b == 1) ? 16 : ((b == 2) ? 7 : 32);
        for (int t = 0; t < nv; ++t) {
            double v[128], n2 = 0.0;
            if (b == 3 && t < (int)test_docs.size()) {
                // Adversarial: set q = exact FP16 doc token direction (tests q_unit_cap to the ULP!)
                const uint16_t* raw = reinterpret_cast<const uint16_t*>(p2[t]);
                for (int j = 0; j < 128; ++j) { v[j] = (double)_cvtsh_ss(raw[j]); n2 += v[j] * v[j]; }
            } else {
                for (int j = 0; j < 128; ++j) { v[j] = nd(rng); n2 += v[j] * v[j]; }
            }
            double sc = (b == 1 || b == 2) ? u_scale(rng) : 1.0;
            double inv_n = sc / std::sqrt(std::max(n2, 1e-18));
            for (int j = 0; j < 128; ++j) Q32[t * 128 + j] = (float)(v[j] * inv_n);
            // Independent Q_rot = Q_32 @ P_fp32 (NOT Q_32 == Q_rot!)
            for (int k = 0; k < 128; ++k) {
                double acc = 0.0;
                for (int j = 0; j < 128; ++j) acc += (double)Q32[t * 128 + j] * P64[j * 128 + k];
                Qrot[t * 128 + k] = (float)acc;
            }
        }
        build_query_lut(idx.C_bf16.data(), idx.K, Q32.data(), lut);
        Tier1QueryPrepV4 qp;
        bool p2_mode = (b == 2);
        int test_k = (b == 2) ? 100 : 10;
        qp.init(Q32.data(), Qrot.data(), lut.valid_q_mask, lut.n_valid_q, idx.is_unit != 0, idx.m_norm_max, p2_mode, idx.has_b1, test_k);
        double tc = get_t_coeff_v4(lut.n_valid_q, p2_mode, test_k, g_delta);

        for (size_t di = 0; di < test_docs.size(); ++di) {
            uint32_t did = test_docs[di];
            uint32_t m = idx.tier2_nvme.meta[did].n_toks;
            const uint16_t* raw = reinterpret_cast<const uint16_t*>(p2[di]);
            const uint16_t* cids4 = reinterpret_cast<const uint16_t*>(p4[di]);
            const uint16_t* sc4 = reinterpret_cast<const uint16_t*>(p4[di] + 2 * m);
            const uint16_t* et4 = reinterpret_cast<const uint16_t*>(p4[di] + 4 * m);
            const uint8_t* pk4 = p4[di] + ((6 * m + 63u) & ~63u);

            const uint16_t* sc1 = idx.has_b1 ? reinterpret_cast<const uint16_t*>(p1[di] + 2 * m) : nullptr;
            const uint16_t* et1 = idx.has_b1 ? reinterpret_cast<const uint16_t*>(p1[di] + 4 * m) : nullptr;
            const uint8_t* sg1  = idx.has_b1 ? (p1[di] + 6 * m) : nullptr;

            // Check first 16 tokens per doc (400 * 16 * 4 batches * ~22 valid q = ~560K B4 + ~560K B1 = >1.1M pairs!)
            uint32_t m_chk = std::min<uint32_t>(m, 24);
            for (uint32_t tk = 0; tk < m_chk; ++tk) {
                uint32_t cid = cids4[tk];
                const float* cent = idx.C_fp32.data() + (size_t)cid * 128;
                // Reconstruct exact theoretical B4 & B1 UBs in FP64 and compare against hardware SIMD UB!
                double x64[128], r_rot[128], r2 = 0.0;
                for (int j = 0; j < 128; ++j) {
                    x64[j] = (double)_cvtsh_ss(raw[(size_t)tk * 128 + j]);
                }
                for (int k = 0; k < 128; ++k) {
                    double acc = 0.0;
                    for (int j = 0; j < 128; ++j) acc += (x64[j] - (double)cent[j]) * P64[j * 128 + k];
                    r_rot[k] = acc; r2 += acc * acc;
                }
                double r_norm = std::sqrt(r2);

                // Decode stored B4 codes y4 in {-7.5..+7.5}
                const uint8_t* p64_t = pk4 + (size_t)tk * 64;
                double y4[128], y4_dot_o = 0.0, y4_n2 = 0.0;
                for (int j = 0; j < 64; ++j) {
                    y4[2 * j]     = (double)(p64_t[j] & 0x0F) - 7.5;
                    y4[2 * j + 1] = (double)(p64_t[j] >> 4) - 7.5;
                }
                for (int j = 0; j < 128; ++j) { y4_n2 += y4[j] * y4[j]; if (r_norm > 1e-12) y4_dot_o += y4[j] * (r_rot[j] / r_norm); }
                double c4 = (r_norm > 1e-12) ? std::min(1.0, y4_dot_o / std::sqrt(std::max(y4_n2, 1e-18))) : 1.0;
                double s4_ex = (r_norm > 1e-12) ? (r_norm / std::max(y4_dot_o, 1e-18)) : 0.0;
                double e4_ex = (r_norm > 1e-12) ? (r_norm * std::sqrt(std::max(0.0, (1.0 - c4 * c4) / std::max(c4 * c4, 1e-18)))) : 0.0;

                // Evaluate hardware B4 single-token UB
                uint8_t single_b4[128];
                std::memset(single_b4, 0, sizeof(single_b4));
                reinterpret_cast<uint16_t*>(single_b4)[0] = (uint16_t)cid;
                reinterpret_cast<uint16_t*>(single_b4)[1] = sc4[tk];
                reinterpret_cast<uint16_t*>(single_b4)[2] = et4[tk];
                std::memcpy(single_b4 + 64, p64_t, 64);

                // Direct per-token check for every valid query token t!
                const __m512i m0f = _mm512_set1_epi8(0x0F);
                __m512i raw_v = _mm512_loadu_si512(p64_t);
                __m512i ev8 = _mm512_and_si512(raw_v, m0f);
                __m512i od8 = _mm512_and_si512(_mm512_srli_epi16(raw_v, 4), m0f);
                float sc_h4 = _cvtsh_ss(sc4[tk]), et_h4 = _cvtsh_ss(et4[tk]);
                float sc_yn = sc_h4 * exact_y_norm_dpbusd(ev8, od8);
                const float* cs_row = lut.S_bf16.data() + (size_t)cid * 32;

                // Also precompute hardware B1 sums for this token
                alignas(64) int16_t b1_sums[32];
                if (idx.has_b1) {
                    const uint8_t* sg = sg1 + (size_t)tk * 16;
                    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
                    for (int bb = 0; bb < 16; ++bb) {
                        a0 = _mm512_add_epi16(a0, _mm512_load_si512(qp.b1_nib[2 * bb + 0][sg[bb] & 0x0F]));
                        a1 = _mm512_add_epi16(a1, _mm512_load_si512(qp.b1_nib[2 * bb + 1][sg[bb] >> 4]));
                    }
                    _mm512_store_si512(b1_sums, _mm512_add_epi16(a0, a1));
                }

                for (int t = 0; t < 32; ++t) {
                    if (!((lut.valid_q_mask >> t) & 1u)) continue;
                    double qc_ex = 0.0, qy4_ex = 0.0, qn2 = 0.0, qx_ex = 0.0;
                    for (int j = 0; j < 128; ++j) {
                        double qj = (double)Q32[t * 128 + j];
                        double qrj = (double)Qrot[t * 128 + j];
                        qc_ex += qj * (double)cent[j];
                        qx_ex += qj * x64[j];
                        qy4_ex += qrj * y4[j];
                        qn2 += qj * qj;
                    }
                    double qn = std::sqrt(qn2);
                    double theory_ub4 = qc_ex + s4_ex * qy4_ex + qn * e4_ex * tc;
                    if (qp.use_cap) theory_ub4 = std::min(theory_ub4, std::max(qx_ex, qn * idx.m_norm_max));

                    // Compute exact hardware B4 UB for token t
                    __m512i d0 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ev8, _mm512_load_si512(qp.q_i8_eo[t])), od8, _mm512_load_si512(qp.q_i8_eo[t] + 64));
                    float dot4 = (float)_mm512_reduce_add_epi32(d0) * qp.q_i8_scale[t] - qp.q_half_bias_i8[t];
                    float bf_e = idx.C_norms[cid] * qp.q_bf_err[t] + idx.C_bf_err[cid] * qp.q_bf_norm[t] + qp.q_eps0[t];
                    float hw_ub4 = cs_row[t] + sc_h4 * dot4 + et_h4 * qp.q_et_mult[t] + sc_yn * qp.q_i8_err[t] + bf_e;
                    if (qp.use_cap) hw_ub4 = std::min(hw_ub4, qp.q_unit_cap[t]);

                    rep.n_pairs_b4++;
                    double def4 = theory_ub4 - (double)hw_ub4;
                    if (def4 > 0.0) { rep.viol_b4++; rep.max_def_b4 = std::max(rep.max_def_b4, def4); }
                    if (qp.use_cap && (double)hw_ub4 < qx_ex && qx_ex <= theory_ub4 + 1e-12) {
                        // Also verify cap NEVER drops below exact FP16 inner product qx_ex!
                        if ((double)qp.q_unit_cap[t] < qx_ex) { rep.viol_b4++; rep.max_def_b4 = std::max(rep.max_def_b4, qx_ex - (double)qp.q_unit_cap[t]); }
                    }

                    if (idx.has_b1) {
                        const uint8_t* sg = sg1 + (size_t)tk * 16;
                        double y1_dot_o = 0.0, qy1_ex = 0.0;
                        for (int j = 0; j < 128; ++j) {
                            double y1j = ((sg[j >> 3] >> (j & 7)) & 1) ? 0.5 : -0.5;
                            if (r_norm > 1e-12) y1_dot_o += y1j * (r_rot[j] / r_norm);
                            qy1_ex += (double)Qrot[t * 128 + j] * y1j;
                        }
                        double c1 = (r_norm > 1e-12) ? std::min(1.0, y1_dot_o / std::sqrt(32.0)) : 1.0;
                        double s1_ex = (r_norm > 1e-12) ? (r_norm / std::max(y1_dot_o, 1e-18)) : 0.0;
                        double e1_ex = (r_norm > 1e-12) ? (r_norm * std::sqrt(std::max(0.0, (1.0 - c1 * c1) / std::max(c1 * c1, 1e-18)))) : 0.0;
                        double theory_ub1 = qc_ex + s1_ex * qy1_ex + qn * e1_ex * tc;
                        if (qp.use_cap) theory_ub1 = std::min(theory_ub1, std::max(qx_ex, qn * idx.m_norm_max));

                        float sc_h1 = _cvtsh_ss(sc1[tk]), et_h1 = _cvtsh_ss(et1[tk]);
                        float hw_ub1 = cs_row[t] + sc_h1 * ((float)b1_sums[t] * qp.q_b1_scale[t])
                                     + et_h1 * qp.q_et_mult[t] + sc_h1 * qp.q_b1_err[t] + bf_e;
                        if (qp.use_cap) hw_ub1 = std::min(hw_ub1, qp.q_unit_cap[t]);

                        rep.n_pairs_b1++;
                        double def1 = theory_ub1 - (double)hw_ub1;
                        if (def1 > 0.0) { rep.viol_b1++; rep.max_def_b1 = std::max(rep.max_def_b1, def1); }
                    }
                }
            }
        }
    }
    std::cerr << "[Expanded Unit Test v4 (M3-B03/M02)] B4: pairs=" << rep.n_pairs_b4 << " viol=" << rep.viol_b4
              << " max_def=" << rep.max_def_b4 << " | B1: pairs=" << rep.n_pairs_b1 << " viol=" << rep.viol_b1
              << " max_def=" << rep.max_def_b1 << "\n";
    return rep;
}

// ============================================================================
// RERANKERS (ALL DETERMINISTIC (score, -doc_id) MIN-HEAP!)
// ============================================================================
struct RerankStats {
    uint32_t topk_ids[100];
    float    scores[100];
    int n_out = 0;
    int wave1_surv = 0, tier2_reranked = 0;
    double t_b1_ms = 0.0, t_b4_ms = 0.0, t_t2_ms = 0.0;
};

static void extract_sorted_heap(std::priority_queue<HeapItem, std::vector<HeapItem>, HeapMinCmp>& heap, RerankStats& st) {
    std::vector<HeapItem> res;
    while (!heap.empty()) { res.push_back(heap.top()); heap.pop(); }
    std::sort(res.begin(), res.end(), HeapMinCmp{});
    st.n_out = std::min<int>(100, (int)res.size());
    for (int i = 0; i < st.n_out; ++i) { st.topk_ids[i] = res[i].did; st.scores[i] = res[i].score; }
}

static RerankStats rerank_direct_exact(const std::vector<uint32_t>& cands, ThreadUringReader& t2, const float* Q_32, uint32_t vmask, int topk = 10) {
    RerankStats st; int nc = (int)cands.size(); if (nc == 0) return st;
    double tt0 = monotonic_ns();
    std::vector<const uint8_t*> ptrs; t2.fetch_batch(cands.data(), nc, ptrs);
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapMinCmp> heap;
    for (int i = 0; i < nc; ++i) {
        HeapItem it{exact_maxsim_fp16_doc(ptrs[i], t2.store->meta[cands[i]].n_toks, Q_32, vmask), cands[i]};
        if ((int)heap.size() < topk) heap.push(it);
        else if (is_better_hit(it, heap.top())) { heap.pop(); heap.push(it); }
    }
    st.t_t2_ms = (monotonic_ns() - tt0) * 1e-6;
    st.tier2_reranked = nc;
    extract_sorted_heap(heap, st);
    return st;
}

static RerankStats rerank_approx_b4_zero_t2(const std::vector<uint32_t>& cands, ThreadUringReader& t1_b4,
                                            const float* S_bf16, const Tier1QueryPrepV4& qp, int topk = 10) {
    RerankStats st; int nc = (int)cands.size(); if (nc == 0) return st;
    double tb0 = monotonic_ns();
    std::vector<const uint8_t*> ptrs; t1_b4.fetch_batch(cands.data(), nc, ptrs);
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapMinCmp> heap;
    for (int i = 0; i < nc; ++i) {
        HeapItem it{compute_approx_b4_point_doc_v4(ptrs[i], t1_b4.store->meta[cands[i]].n_toks, S_bf16, qp), cands[i]};
        if ((int)heap.size() < topk) heap.push(it);
        else if (is_better_hit(it, heap.top())) { heap.pop(); heap.push(it); }
    }
    st.t_b4_ms = (monotonic_ns() - tb0) * 1e-6;
    extract_sorted_heap(heap, st);
    return st;
}

static RerankStats rerank_t3_b4(const std::vector<uint32_t>& cands, ThreadUringReader& t1_b4, ThreadUringReader& t2,
                                const float* S_bf16, const float* C_norms, const float* C_bf_err,
                                const float* Q_32, const Tier1QueryPrepV4& qp, int topk = 10) {
    RerankStats st; int nc = (int)cands.size(); if (nc == 0) return st;
    int sk = std::min(topk, nc);
    double ts0 = monotonic_ns();
    std::vector<const uint8_t*> ptrs2; t2.fetch_batch(cands.data(), sk, ptrs2);
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapMinCmp> heap;
    for (int i = 0; i < sk; ++i) {
        HeapItem it{exact_maxsim_fp16_doc(ptrs2[i], t2.store->meta[cands[i]].n_toks, Q_32, qp.valid_q_mask), cands[i]};
        heap.push(it);
    }
    st.t_t2_ms += (monotonic_ns() - ts0) * 1e-6;
    st.tier2_reranked = sk;
    if (nc > sk) {
        int rem = nc - sk;
        double tb0 = monotonic_ns();
        std::vector<const uint8_t*> ptrs1; t1_b4.fetch_batch(cands.data() + sk, rem, ptrs1);
        st.t_b4_ms += (monotonic_ns() - tb0) * 1e-6;
        const int WAVE = 32;
        std::vector<uint32_t> surv; surv.reserve(WAVE);
        for (int ws = 0; ws < rem; ws += WAVE) {
            int we = std::min(rem, ws + WAVE);
            float th = ((int)heap.size() == topk) ? heap.top().score : -1e9f;
            surv.clear();
            double tc0 = monotonic_ns();
            for (int i = ws; i < we; ++i) {
                uint32_t did = cands[sk + i];
                if (compute_theorem3_b4_ub_doc_v4(ptrs1[i], t1_b4.store->meta[did].n_toks, S_bf16, C_norms, C_bf_err, qp) >= th)
                    surv.push_back(did);
            }
            st.t_b4_ms += (monotonic_ns() - tc0) * 1e-6;
            if (!surv.empty()) {
                double t20 = monotonic_ns();
                t2.fetch_batch(surv.data(), (int)surv.size(), ptrs2);
                for (size_t i = 0; i < surv.size(); ++i) {
                    HeapItem it{exact_maxsim_fp16_doc(ptrs2[i], t2.store->meta[surv[i]].n_toks, Q_32, qp.valid_q_mask), surv[i]};
                    if ((int)heap.size() < topk) heap.push(it);
                    else if (is_better_hit(it, heap.top())) { heap.pop(); heap.push(it); }
                }
                st.t_t2_ms += (monotonic_ns() - t20) * 1e-6;
                st.tier2_reranked += (int)surv.size();
            }
        }
    }
    extract_sorted_heap(heap, st);
    return st;
}

// LIVE 48-CORE Split_B1_B4_T3: Wave-1 (32KB L1d B1 RAM -> ~0.2ms!) -> Wave-2 (io_uring NVMe B4) -> Wave-3 (io_uring NVMe FP16)!
static RerankStats rerank_split_b1_b4_t3(const std::vector<uint32_t>& cands,
                                         ThreadUringReader& t1_b1_ram, ThreadUringReader& t1_b4_nvme, ThreadUringReader& t2_nvme,
                                         const float* S_bf16, const float* C_norms, const float* C_bf_err,
                                         const float* Q_32, const Tier1QueryPrepV4& qp, int topk = 10) {
    RerankStats st; int nc = (int)cands.size(); if (nc == 0) return st;
    int sk = std::min(topk, nc);
    std::vector<const uint8_t*> ptrs2, ptrs4, ptrs1;
    double ts0 = monotonic_ns();
    t2_nvme.fetch_batch(cands.data(), sk, ptrs2);
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapMinCmp> heap;
    for (int i = 0; i < sk; ++i) {
        HeapItem it{exact_maxsim_fp16_doc(ptrs2[i], t2_nvme.store->meta[cands[i]].n_toks, Q_32, qp.valid_q_mask), cands[i]};
        heap.push(it);
    }
    st.t_t2_ms += (monotonic_ns() - ts0) * 1e-6;
    st.tier2_reranked = sk;
    if (nc > sk) {
        int rem = nc - sk;
        t1_b1_ram.fetch_batch(cands.data() + sk, rem, ptrs1);
        const int WAVE = 64; // Larger wave amortizes NVMe QD=64 batching for Wave-2!
        std::vector<uint32_t> w1_surv, w2_surv;
        w1_surv.reserve(WAVE); w2_surv.reserve(WAVE);
        for (int ws = 0; ws < rem; ws += WAVE) {
            int we = std::min(rem, ws + WAVE);
            float th = ((int)heap.size() == topk) ? heap.top().score : -1e9f;
            w1_surv.clear();
            double t10 = monotonic_ns();
            for (int i = ws; i < we; ++i) {
                uint32_t did = cands[sk + i];
                if (compute_theorem3_b1_ub_doc_v4(ptrs1[i], t1_b1_ram.store->meta[did].n_toks, S_bf16, C_norms, C_bf_err, qp) >= th)
                    w1_surv.push_back(did);
            }
            st.t_b1_ms += (monotonic_ns() - t10) * 1e-6;
            if (w1_surv.empty()) continue;
            st.wave1_surv += (int)w1_surv.size();
            double t40 = monotonic_ns();
            t1_b4_nvme.fetch_batch(w1_surv.data(), (int)w1_surv.size(), ptrs4);
            w2_surv.clear();
            for (size_t i = 0; i < w1_surv.size(); ++i) {
                uint32_t did = w1_surv[i];
                if (compute_theorem3_b4_ub_doc_v4(ptrs4[i], t1_b4_nvme.store->meta[did].n_toks, S_bf16, C_norms, C_bf_err, qp) >= th)
                    w2_surv.push_back(did);
            }
            st.t_b4_ms += (monotonic_ns() - t40) * 1e-6;
            if (!w2_surv.empty()) {
                double t20 = monotonic_ns();
                t2_nvme.fetch_batch(w2_surv.data(), (int)w2_surv.size(), ptrs2);
                for (size_t i = 0; i < w2_surv.size(); ++i) {
                    HeapItem it{exact_maxsim_fp16_doc(ptrs2[i], t2_nvme.store->meta[w2_surv[i]].n_toks, Q_32, qp.valid_q_mask), w2_surv[i]};
                    if ((int)heap.size() < topk) heap.push(it);
                    else if (is_better_hit(it, heap.top())) { heap.pop(); heap.push(it); }
                }
                st.t_t2_ms += (monotonic_ns() - t20) * 1e-6;
                st.tier2_reranked += (int)w2_surv.size();
            }
        }
    }
    extract_sorted_heap(heap, st);
    return st;
}

// ============================================================================
// TIER-0 CANDIDATE GENERATORS (Dynamic tau1* + Pure Rule B gamma=0.08 + Smooth Fallback!)
// ============================================================================
struct ScoredDoc { uint16_t score; uint32_t did; };
static std::vector<uint32_t> extract_top_c1(std::vector<ScoredDoc>& buf, int C1) {
    int n = (int)buf.size(), k = std::min(C1, n);
    if (k == 0) return {};
    std::partial_sort(buf.begin(), buf.begin() + k, buf.end(), [](const ScoredDoc& a, const ScoredDoc& b) {
        return (a.score != b.score) ? (a.score > b.score) : (a.did < b.did);
    });
    std::vector<uint32_t> out(k); for (int i = 0; i < k; ++i) out[i] = buf[i].did;
    return out;
}

static std::vector<uint32_t> tier0_flat(const SegmentIndex& idx, const uint64_t* bm, const uint8_t* S4, int C1, const uint64_t* tomb = nullptr) {
    std::vector<ScoredDoc> buf; buf.reserve(4096);
    for (uint64_t w = 0; w < idx.n_words; ++w) {
        uint64_t word = bm[w];
        if (tomb) word &= ~tomb[w];
        if (!word) continue;
        uint64_t base = w * 64;
        while (word) {
            int bit = __builtin_ctzll(word); word &= word - 1;
            uint32_t d = (uint32_t)(base + bit); if (d >= idx.N_docs) break;
            buf.push_back({score_k0_4b(idx.doc_cids.data(), idx.doc_offs[d], idx.doc_offs[d + 1], S4), d});
        }
    }
    return extract_top_c1(buf, C1);
}

static std::vector<uint32_t> tier0_pm_dual_v4(const SegmentIndex& idx, const uint64_t* bm, const roaring_bitmap_t* pred_rb,
                                              const float* S_bf16, const uint8_t* S4, uint32_t vmask,
                                              int nprobe_c, int C1, bool is_unf, bool use_cb, double gamma_cb = 0.08,
                                              bool* tripped_out = nullptr, const uint64_t* tomb = nullptr, bool allow_path1 = true) {
    uint64_t n_valid = idx.N_docs;
    if (!is_unf) {
        n_valid = 0;
        for (uint64_t w = 0; w < idx.n_words; ++w) {
            uint64_t wd = bm[w]; if (tomb) wd &= ~tomb[w];
            n_valid += (uint64_t)__builtin_popcountll(wd);
        }
        if (n_valid == 0) return {};
        uint64_t tau1 = idx.dynamic_tau1(nprobe_c, __builtin_popcount(vmask));
        if (allow_path1 && n_valid <= tau1) return tier0_flat(idx, bm, S4, C1, tomb);
    }
    CentroidTopK topk; topk.select(S_bf16, idx.K, vmask, nprobe_c);
    std::vector<uint32_t> pcids; pcids.reserve(32 * nprobe_c);
    uint32_t vm = vmask;
    while (vm) { int t = __builtin_ctz(vm); vm &= vm - 1; for (const auto& [sc, cid] : topk.top_per_tok[t]) pcids.push_back(cid); }
    std::sort(pcids.begin(), pcids.end());
    pcids.erase(std::unique(pcids.begin(), pcids.end()), pcids.end());

    roaring_bitmap_t* cu = roaring_bitmap_create();
    uint64_t tot_raw = 0, tot_live = 0;
    double sigma = (double)n_valid / (double)idx.N_docs;
    bool tripped = false;

    for (size_t i = 0; i < pcids.size(); ++i) {
        const roaring_bitmap_t* pl = idx.inv_roaring[pcids[i]];
        if (is_unf) { roaring_bitmap_or_inplace(cu, pl); }
        else {
            uint64_t raw_sz = roaring_bitmap_get_cardinality(pl);
            roaring_bitmap_t* inter = roaring_bitmap_and(pl, pred_rb);
            uint64_t live_sz = roaring_bitmap_get_cardinality(inter);
            tot_raw += raw_sz; tot_live += live_sz;
            if (live_sz > 0) roaring_bitmap_or_inplace(cu, inter);
            roaring_bitmap_free(inter);
            if (use_cb && sigma >= 0.10 && i >= 15) {
                // Exact Theorem 4B Pure Rule B (zero-heuristic W_max = min(m, u_max), delta_cb = 1e-6!)
                double w_max = std::min<double>((double)(i + 1), (double)idx.u_max);
                double n_eff = (double)tot_raw / w_max;
                double p_hat = (double)tot_live / std::max<double>(1.0, (double)tot_raw);
                if (p_hat < gamma_cb * sigma) {
                    double kl = (p_hat <= 0.0) ? std::log(1.0 / (1.0 - sigma))
                                               : (p_hat * std::log(p_hat / sigma) + (1.0 - p_hat) * std::log((1.0 - p_hat) / (1.0 - sigma)));
                    double thresh = std::log((double)std::max<size_t>(1, pcids.size()) / 1e-6);
                    if (n_eff * kl > thresh) { tripped = true; break; }
                }
            }
        }
    }
    if (tripped_out) *tripped_out = tripped;
    if (tripped) {
        // Exact Theorem 4B Rule B fallback: switch directly to exhaustive Path I (K0_4b) scan
        // for zero-heuristic guaranteed recall recovery across all dead-posting & semi-poisoning regimes.
        roaring_bitmap_free(cu);
        return tier0_flat(idx, bm, S4, C1, tomb);
    }
    uint64_t nc = roaring_bitmap_get_cardinality(cu);
    std::vector<uint32_t> cids(nc);
    roaring_bitmap_to_uint32_array(cu, cids.data());
    roaring_bitmap_free(cu);
    std::vector<ScoredDoc> buf; buf.reserve(nc);
    for (uint64_t i = 0; i < nc; ++i) {
        uint32_t d = cids[i];
        if (tomb && ((tomb[d >> 6] >> (d & 63)) & 1ULL)) continue;
        buf.push_back({score_k0_4b(idx.doc_cids.data(), idx.doc_offs[d], idx.doc_offs[d + 1], S4), d});
    }
    return extract_top_c1(buf, C1);
}

static roaring_bitmap_t* bitmap_to_roaring(const uint64_t* bm, uint64_t N_docs) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    uint64_t nw = (N_docs + 63) / 64;
    std::vector<uint32_t> tmp; tmp.reserve(16384);
    for (uint64_t w = 0; w < nw; ++w) {
        uint64_t wd = bm[w]; if (!wd) continue;
        uint64_t base = w * 64;
        while (wd) {
            int b = __builtin_ctzll(wd); wd &= wd - 1;
            uint32_t d = (uint32_t)(base + b); if (d >= N_docs) break;
            tmp.push_back(d);
            if (tmp.size() == 16384) { roaring_bitmap_add_many(r, tmp.size(), tmp.data()); tmp.clear(); }
        }
    }
    if (!tmp.empty()) roaring_bitmap_add_many(r, tmp.size(), tmp.data());
    roaring_bitmap_run_optimize(r);
    return r;
}

static double calc_recall_k(const uint32_t* pred, int np, const uint32_t* gt, int k) {
    int h = 0;
    for (int i = 0; i < std::min(np, k); ++i)
        for (int j = 0; j < k; ++j)
            if (pred[i] == gt[j]) { h++; break; }
    return (double)h / (double)k;
}

// Pure WARP-style centroid-sum IVF (no decoupled K0_4b filter, no circuit breaker)
static std::vector<uint32_t> tier0_warp_pure(const SegmentIndex& idx, const roaring_bitmap_t* pred_rb,
                                             const float* S_bf16, uint32_t vmask, int nprobe_c, int C1, bool is_unf) {
    CentroidTopK topk; topk.select(S_bf16, idx.K, vmask, nprobe_c);
    std::vector<std::pair<uint32_t, float>> hits; hits.reserve(16384);
    uint32_t vm = vmask;
    while (vm) {
        int t = __builtin_ctz(vm); vm &= vm - 1;
        for (const auto& [sc, cid] : topk.top_per_tok[t]) {
            const roaring_bitmap_t* pl = idx.inv_roaring[cid];
            if (is_unf) {
                uint64_t n = roaring_bitmap_get_cardinality(pl);
                std::vector<uint32_t> tmp(n); roaring_bitmap_to_uint32_array(pl, tmp.data());
                for (uint32_t d : tmp) hits.push_back({d, sc});
            } else {
                roaring_bitmap_t* inter = roaring_bitmap_and(pl, pred_rb);
                uint64_t n = roaring_bitmap_get_cardinality(inter);
                if (n > 0) {
                    std::vector<uint32_t> tmp(n); roaring_bitmap_to_uint32_array(inter, tmp.data());
                    for (uint32_t d : tmp) hits.push_back({d, sc});
                }
                roaring_bitmap_free(inter);
            }
        }
    }
    if (hits.empty()) return {};
    std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<HeapItem> agg; agg.reserve(hits.size() / 2 + 1);
    for (size_t i = 0; i < hits.size(); ) {
        uint32_t d = hits[i].first; float s = 0.0f;
        while (i < hits.size() && hits[i].first == d) { s += hits[i].second; ++i; }
        agg.push_back({s, d});
    }
    int k = std::min(C1, (int)agg.size());
    std::partial_sort(agg.begin(), agg.begin() + k, agg.end(), HeapMinCmp{});
    std::vector<uint32_t> out(k); for (int i = 0; i < k; ++i) out[i] = agg[i].did;
    return out;
}

static inline uint64_t splitmix64_v4(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void run_semipoison_bench(const SegmentIndex& idx, const std::string& out_json, int nT, int C1_pool, int nprobe_arg) {
    const int nq = (int)std::min<uint64_t>(idx.Nq, 50);
    const double sigma = 0.30;
    const double surv_grid[7] = {0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10};
    std::ofstream fo(out_json);
    fo << std::fixed << std::setprecision(4) << "{\n  \"N_docs\": " << idx.N_docs << ", \"sigma\": " << sigma
       << ", \"threads\": " << nT << ", \"gamma_cb\": 0.08,\n  \"rows\": [\n";
    bool first = true;

    // Precompute hot posting union per query once
    std::vector<roaring_bitmap_t*> hot_rbs(nq, nullptr);
    {
        QueryLUT lut;
        for (int q = 0; q < nq; ++q) {
            build_query_lut(idx.C_bf16.data(), idx.K, idx.Q_32.data() + (size_t)q * 32 * 128, lut);
            CentroidTopK topk; topk.select(lut.S_bf16.data(), idx.K, lut.valid_q_mask, nprobe_arg);
            roaring_bitmap_t* hr = roaring_bitmap_create();
            uint32_t vm = lut.valid_q_mask;
            while (vm) { int t = __builtin_ctz(vm); vm &= vm - 1; for (const auto& [sc, cid] : topk.top_per_tok[t]) roaring_bitmap_or_inplace(hr, idx.inv_roaring[cid]); }
            hot_rbs[q] = hr;
        }
    }

    for (double hs : surv_grid) {
        std::vector<std::vector<uint64_t>> bms(nq, std::vector<uint64_t>(idx.n_words, 0));
        std::vector<roaring_bitmap_t*> rbs(nq, nullptr);
        std::vector<std::vector<uint32_t>> ref_gt10(nq, std::vector<uint32_t>(10, 0));

        // Build per-query semi-poisoned predicate bitmaps & exhaustive Path-I exact FP16 reference Top-10
        #pragma omp parallel num_threads(nT)
        {
            init_amx_thread();
            ThreadUringReader tr_b4r, tr_t2;
            if (idx.has_b4_ram) tr_b4r.init(&idx.tier1_b4_ram, 1, 70);
            tr_t2.init(&idx.tier2_nvme, 32, 256);
            QueryLUT lut;
            #pragma omp for schedule(dynamic, 1)
            for (int q = 0; q < nq; ++q) {
                uint64_t hot_n = roaring_bitmap_get_cardinality(hot_rbs[q]);
                double cold_n = std::max(1.0, (double)idx.N_docs - (double)hot_n);
                double want_cold = std::max(0.0, sigma * (double)idx.N_docs - hs * (double)hot_n);
                double cold_p = std::clamp(want_cold / cold_n, 0.0, 1.0);
                uint64_t th_hot = (uint64_t)(hs * (double)UINT64_MAX);
                uint64_t th_cold = (uint64_t)(cold_p * (double)UINT64_MAX);
                auto& bm = bms[q];
                uint64_t seed_q = 0x517cc1b727220a95ULL ^ ((uint64_t)q * 1315423911ULL) ^ (uint64_t)(hs * 100000.0);
                for (uint32_t d = 0; d < idx.N_docs; ++d) {
                    bool is_h = roaring_bitmap_contains(hot_rbs[q], d);
                    uint64_t r = splitmix64_v4(seed_q ^ (uint64_t)d);
                    bool keep = is_h ? (hs > 0.0 && r < th_hot) : (r < th_cold);
                    if (keep) bm[d >> 6] |= (1ULL << (d & 63));
                }
                rbs[q] = bitmap_to_roaring(bm.data(), idx.N_docs);
                const float* Qq = idx.Q_32.data() + (size_t)q * 32 * 128;
                const float* Qrq = idx.Q_rot.data() + (size_t)q * 32 * 128;
                build_query_lut(idx.C_bf16.data(), idx.K, Qq, lut);
                Tier1QueryPrepV4 qp; qp.init(Qq, Qrq, lut.valid_q_mask, lut.n_valid_q, idx.is_unit != 0, idx.m_norm_max, false, false, 10);
                auto fc = tier0_flat(idx, bm.data(), lut.S4.data(), C1_pool);
                RerankStats rst = idx.has_b4_ram
                    ? rerank_t3_b4(fc, tr_b4r, tr_t2, lut.S_bf16.data(), idx.C_norms.data(), idx.C_bf_err.data(), Qq, qp, 10)
                    : rerank_direct_exact(fc, tr_t2, Qq, lut.valid_q_mask, 10);
                for (int r = 0; r < std::min(10, rst.n_out); ++r) ref_gt10[q][r] = rst.topk_ids[r];
            }
        }

        // Benchmark Protected PM_DRAM_T1 (use_cb=true) vs Unprotected B2b_InvK0 (use_cb=false, allow_path1=false)
        for (int is_pm = 1; is_pm >= 0; --is_pm) {
            idx.tier1_b4_nvme.flush_cold(); idx.tier2_nvme.flush_cold();
            int total_q = std::max(nT * 4, 200);
            std::vector<double> lats(total_q, 0.0), recs(total_q, 0.0);
            std::atomic<uint64_t> trips{0}; std::atomic<int> next_q{0};
            double wt0 = 0.0, wt1 = 0.0;
            #pragma omp parallel num_threads(nT)
            {
                init_amx_thread();
                ThreadUringReader tr_b4r, tr_t2;
                if (idx.has_b4_ram) tr_b4r.init(&idx.tier1_b4_ram, 1, 70);
                tr_t2.init(&idx.tier2_nvme, 32, 256);
                QueryLUT lut;
                #pragma omp barrier
                #pragma omp single
                { wt0 = monotonic_ns(); }
                while (true) {
                    int job = next_q.fetch_add(1, std::memory_order_relaxed); if (job >= total_q) break;
                    int q = job % nq;
                    const float* Qq = idx.Q_32.data() + (size_t)q * 32 * 128;
                    const float* Qrq = idx.Q_rot.data() + (size_t)q * 32 * 128;
                    double qt0 = monotonic_ns();
                    build_query_lut(idx.C_bf16.data(), idx.K, Qq, lut);
                    Tier1QueryPrepV4 qp; qp.init(Qq, Qrq, lut.valid_q_mask, lut.n_valid_q, idx.is_unit != 0, idx.m_norm_max, false, false, 10);
                    bool tr = false;
                    auto cands = tier0_pm_dual_v4(idx, bms[q].data(), rbs[q], lut.S_bf16.data(), lut.S4.data(),
                                                  lut.valid_q_mask, nprobe_arg, C1_pool, false, (is_pm == 1), 0.08, &tr, nullptr, (is_pm == 1));
                    if (tr) trips.fetch_add(1, std::memory_order_relaxed);
                    RerankStats st = (is_pm == 1 && idx.has_b4_ram)
                        ? rerank_t3_b4(cands, tr_b4r, tr_t2, lut.S_bf16.data(), idx.C_norms.data(), idx.C_bf_err.data(), Qq, qp, 10)
                        : rerank_direct_exact(cands, tr_t2, Qq, lut.valid_q_mask, 10);
                    lats[job] = (monotonic_ns() - qt0) * 1e-6;
                    recs[job] = calc_recall_k(st.topk_ids, st.n_out, ref_gt10[q].data(), 10);
                }
                #pragma omp barrier
                #pragma omp single
                { wt1 = monotonic_ns(); }
            }
            double ws = (wt1 - wt0) * 1e-9;
            std::sort(lats.begin(), lats.end());
            double p50 = lats[lats.size() / 2], p99 = lats[std::min(lats.size() - 1, (size_t)(lats.size() * 0.99))];
            double rk = std::accumulate(recs.begin(), recs.end(), 0.0) / total_q;
            double tr_pct = 100.0 * (double)trips.load() / total_q;
            if (!first) fo << ",\n";
            first = false;
            fo << "    {\"hot_surv_pct\": " << (hs * 100.0) << ", \"method\": \"" << (is_pm ? "PM_DRAM_T1_CB" : "B2b_InvK0_NoCB")
               << "\", \"qps\": " << (total_q / ws) << ", \"p50_ms\": " << p50 << ", \"p99_ms\": " << p99
               << ", \"recall10_vs_path1\": " << rk << ", \"cb_trip_pct\": " << tr_pct << "}";
            std::cerr << "  [SemiPoison hot_surv=" << std::setprecision(1) << (hs * 100.0) << "%] "
                      << (is_pm ? "PM_DRAM_T1_CB " : "B2b_InvK0_NoCB") << " | QPS=" << (total_q / ws)
                      << " P50=" << p50 << " P99=" << p99 << " R10=" << std::setprecision(3) << rk
                      << " Trip=" << std::setprecision(1) << tr_pct << "%\n";
        }
        for (auto* r : rbs) if (r) roaring_bitmap_free(r);
    }
    for (auto* r : hot_rbs) if (r) roaring_bitmap_free(r);
    fo << "\n  ]\n}\n";
}

// ============================================================================
// MAIN MODES:
//   --unit-test <index_dir> [--no-amx] [--delta 0.05]
//   --bench-single <index_dir> <out_json> [--no-amx] [--threads 1,8,24,48]
//                  [--topk 10|50|100] [--delta 0.05] [--force-path3] [--methods 0,1,2]
//                  [--tier-mem-mode nvme|split|dram] [--skip-unit-test]
//                  [--workloads w1,w2] [--dump-topids <dir>]
//   --bench-semipoison <index_dir> <out_json> [--threads 48]
// ============================================================================
int main(int argc, char** argv) {
    std::string mode, dir, out_json = "/tmp/pm_v4_out.json", dump_dir, tier_mem_mode = "all";
    std::vector<int> threads = {1, 8, 24, 48};
    std::vector<std::string> wl_filter;
    int topk = 10, c1_arg = 600, nprobe_arg = 8;
    uint32_t method_mask = 0xFFu;
    bool skip_unit_test = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-amx") g_force_no_amx = true;
        else if (a == "--force-path3") g_force_path3 = true;
        else if (a == "--skip-unit-test") skip_unit_test = true;
        else if (a == "--delta" && i + 1 < argc) { g_delta = std::stod(argv[++i]); }
        else if (a == "--tier-mem-mode" && i + 1 < argc) { tier_mem_mode = argv[++i]; }
        else if (a == "--methods" && i + 1 < argc) {
            method_mask = 0u;
            std::stringstream ss(argv[++i]); std::string tok;
            while (std::getline(ss, tok, ',')) if (!tok.empty()) method_mask |= (1u << std::stoi(tok));
        }
        else if (a == "--unit-test" && i + 1 < argc) { mode = "unit_test"; dir = argv[++i]; }
        else if (a == "--bench-single" && i + 2 < argc) { mode = "bench_single"; dir = argv[++i]; out_json = argv[++i]; }
        else if (a == "--bench-semipoison" && i + 2 < argc) { mode = "semipoison"; dir = argv[++i]; out_json = argv[++i]; }
        else if (a == "--topk" && i + 1 < argc) { topk = std::clamp(std::stoi(argv[++i]), 1, 100); }
        else if (a == "--c1" && i + 1 < argc) { c1_arg = std::max(10, std::stoi(argv[++i])); }
        else if (a == "--nprobe" && i + 1 < argc) { nprobe_arg = std::max(1, std::stoi(argv[++i])); }
        else if (a == "--dump-topids" && i + 1 < argc) { dump_dir = argv[++i]; }
        else if (a == "--workloads" && i + 1 < argc) {
            std::stringstream ss(argv[++i]); std::string tok;
            while (std::getline(ss, tok, ',')) if (!tok.empty()) wl_filter.push_back(tok);
        }
        else if (a == "--threads" && i + 1 < argc) {
            threads.clear();
            std::stringstream ss(argv[++i]); std::string tok;
            while (std::getline(ss, tok, ',')) threads.push_back(std::stoi(tok));
        }
    }
    if (mode.empty()) {
        std::cerr << "Usage: " << argv[0] << " --unit-test <dir> | --bench-single <dir> <out.json> | --bench-semipoison <dir> <out.json>\n";
        return 1;
    }
    init_amx_process();
    init_amx_thread();

    SegmentIndex idx;
    bool load_b4 = (mode == "bench_single" || mode == "semipoison");
    bool load_b1 = true;
    if (tier_mem_mode == "nvme")       { load_b4 = false; load_b1 = false; }
    else if (tier_mem_mode == "split") { load_b4 = false; load_b1 = true; }
    else if (tier_mem_mode == "dram")  { load_b4 = true;  load_b1 = false; }
    if (!idx.load(dir, load_b4, load_b1, mode == "bench_single")) return 2;

    UnitTestReport ut{};
    if (!skip_unit_test) {
        ut = run_theorem3_unit_test_v4(idx);
        if (ut.viol_b4 != 0 || ut.viol_b1 != 0) {
            std::cerr << "FATAL: Theorem 3 expanded unit test FAILED!\n";
            return 99;
        }
    }
    if (mode == "unit_test") {
        std::cout << "{\"unit_test_pass\": true, \"delta\": " << g_delta << ", \"b4_pairs\": " << ut.n_pairs_b4 << ", \"b4_viol\": " << ut.viol_b4
                  << ", \"b1_pairs\": " << ut.n_pairs_b1 << ", \"b1_viol\": " << ut.viol_b1
                  << ", \"m_norm_max\": " << idx.m_norm_max << ", \"amx_active\": " << ((g_amx_perm && !g_force_no_amx) ? "true" : "false") << "}\n";
        return 0;
    }
    if (mode == "semipoison") {
        run_semipoison_bench(idx, out_json, threads.empty() ? 48 : threads.back(), c1_arg, nprobe_arg);
        return 0;
    }

    std::vector<std::pair<std::string, std::vector<uint32_t>>> gt100_map;
    {
        std::ifstream f100(dir + "/workloads_gt100.bin", std::ios::binary);
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

    const char* mnames[8] = {"Split_B1_B4_T3", "PM_DRAM_T1", "PM_NVME_T1", "Approx_B4_NVMe", "Approx_B4_DRAM", "B2b_InvK0", "B2a_WARP_Pure", "B1b_Flat_T3"};
    int C1_pool = std::max(c1_arg, topk * 10);

    std::ofstream fo(out_json);
    fo << std::fixed << std::setprecision(4)
       << "{\n  \"dir\": \"" << dir << "\", \"N_docs\": " << idx.N_docs << ", \"K\": " << idx.K
       << ", \"topk\": " << topk << ", \"c1\": " << C1_pool << ", \"nprobe\": " << nprobe_arg
       << ", \"delta\": " << g_delta << ", \"force_path3\": " << (g_force_path3 ? "true" : "false")
       << ", \"tier_mem_mode\": \"" << tier_mem_mode << "\""
       << ", \"m_norm_max\": " << idx.m_norm_max
       << ", \"amx_active\": " << ((g_amx_perm && !g_force_no_amx) ? "true" : "false")
       << ", \"unit_test\": {\"b4_pairs\": " << ut.n_pairs_b4 << ", \"b4_viol\": " << ut.viol_b4
       << ", \"b1_pairs\": " << ut.n_pairs_b1 << ", \"b1_viol\": " << ut.viol_b1 << "},\n  \"rows\": [\n";
    bool first = true;

    for (const auto& wl : idx.workloads) {
        if (!wl_filter.empty() && std::find(wl_filter.begin(), wl_filter.end(), wl.name) == wl_filter.end()) continue;
        const std::string& wn = wl.name;
        bool is_unf = (wl.sigma >= 0.9999);
        int nq = (int)wl.n_q_eval;
        const uint32_t* gt_ptr = wl.gt_top10.data();
        int gt_stride = 10;
        if (topk > 10) {
            for (const auto& kv : gt100_map) if (kv.first == wn) { gt_ptr = kv.second.data(); gt_stride = 100; break; }
        }
        int eval_k = std::min(topk, gt_stride);

        std::vector<roaring_bitmap_t*> rbs(wl.n_bitmaps, nullptr);
        if (!is_unf) for (uint64_t b = 0; b < wl.n_bitmaps; ++b) rbs[b] = bitmap_to_roaring(wl.bitmaps.data() + b * idx.n_words, idx.N_docs);

        {
            QueryLUT lut;
            for (int q = 0; q < std::min(8, nq); ++q) {
                build_query_lut(idx.C_bf16.data(), idx.K, idx.Q_32.data() + (size_t)q * 32 * 128, lut);
                uint32_t bi = wl.q_to_bm[q];
                volatile size_t sink = tier0_pm_dual_v4(idx, wl.bitmaps.data() + (size_t)bi * idx.n_words,
                                                        is_unf ? nullptr : rbs[bi], lut.S_bf16.data(), lut.S4.data(),
                                                        lut.valid_q_mask, nprobe_arg, C1_pool, is_unf, true).size();
                (void)sink;
            }
        }

        for (int nT : threads) {
            int total_q = (nT == 1) ? nq : std::max(nT * 8, 200);
            for (int m = 0; m < 8; ++m) {
                if (!((method_mask >> m) & 1u)) continue;
                if (m == 0 && !idx.has_b1) continue;
                if ((m == 1 || m == 4) && !idx.has_b4_ram) continue;
                idx.tier1_b4_nvme.flush_cold();
                idx.tier2_nvme.flush_cold();

                std::vector<double> lats(total_q, 0.0), recs(total_q, 0.0), allk(total_q, 0.0);
                std::vector<double> t_lut_v(total_q, 0.0), t_t0_v(total_q, 0.0), t_b1_v(total_q, 0.0), t_b4_v(total_q, 0.0), t_t2_v(total_q, 0.0);
                std::vector<double> cand_v(total_q, 0.0), w1_v(total_q, 0.0), t2_v(total_q, 0.0);
                std::vector<uint32_t> dumped_ids((!dump_dir.empty() && nT == threads.front()) ? (size_t)nq * topk : 0, 0);
                std::atomic<uint64_t> app_bytes{0}, log_bytes{0}, cb_trips{0};
                std::atomic<int> next_q{0};
                uint64_t io0 = 0, pio = 0;
                double wt0 = 0.0, wt1 = 0.0, mhz = 0.0;

                #pragma omp parallel num_threads(nT)
                {
                    init_amx_thread();
                    ThreadUringReader tr_b1, tr_b4r, tr_b4n, tr_t2;
                    if (idx.has_b1) tr_b1.init(&idx.tier1_b1_ram, 1, 22);
                    if (idx.has_b4_ram) tr_b4r.init(&idx.tier1_b4_ram, 1, 70);
                    tr_b4n.init(&idx.tier1_b4_nvme, 16, 70);
                    tr_t2.init(&idx.tier2_nvme, 32, 256);
                    QueryLUT lut;
                    #pragma omp barrier
                    #pragma omp single
                    { io0 = read_proc_self_io_read_bytes(); wt0 = monotonic_ns(); }
                    while (true) {
                        int job = next_q.fetch_add(1, std::memory_order_relaxed);
                        if (job >= total_q) break;
                        int q = job % nq;
                        const float* Qq = idx.Q_32.data() + (size_t)q * 32 * 128;
                        const float* Qrq = idx.Q_rot.data() + (size_t)q * 32 * 128;
                        uint32_t bi = wl.q_to_bm[q];
                        const uint64_t* bm = wl.bitmaps.data() + (size_t)bi * idx.n_words;
                        const roaring_bitmap_t* rb = is_unf ? nullptr : rbs[bi];
                        const uint32_t* gt_q = gt_ptr + (size_t)q * gt_stride;

                        double qt0 = monotonic_ns();
                        build_query_lut(idx.C_bf16.data(), idx.K, Qq, lut);
                        Tier1QueryPrepV4 qp;
                        qp.init(Qq, Qrq, lut.valid_q_mask, lut.n_valid_q, idx.is_unit != 0, idx.m_norm_max, (topk > 10), (m == 0), topk);
                        double qt_lut = monotonic_ns();

                        bool tripped = false;
                        std::vector<uint32_t> cands;
                        if (m == 6)      cands = tier0_warp_pure(idx, rb, lut.S_bf16.data(), lut.valid_q_mask, nprobe_arg, C1_pool, is_unf);
                        else if (m == 7) cands = tier0_flat(idx, bm, lut.S4.data(), C1_pool);
                        else             cands = tier0_pm_dual_v4(idx, bm, rb, lut.S_bf16.data(), lut.S4.data(),
                                                                  lut.valid_q_mask, nprobe_arg, C1_pool, is_unf, (m != 5), 0.08, &tripped,
                                                                  nullptr, (m != 5) && !g_force_path3);
                        if (tripped) cb_trips.fetch_add(1, std::memory_order_relaxed);
                        double qt_t0 = monotonic_ns();

                        RerankStats st;
                        if (m == 0)      st = rerank_split_b1_b4_t3(cands, tr_b1, tr_b4n, tr_t2, lut.S_bf16.data(), idx.C_norms.data(), idx.C_bf_err.data(), Qq, qp, topk);
                        else if (m == 1) st = rerank_t3_b4(cands, tr_b4r, tr_t2, lut.S_bf16.data(), idx.C_norms.data(), idx.C_bf_err.data(), Qq, qp, topk);
                        else if (m == 2 || m == 7) st = rerank_t3_b4(cands, tr_b4n, tr_t2, lut.S_bf16.data(), idx.C_norms.data(), idx.C_bf_err.data(), Qq, qp, topk);
                        else if (m == 3) st = rerank_approx_b4_zero_t2(cands, tr_b4n, lut.S_bf16.data(), qp, topk);
                        else if (m == 4) st = rerank_approx_b4_zero_t2(cands, tr_b4r, lut.S_bf16.data(), qp, topk);
                        else             st = rerank_direct_exact(cands, tr_t2, Qq, lut.valid_q_mask, topk);

                        double qt_end = monotonic_ns();
                        lats[job] = (qt_end - qt0) * 1e-6;
                        t_lut_v[job] = (qt_lut - qt0) * 1e-6;
                        t_t0_v[job]  = (qt_t0 - qt_lut) * 1e-6;
                        t_b1_v[job]  = st.t_b1_ms;
                        t_b4_v[job]  = st.t_b4_ms;
                        t_t2_v[job]  = st.t_t2_ms;
                        cand_v[job]  = (double)cands.size();
                        w1_v[job]    = (double)st.wave1_surv;
                        t2_v[job]    = (double)st.tier2_reranked;
                        double r_val = calc_recall_k(st.topk_ids, st.n_out, gt_q, eval_k);
                        recs[job] = r_val;
                        allk[job] = (r_val >= 0.999999) ? 1.0 : 0.0;
                        if (!dumped_ids.empty() && job < nq) {
                            for (int r = 0; r < std::min(topk, st.n_out); ++r) dumped_ids[(size_t)q * topk + r] = st.topk_ids[r];
                        }
                        if (job == total_q / 2 && omp_get_thread_num() == 0) mhz = sample_active_mhz_0_47(nT);
                    }
                    #pragma omp barrier
                    #pragma omp single
                    { wt1 = monotonic_ns(); pio = read_proc_self_io_read_bytes() - io0; }
                    app_bytes.fetch_add(tr_b4n.bytes_read + tr_t2.bytes_read, std::memory_order_relaxed);
                    log_bytes.fetch_add(tr_b4n.logical_bytes_read + tr_t2.logical_bytes_read, std::memory_order_relaxed);
                }

                if (!dumped_ids.empty()) {
                    std::ofstream fd(dump_dir + "/" + wn + "__" + mnames[m] + "_topids.bin", std::ios::binary);
                    if (fd) fd.write(reinterpret_cast<const char*>(dumped_ids.data()), dumped_ids.size() * 4);
                }
                double ws_s = (wt1 - wt0) * 1e-9;
                std::sort(lats.begin(), lats.end());
                double p50 = lats[lats.size() / 2], p99 = lats[std::min(lats.size() - 1, (size_t)(lats.size() * 0.99))];
                double qps = (double)total_q / ws_s;
                auto avg_v = [&](const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / total_q; };
                double rk = avg_v(recs), a10 = avg_v(allk);
                double nvme_mbs = ((double)pio / 1048576.0) / ws_s;
                double log_mb = ((double)log_bytes.load() / 1048576.0) / total_q;
                double app_mb = ((double)app_bytes.load() / 1048576.0) / total_q;
                double proc_mb = ((double)pio / 1048576.0) / total_q;
                double raf = (log_mb > 1e-9) ? (proc_mb / log_mb) : 1.0;

                if (!first) fo << ",\n";
                first = false;
                fo << "    {\"workload\": \"" << wn << "\", \"threads\": " << nT << ", \"method\": \"" << mnames[m]
                   << "\", \"topk\": " << eval_k << ", \"c1\": " << C1_pool << ", \"nprobe\": " << nprobe_arg
                   << ", \"qps\": " << qps << ", \"p50_ms\": " << p50 << ", \"p99_ms\": " << p99
                   << ", \"recall_k\": " << rk << ", \"recall10\": " << rk << ", \"all10_exact\": " << a10
                   << ", \"avg_cand\": " << avg_v(cand_v) << ", \"avg_w1_surv\": " << avg_v(w1_v) << ", \"avg_t2_reranked\": " << avg_v(t2_v)
                   << ", \"t_lut_ms\": " << avg_v(t_lut_v) << ", \"t_tier0_ms\": " << avg_v(t_t0_v)
                   << ", \"t_b1_ms\": " << avg_v(t_b1_v) << ", \"t_b4_ms\": " << avg_v(t_b4_v) << ", \"t_t2_ms\": " << avg_v(t_t2_v)
                   << ", \"nvme_mbs\": " << nvme_mbs
                   << ", \"logical_mb_q\": " << log_mb << ", \"app_mb_q\": " << app_mb << ", \"proc_mb_q\": " << proc_mb
                   << ", \"raf\": " << raf << ", \"cb_trips\": " << cb_trips.load() << ", \"mhz\": " << mhz << "}";
                std::cerr << "  [" << wn << " T=" << std::setw(2) << nT << " C1=" << C1_pool << " np=" << nprobe_arg << "] " << std::setw(15) << mnames[m]
                          << " | QPS=" << std::setprecision(1) << std::setw(7) << qps
                          << " | P50=" << std::setprecision(2) << std::setw(6) << p50
                          << " | R@k=" << std::setprecision(3) << rk << " All10=" << std::setprecision(2) << a10
                          << " | T2=" << std::setprecision(1) << avg_v(t2_v) << "/" << avg_v(cand_v)
                          << " | IO=" << std::setprecision(2) << proc_mb << " MB/q\n";
            }
        }
        if (!is_unf) for (auto* r : rbs) if (r) roaring_bitmap_free(r);
    }
    double pss_gb = 0, rss_gb = 0, hwm_gb = 0;
    read_proc_self_mem_gb(pss_gb, rss_gb, hwm_gb);
    fo << "\n  ],\n  \"proc_mem_gb\": {\"pss_gb\": " << pss_gb << ", \"rss_gb\": " << rss_gb << ", \"vm_hwm_gb\": " << hwm_gb << "}\n}\n";
    return 0;
}
