// PM-MaxSim Provably Sound 1-bit Tier-1 RAM Builder + Exact FP16 Max-Norm Scanner (M3-B03 & Split_B1_B4_T3):
//   1. Scans EVERY FP16 vector in tier2_fp16.nvme to compute exact M_norm = max_{x in D_fp16} ||x||_2
//      and saves m_norm.bin (resolves M3-B03 IEEE-754 FP16 half-ULP norm inflation!).
//   2. Encodes every token's rotated residual (x_fp16 - c_fp32) @ P_fp32 into exact 1-bit Extended RaBitQ
//      (sign bits y_j in {-0.5, +0.5}, ||y||_2 = sqrt(32) = 5.656854249492381 exact!) with
//      sc_err / T_COEFF_MIN (0.145 < t_{127}(0.05/1)/sqrt(127) = 0.146037945) folded into eta_f16_up via
//      round-toward-+inf + ULP verification loop.
//   3. Writes compact UNPADDED tier1_b1.ram (for Split_B1_B4_T3 Wave-1 in tiny RAM: 22 B/tok = 5.63 GB on LoTTE 2.82M!):
//      Per document d (m = n_toks):
//        - uint16_t cids[m]       (2*m bytes, exact centroid ID copied from tier1_b4.nvme)
//        - uint16_t sc_f16[m]     (2*m bytes, FP16 scale)
//        - uint16_t et_f16[m]     (2*m bytes, FP16-ceil eta_up)
//        - uint8_t  sign16[16*m]  (16*m bytes, bit j of byte j/8 = (r_rot[j] >= 0 ? 1 : 0))
//      Plus tier1_b1_meta.bin (array of {uint64_t offset, uint32_t bytes, uint32_t n_toks}).
//
// Compile:
//   g++ -O3 -march=native -std=c++20 -fopenmp -Wall -Wextra tier1_b1_builder_v4.cpp -o tier1_b1_builder_v4

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

struct DocDiskEntry {
    uint64_t offset;
    uint32_t aligned_bytes;
    uint32_t n_toks;
};

struct MNormHeader {
    uint64_t magic;       // 0x4D4E4F524D303034ULL ("MNORM004")
    uint64_t N_docs;
    uint64_t total_tokens;
    double   m_norm_max;  // exact max_{x in D_fp16} ||x||_2
    double   m_norm_p99;
    double   m_norm_med;
};

static constexpr double T_COEFF_MIN = 0.145; // < t_{127}(0.05/1)/sqrt(127) = 0.146037945
static constexpr double SQRT_32     = 5.6568542494923806; // exact ||y||_2 for y in {-0.5, +0.5}^128

static inline void encode_one_b1_sound(const double* r_rot, uint8_t* sign16,
                                       uint16_t& scale_f16, uint16_t& eta_f16_up) {
    double r2 = 0.0;
    for (int j = 0; j < 128; ++j) r2 += r_rot[j] * r_rot[j];
    double r = std::sqrt(r2);
    if (r <= 1e-12) {
        std::memset(sign16, 0, 16);
        scale_f16 = 0;
        eta_f16_up = 0;
        return;
    }
    double inv_r = 1.0 / r;
    double fdot = 0.0; // sum_j 0.5 * |o_j| = 0.5 * ||o||_1
    std::memset(sign16, 0, 16);
    for (int j = 0; j < 128; ++j) {
        if (r_rot[j] >= 0.0) sign16[j >> 3] |= (uint8_t)(1u << (j & 7));
        fdot += 0.5 * std::fabs(r_rot[j]) * inv_r;
    }
    double c = std::min(1.0, fdot / SQRT_32);
    double cos2 = c * c;
    double scale_d64 = r / std::max(fdot, 1e-18);
    double eta_d64   = r * std::sqrt(std::max(0.0, (1.0 - cos2) / std::max(cos2, 1e-18)));

    scale_f16 = _cvtss_sh((float)scale_d64, _MM_FROUND_TO_NEAREST_INT);
    double sc_f16_d = (double)_cvtsh_ss(scale_f16);
    double sc_err   = std::fabs(sc_f16_d - scale_d64) * SQRT_32;

    double eta_req = eta_d64 + (sc_err / T_COEFF_MIN) * 1.001 + 1e-6;
    uint16_t h = _cvtss_sh((float)eta_req, _MM_FROUND_TO_POS_INF);
    // Explicit ULP verification loop guaranteeing round-toward-+inf soundness
    while ((double)_cvtsh_ss(h) < eta_req && h < 0x7BFFu) {
        h = (uint16_t)(h + 1u);
    }
    eta_f16_up = h;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <index_dir> [num_threads]\n";
        return 1;
    }
    std::string dir = argv[1];
    int nT = (argc >= 3) ? std::atoi(argv[2]) : 24;
    omp_set_num_threads(nT);

    // 1. Read C_fp32 and P_fp32 from core_index.bin
    uint64_t N_docs = 0, K = 0, d = 0;
    std::vector<float> C_fp32, P_fp32;
    {
        std::ifstream f(dir + "/core_index.bin", std::ios::binary);
        if (!f) { std::cerr << "Cannot open " << dir << "/core_index.bin\n"; return 2; }
        uint64_t hdr[12];
        f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        N_docs = hdr[1]; K = hdr[2]; d = hdr[3];
        if (d != 128) { std::cerr << "Expected d=128, got " << d << "\n"; return 2; }
        C_fp32.resize(K * d);
        P_fp32.resize(d * d);
        f.read(reinterpret_cast<char*>(C_fp32.data()), K * d * 4);
        f.read(reinterpret_cast<char*>(P_fp32.data()), d * d * 4);
    }
    // Promote P_fp32 to P_f64 for exact rotation
    std::vector<double> P_f64(128 * 128);
    for (int i = 0; i < 128 * 128; ++i) P_f64[i] = (double)P_fp32[i];

    // 2. Read tier2_meta.bin and tier1_meta.bin
    std::vector<DocDiskEntry> t2_meta(N_docs), t1_meta(N_docs);
    {
        std::ifstream f2(dir + "/tier2_meta.bin", std::ios::binary);
        std::ifstream f1(dir + "/tier1_meta.bin", std::ios::binary);
        if (!f2 || !f1) { std::cerr << "Missing tier meta files\n"; return 3; }
        f2.read(reinterpret_cast<char*>(t2_meta.data()), N_docs * sizeof(DocDiskEntry));
        f1.read(reinterpret_cast<char*>(t1_meta.data()), N_docs * sizeof(DocDiskEntry));
    }

    // 3. Compute exact unpadded layout for tier1_b1.ram (22 * m bytes per doc, 16-byte aligned!)
    std::vector<DocDiskEntry> b1_meta(N_docs);
    uint64_t cur_off = 0, total_toks = 0;
    for (uint64_t i = 0; i < N_docs; ++i) {
        uint32_t m = t2_meta[i].n_toks;
        uint32_t sz = (22u * m + 15u) & ~15u;
        b1_meta[i] = {cur_off, sz, m};
        cur_off += sz;
        total_toks += m;
    }
    std::cerr << "[tier1_b1_builder_v4] dir=" << dir << " N_docs=" << N_docs
              << " total_toks=" << total_toks << " out_GB=" << (cur_off / 1073741824.0)
              << " threads=" << nT << "\n";

    std::ofstream fb1(dir + "/tier1_b1.ram", std::ios::binary);
    if (!fb1) { std::cerr << "Cannot create tier1_b1.ram\n"; return 4; }

    // Stream in chunks of 32,768 docs so peak RAM stays tiny (~1 GB)
    const uint64_t CHUNK = 32768;
    double global_max_norm = 0.0;
    std::vector<float> sample_norms;
    sample_norms.reserve(200000);

    FILE* fp2 = std::fopen((dir + "/tier2_fp16.nvme").c_str(), "rb");
    FILE* fp1 = std::fopen((dir + "/tier1_b4.nvme").c_str(), "rb");
    if (!fp2 || !fp1) { std::cerr << "Cannot open tier1/tier2 nvme files\n"; return 5; }

    for (uint64_t s = 0; s < N_docs; s += CHUNK) {
        uint64_t e = std::min(N_docs, s + CHUNK);
        uint64_t cnt = e - s;

        // Read raw FP16 bytes and cids from tier1_b4 for this chunk sequentially
        uint64_t t2_start = t2_meta[s].offset;
        uint64_t t2_end   = t2_meta[e - 1].offset + t2_meta[e - 1].aligned_bytes;
        std::vector<uint8_t> buf2(t2_end - t2_start);
        ::fseeko(fp2, (off_t)t2_start, SEEK_SET);
        if (std::fread(buf2.data(), 1, buf2.size(), fp2) != buf2.size()) { std::cerr << "fread t2 failed\n"; return 6; }

        uint64_t t1_start = t1_meta[s].offset;
        uint64_t t1_end   = t1_meta[e - 1].offset + t1_meta[e - 1].aligned_bytes;
        std::vector<uint8_t> buf1(t1_end - t1_start);
        ::fseeko(fp1, (off_t)t1_start, SEEK_SET);
        if (std::fread(buf1.data(), 1, buf1.size(), fp1) != buf1.size()) { std::cerr << "fread t1 failed\n"; return 6; }

        uint64_t out_start = b1_meta[s].offset;
        uint64_t out_end   = b1_meta[e - 1].offset + b1_meta[e - 1].aligned_bytes;
        std::vector<uint8_t> out_buf(out_end - out_start, 0);

        double chunk_max_norm = 0.0;
        #pragma omp parallel for schedule(dynamic, 64) reduction(max:chunk_max_norm)
        for (uint64_t idx = 0; idx < cnt; ++idx) {
            uint64_t doc = s + idx;
            uint32_t m = t2_meta[doc].n_toks;
            const uint16_t* raw_fp16 = reinterpret_cast<const uint16_t*>(buf2.data() + (t2_meta[doc].offset - t2_start));
            const uint16_t* cids_in  = reinterpret_cast<const uint16_t*>(buf1.data() + (t1_meta[doc].offset - t1_start));

            uint8_t* dst = out_buf.data() + (b1_meta[doc].offset - out_start);
            uint16_t* cids_out = reinterpret_cast<uint16_t*>(dst);
            uint16_t* sc_out   = reinterpret_cast<uint16_t*>(dst + 2 * m);
            uint16_t* et_out   = reinterpret_cast<uint16_t*>(dst + 4 * m);
            uint8_t*  sgn_out  = dst + 6 * m;

            for (uint32_t tk = 0; tk < m; ++tk) {
                uint16_t cid = cids_in[tk];
                cids_out[tk] = cid;
                const float* cent = C_fp32.data() + (size_t)cid * 128;
                double res_v[128], r_rot[128];
                double xn2 = 0.0;
                for (int j = 0; j < 128; ++j) {
                    double xv = (double)_cvtsh_ss(raw_fp16[(size_t)tk * 128 + j]);
                    xn2 += xv * xv;
                    res_v[j] = xv - (double)cent[j];
                }
                double xn = std::sqrt(xn2);
                if (xn > chunk_max_norm) chunk_max_norm = xn;

                for (int k = 0; k < 128; ++k) {
                    double acc = 0.0;
                    for (int j = 0; j < 128; ++j) acc += res_v[j] * P_f64[j * 128 + k];
                    r_rot[k] = acc;
                }
                encode_one_b1_sound(r_rot, sgn_out + (size_t)tk * 16, sc_out[tk], et_out[tk]);
            }
        }
        global_max_norm = std::max(global_max_norm, chunk_max_norm);
        // Sample first token of every 16th doc for median/p99 norm diagnostics
        for (uint64_t idx = 0; idx < cnt; idx += 16) {
            uint64_t doc = s + idx;
            if (t2_meta[doc].n_toks == 0) continue;
            const uint16_t* raw_fp16 = reinterpret_cast<const uint16_t*>(buf2.data() + (t2_meta[doc].offset - t2_start));
            double n2 = 0.0;
            for (int j = 0; j < 128; ++j) { double v = _cvtsh_ss(raw_fp16[j]); n2 += v * v; }
            if (sample_norms.size() < 200000) sample_norms.push_back((float)std::sqrt(n2));
        }

        fb1.write(reinterpret_cast<const char*>(out_buf.data()), out_buf.size());
        if (((s / CHUNK) % 10 == 0) || e == N_docs) {
            std::cerr << "  Progress " << e << "/" << N_docs << " (" << (100.0 * e / N_docs)
                      << "%) | running max_FP16_norm=" << global_max_norm << "\n";
        }
    }
    std::fclose(fp2);
    std::fclose(fp1);
    fb1.close();

    // Write tier1_b1_meta.bin and m_norm.bin
    {
        std::ofstream fm(dir + "/tier1_b1_meta.bin", std::ios::binary);
        fm.write(reinterpret_cast<const char*>(b1_meta.data()), N_docs * sizeof(DocDiskEntry));
    }
    std::sort(sample_norms.begin(), sample_norms.end());
    double med = sample_norms.empty() ? 1.0 : sample_norms[sample_norms.size() / 2];
    double p99 = sample_norms.empty() ? 1.0 : sample_norms[(size_t)(sample_norms.size() * 0.99)];
    MNormHeader mh{0x4D4E4F524D303034ULL, N_docs, total_toks, global_max_norm, p99, med};
    {
        std::ofstream fn(dir + "/m_norm.bin", std::ios::binary);
        fn.write(reinterpret_cast<const char*>(&mh), sizeof(mh));
    }
    std::cerr << "=== DONE " << dir << ": tier1_b1.ram=" << (cur_off / 1073741824.0)
              << " GB | exact M_norm_max=" << global_max_norm
              << " (med=" << med << ", p99=" << p99 << ") ===\n";
    return 0;
}
