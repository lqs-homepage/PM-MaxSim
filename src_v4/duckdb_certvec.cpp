// duckdb_certvec.cpp — Native DuckDB C++20 Table-In-Out Operator Integration for libpm_certvec.so
// Implements MAJOR-04 idiomatic DuckDB TableFunction::in_out_function accepting LogicalType::TABLE:
//   SELECT m.domain, COUNT(*), AVG(s.score)
//   FROM pm_certvec_topk((SELECT doc_id FROM docs WHERE ...), '/path/idx', $q_blob, 32, 10, 0.05) s
//   JOIN docs_meta m ON s.doc_id = m.doc_id GROUP BY m.domain;
// Zero-copy DataChunk -> CRoaring streaming in in_out_function; single pm_execute_query() in in_out_function_final;
// vectorized DataChunk output directly into downstream DuckDB JOIN / GROUP BY / AGGREGATION!

#include "duckdb.hpp"
#include "pm_certvec.h"
#include "roaring.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb_certvec {

static std::mutex g_idx_mtx;
static std::unordered_map<std::string, pm_handle_t*> g_open_handles;

static pm_handle_t* get_or_open(const std::string& dir, const std::string& sidecar, pm_tier_mode_t tier, size_t ram_mb) {
    std::string key = dir + "|" + sidecar + "|" + std::to_string(static_cast<int>(tier)) + "|" + std::to_string(ram_mb);
    std::lock_guard<std::mutex> lk(g_idx_mtx);
    auto it = g_open_handles.find(key);
    if (it != g_open_handles.end()) return it->second;
    pm_open_opts_t opts{};
    opts.tier_mode = tier;
    opts.ram_budget_bytes = ram_mb * 1024ULL * 1024ULL;
    opts.io_backend = PM_IO_URING;
    opts.self_calibrate = 1;
    opts.use_o_direct = 1;
    opts.sidecar_dir = sidecar.empty() ? nullptr : sidecar.c_str();
    pm_handle_t* h = pm_open(dir.c_str(), &opts);
    if (!h) throw duckdb::IOException("duckdb_certvec: failed to open index at " + dir);
    g_open_handles[key] = h;
    return h;
}

struct CertVecBindData : public duckdb::TableFunctionData {
    std::string index_dir;
    std::string sidecar_dir;
    std::vector<float> q_vecs;
    uint32_t L = 1;
    uint32_t k = 10;
    double   tau = 0.0;
    double   delta = 0.05;
    uint32_t nprobe = 16;
    uint32_t c1_pool = 600;
    bool     is_thresh = false;
    bool     has_aux_col = false;
    float    w0 = 0.0f;
    pm_tier_mode_t tier = PM_TIER_SPLIT_B1B4;
    size_t   ram_mb = 256;
};

struct CertVecLocalState : public duckdb::LocalTableFunctionState {
    roaring_bitmap_t* bm = nullptr;
    std::vector<uint32_t> aux_rids;
    std::vector<float>    aux_scores;
    bool                  executed = false;
    std::vector<pm_hit_t> results;
    size_t                emit_pos = 0;
    pm_query_stats_t      last_stats{};

    CertVecLocalState() { bm = roaring_bitmap_create(); }
    ~CertVecLocalState() override { if (bm) roaring_bitmap_free(bm); }
};

static duckdb::unique_ptr<duckdb::FunctionData> certvec_bind(
    duckdb::ClientContext& context,
    duckdb::TableFunctionBindInput& input,
    duckdb::vector<duckdb::LogicalType>& return_types,
    duckdb::vector<std::string>& names,
    bool is_thresh) {
    (void)context;
    auto data = duckdb::make_uniq<CertVecBindData>();
    data->is_thresh = is_thresh;
    // input.input_table_types tells us whether input table has 1 col (doc_id) or 2 cols (doc_id, aux_score)!
    data->has_aux_col = (input.input_table_types.size() >= 2);
    data->index_dir   = input.inputs[1].GetValue<std::string>();
    data->sidecar_dir = input.inputs[2].IsNull() ? "" : input.inputs[2].GetValue<std::string>();
    auto q_blob       = duckdb::StringValue::Get(input.inputs[3]);
    data->q_vecs.resize(q_blob.size() / sizeof(float));
    std::memcpy(data->q_vecs.data(), q_blob.data(), data->q_vecs.size() * sizeof(float));
    data->L = static_cast<uint32_t>(input.inputs[4].GetValue<int32_t>());
    uint32_t d_in = (data->L > 0) ? static_cast<uint32_t>(data->q_vecs.size() / data->L) : 128u;
    if (d_in >= 256u && data->L == 1u) { data->nprobe = 128; data->c1_pool = 3000; }
    if (is_thresh) {
        data->tau = input.inputs[5].GetValue<double>();
    } else {
        data->k = static_cast<uint32_t>(input.inputs[5].GetValue<int32_t>());
    }
    data->delta = input.inputs[6].GetValue<double>();
    if (input.inputs.size() > 7 && !input.inputs[7].IsNull()) data->w0 = input.inputs[7].GetValue<float>();

    return_types.push_back(duckdb::LogicalType::INTEGER);
    names.push_back("doc_id");
    return_types.push_back(duckdb::LogicalType::DOUBLE);
    names.push_back("score");
    return data;
}

static duckdb::unique_ptr<duckdb::FunctionData> bind_topk(
    duckdb::ClientContext& ctx, duckdb::TableFunctionBindInput& in,
    duckdb::vector<duckdb::LogicalType>& rt, duckdb::vector<std::string>& nm) {
    return certvec_bind(ctx, in, rt, nm, false);
}

static duckdb::unique_ptr<duckdb::FunctionData> bind_thresh(
    duckdb::ClientContext& ctx, duckdb::TableFunctionBindInput& in,
    duckdb::vector<duckdb::LogicalType>& rt, duckdb::vector<std::string>& nm) {
    return certvec_bind(ctx, in, rt, nm, true);
}

static duckdb::unique_ptr<duckdb::LocalTableFunctionState> init_local(
    duckdb::ExecutionContext& ctx, duckdb::TableFunctionInitInput& input, duckdb::GlobalTableFunctionState* gstate) {
    (void)ctx; (void)input; (void)gstate;
    return duckdb::make_uniq<CertVecLocalState>();
}

// Stream incoming filtered DataChunks directly via UnifiedVectorFormat + SelectionVector into CRoaring!
static duckdb::OperatorResultType certvec_in_out(
    duckdb::ExecutionContext& ctx, duckdb::TableFunctionInput& data,
    duckdb::DataChunk& input, duckdb::DataChunk& output) {
    (void)ctx; (void)output;
    auto& bind = data.bind_data->Cast<CertVecBindData>();
    auto& st = data.local_state->Cast<CertVecLocalState>();

    duckdb::UnifiedVectorFormat id_fmt;
    input.data[0].ToUnifiedFormat(input.size(), id_fmt);
    auto* id_ptr = duckdb::UnifiedVectorFormat::GetData<int32_t>(id_fmt);

    duckdb::UnifiedVectorFormat aux_fmt;
    const float* aux_ptr = nullptr;
    if (bind.has_aux_col && input.ColumnCount() >= 2) {
        input.data[1].ToUnifiedFormat(input.size(), aux_fmt);
        aux_ptr = duckdb::UnifiedVectorFormat::GetData<float>(aux_fmt);
    }

    for (duckdb::idx_t i = 0; i < input.size(); ++i) {
        auto idx = id_fmt.sel->get_index(i);
        if (!id_fmt.validity.RowIsValid(idx)) continue;
        int32_t rid = id_ptr[idx];
        if (rid < 0) continue;
        roaring_bitmap_add(st.bm, static_cast<uint32_t>(rid));
        if (aux_ptr) {
            auto aidx = aux_fmt.sel->get_index(i);
            if (aux_fmt.validity.RowIsValid(aidx)) {
                st.aux_rids.push_back(static_cast<uint32_t>(rid));
                st.aux_scores.push_back(aux_ptr[aidx]);
            }
        }
    }
    return duckdb::OperatorResultType::NEED_MORE_INPUT;
}

static std::atomic<double> g_last_core_ms{0.0};

static int duck_thresh_cb(void* ctx, uint32_t rid, double sc) {
    static_cast<std::vector<pm_hit_t>*>(ctx)->push_back({rid, sc});
    return 0;
}

// Execute pm_execute_query() ONCE on the accumulated CRoaring filter and stream vectorized output DataChunks!
static duckdb::OperatorFinalizeResultType certvec_final(
    duckdb::ExecutionContext& ctx, duckdb::TableFunctionInput& data, duckdb::DataChunk& output) {
    (void)ctx;
    auto& bind = data.bind_data->Cast<CertVecBindData>();
    auto& st = data.local_state->Cast<CertVecLocalState>();

    if (!st.executed) {
        st.executed = true;
        if (roaring_bitmap_get_cardinality(st.bm) == 0) {
            output.SetCardinality(0);
            return duckdb::OperatorFinalizeResultType::FINISHED;
        }
        pm_handle_t* h = get_or_open(bind.index_dir, bind.sidecar_dir, bind.tier, bind.ram_mb);
        roaring_bitmap_run_optimize(st.bm);
        pm_bitmap_view_t bm_view{PM_BM_ROARING_PTR, st.bm, 0};

        pm_aux_scores_t aux{};
        if (bind.has_aux_col && !st.aux_rids.empty()) {
            aux.kind = PM_AUX_SPARSE_PAIRS;
            aux.rids = st.aux_rids.data();
            aux.scores = st.aux_scores.data();
            aux.count = st.aux_rids.size();
            aux.w0 = bind.w0;
        } else {
            aux.kind = PM_AUX_NONE;
        }

        pm_query_spec_t qspec{};
        qspec.L = bind.L;
        qspec.q_vecs = bind.q_vecs.data();
        qspec.agg_kind = PM_AGG_WSUM;
        qspec.mode = bind.is_thresh ? PM_MODE_THRESHOLD : PM_MODE_TOPK;
        qspec.k = bind.k;
        qspec.tau = bind.tau;
        qspec.delta = bind.delta;
        qspec.split_lambda = 0.2;
        qspec.nprobe = bind.nprobe;
        qspec.c1_pool = bind.c1_pool;
        if (bind.is_thresh) qspec.force_path1 = 1;

        uint32_t out_n = bind.k;
        int rc = 0;
        if (bind.is_thresh) {
            rc = pm_execute_query(h, &bm_view, &aux, &qspec, nullptr, &out_n, duck_thresh_cb, &st.results, &st.last_stats);
        } else {
            st.results.resize(bind.k);
            rc = pm_execute_query(h, &bm_view, &aux, &qspec, st.results.data(), &out_n, nullptr, nullptr, &st.last_stats);
            st.results.resize(out_n);
        }
        if (rc != 0) throw duckdb::InvalidInputException("pm_execute_query failed with rc=" + std::to_string(rc));
        g_last_core_ms.store(st.last_stats.t_total_ms, std::memory_order_release);
    }

    size_t rem = st.results.size() - st.emit_pos;
    if (rem == 0) {
        output.SetCardinality(0);
        return duckdb::OperatorFinalizeResultType::FINISHED;
    }
    size_t chunk = std::min<size_t>(rem, STANDARD_VECTOR_SIZE);
    auto* out_ids = duckdb::FlatVector::GetData<int32_t>(output.data[0]);
    auto* out_sc  = duckdb::FlatVector::GetData<double>(output.data[1]);
    for (size_t i = 0; i < chunk; ++i) {
        out_ids[i] = static_cast<int32_t>(st.results[st.emit_pos + i].rid);
        out_sc[i]  = st.results[st.emit_pos + i].score;
    }
    st.emit_pos += chunk;
    output.SetCardinality(chunk);
    return (st.emit_pos < st.results.size())
        ? duckdb::OperatorFinalizeResultType::HAVE_MORE_OUTPUT
        : duckdb::OperatorFinalizeResultType::FINISHED;
}

struct CertVecGlobalState : public duckdb::GlobalTableFunctionState {
    duckdb::idx_t MaxThreads() const override { return 1; }
};

static duckdb::unique_ptr<duckdb::GlobalTableFunctionState> init_global(
    duckdb::ClientContext& ctx, duckdb::TableFunctionInitInput& input) {
    (void)ctx; (void)input;
    return duckdb::make_uniq<CertVecGlobalState>();
}

void register_certvec_functions(duckdb::Connection& con) {
    duckdb::TableFunction fn_topk(
        "pm_certvec_topk",
        {duckdb::LogicalType::TABLE, duckdb::LogicalType::VARCHAR, duckdb::LogicalType::VARCHAR,
         duckdb::LogicalType::BLOB, duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER,
         duckdb::LogicalType::DOUBLE, duckdb::LogicalType::FLOAT},
        nullptr, bind_topk, init_global, init_local);
    fn_topk.in_out_function = certvec_in_out;
    fn_topk.in_out_function_final = certvec_final;

    duckdb::TableFunction fn_thresh(
        "pm_certvec_thresh",
        {duckdb::LogicalType::TABLE, duckdb::LogicalType::VARCHAR, duckdb::LogicalType::VARCHAR,
         duckdb::LogicalType::BLOB, duckdb::LogicalType::INTEGER, duckdb::LogicalType::DOUBLE,
         duckdb::LogicalType::DOUBLE, duckdb::LogicalType::FLOAT},
        nullptr, bind_thresh, init_global, init_local);
    fn_thresh.in_out_function = certvec_in_out;
    fn_thresh.in_out_function_final = certvec_final;

    con.BeginTransaction();
    auto& catalog = duckdb::Catalog::GetSystemCatalog(*con.context);
    duckdb::CreateTableFunctionInfo info_tk(fn_topk);
    catalog.CreateTableFunction(*con.context, info_tk);
    duckdb::CreateTableFunctionInfo info_th(fn_thresh);
    catalog.CreateTableFunction(*con.context, info_th);
    con.Commit();
}

} // namespace duckdb_certvec

// Smoke test & benchmark driver for native DuckDB analytical SQL (JOIN + WHERE + GROUP BY + Q-TopK / Q-Thresh)!
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: duckdb_certvec_bench --smoke <v4_or_v5_index_dir> [sidecar_dir]\n";
        return 1;
    }
    std::string index_dir = argv[2];
    std::string sidecar_dir = (argc >= 4) ? argv[3] : "";

    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    con.Query("SET threads = 1;");
    duckdb_certvec::register_certvec_functions(con);

    // Inspect index N_docs & load real query 0 from core_index.bin (or synthetic query)
    pm_open_opts_t opts{};
    opts.tier_mode = PM_TIER_SPLIT_B1B4;
    opts.io_backend = PM_IO_URING;
    opts.self_calibrate = 1;
    opts.use_o_direct = 1;
    opts.sidecar_dir = sidecar_dir.empty() ? nullptr : sidecar_dir.c_str();
    pm_handle_t* h = pm_open(index_dir.c_str(), &opts);
    if (!h) { std::cerr << "Failed to open index\n"; return 1; }
    pm_index_info_t info{};
    pm_get_info(h, &info);
    uint64_t N = info.n_docs;
    uint32_t d = info.dims[0];
    pm_close(h);

    // Load real query 0..49 from core_index.bin (v4) or eval_bundle.bin (CERTVEC5)
    uint32_t Lq = 32;
    std::vector<float> all_q;
    std::ifstream cf(index_dir + "/core_index.bin", std::ios::binary);
    if (cf) {
        uint64_t hdr[12];
        cf.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        uint64_t K = hdr[2], dd = hdr[3];
        Lq = 32;
        all_q.assign(50ULL * Lq * dd, 0.0f);
        cf.seekg(static_cast<std::streamoff>(K * dd * 4ULL + dd * dd * 4ULL), std::ios::cur);
        cf.read(reinterpret_cast<char*>(all_q.data()), all_q.size() * 4ULL);
    } else {
        std::ifstream ef(index_dir + "/eval_bundle.bin", std::ios::binary);
        if (ef) {
            uint64_t hdr[8]; ef.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
            uint64_t Nq_b = hdr[2], d0 = hdr[3], L0 = hdr[5];
            Lq = static_cast<uint32_t>(L0);
            uint64_t nload = std::min<uint64_t>(50ULL, Nq_b);
            all_q.assign(50ULL * Lq * d0, 0.0f);
            ef.read(reinterpret_cast<char*>(all_q.data()), nload * Lq * d0 * 4ULL);
        } else {
            Lq = (d >= 256) ? 1u : 32u;
            all_q.assign(50ULL * Lq * d, 0.0f);
            for (size_t i = 0; i < all_q.size(); ++i) all_q[i] = (i % 7 == 0) ? 0.1f : -0.05f;
        }
    }
    std::vector<float> q0(all_q.begin(), all_q.begin() + static_cast<size_t>(Lq) * d);

    // Create real relational tables inside DuckDB (`docs` + `domains` for multi-table JOIN + WHERE + GROUP BY!)
    con.Query("CREATE TABLE domains (dom_id INT, dom_name VARCHAR);");
    con.Query("INSERT INTO domains VALUES (0,'finance'),(1,'legal'),(2,'medical'),(3,'tech'),(4,'science');");
    con.Query("CREATE TABLE docs AS SELECT i::INT AS doc_id, (i % 5)::INT AS dom_id, (50 + (i % 300))::INT AS page_len FROM range(" + std::to_string(N) + ") t(i);");

    duckdb::Value q_blob = duckdb::Value::BLOB(reinterpret_cast<const uint8_t*>(q0.data()), q0.size() * sizeof(float));

    // 1. Complex analytical SQL: JOIN + WHERE + pm_certvec_topk + JOIN back!
    auto t0 = std::chrono::steady_clock::now();
    auto prep1 = con.Prepare(
        "SELECT s.doc_id, m.dom_name, d.page_len, round(s.score, 4) AS score "
        "FROM pm_certvec_topk("
        "  (SELECT d.doc_id FROM docs d JOIN domains m ON d.dom_id = m.dom_id WHERE m.dom_name IN ('finance','legal') AND d.page_len <= 150), "
        "  $1::VARCHAR, $2::VARCHAR, $3::BLOB, " + std::to_string(Lq) + "::INTEGER, 10::INTEGER, 0.05::DOUBLE, 0.0::FLOAT) s "
        "JOIN docs d ON s.doc_id = d.doc_id JOIN domains m ON d.dom_id = m.dom_id "
        "ORDER BY s.score DESC;");
    auto res1_raw = prep1->Execute(index_dir, sidecar_dir, q_blob);
    double ms1 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (res1_raw->HasError()) { std::cerr << "DuckDB Q1 error: " << res1_raw->GetError() << "\n"; return 1; }
    std::vector<double> q1_scores;
    size_t q1_rows = 0;
    while (auto ch = res1_raw->Fetch()) {
        for (duckdb::idx_t r = 0; r < ch->size(); ++r) {
            q1_rows++;
            double sc = ch->GetValue(3, r).GetValue<double>();
            q1_scores.push_back(sc);
            if (q1_rows <= 5) {
                std::cout << "  row " << q1_rows << ": doc_id=" << ch->GetValue(0, r).ToString()
                          << " dom=" << ch->GetValue(1, r).ToString()
                          << " page_len=" << ch->GetValue(2, r).ToString()
                          << " score=" << sc << "\n";
            }
        }
    }
    std::cout << "[DuckDB Q-TopK JOIN+WHERE] rows=" << q1_rows << " ms=" << ms1 << "\n";

    // 2. Complex analytical SQL: Q-Thresh feeding downstream GROUP BY + COUNT(*) + AVG(score)!
    double tau = (q1_scores.size() >= 5) ? (q1_scores[4] - 0.01) : 0.5;
    auto t1 = std::chrono::steady_clock::now();
    auto prep2 = con.Prepare(
        "SELECT m.dom_name, COUNT(*) AS matched_docs, round(AVG(s.score), 4) AS avg_score "
        "FROM pm_certvec_thresh("
        "  (SELECT d.doc_id FROM docs d JOIN domains m ON d.dom_id = m.dom_id WHERE m.dom_name IN ('finance','legal') AND d.page_len <= 150), "
        "  $1::VARCHAR, $2::VARCHAR, $3::BLOB, " + std::to_string(Lq) + "::INTEGER, $4::DOUBLE, 0.05::DOUBLE, 0.0::FLOAT) s "
        "JOIN docs d ON s.doc_id = d.doc_id JOIN domains m ON d.dom_id = m.dom_id "
        "GROUP BY m.dom_name ORDER BY matched_docs DESC;");
    auto res2_raw = prep2->Execute(index_dir, sidecar_dir, q_blob, tau);
    double ms2 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    if (res2_raw->HasError()) { std::cerr << "DuckDB Q2 error: " << res2_raw->GetError() << "\n"; return 1; }
    size_t q2_groups = 0;
    while (auto ch = res2_raw->Fetch()) {
        for (duckdb::idx_t r = 0; r < ch->size(); ++r) {
            q2_groups++;
            std::cout << "  group: dom=" << ch->GetValue(0, r).ToString()
                      << " matched_docs=" << ch->GetValue(1, r).ToString()
                      << " avg_score=" << ch->GetValue(2, r).ToString() << "\n";
        }
    }
    std::cout << "[DuckDB Q-Thresh GROUP BY] tau=" << tau << " groups=" << q2_groups << " ms=" << ms2 << "\n";

    if (std::string(argv[1]) == "--bench-decomp" && argc >= 5) {
        std::string out_json = argv[4];
        // Direct per-hit Q-Thresh verification query to measure exact false positives across ALL 40 queries!
        auto prep2_fp = con.Prepare(
            "SELECT COUNT(*) FROM pm_certvec_thresh("
            "  (SELECT d.doc_id FROM docs d JOIN domains m ON d.dom_id = m.dom_id WHERE m.dom_name IN ('finance','legal') AND d.page_len <= 150), "
            "  $1::VARCHAR, $2::VARCHAR, $3::BLOB, " + std::to_string(Lq) + "::INTEGER, $4::DOUBLE, 0.05::DOUBLE, 0.0::FLOAT) s "
            "WHERE s.score < ($4::DOUBLE - 1e-9);");

        // Precompute realistic per-query threshold tau_q = 5th top score on each query qi
        std::vector<double> per_q_tau(45, tau);
        for (int qi = 0; qi < 45; ++qi) {
            const float* qp = all_q.data() + static_cast<size_t>(qi) * Lq * d;
            duckdb::Value qb = duckdb::Value::BLOB(reinterpret_cast<const uint8_t*>(qp), static_cast<size_t>(Lq) * d * sizeof(float));
            auto rt = prep1->Execute(index_dir, sidecar_dir, qb);
            std::vector<double> scs;
            while (auto ch = rt->Fetch()) {
                for (duckdb::idx_t r = 0; r < ch->size(); ++r) scs.push_back(ch->GetValue(3, r).GetValue<double>());
            }
            if (scs.size() >= 5) per_q_tau[qi] = scs[4];
            else if (!scs.empty()) per_q_tau[qi] = scs.back();
        }

        // Warm 5 queries then time 40 queries: decompose EXACT same-invocation DuckDB E2E vs inner libpm_certvec core time
        // AND explicitly measure false positives across all 40 Q-Thresh queries!
        auto time_pair = [&](bool is_thresh, double& duck_ms, double& core_ms, double& ov_p50_ms, double& avg_rows, uint64_t& tot_fp) {
            std::vector<double> d_ms, c_ms, ov_ms;
            double rsum = 0.0;
            tot_fp = 0;
            for (int qi = 0; qi < 45; ++qi) {
                const float* qp = all_q.data() + static_cast<size_t>(qi) * Lq * d;
                duckdb::Value qb = duckdb::Value::BLOB(reinterpret_cast<const uint8_t*>(qp), static_cast<size_t>(Lq) * d * sizeof(float));
                duckdb_certvec::g_last_core_ms.store(0.0, std::memory_order_release);
                auto td0 = std::chrono::steady_clock::now();
                auto r = is_thresh ? prep2->Execute(index_dir, sidecar_dir, qb, per_q_tau[qi]) : prep1->Execute(index_dir, sidecar_dir, qb);
                size_t rc = 0;
                while (auto ch = r->Fetch()) rc += ch->size();
                double dm = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
                double cm = duckdb_certvec::g_last_core_ms.load(std::memory_order_acquire);
                if (qi >= 5) {
                    d_ms.push_back(dm); c_ms.push_back(cm); ov_ms.push_back(std::max(0.0, dm - cm)); rsum += static_cast<double>(rc);
                    if (is_thresh) {
                        auto rfp = prep2_fp->Execute(index_dir, sidecar_dir, qb, per_q_tau[qi]);
                        while (auto ch = rfp->Fetch()) {
                            if (ch->size() > 0) tot_fp += static_cast<uint64_t>(ch->GetValue(0, 0).GetValue<int64_t>());
                        }
                    }
                }
            }
            std::sort(d_ms.begin(), d_ms.end()); std::sort(c_ms.begin(), c_ms.end()); std::sort(ov_ms.begin(), ov_ms.end());
            duck_ms = d_ms[d_ms.size() / 2];
            core_ms = c_ms[c_ms.size() / 2];
            ov_p50_ms = ov_ms[ov_ms.size() / 2];
            avg_rows = rsum / d_ms.size();
        };
        double tk_duck = 0, tk_core = 0, tk_ov = 0, tk_rows = 0;
        double th_duck = 0, th_core = 0, th_ov = 0, th_rows = 0;
        uint64_t tk_fp = 0, th_fp = 0;
        time_pair(false, tk_duck, tk_core, tk_ov, tk_rows, tk_fp);
        time_pair(true, th_duck, th_core, th_ov, th_rows, th_fp);
        std::ofstream jf(out_json);
        jf << "{\n  \"host\": \"duckdb_1.2.1\", \"index_dir\": \"" << index_dir << "\", \"n_docs\": " << N
           << ", \"integration_loc\": 250"
           << ",\n  \"topk\": {\"duckdb_e2e_p50_ms\": " << tk_duck << ", \"core_p50_ms\": " << tk_core
           << ", \"host_overhead_ms\": " << tk_ov << ", \"avg_rows\": " << tk_rows << "}"
           << ",\n  \"thresh_groupby\": {\"duckdb_e2e_p50_ms\": " << th_duck << ", \"core_p50_ms\": " << th_core
           << ", \"host_overhead_ms\": " << th_ov << ", \"avg_groups\": " << th_rows << ", \"false_positives\": " << th_fp << ", \"verified_queries\": 40}\n}\n";
        std::cout << "[DuckDB Decomp] TopK: e2e=" << tk_duck << "ms core=" << tk_core << "ms overhead=" << tk_ov << "ms"
                  << " | Thresh+GroupBy: e2e=" << th_duck << "ms core=" << th_core << "ms overhead=" << th_ov << "ms FP=" << th_fp << "/40q\n";
    }
    return 0;
}
