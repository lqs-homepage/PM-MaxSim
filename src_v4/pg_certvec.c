/*
 * pg_certvec.c — Native PostgreSQL 17.12 Extension for PM-MaxSim / CertVec
 *
 * Implements BLOCKING-03 zero-duplication multi-process architecture:
 *  1. shmem_request_hook / shmem_startup_hook allocates ONE Unified 4 KiB S3-FIFO Page Cache
 *     in PostgreSQL shared memory shared across all 48+ concurrent backend processes.
 *  2. Immutable Tier-0 (roaring_bitmap_frozen_view) & B1 are mmap(MAP_SHARED, PROT_READ) inside
 *     libpm_certvec.so — 1 physical RAM copy shared across all PG backends!
 *  3. Idiomatic zero-array SQL interfaces (MVCC-visible via PG's own relational scan + JOIN):
 *     - Composable zero-array aggregate `pm_roaring_agg(int4)` (INTERNAL CRoaring state ->
 *       compact run-optimized bytea, < 2 KB at sigma=100% vs 35 MB int4[]) and `pm_aux_agg(int4, float4)`
 *     - Set-returning functions `pm_certvec_topk(...)`, `pm_certvec_hybrid_topk(...)`, and
 *       `pm_certvec_thresh(...)` supporting optional sparse auxiliary scores (BM25 / SQL column)
 *       and arbitrary multi-table JOIN/GROUP BY CTEs!
 */

#include "postgres.h"

#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"

#include "pm_certvec.h"
#include "roaring.h"

#include <string.h>

PG_MODULE_MAGIC;

/* GUC parameters (configurable in postgresql.conf or SET) */
static int   g_cache_mb = 256;
static char* g_default_index_dir = NULL;
static char* g_default_sidecar_dir = NULL;
static int   g_tier_mode = PM_TIER_SPLIT_B1B4;
static int   g_use_o_direct = 1;

/* Process-shared PG Shmem pointer for Unified 4 KiB S3-FIFO Page Cache */
static shmem_request_hook_type prev_shmem_request_hook = NULL;
static shmem_startup_hook_type prev_shmem_startup_hook = NULL;
static void*  g_pg_shmem_cache = NULL;
static size_t g_pg_shmem_bytes = 0;

/* Backend-local cached pm_handle_t (mmap(MAP_SHARED) Tier-0/B1 + attached PG Shmem S3-FIFO cache!) */
#define MAX_OPEN_INDEXES 8
typedef struct {
    char         path[512];
    char         sidecar[512];
    pm_handle_t* handle;
} CachedIndex;

static CachedIndex g_open_idx[MAX_OPEN_INDEXES];
static int         g_num_open = 0;

static void pg_certvec_shmem_request(void) {
    if (prev_shmem_request_hook) prev_shmem_request_hook();
    if (g_cache_mb > 0) {
        size_t data_bytes = (size_t)g_cache_mb * 1024ULL * 1024ULL;
        g_pg_shmem_bytes = pm_s3fifo_required_shmem_bytes(data_bytes);
        RequestAddinShmemSpace(g_pg_shmem_bytes);
    }
}

static void pg_certvec_shmem_startup(void) {
    bool found = false;
    if (prev_shmem_startup_hook) prev_shmem_startup_hook();
    if (g_pg_shmem_bytes == 0) return;

    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
    g_pg_shmem_cache = ShmemInitStruct("pg_certvec_s3fifo_cache", g_pg_shmem_bytes, &found);
    if (!found && g_pg_shmem_cache != NULL) {
        memset(g_pg_shmem_cache, 0, 64); /* First backend initializes S3-FIFO header on pm_open */
    }
    LWLockRelease(AddinShmemInitLock);
}

static pm_handle_t* get_or_open_index(const char* index_dir, const char* sidecar_dir) {
    const char* sc = sidecar_dir ? sidecar_dir : "";
    for (int i = 0; i < g_num_open; ++i) {
        if (strcmp(g_open_idx[i].path, index_dir) == 0 && strcmp(g_open_idx[i].sidecar, sc) == 0) {
            return g_open_idx[i].handle;
        }
    }
    if (g_num_open >= MAX_OPEN_INDEXES) {
        ereport(ERROR, (errmsg("pg_certvec: max open indexes (%d) exceeded", MAX_OPEN_INDEXES)));
    }
    pm_open_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.tier_mode = (pm_tier_mode_t)g_tier_mode;
    opts.io_backend = PM_IO_URING;
    opts.self_calibrate = 1;
    opts.use_o_direct = (uint8_t)g_use_o_direct;
    opts.shared_cache_mem = g_pg_shmem_cache;
    opts.shared_cache_bytes = g_pg_shmem_bytes;
    opts.sidecar_dir = (sc[0] != '\0') ? sc : NULL;

    /* Serialize first-time PG shared-memory S3-FIFO header initialization across cold concurrent backends! */
    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
    pm_handle_t* h = pm_open(index_dir, &opts);
    LWLockRelease(AddinShmemInitLock);
    if (!h) {
        ereport(ERROR, (errmsg("pg_certvec: failed to open index at '%s'", index_dir)));
    }
    snprintf(g_open_idx[g_num_open].path, sizeof(g_open_idx[g_num_open].path), "%s", index_dir);
    snprintf(g_open_idx[g_num_open].sidecar, sizeof(g_open_idx[g_num_open].sidecar), "%s", sc);
    g_open_idx[g_num_open].handle = h;
    g_num_open++;
    return h;
}

/* ============================================================================
 * 1. Zero-Array Composable CRoaring Aggregate: pm_roaring_agg(int4) -> bytea
 * ============================================================================ */
typedef struct {
    roaring_bitmap_t* bm;
} RoaringAggState;

static void roaring_agg_cleanup(void* arg) {
    RoaringAggState* st = (RoaringAggState*)arg;
    if (st && st->bm) {
        roaring_bitmap_free(st->bm);
        st->bm = NULL;
    }
}

PG_FUNCTION_INFO_V1(pm_roaring_agg_trans);
Datum pm_roaring_agg_trans(PG_FUNCTION_ARGS) {
    MemoryContext agg_ctx;
    RoaringAggState* st;
    if (!AggCheckCallContext(fcinfo, &agg_ctx)) {
        ereport(ERROR, (errmsg("pm_roaring_agg_trans called in non-aggregate context")));
    }
    if (PG_ARGISNULL(0)) {
        MemoryContext old_ctx = MemoryContextSwitchTo(agg_ctx);
        st = (RoaringAggState*)palloc0(sizeof(RoaringAggState));
        st->bm = roaring_bitmap_create();
        MemoryContextCallback* cb = (MemoryContextCallback*)palloc(sizeof(MemoryContextCallback));
        cb->func = roaring_agg_cleanup;
        cb->arg = st;
        MemoryContextRegisterResetCallback(agg_ctx, cb);
        MemoryContextSwitchTo(old_ctx);
    } else {
        st = (RoaringAggState*)PG_GETARG_POINTER(0);
    }
    if (!PG_ARGISNULL(1)) {
        int32 rid = PG_GETARG_INT32(1);
        if (rid >= 0) roaring_bitmap_add(st->bm, (uint32_t)rid);
    }
    PG_RETURN_POINTER(st);
}

PG_FUNCTION_INFO_V1(pm_roaring_agg_final);
Datum pm_roaring_agg_final(PG_FUNCTION_ARGS) {
    if (PG_ARGISNULL(0)) PG_RETURN_NULL();
    RoaringAggState* st = (RoaringAggState*)PG_GETARG_POINTER(0);
    if (!st || !st->bm) PG_RETURN_NULL();
    roaring_bitmap_run_optimize(st->bm);
    size_t sz = roaring_bitmap_portable_size_in_bytes(st->bm);
    bytea* res = (bytea*)palloc(VARHDRSZ + sz);
    SET_VARSIZE(res, VARHDRSZ + sz);
    roaring_bitmap_portable_serialize(st->bm, VARDATA(res));
    PG_RETURN_BYTEA_P(res);
}

/* ============================================================================
 * 1B. Zero-Array Hybrid Sparse Aux Aggregate: pm_aux_agg(int4, float4) -> bytea
 *     Packs (uint32 count, uint32 rids[count], float scores[count]) directly from PG JOIN/WHERE!
 * ============================================================================ */
typedef struct {
    uint32_t* rids;
    float*    scores;
    uint32_t  count;
    uint32_t  cap;
} AuxAggState;

PG_FUNCTION_INFO_V1(pm_aux_agg_trans);
Datum pm_aux_agg_trans(PG_FUNCTION_ARGS) {
    MemoryContext agg_ctx;
    AuxAggState* st;
    if (!AggCheckCallContext(fcinfo, &agg_ctx))
        ereport(ERROR, (errmsg("pm_aux_agg_trans called in non-aggregate context")));
    MemoryContext old_ctx = MemoryContextSwitchTo(agg_ctx);
    if (PG_ARGISNULL(0)) {
        st = (AuxAggState*)palloc0(sizeof(AuxAggState));
        st->cap = 256;
        st->rids = (uint32_t*)palloc(sizeof(uint32_t) * st->cap);
        st->scores = (float*)palloc(sizeof(float) * st->cap);
    } else {
        st = (AuxAggState*)PG_GETARG_POINTER(0);
    }
    if (!PG_ARGISNULL(1) && !PG_ARGISNULL(2)) {
        int32 rid = PG_GETARG_INT32(1);
        float4 sc = PG_GETARG_FLOAT4(2);
        if (rid >= 0) {
            if (st->count >= st->cap) {
                st->cap *= 2u;
                st->rids = (uint32_t*)repalloc(st->rids, sizeof(uint32_t) * st->cap);
                st->scores = (float*)repalloc(st->scores, sizeof(float) * st->cap);
            }
            st->rids[st->count] = (uint32_t)rid;
            st->scores[st->count] = sc;
            st->count++;
        }
    }
    MemoryContextSwitchTo(old_ctx);
    PG_RETURN_POINTER(st);
}

PG_FUNCTION_INFO_V1(pm_aux_agg_final);
Datum pm_aux_agg_final(PG_FUNCTION_ARGS) {
    if (PG_ARGISNULL(0)) PG_RETURN_NULL();
    AuxAggState* st = (AuxAggState*)PG_GETARG_POINTER(0);
    size_t sz = sizeof(uint32_t) + (size_t)st->count * (sizeof(uint32_t) + sizeof(float));
    bytea* res = (bytea*)palloc(VARHDRSZ + sz);
    SET_VARSIZE(res, VARHDRSZ + sz);
    char* p = VARDATA(res);
    memcpy(p, &st->count, sizeof(uint32_t)); p += sizeof(uint32_t);
    if (st->count > 0) {
        memcpy(p, st->rids, (size_t)st->count * sizeof(uint32_t)); p += (size_t)st->count * sizeof(uint32_t);
        memcpy(p, st->scores, (size_t)st->count * sizeof(float));
    }
    PG_RETURN_BYTEA_P(res);
}

static void emit_hits_to_tuplestore(FunctionCallInfo fcinfo, const pm_hit_t* hits, uint32_t count);

PG_FUNCTION_INFO_V1(pm_certvec_hybrid_topk);
Datum pm_certvec_hybrid_topk(PG_FUNCTION_ARGS) {
    char* index_dir = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* sidecar_dir = PG_ARGISNULL(1) ? NULL : text_to_cstring(PG_GETARG_TEXT_PP(1));
    pm_handle_t* h = get_or_open_index(index_dir, sidecar_dir);

    roaring_bitmap_t* bm = NULL;
    pm_bitmap_view_t bm_view;
    memset(&bm_view, 0, sizeof(bm_view));
    if (!PG_ARGISNULL(2)) {
        bytea* b = PG_GETARG_BYTEA_PP(2);
        bm = roaring_bitmap_portable_deserialize_safe(VARDATA_ANY(b), VARSIZE_ANY_EXHDR(b));
        if (!bm) ereport(ERROR, (errmsg("pg_certvec: invalid roaring bitmap")));
        bm_view.kind = PM_BM_ROARING_PTR;
        bm_view.data = bm;
    } else {
        bm_view.kind = PM_BM_NONE;
    }

    pm_aux_scores_t aux;
    memset(&aux, 0, sizeof(aux));
    if (!PG_ARGISNULL(3)) {
        bytea* ab = PG_GETARG_BYTEA_PP(3);
        const char* p = VARDATA_ANY(ab);
        uint32_t cnt = 0;
        memcpy(&cnt, p, sizeof(uint32_t)); p += sizeof(uint32_t);
        aux.kind = PM_AUX_SPARSE_PAIRS;
        aux.count = cnt;
        aux.rids = (const uint32_t*)p;
        aux.scores = (const float*)(p + (size_t)cnt * sizeof(uint32_t));
        aux.w0 = PG_GETARG_FLOAT4(4);
    }

    bytea* qbytes = PG_GETARG_BYTEA_PP(5);
    int32  L = PG_GETARG_INT32(6);
    int32  k = PG_GETARG_INT32(7);
    double delta = PG_GETARG_FLOAT8(8);
    int32  nprobe = (PG_NARGS() > 9 && !PG_ARGISNULL(9)) ? PG_GETARG_INT32(9) : 16;
    int32  c1_pool = (PG_NARGS() > 10 && !PG_ARGISNULL(10)) ? PG_GETARG_INT32(10) : 600;

    uint32_t* fids_arr = NULL;
    pm_set_op_t* sops_arr = NULL;
    uint32_t* rvals_arr = NULL;
    float* wts_arr = NULL;
    if (PG_NARGS() > 11 && !PG_ARGISNULL(11)) {
        ArrayType* a = PG_GETARG_ARRAYTYPE_P(11);
        Datum* elems = NULL; int ne = 0;
        deconstruct_array_builtin(a, INT4OID, &elems, NULL, &ne);
        if (ne == L) {
            fids_arr = (uint32_t*)palloc(sizeof(uint32_t) * (size_t)L);
            for (int i = 0; i < L; ++i) fids_arr[i] = (uint32_t)DatumGetInt32(elems[i]);
        }
    }
    if (PG_NARGS() > 12 && !PG_ARGISNULL(12)) {
        ArrayType* a = PG_GETARG_ARRAYTYPE_P(12);
        Datum* elems = NULL; int ne = 0;
        deconstruct_array_builtin(a, INT4OID, &elems, NULL, &ne);
        if (ne == L) {
            sops_arr = (pm_set_op_t*)palloc(sizeof(pm_set_op_t) * (size_t)L);
            for (int i = 0; i < L; ++i) sops_arr[i] = (pm_set_op_t)DatumGetInt32(elems[i]);
        }
    }
    if (PG_NARGS() > 13 && !PG_ARGISNULL(13)) {
        ArrayType* a = PG_GETARG_ARRAYTYPE_P(13);
        Datum* elems = NULL; int ne = 0;
        deconstruct_array_builtin(a, INT4OID, &elems, NULL, &ne);
        if (ne == L) {
            rvals_arr = (uint32_t*)palloc(sizeof(uint32_t) * (size_t)L);
            for (int i = 0; i < L; ++i) rvals_arr[i] = (uint32_t)DatumGetInt32(elems[i]);
        }
    }
    if (PG_NARGS() > 14 && !PG_ARGISNULL(14)) {
        ArrayType* a = PG_GETARG_ARRAYTYPE_P(14);
        Datum* elems = NULL; int ne = 0;
        deconstruct_array(a, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT, &elems, NULL, &ne);
        if (ne == L) {
            wts_arr = (float*)palloc(sizeof(float) * (size_t)L);
            for (int i = 0; i < L; ++i) wts_arr[i] = DatumGetFloat4(elems[i]);
        }
    }

    pm_query_spec_t qspec;
    memset(&qspec, 0, sizeof(qspec));
    qspec.L = (uint32_t)L;
    qspec.q_vecs = (const float*)VARDATA_ANY(qbytes);
    qspec.field_ids = fids_arr;
    qspec.set_ops = sops_arr;
    qspec.r_vals = rvals_arr;
    qspec.weights = wts_arr;
    qspec.agg_kind = PM_AGG_WSUM;
    qspec.mode = PM_MODE_TOPK;
    qspec.k = (uint32_t)k;
    qspec.delta = delta;
    qspec.split_lambda = 0.2;
    qspec.nprobe = (uint32_t)nprobe;
    qspec.c1_pool = (uint32_t)c1_pool;

    pm_hit_t* hits = (pm_hit_t*)palloc0(sizeof(pm_hit_t) * (size_t)k);
    uint32_t out_cnt = (uint32_t)k;
    pm_query_stats_t st;
    int rc = pm_execute_query(h, &bm_view, &aux, &qspec, hits, &out_cnt, NULL, NULL, &st);
    if (bm) roaring_bitmap_free(bm);
    if (fids_arr) pfree(fids_arr);
    if (sops_arr) pfree(sops_arr);
    if (rvals_arr) pfree(rvals_arr);
    if (wts_arr) pfree(wts_arr);
    if (rc != 0) ereport(ERROR, (errmsg("pg_certvec_hybrid_topk failed with rc=%d", rc)));

    emit_hits_to_tuplestore(fcinfo, hits, out_cnt);
    pfree(hits);
    PG_RETURN_NULL();
}

/* ============================================================================
 * 2. Set-Returning Core Search & Threshold SQL Functions
 *    - pm_certvec_topk(index_dir text, sidecar_dir text, filter_roaring bytea,
 *                      query_f32 bytea, L int4, k int4, delta float8,
 *                      nprobe int4, c1_pool int4) RETURNS TABLE(doc_id int4, score float8)
 *    - pm_certvec_thresh(index_dir text, sidecar_dir text, filter_roaring bytea,
 *                        query_f32 bytea, L int4, tau float8, delta float8,
 *                        nprobe int4) RETURNS TABLE(doc_id int4, score float8)
 * ============================================================================ */

typedef struct {
    pm_hit_t* hits;
    uint32_t  count;
    uint32_t  cap;
} ThreshVec;

static int thresh_collect_cb(void* ctx, uint32_t rid, double sc) {
    ThreshVec* tv = (ThreshVec*)ctx;
    if (tv->count >= tv->cap) {
        tv->cap = (tv->cap == 0) ? 256u : (tv->cap * 2u);
        tv->hits = (pm_hit_t*)repalloc(tv->hits, sizeof(pm_hit_t) * tv->cap);
    }
    tv->hits[tv->count].rid = rid;
    tv->hits[tv->count].score = sc;
    tv->count++;
    return 0;
}

static void emit_hits_to_tuplestore(FunctionCallInfo fcinfo, const pm_hit_t* hits, uint32_t count) {
    ReturnSetInfo* rsinfo = (ReturnSetInfo*)fcinfo->resultinfo;
    TupleDesc tupdesc;
    Tuplestorestate* tupstore;
    MemoryContext per_query_ctx;
    MemoryContext old_ctx;

    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
        ereport(ERROR, (errmsg("set-valued function called in context that cannot accept a set")));
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR, (errmsg("materialize mode required")));

    per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
    old_ctx = MemoryContextSwitchTo(per_query_ctx);
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR, (errmsg("return type must be a row type (doc_id int4, score float8)")));
    tupstore = tuplestore_begin_heap(true, false, work_mem);
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tupstore;
    rsinfo->setDesc = tupdesc;
    MemoryContextSwitchTo(old_ctx);

    for (uint32_t i = 0; i < count; ++i) {
        Datum vals[2];
        bool  nulls[2] = {false, false};
        vals[0] = Int32GetDatum((int32)hits[i].rid);
        vals[1] = Float8GetDatum(hits[i].score);
        tuplestore_putvalues(tupstore, tupdesc, vals, nulls);
    }
}

PG_FUNCTION_INFO_V1(pm_certvec_topk);
Datum pm_certvec_topk(PG_FUNCTION_ARGS) {
    char* index_dir = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* sidecar_dir = PG_ARGISNULL(1) ? NULL : text_to_cstring(PG_GETARG_TEXT_PP(1));
    pm_handle_t* h = get_or_open_index(index_dir, sidecar_dir);

    roaring_bitmap_t* bm = NULL;
    pm_bitmap_view_t bm_view;
    memset(&bm_view, 0, sizeof(bm_view));
    if (!PG_ARGISNULL(2)) {
        bytea* b = PG_GETARG_BYTEA_PP(2);
        bm = roaring_bitmap_portable_deserialize_safe(VARDATA_ANY(b), VARSIZE_ANY_EXHDR(b));
        if (!bm) ereport(ERROR, (errmsg("pg_certvec: invalid serialized roaring bitmap")));
        bm_view.kind = PM_BM_ROARING_PTR;
        bm_view.data = bm;
    } else {
        bm_view.kind = PM_BM_NONE;
    }

    bytea* qbytes = PG_GETARG_BYTEA_PP(3);
    int32  L = PG_GETARG_INT32(4);
    int32  k = PG_GETARG_INT32(5);
    double delta = PG_GETARG_FLOAT8(6);
    int32  nprobe = PG_ARGISNULL(7) ? 16 : PG_GETARG_INT32(7);
    int32  c1_pool = PG_ARGISNULL(8) ? 600 : PG_GETARG_INT32(8);

    pm_query_spec_t qspec;
    memset(&qspec, 0, sizeof(qspec));
    qspec.L = (uint32_t)L;
    qspec.q_vecs = (const float*)VARDATA_ANY(qbytes);
    qspec.agg_kind = PM_AGG_WSUM;
    qspec.mode = PM_MODE_TOPK;
    qspec.k = (uint32_t)k;
    qspec.delta = delta;
    qspec.split_lambda = 0.2;
    qspec.nprobe = (uint32_t)nprobe;
    qspec.c1_pool = (uint32_t)c1_pool;

    pm_hit_t* hits = (pm_hit_t*)palloc0(sizeof(pm_hit_t) * (size_t)k);
    uint32_t out_cnt = (uint32_t)k;
    pm_query_stats_t st;
    int rc = pm_execute_query(h, &bm_view, NULL, &qspec, hits, &out_cnt, NULL, NULL, &st);
    if (bm) roaring_bitmap_free(bm);
    if (rc != 0) ereport(ERROR, (errmsg("pg_certvec_topk failed with rc=%d", rc)));

    emit_hits_to_tuplestore(fcinfo, hits, out_cnt);
    pfree(hits);
    PG_RETURN_NULL();
}

PG_FUNCTION_INFO_V1(pm_certvec_thresh);
Datum pm_certvec_thresh(PG_FUNCTION_ARGS) {
    char* index_dir = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* sidecar_dir = PG_ARGISNULL(1) ? NULL : text_to_cstring(PG_GETARG_TEXT_PP(1));
    pm_handle_t* h = get_or_open_index(index_dir, sidecar_dir);

    roaring_bitmap_t* bm = NULL;
    pm_bitmap_view_t bm_view;
    memset(&bm_view, 0, sizeof(bm_view));
    if (!PG_ARGISNULL(2)) {
        bytea* b = PG_GETARG_BYTEA_PP(2);
        bm = roaring_bitmap_portable_deserialize_safe(VARDATA_ANY(b), VARSIZE_ANY_EXHDR(b));
        if (!bm) ereport(ERROR, (errmsg("pg_certvec: invalid serialized roaring bitmap")));
        bm_view.kind = PM_BM_ROARING_PTR;
        bm_view.data = bm;
    } else {
        bm_view.kind = PM_BM_NONE;
    }

    bytea* qbytes = PG_GETARG_BYTEA_PP(3);
    int32  L = PG_GETARG_INT32(4);
    double tau = PG_GETARG_FLOAT8(5);
    double delta = PG_GETARG_FLOAT8(6);
    int32  nprobe = PG_ARGISNULL(7) ? 16 : PG_GETARG_INT32(7);

    pm_query_spec_t qspec;
    memset(&qspec, 0, sizeof(qspec));
    qspec.L = (uint32_t)L;
    qspec.q_vecs = (const float*)VARDATA_ANY(qbytes);
    qspec.agg_kind = PM_AGG_WSUM;
    qspec.mode = PM_MODE_THRESHOLD;
    qspec.tau = tau;
    qspec.delta = delta;
    qspec.split_lambda = 0.2;
    qspec.nprobe = (uint32_t)nprobe;
    qspec.force_path1 = 1; /* Global unconditional Path I guarantee with rho_max det pruning! */

    ThreshVec tv;
    tv.cap = 256;
    tv.count = 0;
    tv.hits = (pm_hit_t*)palloc(sizeof(pm_hit_t) * tv.cap);
    uint32_t out_cnt = 0;
    pm_query_stats_t st;
    int rc = pm_execute_query(h, &bm_view, NULL, &qspec, NULL, &out_cnt, thresh_collect_cb, &tv, &st);
    if (bm) roaring_bitmap_free(bm);
    if (rc != 0) ereport(ERROR, (errmsg("pg_certvec_thresh failed with rc=%d", rc)));

    emit_hits_to_tuplestore(fcinfo, tv.hits, tv.count);
    pfree(tv.hits);
    PG_RETURN_NULL();
}

PG_FUNCTION_INFO_V1(pm_certvec_info);
Datum pm_certvec_info(PG_FUNCTION_ARGS) {
    char* index_dir = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* sidecar_dir = PG_ARGISNULL(1) ? NULL : text_to_cstring(PG_GETARG_TEXT_PP(1));
    pm_handle_t* h = get_or_open_index(index_dir, sidecar_dir);
    pm_index_info_t info;
    pm_get_info(h, &info);
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"n_docs\":%llu,\"n_fields\":%u,\"tier0_mb\":%.2f,\"b1_mb\":%.2f,\"b4_mb\":%.2f,\"shmem_cache_mb\":%.2f,\"tier_mode\":%d,\"tau1\":%llu}",
             (unsigned long long)info.n_docs, info.n_fields,
             info.tier0_ram_bytes / 1048576.0, info.b1_ram_bytes / 1048576.0,
             info.b4_ram_bytes / 1048576.0, info.cache_ram_bytes / 1048576.0,
             info.active_tier_mode, (unsigned long long)info.calibrated_tau1);
    PG_RETURN_TEXT_P(cstring_to_text(buf));
}

void _PG_init(void) {
    DefineCustomIntVariable("pg_certvec.cache_mb",
                            "Shared Unified 4 KiB S3-FIFO page cache size in MB.",
                            NULL, &g_cache_mb, 256, 0, 131072,
                            PGC_POSTMASTER, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("pg_certvec.tier_mode",
                            "Default tier mode (1=NVME, 2=SPLIT_B1B4, 3=DRAM_T1).",
                            NULL, &g_tier_mode, PM_TIER_SPLIT_B1B4, 0, 3,
                            PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("pg_certvec.use_o_direct",
                            "Enable O_DIRECT on NVMe stores (1=on, 0=off).",
                            NULL, &g_use_o_direct, 1, 0, 1,
                            PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomStringVariable("pg_certvec.default_index_dir",
                               "Default CertVec index directory.",
                               NULL, &g_default_index_dir, "",
                               PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomStringVariable("pg_certvec.default_sidecar_dir",
                               "Default CertVec sidecar directory.",
                               NULL, &g_default_sidecar_dir, "",
                               PGC_USERSET, 0, NULL, NULL, NULL);

    if (process_shared_preload_libraries_in_progress) {
        prev_shmem_request_hook = shmem_request_hook;
        shmem_request_hook = pg_certvec_shmem_request;
        prev_shmem_startup_hook = shmem_startup_hook;
        shmem_startup_hook = pg_certvec_shmem_startup;
    }
}
