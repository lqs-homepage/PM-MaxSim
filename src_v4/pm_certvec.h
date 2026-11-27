#ifndef PM_CERTVEC_H
#define PM_CERTVEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pm_handle pm_handle_t;

typedef enum {
    PM_OP_MAX        = 0,
    PM_OP_TOP_R_SUM  = 1,
    PM_OP_TOP_R_MEAN = 2,
    PM_OP_AVG        = 3
} pm_set_op_t;

typedef enum {
    PM_AGG_WSUM      = 0,
    PM_AGG_RELU_WSUM = 1,
    PM_AGG_MIN       = 2
} pm_agg_kind_t;

typedef enum {
    PM_MODE_TOPK      = 0,
    PM_MODE_THRESHOLD = 1
} pm_query_mode_t;

typedef enum {
    PM_BM_NONE           = 0,
    PM_BM_ROARING_PTR    = 1, // const roaring_bitmap_t*
    PM_BM_FROZEN_ROARING = 2, // const char* (32-byte aligned frozen CRoaring buffer)
    PM_BM_RID_ARRAY      = 3, // const uint32_t* (sorted or unsorted RID list)
    PM_BM_BITSET64       = 4  // const uint64_t* (N_words dense bitset)
} pm_bm_kind_t;

typedef enum {
    PM_AUX_NONE         = 0,
    PM_AUX_DENSE_F32    = 1, // float[N_docs]
    PM_AUX_SPARSE_PAIRS = 2  // (uint32_t rids[count], float scores[count], default_val)
} pm_aux_kind_t;

typedef enum {
    PM_IO_URING      = 0,
    PM_IO_PREAD_POOL = 1
} pm_io_backend_t;

typedef enum {
    PM_TIER_AUTO_KKT   = 0, // Continuous Theorem 5 KKT allocation from ram_budget_bytes
    PM_TIER_NVME_T1    = 1, // Tier-0 RAM only; B4 + FP16 on NVMe (optional S3-FIFO cache)
    PM_TIER_SPLIT_B1B4 = 2, // B1 RAM + B4/FP16 NVMe (Theorem 3B strict two-wave split)
    PM_TIER_DRAM_T1    = 3  // B4 RAM + FP16 NVMe
} pm_tier_mode_t;

typedef struct {
    pm_bm_kind_t kind;
    const void*  data;
    size_t       len;  // byte length for FROZEN_ROARING, element count for RID_ARRAY / BITSET64
} pm_bitmap_view_t;

typedef struct {
    pm_aux_kind_t   kind;
    const float*    scores;
    const uint32_t* rids;
    size_t          count;
    float           default_val;
    float           w0; // Signed scalar weight in R (allows negative penalties)
} pm_aux_scores_t;

typedef struct {
    uint32_t           L;             // Number of query vector slots (zero-norm / zero-weight auto-pruned)
    const float*       q_vecs;        // Concatenated query vectors (length sum_i d_{field_ids[i]})
    const uint32_t*    field_ids;     // [L] target field index in [0..F-1] (NULL => all 0)
    const pm_set_op_t* set_ops;       // [L] MAX | TOP_R_SUM | TOP_R_MEAN | AVG (NULL => all MAX)
    const uint32_t*    r_vals;        // [L] r_i >= 1 for TOP_R_* (NULL => all 1)
    const float*       weights;       // [L] STRICTLY >= 0 (any w_i < 0 rejected with -EINVAL; NULL => all 1.0)
    const float*       relu_taus;     // [L] thresholds for PM_AGG_RELU_WSUM (NULL otherwise)
    pm_agg_kind_t      agg_kind;
    pm_query_mode_t    mode;
    uint32_t           k;             // Top-k for PM_MODE_TOPK
    double             tau;           // Exact FP64 threshold for PM_MODE_THRESHOLD
    double             delta;         // Target failure probability (default 0.05; alpha clamped <= 0.05)
    double             split_lambda;  // Theorem 3B Wave-1 budget share (default 0.2)
    uint8_t            part2_allk;    // 0 = Part 1 expected recall, 1 = Part 2 exact All-k
    uint32_t           nprobe;        // Tier-0 centroid probes per query vector (0 = auto default 16)
    uint32_t           c1_pool;       // Tier-0 candidate pool size C1 (0 = auto default 600)
    uint8_t            disable_stage0a; // 1 = bypass Theorem 4C container pre-filter (ablation)
    uint8_t            force_path1;     // 1 = force Path I (ablation)
    uint8_t            force_path3;     // 1 = force Path III (ablation)
    uint8_t            uncertified_approx; // 1 = B4 point-estimate without Tier-2 (ablation)
} pm_query_spec_t;

typedef struct {
    uint32_t rid;
    double   score;
} pm_hit_t;

// Growable callback for variable-cardinality Q-Thresh output (return 0 to continue, non-zero to abort)
typedef int (*pm_emit_cb_t)(void* ctx, uint32_t rid, double exact_score);

typedef struct {
    uint32_t path_used;           // 1 = Path I, 3 = Path III, 31 = Path III -> CB fallback to Path I
    uint32_t stage0a_used;        // 1 if Theorem 4C container pre-filter fired
    uint32_t cb_tripped;          // 1 if Theorem 4B KL circuit breaker tripped
    uint64_t valid_filtered_docs; // N_P
    uint64_t stage0a_shortlist;   // |U_short| if Stage-0a fired
    uint64_t c1_evaluated;        // Entities entering Tier-1
    uint64_t wave1_survivors;     // Entities surviving B1 Wave 1 (Split mode)
    uint64_t tier2_fetched;       // Entities fetched at Tier-2 FP16
    uint64_t nvme_bytes_read;     // Physical O_DIRECT NVMe bytes read (post S3-FIFO cache misses)
    uint64_t cache_hits_b4;       // Unified S3-FIFO 4 KiB page hits for B4
    uint64_t cache_hits_t2;       // Unified S3-FIFO 4 KiB page hits for Tier-2 FP16
    uint64_t cache_misses;        // Unified S3-FIFO 4 KiB page misses
    double   t_lut_ms;
    double   t_tier0_ms;
    double   t_b1_ms;
    double   t_b4_ms;
    double   t_t2_ms;
    double   t_total_ms;
} pm_query_stats_t;

typedef struct {
    size_t          ram_budget_bytes;   // Discretionary RAM = MemoryMax - M_host - M_0 (0 => use tier_mode)
    pm_tier_mode_t  tier_mode;          // Explicit tier mode or PM_TIER_AUTO_KKT
    pm_io_backend_t io_backend;         // PM_IO_URING or PM_IO_PREAD_POOL
    uint8_t         self_calibrate;     // 1 => 1-2s startup micro-probe sets host tau1*, L_strip, QD
    uint8_t         disable_amx;        // 1 => force AVX-512 / scalar fallback (ablation)
    uint8_t         use_o_direct;       // 1 => O_DIRECT (default); 0 => buffered (fallback)
    void*           shared_cache_mem;   // Optional PG Shared Memory region for unified 4 KiB S3-FIFO cache
    size_t          shared_cache_bytes; // Size of shared_cache_mem (0 if process-local)
    const char*     sidecar_dir;        // Optional read-only sidecar dir for legacy v4 indexes (frozen postings, rho_max)
} pm_open_opts_t;

typedef struct {
    uint32_t     field_id;
    uint32_t     n_vecs;
    const float* vecs_f32; // [n_vecs * d_f]
} pm_field_vecs_t;

typedef struct {
    uint32_t               n_fields;
    const pm_field_vecs_t* fields;
} pm_entity_vecs_t;

typedef struct {
    uint64_t n_docs;
    uint32_t n_fields;
    uint32_t dims[8];
    uint32_t codebooks[8];
    double   m_norm_max[8];
    uint64_t tier0_ram_bytes;
    uint64_t b1_ram_bytes;
    uint64_t b4_ram_bytes;
    uint64_t cache_ram_bytes;
    uint64_t calibrated_tau1;
    double   calibrated_alpha_k0_ns;
    int      active_tier_mode;
    int      amx_enabled;
    int      io_backend_active;
} pm_index_info_t;

// 4-Call Core C ABI (+ info helper)
pm_handle_t* pm_open(const char* index_dir, const pm_open_opts_t* opts);

int pm_execute_query(
    pm_handle_t*            h,
    const pm_bitmap_view_t* filter_bm,
    const pm_aux_scores_t*  aux_scores,
    const pm_query_spec_t*  qspec,
    pm_hit_t*               topk_out,
    uint32_t*               out_count,
    pm_emit_cb_t            thresh_cb,
    void*                   thresh_ctx,
    pm_query_stats_t*       stats);

int  pm_insert_entity(pm_handle_t* h, uint32_t ext_id, const pm_entity_vecs_t* vecs);
int  pm_delete_entity(pm_handle_t* h, uint32_t ext_id);
int  pm_compact_delta(pm_handle_t* h);
int  pm_get_info(const pm_handle_t* h, pm_index_info_t* info);
void pm_close(pm_handle_t* h);

// Utility: compute required shared memory size for Unified 4 KiB S3-FIFO Cache of `n_pages` 4 KiB pages
size_t pm_s3fifo_required_shmem_bytes(size_t cache_data_bytes);

#ifdef __cplusplus
}
#endif

#endif // PM_CERTVEC_H
