-- pg_certvec--1.0.sql: Native PostgreSQL 17 SQL Surface for PM-MaxSim / CertVec

CREATE FUNCTION pm_roaring_agg_trans(internal, int4)
RETURNS internal AS 'MODULE_PATHNAME', 'pm_roaring_agg_trans'
LANGUAGE C;

CREATE FUNCTION pm_roaring_agg_final(internal)
RETURNS bytea AS 'MODULE_PATHNAME', 'pm_roaring_agg_final'
LANGUAGE C;

CREATE AGGREGATE pm_roaring_agg(int4) (
    SFUNC = pm_roaring_agg_trans,
    STYPE = internal,
    FINALFUNC = pm_roaring_agg_final
);

CREATE FUNCTION pm_aux_agg_trans(internal, int4, float4)
RETURNS internal AS 'MODULE_PATHNAME', 'pm_aux_agg_trans'
LANGUAGE C;

CREATE FUNCTION pm_aux_agg_final(internal)
RETURNS bytea AS 'MODULE_PATHNAME', 'pm_aux_agg_final'
LANGUAGE C;

CREATE AGGREGATE pm_aux_agg(int4, float4) (
    SFUNC = pm_aux_agg_trans,
    STYPE = internal,
    FINALFUNC = pm_aux_agg_final
);

CREATE FUNCTION pm_certvec_topk(
    index_dir text,
    sidecar_dir text,
    filter_roaring bytea,
    query_f32 bytea,
    L int4,
    k int4,
    delta float8 DEFAULT 0.05,
    nprobe int4 DEFAULT 16,
    c1_pool int4 DEFAULT 600
) RETURNS TABLE(doc_id int4, score float8)
AS 'MODULE_PATHNAME', 'pm_certvec_topk'
LANGUAGE C VOLATILE;

CREATE FUNCTION pm_certvec_thresh(
    index_dir text,
    sidecar_dir text,
    filter_roaring bytea,
    query_f32 bytea,
    L int4,
    tau float8,
    delta float8 DEFAULT 0.05,
    nprobe int4 DEFAULT 16
) RETURNS TABLE(doc_id int4, score float8)
AS 'MODULE_PATHNAME', 'pm_certvec_thresh'
LANGUAGE C VOLATILE;

CREATE FUNCTION pm_certvec_hybrid_topk(
    index_dir text,
    sidecar_dir text,
    filter_roaring bytea,
    aux_pairs bytea,
    w0 float4,
    query_f32 bytea,
    L int4,
    k int4,
    delta float8 DEFAULT 0.05,
    nprobe int4 DEFAULT 16,
    c1_pool int4 DEFAULT 600,
    field_ids int4[] DEFAULT NULL,
    set_ops int4[] DEFAULT NULL,
    r_vals int4[] DEFAULT NULL,
    weights float4[] DEFAULT NULL
) RETURNS TABLE(doc_id int4, score float8)
AS 'MODULE_PATHNAME', 'pm_certvec_hybrid_topk'
LANGUAGE C VOLATILE;

CREATE FUNCTION pm_certvec_info(index_dir text, sidecar_dir text DEFAULT NULL)
RETURNS text AS 'MODULE_PATHNAME', 'pm_certvec_info'
LANGUAGE C VOLATILE;
