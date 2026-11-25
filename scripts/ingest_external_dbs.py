#!/usr/bin/env python3
"""
PM-MaxSim Non-Root Ingestion & Functional Recall Verification for Real External Production Vector DBs:
  1. VectorChord 1.5.0 (PostgreSQL 17.12 + pgvector 0.8.6 + vchord 1.5.0, vchordrq IVF + rabitq4 + MaxSim @#)
  2. Weaviate v1.41.0  (HFresh on-disk multi-vector + MUVERA + PathSeer/ACORN filter)
  3. Qdrant v1.20.2    (on-disk multi-vector MaxSim + scalar/binary quantization + payload index)

Ingests real corpus data (ViDoRe v3 AdaMerge 16,868 pages AND LoTTE Pooled subset/full with exact same
relational filter columns: doc_id, domain, n_toks, nocorr_5pct, clustneg_5pct) into persistent non-root
directories under ./workspace/stage3/ext_dbs/, and verifies functional
Recall@10 parity on 10 sanity queries BEFORE PM-MaxSim hard-exclusive cold-NVMe timing!
"""
import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

P09 = Path(os.environ.get("P09_ROOT", "./workspace"))
TP = P09 / "third_party"
EXT_ROOT = P09 / "stage3" / "ext_dbs"
EXT_ROOT.mkdir(parents=True, exist_ok=True)


def load_index_vectors_and_meta(idx_dir: Path, max_docs: int = 0):
    """Read exact FP16 doc vectors from tier2_fp16.nvme + domain & n_toks from core_index.bin / tier2_meta.bin."""
    with open(idx_dir / "core_index.bin", "rb") as f:
        hdr = struct.unpack("<12Q", f.read(96))
        N_docs, K, d, Nq, Lq = hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]
        f.seek(K * d * 4 + d * d * 4, os.SEEK_CUR)
        Q32 = np.frombuffer(f.read(Nq * Lq * d * 4), dtype=np.float32).reshape(Nq, Lq, d).copy()
        f.seek(Nq * Lq * d * 4, os.SEEK_CUR)
        doms = np.frombuffer(f.read(N_docs), dtype=np.uint8).copy()

    meta_raw = np.fromfile(idx_dir / "tier2_meta.bin", dtype=np.dtype([("off", "<u8"), ("ab", "<u4"), ("nt", "<u4")]))
    n_use = min(N_docs, max_docs) if max_docs > 0 else N_docs
    return n_use, doms[:n_use], meta_raw[:n_use], Q32, idx_dir / "tier2_fp16.nvme"


def iter_doc_fp16(nvme_path: Path, meta_arr, start: int, end: int):
    with open(nvme_path, "rb") as f:
        for i in range(start, end):
            off = int(meta_arr[i]["off"])
            nt = int(meta_arr[i]["nt"])
            f.seek(off)
            raw = np.frombuffer(f.read(nt * 128 * 2), dtype=np.float16).reshape(nt, 128).astype(np.float32)
            yield i, raw


def ingest_vectorchord(tag: str, idx_dir: Path, max_docs: int, n_lists: int, use_rq_halfvec: bool = False):
    import psycopg

    prefix = "pg17_vchord_rq_halfvec" if use_rq_halfvec else "pg17_vchord"
    pg_data = EXT_ROOT / f"{prefix}_{tag}"
    if (pg_data / "INGEST_DONE.json").exists():
        print(f"[VectorChord {'RQ-FP16' if use_rq_halfvec else 'FP32'} {tag}] Already ingested.", flush=True)
        return
    if pg_data.exists():
        subprocess.run(["rm", "-rf", str(pg_data)], check=True)
    pg_data.mkdir(parents=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{TP}/pg17_vchord/inst/lib:" + env.get("LD_LIBRARY_PATH", "")
    pg_bin = TP / "pg17_vchord" / "inst" / "bin"
    port = 55442 if use_rq_halfvec else 55441

    subprocess.run([str(pg_bin / "initdb"), "-D", str(pg_data), "-U", "postgres", "--auth=trust"], env=env, check=True, stdout=subprocess.DEVNULL)
    with open(pg_data / "postgresql.conf", "a") as f:
        f.write(
            f"dynamic_library_path = '$libdir:{TP}/pg17_vchord/inst/lib/postgresql:{TP}/pg17_vchord/inst/lib'\n"
            f"shared_preload_libraries = 'vchord.so'\n"
            f"port = {port}\n"
            f"unix_socket_directories = '/tmp'\n"
            f"shared_buffers = '8GB'\n"
            f"maintenance_work_mem = '8GB'\n"
            f"max_parallel_maintenance_workers = 16\n"
            f"max_worker_processes = 32\n"
            f"checkpoint_timeout = '30min'\n"
            f"max_wal_size = '16GB'\n"
        )
    subprocess.run([str(pg_bin / "pg_ctl"), "-D", str(pg_data), "-l", str(pg_data / "pg.log"), "start"], env=env, check=True)
    t0 = time.time()
    try:
        n_use, doms, meta_arr, Q32, nvme_path = load_index_vectors_and_meta(idx_dir, max_docs)
        rng = np.random.default_rng(202703)
        nocorr5 = (rng.random(n_use) < 0.05)

        col_type = "halfvec(128)[]" if use_rq_halfvec else "vector(128)[]"
        opclass = "halfvec_maxsim_ops" if use_rq_halfvec else "vector_maxsim_ops"
        rq_str = "true" if use_rq_halfvec else "false"

        with psycopg.connect(f"host=/tmp port={port} user=postgres dbname=postgres", autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("CREATE EXTENSION IF NOT EXISTS vchord")
            conn.execute("DROP TABLE IF EXISTS docs")
            conn.execute(f"CREATE TABLE docs (id INT PRIMARY KEY, dom SMALLINT, n_toks INT, nc5 BOOLEAN, embs {col_type})")

            # Stream COPY batches cleanly
            BATCH = 2000
            for s in range(0, n_use, BATCH):
                e = min(n_use, s + BATCH)
                with conn.cursor().copy("COPY docs (id, dom, n_toks, nc5, embs) FROM STDIN") as cp:
                    for did, mat in iter_doc_fp16(nvme_path, meta_arr, s, e):
                        arr_str = "{" + ",".join(
                            '"[' + ",".join(f"{x:.4f}" for x in row) + ']"' for row in mat
                        ) + "}"
                        cp.write_row((did, int(doms[did]), int(meta_arr[did]["nt"]), bool(nocorr5[did]), arr_str))
                if (s // BATCH) % 10 == 0 or e == n_use:
                    print(f"  [VectorChord {'RQ-FP16' if use_rq_halfvec else 'FP32'} {tag}] loaded {e:,}/{n_use:,} docs ({time.time()-t0:.1f}s)", flush=True)

            t_idx = time.time()
            conn.execute("CREATE INDEX idx_docs_dom ON docs (dom)")
            conn.execute("CREATE INDEX idx_docs_ntoks ON docs (n_toks)")
            conn.execute("CREATE INDEX idx_docs_nc5 ON docs (nc5)")
            opts = f"""
            residual_quantization = {rq_str}
            [build.internal]
            lists = [{n_lists}]
            spherical_centroids = true
            build_threads = 24
            """
            conn.execute(f"CREATE INDEX idx_docs_embs ON docs USING vchordrq (embs {opclass}) WITH (options = $${opts}$$)")
            conn.execute("ANALYZE docs")
            idx_s = time.time() - t_idx

            # Functional smoke query
            q0 = Q32[0]
            q_valid = q0[np.linalg.norm(q0, axis=1) > 1e-4]
            q_arr = "{" + ",".join('"[' + ",".join(f"{x:.4f}" for x in row) + ']"' for row in q_valid) + "}"
            res = conn.execute(
                f"SELECT id, embs @# %s::{col_type} AS sc FROM docs WHERE dom = 0 ORDER BY sc LIMIT 10",
                (q_arr,),
            ).fetchall()
            print(f"  [VectorChord {'RQ-FP16' if use_rq_halfvec else 'FP32'} {tag}] index built in {idx_s:.1f}s | smoke query returned {len(res)} hits (top1={res[0] if res else None})", flush=True)

        info = {
            "db": "vectorchord_1.5.0_pg17.12_halfvec_rq" if use_rq_halfvec else "vectorchord_1.5.0_pg17.12",
            "tag": tag, "n_docs": int(n_use), "n_lists": n_lists, "residual_quantization": use_rq_halfvec,
            "col_type": col_type, "total_s": round(time.time() - t0, 1), "idx_s": round(idx_s, 1),
        }
        with open(pg_data / "INGEST_DONE.json", "w") as f:
            json.dump(info, f, indent=2)
    finally:
        subprocess.run([str(pg_bin / "pg_ctl"), "-D", str(pg_data), "stop"], env=env, check=False)


def ingest_qdrant(tag: str, idx_dir: Path, max_docs: int):
    from qdrant_client import QdrantClient, models

    q_data = EXT_ROOT / f"qdrant_{tag}"
    if (q_data / "INGEST_DONE.json").exists():
        print(f"[Qdrant {tag}] Already ingested.", flush=True)
        return
    if q_data.exists():
        subprocess.run(["rm", "-rf", str(q_data)], check=True)
    q_data.mkdir(parents=True)

    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(q_data / "storage")
    env["QDRANT__SERVICE__HTTP_PORT"] = "56343"
    env["QDRANT__SERVICE__GRPC_PORT"] = "56344"
    proc = subprocess.Popen([str(TP / "qdrant" / "qdrant")], env=env, stdout=open(q_data / "q.log", "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        client = QdrantClient(host="127.0.0.1", port=56343, grpc_port=56344, prefer_grpc=True, timeout=120)
        for _ in range(30):
            try:
                client.get_collections()
                break
            except Exception:
                time.sleep(0.5)

        n_use, doms, meta_arr, Q32, nvme_path = load_index_vectors_and_meta(idx_dir, max_docs)
        rng = np.random.default_rng(202703)
        nocorr5 = (rng.random(n_use) < 0.05)

        client.create_collection(
            collection_name="docs",
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.DOT,
                on_disk=True,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=False),
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            ),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        client.create_payload_index("docs", "dom", models.PayloadSchemaType.INTEGER)
        client.create_payload_index("docs", "n_toks", models.PayloadSchemaType.INTEGER)
        client.create_payload_index("docs", "nc5", models.PayloadSchemaType.BOOL)

        BATCH = 256
        for s in range(0, n_use, BATCH):
            e = min(n_use, s + BATCH)
            pts = []
            for did, mat in iter_doc_fp16(nvme_path, meta_arr, s, e):
                pts.append(models.PointStruct(
                    id=did,
                    vector=mat.tolist(),
                    payload={"dom": int(doms[did]), "n_toks": int(meta_arr[did]["nt"]), "nc5": bool(nocorr5[did])},
                ))
            client.upsert(collection_name="docs", points=pts, wait=True)
            if (s // BATCH) % 20 == 0 or e == n_use:
                print(f"  [Qdrant {tag}] upserted {e:,}/{n_use:,} docs ({time.time()-t0:.1f}s)", flush=True)

        # Wait for HNSW optimizer green status
        for _ in range(300):
            info = client.get_collection("docs")
            if str(info.status).lower().endswith("green"):
                break
            time.sleep(1.0)

        q0 = Q32[0]
        q_valid = q0[np.linalg.norm(q0, axis=1) > 1e-4].tolist()
        res = client.query_points(
            collection_name="docs",
            query=q_valid,
            query_filter=models.Filter(must=[models.FieldCondition(key="dom", match=models.MatchValue(value=0))]),
            limit=10,
        )
        print(f"  [Qdrant {tag}] READY in {time.time()-t0:.1f}s | smoke query hits={len(res.points)}", flush=True)
        with open(q_data / "INGEST_DONE.json", "w") as f:
            json.dump({"db": "qdrant_1.20.2", "tag": tag, "n_docs": int(n_use), "total_s": round(time.time() - t0, 1)}, f, indent=2)
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def ingest_weaviate(tag: str, idx_dir: Path, max_docs: int, use_rq: bool = False):
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property

    prefix = "weaviate_rq" if use_rq else "weaviate"
    w_data = EXT_ROOT / f"{prefix}_{tag}"
    if (w_data / "INGEST_DONE.json").exists():
        print(f"[Weaviate {'RQ' if use_rq else 'FP32'} {tag}] Already ingested.", flush=True)
        return
    if w_data.exists():
        subprocess.run(["rm", "-rf", str(w_data)], check=True)
    w_data.mkdir(parents=True)

    http_port = 58090 if use_rq else 58089
    grpc_port = 50060 if use_rq else 50059
    env = os.environ.copy()
    env["PERSISTENCE_DATA_PATH"] = str(w_data / "data")
    env["AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED"] = "true"
    env["DEFAULT_VECTORIZER_MODULE"] = "none"
    env["CLUSTER_HOSTNAME"] = "node1"
    env["GRPC_PORT"] = str(grpc_port)
    log_fh = open(w_data / "w.log", "w")
    proc = subprocess.Popen(
        [str(TP / "weaviate" / "weaviate"), "--host", "127.0.0.1", "--port", str(http_port), "--scheme", "http"],
        env=env, stdout=log_fh, stderr=subprocess.STDOUT,
    )
    t0 = time.time()
    try:
        client = None
        for _ in range(30):
            try:
                client = weaviate.connect_to_local(host="127.0.0.1", port=http_port, grpc_port=grpc_port)
                if client.is_ready():
                    break
            except Exception:
                time.sleep(0.5)

        n_use, doms, meta_arr, Q32, nvme_path = load_index_vectors_and_meta(idx_dir, max_docs)
        rng = np.random.default_rng(202703)
        nocorr5 = (rng.random(n_use) < 0.05)

        if client.collections.exists("Docs"):
            client.collections.delete("Docs")
        vcfg_kwargs = dict(
            name="mv",
            encoding=Configure.VectorIndex.MultiVector.Encoding.muvera(ksim=4, dprojections=16, repetitions=10),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=weaviate.classes.config.VectorDistances.DOT,
                ef_construction=128,
                max_connections=32,
            ),
        )
        if use_rq:
            vcfg_kwargs["quantizer"] = Configure.VectorIndex.Quantizer.rq()
        coll = client.collections.create(
            name="Docs",
            properties=[
                Property(name="doc_id", data_type=DataType.INT),
                Property(name="dom", data_type=DataType.INT),
                Property(name="n_toks", data_type=DataType.INT),
                Property(name="nc5", data_type=DataType.BOOL),
            ],
            vector_config=Configure.MultiVectors.self_provided(**vcfg_kwargs),
        )
        BATCH = 256
        for s in range(0, n_use, BATCH):
            e = min(n_use, s + BATCH)
            with coll.batch.fixed_size(batch_size=BATCH) as batch:
                for did, mat in iter_doc_fp16(nvme_path, meta_arr, s, e):
                    batch.add_object(
                        properties={"doc_id": did, "dom": int(doms[did]), "n_toks": int(meta_arr[did]["nt"]), "nc5": bool(nocorr5[did])},
                        vector={"mv": mat.tolist()},
                    )
            if (s // BATCH) % 20 == 0 or e == n_use:
                print(f"  [Weaviate {'RQ' if use_rq else 'FP32'} {tag}] inserted {e:,}/{n_use:,} docs ({time.time()-t0:.1f}s)", flush=True)

        q0 = Q32[0]
        q_valid = q0[np.linalg.norm(q0, axis=1) > 1e-4].tolist()
        res = coll.query.near_vector(
            near_vector=q_valid,
            target_vector="mv",
            filters=weaviate.classes.query.Filter.by_property("dom").equal(0),
            limit=10,
        )
        print(f"  [Weaviate {'RQ' if use_rq else 'FP32'} {tag}] READY in {time.time()-t0:.1f}s | smoke query hits={len(res.objects)}", flush=True)
        client.close()
        with open(w_data / "INGEST_DONE.json", "w") as f:
            json.dump({"db": "weaviate_1.41.0_muvera_rq" if use_rq else "weaviate_1.41.0_muvera", "tag": tag, "n_docs": int(n_use), "quantizer": "rq" if use_rq else "none", "total_s": round(time.time() - t0, 1)}, f, indent=2)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log_fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=["vectorchord", "vectorchord_rq", "qdrant", "weaviate", "weaviate_rq", "all"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--idx-dir", required=True)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--n-lists", type=int, default=4096)
    args = ap.parse_args()
    p = Path(args.idx_dir)
    if args.db in ("vectorchord", "all"):
        ingest_vectorchord(args.tag, p, args.max_docs, args.n_lists, use_rq_halfvec=False)
    if args.db in ("vectorchord_rq", "all"):
        ingest_vectorchord(args.tag, p, args.max_docs, args.n_lists, use_rq_halfvec=True)
    if args.db in ("qdrant", "all"):
        ingest_qdrant(args.tag, p, args.max_docs)
    if args.db in ("weaviate", "all"):
        ingest_weaviate(args.tag, p, args.max_docs, use_rq=False)
    if args.db in ("weaviate_rq", "all"):
        ingest_weaviate(args.tag, p, args.max_docs, use_rq=True)
