#!/usr/bin/env python3
"""
Layer 3 Milestone 3D: Non-Root Ingestion of New Large Out-of-Core Corpora Into Unmodified Production DBs
  1. Corpus #5 (`msmarco_docs_768d`, 300K parent docs x ~6M 768D chunks + title + real metadata):
     - Qdrant v1.20.2 (INT8 & BQ On-Disk, chunk-level HNSW + payload index on `doc_id` for native `group_by`!)
     - Weaviate v1.41.0 (RQ HNSW, chunk-level + native `group_by` on `doc_id`!)
     - PostgreSQL 17.12 + VectorChord 1.5.0 (`halfvec(768)` vchordrq RQ + SQL `DISTINCT ON (doc_id)`)
  2. Corpus #6 (`bgem3_1024d_hybrid`, 150K docs x 11.73M 1024D ColBERT tokens + 1024D Dense + Native Sparse):
     - Qdrant v1.20.2 (BQ On-Disk + INT8 On-Disk: native MultiVector `size=1024, MAX_SIM` + `dense` + `sparse` Hybrid!)
     - PostgreSQL 17.12 + VectorChord 1.5.0 (`halfvec(1024)[]` `halfvec_maxsim_ops` + `dense halfvec(1024)`)
All instances live under `$P09_ROOT/stage3_v4/ext_dbs/` (zero touch on frozen `stage3/ext_dbs/`).
"""
import argparse
import json
import os
import struct
import subprocess
import time
from pathlib import Path
import numpy as np

P09 = Path(os.environ.get("P09_ROOT", "${P09_ROOT}"))
TP = P09 / "third_party"
EXT_ROOT = P09 / "stage3_v4" / "ext_dbs"
EXT_ROOT.mkdir(parents=True, exist_ok=True)


def iter_field_fp16(field_dir: Path, dim: int, start_doc: int, end_doc: int):
    meta = np.fromfile(field_dir / "tier2_meta.bin", dtype=np.dtype([("off", "<u8"), ("ab", "<u4"), ("nt", "<u4")]))
    with open(field_dir / "tier2_fp16.nvme", "rb") as f:
        for i in range(start_doc, end_doc):
            off = int(meta[i]["off"])
            nt = int(meta[i]["nt"])
            f.seek(off)
            mat = np.frombuffer(f.read(nt * dim * 2), dtype=np.float16).reshape(nt, dim).astype(np.float32)
            yield i, mat


def ingest_qdrant_parentdoc_768d(idx_dir: Path, quant: str = "bq", max_docs: int = 0):
    from qdrant_client import QdrantClient, models

    q_data = EXT_ROOT / f"qdrant_{quant}_msmarco_docs_768d"
    if (q_data / "INGEST_DONE.json").exists():
        print(f"[Qdrant {quant} docs768] Already done.", flush=True)
        return
    if q_data.exists():
        subprocess.run(["rm", "-rf", str(q_data)], check=True)
    q_data.mkdir(parents=True)

    http_p, grpc_p = (56413, 56414) if quant == "bq" else (56415, 56416)
    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(q_data / "storage")
    env["QDRANT__SERVICE__HTTP_PORT"] = str(http_p)
    env["QDRANT__SERVICE__GRPC_PORT"] = str(grpc_p)
    proc = subprocess.Popen([str(TP / "qdrant" / "qdrant")], env=env, stdout=open(q_data / "q.log", "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        client = QdrantClient(host="127.0.0.1", port=http_p, grpc_port=grpc_p, prefer_grpc=True, timeout=180)
        for _ in range(40):
            try:
                client.get_collections()
                break
            except Exception:
                time.sleep(0.5)

        dom_ids = np.load(idx_dir / "dom_ids.npy")
        word_lens = np.load(idx_dir / "doc_word_lens.npy")
        N = len(dom_ids) if max_docs <= 0 else min(len(dom_ids), max_docs)
        gt5 = np.load(idx_dir / "gt_NoCorr_5pct.npz")["mask"]

        qcfg = (
            models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))
            if quant == "bq"
            else models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True))
        )
        client.create_collection(
            collection_name="chunks",
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.DOT, on_disk=True,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=False),
            ),
            quantization_config=qcfg,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=50000),
        )
        client.create_payload_index("chunks", "doc_id", models.PayloadSchemaType.INTEGER)
        client.create_payload_index("chunks", "dom", models.PayloadSchemaType.INTEGER)
        client.create_payload_index("chunks", "word_len", models.PayloadSchemaType.INTEGER)
        client.create_payload_index("chunks", "nc5", models.PayloadSchemaType.BOOL)

        chunk_uid = 0
        pts = []
        BATCH_PTS = 2048
        for did, mat in iter_field_fp16(idx_dir / "field_0", 768, 0, N):
            pay = {"doc_id": int(did), "dom": int(dom_ids[did]), "word_len": int(word_lens[did]), "nc5": bool(gt5[did])}
            for row in mat:
                pts.append(models.PointStruct(id=chunk_uid, vector=row.tolist(), payload=pay))
                chunk_uid += 1
                if len(pts) >= BATCH_PTS:
                    client.upsert(collection_name="chunks", points=pts, wait=False)
                    pts.clear()
            if (did + 1) % 25000 == 0 or (did + 1) == N:
                print(f"  [Qdrant {quant} docs768] docs={did+1:,}/{N:,} chunks={chunk_uid:,} ({time.time()-t0:.1f}s)", flush=True)
        if pts:
            client.upsert(collection_name="chunks", points=pts, wait=True)

        # Wait green
        for _ in range(600):
            if str(client.get_collection("chunks").status).lower().endswith("green"):
                break
            time.sleep(1.5)

        q0 = np.load(idx_dir / "queries_f32.npy")[0].tolist()
        res = client.query_points_groups(
            collection_name="chunks",
            query=q0,
            group_by="doc_id",
            limit=10,
            group_size=1,
            query_filter=models.Filter(must=[models.FieldCondition(key="nc5", match=models.MatchValue(value=True))]),
        )
        print(f"  [Qdrant {quant} docs768] READY in {time.time()-t0:.1f}s | group_by smoke groups={len(res.groups)}", flush=True)
        with open(q_data / "INGEST_DONE.json", "w") as f:
            json.dump({"db": f"qdrant_1.20.2_{quant}", "corpus": "msmarco_docs_768d", "n_docs": N, "n_chunks": chunk_uid, "total_s": round(time.time() - t0, 1)}, f, indent=2)
    finally:
        proc.terminate()
        proc.wait(timeout=45)


def ingest_qdrant_bgem3_1024d(idx_dir: Path, quant: str = "bq", max_docs: int = 0):
    from qdrant_client import QdrantClient, models

    q_data = EXT_ROOT / f"qdrant_{quant}_bgem3_1024d"
    if (q_data / "INGEST_DONE.json").exists():
        print(f"[Qdrant {quant} bgem3] Already done.", flush=True)
        return
    if q_data.exists():
        subprocess.run(["rm", "-rf", str(q_data)], check=True)
    q_data.mkdir(parents=True)

    http_p, grpc_p = (56423, 56424) if quant == "bq" else (56425, 56426)
    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(q_data / "storage")
    env["QDRANT__SERVICE__HTTP_PORT"] = str(http_p)
    env["QDRANT__SERVICE__GRPC_PORT"] = str(grpc_p)
    proc = subprocess.Popen([str(TP / "qdrant" / "qdrant")], env=env, stdout=open(q_data / "q.log", "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        client = QdrantClient(host="127.0.0.1", port=http_p, grpc_port=grpc_p, prefer_grpc=True, timeout=180)
        for _ in range(40):
            try:
                client.get_collections()
                break
            except Exception:
                time.sleep(0.5)

        topics = np.load(idx_dir / "doc_topics.npy")
        char_lens = np.load(idx_dir / "doc_char_lens.npy")
        N = len(topics) if max_docs <= 0 else min(len(topics), max_docs)
        gt5 = np.load(idx_dir / "gt_NoCorr_5pct.npz")["mask"]

        qcfg = (
            models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))
            if quant == "bq"
            else models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True))
        )
        client.create_collection(
            collection_name="docs",
            vectors_config={
                "colbert": models.VectorParams(
                    size=1024, distance=models.Distance.DOT, on_disk=True,
                    multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=False),
                    quantization_config=qcfg,
                ),
                "dense": models.VectorParams(
                    size=1024, distance=models.Distance.DOT, on_disk=True,
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=False),
                    quantization_config=qcfg,
                ),
            },
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        client.create_payload_index("docs", "topic", models.PayloadSchemaType.INTEGER)
        client.create_payload_index("docs", "char_len", models.PayloadSchemaType.INTEGER)
        client.create_payload_index("docs", "nc5", models.PayloadSchemaType.BOOL)

        f1_iter = iter_field_fp16(idx_dir / "field_1", 1024, 0, N)
        pts = []
        BATCH = 64
        for (did, colb_mat), (did1, dense_mat) in zip(iter_field_fp16(idx_dir / "field_0", 1024, 0, N), f1_iter):
            assert did == did1
            pts.append(models.PointStruct(
                id=int(did),
                vector={"colbert": colb_mat.tolist(), "dense": dense_mat[0].tolist()},
                payload={"topic": int(topics[did]), "char_len": int(char_lens[did]), "nc5": bool(gt5[did])},
            ))
            if len(pts) >= BATCH:
                client.upsert(collection_name="docs", points=pts, wait=False)
                pts.clear()
            if (did + 1) % 10000 == 0 or (did + 1) == N:
                print(f"  [Qdrant {quant} bgem3] docs={did+1:,}/{N:,} ({time.time()-t0:.1f}s)", flush=True)
        if pts:
            client.upsert(collection_name="docs", points=pts, wait=True)

        for _ in range(600):
            if str(client.get_collection("docs").status).lower().endswith("green"):
                break
            time.sleep(1.5)

        q0_c = np.load(idx_dir / "q_colb_f32.npy")[0]
        q0_len = int(np.load(idx_dir / "q_lens.npy")[0])
        res = client.query_points(
            collection_name="docs", using="colbert", query=q0_c[:q0_len].tolist(),
            query_filter=models.Filter(must=[models.FieldCondition(key="nc5", match=models.MatchValue(value=True))]),
            limit=10,
        )
        print(f"  [Qdrant {quant} bgem3] READY in {time.time()-t0:.1f}s | smoke hits={len(res.points)}", flush=True)
        with open(q_data / "INGEST_DONE.json", "w") as f:
            json.dump({"db": f"qdrant_1.20.2_{quant}", "corpus": "bgem3_1024d_hybrid", "n_docs": N, "total_s": round(time.time() - t0, 1)}, f, indent=2)
    finally:
        proc.terminate()
        proc.wait(timeout=45)


def ingest_pg_vchord_parentdoc_768d(idx_dir: Path, max_docs: int = 0):
    import psycopg

    pg_data = EXT_ROOT / "pg17_vchord_rq_msmarco_docs_768d"
    if (pg_data / "INGEST_DONE.json").exists():
        print("[VectorChord docs768] Already done.", flush=True)
        return
    if pg_data.exists():
        subprocess.run(["rm", "-rf", str(pg_data)], check=True)
    pg_data.mkdir(parents=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{TP}/pg17_vchord/inst/lib:{P09}/stage3_v4/src:" + env.get("LD_LIBRARY_PATH", "")
    pg_bin = TP / "pg17_vchord" / "inst" / "bin"
    port = 55451
    sock_dir = EXT_ROOT / "sock_docs768"
    sock_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([str(pg_bin / "initdb"), "-D", str(pg_data), "-U", "postgres", "--auth=trust"], env=env, check=True, stdout=subprocess.DEVNULL)
    with open(pg_data / "postgresql.conf", "a") as f:
        f.write(
            f"dynamic_library_path = '$libdir:{TP}/pg17_vchord/inst/lib/postgresql:{TP}/pg17_vchord/inst/lib'\n"
            f"shared_preload_libraries = 'vchord.so,pg_certvec.so'\n"
            f"pg_certvec.cache_mb = 512\n"
            f"port = {port}\n"
            f"unix_socket_directories = '{sock_dir}'\n"
            f"shared_buffers = '4GB'\n"
            f"maintenance_work_mem = '8GB'\n"
            f"max_parallel_maintenance_workers = 16\n"
            f"max_worker_processes = 32\n"
            f"max_connections = 80\n"
        )
    subprocess.run([str(pg_bin / "pg_ctl"), "-D", str(pg_data), "-l", str(pg_data / "pg.log"), "start"], env=env, check=True)
    t0 = time.time()
    try:
        dom_ids = np.load(idx_dir / "dom_ids.npy")
        word_lens = np.load(idx_dir / "doc_word_lens.npy")
        N = len(dom_ids) if max_docs <= 0 else min(len(dom_ids), max_docs)
        gt5 = np.load(idx_dir / "gt_NoCorr_5pct.npz")["mask"]

        with psycopg.connect(f"host={sock_dir} port={port} user=postgres dbname=postgres", autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("CREATE EXTENSION IF NOT EXISTS vchord")
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_certvec")
            # Parent relational table (used by BOTH pg_certvec AND SQL JOINs!)
            conn.execute("CREATE TABLE docs (doc_id INT PRIMARY KEY, dom SMALLINT, word_len INT, nc5 BOOLEAN)")
            with conn.cursor().copy("COPY docs (doc_id, dom, word_len, nc5) FROM STDIN") as cp:
                for did in range(N):
                    cp.write_row((did, int(dom_ids[did]), int(word_lens[did]), bool(gt5[did])))
            conn.execute("CREATE INDEX idx_docs_dom ON docs(dom)")
            conn.execute("CREATE INDEX idx_docs_wlen ON docs(word_len)")
            conn.execute("CREATE INDEX idx_docs_nc5 ON docs(nc5)")

            # Chunk table for VectorChord single-vector RQ + DISTINCT ON / GROUP BY baseline!
            conn.execute("CREATE TABLE chunks (chunk_id INT PRIMARY KEY, doc_id INT, dom SMALLINT, word_len INT, nc5 BOOLEAN, emb halfvec(768))")
            cid_seq = 0
            BATCH_DOCS = 2000
            for s in range(0, N, BATCH_DOCS):
                e = min(N, s + BATCH_DOCS)
                with conn.cursor().copy("COPY chunks (chunk_id, doc_id, dom, word_len, nc5, emb) FROM STDIN") as cp:
                    for did, mat in iter_field_fp16(idx_dir / "field_0", 768, s, e):
                        d_i, wl_i, nc_b = int(dom_ids[did]), int(word_lens[did]), bool(gt5[did])
                        for row in mat:
                            vec_s = "[" + ",".join(f"{x:.4f}" for x in row) + "]"
                            cp.write_row((cid_seq, did, d_i, wl_i, nc_b, vec_s))
                            cid_seq += 1
                if (s // BATCH_DOCS) % 20 == 0 or e == N:
                    print(f"  [PG17+VChord docs768] loaded {e:,}/{N:,} docs ({cid_seq:,} chunks, {time.time()-t0:.1f}s)", flush=True)

            t_idx = time.time()
            conn.execute("CREATE INDEX idx_chunks_dom ON chunks(dom)")
            conn.execute("CREATE INDEX idx_chunks_nc5 ON chunks(nc5)")
            conn.execute("CREATE INDEX idx_chunks_wlen ON chunks(word_len)")
            opts = """
            residual_quantization = true
            [build.internal]
            lists = [8192]
            spherical_centroids = true
            build_threads = 24
            """
            conn.execute(f"CREATE INDEX idx_chunks_emb ON chunks USING vchordrq (emb halfvec_ip_ops) WITH (options = $${opts}$$)")
            conn.execute("ANALYZE docs; ANALYZE chunks;")
            idx_s = time.time() - t_idx
            print(f"  [PG17+VChord docs768] READY idx_s={idx_s:.1f}s total_s={time.time()-t0:.1f}s chunks={cid_seq:,}", flush=True)
            with open(pg_data / "INGEST_DONE.json", "w") as f:
                json.dump({"db": "pg17_vchord_rq_and_certvec", "corpus": "msmarco_docs_768d", "n_docs": N, "n_chunks": cid_seq, "idx_s": round(idx_s, 1), "total_s": round(time.time() - t0, 1)}, f, indent=2)
    finally:
        subprocess.run([str(pg_bin / "pg_ctl"), "-D", str(pg_data), "stop"], env=env, check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qdrant_bq_docs", "qdrant_int8_docs", "pg_docs", "qdrant_bq_bgem3", "qdrant_int8_bgem3", "all"], required=True)
    ap.add_argument("--max-docs", type=int, default=0)
    args = ap.parse_args()
    docs_dir = P09 / "stage3_v4" / "indexes" / "msmarco_docs_768d"
    bgem3_dir = P09 / "stage3_v4" / "indexes" / "bgem3_1024d_hybrid"
    if args.task in ("qdrant_bq_docs", "all"):
        ingest_qdrant_parentdoc_768d(docs_dir, "bq", args.max_docs)
    if args.task in ("qdrant_int8_docs", "all"):
        ingest_qdrant_parentdoc_768d(docs_dir, "int8", args.max_docs)
    if args.task in ("pg_docs", "all"):
        ingest_pg_vchord_parentdoc_768d(docs_dir, args.max_docs)
    if args.task in ("qdrant_bq_bgem3", "all"):
        ingest_qdrant_bgem3_1024d(bgem3_dir, "bq", args.max_docs)
    if args.task in ("qdrant_int8_bgem3", "all"):
        ingest_qdrant_bgem3_1024d(bgem3_dir, "int8", args.max_docs)
