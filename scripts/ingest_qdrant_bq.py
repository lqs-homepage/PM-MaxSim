#!/usr/bin/env python3
"""
Non-timing ingestion of unmodified Qdrant v1.20.2 with Binary Quantization (BQ, always_ram=True) + On-Disk MaxSim
on exact 1:1 matched lotte_200k and vidore_v3_am corpora.
"""
import json
import os
import struct
import subprocess
import time
from pathlib import Path
import numpy as np

P09 = Path(os.environ.get("P09_ROOT", "./workspace"))
TP = P09 / "third_party"
EXT_ROOT = P09 / "stage3" / "ext_dbs"


def load_index_vectors_and_meta(idx_dir: Path, max_docs: int):
    with open(idx_dir / "core_index.bin", "rb") as f:
        hdr = struct.unpack("<12Q", f.read(96))
        N_docs, K, d, Nq_tot, Lq = hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]
        f.seek(K * d * 4 + d * d * 4, os.SEEK_CUR)
        Q32 = np.frombuffer(f.read(Nq_tot * Lq * d * 4), dtype=np.float32).reshape(Nq_tot, Lq, d).copy()
        f.seek(Nq_tot * Lq * d * 4, os.SEEK_CUR)
        doms = np.frombuffer(f.read(N_docs), dtype=np.uint8).copy()
    assert int(doms.max()) <= 10, f"Unexpected domain value {doms.max()}!"
    dt = np.dtype([("off", "<u8"), ("ab", "<u4"), ("nt", "<u4")])
    meta = np.fromfile(idx_dir / "tier2_meta.bin", dtype=dt)
    n_use = min(N_docs, max_docs)
    return n_use, doms[:n_use], meta[:n_use], Q32, idx_dir / "tier2_fp16.nvme"


def iter_doc_fp16(nvme_path: Path, meta_arr, start: int, end: int):
    with open(nvme_path, "rb") as f:
        for did in range(start, end):
            off = int(meta_arr[did]["off"])
            nt = int(meta_arr[did]["nt"])
            f.seek(off)
            raw = f.read(nt * 128 * 2)
            yield did, np.frombuffer(raw, dtype=np.float16).reshape(nt, 128).astype(np.float32)


def ingest_qdrant_bq(tag: str, idx_dir: Path, max_docs: int):
    from qdrant_client import QdrantClient, models

    q_data = EXT_ROOT / f"qdrant_bq_{tag}"
    if (q_data / "INGEST_DONE.json").exists():
        print(f"[Qdrant BQ {tag}] Already ingested.", flush=True)
        return
    if q_data.exists():
        subprocess.run(["rm", "-rf", str(q_data)], check=True)
    q_data.mkdir(parents=True)

    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(q_data / "storage")
    env["QDRANT__SERVICE__HTTP_PORT"] = "56345"
    env["QDRANT__SERVICE__GRPC_PORT"] = "56346"
    log_fh = open(q_data / "q.log", "w")
    proc = subprocess.Popen([str(TP / "qdrant" / "qdrant")], env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        client = QdrantClient(host="127.0.0.1", port=56345, grpc_port=56346, prefer_grpc=True, timeout=120)
        for _ in range(40):
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
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
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
            if (s // BATCH) % 50 == 0 or e == n_use:
                print(f"  [Qdrant BQ {tag}] upserted {e:,}/{n_use:,} docs ({time.time()-t0:.1f}s)", flush=True)

        for _ in range(300):
            info = client.get_collection("docs")
            if str(info.status).lower().endswith("green"):
                break
            time.sleep(1.0)

        q0 = Q32[0]
        q_valid = q0[np.linalg.norm(q0, axis=1) > 1e-4].tolist()
        res0 = client.query_points(
            collection_name="docs",
            query=q_valid,
            query_filter=models.Filter(must=[models.FieldCondition(key="dom", match=models.MatchValue(value=0))]),
            limit=10,
        )
        res1 = client.query_points(
            collection_name="docs",
            query=q_valid,
            query_filter=models.Filter(must=[models.FieldCondition(key="dom", match=models.MatchValue(value=1))]),
            limit=10,
        )
        assert len(res0.points) == 10 and len(res1.points) == 10, f"Smoke failed: dom0={len(res0.points)}, dom1={len(res1.points)}"
        disk_out = subprocess.check_output(["du", "-sh", str(q_data)]).decode().split()[0]
        tot_s = round(time.time() - t0, 1)
        print(f"  [Qdrant BQ {tag}] READY in {tot_s}s | disk={disk_out} | smoke dom0={len(res0.points)} dom1={len(res1.points)}", flush=True)
        with open(q_data / "INGEST_DONE.json", "w") as f:
            json.dump({"db": "qdrant_1.20.2_bq_ondisk", "tag": tag, "n_docs": int(n_use), "total_s": tot_s, "disk": disk_out, "smoke_hits": len(res0.points)}, f, indent=2)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log_fh.close()


if __name__ == "__main__":
    ingest_qdrant_bq("vidore_v3_am", P09 / "stage3" / "indexes" / "vidore_v3_adamerge", 20000)
    ingest_qdrant_bq("lotte_200k", P09 / "stage3" / "indexes" / "lotte_200k_matched", 200000)
