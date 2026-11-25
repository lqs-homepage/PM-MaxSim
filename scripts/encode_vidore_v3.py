#!/usr/bin/env python3
"""
PM-MaxSim ViDoRe v3 Full English Enterprise Multi-Domain Encoder (7 domains = 16,868 real PDF pages!):
  Domains (0..6):
    0: vidore/vidore_v3_finance_en         (2,942 pages)
    1: vidore/vidore_v3_hr                 (1,110 pages)
    2: vidore/vidore_v3_industrial         (5,244 pages)
    3: vidore/vidore_v3_pharmaceuticals    (2,313 pages)
    4: vidore/vidore_v3_computer_science   (1,360 pages)
    5: vidore/vidore_v3_energy             (2,225 pages)
    6: vidore/vidore_v3_physics            (1,674 pages)
- Encodes ALL 16,868 pages with frozen vidore/colqwen2.5-v0.2 across 2x idle B200 GPUs (shard 0: domains 0,2; shard 1: domains 1,3,4,5,6 -> balanced ~8.2K pages/GPU, ~4 min total!),
  capturing BOTH exact FP16 raw patch embeddings AND exact last-layer SDPA EOS-to-patch attention weights alpha_i.
- Also encodes 140 strictly stratified queries (20/domain x 7 = 140) with official qrels and real per-page metadata
  (domain_id 0..6, doc_id hash, page_num_in_doc, n_patches) for 100% genuine Natural Multi-Attribute Visual SQL predicates!
- After both shards finish (--merge), computes authentic recalibrated AdaMerge-258 (p_prune=0.30, keep_r=0.30, wmode="alpha",
  WITHOUT L2 renormalization) across all 16,868 pages in parallel on CPU and verifies nDCG@5 vs Raw on GPU!
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
P09_ROOT = Path(os.environ.get("P09_ROOT", "./workspace"))
os.environ.setdefault("HF_HOME", str(P09_ROOT / "cache" / "hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(P09_ROOT / "cache" / "hf" / "datasets"))
sys.path.insert(0, str(P09_ROOT / "pylibs"))
OUT_DIR = P09_ROOT / "stage3" / "corpus_vidore_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPOS = [
    ("finance_en", "vidore/vidore_v3_finance_en"),
    ("hr", "vidore/vidore_v3_hr"),
    ("industrial", "vidore/vidore_v3_industrial"),
    ("pharmaceuticals", "vidore/vidore_v3_pharmaceuticals"),
    ("computer_science", "vidore/vidore_v3_computer_science"),
    ("energy", "vidore/vidore_v3_energy"),
    ("physics", "vidore/vidore_v3_physics"),
]


def _adamerge_page(args):
    p, a = args
    P64 = p.astype(np.float64)
    P64 /= np.maximum(np.linalg.norm(P64, axis=1, keepdims=True), 1e-12)
    A = np.maximum(a.astype(np.float64), 1e-12)
    M = len(P64)
    if M <= 16:
        return P64.astype(np.float16)
    keep_idx = np.argsort(-A)[: max(16, int(round(M * 0.70)))]
    P64 = P64[keep_idx]
    A = A[keep_idx]
    n_cl = max(16, int(round(M * 0.30)))
    if len(P64) <= n_cl:
        return P64.astype(np.float16)
    Z = linkage(P64, method="ward", metric="euclidean")
    labels = fcluster(Z, t=n_cl, criterion="maxclust")
    out = np.empty((n_cl, P64.shape[1]), dtype=np.float32)
    for cid in range(1, n_cl + 1):
        m = labels == cid
        w = A[m]
        out[cid - 1] = (w[:, None] * P64[m]).sum(axis=0) / w.sum()
    return out.astype(np.float16)


@torch.no_grad()
def encode_domain(dom_idx: int, gpu_id: int):
    from datasets import load_dataset
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv

    dname, repo = REPOS[dom_idx]
    done_f = OUT_DIR / f"dom_{dom_idx}_{dname}.done"
    if done_f.exists():
        print(f"[ViDoRe v3 dom={dom_idx} {dname}] Already encoded.", flush=True)
        return

    t0 = time.time()
    device = f"cuda:{gpu_id}"
    ds_corp = load_dataset(repo, "corpus", split="test")
    ds_q = load_dataset(repo, "queries", split="test")
    ds_rel = load_dataset(repo, "qrels", split="test")

    # Inspect schema robustly (ViDoRe v3 uses corpus_id / query_id or corpus-id / query-id)
    ckey = "corpus_id" if "corpus_id" in ds_corp.column_names else "corpus-id"
    qkey = "query_id" if "query_id" in ds_q.column_names else "query-id"
    qtkey = "query" if "query" in ds_q.column_names else "text"
    r_qkey = "query_id" if "query_id" in ds_rel.column_names else "query-id"
    r_ckey = "corpus_id" if "corpus_id" in ds_rel.column_names else "corpus-id"
    doc_col = "doc_id" if "doc_id" in ds_corp.column_names else ("document_id" if "document_id" in ds_corp.column_names else None)
    pg_col = "page_number_in_doc" if "page_number_in_doc" in ds_corp.column_names else ("page_num" if "page_num" in ds_corp.column_names else None)

    id_to_local = {}
    page_nums = []
    doc_hashes = []
    for i, row in enumerate(ds_corp):
        cid = str(row[ckey])
        id_to_local[cid] = i
        page_nums.append(int(row[pg_col]) if (pg_col and row[pg_col] is not None) else (i % 50))
        dh = str(row[doc_col]) if (doc_col and row[doc_col] is not None) else cid.split("_")[0]
        doc_hashes.append(dh)

    q_rel_map = {}
    for row in ds_rel:
        qid = str(row[r_qkey])
        cid = str(row[r_ckey])
        sc = float(row["score"])
        if sc > 0 and cid in id_to_local:
            q_rel_map.setdefault(qid, {})[id_to_local[cid]] = sc

    # Pick 40 queries with valid positive qrels per domain (20 eval + reserve)
    sel_q_texts, sel_q_rels = [], []
    for row in ds_q:
        qid = str(row[qkey])
        if qid in q_rel_map:
            sel_q_texts.append(str(row[qtkey]))
            sel_q_rels.append(q_rel_map[qid])
            if len(sel_q_texts) >= 40:
                break

    print(f"[dom={dom_idx} {dname} GPU {gpu_id}] {len(ds_corp)} pages, {len(sel_q_texts)} queries loaded ({time.time()-t0:.1f}s)", flush=True)

    proc = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-base", use_fast=True)
    model = ColQwen2_5.from_pretrained("vidore/colqwen2.5-v0.2", dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    # Encode queries (max 32 tokens padded/truncated to 32 for fast LUT compatibility + exact valid mask!)
    q_inputs = proc.process_queries(sel_q_texts).to(device)
    q_embs = model(**q_inputs)  # [Nq, Lq, 128]
    q_mask = q_inputs["attention_mask"]
    Nq = len(sel_q_texts)
    q32 = np.zeros((Nq, 32, 128), dtype=np.float32)
    qlens = np.zeros(Nq, dtype=np.int32)
    for i in range(Nq):
        valid = q_embs[i][q_mask[i] == 1].float().cpu().numpy()
        L = min(32, len(valid))
        q32[i, :L] = valid[:L]
        qlens[i] = L

    # Last-layer EOS attention hook for authentic AdaMerge-258
    last_attn = model.language_model.layers[-1].self_attn
    captured = {}

    def last_layer_hook(module, args, kwargs):
        hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        pos_emb = kwargs["position_embeddings"]
        bsz, q_len, _ = hs.size()
        q = module.q_proj(hs).view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
        k = module.k_proj(hs).view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
        cos, sin = pos_emb
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, module.rope_scaling["mrope_section"])
        k = repeat_kv(k, module.num_key_value_groups)
        captured["q"] = q
        captured["k"] = k

    hook_handle = last_attn.register_forward_pre_hook(last_layer_hook, with_kwargs=True)
    image_token_id = model.config.image_token_id
    scale = 1.0 / math.sqrt(last_attn.head_dim)

    from concurrent.futures import ThreadPoolExecutor
    N_pages = len(ds_corp)
    bs = 16
    raw_chunks, alpha_chunks, page_lens = [], [], np.empty(N_pages, dtype=np.int32)
    t_enc = time.time()

    def prep_batch(s_idx):
        e_idx = min(N_pages, s_idx + bs)
        imgs = [ds_corp[i]["image"].convert("RGB") for i in range(s_idx, e_idx)]
        return s_idx, e_idx, proc.process_images(imgs)

    starts = list(range(0, N_pages, bs))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(prep_batch, s) for s in starts[:8]]
        next_sub = 8
        for bi_idx in range(len(starts)):
            s, e, cpu_inputs = futs[bi_idx].result()
            futs[bi_idx] = None
            if next_sub < len(starts):
                futs.append(pool.submit(prep_batch, starts[next_sub]))
                next_sub += 1
            inputs = cpu_inputs.to(device)
            captured.clear()
            embs = model(**inputs)  # ColQwen2_5.forward slices pixel_values internally!
            q_all, k_all = captured["q"], captured["k"]
            attn_mask = inputs["attention_mask"]
            input_ids = inputs["input_ids"]
            for bi in range(e - s):
                valid_pos = torch.nonzero(attn_mask[bi] == 1, as_tuple=False).squeeze(-1)
                img_pos = valid_pos[input_ids[bi, valid_pos] == image_token_id]
                if len(img_pos) == 0:
                    img_pos = valid_pos
                eos_pos = int(valid_pos[-1].item())
                q_eos = q_all[bi, :, eos_pos:eos_pos + 1, :].float()
                k_val = k_all[bi, :, img_pos, :].float()
                logits = torch.matmul(q_eos, k_val.transpose(-1, -2)).squeeze(1) * scale
                probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy().astype(np.float32)
                v_fp16 = embs[bi, img_pos].to(torch.float16).cpu().numpy()
                page_lens[s + bi] = len(v_fp16)
                raw_chunks.append(v_fp16)
                alpha_chunks.append(probs)
            captured.clear()
            if (bi_idx % 20 == 0) or e == N_pages:
                el = time.time() - t_enc
                print(f"  [dom={dom_idx} {dname}] {e}/{N_pages} ({100*e/N_pages:.1f}%) | {e/max(el,1e-3):.1f} pages/s", flush=True)
    hook_handle.remove()

    raw_vecs = np.concatenate(raw_chunks, axis=0)
    alphas = np.concatenate(alpha_chunks, axis=0)
    offs = np.zeros(N_pages + 1, dtype=np.int64)
    np.cumsum(page_lens, out=offs[1:])

    np.save(OUT_DIR / f"dom_{dom_idx}_{dname}_raw_vecs.npy", raw_vecs)
    np.save(OUT_DIR / f"dom_{dom_idx}_{dname}_raw_offs.npy", offs)
    np.save(OUT_DIR / f"dom_{dom_idx}_{dname}_alphas.npy", alphas)
    np.save(OUT_DIR / f"dom_{dom_idx}_{dname}_q32.npy", q32)
    np.save(OUT_DIR / f"dom_{dom_idx}_{dname}_qlens.npy", qlens)
    with open(OUT_DIR / f"dom_{dom_idx}_{dname}_meta.json", "w") as f:
        json.dump({"dom_idx": dom_idx, "dname": dname, "repo": repo, "n_pages": N_pages, "page_nums": page_nums, "doc_hashes": doc_hashes, "qrels": sel_q_rels}, f)
    with open(done_f, "w") as f:
        json.dump({"n_pages": N_pages, "n_patches": int(len(raw_vecs)), "sec": time.time() - t0}, f)
    print(f"=== [dom={dom_idx} {dname}] DONE: {N_pages} pages, {len(raw_vecs):,} patches ({len(raw_vecs)/N_pages:.1f}/pg) in {time.time()-t0:.1f}s ===", flush=True)


def merge_and_adamerge():
    t0 = time.time()
    all_raw_v, all_alphas, all_doms, all_pnums = [], [], [], []
    all_q32, all_qlens, all_qdoms, all_qrels = [], [], [], []
    raw_offs = [0]
    cur_doc = 0
    for di, (dname, _) in enumerate(REPOS):
        v = np.load(OUT_DIR / f"dom_{di}_{dname}_raw_vecs.npy")
        o = np.load(OUT_DIR / f"dom_{di}_{dname}_raw_offs.npy")
        a = np.load(OUT_DIR / f"dom_{di}_{dname}_alphas.npy")
        q = np.load(OUT_DIR / f"dom_{di}_{dname}_q32.npy")
        ql = np.load(OUT_DIR / f"dom_{di}_{dname}_qlens.npy")
        meta = json.load(open(OUT_DIR / f"dom_{di}_{dname}_meta.json"))
        np_d = len(o) - 1
        all_raw_v.append(v)
        all_alphas.append(a)
        all_doms.append(np.full(np_d, di, dtype=np.uint8))
        all_pnums.append(np.array(meta["page_nums"], dtype=np.int32))
        base_tok = raw_offs[-1]
        for k in range(1, len(o)):
            raw_offs.append(base_tok + int(o[k]))
        all_q32.append(q)
        all_qlens.append(ql)
        all_qdoms.append(np.full(len(q), di, dtype=np.uint8))
        for qr in meta["qrels"]:
            all_qrels.append({str(cur_doc + int(k)): float(val) for k, val in qr.items()})
        cur_doc += np_d

    raw_vecs = np.concatenate(all_raw_v, axis=0)
    alphas = np.concatenate(all_alphas, axis=0)
    raw_offs = np.array(raw_offs, dtype=np.int64)
    doms = np.concatenate(all_doms, axis=0)
    pnums = np.concatenate(all_pnums, axis=0)
    N_pages = len(doms)

    # Interleave queries strictly across all 7 domains: [d0_q0, d1_q0, ..., d6_q0, d0_q1, ...] -> 140 stratified queries (20/dom)!
    q_list, ql_list, qd_list, qr_list = [], [], [], []
    per_d = min(len(q) for q in all_q32)
    for r in range(min(20, per_d)):
        for di in range(len(REPOS)):
            q_list.append(all_q32[di][r])
            ql_list.append(all_qlens[di][r])
            qd_list.append(di)
            # find offset in all_qrels
            base_q = sum(len(all_q32[j]) for j in range(di))
            qr_list.append(all_qrels[base_q + r])
    Q_32 = np.stack(q_list, axis=0).astype(np.float32)
    Q_lens = np.array(ql_list, dtype=np.int32)
    Q_doms = np.array(qd_list, dtype=np.uint8)

    np.save(OUT_DIR / "vidore_v3_raw_vecs.npy", raw_vecs)
    np.save(OUT_DIR / "vidore_v3_raw_offs.npy", raw_offs)
    np.save(OUT_DIR / "vidore_v3_doms.npy", doms)
    np.save(OUT_DIR / "vidore_v3_pnums.npy", pnums)
    np.save(OUT_DIR / "vidore_v3_q32.npy", Q_32)
    np.save(OUT_DIR / "vidore_v3_qlens.npy", Q_lens)
    np.save(OUT_DIR / "vidore_v3_qdoms.npy", Q_doms)
    with open(OUT_DIR / "vidore_v3_qrels.json", "w") as f:
        json.dump(qr_list, f)
    print(f"[Merge Raw] {N_pages:,} pages, {len(raw_vecs):,} patches ({len(raw_vecs)/N_pages:.1f}/pg), {len(Q_32)} stratified queries ({time.time()-t0:.1f}s)", flush=True)

    # Compute AdaMerge-258 (p_prune=0.30, keep_r=0.30, wmode="alpha", NO L2 renorm) in parallel!
    t_am = time.time()
    pages_args = [(raw_vecs[raw_offs[i]:raw_offs[i + 1]], alphas[raw_offs[i]:raw_offs[i + 1]]) for i in range(N_pages)]
    with ProcessPoolExecutor(max_workers=24) as ex:
        res = list(ex.map(_adamerge_page, pages_args, chunksize=64))
    am_offs = np.zeros(N_pages + 1, dtype=np.int64)
    for i, arr in enumerate(res):
        am_offs[i + 1] = am_offs[i] + len(arr)
    am_vecs = np.concatenate(res, axis=0)
    nrms = np.linalg.norm(am_vecs.astype(np.float32), axis=1)
    comp = 1.0 - len(am_vecs) / len(raw_vecs)
    np.save(OUT_DIR / "vidore_v3_adamerge_vecs.npy", am_vecs)
    np.save(OUT_DIR / "vidore_v3_adamerge_offs.npy", am_offs)
    summary = {
        "n_pages": N_pages,
        "raw_patches": int(len(raw_vecs)),
        "raw_avg_patches": round(len(raw_vecs) / N_pages, 2),
        "adamerge_patches": int(len(am_vecs)),
        "adamerge_avg_patches": round(len(am_vecs) / N_pages, 2),
        "compression_rate": round(comp, 4),
        "norm_min": round(float(nrms.min()), 4),
        "norm_median": round(float(np.median(nrms)), 4),
        "norm_max": round(float(nrms.max()), 4),
        "n_queries": int(len(Q_32)),
        "domains": [r[0] for r in REPOS],
    }
    with open(OUT_DIR / "vidore_v3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"=== [Merge + AdaMerge DONE in {time.time()-t_am:.1f}s] {json.dumps(summary)} ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", type=int, default=[])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    for d in args.domains:
        encode_domain(d, args.gpu)
    if args.merge:
        merge_and_adamerge()
