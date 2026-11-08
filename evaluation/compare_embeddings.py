"""So sánh nhanh text-embedding-3-small (đang dùng production) vs BAAI/bge-m3 vs
Qwen/Qwen3-Embedding-0.6B trên MẪU NHỎ (không đụng Milvus production, không
reseed) — thí nghiệm kiểu Phase 2 trong docs/DEVELOPMENT_PLAN.md nhưng ở quy mô
nhỏ để chạy nhanh trên CPU.

So sánh ĐỊNH TÍNH (so top-K kết quả giữa các model cho cùng câu hỏi), KHÔNG tính
được Precision/Recall chuẩn vì gold_docs trong data.json đang bị lệch offset sau
khi crawl lại (xem README mục Đánh giá — hạn chế đã biết).

Chạy: cd evaluation && python compare_embeddings.py
"""
import gc
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pymysql

SAMPLE_SIZE = 500
TOP_K = 5

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456789")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "law")


def load_sample(n: int) -> list[dict]:
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DATABASE, charset="utf8mb4",
    )
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("""
        SELECT d.mapc AS mapc, cd.ten AS tenchude, dm.ten AS tendemuc,
               ch.ten AS tenchuong, d.ten AS tendieu, d.noidung AS noidung
        FROM pddieu d
        LEFT JOIN pdchuong ch ON d.chuong_id = ch.mapc
        LEFT JOIN pddemuc  dm ON d.demuc_id  = dm.id
        LEFT JOIN pdchude  cd ON d.chude_id  = cd.id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    random.seed(42)
    return random.sample(rows, min(n, len(rows)))


def build_text(rec: dict) -> str:
    parts = [(rec.get(k) or "").strip() for k in ("tenchude", "tendemuc", "tenchuong", "tendieu")]
    parts = [p for p in parts if p]
    nd = (rec.get("noidung") or "").strip()
    if nd:
        parts.append(nd)
    return "\n\n".join(parts)[:3000]


def load_queries() -> list[str]:
    with open(ROOT / "evaluation" / "data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["query"] for item in data]


def cos_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def top_k_for_queries(query_emb: np.ndarray, corpus_emb: np.ndarray, sample: list[dict], k: int = TOP_K):
    sims = cos_sim_matrix(query_emb, corpus_emb)
    results = []
    for row in sims:
        idx = np.argsort(-row)[:k]
        results.append([(sample[i]["mapc"], sample[i].get("tendieu"), float(row[i])) for i in idx])
    return results


def run_openai(sample: list[dict], queries: list[str]):
    from openai import OpenAI
    client = OpenAI()
    texts = [build_text(r) for r in sample]

    def embed(texts_batch):
        out = []
        for i in range(0, len(texts_batch), 100):
            resp = client.embeddings.create(model="text-embedding-3-small", input=texts_batch[i:i + 100])
            out.extend([d.embedding for d in resp.data])
        return np.array(out)

    corpus_emb = embed(texts)
    query_emb = embed(queries)
    return top_k_for_queries(query_emb, corpus_emb, sample)


def _encode_with_fallback(model_name: str, device: str, texts: list[str], batch_size: int):
    """Encode trên GPU; nếu OOM (GTX 1650 chỉ 4GB, model lớn dễ tràn) thì tự giảm
    batch_size rồi thử lại, hết cỡ vẫn OOM thì rơi về CPU."""
    import torch
    from sentence_transformers import SentenceTransformer

    for bs in ([batch_size, max(4, batch_size // 4), 1] if device == "cuda" else [batch_size]):
        try:
            model = SentenceTransformer(model_name, device=device)
            emb = model.encode(texts, batch_size=bs, show_progress_bar=True, convert_to_numpy=True)
            return model, emb
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠ CUDA OOM với batch_size={bs}, thử lại nhỏ hơn...")
            try:
                del model
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    print("  ⚠ Vẫn OOM trên GPU — rơi về CPU.")
    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return model, emb


def run_sentence_transformer(model_name: str, sample: list[dict], queries: list[str], query_prefix: str = ""):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  (device: {device})")
    texts = [build_text(r) for r in sample]
    batch_size = 8 if device == "cuda" else 16  # GTX 1650 chỉ 4GB VRAM — batch nhỏ để tránh OOM

    model, corpus_emb = _encode_with_fallback(model_name, device, texts, batch_size)
    prefixed_queries = [f"{query_prefix}{q}" for q in queries] if query_prefix else queries
    query_emb = model.encode(prefixed_queries, convert_to_numpy=True)
    result = top_k_for_queries(query_emb, corpus_emb, sample)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


CACHE_PATH = ROOT / "evaluation" / "results" / "embedding_compare.json"


def load_cached_result(model_name: str, queries: list[str], sample_size: int):
    """Tái dùng kết quả model đã chạy lần trước (lưu trong embedding_compare.json)
    nếu cùng bộ câu hỏi + cùng sample_size — tránh gọi lại OpenAI tốn tiền/thời
    gian không cần thiết khi chỉ muốn chạy thêm model khác."""
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if cached.get("queries") != queries or cached.get("sample_size") != sample_size:
        return None
    res = cached.get("results", {}).get(model_name)
    if not res:
        return None
    return [[(r["mapc"], r["tendieu"], r["score"]) for r in per_query] for per_query in res]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["openai", "bge-m3", "qwen3"],
        choices=["openai", "bge-m3", "qwen3"],
        help="Chỉ chạy các model được chọn (vd: --models openai bge-m3, bỏ qua qwen3 cho nhanh)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bỏ qua cache, luôn gọi lại model (kể cả OpenAI) dù đã có kết quả lưu từ trước",
    )
    args = parser.parse_args()

    print(f"Lấy mẫu {SAMPLE_SIZE} điều luật từ MySQL (seed cố định để tái lập được)...")
    sample = load_sample(SAMPLE_SIZE)
    queries = load_queries()
    print(f"Mẫu: {len(sample)} điều, {len(queries)} câu hỏi test — model: {args.models}\n")

    results = {}

    if "openai" in args.models:
        cached = None if args.force else load_cached_result("text-embedding-3-small", queries, len(sample))
        if cached:
            print("=== text-embedding-3-small: dùng lại kết quả đã lưu (không gọi lại OpenAI) ===")
            results["text-embedding-3-small"] = cached
        else:
            print("=== text-embedding-3-small (OpenAI, đang dùng production) ===")
            results["text-embedding-3-small"] = run_openai(sample, queries)

    if "bge-m3" in args.models:
        print("\n=== BAAI/bge-m3 ===")
        results["bge-m3"] = run_sentence_transformer("BAAI/bge-m3", sample, queries)

    if "qwen3" in args.models:
        print("\n=== Qwen/Qwen3-Embedding-0.6B ===")
        qwen_prefix = "Instruct: Given a legal question, retrieve relevant Vietnamese law articles\nQuery: "
        results["qwen3-embedding-0.6b"] = run_sentence_transformer(
            "Qwen/Qwen3-Embedding-0.6B", sample, queries, query_prefix=qwen_prefix
        )

    for qi, query in enumerate(queries):
        print(f"\n{'=' * 80}\nCâu hỏi {qi + 1}: {query}\n{'=' * 80}")
        for model_name, res in results.items():
            print(f"\n[{model_name}] top-{TOP_K}:")
            for mapc, tendieu, score in res[qi]:
                label = (tendieu or mapc or "")[:70]
                print(f"  {score:.4f}  {label}")

    out_path = ROOT / "evaluation" / "results" / "embedding_compare.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "sample_size": len(sample),
            "queries": queries,
            "results": {
                model_name: [
                    [{"mapc": m, "tendieu": t, "score": s} for m, t, s in res[qi]]
                    for qi in range(len(queries))
                ]
                for model_name, res in results.items()
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu: {out_path}")


if __name__ == "__main__":
    main()
