"""Seed Milvus bằng BGE-M3 vào collection RIÊNG `phapdien_bge_m3` — KHÔNG đụng
production (`phapdien_simple_tendieu`, vẫn dùng OpenAI qua seed_data_batch.py).
Thí nghiệm so sánh embedding model (xem README mục Đánh giá, docs/DEVELOPMENT_PLAN.md
Phase 2) — cùng chiến lược chunking (400 token/20% overlap) để so sánh công bằng.

Dùng GPU nếu có (tự in ra device đang dùng), batch nhỏ (mặc định 8, chỉnh qua
BGE_BATCH_SIZE) để tránh CUDA OOM trên GPU VRAM thấp (vd GTX 1650 4GB). Chạy lâu
(ước tính vài giờ cho 66k điều có chunking, ~7-8 điều/giây trên GTX 1650) — có
checkpoint resume RIÊNG (.seed_checkpoint_bge.json), độc lập với seed chính.

Chạy: cd data_processing && python seed_data_batch_bge.py [--resume]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility,
)

from seed_data_batch import (
    _chunked,
    build_block_text,
    build_chunk_records,
    build_metadata,
    load_data_from_mysql,
    select_batch,
)

MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
COLLECTION_NAME = "phapdien_bge_m3"
GPU_BATCH_SIZE = int(os.getenv("BGE_BATCH_SIZE", "8"))  # nhỏ để tránh CUDA OOM (GTX 1650 4GB)
MYSQL_BATCH_SIZE = 200  # số ĐIỀU mỗi batch MySQL — giống seed_data_batch.py
INSERT_SUB_BATCH = int(os.getenv("EMBED_API_BATCH_SIZE", "100"))  # giới hạn insert Milvus/lần

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), ".seed_checkpoint_bge.json")

_model = None
_device = None


def get_model():
    global _model, _device
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[BGE] device: {_device}")
        _model = SentenceTransformer(MODEL_NAME, device=_device)
    return _model


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Encode bằng BGE-M3; SentenceTransformer tự chia mini-batch theo
    GPU_BATCH_SIZE nội bộ nên có thể truyền cả danh sách lớn (vd 100 chunk)."""
    model = get_model()
    emb = model.encode(texts, batch_size=GPU_BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
    return emb.tolist()


fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="metadata", dtype=DataType.JSON),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
]
schema = CollectionSchema(fields, description="Phapdien (thí nghiệm BGE-M3): vector + metadata JSON")


def load_checkpoint() -> dict | None:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_checkpoint(doc_offset: int, next_id: int) -> None:
    tmp_path = CHECKPOINT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"doc_offset": doc_offset, "next_id": next_id}, f)
    os.replace(tmp_path, CHECKPOINT_PATH)


def setup_collection(*, resume: bool):
    if resume and utility.has_collection(COLLECTION_NAME):
        col = Collection(COLLECTION_NAME)
        col.load()
        ckpt = load_checkpoint()
        doc_offset, next_id = (ckpt["doc_offset"], ckpt["next_id"]) if ckpt else (0, 1)
        print(f"Resume collection '{COLLECTION_NAME}', doc_offset={doc_offset}, next_id={next_id}")
        return col, doc_offset, next_id

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    col = Collection(COLLECTION_NAME, schema)
    col.create_index(
        "embedding",
        {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
    )
    col.load()
    print("Created collection:", COLLECTION_NAME)
    return col, 0, 1


def insert_batch(chunk_records: list[dict], next_id: int, error_rows: list[dict], collection: Collection) -> int:
    reserved_ids = list(range(next_id, next_id + len(chunk_records)))
    print(f"[BGE] Embedding {len(chunk_records)} chunks...")

    total_ok = 0
    for sub_records, sub_ids in zip(_chunked(chunk_records, INSERT_SUB_BATCH),
                                     _chunked(reserved_ids, INSERT_SUB_BATCH)):
        try:
            texts = [build_block_text(rec) for rec in sub_records]
            vectors = embed_batch(texts)
            metadatas = [build_metadata(rec) for rec in sub_records]
            collection.insert([sub_ids, metadatas, vectors])
            total_ok += len(sub_ids)
        except Exception as e:
            print(f"⚠ Lỗi embed/insert lô {len(sub_records)} chunk ({e!r}), bỏ qua lô này")
            for rec, this_id in zip(sub_records, sub_ids):
                error_rows.append({"id": this_id, "mapc": rec.get("mapc"), "error": str(e)})

    total = len(chunk_records)
    print(f"[BGE] Inserted {total_ok}/{total} chunks (id {reserved_ids[0]} → {reserved_ids[-1]})"
          + ("" if total_ok == total else f", lỗi={total - total_ok}"))
    return next_id + len(chunk_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Tiếp tục ingest vào collection hiện có.")
    args = parser.parse_args()

    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus")

    collection, doc_offset, next_id = setup_collection(resume=args.resume)
    get_model()  # load model ngay để in rõ device đang dùng trước khi bắt đầu

    data = load_data_from_mysql()
    total = len(data)
    print("Total điều:", total)
    if doc_offset:
        print(f"Bỏ qua {doc_offset} điều đã ingest.")

    error_rows: list[dict] = []
    while doc_offset < total:
        batch_articles = select_batch(data, MYSQL_BATCH_SIZE, doc_offset)
        chunk_records = []
        for art in batch_articles:
            chunk_records.extend(build_chunk_records(art))

        next_id = insert_batch(chunk_records, next_id, error_rows, collection)
        doc_offset += len(batch_articles)
        save_checkpoint(doc_offset, next_id)

    collection.flush()

    if error_rows:
        import pandas as pd
        pd.DataFrame(error_rows).to_csv("embed_errors_bge.csv", index=False, encoding="utf-8-sig")
        print(f"⚠ Đã ghi {len(error_rows)} bản ghi lỗi vào embed_errors_bge.csv")
    else:
        print("✅ Không có bản ghi lỗi nào.")

    print("🎉 DONE - Ingest xong tất cả điều luật bằng BGE-M3 (đã chunking).")


if __name__ == "__main__":
    main()
