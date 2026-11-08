from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from openai import OpenAI
import argparse
import json
import os
import pandas as pd  # vẫn dùng cho error_rows -> csv
import pymysql
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456789")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "law")

# Milvus giới hạn độ dài chuỗi JSON cho mỗi field metadata
MAX_METADATA_JSON_BYTES = 64_000

# text-embedding-3-small giới hạn 8191 token/input; cắt bớt text dùng để embed
# (không phải noidung lưu trong metadata) để tránh lỗi cả batch vì 1 điều quá dài
MAX_EMBED_TEXT_CHARS = 12_000

# Số chunk tối đa gửi trong 1 lệnh gọi OpenAI embeddings.create(). Batch MySQL
# (200 điều) có thể sinh ra >1000 chunk nếu nhiều điều dài — gửi hết 1 lần dễ
# vượt giới hạn request của OpenAI (lỗi cả batch, rơi xuống embed từng chunk một
# cực chậm). Chia nhỏ theo lô này để lỗi (nếu có) chỉ ảnh hưởng 1 lô nhỏ.
EMBED_API_BATCH_SIZE = int(os.getenv("EMBED_API_BATCH_SIZE", "100"))

# ----------------- CHUNKING -----------------
# Điều luật dài được chia thành nhiều chunk theo số token (thay vì embed 1 khối
# rồi cắt bỏ phần vượt quá) để không mất nội dung và cải thiện độ chính xác
# retrieval. Mỗi chunk -> 1 vector riêng, cùng `mapc` với điều gốc.
_ENCODER = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "400"))
# Overlap tuyệt đối theo SỐ TOKEN (không phải %) — mặc định 80 = đúng 20% của
# 400 (giữ hành vi giống cấu hình cũ dùng tỉ lệ), nhưng giờ cố định theo số
# token tuyệt đối nên không tự đổi theo nếu bạn chỉnh CHUNK_SIZE_TOKENS.
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))

# Thứ tự ưu tiên tách khi recursive-chunking: đoạn > dòng > câu > mệnh đề > từ
# > ký tự (chuỗi rỗng "" là fallback cắt cứng khi không còn ranh giới nào phù hợp).
# Separator câu dùng regex (?<!\d)\.\s+ thay vì ". " thuần: văn bản luật VN đánh
# số khoản/điểm kiểu "1. ...", "2. ..." — tách thuần ". " sẽ cắt ngay sau số thứ
# tự (vd biến "2. Thông tư này..." thành 2 chunk: chunk rác "2" + phần còn lại
# mất luôn số khoản). Negative lookbehind (?<!\d) chặn việc này vì số khoản luôn
# đứng ngay trước dấu chấm đó, trong khi câu thường kết thúc bằng chữ cái.
_RECURSIVE_SEPARATORS = ["\n\n", "\n", r"(?<!\d)\.\s+", r";\s+", r",\s+", " ", ""]
_RECURSIVE_SEPARATORS_ARE_REGEX = True

# ----------------- OpenAI EMBEDDING -----------------
client = OpenAI()

def embed_batch(texts):
    """Embed 1 batch text, lỗi thì ném exception ra ngoài."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in resp.data]

def embed_text(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

# ----------------- MILVUS -----------------
collection_name = "phapdien_simple_tendieu"
embedding_dim = 1536

fields = [
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=False
    ),
    FieldSchema(
        name="metadata",
        dtype=DataType.JSON,  # object chứa toàn bộ metadata
    ),
    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=embedding_dim
    ),
]
schema = CollectionSchema(fields, description="Phapdien: vector + metadata JSON")

def _noidung_text(rec: dict) -> str:
    nd = rec.get("noidung") or []
    if isinstance(nd, list):
        body = "\n".join(nd)
    else:
        body = str(nd)
    return body.strip()


def build_metadata(rec: dict) -> dict:
    """Metadata JSON phải < 65536 byte; cắt noidung nếu cần.
    `rec` có thể là 1 chunk-record (có `noidung_chunk`/`chunk_index`/`chunk_total`)
    hoặc 1 điều gốc chưa chunk (fallback dùng toàn bộ noidung)."""
    noidung = rec.get("noidung_chunk")
    if noidung is None:
        noidung = _noidung_text(rec)
    meta = {
        "mapc": rec.get("mapc") or None,
        "tenchude": rec.get("tenchude") or None,
        "tendemuc": rec.get("tendemuc") or None,
        "tenchuong": rec.get("tenchuong") or None,
        "tendieu": rec.get("tendieu") or None,
        "chunk_index": rec.get("chunk_index", 0),
        "chunk_total": rec.get("chunk_total", 1),
        "noidung": noidung,
    }
    while meta["noidung"]:
        size = len(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        if size <= MAX_METADATA_JSON_BYTES:
            break
        meta["noidung"] = meta["noidung"][: max(0, len(meta["noidung"]) - 4000)]
    return meta


def build_header_text(rec: dict) -> str:
    parts = []
    for key in ("tenchude", "tendemuc", "tenchuong", "tendieu"):
        v = (rec.get(key) or "").strip()
        if v:
            parts.append(v)
    return "\n".join(parts)


def chunk_text_by_tokens(text: str, chunk_size: int = CHUNK_SIZE_TOKENS,
                          overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Recursive chunking: ưu tiên cắt tại ranh giới ngữ nghĩa (đoạn, dòng, câu,
    mệnh đề...) theo _RECURSIVE_SEPARATORS, chỉ cắt cứng giữa từ/ký tự khi không
    còn ranh giới nào giúp chunk vừa khít chunk_size. Kích thước/overlap đo bằng
    số token (tiktoken cl100k_base, khớp tokenizer của text-embedding-3-small),
    overlap là số token TUYỆT ĐỐI (không phải %). Văn bản ngắn hơn chunk_size
    trả về nguyên 1 chunk."""
    if not text:
        return [""]
    if len(_ENCODER.encode(text)) <= chunk_size:
        return [text]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap_tokens,
        separators=_RECURSIVE_SEPARATORS,
        is_separator_regex=_RECURSIVE_SEPARATORS_ARE_REGEX,
    )
    return splitter.split_text(text)


def build_chunk_records(rec: dict) -> list[dict]:
    """Từ 1 điều luật -> list bản ghi chunk (mỗi chunk sẽ thành 1 vector Milvus
    riêng, cùng `mapc`). Header (chủ đề/đề mục/chương/tên điều) được giữ nguyên
    ở mỗi chunk để embedding không mất ngữ cảnh văn bản gốc."""
    header = build_header_text(rec)
    noidung_full = _noidung_text(rec)
    chunks = chunk_text_by_tokens(noidung_full)
    total = len(chunks)
    return [
        {
            **rec,
            "header_text": header,
            "noidung_chunk": chunk_body,
            "chunk_index": idx,
            "chunk_total": total,
        }
        for idx, chunk_body in enumerate(chunks)
    ]


# Checkpoint ingest: 1 điều có thể ra nhiều chunk/vector nên không thể suy ra
# "đã ingest bao nhiêu điều" chỉ từ col.num_entities nữa (khác behaviour cũ).
# File này lưu lại doc_offset (số điều đã xử lý) + next_id (id Milvus tiếp theo).
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), ".seed_checkpoint.json")


def load_checkpoint() -> dict | None:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_checkpoint(doc_offset: int, next_id: int) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"doc_offset": doc_offset, "next_id": next_id}, f)


def setup_collection(*, resume: bool) -> tuple[Collection, int, int]:
    """Tạo collection mới hoặc tiếp tục ingest.
    Trả về (collection, doc_offset, next_id)."""
    if resume and utility.has_collection(collection_name):
        col = Collection(collection_name)
        col.load()
        ckpt = load_checkpoint()
        if ckpt:
            doc_offset, next_id = ckpt["doc_offset"], ckpt["next_id"]
        else:
            # Collection cũ (trước khi có chunking, không có checkpoint):
            # lúc đó 1 vector = 1 điều nên num_entities suy ra đúng doc_offset.
            doc_offset = col.num_entities
            next_id = col.num_entities + 1
        print(f"Resume collection '{collection_name}', doc_offset={doc_offset}, next_id={next_id}")
        return col, doc_offset, next_id

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    col = Collection(collection_name, schema)
    col.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    col.load()
    print("Created collection:", collection_name)
    return col, 0, 1


# ----------------- DATA: đọc từ JSON -----------------

def load_data_from_mysql():
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4"
    )
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    cursor.execute("""
        SELECT
            d.mapc    AS mapc,
            cd.ten    AS tenchude,
            dm.ten    AS tendemuc,
            ch.ten    AS tenchuong,
            d.ten     AS tendieu,
            d.noidung AS noidung
        FROM pddieu d
        LEFT JOIN pdchuong ch ON d.chuong_id = ch.mapc
        LEFT JOIN pddemuc  dm ON d.demuc_id  = dm.id
        LEFT JOIN pdchude  cd ON d.chude_id  = cd.id
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def select_batch(data, batch_size, offset):
    """Lấy 1 batch từ list data."""
    return data[offset: offset + batch_size]

def build_block_text(rec: dict) -> str:
    """
    Ghép block text để embed: header (tenchude, tendemuc, tenchuong, tendieu)
    + nội dung của CHUNK này (rec["noidung_chunk"], nếu rec đã qua
    build_chunk_records) — hoặc toàn bộ noidung nếu rec chưa chunk.
    """
    header = rec.get("header_text")
    if header is None:
        header = build_header_text(rec)

    body = rec.get("noidung_chunk")
    if body is None:
        body = _noidung_text(rec)
    else:
        body = body.strip()

    parts = [p for p in (header, body) if p]
    text = "\n\n".join(parts)
    return text[:MAX_EMBED_TEXT_CHARS]

# ----------------- INSERT BATCH -----------------
def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def insert_batch(chunk_records, next_id, error_rows) -> int:
    """
    chunk_records: list[dict], mỗi phần tử là 1 CHUNK (đã qua build_chunk_records):
      - tenchude, tendemuc, tenchuong, tendieu, mapc
      - header_text, noidung_chunk, chunk_index, chunk_total
    Trả về next_id đã cập nhật (next_id + len(chunk_records)) để batch sau
    không bị trùng id, kể cả khi có bản ghi lỗi embed trong batch này.

    Gửi embed theo từng lô nhỏ (EMBED_API_BATCH_SIZE) thay vì cả chunk_records
    1 lần — batch MySQL 200 điều có thể sinh >1000 chunk nếu nhiều điều dài,
    dễ vượt giới hạn request OpenAI; lỗi (nếu có) nhờ vậy chỉ rơi xuống
    embed-từng-chunk cho 1 lô nhỏ, không phải toàn bộ.
    """
    reserved_ids = list(range(next_id, next_id + len(chunk_records)))
    print(f"Embedding {len(chunk_records)} chunks (lô {EMBED_API_BATCH_SIZE} chunk/lần)…")

    total_ok = 0

    for sub_records, sub_ids in zip(_chunked(chunk_records, EMBED_API_BATCH_SIZE),
                                     _chunked(reserved_ids, EMBED_API_BATCH_SIZE)):
        ids, metadatas, vectors = [], [], []
        try:
            embedded = embed_batch([build_block_text(rec) for rec in sub_records])
            ids.extend(sub_ids)
            metadatas.extend(build_metadata(rec) for rec in sub_records)
            vectors.extend(embedded)
        except Exception as e:
            print(f"⚠ Lỗi embed lô {len(sub_records)} chunk ({e!r}), chuyển sang embed từng chunk")
            for rec, this_id in zip(sub_records, sub_ids):
                text = build_block_text(rec)
                try:
                    vec = embed_text(text)
                except Exception as e2:
                    error_rows.append({
                        "id": this_id,
                        "mapc": rec.get("mapc"),
                        "tendieu": rec.get("tendieu"),
                        "chunk_index": rec.get("chunk_index"),
                        "len_text": len(text),
                        "error": str(e2),
                    })
                    print(f"❌ Lỗi embed id={this_id}, mapc={rec.get('mapc')}, chunk={rec.get('chunk_index')}: {e2}")
                    continue
                ids.append(this_id)
                metadatas.append(build_metadata(rec))
                vectors.append(vec)

        # Insert ngay theo từng lô nhỏ (không tích luỹ cả batch MySQL rồi insert 1
        # lần) — batch MySQL (200 điều) có thể sinh >10.000 chunk nếu nhiều điều
        # rất dài, 1 lệnh insert() duy nhất cho toàn bộ dễ vượt giới hạn message
        # gRPC của Milvus (mặc định 64MB) → lỗi RESOURCE_EXHAUSTED, mất cả batch.
        if ids:
            collection.insert([ids, metadatas, vectors])
            total_ok += len(ids)

    total = len(chunk_records)
    if total_ok:
        print(f"Inserted {total_ok}/{total} chunks (id {reserved_ids[0]} → {reserved_ids[-1]})"
              + ("" if total_ok == total else f", lỗi={total - total_ok}"))
    else:
        print("⚠ Batch không insert được chunk nào (tất cả lỗi)")

    return next_id + len(chunk_records)

# ----------------- MAIN -----------------
def main():
    global collection

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục ingest vào collection hiện có (không drop).",
    )
    args = parser.parse_args()

    collection, doc_offset, next_id = setup_collection(resume=args.resume)

    data = load_data_from_mysql()
    total = len(data)
    batch_size = 200  # số ĐIỀU LUẬT mỗi batch — số chunk/vector thực tế có thể nhiều hơn

    num_batches = (total + batch_size - 1) // batch_size
    print("Total điều:", total)
    print("Total batches:", num_batches)
    if doc_offset:
        print(f"Bỏ qua {doc_offset} điều đã ingest.")

    error_rows = []

    while doc_offset < total:
        batch_articles = select_batch(data, batch_size, doc_offset)
        chunk_records = []
        for art in batch_articles:
            chunk_records.extend(build_chunk_records(art))

        next_id = insert_batch(chunk_records, next_id, error_rows)
        doc_offset += len(batch_articles)
        save_checkpoint(doc_offset, next_id)

    collection.flush()

    if error_rows:
        errors_df = pd.DataFrame(error_rows)
        errors_df.to_csv("embed_errors.csv", index=False, encoding="utf-8-sig")
        print(f"⚠ Đã ghi {len(error_rows)} bản ghi lỗi vào embed_errors.csv")
    else:
        print("✅ Không có bản ghi lỗi nào.")

    print("🎉 DONE - Ingest xong tất cả điều luật (đã chunking).")

collection = None  # gán trong main()

if __name__ == "__main__":
    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus")
    main()
