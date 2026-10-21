import sys
from pathlib import Path

# Cho phép chạy: python retrieve/build_graph.py (từ thư mục gốc repo)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import re
import pymysql
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

from retrieve.tavily_fallback import (
    is_context_sufficient,
    search_tavily,
    decompose_query,
    search_tavily_realtime,
)

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "123456789"),
    "database": os.getenv("MYSQL_DATABASE", "law"),
    "charset": "utf8mb4",
}

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
 
BATCH_SIZE = 500   # số node insert mỗi lần (tối ưu tốc độ)
 
# ─── KẾT NỐI ──────────────────────────────────────────────────────────────────
 
def get_mysql():
    return pymysql.connect(**MYSQL_CONFIG, cursorclass=pymysql.cursors.DictCursor)
 
def get_neo4j():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
 
# ─── BƯỚC 0: TẠO CONSTRAINTS & INDEX ─────────────────────────────────────────
 
def create_constraints(driver):
    """
    Chạy 1 lần để tạo index và unique constraint.
    Giúp tăng tốc MERGE và tìm kiếm node.
    """
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ChuDe)  REQUIRE n.id   IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DeMuc)  REQUIRE n.id   IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Chuong) REQUIRE n.mapc IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Dieu)   REQUIRE n.mapc IS UNIQUE",
        # Full-text search index cho tìm kiếm keyword
        """CREATE FULLTEXT INDEX dieu_fulltext IF NOT EXISTS
           FOR (n:Dieu) ON EACH [n.ten, n.noidung]""",
    ]
    with driver.session() as session:
        for cypher in constraints:
            try:
                session.run(cypher)
            except Exception as e:
                print(f"[constraint] {e}")
    print("✅ Constraints & indexes OK")
 
# ─── BƯỚC 1: INSERT PHÂN CẤP ──────────────────────────────────────────────────
 
def insert_chude(driver, conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, ten, stt FROM pdchude")
        rows = cur.fetchall()
 
    with driver.session() as session:
        session.run("""
            UNWIND $rows AS r
            MERGE (n:ChuDe {id: r.id})
            SET n.ten = r.ten, n.stt = r.stt
        """, rows=rows)
    print(f"✅ ChuDe: {len(rows)} nodes")
 
 
def insert_demuc(driver, conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, ten, stt, chude_id FROM pddemuc")
        rows = cur.fetchall()
 
    with driver.session() as session:
        # Insert node
        session.run("""
            UNWIND $rows AS r
            MERGE (n:DeMuc {id: r.id})
            SET n.ten = r.ten, n.stt = r.stt
        """, rows=rows)
        # Tạo relationship với ChuDe
        session.run("""
            UNWIND $rows AS r
            MATCH (cd:ChuDe {id: r.chude_id})
            MATCH (dm:DeMuc {id: r.id})
            MERGE (cd)-[:CO_DE_MUC]->(dm)
        """, rows=rows)
    print(f"✅ DeMuc: {len(rows)} nodes + relationships")
 
 
def insert_chuong(driver, conn):
    with conn.cursor() as cur:
        cur.execute("SELECT mapc, ten, chimuc, stt, demuc_id FROM pdchuong")
        rows = cur.fetchall()
 
    with driver.session() as session:
        session.run("""
            UNWIND $rows AS r
            MERGE (n:Chuong {mapc: r.mapc})
            SET n.ten = r.ten, n.chimuc = r.chimuc, n.stt = r.stt
        """, rows=rows)
        session.run("""
            UNWIND $rows AS r
            MATCH (dm:DeMuc {id: r.demuc_id})
            MATCH (ch:Chuong {mapc: r.mapc})
            MERGE (dm)-[:CO_CHUONG]->(ch)
        """, rows=rows)
    print(f"✅ Chuong: {len(rows)} nodes + relationships")
 
# ─── BƯỚC 2: INSERT DIEU (batch vì 66k rows) ──────────────────────────────────
 
def insert_dieu(driver, conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT mapc, ten, noidung, chimuc, stt,
                   chuong_id, demuc_id, chude_id
            FROM pddieu
        """)
        rows = cur.fetchall()
 
    total = len(rows)
    print(f"  Inserting {total} Dieu nodes...")
 
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Dieu nodes"):
        batch = rows[i : i + BATCH_SIZE]

        with driver.session() as session:
            session.run("""
                UNWIND $batch AS r
                MERGE (d:Dieu {mapc: r.mapc})
                SET d.ten       = r.ten,
                    d.noidung   = r.noidung,
                    d.chimuc    = r.chimuc,
                    d.stt       = r.stt,
                    d.demuc_id  = r.demuc_id,
                    d.chude_id  = r.chude_id
            """, batch=batch)
 
    print(f"  Linking Dieu → Chuong...")
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Dieu→Chuong"):
        batch = rows[i : i + BATCH_SIZE]
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS r
                MATCH (d:Dieu   {mapc: r.mapc})
                MATCH (ch:Chuong {mapc: r.chuong_id})
                MERGE (ch)-[:CO_DIEU]->(d)
            """, batch=batch)
 
    print(f"✅ Dieu: {total} nodes")
 
# ─── BƯỚC 3: INSERT LIEN_QUAN (38k edges) ─────────────────────────────────────
 
def insert_lienquan(driver, conn):
    with conn.cursor() as cur:
        cur.execute("SELECT dieu_id1_id, dieu_id2_id FROM pdmuclienquan")
        rows = cur.fetchall()
 
    total = len(rows)
    print(f"  Inserting {total} LIEN_QUAN edges...")
 
    for i in tqdm(range(0, total, BATCH_SIZE), desc="LIEN_QUAN"):
        batch = rows[i : i + BATCH_SIZE]
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS r
                MATCH (a:Dieu {mapc: r.dieu_id1_id})
                MATCH (b:Dieu {mapc: r.dieu_id2_id})
                MERGE (a)-[:LIEN_QUAN]->(b)
                MERGE (b)-[:LIEN_QUAN]->(a)
            """, batch=batch)
 
    print(f"✅ LIEN_QUAN: {total} edges (bidirectional)")
 
# ─── BƯỚC 4: PARSE THAM_CHIEU TỪ NỘI DUNG ────────────────────────────────────
#
# Lưu ý: mapc thật là chuỗi số dài (vd "0100100000000000100000100000000000000000"),
# KHÔNG phải dạng "1.1.LQ.1" (đó là mã phân loại nội bộ của Pháp Điển, nằm trong
# trường `ten`, không bao giờ xuất hiện trong `noidung` làm trích dẫn). Trích dẫn
# chéo thật trong `noidung` được viết tự nhiên kiểu "Điều 6", "Điều 12" — không kèm
# tên văn bản, nên phải suy ra "Điều N" đó thuộc văn bản nào bằng cách gom các điều
# theo văn bản nguồn qua trường `vbqppl` (vd "(Điều 1 Luật số 32/2004/QH11 ...)").

# "(Điều N <mô tả văn bản nguồn>..." — N là số điều gốc trong văn bản nguồn
LEAD_DIEU_PATTERN = re.compile(r'^\(?\s*[Đđ]iều\s+(\d+[a-z]?)\s+(.+)', re.DOTALL)
# định danh văn bản nguồn, vd "số 32/2004/QH11", "số 16/2006/NĐ-CP"
DOC_KEY_PATTERN = re.compile(r'số\s+[\w\./\-]+')
# trích dẫn "Điều N" xuất hiện trong nội dung điều luật khác
REF_PATTERN = re.compile(r'[Đđ]iều\s+(\d+[a-z]?)\b')


def insert_thamchieu(driver, conn):
    with conn.cursor() as cur:
        cur.execute("SELECT mapc, noidung, vbqppl FROM pddieu")
        rows = cur.fetchall()

    # Gom điều luật theo văn bản nguồn: doc_key -> {số điều gốc: mapc}
    article_doc = {}  # mapc -> (doc_key, số điều gốc)
    doc_index = {}
    for row in rows:
        m = LEAD_DIEU_PATTERN.match((row["vbqppl"] or "").strip())
        if not m:
            continue
        num, rest = m.groups()
        dm = DOC_KEY_PATTERN.search(rest)
        if not dm:
            continue
        key = dm.group(0)
        article_doc[row["mapc"]] = (key, num)
        doc_index.setdefault(key, {})[num] = row["mapc"]

    # Với mỗi điều, "Điều N" trong noidung được suy ra là điều N của CÙNG văn bản
    # nguồn (bỏ qua trích dẫn chéo sang văn bản khác — không đủ ngữ cảnh để khớp
    # chắc chắn, tránh tạo edge sai).
    edges = []
    for row in rows:
        src = row["mapc"]
        info = article_doc.get(src)
        if not info or not row["noidung"]:
            continue
        doc_key, own_num = info
        for tgt_num in set(REF_PATTERN.findall(row["noidung"])):
            if tgt_num == own_num:
                continue
            tgt = doc_index.get(doc_key, {}).get(tgt_num)
            if tgt and tgt != src:
                edges.append({"src": src, "tgt": tgt})

    total = len(edges)
    print(f"  Parsed {total} THAM_CHIEU edges (khớp 'Điều N' trong cùng văn bản nguồn)...")
 
    for i in tqdm(range(0, total, BATCH_SIZE), desc="THAM_CHIEU"):
        batch = edges[i : i + BATCH_SIZE]
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS r
                MATCH (a:Dieu {mapc: r.src})
                MATCH (b:Dieu {mapc: r.tgt})
                MERGE (a)-[:THAM_CHIEU]->(b)
            """, batch=batch)
 
    print(f"✅ THAM_CHIEU: {total} edges")
 
# ─── CACHE (sqlitedict) — tránh gọi lại OpenAI embedding cho query lặp lại ─────

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
_VECTOR_CACHE_PATH = os.path.join(_CACHE_DIR, "vector_search_cache.sqlite")
_CACHE_ENABLED = os.getenv("RAG_CACHE_ENABLED", "1").strip().lower() in ("1", "true", "yes")


def _vector_cache_get(key: str):
    if not _CACHE_ENABLED:
        return None
    try:
        import sqlitedict
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with sqlitedict.SqliteDict(_VECTOR_CACHE_PATH) as db:
            return db.get(key)
    except Exception:
        return None


def _vector_cache_set(key: str, value) -> None:
    if not _CACHE_ENABLED:
        return
    try:
        import sqlitedict
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with sqlitedict.SqliteDict(_VECTOR_CACHE_PATH, autocommit=True) as db:
            db[key] = value
    except Exception:
        pass


# ─── HYBRID RERANK HELPERS (BM25 + embedding cục bộ, không gọi OpenAI) ─────────

_TOKEN_RE = re.compile(r"\w+")


def _bm25_scores_normalized(query: str, texts: list[str]) -> list[float]:
    """Điểm BM25 (tìm kiếm theo từ khoá) cho từng text so với query, chuẩn hoá về [0,1]."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_TOKEN_RE.findall(t.lower()) for t in texts]
    query_tokens = _TOKEN_RE.findall(query.lower())
    if not any(tokenized_corpus) or not query_tokens:
        return [0.0] * len(texts)

    bm25 = BM25Okapi(tokenized_corpus)
    raw = list(bm25.get_scores(query_tokens))
    lo, hi = min(raw), max(raw)
    if hi - lo < 1e-9:
        return [0.0] * len(raw)
    return [(v - lo) / (hi - lo) for v in raw]


def _semantic_scores(query: str, texts: list[str]) -> list[float]:
    """Độ tương đồng ngữ nghĩa (cosine) giữa query và từng text, dùng embedding cục bộ."""
    try:
        from services.local_embeddings import get_local_embedder
        from sentence_transformers import util as st_util

        model = get_local_embedder()
        query_emb = model.encode(query, convert_to_tensor=True)
        text_emb = model.encode(texts, convert_to_tensor=True)
        return st_util.cos_sim(query_emb, text_emb)[0].tolist()
    except Exception:
        return [0.0] * len(texts)


# ─── GRAPH RAG RETRIEVER ───────────────────────────────────────────────────────

class GraphRAGRetriever:
    """
    Kết hợp Milvus (vector search) + Neo4j (graph traversal).
    
    Cách dùng:
        retriever = GraphRAGRetriever(neo4j_driver, milvus_collection, openai_client)
        results = retriever.retrieve("hành vi chống người thi hành công vụ", top_k=5, hop=2)
        # results: list[dict] với keys: mapc, ten, noidung, score, source
    """
 
    def __init__(self, neo4j_driver, milvus_collection, openai_client):
        self.driver     = neo4j_driver
        self.collection = milvus_collection
        self.client     = openai_client
 
    # ── Embedding ────────────────────────────────────────────────────────────
 
    def _embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding
 
    # ── Stage 1: Milvus vector search ────────────────────────────────────────
 
    def _vector_search(self, query: str, top_k: int) -> list[str]:
        """Trả về list mapc của các Dieu gần nhất về ngữ nghĩa.
        Kết quả được cache theo (query, top_k) để tránh gọi lại OpenAI embedding
        khi cùng một câu hỏi lặp lại (vd: khi chạy đánh giá hoặc người dùng hỏi lại)."""
        cache_key = f"{query.strip().lower()}::{top_k}"
        cached = _vector_cache_get(cache_key)
        if cached is not None:
            return cached

        mapcs = self._vector_search_uncached(query, top_k)
        _vector_cache_set(cache_key, mapcs)
        return mapcs

    def _vector_search_uncached(self, query: str, top_k: int) -> list[str]:
        try:
            vec = self._embed(query)
            results = self.collection.search(
                data=[vec],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 128}},
                limit=top_k,
                output_fields=["metadata"],
            )
        except Exception as e:
            print(f"[GraphRAG] Milvus search lỗi: {e}")
            return []

        if not results or not results[0]:
            print("[GraphRAG] Milvus trả về rỗng")
            return []

        mapcs = []
        for hit in results[0]:
            try:
                meta = hit.entity.get("metadata") or {}
            except Exception:
                meta = {}

            if isinstance(meta, str):
                import json
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}

            mapc = meta.get("mapc") or (meta.get("tendieu") or "").split(".")[0]
            if mapc:
                mapcs.append(mapc)

        print(f"[GraphRAG] _vector_search raw hits: {len(results[0])}, mapcs tìm được: {len(mapcs)}")

        # Nếu không lấy được mapc nào, in debug để kiểm tra cấu trúc metadata
        if not mapcs and results[0]:
            sample = results[0][0]
            try:
                print(f"[DEBUG] Sample entity fields: {sample.entity.fields}")
                print(f"[DEBUG] Sample metadata raw: {sample.entity.get('metadata')}")
            except Exception as e:
                print(f"[DEBUG] Không đọc được entity: {e}")

        return mapcs
 
    # ── Stage 2: Graph expansion ──────────────────────────────────────────────
 
    def _graph_expand(self, seed_mapcs: list[str], hop: int) -> list[dict]:
        if not seed_mapcs:
            return []

        # Neo4j không cho dùng $param trong *1..n — phải format trực tiếp
        cypher = f"""
            MATCH (seed:Dieu)
            WHERE seed.mapc IN $mapcs

            CALL {{
                WITH seed
                MATCH (seed)-[:LIEN_QUAN|THAM_CHIEU*1..{hop}]-(neighbor:Dieu)
                RETURN neighbor
                UNION
                WITH seed
                RETURN seed AS neighbor
            }}

            OPTIONAL MATCH (ch:Chuong)-[:CO_DIEU]->(neighbor)
            OPTIONAL MATCH (dm:DeMuc)-[:CO_CHUONG]->(ch)
            OPTIONAL MATCH (cd:ChuDe)-[:CO_DE_MUC]->(dm)

            RETURN DISTINCT
                neighbor.mapc    AS mapc,
                neighbor.ten     AS ten,
                neighbor.noidung AS noidung,
                ch.ten           AS tenchuong,
                dm.ten           AS tendemuc,
                cd.ten           AS tenchude,
                CASE WHEN neighbor.mapc IN $mapcs THEN 1.0 ELSE 0.5 END AS base_score
            ORDER BY base_score DESC
            LIMIT 40
        """

        with self.driver.session() as session:
            result = session.run(cypher, mapcs=seed_mapcs)

            rows = []
            for r in result:
                passage = "\n".join(filter(None, [
                    r["tenchude"],
                    r["tendemuc"],
                    r["tenchuong"],
                    r["ten"],
                    r["noidung"],
                ]))
                rows.append({
                    "mapc":       r["mapc"],
                    "ten":        r["ten"],
                    "noidung":    r["noidung"],
                    "tenchuong":  r["tenchuong"],
                    "tendemuc":   r["tendemuc"],
                    "tenchude":   r["tenchude"],
                    "passage":    passage,
                    "base_score": r["base_score"],
                })
            return rows
 
    # ── Stage 3: Hybrid rerank — BM25 (từ khoá) + embedding cục bộ (ngữ nghĩa) ──
    # Cả hai đều tính tại chỗ, không gọi OpenAI/LLM nào — không phát sinh chi phí.

    def _rerank(
        self,
        candidates: list[dict],
        query: str,
        top_k: int,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.3,
    ) -> list[dict]:
        seen = set()
        deduped = []
        for c in candidates:
            key = c.get("mapc") or c.get("passage", "")[:80]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        if not deduped:
            return []

        texts = [((c.get("ten") or "") + " " + (c.get("noidung") or "")) for c in deduped]
        bm25_norm = _bm25_scores_normalized(query, texts)
        semantic_scores = _semantic_scores(query, texts)

        for c, bm25_s, sem_s in zip(deduped, bm25_norm, semantic_scores):
            c["bm25_score"] = bm25_s
            c["semantic_score"] = sem_s
            c["score"] = c["base_score"] + bm25_weight * bm25_s + semantic_weight * max(0.0, sem_s)

        deduped.sort(key=lambda x: x["score"], reverse=True)
        return deduped[:top_k]
    # ── Main retrieve ─────────────────────────────────────────────────────────
 
    def retrieve(self, query, top_k=20, seed_k=5, hop=2) -> list[dict]:
        # Decompose trước khi search
        decomposed     = decompose_query(query)
        legal_query    = decomposed.get("legal_query") or query
        realtime_query = decomposed.get("realtime_query")
        calc_hint      = decomposed.get("calc_hint")

        if realtime_query:
            print(f"[GraphRAG] Phát hiện query thực tế: '{realtime_query}'")

        # Stage 1–3: Graph RAG với legal_query
        seed_mapcs = self._vector_search(legal_query, top_k=seed_k)
        print(f"[GraphRAG] Seeds: {len(seed_mapcs)}")

        candidates = self._graph_expand(seed_mapcs, hop=hop)
        print(f"[GraphRAG] Candidates: {len(candidates)}")

        final = self._rerank(candidates, legal_query, top_k=top_k)

        for r in final:
            r.setdefault("source", "phapdien")
            r.setdefault("source_label", "[Pháp Điển]")

        # Stage 4a: Fallback pháp lý — TRUYỀN query VÀO để check relevance + thời sự
        if not is_context_sufficient(final, query=query):
            print(f"[GraphRAG] Context không đủ/không liên quan → Tavily legal fallback")
            web_legal = search_tavily(legal_query)
            final = final + web_legal
            print(f"[GraphRAG] Sau legal fallback: {len(final)} kết quả")

        # Stage 4b: Realtime data
        if realtime_query:
            realtime_results = search_tavily_realtime(realtime_query)
            final = realtime_results + final
            print(f"[GraphRAG] Sau realtime: {len(final)} kết quả")

        # Stage 4c: Gắn calc hint nếu có
        if calc_hint:
            final.insert(0, {
                "mapc": None, "ten": "Yêu cầu tính toán",
                "noidung": calc_hint,
                "passage": f"[Hướng dẫn tính toán]\n{calc_hint}",
                "score": 1.0, "source": "system",
                "source_label": "[Hệ thống]",
            })

        print(f"[GraphRAG] Tổng kết quả trả về: {len(final)}")
        return final
 
 
# ─── NOTE: Thêm mapc vào metadata khi seed ────────────────────────────────────
#
# Trong seed_data_batch.py, sửa phần build metadata:
#
#   meta = {
#       "mapc":     rec.get("mapc") or None,   # ← THÊM DÒNG NÀY
#       "tenchude": rec.get("tenchude") or None,
#       "tendemuc": rec.get("tendemuc") or None,
#       "tenchuong":rec.get("tenchuong") or None,
#       "tendieu":  rec.get("tendieu") or None,
#       "noidung":  body[:500],
#   }
#
# Và trong query MySQL thêm d.mapc:
#   SELECT d.mapc, cd.ten AS tenchude, dm.ten AS tendemuc, ...
#
# ──────────────────────────────────────────────────────────────────────────────
 
 
# ─── MAIN ─────────────────────────────────────────────────────────────────────
 
def main():
    print("=" * 60)
    print("Graph RAG Builder — MySQL → Neo4j")
    print("=" * 60)
 
    conn   = get_mysql()
    driver = get_neo4j()
 
    try:
        print("\n[0] Tạo constraints & indexes...")
        create_constraints(driver)
 
        print("\n[1] Insert ChuDe...")
        insert_chude(driver, conn)
 
        print("\n[2] Insert DeMuc...")
        insert_demuc(driver, conn)
 
        print("\n[3] Insert Chuong...")
        insert_chuong(driver, conn)
 
        print("\n[4] Insert Dieu (66k nodes — sẽ mất vài phút)...")
        insert_dieu(driver, conn)
 
        print("\n[5] Insert LIEN_QUAN (38k edges)...")
        insert_lienquan(driver, conn)
 
        print("\n[6] Parse & insert THAM_CHIEU từ noidung...")
        insert_thamchieu(driver, conn)
 
        print("\n" + "=" * 60)
        print("🎉 DONE — Graph đã được build thành công!")
        print("   Neo4j Browser: http://localhost:7474")
        print(f"   Login: {NEO4J_USER} / (xem NEO4J_PASSWORD trong .env)")
        print("=" * 60)
 
    finally:
        conn.close()
        driver.close()
 
 
if __name__ == "__main__":
    main()
 