import sys
from pathlib import Path

# Cho phép chạy: python retrieve/build_graph.py (từ thư mục gốc repo)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import re
import pymysql
from neo4j import GraphDatabase
from tqdm import tqdm

from retrieve.tavily_fallback import (
    is_context_sufficient,
    search_tavily,
    decompose_query,
    search_tavily_realtime,
)
 
# ─── CONFIG ───────────────────────────────────────────────────────────────────
 
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456789",
    "database": "law",
    "charset": "utf8mb4",
}
 
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"
 
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
 
# Pattern nhận dạng mã điều luật trong nội dung (vd: 1.1.LQ.1, 2.3.NĐ.5.2)
MAPC_PATTERN = re.compile(r'\b(\d+\.\d+\.[A-ZĐ]+\.\d+(?:\.\d+)*)\b')
 
def insert_thamchieu(driver, conn):
    with conn.cursor() as cur:
        cur.execute("SELECT mapc, noidung FROM pddieu WHERE noidung IS NOT NULL LIMIT 5")
        rows = cur.fetchall()

    # ── DEBUG: in thử noidung và kết quả regex ──
    for row in rows:
        print(f"\n--- mapc: {row['mapc']} ---")
        print(f"noidung (200 ký tự đầu): {row['noidung'][:200]}")
        found = MAPC_PATTERN.findall(row['noidung'] or "")
        print(f"MAPC_PATTERN tìm được: {found}")
 
    # Lấy tập hợp mapc hợp lệ để tránh tạo edge tới node không tồn tại
    with conn.cursor() as cur:
        cur.execute("SELECT mapc FROM pddieu")
        valid_mapcs = {r["mapc"] for r in cur.fetchall()}
 
    edges = []
    for row in rows:
        src = row["mapc"]
        found = MAPC_PATTERN.findall(row["noidung"] or "")
        for tgt in set(found):
            if tgt != src and tgt in valid_mapcs:
                edges.append({"src": src, "tgt": tgt})
 
    total = len(edges)
    print(f"  Parsed {total} THAM_CHIEU edges từ noidung...")
 
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
        """Trả về list mapc của các Dieu gần nhất về ngữ nghĩa."""
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
 
    # ── Stage 3: Simple rerank bằng keyword overlap ───────────────────────────
 
    def _rerank(self, candidates: list[dict], query: str, top_k: int) -> list[dict]:
        query_tokens = set(query.lower().split())
        seen = set()
        deduped = []

        for c in candidates:
            key = c.get("mapc") or c.get("passage", "")[:80]
            if key in seen:
                continue
            seen.add(key)
            text    = ((c.get("ten") or "") + " " + (c.get("noidung") or "")).lower()
            overlap = sum(1 for t in query_tokens if t in text)
            c["score"] = c["base_score"] + overlap * 0.05
            deduped.append(c)

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
        print("   Login: neo4j / password123")
        print("=" * 60)
 
    finally:
        conn.close()
        driver.close()
 
 
if __name__ == "__main__":
    main()
 