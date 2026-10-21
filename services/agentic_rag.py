"""
Agentic RAG: OpenAI tool-calling loop (vector + graph + web + full article).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

# Giới hạn nội dung đưa vào prompt tool (UI bucket giữ full)
LLM_MAX_NOIDUNG_CHARS = int(os.getenv("AGENT_LLM_MAX_NOIDUNG_CHARS", "4000"))
LLM_MAX_PASSAGE_CHARS = int(os.getenv("AGENT_LLM_MAX_PASSAGE_CHARS", "4000"))
# Rerank: 1 = sau overlap gọi LLM chấm lại top-N (tốn API); 0 = chỉ overlap + Milvus order
RAG_AGENT_LLM_RERANK = os.getenv("RAG_AGENT_LLM_RERANK", "0").strip().lower() in ("1", "true", "yes")
AGENT_VECTOR_POOL_MULT = max(1, int(os.getenv("AGENT_VECTOR_POOL_MULT", "3")))

_MYSQL_CFG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "123456789"),
    "database": os.getenv("MYSQL_DATABASE", "law"),
    "charset": "utf8mb4",
}

_DIEU_CONTEXT_CYPHER = """
MATCH (d:Dieu) WHERE d.mapc IN $mapcs
OPTIONAL MATCH (ch:Chuong)-[:CO_DIEU]->(d)
OPTIONAL MATCH (dm:DeMuc)-[:CO_CHUONG]->(ch)
OPTIONAL MATCH (cd:ChuDe)-[:CO_DE_MUC]->(dm)
RETURN d.mapc AS mapc, d.ten AS ten, d.noidung AS noidung,
       ch.ten AS tenchuong, dm.ten AS tendemuc, cd.ten AS tenchude
"""

_DIEU_SINGLE_CYPHER = """
MATCH (d:Dieu {mapc: $mapc})
OPTIONAL MATCH (ch:Chuong)-[:CO_DIEU]->(d)
OPTIONAL MATCH (dm:DeMuc)-[:CO_CHUONG]->(ch)
OPTIONAL MATCH (cd:ChuDe)-[:CO_DE_MUC]->(dm)
RETURN d.mapc AS mapc, d.ten AS ten, d.noidung AS noidung,
       ch.ten AS tenchuong, dm.ten AS tendemuc, cd.ten AS tenchude
"""

# Tool definitions for OpenAI Chat Completions API
RAG_AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Tìm kiếm semantic trong Milvus. Dùng khi cần tìm điều luật liên quan đến chủ đề.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Cụm từ tìm kiếm pháp lý"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_traverse",
            "description": "Duyệt graph Neo4j để tìm các điều luật có quan hệ LIEN_QUAN hoặc THAM_CHIEU. Dùng sau vector_search để mở rộng context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mapcs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Danh sách mapc của các điều luật seed",
                    },
                    "hop": {"type": "integer", "default": 2},
                },
                "required": ["mapcs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Tìm kiếm thông tin pháp luật mới nhất trên web. Dùng khi cần thông tin cập nhật hoặc văn bản chưa có trong Pháp Điển.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_article",
            "description": "Lấy toàn bộ nội dung một điều luật theo mapc. Dùng khi cần đọc chi tiết hơn snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mapc": {"type": "string"},
                },
                "required": ["mapc"],
            },
        },
    },
]


def agent_chat_model() -> str:
    return (
        os.getenv("OPENAI_AGENT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )


def agent_system_prompt(*, situation_mode: bool) -> str:
    base = """Bạn là trợ lý pháp luật Việt Nam thông minh với khả năng tìm kiếm đa chiều.

Quy trình làm việc:
1. Phân tích câu hỏi và lên kế hoạch tìm kiếm
2. Gọi vector_search để tìm điều luật liên quan
3. Nếu tìm thấy điều luật phù hợp, dùng graph_traverse để mở rộng context
4. Nếu cần thông tin cập nhật, gọi web_search
5. Nếu cần đọc chi tiết hơn, gọi get_full_article
6. Sau khi đủ thông tin (thường 2-4 tool calls), tổng hợp câu trả lời

Nguyên tắc:
- Luôn trích dẫn điều luật cụ thể (tên điều, chương, đề mục) khi có trong công cụ Pháp Điển
- Nếu thông tin mâu thuẫn giữa Pháp Điển và Web, ưu tiên Pháp Điển
- Nếu không đủ thông tin sau khi đã thử các công cụ phù hợp, thành thật nói không biết
- Không bịa thông tin pháp lý"""
    if situation_mode:
        base += """

Chế độ tình huống: người dùng mô tả một tình huống thực tế. Hãy:
- Chỉ ra các hành vi/căn cứ pháp lý có thể áp dụng
- Trích dẫn điều luật rõ ràng
- Tóm tắt rủi ro và hướng xử lý gợi ý (không thay thế tư vấn của luật sư)"""
    return base


def truncate_for_llm(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n… [đã rút gọn]"


def hit_dict_to_row(h: dict[str, Any]) -> dict[str, Any]:
    return {
        "mapc": h.get("mapc"),
        "ten": h.get("ten"),
        "noidung": h.get("noidung"),
        "tenchuong": h.get("tenchuong"),
        "tendemuc": h.get("tendemuc"),
        "tenchude": h.get("tenchude"),
    }


def _llm_score_passage(client: Any, query: str, passage: str, *, model: str | None = None) -> float:
    """Chấm liên quan query–passage (0–1), cùng logic TwoStageRetriever._score_pair."""
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    passage = (passage or "")[:8000]
    if not passage.strip():
        return 0.0
    system_prompt = (
        "Bạn là một hệ thống chấm điểm truy hồi văn bản phục vụ cho trả lời truy vấn pháp luật. "
        "Nhiệm vụ của bạn là đánh giá mức độ đoạn văn (passage) trả lời được câu hỏi (query). "
        "Đoạn văn nào trực tiếp cần dùng để trả lời câu hỏi thì điểm cao."
        "Hãy trả về MỘT số thực trong khoảng từ 0 đến 1:\n"
        "- 0: hoàn toàn không liên quan\n"
        "- 1: rất liên quan, trả lời trực tiếp câu hỏi\n"
        "Chỉ in ra đúng MỘT số, không giải thích gì thêm."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query:\n{query}\n\nPassage:\n{passage}"},
            ],
            temperature=0.0,
            max_tokens=8,
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"([01](?:\.\d+)?)", text)
        if not m:
            return 0.0
        return max(0.0, min(1.0, float(m.group(1))))
    except Exception:
        return 0.0


def _apply_llm_rerank_head(
    retriever: Any,
    query: str,
    ranked: list[dict[str, Any]],
    *,
    head_n: int = 12,
) -> list[dict[str, Any]]:
    """Chấm LLM lại phần đầu danh sách đã overlap-rerank, rồi gộp đuôi."""
    client = getattr(retriever, "client", None)
    if not client or not ranked:
        return ranked
    n = min(head_n, len(ranked))
    head = [dict(x) for x in ranked[:n]]
    tail = ranked[n:]
    for c in head:
        passage = (c.get("passage") or "").strip()
        if not passage:
            passage = "\n".join(
                filter(None, [c.get("ten"), c.get("noidung")])
            )
        c["score"] = _llm_score_passage(client, query, passage)
    head.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    return head + tail


def rerank_phapdien_hits(
    retriever: Any,
    rerank_query: str,
    full_hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Sắp xếp lại passage Pháp Điển theo truy vấn người dùng:
    1) GraphRAGRetriever._rerank (overlap từ khóa + base_score)
    2) Tuỳ chọn: LLM chấm lại top-N (RAG_AGENT_LLM_RERANK=1)
    """
    q = (rerank_query or "").strip()
    if not full_hits or not q or not hasattr(retriever, "_rerank"):
        return full_hits

    candidates: list[dict[str, Any]] = []
    for i, h in enumerate(full_hits):
        base = float(h.get("score", max(0.35, 1.0 - i * 0.012)))
        candidates.append(
            {
                "mapc": h.get("mapc"),
                "ten": h.get("ten") or "",
                "noidung": (h.get("noidung") or "")[:12000],
                "passage": (h.get("passage") or "")[:12000],
                "tenchuong": h.get("tenchuong"),
                "tendemuc": h.get("tendemuc"),
                "tenchude": h.get("tenchude"),
                "base_score": base,
            }
        )

    pool = max(top_k, min(len(candidates), 40))
    ranked = retriever._rerank(candidates, q, top_k=pool)
    if RAG_AGENT_LLM_RERANK and ranked:
        ranked = _apply_llm_rerank_head(retriever, q, ranked, head_n=12)

    out: list[dict[str, Any]] = []
    for c in ranked[:top_k]:
        row = {
            "mapc": c.get("mapc"),
            "ten": c.get("ten"),
            "noidung": c.get("noidung"),
            "tenchuong": c.get("tenchuong"),
            "tendemuc": c.get("tendemuc"),
            "tenchude": c.get("tenchude"),
        }
        out.append(row_to_full_hit(row, score=float(c.get("score", 0.5))))
    return out


def build_passage_text(row: dict[str, Any]) -> str:
    return "\n".join(
        filter(
            None,
            [
                row.get("tenchude"),
                row.get("tendemuc"),
                row.get("tenchuong"),
                row.get("ten"),
                row.get("noidung"),
            ],
        )
    )


def row_to_full_hit(row: dict[str, Any], *, score: float, source: str = "phapdien") -> dict[str, Any]:
    return {
        "mapc": row.get("mapc", ""),
        "ten": row.get("ten", ""),
        "tenchuong": row.get("tenchuong"),
        "tendemuc": row.get("tendemuc"),
        "tenchude": row.get("tenchude"),
        "noidung": row.get("noidung") or "",
        "passage": build_passage_text(row),
        "score": score,
        "source": source,
    }


def row_to_llm_item(row: dict[str, Any]) -> dict[str, Any]:
    noidung = row.get("noidung") or ""
    preview = truncate_for_llm(noidung, LLM_MAX_NOIDUNG_CHARS)
    passage = build_passage_text(row)
    passage_preview = truncate_for_llm(passage, LLM_MAX_PASSAGE_CHARS)
    return {
        "mapc": row.get("mapc", ""),
        "ten": row.get("ten", ""),
        "tenchuong": row.get("tenchuong"),
        "tendemuc": row.get("tendemuc"),
        "tenchude": row.get("tenchude"),
        "noidung": preview,
        "passage": passage_preview,
        "noidung_truncated": len(noidung) > len(preview),
    }


def _fetch_dieu_mysql(mapcs: list[str]) -> list[dict[str, Any]]:
    if not mapcs:
        return []
    try:
        import pymysql
    except ImportError:
        return []
    placeholders = ",".join(["%s"] * len(mapcs))
    sql = f"""
        SELECT
            d.mapc AS mapc,
            cd.ten AS tenchude,
            dm.ten AS tendemuc,
            ch.ten AS tenchuong,
            d.ten AS ten,
            d.noidung AS noidung
        FROM pddieu d
        LEFT JOIN pdchuong ch ON d.chuong_id = ch.mapc
        LEFT JOIN pddemuc dm ON d.demuc_id = dm.id
        LEFT JOIN pdchude cd ON d.chude_id = cd.id
        WHERE d.mapc IN ({placeholders})
    """
    try:
        conn = pymysql.connect(**_MYSQL_CFG)
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(sql, mapcs)
            rows = list(cur.fetchall())
        conn.close()
        return rows
    except Exception:
        return []


def fetch_dieu_records(neo4j_driver: Any, mapcs: list[str]) -> list[dict[str, Any]]:
    """Lấy full điều luật + ngữ cảnh; giữ thứ tự mapcs từ Milvus."""
    ordered_ids = [str(m).strip() for m in mapcs if m]
    if not ordered_ids:
        return []

    by_mapc: dict[str, dict[str, Any]] = {}
    try:
        with neo4j_driver.session() as session:
            for r in session.run(_DIEU_CONTEXT_CYPHER, mapcs=ordered_ids):
                by_mapc[r["mapc"]] = dict(r)
    except Exception:
        pass

    missing = [m for m in ordered_ids if m not in by_mapc]
    for row in _fetch_dieu_mysql(missing):
        by_mapc[row["mapc"]] = row

    return [by_mapc[m] for m in ordered_ids if m in by_mapc]


def fetch_dieu_single(neo4j_driver: Any, mapc: str) -> dict[str, Any] | None:
    mapc = (mapc or "").strip()
    if not mapc:
        return None
    try:
        with neo4j_driver.session() as session:
            row = session.run(_DIEU_SINGLE_CYPHER, mapc=mapc).single()
        if row:
            return dict(row)
    except Exception:
        pass
    rows = _fetch_dieu_mysql([mapc])
    return rows[0] if rows else None


def merge_passage_hits(
    hits: list[dict[str, Any]],
    bucket: list[dict[str, Any]],
    *,
    max_extend: int = 12,
) -> None:
    seen: set[str] = set()
    for h in bucket:
        key = h.get("mapc") or h.get("url") or ""
        if key:
            seen.add(str(key))
    for h in hits[:max_extend]:
        key = h.get("mapc") or h.get("url") or (h.get("passage") or "")[:80]
        sk = str(key)
        if not sk or sk in seen:
            continue
        seen.add(sk)
        bucket.append(h)


def assistant_message_to_dict(msg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    tcalls = getattr(msg, "tool_calls", None)
    if tcalls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": getattr(tc, "type", None) or "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in tcalls
        ]
    return out


def execute_agent_tool(
    name: str,
    args: dict[str, Any],
    *,
    retriever: Any,
    neo4j_driver: Any,
    rerank_query: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """
  Thực thi tool.
  Returns:
    - JSON rút gọn cho LLM (tool message)
    - full hits cho bucket UI
    """
    args = args or {}
    empty: list[dict[str, Any]] = []
    try:
        if name == "vector_search":
            q = str(args.get("query", "")).strip()
            top_k = int(args.get("top_k", 10))
            if not q:
                return json.dumps([], ensure_ascii=False), empty

            pool_k = min(max(top_k * AGENT_VECTOR_POOL_MULT, top_k), 40)
            mapcs = retriever._vector_search(q, top_k=pool_k)
            if not mapcs:
                return json.dumps([], ensure_ascii=False), empty

            rows = fetch_dieu_records(neo4j_driver, mapcs)
            full_hits = [row_to_full_hit(r, score=0.8) for r in rows]
            rq = (rerank_query or "").strip() or q
            full_hits = rerank_phapdien_hits(retriever, rq, full_hits, top_k=top_k)
            llm_items = [row_to_llm_item(hit_dict_to_row(h)) for h in full_hits]
            return json.dumps(llm_items, ensure_ascii=False), full_hits

        if name == "graph_traverse":
            mapcs = args.get("mapcs") or []
            if not isinstance(mapcs, list):
                mapcs = []
            mapcs = [str(x) for x in mapcs if x]
            hop = max(1, int(args.get("hop", 2)))
            results = retriever._graph_expand(mapcs, hop=hop)[:40]

            full_hits = []
            for r in results:
                row = {
                    "mapc": r.get("mapc"),
                    "ten": r.get("ten"),
                    "noidung": r.get("noidung"),
                    "tenchuong": r.get("tenchuong"),
                    "tendemuc": r.get("tendemuc"),
                    "tenchude": r.get("tenchude"),
                }
                score = float(r.get("base_score", 0.85))
                full_hits.append(row_to_full_hit(row, score=score))

            graph_top = min(len(full_hits), 20) if full_hits else 0
            if graph_top:
                rq = (rerank_query or "").strip() or " ".join(mapcs[:5])
                full_hits = rerank_phapdien_hits(retriever, rq, full_hits, top_k=graph_top)
            llm_items = [row_to_llm_item(hit_dict_to_row(h)) for h in full_hits]

            return json.dumps(llm_items, ensure_ascii=False), full_hits

        if name == "web_search":
            from retrieve.tavily_fallback import search_tavily

            q = str(args.get("query", "")).strip()
            if not q:
                return json.dumps([], ensure_ascii=False), empty

            res = search_tavily(q, top_k=5)
            full_hits = []
            llm_items = []
            for r in res:
                content = r.get("noidung") or ""
                full_hits.append({
                    "mapc": r.get("mapc") or "",
                    "ten": r.get("ten") or "",
                    "noidung": content,
                    "passage": content,
                    "score": 0.7,
                    "source": "web",
                    "url": r.get("url", ""),
                    "trust_level": r.get("trust_level", "medium"),
                    "source_label": r.get("source_label", "[Web]"),
                })
                llm_items.append({
                    "title": r.get("ten"),
                    "ten": r.get("ten"),
                    "content": truncate_for_llm(content, 400),
                    "noidung": truncate_for_llm(content, 400),
                    "url": r.get("url", ""),
                    "mapc": r.get("mapc") or "",
                })
            return json.dumps(llm_items, ensure_ascii=False), full_hits

        if name == "get_full_article":
            mapc = str(args.get("mapc", "")).strip()
            if not mapc:
                return json.dumps({"error": "Thiếu mapc"}, ensure_ascii=False), empty

            row = fetch_dieu_single(neo4j_driver, mapc)
            if not row:
                return json.dumps({"error": "Không tìm thấy"}, ensure_ascii=False), empty

            full_hit = row_to_full_hit(row, score=1.0)
            llm_payload = {
                "mapc": mapc,
                "ten": row.get("ten"),
                "tenchuong": row.get("tenchuong"),
                "tendemuc": row.get("tendemuc"),
                "tenchude": row.get("tenchude"),
                "noidung": row.get("noidung") or "",
            }
            return json.dumps(llm_payload, ensure_ascii=False), [full_hit]

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False), empty

    return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False), empty


def apply_agent_tool_result(
    tool_name: str,
    tool_content: str,
    full_hits: list[dict[str, Any]],
    bucket: list[dict[str, Any]],
    *,
    max_extend: int = 12,
) -> None:
    """Gộp full hits vào bucket; fallback parse JSON nếu không có hits."""
    if full_hits:
        merge_passage_hits(full_hits, bucket, max_extend=max_extend)
    else:
        merge_tool_result_into_passages(
            tool_name, tool_content, bucket, max_extend=max_extend
        )


def hits_for_passages(tool_name: str, result_json: str) -> list[dict[str, Any]]:
    """Chuẩn hoá kết quả tool thành dict dùng được với _build_passages."""
    try:
        data = json.loads(result_json)
    except Exception:
        return []

    out: list[dict[str, Any]] = []

    if tool_name == "get_full_article":
        if not isinstance(data, dict) or data.get("error"):
            return []
        out.append({
            "mapc": data.get("mapc", ""),
            "ten": data.get("ten", ""),
            "noidung": data.get("noidung", ""),
            "passage": "\n".join(filter(None, [data.get("ten"), data.get("noidung")])),
            "score": 1.0,
            "source": "phapdien",
        })
        return out

    if not isinstance(data, list):
        return []

    for r in data:
        if not isinstance(r, dict):
            continue
        url = r.get("url", "")
        if url or tool_name == "web_search":
            out.append({
                "mapc": r.get("mapc") or "",
                "ten": r.get("ten") or r.get("title") or "",
                "noidung": r.get("noidung") or r.get("content") or "",
                "passage": r.get("passage")
                or "\n".join(filter(None, [r.get("title"), r.get("content"), r.get("noidung")])),
                "score": float(r.get("score", 0.7)),
                "source": "web",
                "url": url,
                "trust_level": r.get("trust_level", "medium"),
                "source_label": r.get("source_label", "[Web]"),
            })
        elif tool_name == "graph_traverse":
            out.append({
                "mapc": r.get("mapc", ""),
                "ten": r.get("ten", ""),
                "tenchuong": r.get("tenchuong"),
                "tendemuc": r.get("tendemuc"),
                "tenchude": r.get("tenchude"),
                "noidung": r.get("noidung") or r.get("passage", ""),
                "passage": r.get("passage") or r.get("noidung", ""),
                "score": 0.85,
                "source": "phapdien",
            })
        else:
            # vector_search
            out.append({
                "mapc": r.get("mapc", ""),
                "ten": r.get("ten", ""),
                "noidung": r.get("noidung", ""),
                "passage": "\n".join(filter(None, [r.get("ten"), r.get("noidung")])),
                "score": 0.8,
                "source": "phapdien",
            })
    return out


def merge_tool_result_into_passages(
    tool_name: str,
    result_json: str,
    bucket: list[dict[str, Any]],
    *,
    max_extend: int = 8,
) -> None:
    """Fallback: parse JSON tool (thường bản rút gọn) vào bucket."""
    hits = hits_for_passages(tool_name, result_json)
    merge_passage_hits(hits, bucket, max_extend=max_extend)


SYNTHESIS_PROMPT = "Hãy tổng hợp câu trả lời dựa trên thông tin đã thu thập (tiếng Việt)."

GROUNDING_SIM_THRESHOLD = float(os.getenv("AGENT_GROUNDING_THRESHOLD", "0.35"))

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?…])\s+|\n+")


def check_answer_grounding(answer: str, passages: list[dict[str, Any]]) -> dict[str, Any]:
    """Kiểm tra độ trung thực (faithfulness/grounding): mỗi câu trong câu trả lời có thực
    sự bám sát nội dung đã truy xuất hay không — so sánh bằng embedding cục bộ (không tốn
    phí gọi LLM/OpenAI), không phải chấm điểm LLM-as-judge.

    Trả về {"grounding_score": 0..1, "flagged_sentences": [câu nghi ngờ bịa]}.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(answer or "") if len(s.strip()) > 8]
    context_texts = [str(p.get("noidung") or p.get("passage") or "") for p in passages]
    context_texts = [t for t in context_texts if t.strip()]

    if not sentences or not context_texts:
        return {"grounding_score": None, "flagged_sentences": []}

    try:
        from services.local_embeddings import get_local_embedder
        model = get_local_embedder()
        from sentence_transformers import util as st_util

        sent_emb = model.encode(sentences, convert_to_tensor=True)
        ctx_emb = model.encode(context_texts, convert_to_tensor=True)
        sims = st_util.cos_sim(sent_emb, ctx_emb)  # [n_sentences, n_contexts]
        best_per_sentence = sims.max(dim=1).values.tolist()
    except Exception:
        return {"grounding_score": None, "flagged_sentences": []}

    flagged = [s for s, score in zip(sentences, best_per_sentence) if score < GROUNDING_SIM_THRESHOLD]
    grounding_score = sum(best_per_sentence) / len(best_per_sentence)
    return {"grounding_score": round(grounding_score, 4), "flagged_sentences": flagged}


# ── SSE / response shaping (dùng chung bởi run_agentic_rag, trước nằm trùng lặp
#    trong rag_api.py ở cả /rag/chat và /rag/stream) ────────────────────────────
def _web_sources_markdown(web_sources: list[dict[str, str]]) -> str:
    if not web_sources:
        return ""
    lines = ["\n\n---\n**Nguồn tham khảo từ web:**"]
    for src in web_sources:
        icon = "✅" if src.get("level") == "high" else "⚠️"
        lines.append(
            f"{icon} {src.get('label', '[Web]')} [{src.get('title', '')}]({src.get('url', '')})"
        )
    return "\n".join(lines)


def _build_passages(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passages = []
    for hit in results:
        passages.append({
            "mapc":      hit.get("mapc", ""),
            "ten":       hit.get("ten", ""),
            "tenchuong": hit.get("tenchuong", ""),
            "tendemuc":  hit.get("tendemuc", ""),
            "tenchude":  hit.get("tenchude", ""),
            "noidung":   str(hit.get("noidung") or hit.get("passage", "")),
            "score":     round(float(hit.get("score", 0)), 4),
            "source":    hit.get("source", "phapdien"),
            "url":       hit.get("url", ""),
            "trust_level": hit.get("trust_level", "medium"),
            "source_label": hit.get("source_label", "[Web]"),
        })
    return passages


def _web_sources_from_passages(passages: list[dict[str, Any]]) -> list[dict[str, str]]:
    web_sources: list[dict[str, str]] = []
    for p in passages:
        if p.get("source") in ("web", "web_realtime") and p.get("url"):
            web_sources.append({
                "title": str(p.get("ten", "")),
                "url":   str(p.get("url", "")),
                "level": str(p.get("trust_level", "medium")),
                "label": str(p.get("source_label", "[Web]")),
            })
    return web_sources


def _dedup_results(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_keys: set = set()
    web_r, pd_r = [], []
    for hit in raw_results:
        key = hit.get("mapc") or hit.get("url") or hit.get("passage", "")[:100]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if hit.get("source") in ("web", "web_realtime"):
            web_r.append(hit)
        else:
            pd_r.append(hit)
    return (web_r + pd_r)[:20]


def _chunk_stream_text(text: str, max_chars: int = 32):
    """Tách văn bản thành nhiều mẩu nhỏ (theo từ / dòng) để client hiển thị mượt."""
    rest = text or ""
    if not rest:
        return
    while rest:
        if rest[0] == "\n":
            n = 1
            while n < len(rest) and rest[n] == "\n":
                n += 1
            yield rest[:n]
            rest = rest[n:]
            continue
        m = re.match(r"(\S+)(\s*)", rest)
        if m:
            piece = m.group(0)
            if len(piece) > max_chars:
                yield piece[:max_chars]
                rest = rest[max_chars:]
            else:
                yield piece
                rest = rest[len(piece):]
        else:
            yield rest[0]
            rest = rest[1:]


def _build_meta_event(
    bucket: list[dict[str, Any]], *, normalized_query: str, iterations: int
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    passages = _build_passages(_dedup_results(bucket))
    web_sources = _web_sources_from_passages(passages)
    event = {
        "type": "meta",
        "action": "proceed",
        "normalized_query": normalized_query,
        "passages": passages,
        "sources": web_sources,
        "iterations": iterations,
        "agentic": True,
    }
    return event, web_sources


def run_agentic_rag(
    *,
    openai_client: Any,
    retriever: Any,
    neo4j_driver: Any,
    user_prompt: str,
    history: list[dict[str, str]],
    normalized_query: str,
    situation_mode: bool = False,
    max_iterations: int = 6,
    rerank_query: str | None = None,
):
    """
    Vòng lặp tool-calling, dùng chung cho /rag/chat (gom hết) và /rag/stream (pass-through SSE).
    Yield các event dict theo đúng schema SSE đã có từ trước:
      {"type": "status", "text", "iteration", "max"}
      {"type": "meta", "action", "normalized_query", "passages", "sources", "iterations", "agentic"}
      {"type": "token", "text"}
    """
    rerank_ctx = (rerank_query if rerank_query is not None else user_prompt) or ""
    model = agent_chat_model()
    system_prompt = agent_system_prompt(situation_mode=situation_mode)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *[{"role": m["role"], "content": m["content"]} for m in history[-8:]],
        {"role": "user", "content": user_prompt},
    ]
    bucket: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=RAG_AGENT_TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = response.choices[0].message
        messages.append(assistant_message_to_dict(msg))

        if not msg.tool_calls:
            meta_event, web_sources = _build_meta_event(
                bucket, normalized_query=normalized_query, iterations=iteration + 1
            )
            yield meta_event

            full_answer_parts: list[str] = []
            content = (msg.content or "").strip()
            if content:
                for part in _chunk_stream_text(content):
                    full_answer_parts.append(part)
                    yield {"type": "token", "text": part}
            else:
                stream = openai_client.chat.completions.create(
                    model=model,
                    messages=messages + [{"role": "user", "content": SYNTHESIS_PROMPT}],
                    temperature=0.2,
                    max_tokens=1200,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_answer_parts.append(delta)
                        yield {"type": "token", "text": delta}

            if web_sources:
                yield {"type": "token", "text": _web_sources_markdown(web_sources)}
            yield {"type": "grounding", **check_answer_grounding("".join(full_answer_parts), meta_event["passages"])}
            return

        for tc in msg.tool_calls:
            yield {
                "type": "status",
                "text": f"🔍 Đang gọi {tc.function.name}...",
                "iteration": iteration + 1,
                "max": max_iterations,
            }
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_content, full_hits = execute_agent_tool(
                tc.function.name,
                args,
                retriever=retriever,
                neo4j_driver=neo4j_driver,
                rerank_query=rerank_ctx,
            )
            apply_agent_tool_result(
                tc.function.name, tool_content, full_hits, bucket
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_content,
            })

    # Hết max_iterations mà model vẫn muốn gọi tool tiếp — ép tổng hợp câu trả lời.
    meta_event, web_sources = _build_meta_event(
        bucket, normalized_query=normalized_query, iterations=max_iterations
    )
    yield meta_event

    full_answer_parts: list[str] = []
    stream = openai_client.chat.completions.create(
        model=model,
        messages=messages + [{"role": "user", "content": SYNTHESIS_PROMPT}],
        temperature=0.2,
        max_tokens=1200,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_answer_parts.append(delta)
            yield {"type": "token", "text": delta}

    if web_sources:
        yield {"type": "token", "text": _web_sources_markdown(web_sources)}
    yield {"type": "grounding", **check_answer_grounding("".join(full_answer_parts), meta_event["passages"])}
