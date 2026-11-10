"""
Truy vấn toàn bộ 19 chỉ số đã cấu hình trong dashboard "RAG Ops Overview"
(scripts/setup_langfuse_dashboard.py) trực tiếp qua Langfuse Metrics API, in ra
JSON — dùng để tự phân tích định kỳ mà không cần mở UI, hoặc để đối chiếu số
liệu dashboard có đúng không.

Chạy: python scripts/query_langfuse_metrics.py [--days N]
  --days N   số ngày gần nhất để lấy dữ liệu (mặc định 30)

Yêu cầu: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL trong .env
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langfuse import get_client

client = get_client()
if not client.auth_check():
    sys.exit("❌ Langfuse auth thất bại — kiểm tra LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL trong .env")
api = client.api

REQUEST_NAMES = ["rag_chat_request", "rag_stream_request"]
TOOL_NAMES = ["tool:vector_search", "tool:graph_traverse", "tool:web_search", "tool:get_full_article"]


def f_name_in(names: list[str]) -> dict:
    return {"column": "name", "operator": "any of", "type": "stringOptions", "value": names}


def f_name_eq(name: str) -> dict:
    return {"column": "name", "operator": "=", "type": "string", "value": name}


def build_queries() -> list[tuple[str, dict]]:
    return [
        # ── 1. Performance ──
        ("Performance — Request latency P50/P95/P99", dict(
            view="observations", dimensions=[], filters=[f_name_in(REQUEST_NAMES)],
            metrics=[{"measure": "latency", "aggregation": a} for a in ("p50", "p95", "p99")])),
        ("Performance — Request volume theo tên", dict(
            view="observations", dimensions=[{"field": "name"}], filters=[f_name_in(REQUEST_NAMES)],
            metrics=[{"measure": "count", "aggregation": "count"}])),

        # ── 2. Cost ──
        ("Cost — Token input/output", dict(
            view="observations", dimensions=[], filters=[],
            metrics=[{"measure": "inputTokens", "aggregation": "sum"}, {"measure": "outputTokens", "aggregation": "sum"}])),
        ("Cost — Tổng chi phí ($)", dict(
            view="observations", dimensions=[], filters=[],
            metrics=[{"measure": "totalCost", "aggregation": "sum"}])),
        ("Cost — Chi phí theo model", dict(
            view="observations", dimensions=[{"field": "providedModelName"}], filters=[],
            metrics=[{"measure": "totalCost", "aggregation": "sum"}])),

        # ── 3. Reliability ──
        ("Reliability — Error rate", dict(
            view="scores-boolean", dimensions=[], filters=[f_name_eq("request_error")],
            metrics=[{"measure": "value", "aggregation": "avg"}, {"measure": "value", "aggregation": "count"}])),
        ("Reliability — Timeout rate", dict(
            view="scores-boolean", dimensions=[], filters=[f_name_eq("timeout")],
            metrics=[{"measure": "value", "aggregation": "avg"}, {"measure": "value", "aggregation": "count"}])),

        # ── 4. RAG Quality ──
        ("RAG Quality — Faithfulness", dict(
            view="scores-numeric", dimensions=[], filters=[f_name_eq("faithfulness")],
            metrics=[{"measure": "value", "aggregation": "avg"}, {"measure": "value", "aggregation": "count"}])),
        ("RAG Quality — Answer relevancy", dict(
            view="scores-numeric", dimensions=[], filters=[f_name_eq("answer_relevancy")],
            metrics=[{"measure": "value", "aggregation": "avg"}, {"measure": "value", "aggregation": "count"}])),
        ("RAG Quality — Context precision", dict(
            view="scores-numeric", dimensions=[], filters=[f_name_eq("context_precision")],
            metrics=[{"measure": "value", "aggregation": "avg"}, {"measure": "value", "aggregation": "count"}])),

        # ── 5. Agent Quality ──
        ("Agent Quality — Tool call count theo loại", dict(
            view="observations", dimensions=[{"field": "name"}], filters=[f_name_in(TOOL_NAMES)],
            metrics=[{"measure": "count", "aggregation": "count"}])),
        ("Agent Quality — Tool success rate theo loại tool", dict(
            view="scores-boolean", dimensions=[{"field": "observationName"}], filters=[f_name_eq("tool_success")],
            metrics=[{"measure": "value", "aggregation": "avg"}, {"measure": "value", "aggregation": "count"}])),

        # ── 6. Safety ──
        ("Safety — Guardrail pass rate", dict(
            view="scores-boolean", dimensions=[], filters=[f_name_eq("guardrail_pass")],
            metrics=[{"measure": "value", "aggregation": "avg"}, {"measure": "value", "aggregation": "count"}])),
        ("Safety — Unsafe output rate", dict(
            view="scores-boolean", dimensions=[], filters=[f_name_eq("unsafe_output")],
            metrics=[{"measure": "value", "aggregation": "avg"}, {"measure": "value", "aggregation": "count"}])),
        ("Safety/Request log — Phân bố outcome", dict(
            view="scores-categorical", dimensions=[{"field": "stringValue"}], filters=[f_name_eq("request_outcome")],
            metrics=[{"measure": "count", "aggregation": "count"}])),

        # ── 7. UX ──
        ("User Experience — Fallback rate", dict(
            view="scores-boolean", dimensions=[], filters=[f_name_eq("fallback")],
            metrics=[{"measure": "value", "aggregation": "avg"}, {"measure": "value", "aggregation": "count"}])),

        # ── 8. Retrieval ──
        ("Retrieval — Latency P95 theo loại tool", dict(
            view="observations", dimensions=[{"field": "name"}], filters=[f_name_in(TOOL_NAMES)],
            metrics=[{"measure": "latency", "aggregation": "p95"}, {"measure": "latency", "aggregation": "avg"}])),
        ("Retrieval — Empty retrieval rate", dict(
            view="scores-boolean", dimensions=[], filters=[f_name_eq("retrieval_empty")],
            metrics=[{"measure": "value", "aggregation": "avg"}, {"measure": "value", "aggregation": "count"}])),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    to_ts = datetime.datetime.now(datetime.timezone.utc)
    from_ts = to_ts - datetime.timedelta(days=args.days)

    results: dict[str, object] = {}
    for label, q in build_queries():
        q["fromTimestamp"] = from_ts.isoformat()
        q["toTimestamp"] = to_ts.isoformat()
        try:
            res = api.metrics.metrics(query=json.dumps(q))
            d = res.dict() if hasattr(res, "dict") else res
            results[label] = d.get("data", d)
        except Exception as e:
            results[label] = {"ERROR": f"{type(e).__name__}: {e}"}

    print(json.dumps(results, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
