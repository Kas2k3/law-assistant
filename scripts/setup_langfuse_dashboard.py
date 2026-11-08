"""
Thiết lập Langfuse cho toàn bộ bộ chỉ số đánh giá (thay cho dashboard Streamlit
tự build): Score Configs + 1 Dashboard "RAG Ops Overview" với các widget phủ
đủ 8 nhóm người dùng yêu cầu + 1 Annotation Queue để chấm tay "routing accuracy"
(không thể tự động hoá vì không có nhãn đúng/sai sẵn có).

Idempotent: chạy lại an toàn — bỏ qua nếu đã có config/widget/dashboard/queue
cùng tên.

Yêu cầu: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL trong .env
Chạy: python scripts/setup_langfuse_dashboard.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langfuse import get_client
from langfuse.api.commons.types.config_category import ConfigCategory
from langfuse.api.unstable.dashboard_widgets.types import (
    DashboardWidgetChartType as Chart,
    DashboardWidgetDimension as Dim,
    DashboardWidgetFilter as Filt,
    DashboardWidgetMetric as Metric,
    DashboardWidgetMetricAggregation as Agg,
    DashboardWidgetView as View,
)
from langfuse.api.unstable.dashboards.types import CreateDashboardPlacementRequest_Widget as Placement

client = get_client()
if not client.auth_check():
    sys.exit("❌ Langfuse auth thất bại — kiểm tra LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL trong .env")
api = client.api

REQUEST_NAMES = ["rag_chat_request", "rag_stream_request"]
TOOL_NAMES = ["tool:vector_search", "tool:graph_traverse", "tool:web_search", "tool:get_full_article"]


# ───────────────────────── 1. Score configs ─────────────────────────
OUTCOME_LABELS = ["quick", "spam", "escalate", "rejected", "proceed", "error"]

SCORE_CONFIGS = [
    dict(name="faithfulness", data_type="NUMERIC", min_value=0, max_value=1,
         description="RAG Quality — câu trả lời có bám sát passage đã truy xuất không (embedding cục bộ, 0-1)"),
    dict(name="answer_relevancy", data_type="NUMERIC", min_value=0, max_value=1,
         description="RAG Quality — câu trả lời có đúng trọng tâm câu hỏi không (embedding cục bộ, 0-1)"),
    dict(name="context_precision", data_type="NUMERIC", min_value=0, max_value=1,
         description="RAG Quality — các đoạn truy xuất có liên quan câu hỏi không (embedding cục bộ, 0-1)"),
    dict(name="fallback", data_type="BOOLEAN",
         description="User Experience — true nếu hết max_iterations mà chưa trả lời tự nhiên, phải ép tổng hợp"),
    dict(name="empty_retrieval", data_type="BOOLEAN",
         description="Retrieval — true nếu cả request không truy xuất được passage nào"),
    dict(name="unsafe_output", data_type="BOOLEAN",
         description="Safety — OpenAI Moderation API gắn cờ câu trả lời cuối cùng"),
    dict(name="guardrail_pass", data_type="BOOLEAN",
         description="Safety — false nếu bị chặn bởi guardrail (spam/escalate/rejected)"),
    dict(name="request_error", data_type="BOOLEAN",
         description="Reliability — true nếu nhánh proceed gặp exception"),
    dict(name="timeout", data_type="BOOLEAN",
         description="Reliability — true nếu lỗi là timeout (chỉ có ý nghĩa khi request_error=true)"),
    dict(name="tool_success", data_type="BOOLEAN",
         description="Agent Quality — tool có trả lỗi không (chấm theo từng lần gọi tool)"),
    dict(name="retrieval_empty", data_type="BOOLEAN",
         description="Retrieval — 1 lần gọi tool truy xuất (vector/graph/web/full_article) có trả về 0 kết quả không"),
    dict(name="request_outcome", data_type="CATEGORICAL",
         categories=[ConfigCategory(value=float(i), label=v) for i, v in enumerate(OUTCOME_LABELS)],
         description="Request log — nhánh xử lý cuối cùng của request"),
]


def setup_score_configs() -> None:
    existing = {c.name for c in api.score_configs.get(limit=100).data}
    for cfg in SCORE_CONFIGS:
        if cfg["name"] in existing:
            print(f"  [skip] score_config '{cfg['name']}' đã tồn tại")
            continue
        try:
            api.score_configs.create(**cfg)
            print(f"  [ok]   score_config '{cfg['name']}' đã tạo")
        except Exception as e:
            print(f"  [err]  score_config '{cfg['name']}': {e}")


# ───────────────────────── 2. Dashboard widgets ─────────────────────────
def f_name_in(names: list[str]) -> Filt:
    return Filt(column="name", operator="any of", type="stringOptions", value=names)


def f_name_eq(name: str) -> Filt:
    return Filt(column="name", operator="=", type="string", value=name)


WIDGETS = [
    # ── 1. Performance ──────────────────────────────────────────────
    dict(name="Performance — Request latency P50/P95/P99", view=View.OBSERVATIONS,
         dimensions=[], filters=[f_name_in(REQUEST_NAMES)], chart_type=Chart.LINE_TIME_SERIES,
         metrics=[Metric(measure="latency", agg=Agg.P50), Metric(measure="latency", agg=Agg.P95),
                  Metric(measure="latency", agg=Agg.P99)],
         description="Độ trễ end-to-end của /rag/chat + /rag/stream theo thời gian"),
    dict(name="Performance — Request volume (log)", view=View.OBSERVATIONS,
         dimensions=[Dim(field="name")], filters=[f_name_in(REQUEST_NAMES)], chart_type=Chart.BAR_TIME_SERIES,
         metrics=[Metric(measure="count", agg=Agg.COUNT)],
         description="Số lượng request theo thời gian — xem chi tiết từng request ở tab Traces"),

    # ── 2. Cost ──────────────────────────────────────────────────────
    dict(name="Cost — Token input/output theo thời gian", view=View.OBSERVATIONS,
         dimensions=[], filters=[], chart_type=Chart.AREA_TIME_SERIES,
         metrics=[Metric(measure="inputTokens", agg=Agg.SUM), Metric(measure="outputTokens", agg=Agg.SUM)],
         description="Tổng token input/output mỗi ngày (mọi LLM call: chat, rerank, intent...)"),
    dict(name="Cost — Chi phí ($) theo thời gian", view=View.OBSERVATIONS,
         dimensions=[], filters=[], chart_type=Chart.LINE_TIME_SERIES,
         metrics=[Metric(measure="totalCost", agg=Agg.SUM)],
         description="Tổng chi phí OpenAI mỗi ngày, tự tính theo bảng giá model"),
    dict(name="Cost — Chi phí theo model", view=View.OBSERVATIONS,
         dimensions=[Dim(field="providedModelName")], filters=[], chart_type=Chart.HORIZONTAL_BAR,
         metrics=[Metric(measure="totalCost", agg=Agg.SUM)],
         description="Model nào tốn tiền nhất (chat model vs embedding vs rerank)"),

    # ── 3. Reliability ───────────────────────────────────────────────
    dict(name="Reliability — Error rate", view=View.SCORES_BOOLEAN,
         dimensions=[], filters=[f_name_eq("request_error")], chart_type=Chart.NUMBER,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Tỉ lệ request (nhánh proceed) gặp exception"),
    dict(name="Reliability — Timeout rate", view=View.SCORES_BOOLEAN,
         dimensions=[], filters=[f_name_eq("timeout")], chart_type=Chart.NUMBER,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Tỉ lệ lỗi là timeout (trong số request gặp lỗi)"),
    dict(name="Reliability — Error/Timeout rate theo thời gian", view=View.SCORES_BOOLEAN,
         dimensions=[], filters=[f_name_in(["request_error", "timeout"])], chart_type=Chart.LINE_TIME_SERIES,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Xu hướng error rate / timeout rate theo thời gian"),

    # ── 4. RAG Quality ───────────────────────────────────────────────
    dict(name="RAG Quality — Faithfulness theo thời gian", view=View.SCORES_NUMERIC,
         dimensions=[], filters=[f_name_eq("faithfulness")], chart_type=Chart.LINE_TIME_SERIES,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Độ trung thực câu trả lời so với passage đã truy xuất (0-1, trung bình mỗi ngày)"),
    dict(name="RAG Quality — Answer relevancy theo thời gian", view=View.SCORES_NUMERIC,
         dimensions=[], filters=[f_name_eq("answer_relevancy")], chart_type=Chart.LINE_TIME_SERIES,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Câu trả lời có đúng trọng tâm câu hỏi không (0-1, trung bình mỗi ngày)"),
    dict(name="RAG Quality — Context precision theo thời gian", view=View.SCORES_NUMERIC,
         dimensions=[], filters=[f_name_eq("context_precision")], chart_type=Chart.LINE_TIME_SERIES,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Các đoạn truy xuất có liên quan câu hỏi không (0-1, trung bình mỗi ngày). "
                     "LƯU Ý: context_recall KHÔNG có trong dashboard — cần gold_docs đáng tin cậy để tính, "
                     "hiện data.json dùng offset MySQL đã lỗi thời (xem README mục Known limitation)."),

    # ── 5. Agent Quality ─────────────────────────────────────────────
    dict(name="Agent Quality — Tool call count theo loại tool", view=View.OBSERVATIONS,
         dimensions=[Dim(field="calledToolNames")], filters=[], chart_type=Chart.VERTICAL_BAR,
         metrics=[Metric(measure="count", agg=Agg.COUNT)],
         description="Số lần mỗi loại tool (vector_search/graph_traverse/web_search/get_full_article) được gọi"),
    dict(name="Agent Quality — Tool success rate theo loại tool", view=View.SCORES_BOOLEAN,
         dimensions=[Dim(field="observationName")], filters=[f_name_eq("tool_success")], chart_type=Chart.HORIZONTAL_BAR,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Tỉ lệ tool chạy không lỗi, chia theo từng loại tool. "
                     "routing_accuracy KHÔNG có ở đây — không có nhãn 'tool đúng' tự động, "
                     "xem Annotation Queue 'routing-accuracy-review' để chấm tay."),

    # ── 6. Safety ────────────────────────────────────────────────────
    dict(name="Safety — Guardrail pass rate", view=View.SCORES_BOOLEAN,
         dimensions=[], filters=[f_name_eq("guardrail_pass")], chart_type=Chart.NUMBER,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Tỉ lệ request KHÔNG bị chặn bởi guardrail (spam/escalate/rejected)"),
    dict(name="Safety — Unsafe output rate", view=View.SCORES_BOOLEAN,
         dimensions=[], filters=[f_name_eq("unsafe_output")], chart_type=Chart.NUMBER,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Tỉ lệ câu trả lời bị OpenAI Moderation gắn cờ"),
    dict(name="Safety / Request log — Phân bố outcome", view=View.SCORES_CATEGORICAL,
         dimensions=[Dim(field="stringValue")], filters=[f_name_eq("request_outcome")], chart_type=Chart.PIE,
         metrics=[Metric(measure="count", agg=Agg.COUNT)],
         description="Tỉ lệ quick/spam/escalate/rejected/proceed/error trong toàn bộ request"),

    # ── 7. User Experience ───────────────────────────────────────────
    dict(name="User Experience — Fallback rate", view=View.SCORES_BOOLEAN,
         dimensions=[], filters=[f_name_eq("fallback")], chart_type=Chart.NUMBER,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Tỉ lệ request phải ép tổng hợp câu trả lời vì hết max_iterations"),

    # ── 8. Retrieval ─────────────────────────────────────────────────
    dict(name="Retrieval — Latency P95 theo loại tool", view=View.OBSERVATIONS,
         dimensions=[Dim(field="name")], filters=[f_name_in(TOOL_NAMES)], chart_type=Chart.HORIZONTAL_BAR,
         metrics=[Metric(measure="latency", agg=Agg.P95)],
         description="Độ trễ P95 của từng loại tool truy xuất"),
    dict(name="Retrieval — Empty retrieval rate", view=View.SCORES_BOOLEAN,
         dimensions=[], filters=[f_name_eq("retrieval_empty")], chart_type=Chart.NUMBER,
         metrics=[Metric(measure="value", agg=Agg.AVG)],
         description="Tỉ lệ 1 lần gọi tool truy xuất trả về 0 kết quả"),
]


def setup_widgets() -> dict[str, str]:
    existing = {w.name: w.id for w in api.unstable.dashboard_widgets.list(limit=100).data}
    widget_ids: dict[str, str] = {}
    for w in WIDGETS:
        if w["name"] in existing:
            print(f"  [skip] widget '{w['name']}' đã tồn tại")
            widget_ids[w["name"]] = existing[w["name"]]
            continue
        try:
            created = api.unstable.dashboard_widgets.create(**w)
            widget_ids[w["name"]] = created.id
            print(f"  [ok]   widget '{w['name']}' đã tạo")
        except Exception as e:
            print(f"  [err]  widget '{w['name']}': {e}")
    return widget_ids


# ───────────────────────── 3. Dashboard + placements ─────────────────────────
DASHBOARD_NAME = "RAG Ops Overview"
COLS = 2  # 2 widget/hàng, mỗi widget rộng 6/12 cột


def setup_dashboard(widget_ids: dict[str, str]) -> None:
    existing = [d for d in api.unstable.dashboards.list(limit=100).data if d.name == DASHBOARD_NAME]
    if existing:
        dash_id = existing[0].id
        print(f"  [skip] dashboard '{DASHBOARD_NAME}' đã tồn tại (id={dash_id}) — không thêm lại placement")
        print(f"         Mở: xem project Langfuse > Dashboards > {DASHBOARD_NAME}")
        return

    dash = api.unstable.dashboards.create(
        name=DASHBOARD_NAME,
        description="Tổng hợp Performance/Cost/Reliability/RAG Quality/Agent Quality/Safety/UX/Retrieval cho hệ thống RAG pháp luật",
    )
    print(f"  [ok]   dashboard '{DASHBOARD_NAME}' đã tạo (id={dash.id})")

    for i, w in enumerate(WIDGETS):
        wid = widget_ids.get(w["name"])
        if not wid:
            continue
        row, col = divmod(i, COLS)
        try:
            api.unstable.dashboards.add_placement(
                dash.id,
                request=Placement(widget_id=wid, x=col * 6, y=row * 4, width=6, height=4),
            )
        except Exception as e:
            print(f"  [err]  placement cho '{w['name']}': {e}")
    print(f"  [ok]   đã xếp {len(WIDGETS)} widget vào dashboard")


# ───────────────────────── 4. Annotation queue (routing accuracy) ─────────────────────────
ROUTING_QUEUE_NAME = "routing-accuracy-review"


def setup_routing_annotation_queue() -> None:
    """routing_accuracy (tool được chọn có đúng không) không thể tự động hoá —
    không có nhãn 'đúng' sẵn có để so sánh. Tạo sẵn 1 annotation queue + score
    config CATEGORICAL để con người review mẫu trace và chấm tay trong Langfuse UI
    (Traces > chọn trace > Add to queue 'routing-accuracy-review')."""
    cfg_name = "routing_accuracy_human"
    existing_cfgs = {c.name: c.id for c in api.score_configs.get(limit=100).data}
    if cfg_name in existing_cfgs:
        cfg_id = existing_cfgs[cfg_name]
        print(f"  [skip] score_config '{cfg_name}' đã tồn tại")
    else:
        cfg = api.score_configs.create(
            name=cfg_name, data_type="CATEGORICAL",
            categories=[ConfigCategory(value=1.0, label="correct"), ConfigCategory(value=0.0, label="incorrect")],
            description="Chấm tay: agent có chọn đúng tool (vector_search/graph_traverse/web_search/get_full_article) không",
        )
        cfg_id = cfg.id
        print(f"  [ok]   score_config '{cfg_name}' đã tạo")

    existing_queues = {q.name for q in api.annotation_queues.list_queues(limit=100).data}
    if ROUTING_QUEUE_NAME in existing_queues:
        print(f"  [skip] annotation queue '{ROUTING_QUEUE_NAME}' đã tồn tại")
        return
    try:
        api.annotation_queues.create_queue(
            name=ROUTING_QUEUE_NAME, score_config_ids=[cfg_id],
            description="Thả vào đây 1 mẫu trace để review tay: agent có gọi đúng tool không? "
                         "(Agent Quality > routing accuracy — không tự động hoá được vì thiếu nhãn đúng/sai)",
        )
        print(f"  [ok]   annotation queue '{ROUTING_QUEUE_NAME}' đã tạo")
    except Exception as e:
        print(f"  [err]  annotation queue: {e}")


if __name__ == "__main__":
    print("1) Score configs")
    setup_score_configs()
    print("\n2) Dashboard widgets")
    wids = setup_widgets()
    print("\n3) Dashboard + placements")
    setup_dashboard(wids)
    print("\n4) Annotation queue cho routing accuracy (chấm tay)")
    setup_routing_annotation_queue()
    print("\n✅ Xong. Mở Langfuse > Dashboards > 'RAG Ops Overview'.")
