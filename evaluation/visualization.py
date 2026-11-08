"""Dashboard thống kê kết quả đánh giá (evaluation).

Tự nhận diện 2 định dạng file JSON:
- Mới — `run_evaluation.py` (đo đúng pipeline agentic thật, 1 pipeline duy nhất):
  {"summary": {...}, "detailed": [...]}
- Cũ — `retrieve/retrieval_strategies.py` (so sánh nhiều chiến lược độc lập):
  {"semantic_search": {"k=10": {...}}, "hybrid_search": {...}, ...}

Chạy: streamlit run evaluation/visualization.py
"""
import glob
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from metric_docs import FULL_GUIDE_MARKDOWN, METRIC_HELP
from live_log import read_live_log

st.set_page_config(page_title="Dashboard đánh giá RAG", layout="wide", page_icon="📊")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LIVE_PROGRESS_PATH = os.path.join(RESULTS_DIR, "_live_progress.json")

st.title("📊 Dashboard đánh giá hệ thống RAG")

with st.expander("📖 Giải thích các chỉ số đánh giá & Top-K", expanded=False):
    st.markdown(FULL_GUIDE_MARKDOWN)

st.markdown("---")


@st.fragment(run_every=3)
def _live_banner():
    """Tự refresh mỗi 3s trong lúc evaluate() đang chạy — không cần tải lại cả trang."""
    if not os.path.exists(LIVE_PROGRESS_PATH):
        return False
    try:
        with open(LIVE_PROGRESS_PATH, "r", encoding="utf-8") as f:
            live = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False  # đang ghi dở (os.replace là atomic nhưng phòng trường hợp đọc trúng lúc file bị xoá)

    status = live.get("status")
    completed, total = live.get("completed", 0), live.get("total", 0) or 1
    if status == "running":
        st.warning(f"🔴 **ĐANG CHẠY ĐÁNH GIÁ** — `{live.get('data_file')}` — {completed}/{total} câu hỏi")
        st.progress(completed / total)
    elif status == "error":
        st.error(f"⚠️ Lần chạy gần nhất bị dừng giữa chừng ({completed}/{total} câu hỏi) — xem terminal để biết lỗi.")
    return status == "running"


is_live_running = _live_banner()

# ════════════════════════════════════════════════════════════════════════════
# LỊCH SỬ CÂU HỎI THỰC TẾ — ghi tự động từ /rag/chat + /rag/stream, không cần
# chạy run_evaluation.py hay upload file JSON nào (xem evaluation/live_log.py).
# ════════════════════════════════════════════════════════════════════════════
st.header("💬 Lịch sử câu hỏi thực tế (live)")
st.caption(
    "Mọi câu hỏi gửi qua chat UI được ghi tự động ở đây — không cần chạy "
    "`run_evaluation.py` hay upload file. **Không tính được** Precision/Recall/"
    "ROUGE-L/BLEU (các chỉ số đó bắt buộc phải so với `gold_docs`/`gold_response` "
    "— câu hỏi tự do không có đáp án mẫu) — chỉ hiện chỉ số không cần đáp án mẫu: "
    "độ trung thực (grounding), số passage retrieve được, thời gian xử lý."
)


@st.fragment(run_every=5)
def _live_chat_section():
    rows = read_live_log()
    if not rows:
        st.info("Chưa có câu hỏi nào được hỏi qua chat UI — hỏi thử 1 câu trên giao diện rồi quay lại đây.")
        return

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("time", ascending=False).reset_index(drop=True)

    proceed_df = df[df["action"] == "proceed"]

    def _avg(colname: str):
        if proceed_df.empty or colname not in proceed_df.columns:
            return None
        v = proceed_df[colname].dropna().mean()
        return float(v) if pd.notna(v) else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Tổng câu hỏi", len(df))
    c2.metric("Số câu xử lý RAG", len(proceed_df))
    avg_g = _avg("grounding_score")
    c3.metric("Grounding TB", f"{avg_g:.3f}" if avg_g is not None else "—", help=METRIC_HELP["grounding"])
    avg_ar = _avg("answer_relevance")
    c4.metric("Answer Relevance TB", f"{avg_ar:.3f}" if avg_ar is not None else "—",
              help=METRIC_HELP["answer_relevance"])
    avg_cr = _avg("context_relevance")
    c5.metric("Context Relevance TB", f"{avg_cr:.3f}" if avg_cr is not None else "—",
              help=METRIC_HELP["context_relevance"])
    avg_t = df["elapsed_ms"].dropna().mean() if "elapsed_ms" in df.columns else None
    c6.metric("Xử lý TB", f"{avg_t / 1000:.1f}s" if avg_t is not None and pd.notna(avg_t) else "—")

    low_threshold = st.slider(
        "Ngưỡng cảnh báo độ trung thực thấp", 0.0, 1.0, 0.35, 0.05, key="live_threshold",
    )

    def _round_or_none(v):
        return round(v, 3) if v is not None and pd.notna(v) else None

    display_rows = []
    for _, r in df.iterrows():
        gs = r.get("grounding_score")
        has_gs = gs is not None and pd.notna(gs)
        et = r.get("elapsed_ms")
        display_rows.append({
            "Thời gian": r["time"].strftime("%Y-%m-%d %H:%M:%S"),
            "Câu hỏi": r["query"],
            "Hành động": r["action"],
            "Số passage": r.get("num_passages", 0),
            "Grounding": _round_or_none(gs),
            "Answer Rel.": _round_or_none(r.get("answer_relevance")),
            "Context Rel.": _round_or_none(r.get("context_relevance")),
            "⚠️ Nghi bịa": "⚠️" if (has_gs and gs < low_threshold) else "",
            "Xử lý (s)": round(et / 1000, 1) if et is not None and pd.notna(et) else None,
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    with st.expander("🔍 Xem chi tiết 1 câu hỏi"):
        idx = st.selectbox(
            "Chọn câu hỏi:", range(len(df)),
            format_func=lambda i: f"{df.iloc[i]['time'].strftime('%H:%M:%S')} — {df.iloc[i]['query'][:60]}",
            key="live_detail_picker",
        )
        row = df.iloc[idx]
        st.markdown(f"**Câu hỏi:** {row['query']}")
        if row.get("normalized_query"):
            st.markdown(f"**Câu hỏi đã rewrite:** {row['normalized_query']}")
        st.markdown(f"**Câu trả lời:** {row['answer']}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Grounding", _round_or_none(row.get("grounding_score")) or "—", help=METRIC_HELP["grounding"])
        m2.metric("Answer Relevance", _round_or_none(row.get("answer_relevance")) or "—",
                  help=METRIC_HELP["answer_relevance"])
        m3.metric("Context Relevance", _round_or_none(row.get("context_relevance")) or "—",
                  help=METRIC_HELP["context_relevance"])

        mapcs = row.get("retrieved_mapcs") or []
        if len(mapcs):
            st.markdown(f"**Điều luật đã dùng ({len(mapcs)}):** {', '.join(mapcs[:10])}")
        flagged = row.get("flagged_sentences") or []
        if len(flagged):
            st.warning("Câu nghi bịa: " + " | ".join(flagged))

    st.download_button(
        "📥 Export lịch sử (CSV)",
        pd.DataFrame(display_rows).to_csv(index=False),
        file_name="live_chat_log.csv",
        mime="text/csv",
    )


_live_chat_section()

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SO SÁNH VỚI BỘ TEST CHUẨN (cần gold_docs/gold_response) — từ run_evaluation.py
# ════════════════════════════════════════════════════════════════════════════
st.header("📂 So sánh với bộ test chuẩn (cần gold data)")

# ── Chọn nguồn file: có sẵn trong evaluation/results/ hoặc tự upload ───────────
existing_files = sorted(
    (f for f in glob.glob(os.path.join(RESULTS_DIR, "*.json")) if os.path.basename(f) != "_live_progress.json"),
    key=os.path.getmtime, reverse=True,
)
existing_names = [os.path.basename(f) for f in existing_files]

live_option = "🔴 Kết quả đang chạy realtime (_live_progress.json)"
options = ([live_option] if os.path.exists(LIVE_PROGRESS_PATH) else []) + existing_names

col_src1, col_src2 = st.columns([2, 1])
with col_src1:
    picked = st.selectbox(
        "📁 Chọn file kết quả có sẵn trong evaluation/results/",
        ["(không chọn)"] + options,
        index=1 if (is_live_running and options and options[0] == live_option) else 0,
    )
with col_src2:
    uploaded_file = st.file_uploader("...hoặc tải file JSON khác", type=["json"])

all_data = None
if uploaded_file is not None:
    all_data = json.load(uploaded_file)
elif picked == live_option:
    with open(LIVE_PROGRESS_PATH, "r", encoding="utf-8") as f:
        live = json.load(f)
    if live.get("summary"):
        all_data = {"summary": live["summary"], "detailed": live["detailed"]}
    else:
        st.info("Chưa có câu hỏi nào hoàn tất — đợi vài giây rồi xem lại.")
        st.stop()
elif picked != "(không chọn)":
    with open(os.path.join(RESULTS_DIR, picked), "r", encoding="utf-8") as f:
        all_data = json.load(f)

if all_data is None:
    st.info("👆 Chọn hoặc tải lên một file kết quả đánh giá để xem thống kê.")
    st.markdown(
        """
        ### Cách tạo file kết quả

        **Đánh giá đúng pipeline agentic thật (khuyến nghị):**
        ```bash
        cd evaluation
        python run_evaluation.py --limit 2 --yes   # thử rẻ trước — tốn phí OpenAI thật
        python run_evaluation.py --yes             # chạy toàn bộ data.json
        ```
        Kết quả lưu ở `evaluation/results/agentic_evaluation_results.json`.

        **So sánh các chiến lược retrieval độc lập (tham khảo, không phải pipeline production):**
        ```bash
        python run_evaluation.py  # (bản cũ retrieve/retrieval_strategies.py, nếu còn dùng)
        ```
        """
    )
    st.stop()

IS_NEW_FORMAT = isinstance(all_data, dict) and "summary" in all_data and "detailed" in all_data

# ════════════════════════════════════════════════════════════════════════════
# ĐỊNH DẠNG MỚI — 1 pipeline agentic thật
# ════════════════════════════════════════════════════════════════════════════
if IS_NEW_FORMAT:
    summary = all_data["summary"]
    detailed = all_data["detailed"]
    top_k_values = sorted(int(k.split("=")[1]) for k in summary["retrieval_metrics"].keys())

    st.caption("Đo trực tiếp trên `run_agentic_rag()` — đúng pipeline đang chạy trong backend/rag_api.py")

    all_retrieval_zero = all(
        v == 0 for k_vals in summary["retrieval_metrics"].values() for v in k_vals.values()
    )
    if all_retrieval_zero and summary["num_queries"] > 0:
        st.warning(
            "⚠️ **Retrieval metrics (Precision/Recall/Hit/MRR/NDCG) đều bằng 0 ở mọi k** — "
            "nhiều khả năng do `gold_docs` trong `data.json` bị lệch sau khi crawl lại dữ liệu "
            "(xem mục 📖 Giải thích ở đầu trang, phần \"Hạn chế đã biết\"). "
            "**Nên đọc ROUGE-L/BLEU/Semantic Similarity/grounding bên dưới để đánh giá chất lượng thật.**"
        )

    # ── Thẻ tổng quan ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Số câu hỏi đã test", summary["num_queries"])
    resp = summary.get("response_metrics") or {}
    c2.metric("ROUGE-L", f"{resp.get('rouge_l', 0):.3f}" if resp else "—", help=METRIC_HELP["rouge_l"])
    c3.metric("BLEU", f"{resp.get('bleu', 0):.3f}" if resp else "—", help=METRIC_HELP["bleu"])
    c4.metric("Semantic Similarity", f"{resp.get('semantic_similarity', 0):.3f}" if resp else "—",
              help=METRIC_HELP["semantic_similarity"])
    g = summary.get("avg_grounding_score")
    c5.metric("Độ trung thực TB (grounding)", f"{g:.3f}" if g is not None else "—", help=METRIC_HELP["grounding"])

    st.markdown("---")

    # ── Biểu đồ retrieval metrics theo k ─────────────────────────────────────
    st.header("📈 Retrieval metrics theo Top-K")
    metric_names = ["Precision@K", "Recall@K", "Hit@K", "MRR@K", "NDCG@K"]
    metric_keys = ["precision", "recall", "hit", "mrr", "ndcg"]

    fig = make_subplots(rows=2, cols=3, subplot_titles=metric_names)
    colors = ["#667eea", "#f093fb", "#4facfe", "#ef034a", "#22d031"]
    for idx, (name, key) in enumerate(zip(metric_names, metric_keys)):
        row, col = idx // 3 + 1, idx % 3 + 1
        values = [summary["retrieval_metrics"][f"k={k}"].get(key, 0) for k in top_k_values]
        fig.add_trace(
            go.Scatter(x=top_k_values, y=values, mode="lines+markers",
                       line=dict(color=colors[idx % len(colors)], width=2), marker=dict(size=8),
                       showlegend=False),
            row=row, col=col,
        )
        fig.update_xaxes(title_text="Top-K", row=row, col=col)
        fig.update_yaxes(title_text="Score", range=[0, 1], row=row, col=col)
    fig.update_layout(height=600, margin=dict(t=60, b=40, l=60, r=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "💡 So Recall@10 với Recall@3: nếu Recall@10 cao nhưng Recall@3 thấp → điều luật đúng "
        "**có được tìm thấy** nhưng rerank xếp hạng chưa tốt. Nếu Recall@10 vẫn thấp → điều luật "
        "đúng **không nằm trong tập ứng viên** ngay từ đầu (lỗi ở vector_search, không phải rerank)."
    )

    st.markdown("---")

    # ── Bảng chi tiết từng câu hỏi ────────────────────────────────────────────
    st.header("📋 Chi tiết từng câu hỏi")
    low_grounding_threshold = st.slider(
        "Ngưỡng cảnh báo độ trung thực thấp (câu trả lời có nguy cơ “bịa”)",
        0.0, 1.0, 0.35, 0.05,
    )

    rows = []
    for r in detailed:
        gold = set(r.get("gold_mapcs", []))
        retrieved = r.get("retrieved_mapcs", [])
        hit_count = len(gold & set(retrieved))
        gs = r.get("grounding_score")
        rows.append({
            "Câu hỏi": r["query"],
            "Điều đúng tìm được": f"{hit_count}/{len(gold)}" if gold else "—",
            "Độ trung thực": round(gs, 3) if gs is not None else None,
            "⚠️ Nghi bịa": "⚠️" if (gs is not None and gs < low_grounding_threshold) else "",
            "Câu trả lời (rút gọn)": (r.get("answer") or "")[:160] + ("…" if len(r.get("answer") or "") > 160 else ""),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Xem chi tiết đầy đủ từng câu hỏi (đủ metric @ mọi k, không chỉ rút gọn) ──
    with st.expander("🔍 Xem chi tiết toàn bộ metric của 1 câu hỏi"):
        q_idx = st.selectbox(
            "Chọn câu hỏi:", range(len(detailed)),
            format_func=lambda i: detailed[i]["query"], key="detail_q_picker",
        )
        r = detailed[q_idx]
        st.markdown(f"**Câu hỏi:** {r['query']}")
        st.markdown(f"**Câu trả lời:** {r.get('answer', '') or '_(chưa có — câu hỏi này chưa chạy xong)_'}")

        ret_metrics = r.get("retrieval_metrics") or {}
        if ret_metrics:
            k_rows = []
            for k_key, vals in ret_metrics.items():
                k_rows.append({"Top-K": k_key, **{m.capitalize(): round(v, 3) for m, v in vals.items()}})
            st.dataframe(pd.DataFrame(k_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Mỗi hàng là 1 giá trị k khác nhau — xem mục 📖 Giải thích ở đầu trang để biết ý nghĩa từng cột."
            )

        resp_m = r.get("response_metrics") or {}
        gs = r.get("grounding_score")
        if resp_m or gs is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ROUGE-L", f"{resp_m.get('rouge_l', 0):.3f}" if resp_m else "—", help=METRIC_HELP["rouge_l"])
            m2.metric("BLEU", f"{resp_m.get('bleu', 0):.3f}" if resp_m else "—", help=METRIC_HELP["bleu"])
            m3.metric("Semantic Similarity", f"{resp_m.get('semantic_similarity', 0):.3f}" if resp_m else "—",
                      help=METRIC_HELP["semantic_similarity"])
            m4.metric("Độ trung thực", f"{gs:.3f}" if gs is not None else "—", help=METRIC_HELP["grounding"])

    flagged = [r for r in detailed if r.get("grounding_score") is not None and r["grounding_score"] < low_grounding_threshold]
    if flagged:
        st.warning(f"{len(flagged)} câu trả lời có độ trung thực dưới ngưỡng {low_grounding_threshold} — xem chi tiết bên dưới.")
        for r in flagged:
            with st.expander(f"⚠️ {r['query']}"):
                st.write("**Câu trả lời:**", r.get("answer", ""))
                st.write("**Điểm trung thực:**", r.get("grounding_score"))

    st.download_button(
        "📥 Export bảng chi tiết (CSV)",
        df.to_csv(index=False),
        file_name="evaluation_detailed.csv",
        mime="text/csv",
    )

# ════════════════════════════════════════════════════════════════════════════
# ĐỊNH DẠNG CŨ — so sánh nhiều chiến lược retrieval độc lập
# ════════════════════════════════════════════════════════════════════════════
else:
    st.caption("Định dạng cũ — so sánh các chiến lược trong retrieve/retrieval_strategies.py (không phải pipeline production)")
    strategies = list(all_data.keys())
    first_strategy = strategies[0]
    top_k_values = sorted(int(k.split("=")[1]) for k in all_data[first_strategy].keys())

    st.sidebar.header("🎯 Lựa chọn")
    selected_strategies = st.sidebar.multiselect("Chọn strategies để so sánh:", strategies, default=strategies)

    if not selected_strategies:
        st.warning("Vui lòng chọn ít nhất 1 strategy")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📊 Retrieval Metrics", "📝 Response Metrics", "📋 Bảng so sánh"])
    colors = ["#667eea", "#f093fb", "#4facfe", "#ef034a", "#22d031", "#30cfd0"]

    with tab1:
        metric_names = ["Precision@K", "Recall@K", "Hit@K", "MRR@K", "NDCG@K"]
        metric_keys = ["precision@k", "recall@k", "hit@k", "mrr@k", "ndcg@k"]
        fig = make_subplots(rows=2, cols=3, subplot_titles=metric_names)
        for idx, (name, key) in enumerate(zip(metric_names, metric_keys)):
            row, col = idx // 3 + 1, idx % 3 + 1
            for i, strategy in enumerate(selected_strategies):
                values = [all_data[strategy][f"k={k}"]["average_metrics"]["retrieval_metrics"].get(key, 0) for k in top_k_values]
                fig.add_trace(
                    go.Scatter(x=top_k_values, y=values, mode="lines+markers", name=strategy,
                               line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=8),
                               showlegend=(idx == 0)),
                    row=row, col=col,
                )
            fig.update_xaxes(title_text="Top-K", row=row, col=col)
            fig.update_yaxes(title_text="Score", row=row, col=col)
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        metric_names = ["ROUGE-L", "BLEU", "Semantic Similarity"]
        metric_keys = ["rouge_l", "bleu", "semantic_similarity"]
        fig2 = make_subplots(rows=1, cols=3, subplot_titles=metric_names)
        for idx, (name, key) in enumerate(zip(metric_names, metric_keys)):
            for i, strategy in enumerate(selected_strategies):
                values = [all_data[strategy][f"k={k}"]["average_metrics"]["response_metrics"][key] for k in top_k_values]
                fig2.add_trace(
                    go.Scatter(x=top_k_values, y=values, mode="lines+markers", name=strategy,
                               line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=8),
                               showlegend=(idx == 0)),
                    row=1, col=idx + 1,
                )
        fig2.update_layout(height=450, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        selected_k = st.selectbox("Chọn Top-K:", top_k_values, index=len(top_k_values) // 2)
        key = f"k={selected_k}"
        table = []
        for strategy in selected_strategies:
            ret = all_data[strategy][key]["average_metrics"]["retrieval_metrics"]
            resp = all_data[strategy][key]["average_metrics"]["response_metrics"]
            table.append({"Strategy": strategy, **ret, **resp})
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
