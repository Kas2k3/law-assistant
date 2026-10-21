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

st.set_page_config(page_title="Dashboard đánh giá RAG", layout="wide", page_icon="📊")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

st.title("📊 Dashboard đánh giá hệ thống RAG")
st.markdown("---")

# ── Chọn nguồn file: có sẵn trong evaluation/results/ hoặc tự upload ───────────
existing_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")), key=os.path.getmtime, reverse=True)
existing_names = [os.path.basename(f) for f in existing_files]

col_src1, col_src2 = st.columns([2, 1])
with col_src1:
    picked = st.selectbox(
        "📁 Chọn file kết quả có sẵn trong evaluation/results/",
        ["(không chọn)"] + existing_names,
    )
with col_src2:
    uploaded_file = st.file_uploader("...hoặc tải file JSON khác", type=["json"])

all_data = None
if uploaded_file is not None:
    all_data = json.load(uploaded_file)
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

    # ── Thẻ tổng quan ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Số câu hỏi đã test", summary["num_queries"])
    resp = summary.get("response_metrics") or {}
    c2.metric("ROUGE-L", f"{resp.get('rouge_l', 0):.3f}" if resp else "—")
    c3.metric("BLEU", f"{resp.get('bleu', 0):.3f}" if resp else "—")
    c4.metric("Semantic Similarity", f"{resp.get('semantic_similarity', 0):.3f}" if resp else "—")
    g = summary.get("avg_grounding_score")
    c5.metric("Độ trung thực TB (grounding)", f"{g:.3f}" if g is not None else "—")

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
