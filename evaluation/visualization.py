import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Kết Quả Đánh Giá Strategies", layout="wide", page_icon="📊")

st.title("📊 Kết Quả Đánh Giá Các Chiến Lược Retrieval")
st.markdown("So sánh hiệu suất của các chiến lược với nhiều giá trị TOP_K")
st.markdown("---")

# Upload file JSON
uploaded_file = st.file_uploader("📁 Tải file kết quả JSON", type=['json'])

if uploaded_file is not None:
    # Load data
    all_data = json.load(uploaded_file)
    
    # Extract strategies và top_k values
    strategies = list(all_data.keys())
    
    # Lấy top_k values từ strategy đầu tiên
    first_strategy = strategies[0]
    top_k_values = sorted([int(k.split("=")[1]) for k in all_data[first_strategy].keys()])
    
    # st.success(f"✔ Đã load {len(strategies)} strategies với {len(top_k_values)} giá trị TOP_K")
    st.toast(f"✔ Đã load {len(strategies)} strategies với {len(top_k_values)} giá trị TOP_K", icon="✅")
    
    # Sidebar để chọn strategies
    st.sidebar.header("🎯 Lựa Chọn")
    selected_strategies = st.sidebar.multiselect(
        "Chọn strategies để so sánh:",
        strategies,
        default=strategies
    )
    
    if not selected_strategies:
        st.warning("Vui lòng chọn ít nhất 1 strategy")
    else:
        # Tabs cho từng loại metrics
        tab1, tab2, tab3 = st.tabs(["📊 Retrieval Metrics", "📝 Response Metrics", "📋 Bảng So Sánh"])
        
        with tab1:
            st.header("📈 Retrieval Metrics")
            
            # Prepare data for charts - thêm MRR@K
            retrieval_metrics = ["Precision@K", "Recall@K", "Hit@K", "MRR@K", "NDCG@K"] #"Precision@K", "Recall@K", "Hit@K", "MRR@K", "NDCG@K"
            metric_keys = ["precision@k", "recall@k", "hit@k", "mrr@k", "ndcg@k"]   #"precision@k", "recall@k", "hit@k", "mrr@k", "ndcg@k"
            
            # Create 2x3 subplots (thêm 1 chart cho MRR)
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=retrieval_metrics,
                specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}, None]],
                vertical_spacing=0.15,
                horizontal_spacing=0.10
            )
            
            colors = ['#667eea', '#f093fb', '#4facfe', "#ef034a", "#22d031", '#30cfd0']
            
            for idx, (metric_name, metric_key) in enumerate(zip(retrieval_metrics, metric_keys)):
                row = idx // 3 + 1
                col = idx % 3 + 1
                
                for i, strategy in enumerate(selected_strategies):
                    values = []
                    for k in top_k_values:
                        key = f"k={k}"
                        value = all_data[strategy][key]["average_metrics"]["retrieval_metrics"].get(metric_key, 0)
                        values.append(value)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=top_k_values,
                            y=values,
                            mode='lines+markers',
                            name=strategy,
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=8),
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )
                
                fig.update_xaxes(title_text="TOP_K", row=row, col=col)
                fig.update_yaxes(title_text="Score", row=row, col=col)
            
            fig.update_layout(
                height=900,
                margin=dict(t=150, b=80, l=80, r=80),
                showlegend=True,
                legend=dict(
                    orientation="h",    
                    yanchor="bottom",  
                    y=1.08,
                    xanchor="center", 
                    x=0.5
                )
                # margin=dict(t=120, b=80, l=60, r=200),
                # showlegend=True,
                # legend=dict(
                #     orientation="v",
                #     yanchor="middle",
                #     y=0.8,
                #     xanchor="center",
                #     x=0.75
                # )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("📝 Response Metrics")
            
            # Response metrics charts
            response_metrics = ["ROUGE-L", "BLEU", "Semantic Similarity"]
            response_keys = ["rouge_l", "bleu", "semantic_similarity"]
            
            fig2 = make_subplots(
                rows=1, cols=3,
                subplot_titles=response_metrics,
                horizontal_spacing=0.12
            )
            
            for idx, (metric_name, metric_key) in enumerate(zip(response_metrics, response_keys)):
                col = idx + 1
                
                for i, strategy in enumerate(selected_strategies):
                    values = []
                    for k in top_k_values:
                        key = f"k={k}"
                        value = all_data[strategy][key]["average_metrics"]["response_metrics"][metric_key]
                        values.append(value)
                    
                    fig2.add_trace(
                        go.Scatter(
                            x=top_k_values,
                            y=values,
                            mode='lines+markers',
                            name=strategy,
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=8),
                            showlegend=(idx == 0)
                        ),
                        row=1, col=col
                    )
                
                fig2.update_xaxes(title_text="TOP_K", row=1, col=col)
                fig2.update_yaxes(title_text="Score", row=1, col=col)
            
            fig2.update_layout(
                height=500,
                margin=dict(t=120, b=80, l=60, r=200),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                )
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.header("📋 Bảng So Sánh Chi Tiết")
            
            # Chọn TOP_K để hiển thị
            selected_k = st.selectbox("Chọn TOP_K:", top_k_values, index=len(top_k_values)//2)
            
            key = f"k={selected_k}"
            
            # Retrieval metrics table - thêm MRR@K
            st.subheader(f"📊 Retrieval Metrics @ TOP_{selected_k}")
            
            retrieval_table = []
            for strategy in selected_strategies:
                metrics = all_data[strategy][key]["average_metrics"]["retrieval_metrics"]
                retrieval_table.append({
                    "Strategy": strategy,
                    "Precision@K": metrics['precision@k'],
                    "Recall@K": metrics['recall@k'],
                    "Hit@K": metrics['hit@k'],
                    "MRR@K": metrics.get('mrr@k', 0),
                    "NDCG@K": metrics['ndcg@k']
                })
            
            df_ret = pd.DataFrame(retrieval_table)
            
            # Hàm highlight giá trị max trong mỗi cột
            def highlight_max(s):
                if s.dtype == 'object':  # Bỏ qua cột Strategy
                    return [''] * len(s)
                is_max = s == s.max()
                return ['font-weight: bold' if v else '' for v in is_max]   #; background-color: #d4edda
            
            # Format số và highlight max
            styled_df_ret = df_ret.style.apply(highlight_max).format({
                "Precision@K": "{:.4f}",
                "Recall@K": "{:.4f}",
                "Hit@K": "{:.4f}",
                "MRR@K": "{:.4f}",
                "NDCG@K": "{:.4f}"
            })
            
            st.dataframe(styled_df_ret, use_container_width=True, hide_index=True)
            
            # Response metrics table
            st.subheader(f"📝 Response Metrics @ TOP_{selected_k}")
            
            response_table = []
            for strategy in selected_strategies:
                metrics = all_data[strategy][key]["average_metrics"]["response_metrics"]
                response_table.append({
                    "Strategy": strategy,
                    "ROUGE-L": metrics['rouge_l'],
                    "BLEU": metrics['bleu'],
                    "Semantic Similarity": metrics['semantic_similarity']
                })
            
            df_resp = pd.DataFrame(response_table)
            
            # Format số và highlight max
            styled_df_resp = df_resp.style.apply(highlight_max).format({
                "ROUGE-L": "{:.4f}",
                "BLEU": "{:.4f}",
                "Semantic Similarity": "{:.4f}"
            })
            
            st.dataframe(styled_df_resp, use_container_width=True, hide_index=True)
            
            # Best metrics summary - thêm MRR@K
            st.subheader("🏆 Chiến Lược Tốt Nhất")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Retrieval Metrics:**")
                for metric_name, metric_key in zip(retrieval_metrics, metric_keys):
                    best_value = -1
                    best_strategy = ""
                    
                    for strategy in selected_strategies:
                        value = all_data[strategy][key]["average_metrics"]["retrieval_metrics"].get(metric_key, 0)
                        if value > best_value:
                            best_value = value
                            best_strategy = strategy
                    
                    st.metric(metric_name, f"{best_value:.4f}", delta=best_strategy)
            
            with col2:
                st.markdown("**Response Metrics:**")
                for metric_name, metric_key in zip(response_metrics, response_keys):
                    best_value = -1
                    best_strategy = ""
                    
                    for strategy in selected_strategies:
                        value = all_data[strategy][key]["average_metrics"]["response_metrics"][metric_key]
                        if value > best_value:
                            best_value = value
                            best_strategy = strategy
                    
                    st.metric(metric_name, f"{best_value:.4f}", delta=best_strategy)
        
        # Export section
        st.markdown("---")
        st.header("💾 Export Dữ Liệu")
        
        selected_k_export = st.selectbox("Chọn TOP_K để export:", top_k_values, key="export_k")
        
        if st.button("📥 Tạo CSV Export"):
            export_data = []
            
            for strategy in selected_strategies:
                key = f"k={selected_k_export}"
                ret_metrics = all_data[strategy][key]["average_metrics"]["retrieval_metrics"]
                resp_metrics = all_data[strategy][key]["average_metrics"]["response_metrics"]
                
                export_data.append({
                    "Strategy": strategy,
                    "TOP_K": selected_k_export,
                    **{f"Retrieval_{k}": v for k, v in ret_metrics.items()},
                    **{f"Response_{k}": v for k, v in resp_metrics.items()}
                })
            
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"evaluation_results_k{selected_k_export}.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Vui lòng tải file JSON kết quả đánh giá")
    st.markdown("""
    ### Hướng dẫn:
    1. Chạy script `evaluate_retrieval_strategies.py` để tạo file kết quả
    2. File JSON sẽ được lưu tại `results/retrieval_evaluation_results.json`
    3. Tải file lên đây để xem phân tích chi tiết
    
    ### Cấu trúc file JSON:
    ```json
    {
      "semantic_search": {
        "k=10": {
          "average_metrics": {
            "retrieval_metrics": {
              "precision@k": 0.5,
              "recall@k": 0.8,
              "hit@k": 0.9,
              "mrr@k": 0.75,
              "ndcg@k": 0.65
            },
            "response_metrics": {...}
          }
        }
      },
      "hybrid_search": {...}
    }
    ```
    """)