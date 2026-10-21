"""
Evaluation Framework — đánh giá ĐÚNG pipeline agentic RAG thực tế đang chạy
===========================================================================

Gọi thẳng run_agentic_rag() qua GraphRAGRetriever (giống hệt backend/rag_api.py),
thay vì các chiến lược độc lập trong retrieval_strategies.py (trỏ vào collection
Milvus "phapdien_articles"/"phapdien_tendieu" không còn tồn tại trong production —
production dùng "phapdien_simple_tendieu" qua retrieve/two_stage_search.py).

Đo:
- Retrieval: Precision/Recall/Hit/MRR/NDCG @k — so khớp theo mapc (mã điều luật thật),
  không phải theo offset dòng MySQL như data.json gốc (xem load_dataset_with_mapc).
- Response: ROUGE-L, BLEU, Semantic Similarity (so với gold_response).
- Grounding: điểm trung thực trung bình do run_agentic_rag() tự tính (embedding cục bộ,
  không tốn phí) — xem services/agentic_rag.py::check_answer_grounding.

⚠ CẢNH BÁO CHI PHÍ: evaluate() gọi OpenAI THẬT (embedding cho vector_search + GPT-4o-mini
cho vòng lặp agentic, tối đa 6 lượt/câu hỏi). Mặc định script CHỈ in ước tính chi phí rồi
dừng — phải thêm --yes mới thực sự chạy. Dùng --limit N để thử với ít câu hỏi trước.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from metrics.retrieval_metrics import precision_at_k, recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k
from metrics.response_metrics import rouge_l_score, bleu_score, semantic_similarity


def offsets_to_mapcs(offsets: set) -> dict:
    """data.json gốc dùng offset dòng MySQL (dataset cũ) làm gold_docs. Pipeline thật
    định danh điều luật bằng mapc. Hàm này tra MySQL 1 lần để quy đổi — chỉ đọc dữ liệu,
    KHÔNG gọi OpenAI, không tốn phí."""
    import pymysql

    cfg = dict(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "123456789"),
        database=os.getenv("MYSQL_DATABASE", "law"),
        charset="utf8mb4",
    )
    conn = pymysql.connect(**cfg, cursorclass=pymysql.cursors.DictCursor)
    out = {}
    try:
        with conn.cursor() as cur:
            for off in offsets:
                cur.execute("SELECT mapc FROM pddieu LIMIT 1 OFFSET %s", (off - 1,))
                row = cur.fetchone()
                if row:
                    out[off] = row["mapc"]
    finally:
        conn.close()
    return out


def load_dataset_with_mapc(data_file: str) -> list:
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    all_offsets = {off for item in dataset for off in item["gold_docs"]}
    offset_map = offsets_to_mapcs(all_offsets)

    for item in dataset:
        item["gold_mapcs"] = [offset_map[o] for o in item["gold_docs"] if o in offset_map]
    return dataset


def run_agentic_query(retriever, neo4j_driver, oai_client, query: str) -> dict:
    """Chạy đúng pipeline thật (giống hệt backend/rag_api.py) cho 1 câu hỏi."""
    from services.agentic_rag import run_agentic_rag

    answer_parts = []
    retrieved_mapcs = []
    grounding_score = None

    for ev in run_agentic_rag(
        openai_client=oai_client,
        retriever=retriever,
        neo4j_driver=neo4j_driver,
        user_prompt=query,
        history=[],
        normalized_query=query,
    ):
        if ev["type"] == "token":
            answer_parts.append(ev["text"])
        elif ev["type"] == "meta":
            retrieved_mapcs = [p["mapc"] for p in ev.get("passages", []) if p.get("mapc")]
        elif ev["type"] == "grounding":
            grounding_score = ev.get("grounding_score")

    return {
        "answer": "".join(answer_parts).strip(),
        "retrieved_mapcs": retrieved_mapcs,
        "grounding_score": grounding_score,
    }


def evaluate(data_file: str, top_k_values: list, limit: int = None) -> list:
    from retrieve.build_graph import GraphRAGRetriever, get_neo4j
    from retrieve.two_stage_search import client, collection

    dataset = load_dataset_with_mapc(data_file)
    if limit:
        dataset = dataset[:limit]

    neo4j_driver = get_neo4j()
    retriever = GraphRAGRetriever(neo4j_driver=neo4j_driver, milvus_collection=collection, openai_client=client)

    per_query_results = []
    for i, item in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] {item['query'][:60]}...")
        out = run_agentic_query(retriever, neo4j_driver, client, item["query"])

        gold = item["gold_mapcs"]
        retrieved = out["retrieved_mapcs"]
        row = {
            "query": item["query"],
            "gold_mapcs": gold,
            "retrieved_mapcs": retrieved,
            "answer": out["answer"],
            "gold_response": item.get("gold_response", ""),
            "grounding_score": out["grounding_score"],
            "retrieval_metrics": {},
            "response_metrics": {},
        }
        for k in top_k_values:
            row["retrieval_metrics"][f"k={k}"] = {
                "precision": precision_at_k(retrieved, gold, k),
                "recall": recall_at_k(retrieved, gold, k),
                "hit": hit_at_k(retrieved, gold, k),
                "mrr": mrr_at_k(retrieved, gold, k),
                "ndcg": ndcg_at_k(retrieved, gold, k),
            }
        if out["answer"] and item.get("gold_response"):
            row["response_metrics"] = {
                "rouge_l": rouge_l_score(out["answer"], item["gold_response"]),
                "bleu": bleu_score(out["answer"], item["gold_response"]),
                "semantic_similarity": semantic_similarity(out["answer"], item["gold_response"]),
            }
        per_query_results.append(row)

    return per_query_results


def summarize(results: list, top_k_values: list) -> dict:
    summary = {"num_queries": len(results), "retrieval_metrics": {}, "response_metrics": {}, "avg_grounding_score": None}
    metric_keys = ["precision", "recall", "hit", "mrr", "ndcg"]
    for k in top_k_values:
        summary["retrieval_metrics"][f"k={k}"] = (
            {key: sum(r["retrieval_metrics"][f"k={k}"][key] for r in results) / len(results) for key in metric_keys}
            if results else {}
        )

    scored = [r for r in results if r["response_metrics"]]
    if scored:
        summary["response_metrics"] = {
            key: sum(r["response_metrics"][key] for r in scored) / len(scored)
            for key in ["rouge_l", "bleu", "semantic_similarity"]
        }

    grounded = [r["grounding_score"] for r in results if r["grounding_score"] is not None]
    if grounded:
        summary["avg_grounding_score"] = sum(grounded) / len(grounded)

    return summary


def estimate_cost(num_queries: int) -> str:
    """Ước tính RẤT thô: mỗi câu hỏi tốn vài lượt gọi GPT-4o-mini (vòng lặp agentic,
    tối đa 6 lượt) + 1-3 lượt embedding nhỏ (vector_search). Chỉ để tham khảo trước khi
    chạy thật, không phải con số chính xác."""
    low, high = num_queries * 0.01, num_queries * 0.05
    return f"~{low:.2f}-{high:.2f} USD cho {num_queries} câu hỏi"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá pipeline agentic RAG thật (gọi OpenAI thật — tốn phí)")
    parser.add_argument("--data", default="data.json")
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu hỏi đầu (để test nhanh/rẻ)")
    parser.add_argument("--output", default="results/agentic_evaluation_results.json")
    parser.add_argument("--yes", action="store_true", help="Xác nhận đã hiểu sẽ tốn phí OpenAI thật và muốn chạy")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        n_total = len(json.load(f))
    n_run = args.limit or n_total

    if not args.yes:
        print(f"Script này sẽ gọi OpenAI THẬT cho {n_run} câu hỏi (ước tính {estimate_cost(n_run)}).")
        print("Thêm --yes để xác nhận chạy. Dùng --limit N để chạy thử với ít câu hỏi hơn trước.")
        sys.exit(0)

    results = evaluate(args.data, args.top_k, limit=args.limit)
    summary = summarize(results, args.top_k)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed": results}, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu: {args.output}")
