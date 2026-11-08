"""Biến thể của run_evaluation.py nhưng RETRIEVAL trỏ vào collection `phapdien_bge_m3`
(BGE-M3) thay vì `phapdien_simple_tendieu` (OpenAI) — để so sánh chất lượng câu trả
lời cuối cùng giữa 2 embedding model trên CÙNG pipeline agentic thật, CÙNG bộ câu
hỏi (data.json). LLM sinh câu trả lời (chat completions) vẫn dùng OpenAI như bình
thường — chỉ đổi phần tìm kiếm vector.

So dựa trên response metrics (ROUGE-L/BLEU/Semantic Similarity so với gold_response,
KHÔNG bị ảnh hưởng bởi lỗi offset) + grounding/answer_relevance/context_relevance
(reference-free). Vẫn tính retrieval metrics (Precision/Recall/...) cho đồng bộ với
run_evaluation.py, nhưng xem README mục Đánh giá — gold_docs đang lệch offset nên
các chỉ số đó không đáng tin ở cả 2 lần chạy (OpenAI lẫn BGE-M3), không chỉ riêng bản này.

Chạy: cd evaluation && python run_evaluation_bge.py --yes
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from metrics.retrieval_metrics import precision_at_k, recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k
from metrics.response_metrics import rouge_l_score, bleu_score, semantic_similarity

BGE_COLLECTION_NAME = "phapdien_bge_m3"
LIVE_PROGRESS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "_live_progress_bge.json")


class _BgeEmbedClient:
    """Giả lập interface `openai_client.embeddings.create(model=, input=)` của
    OpenAI SDK, nhưng dùng BGE-M3 cục bộ (GPU nếu có) — để GraphRAGRetriever._embed()
    dùng được nguyên xi mà không cần sửa code gốc. KHÔNG dùng để sinh câu trả lời
    (chat completions) — chỉ dùng cho bước embed câu hỏi khi vector_search."""

    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[BGE embed client] device: {device}")
        self._model = SentenceTransformer("BAAI/bge-m3", device=device)
        self.embeddings = self

    def create(self, model=None, input=None):
        text = input if isinstance(input, str) else input[0]
        vec = self._model.encode(text, convert_to_numpy=True).tolist()
        return SimpleNamespace(data=[SimpleNamespace(embedding=vec)])


def _write_live_progress(*, status: str, completed: int, total: int, data_file: str,
                          top_k_values: list, results: list) -> None:
    os.makedirs(os.path.dirname(LIVE_PROGRESS_PATH), exist_ok=True)
    payload = {
        "status": status,
        "data_file": data_file,
        "completed": completed,
        "total": total,
        "summary": summarize(results, top_k_values) if results else None,
        "detailed": results,
    }
    tmp_path = LIVE_PROGRESS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, LIVE_PROGRESS_PATH)


def offsets_to_mapcs(offsets: set) -> dict:
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
    from services.agentic_rag import run_agentic_rag

    answer_parts = []
    retrieved_mapcs = []
    grounding_score = None
    answer_relevance = None
    context_relevance = None

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
            answer_relevance = ev.get("answer_relevance")
            context_relevance = ev.get("context_relevance")

    return {
        "answer": "".join(answer_parts).strip(),
        "retrieved_mapcs": retrieved_mapcs,
        "grounding_score": grounding_score,
        "answer_relevance": answer_relevance,
        "context_relevance": context_relevance,
    }


def evaluate(data_file: str, top_k_values: list, limit: int = None) -> list:
    from pymilvus import connections, Collection
    from openai import OpenAI
    from retrieve.build_graph import GraphRAGRetriever, get_neo4j

    connections.connect("default", host="localhost", port="19530")
    bge_collection = Collection(BGE_COLLECTION_NAME)
    bge_collection.load()

    dataset = load_dataset_with_mapc(data_file)
    if limit:
        dataset = dataset[:limit]

    neo4j_driver = get_neo4j()
    bge_embed_client = _BgeEmbedClient()
    retriever = GraphRAGRetriever(neo4j_driver=neo4j_driver, milvus_collection=bge_collection, openai_client=bge_embed_client)
    chat_client = OpenAI()  # LLM sinh câu trả lời — vẫn OpenAI thật, không đổi

    total = len(dataset)
    per_query_results = []
    _write_live_progress(status="running", completed=0, total=total, data_file=data_file,
                          top_k_values=top_k_values, results=per_query_results)

    for i, item in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] {item['query'][:60]}...")
        out = run_agentic_query(retriever, neo4j_driver, chat_client, item["query"])

        gold = item["gold_mapcs"]
        retrieved = out["retrieved_mapcs"]
        row = {
            "query": item["query"],
            "gold_mapcs": gold,
            "retrieved_mapcs": retrieved,
            "answer": out["answer"],
            "gold_response": item.get("gold_response", ""),
            "grounding_score": out["grounding_score"],
            "answer_relevance": out["answer_relevance"],
            "context_relevance": out["context_relevance"],
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
        _write_live_progress(status="running", completed=i, total=total, data_file=data_file,
                              top_k_values=top_k_values, results=per_query_results)

    _write_live_progress(status="done", completed=total, total=total, data_file=data_file,
                          top_k_values=top_k_values, results=per_query_results)
    return per_query_results


def summarize(results: list, top_k_values: list) -> dict:
    summary = {"num_queries": len(results), "retrieval_metrics": {}, "response_metrics": {},
               "avg_grounding_score": None, "avg_answer_relevance": None, "avg_context_relevance": None}
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

    for key, out_key in [("grounding_score", "avg_grounding_score"),
                          ("answer_relevance", "avg_answer_relevance"),
                          ("context_relevance", "avg_context_relevance")]:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            summary[out_key] = sum(vals) / len(vals)

    return summary


def estimate_cost(num_queries: int) -> str:
    low, high = num_queries * 0.01, num_queries * 0.05
    return f"~{low:.2f}-{high:.2f} USD cho {num_queries} câu hỏi (chỉ phần chat completions — embed BGE-M3 miễn phí, chạy cục bộ)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá pipeline agentic RAG với retrieval BGE-M3 (gọi OpenAI thật cho chat — tốn phí)")
    parser.add_argument("--data", default="data.json")
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/agentic_evaluation_results_bge.json")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        n_total = len(json.load(f))
    n_run = args.limit or n_total

    if not args.yes:
        print(f"Script này sẽ gọi OpenAI THẬT (chat completions) cho {n_run} câu hỏi (ước tính {estimate_cost(n_run)}).")
        print("Thêm --yes để xác nhận chạy. Dùng --limit N để chạy thử với ít câu hỏi hơn trước.")
        sys.exit(0)

    try:
        results = evaluate(args.data, args.top_k, limit=args.limit)
    except BaseException:
        if os.path.exists(LIVE_PROGRESS_PATH):
            with open(LIVE_PROGRESS_PATH, "r", encoding="utf-8") as f:
                live = json.load(f)
            live["status"] = "error"
            with open(LIVE_PROGRESS_PATH, "w", encoding="utf-8") as f:
                json.dump(live, f, ensure_ascii=False, indent=2)
        raise
    summary = summarize(results, args.top_k)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed": results}, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu: {args.output}")
