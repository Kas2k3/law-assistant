"""Model embedding cục bộ dùng chung (miễn phí, không gọi OpenAI) cho rerank
(retrieve/build_graph.py) và kiểm tra độ trung thực câu trả lời (services/agentic_rag.py)."""
_model = None


def get_local_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _model
