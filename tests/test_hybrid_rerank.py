"""Test GraphRAGRetriever._rerank() (BM25 + semantic cục bộ). Mock phần semantic để
không cần tải model HuggingFace trong CI — BM25 vẫn chạy thật (rank_bm25, không cần mạng)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import retrieve.build_graph as build_graph


def test_rerank_prefers_keyword_match(monkeypatch):
    # BM25 IDF cần corpus đủ lớn để phân biệt (2 văn bản thì idf của từ xuất hiện
    # đúng 1 lần luôn bằng 0 theo công thức BM25 cổ điển) — dùng vài candidate nhiễu
    # để mô phỏng đúng quy mô thật (pool ~40 candidate khi gọi từ rerank_phapdien_hits).
    monkeypatch.setattr(build_graph, "_semantic_scores", lambda q, texts: [0.0] * len(texts))

    retriever = build_graph.GraphRAGRetriever(neo4j_driver=None, milvus_collection=None, openai_client=None)
    candidates = [
        {"mapc": "A1", "ten": "Dieu an ninh quoc gia", "noidung": "quy dinh ve an ninh quoc gia", "base_score": 1.0},
        {"mapc": "A2", "ten": "Dieu xu phat giao thong", "noidung": "phat tien vi pham giao thong", "base_score": 1.0},
        {"mapc": "A3", "ten": "Dieu lao dong", "noidung": "hop dong lao dong va tien luong", "base_score": 1.0},
        {"mapc": "A4", "ten": "Dieu dat dai", "noidung": "quyen su dung dat cua ho gia dinh", "base_score": 1.0},
    ]

    out = retriever._rerank(candidates, "an ninh quoc gia", top_k=4)
    assert out[0]["mapc"] == "A1"
    assert out[0]["score"] > out[1]["score"]


def test_rerank_dedupes_by_mapc(monkeypatch):
    monkeypatch.setattr(build_graph, "_semantic_scores", lambda q, texts: [0.0] * len(texts))

    retriever = build_graph.GraphRAGRetriever(neo4j_driver=None, milvus_collection=None, openai_client=None)
    candidates = [
        {"mapc": "A1", "ten": "x", "noidung": "y", "base_score": 1.0},
        {"mapc": "A1", "ten": "x", "noidung": "y", "base_score": 1.0},
    ]

    out = retriever._rerank(candidates, "x", top_k=5)
    assert len(out) == 1


def test_rerank_empty_candidates():
    retriever = build_graph.GraphRAGRetriever(neo4j_driver=None, milvus_collection=None, openai_client=None)
    assert retriever._rerank([], "query", top_k=5) == []
