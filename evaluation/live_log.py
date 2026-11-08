"""Ghi lại MỌI câu hỏi thật người dùng hỏi qua chat UI (/rag/chat, /rag/stream) —
không cần gold_docs/gold_response, không cần chạy run_evaluation.py hay upload
file JSON thủ công. Dashboard (visualization.py) đọc thẳng file này để hiển thị
realtime các câu đã hỏi, kèm 3 chỉ số kiểu RAGAS "reference-free" (không cần đáp
án mẫu, tính bằng embedding cục bộ — miễn phí, không gọi OpenAI):
  - grounding_score (faithfulness): câu trả lời có bám sát passage đã retrieve
  - answer_relevance: câu trả lời có đúng trọng tâm câu hỏi
  - context_relevance: passage đã retrieve có liên quan tới câu hỏi

Không tính được Precision/Recall/Hit/MRR/NDCG/ROUGE-L/BLEU/Semantic Similarity ở
đây vì các metric đó BẮT BUỘC phải so với gold_docs/gold_response — câu hỏi tự do
không có đáp án mẫu nên không thể tự động chấm theo các chỉ số đó.
"""
import json
import time
from pathlib import Path
from typing import Any, Optional

LIVE_LOG_PATH = Path(__file__).resolve().parent / "results" / "live_chat_log.jsonl"


def log_live_turn(
    *,
    query: str,
    normalized_query: Optional[str],
    action: str,
    answer: str,
    retrieved_mapcs: Optional[list] = None,
    grounding_score: Optional[float] = None,
    flagged_sentences: Optional[list] = None,
    answer_relevance: Optional[float] = None,
    context_relevance: Optional[float] = None,
    iterations: Optional[int] = None,
    elapsed_ms: Optional[float] = None,
    situation_mode: bool = False,
) -> None:
    """Append 1 dòng JSON vào live_chat_log.jsonl. Không bao giờ raise — lỗi ghi
    log không được phép làm hỏng câu trả lời thật đang trả về cho người dùng."""
    try:
        LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "query": query,
            "normalized_query": normalized_query,
            "action": action,
            "answer": answer,
            "retrieved_mapcs": retrieved_mapcs or [],
            "num_passages": len(retrieved_mapcs or []),
            "grounding_score": grounding_score,
            "flagged_sentences": flagged_sentences or [],
            "answer_relevance": answer_relevance,
            "context_relevance": context_relevance,
            "iterations": iterations,
            "elapsed_ms": elapsed_ms,
            "situation_mode": situation_mode,
        }
        with open(LIVE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[live_log] Lỗi ghi log (bỏ qua, không ảnh hưởng câu trả lời): {e}")


def read_live_log() -> list[dict]:
    """Đọc toàn bộ live_chat_log.jsonl — dùng bởi dashboard."""
    if not LIVE_LOG_PATH.exists():
        return []
    rows = []
    with open(LIVE_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
