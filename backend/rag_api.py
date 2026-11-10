import sys
from pathlib import Path

# Cho phép chạy "uvicorn backend.rag_api:app" (hoặc trực tiếp) từ bất kỳ thư mục nào
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import tempfile
import time
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File
from langfuse import get_client, observe, propagate_attributes
from openai import OpenAI
from pydantic import BaseModel, Field
import json as _json

# Nạp .env trước khi import các module khác
load_dotenv()

from agents.pipeline import Action, run_pre_retrieve
from evaluation.live_log import log_live_turn
from retrieve.build_graph import GraphRAGRetriever, get_neo4j
from retrieve.two_stage_search import client, collection
from services.agentic_rag import _chunk_stream_text, run_agentic_rag
from services.utils import (
    client as oai_client,
    detect_intent,
    rewrite_query_v2,
)

app = FastAPI(title="RAG Backend API", version="1.0.0")

neo4j_driver = get_neo4j()
two_stage_retriever = GraphRAGRetriever(
    neo4j_driver=neo4j_driver,
    milvus_collection=collection,
    openai_client=client,
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RagChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    history: List[ChatMessage] = Field(default_factory=list)
    query_mode: Literal["normal", "situation"] = "normal"
    # id cuộc hội thoại ở frontend (chat._id, crypto.randomUUID()) — gắn làm
    # session_id trên Langfuse để nhóm toàn bộ lượt hỏi-đáp của 1 cuộc hội
    # thoại lại trong tab Sessions, xem được chi tiết cả cuộc chứ không chỉ
    # từng trace rời rạc.
    chat_id: Optional[str] = None


class RagChatResponse(BaseModel):
    answer: str
    action: str
    normalized_query: Optional[str] = None
    sources: List[Dict[str, str]] = Field(default_factory=list)
    passages: List[Dict[str, Any]] = Field(default_factory=list)
    grounding_score: Optional[float] = None
    flagged_sentences: List[str] = Field(default_factory=list)
    answer_relevance: Optional[float] = None
    context_relevance: Optional[float] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
# (xây passages/sources/token-chunking giờ nằm trong services.agentic_rag,
#  dùng chung bởi run_agentic_rag - xem import ở trên)


def _is_timeout_error(e: Exception) -> bool:
    return "timeout" in type(e).__name__.lower() or "timeout" in str(e).lower()


def _score_outcome(outcome: str, *, is_error: bool = False, is_timeout: bool = False) -> None:
    """Đẩy score lên Langfuse cho Dashboard (Safety / Reliability / request log):
      - request_outcome (CATEGORICAL): quick/spam/escalate/rejected/proceed/error
      - guardrail_pass (BOOLEAN): false nếu bị chặn bởi guardrail (spam/escalate/rejected)
      - request_error, timeout (BOOLEAN): chỉ có ý nghĩa ở nhánh có gọi LLM/tool thật
        (proceed/error) — các nhánh rule-based (quick/spam/escalate/rejected) không
        có rủi ro timeout nên không chấm.
    """
    client = get_client()
    client.score_current_trace(name="request_outcome", value=outcome, data_type="CATEGORICAL")
    client.score_current_trace(
        name="guardrail_pass",
        value=0 if outcome in ("spam", "escalate", "rejected") else 1,
        data_type="BOOLEAN",
    )
    if outcome in ("proceed", "error"):
        client.score_current_trace(name="request_error", value=1 if is_error else 0, data_type="BOOLEAN")
        client.score_current_trace(name="timeout", value=1 if is_timeout else 0, data_type="BOOLEAN")


def _log_trace_io(
    *,
    chat_id: Optional[str],
    query: str,
    normalized_query: Optional[str],
    action: str,
    answer: str,
    **extra: Any,
) -> None:
    """Gắn input/output/metadata chi tiết lên root span (= trace) của lượt hỏi
    này — để khi mở theo Session (nhóm theo chat_id, xem agents/propagate_attributes
    ở rag_chat/rag_stream) thấy được đầy đủ nội dung từng lượt trong cả cuộc hội
    thoại, không chỉ các con số score rời rạc."""
    get_client().update_current_span(
        input={"chat_id": chat_id, "query": query, "normalized_query": normalized_query},
        output={"action": action, "answer": answer},
        metadata={k: v for k, v in extra.items() if v is not None},
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/rag/chat", response_model=RagChatResponse)
@observe(name="rag_chat_request")
def rag_chat(req: RagChatRequest) -> RagChatResponse:
    with propagate_attributes(session_id=req.chat_id):
        return _rag_chat_impl(req)


def _rag_chat_impl(req: RagChatRequest) -> RagChatResponse:
    t0 = time.perf_counter()
    prompt = req.prompt.strip()
    history_for_processing = [m.model_dump() for m in req.history]
    decision = run_pre_retrieve(prompt, history_for_processing)

    if decision.action == Action.QUICK_ANSWER:
        answer = decision.answer_text or ""
        _score_outcome("quick")
        _log_trace_io(chat_id=req.chat_id, query=prompt, normalized_query=None, action="quick", answer=answer)
        log_live_turn(query=prompt, normalized_query=None, action="quick", answer=answer,
                      elapsed_ms=(time.perf_counter() - t0) * 1000)
        return RagChatResponse(answer=answer, action="quick")

    if decision.action == Action.SPAM:
        answer = (
            "Xin lỗi, tôi không thể xử lý yêu cầu của bạn vì nó có dấu hiệu spam. "
            "Bạn hãy cung cấp thông tin cụ thể hơn để tôi có thể hỗ trợ tốt nhất. Cảm ơn bạn!"
        )
        _score_outcome("spam")
        _log_trace_io(chat_id=req.chat_id, query=prompt, normalized_query=None, action="spam", answer=answer)
        log_live_turn(query=prompt, normalized_query=None, action="spam", answer=answer,
                      elapsed_ms=(time.perf_counter() - t0) * 1000)
        return RagChatResponse(action="spam", answer=answer)

    if decision.action == Action.ESCALATE:
        answer = (
            "Mình rất tiếc khi đem lại trải nghiệm không tốt cho bạn. "
            "Bạn có muốn mình chuyển tiếp vấn đề cho cán bộ trực tiếp xử lý không?"
        )
        _score_outcome("escalate")
        _log_trace_io(chat_id=req.chat_id, query=prompt, normalized_query=None, action="escalate", answer=answer)
        log_live_turn(query=prompt, normalized_query=None, action="escalate", answer=answer,
                      elapsed_ms=(time.perf_counter() - t0) * 1000)
        return RagChatResponse(action="escalate", answer=answer)

    # Action.PROCEED
    base = (
        history_for_processing[:-1]
        if history_for_processing and history_for_processing[-1].get("role") == "user"
        else history_for_processing
    )
    latest_history = base[-10:]
    normalized_query = rewrite_query_v2(prompt, latest_history) or prompt

    intent_result = detect_intent(normalized_query) or "Có"
    if "không" in intent_result.lower():
        answer = (
            "Có vẻ yêu cầu của bạn chưa rõ ràng hoặc không nằm trong phạm vi mình xử lý. "
            "Mình là trợ lý pháp lý hỗ trợ tìm kiếm và giải thích luật. "
            "Bạn có thể hỏi về các quy định, điều luật cụ thể, hoặc quy trình pháp lý mà bạn đang thắc mắc."
        )
        _score_outcome("rejected")
        _log_trace_io(chat_id=req.chat_id, query=prompt, normalized_query=normalized_query, action="rejected", answer=answer)
        log_live_turn(query=prompt, normalized_query=normalized_query, action="rejected", answer=answer,
                      elapsed_ms=(time.perf_counter() - t0) * 1000)
        return RagChatResponse(action="proceed", normalized_query=normalized_query, answer=answer)

    is_situation = req.query_mode == "situation"
    user_for_agent = prompt if is_situation else normalized_query

    try:
        meta: Dict[str, Any] = {}
        grounding: Dict[str, Any] = {}
        answer_parts: List[str] = []
        for ev in run_agentic_rag(
            openai_client=oai_client,
            retriever=two_stage_retriever,
            neo4j_driver=neo4j_driver,
            user_prompt=user_for_agent,
            history=latest_history,
            normalized_query=normalized_query,
            situation_mode=is_situation,
            rerank_query=normalized_query,
        ):
            if ev["type"] == "meta":
                meta = ev
            elif ev["type"] == "token":
                answer_parts.append(ev["text"])
            elif ev["type"] == "grounding":
                grounding = ev
    except Exception as e:
        answer = f"Mình gặp lỗi khi chạy tác tử tìm kiếm: {e}"
        get_client().update_current_span(level="ERROR", status_message=str(e))
        _score_outcome("error", is_error=True, is_timeout=_is_timeout_error(e))
        _log_trace_io(chat_id=req.chat_id, query=prompt, normalized_query=normalized_query, action="error",
                      answer=answer, error=str(e), situation_mode=is_situation)
        log_live_turn(query=prompt, normalized_query=normalized_query, action="error", answer=answer,
                      elapsed_ms=(time.perf_counter() - t0) * 1000, situation_mode=is_situation)
        return RagChatResponse(action="proceed", normalized_query=normalized_query, answer=answer, passages=[])

    raw_response = "".join(answer_parts).strip() or (
        "Mình chưa thể tạo câu trả lời lúc này do lỗi dịch vụ AI. "
        "Bạn vui lòng thử lại sau ít phút."
    )
    _score_outcome("proceed")
    _log_trace_io(
        chat_id=req.chat_id, query=prompt, normalized_query=normalized_query, action="proceed",
        answer=raw_response, iterations=meta.get("iterations"), situation_mode=is_situation,
        sources_count=len(meta.get("sources", [])), passages_count=len(meta.get("passages", [])),
        grounding_score=grounding.get("grounding_score"),
        answer_relevance=grounding.get("answer_relevance"),
        context_relevance=grounding.get("context_relevance"),
    )

    log_live_turn(
        query=prompt, normalized_query=normalized_query, action="proceed", answer=raw_response,
        retrieved_mapcs=[p.get("mapc") for p in meta.get("passages", []) if p.get("mapc")],
        grounding_score=grounding.get("grounding_score"),
        flagged_sentences=grounding.get("flagged_sentences", []),
        answer_relevance=grounding.get("answer_relevance"),
        context_relevance=grounding.get("context_relevance"),
        iterations=meta.get("iterations"),
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        situation_mode=is_situation,
    )

    return RagChatResponse(
        action="proceed",
        normalized_query=normalized_query,
        answer=raw_response,
        sources=meta.get("sources", []),
        passages=meta.get("passages", []),
        grounding_score=grounding.get("grounding_score"),
        flagged_sentences=grounding.get("flagged_sentences", []),
        answer_relevance=grounding.get("answer_relevance"),
        context_relevance=grounding.get("context_relevance"),
    )


@app.post("/rag/stream")
def rag_stream(req: RagChatRequest):
    """
    Server-Sent Events stream.
      data: {"type":"status", "text":"...", "iteration"?: n, "max"?: m}
      data: {"type":"meta",  "action":"...", "normalized_query":"...", "passages":[...], "iterations"?: n}
      data: {"type":"token", "text":"..."}
      data: {"type":"done"}
      data: {"type":"error", "message":"..."}
    """
    @observe(name="rag_stream_request")
    def event_stream():
        with propagate_attributes(session_id=req.chat_id):
            yield from _event_stream_impl()

    def _event_stream_impl():
        t0 = time.perf_counter()
        prompt = req.prompt.strip()
        history_for_processing = [m.model_dump() for m in req.history]
        # Yield ngay dòng đầu tiên để SSE connection mở và client thấy phản hồi
        # tức thì — run_pre_retrieve (guardrail) là lệnh gọi đồng bộ, nếu không
        # có yield nào trước nó thì trình duyệt không nhận được byte nào trong
        # lúc nó chạy, UI sẽ im lặng.
        yield f"data: {_json.dumps({'type':'status','text':'🤔 Đang kiểm tra nội dung...'}, ensure_ascii=False)}\n\n"
        decision = run_pre_retrieve(prompt, history_for_processing)

        # ── QUICK / SPAM / ESCALATE ───────────────────────────────────────────
        if decision.action in (Action.QUICK_ANSWER, Action.SPAM, Action.ESCALATE):
            if decision.action == Action.QUICK_ANSWER:
                answer = decision.answer_text or ""
                outcome = "quick"
            elif decision.action == Action.SPAM:
                answer = "Xin lỗi, yêu cầu có dấu hiệu spam. Vui lòng cung cấp thông tin cụ thể hơn."
                outcome = "spam"
            else:
                answer = "Mình rất tiếc. Bạn có muốn mình chuyển tiếp cho cán bộ xử lý không?"
                outcome = "escalate"
            _score_outcome(outcome)
            _log_trace_io(chat_id=req.chat_id, query=prompt, normalized_query=None, action=outcome, answer=answer)
            yield f"data: {_json.dumps({'type':'meta','action':decision.action.value,'passages':[]}, ensure_ascii=False)}\n\n"
            for part in _chunk_stream_text(answer):
                yield f"data: {_json.dumps({'type':'token','text':part}, ensure_ascii=False)}\n\n"
            yield f"data: {_json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"
            log_live_turn(query=prompt, normalized_query=None, action=outcome, answer=answer,
                          elapsed_ms=(time.perf_counter() - t0) * 1000)
            return

        # ── PROCEED ───────────────────────────────────────────────────────────
        base = (
            history_for_processing[:-1]
            if history_for_processing and history_for_processing[-1].get("role") == "user"
            else history_for_processing
        )
        latest_history   = base[-10:]
        yield f"data: {_json.dumps({'type':'status','text':'✍️ Đang chuẩn hoá câu hỏi...'}, ensure_ascii=False)}\n\n"
        normalized_query = rewrite_query_v2(prompt, latest_history) or prompt

        intent_result = detect_intent(normalized_query) or "Có"
        if "không" in intent_result.lower():
            answer = (
                "Yêu cầu chưa rõ ràng hoặc ngoài phạm vi. "
                "Mình hỗ trợ tìm kiếm và giải thích pháp luật Việt Nam."
            )
            yield f"data: {_json.dumps({'type':'meta','action':'proceed','normalized_query':normalized_query,'passages':[]}, ensure_ascii=False)}\n\n"
            for part in _chunk_stream_text(answer):
                yield f"data: {_json.dumps({'type':'token','text':part}, ensure_ascii=False)}\n\n"
            yield f"data: {_json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"
            _score_outcome("rejected")
            _log_trace_io(chat_id=req.chat_id, query=prompt, normalized_query=normalized_query, action="rejected", answer=answer)
            log_live_turn(query=prompt, normalized_query=normalized_query, action="rejected", answer=answer,
                          elapsed_ms=(time.perf_counter() - t0) * 1000)
            return

        is_situation = req.query_mode == "situation"
        user_for_agent = prompt if is_situation else normalized_query

        meta: Dict[str, Any] = {}
        grounding: Dict[str, Any] = {}
        answer_parts: List[str] = []
        error_msg: Optional[str] = None
        try:
            for ev in run_agentic_rag(
                openai_client=oai_client,
                retriever=two_stage_retriever,
                neo4j_driver=neo4j_driver,
                user_prompt=user_for_agent,
                history=latest_history,
                normalized_query=normalized_query,
                situation_mode=is_situation,
                rerank_query=normalized_query,
            ):
                if ev["type"] == "meta":
                    meta = ev
                elif ev["type"] == "token":
                    answer_parts.append(ev["text"])
                elif ev["type"] == "grounding":
                    grounding = ev
                yield f"data: {_json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_msg = str(e)
            get_client().update_current_span(level="ERROR", status_message=error_msg)
            _score_outcome("error", is_error=True, is_timeout=_is_timeout_error(e))
            _log_trace_io(chat_id=req.chat_id, query=prompt, normalized_query=normalized_query, action="error",
                          answer=error_msg, error=error_msg, situation_mode=is_situation)
            yield f"data: {_json.dumps({'type':'error','message':error_msg}, ensure_ascii=False)}\n\n"

        yield f"data: {_json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"

        if error_msg is not None:
            log_live_turn(query=prompt, normalized_query=normalized_query, action="error", answer=error_msg,
                          elapsed_ms=(time.perf_counter() - t0) * 1000, situation_mode=is_situation)
        else:
            _score_outcome("proceed")
            final_answer = "".join(answer_parts).strip()
            _log_trace_io(
                chat_id=req.chat_id, query=prompt, normalized_query=normalized_query, action="proceed",
                answer=final_answer, iterations=meta.get("iterations"), situation_mode=is_situation,
                sources_count=len(meta.get("sources", [])), passages_count=len(meta.get("passages", [])),
                grounding_score=grounding.get("grounding_score"),
                answer_relevance=grounding.get("answer_relevance"),
                context_relevance=grounding.get("context_relevance"),
            )
            log_live_turn(
                query=prompt, normalized_query=normalized_query, action="proceed",
                answer=final_answer,
                retrieved_mapcs=[p.get("mapc") for p in meta.get("passages", []) if p.get("mapc")],
                grounding_score=grounding.get("grounding_score"),
                flagged_sentences=grounding.get("flagged_sentences", []),
                answer_relevance=grounding.get("answer_relevance"),
                context_relevance=grounding.get("context_relevance"),
                iterations=meta.get("iterations"),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                situation_mode=is_situation,
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    from fastapi import UploadFile, File
    
    audio_bytes = await file.read()
    
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        stt_client = OpenAI()
        with open(tmp_path, "rb") as audio_file:
            transcription = stt_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="vi",
            )
        return {"text": transcription.text}
    finally:
        os.unlink(tmp_path)