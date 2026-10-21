import sys
from pathlib import Path

# Cho phép chạy "uvicorn backend.rag_api:app" (hoặc trực tiếp) từ bất kỳ thư mục nào
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import tempfile
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
from pydantic import BaseModel, Field
import json as _json

# Nạp .env trước khi import các module khác
load_dotenv()

from agents.pipeline import Action, run_pre_retrieve
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


class RagChatResponse(BaseModel):
    answer: str
    action: str
    normalized_query: Optional[str] = None
    sources: List[Dict[str, str]] = Field(default_factory=list)
    passages: List[Dict[str, Any]] = Field(default_factory=list)
    grounding_score: Optional[float] = None
    flagged_sentences: List[str] = Field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────
# (xây passages/sources/token-chunking giờ nằm trong services.agentic_rag,
#  dùng chung bởi run_agentic_rag - xem import ở trên)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/rag/chat", response_model=RagChatResponse)
def rag_chat(req: RagChatRequest) -> RagChatResponse:
    prompt = req.prompt.strip()
    history_for_processing = [m.model_dump() for m in req.history]
    decision = run_pre_retrieve(prompt, history_for_processing)

    if decision.action == Action.QUICK_ANSWER:
        return RagChatResponse(answer=decision.answer_text or "", action="quick")

    if decision.action == Action.SPAM:
        return RagChatResponse(
            action="spam",
            answer=(
                "Xin lỗi, tôi không thể xử lý yêu cầu của bạn vì nó có dấu hiệu spam. "
                "Bạn hãy cung cấp thông tin cụ thể hơn để tôi có thể hỗ trợ tốt nhất. Cảm ơn bạn!"
            ),
        )

    if decision.action == Action.ESCALATE:
        return RagChatResponse(
            action="escalate",
            answer=(
                "Mình rất tiếc khi đem lại trải nghiệm không tốt cho bạn. "
                "Bạn có muốn mình chuyển tiếp vấn đề cho cán bộ trực tiếp xử lý không?"
            ),
        )

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
        return RagChatResponse(
            action="proceed",
            normalized_query=normalized_query,
            answer=(
                "Có vẻ yêu cầu của bạn chưa rõ ràng hoặc không nằm trong phạm vi mình xử lý. "
                "Mình là trợ lý pháp lý hỗ trợ tìm kiếm và giải thích luật. "
                "Bạn có thể hỏi về các quy định, điều luật cụ thể, hoặc quy trình pháp lý mà bạn đang thắc mắc."
            ),
        )

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
        return RagChatResponse(
            action="proceed",
            normalized_query=normalized_query,
            answer=f"Mình gặp lỗi khi chạy tác tử tìm kiếm: {e}",
            passages=[],
        )

    raw_response = "".join(answer_parts).strip() or (
        "Mình chưa thể tạo câu trả lời lúc này do lỗi dịch vụ AI. "
        "Bạn vui lòng thử lại sau ít phút."
    )

    return RagChatResponse(
        action="proceed",
        normalized_query=normalized_query,
        answer=raw_response,
        sources=meta.get("sources", []),
        passages=meta.get("passages", []),
        grounding_score=grounding.get("grounding_score"),
        flagged_sentences=grounding.get("flagged_sentences", []),
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
    def event_stream():
        prompt = req.prompt.strip()
        history_for_processing = [m.model_dump() for m in req.history]
        decision = run_pre_retrieve(prompt, history_for_processing)

        # ── QUICK / SPAM / ESCALATE ───────────────────────────────────────────
        if decision.action in (Action.QUICK_ANSWER, Action.SPAM, Action.ESCALATE):
            if decision.action == Action.QUICK_ANSWER:
                answer = decision.answer_text or ""
            elif decision.action == Action.SPAM:
                answer = "Xin lỗi, yêu cầu có dấu hiệu spam. Vui lòng cung cấp thông tin cụ thể hơn."
            else:
                answer = "Mình rất tiếc. Bạn có muốn mình chuyển tiếp cho cán bộ xử lý không?"
            yield f"data: {_json.dumps({'type':'meta','action':decision.action.value,'passages':[]}, ensure_ascii=False)}\n\n"
            for part in _chunk_stream_text(answer):
                yield f"data: {_json.dumps({'type':'token','text':part}, ensure_ascii=False)}\n\n"
            yield f"data: {_json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"
            return

        # ── PROCEED ───────────────────────────────────────────────────────────
        base = (
            history_for_processing[:-1]
            if history_for_processing and history_for_processing[-1].get("role") == "user"
            else history_for_processing
        )
        latest_history   = base[-10:]
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
            return

        is_situation = req.query_mode == "situation"
        user_for_agent = prompt if is_situation else normalized_query

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
                yield f"data: {_json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'type':'error','message':str(e)}, ensure_ascii=False)}\n\n"

        yield f"data: {_json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"

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