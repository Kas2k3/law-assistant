import os
import re
import tempfile
import uuid as _uuid
from typing import Any, Dict, List, Literal, Optional

import firebase_admin
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File
from firebase_admin import credentials, storage as fb_storage
from openai import OpenAI
from pydantic import BaseModel, Field
import json as _json

# Nạp .env trước khi import các module khác
load_dotenv()

_FIREBASE_CRED   = os.getenv("FIREBASE_CRED_PATH", "firebase-credentials.json")
_FIREBASE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "")

# ── Init Firebase ─────────────────────────────────────────────────────────────
if not firebase_admin._apps:
    if _FIREBASE_CRED and os.path.exists(_FIREBASE_CRED) and _FIREBASE_BUCKET:
        cred = credentials.Certificate(_FIREBASE_CRED)
        firebase_admin.initialize_app(cred, {"storageBucket": _FIREBASE_BUCKET})
        print(f"[Firebase] Initialized — bucket: {_FIREBASE_BUCKET}")
    else:
        print("[Firebase] Bỏ qua init — thiếu credentials hoặc bucket")

from agents.pipeline import Action, run_pre_retrieve
from retrieve.build_graph import GraphRAGRetriever, get_neo4j
from retrieve.two_stage_search import client, collection
from services.agentic_rag import (
    RAG_AGENT_TOOLS,
    agent_chat_model,
    agent_system_prompt,
    apply_agent_tool_result,
    assistant_message_to_dict,
    execute_agent_tool,
    run_agentic_rag_sync,
)
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

# Agentic RAG — OpenAI function tools (bản đầy đủ trong services/agentic_rag.py)
TOOLS = RAG_AGENT_TOOLS


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


# ── Helpers ───────────────────────────────────────────────────────────────────
def _web_sources_markdown(web_sources: List[Dict[str, str]]) -> str:
    if not web_sources:
        return ""
    lines = ["\n\n---\n**Nguồn tham khảo từ web:**"]
    for src in web_sources:
        icon = "✅" if src.get("level") == "high" else "⚠️"
        lines.append(
            f"{icon} {src.get('label', '[Web]')} [{src.get('title', '')}]({src.get('url', '')})"
        )
    return "\n".join(lines)


def _build_passages(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    passages = []
    for hit in results:
        passages.append({
            "mapc":      hit.get("mapc", ""),
            "ten":       hit.get("ten", ""),
            "tenchuong": hit.get("tenchuong", ""),
            "tendemuc":  hit.get("tendemuc", ""),
            "tenchude":  hit.get("tenchude", ""),
            "noidung":   str(hit.get("noidung") or hit.get("passage", "")),
            "score":     round(float(hit.get("score", 0)), 4),
            "source":    hit.get("source", "phapdien"),
            "url":       hit.get("url", ""),
            "trust_level": hit.get("trust_level", "medium"),
            "source_label": hit.get("source_label", "[Web]"),
        })
    return passages


def _web_sources_from_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    web_sources: List[Dict[str, str]] = []
    for p in passages:
        if p.get("source") in ("web", "web_realtime") and p.get("url"):
            web_sources.append({
                "title": str(p.get("ten", "")),
                "url":   str(p.get("url", "")),
                "level": str(p.get("trust_level", "medium")),
                "label": str(p.get("source_label", "[Web]")),
            })
    return web_sources


def _dedup_results(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_keys: set = set()
    web_r, pd_r = [], []
    for hit in raw_results:
        key = hit.get("mapc") or hit.get("url") or hit.get("passage", "")[:100]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if hit.get("source") in ("web", "web_realtime"):
            web_r.append(hit)
        else:
            pd_r.append(hit)
    return (web_r + pd_r)[:20]


def _chunk_stream_text(text: str, max_chars: int = 32):
    """
    Tách văn bản thành nhiều SSE nhỏ (theo từ / dòng) để client hiển thị mượt,
    thay vì gửi một lần ~400 ký tự.
    """
    rest = text or ""
    if not rest:
        return
    while rest:
        if rest[0] == "\n":
            n = 1
            while n < len(rest) and rest[n] == "\n":
                n += 1
            yield rest[:n]
            rest = rest[n:]
            continue
        m = re.match(r"(\S+)(\s*)", rest)
        if m:
            piece = m.group(0)
            if len(piece) > max_chars:
                yield piece[:max_chars]
                rest = rest[max_chars:]
            else:
                yield piece
                rest = rest[len(piece) :]
        else:
            yield rest[0]
            rest = rest[1:]


def _strip_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',     r'\1', text)
    text = re.sub(r'#{1,6}\s',       '',   text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
    text = re.sub(r'---+', '', text)
    text = re.sub(r'\n+', '. ', text).strip()
    return text[:4096]


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
        agent_out = run_agentic_rag_sync(
            openai_client=oai_client,
            retriever=two_stage_retriever,
            neo4j_driver=neo4j_driver,
            user_prompt=user_for_agent,
            history=latest_history,
            situation_mode=is_situation,
            rerank_query=normalized_query,
        )
    except Exception as e:
        return RagChatResponse(
            action="proceed",
            normalized_query=normalized_query,
            answer=f"Mình gặp lỗi khi chạy tác tử tìm kiếm: {e}",
            passages=[],
        )

    hits = _dedup_results(agent_out.get("passage_hits") or [])
    passages = _build_passages(hits)
    web_sources = _web_sources_from_passages(passages)
    raw_response = (agent_out.get("answer") or "").strip() or (
        "Mình chưa thể tạo câu trả lời lúc này do lỗi dịch vụ AI. "
        "Bạn vui lòng thử lại sau ít phút."
    )
    raw_response += _web_sources_markdown(web_sources)

    return RagChatResponse(
        action="proceed",
        normalized_query=normalized_query,
        answer=raw_response,
        sources=web_sources,
        passages=passages,
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
        system_prompt = agent_system_prompt(situation_mode=is_situation)
        messages: List[Any] = [
            {"role": "system", "content": system_prompt},
            *[{"role": m["role"], "content": m["content"]} for m in latest_history[-8:]],
            {"role": "user", "content": user_for_agent},
        ]
        bucket: List[Dict[str, Any]] = []
        max_it = 6
        model = agent_chat_model()

        try:
            for iteration in range(max_it):
                response = oai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=RAG_AGENT_TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                )
                msg = response.choices[0].message
                messages.append(assistant_message_to_dict(msg))

                if not msg.tool_calls:
                    ded = _dedup_results(bucket)
                    passages = _build_passages(ded)
                    web_sources = _web_sources_from_passages(passages)
                    yield f"data: {_json.dumps({'type':'meta','action':'proceed','normalized_query':normalized_query,'passages':passages,'sources':web_sources,'iterations':iteration + 1,'agentic':True}, ensure_ascii=False)}\n\n"

                    content = (msg.content or "").strip()
                    if content:
                        for part in _chunk_stream_text(content):
                            yield f"data: {_json.dumps({'type':'token','text':part}, ensure_ascii=False)}\n\n"
                    else:
                        stream = oai_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.2,
                            max_tokens=1200,
                            stream=True,
                        )
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                yield f"data: {_json.dumps({'type':'token','text':delta}, ensure_ascii=False)}\n\n"

                    if web_sources:
                        footer = "\n\n---\n**Nguồn tham khảo từ web:**\n" + "\n".join(
                            f"{'✅' if s['level'] == 'high' else '⚠️'} {s['label']} [{s['title']}]({s['url']})"
                            for s in web_sources
                        )
                        yield f"data: {_json.dumps({'type':'token','text':footer}, ensure_ascii=False)}\n\n"

                    yield f"data: {_json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"
                    return

                for tc in msg.tool_calls:
                    tname = tc.function.name
                    yield f"data: {_json.dumps({'type':'status','text':f'🔍 Đang gọi {tname}...','iteration':iteration + 1,'max':max_it}, ensure_ascii=False)}\n\n"
                    try:
                        args = _json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    tool_content, full_hits = execute_agent_tool(
                        tname,
                        args,
                        retriever=two_stage_retriever,
                        neo4j_driver=neo4j_driver,
                        rerank_query=normalized_query,
                    )
                    apply_agent_tool_result(tname, tool_content, full_hits, bucket)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    })

            ded = _dedup_results(bucket)
            passages = _build_passages(ded)
            web_sources = _web_sources_from_passages(passages)
            yield f"data: {_json.dumps({'type':'meta','action':'proceed','normalized_query':normalized_query,'passages':passages,'sources':web_sources,'iterations':max_it,'agentic':True}, ensure_ascii=False)}\n\n"

            stream = oai_client.chat.completions.create(
                model=model,
                messages=messages
                + [{
                    "role": "user",
                    "content": "Hãy tổng hợp câu trả lời dựa trên thông tin đã thu thập (tiếng Việt).",
                }],
                temperature=0.2,
                max_tokens=1200,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {_json.dumps({'type':'token','text':delta}, ensure_ascii=False)}\n\n"

            if web_sources:
                footer = "\n\n---\n**Nguồn tham khảo từ web:**\n" + "\n".join(
                    f"{'✅' if s['level'] == 'high' else '⚠️'} {s['label']} [{s['title']}]({s['url']})"
                    for s in web_sources
                )
                yield f"data: {_json.dumps({'type':'token','text':footer}, ensure_ascii=False)}\n\n"

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


@app.post("/tts")
async def text_to_speech(req: dict):
    from services.history import save_tts_url

    text    = (req.get("text") or "").strip()
    voice   = req.get("voice", "nova")
    chat_id = req.get("chat_id", "")
    msg_idx = req.get("msg_idx")

    if not text:
        return {"error": "text is required"}

    text = _strip_markdown(text)

    # ── Tạo audio bytes ───────────────────────────────────────────────────────
    tts_client  = OpenAI()
    audio_bytes = b""
    with tts_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="mp3",
    ) as response:
        for chunk in response.iter_bytes(chunk_size=4096):
            audio_bytes += chunk

    # ── Upload lên Firebase Storage ───────────────────────────────────────────
    firebase_url = None
    if firebase_admin._apps and _FIREBASE_BUCKET:
        try:
            file_name = f"tts/{chat_id or 'unknown'}/{_uuid.uuid4().hex}.mp3"
            bucket    = fb_storage.bucket(_FIREBASE_BUCKET)   # ← explicit bucket name
            blob      = bucket.blob(file_name)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            blob.upload_from_filename(tmp_path, content_type="audio/mpeg")
            os.unlink(tmp_path)

            blob.make_public()
            firebase_url = blob.public_url
            print(f"[TTS] Uploaded: {firebase_url}")

            if chat_id and msg_idx is not None:
                save_tts_url(chat_id, int(msg_idx), firebase_url)

        except Exception as e:
            print(f"[TTS Firebase] Lỗi upload: {e}")
    else:
        print("[TTS] Firebase chưa init — fallback stream trực tiếp")

    # ── Trả về ────────────────────────────────────────────────────────────────
    if firebase_url:
        return {"url": firebase_url, "chat_id": chat_id, "msg_idx": msg_idx}

    # Fallback: stream audio trực tiếp
    def generate():
        yield audio_bytes

    return StreamingResponse(
        generate(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache"},
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