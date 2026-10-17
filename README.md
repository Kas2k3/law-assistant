# Chatbot RAG — Hỏi đáp pháp luật (Pháp Điển)

Hệ thống hỏi–đáp pháp luật tiếng Việt dựa trên **Retrieval-Augmented Generation (RAG)** và **agentic tool-calling**: lớp pre-retrieve (quick / spam / sentiment), truy xuất vector (Milvus), mở rộng đồ thị (Neo4j), tùy chọn web (Tavily), rồi tổng hợp câu trả lời bằng LLM.

## Kiến trúc tổng quan

```
┌─────────────────┐     SSE / REST      ┌──────────────────┐
│  Next.js UI     │ ◄──────────────────►│  rag_api.py      │
│  (frontend/)    │   /api/stream       │  FastAPI :8001   │
│  MongoDB chats  │   /api/chats        └────────┬─────────┘
└────────┬────────┘                              │
         │                                       ├── agents/pipeline (pre-retrieve)
         │                                       ├── services/agentic_rag.py
         │                                       └── retrieve/GraphRAGRetriever
         │                                              ├── Milvus (vector)
         │                                              ├── Neo4j (graph)
         │                                              └── Tavily (web, tùy chọn)
         │
┌────────┴────────┐
│ MySQL `law`     │  ← nguồn gốc từ law-crawler
└─────────────────┘
```

| Thành phần | Vai trò |
|-----------|---------|
| **MySQL** | Lưu Pháp Điển đã crawl (`pddieu`, chủ đề, chương, đề mục) |
| **Milvus** | Embedding + tìm kiếm semantic (`phapdien_simple_tendieu`) |
| **Neo4j** | Đồ thị điều luật, quan hệ `LIEN_QUAN` / `THAM_CHIEU`, nội dung full cho agent |
| **rag_api** | API RAG, stream SSE, TTS/STT (Firebase tùy chọn) |
| **frontend** | Chat UI (dark mode), markdown, passages, TTS/STT, lịch sử MongoDB |
| **app.py** | Giao diện Streamlit (luồng cũ, tùy chọn) |

> Docker dùng cho **MySQL** (`law-crawler/`), **Milvus + Neo4j** (`test_raptor/`). Code Python và Next.js chạy trên host (venv + `npm`).

---

## Yêu cầu

- Python ≥ 3.10
- Node.js ≥ 20 (frontend Next.js 16)
- Docker & Docker Compose
- `OPENAI_API_KEY` (embedding `text-embedding-3-small`, chat, rerank LLM tùy chọn)
- Tuỳ chọn: `TAVILY_API_KEY`, MongoDB (lịch sử chat), Firebase (TTS lưu file công khai)

---

## 1. Cài đặt Python

```bash
cd project2-chatbot-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pymysql           # seed Milvus từ MySQL
pip install firebase-admin    # tùy chọn — TTS upload Firebase
```

Tạo file `.env` ở thư mục gốc (tham khảo các biến ở mục 3).

---

## 2. Chuẩn bị dữ liệu

### 2.1 Crawl Pháp Điển → MySQL

Tải dữ liệu từ [Pháp Điển](https://phapdien.moj.gov.vn/), giải nén vào `law-crawler/phap-dien/` (`chude.json`, `demuc.json`, `treeNode.json`, thư mục `demuc/`).

```bash
cd law-crawler
docker compose up -d          # MySQL + phpMyAdmin
pip install -r requirements.txt
# Lần đầu: bỏ comment db.create_tables trong main.py nếu DB trống
python main.py
```

MySQL mặc định: `localhost:3306`, DB `law`, user `root`, password `123456789`, phpMyAdmin `http://localhost:8081`.

### 2.2 Index Milvus

```bash
cd test_raptor
docker compose up -d          # etcd, minio, milvus-standalone, neo4j
```

Đợi healthy: `curl http://localhost:9091/healthz` → `OK`.

```bash
cd ..
source .venv/bin/activate
cd data_processing
python seed_data_batch.py              # ingest mới (drop collection cũ)
python seed_data_batch.py --resume     # tiếp tục nếu bị gián đoạn
```

- Collection: `phapdien_simple_tendieu`
- Embed theo **tên điều** (`tendieu`), metadata JSON (có `mapc`, `noidung` tự cắt nếu vượt ~64KB Milvus)
- Lỗi embed ghi vào `embed_errors.csv`

### 2.3 Build đồ thị Neo4j

```bash
cd ..
python retrieve/build_graph.py
# hoặc: python -m retrieve.build_graph
```

Neo4j Browser: `http://localhost:7474` — `neo4j` / `password123`.

---

## 3. Chạy backend RAG (FastAPI)

```bash
source .venv/bin/activate
uvicorn rag_api:app --host 0.0.0.0 --port 8001 --reload
```

| Endpoint | Mô tả |
|----------|--------|
| `GET /health` | Health check |
| `POST /rag/chat` | Chat đồng bộ (JSON) |
| `POST /rag/stream` | Chat SSE (xem bảng sự kiện bên dưới) |
| `POST /tts` | Text-to-speech (MP3 stream hoặc URL Firebase) |
| `POST /stt` | Speech-to-text (upload audio) |

**Body chung** (`RagChatRequest`):

```json
{
  "prompt": "Điều kiện ly hôn?",
  "history": [{"role": "user", "content": "..."}],
  "query_mode": "normal"
}
```

`query_mode`: `"normal"` (rewrite + intent) hoặc `"situation"` (mô tả tình huống — agent dùng prompt gốc, system prompt chế độ tình huống).

### Luồng xử lý

**Bước 1 — Pre-retrieve** (`agents/pipeline.py` → `AgentsManager`):

| Action | Ý nghĩa |
|--------|---------|
| `quick` | Câu hỏi FAQ — trả lời ngay, không RAG |
| `spam` | Từ chối spam |
| `escalate` | Sentiment tiêu cực — đề xuất chuyển cán bộ |
| `proceed` | Tiếp tục agentic RAG |

**Bước 2 — Khi `proceed`:** `rewrite_query_v2` + `detect_intent`. Nếu intent ngoài phạm vi → trả lời từ chối.

**Bước 3 — Agentic loop** (`services/agentic_rag.py`, tối đa 6 vòng):

| Tool | Mô tả |
|------|--------|
| `vector_search` | Milvus → `mapc` → hydrate Neo4j (full nội dung) → **rerank** |
| `graph_traverse` | Mở rộng Neo4j theo `LIEN_QUAN` / `THAM_CHIEU` → rerank |
| `web_search` | Tavily (khi có `TAVILY_API_KEY`) |
| `get_full_article` | Full một điều theo `mapc` |

Passages trả về UI lấy **full text từ Neo4j** (bucket agent), không phụ thuộc snippet Milvus. MySQL là fallback khi Neo4j thiếu node.

**Rerank:** mặc định overlap từ khóa + thứ tự Milvus; bật LLM rerank: `RAG_AGENT_LLM_RERANK=1`.

### Sự kiện SSE (`POST /rag/stream`)

| `type` | Nội dung |
|--------|----------|
| `status` | Tiến trình agent (`text`, `iteration`, `max`) |
| `meta` | `action`, `normalized_query`, `passages`, `sources`, `iterations`, `agentic` |
| `token` | Đoạn text câu trả lời |
| `done` | Kết thúc stream |
| `error` | Lỗi (`message`) |

### Biến môi trường (backend)

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_AGENT_MODEL=gpt-4o-mini

# MySQL (fallback khi Neo4j thiếu node)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=123456789
MYSQL_DATABASE=law

# Agent / RAG
AGENT_LLM_MAX_NOIDUNG_CHARS=4000
AGENT_LLM_MAX_PASSAGE_CHARS=4000
AGENT_VECTOR_POOL_MULT=3
RAG_AGENT_LLM_RERANK=0

# Tavily (web_search)
TAVILY_API_KEY=tvly-...

# Lịch sử chat Python/Streamlit (tùy chọn)
MONGODB_URI=
MONGODB_DB=chatbot_rag
MONGODB_COLLECTION=conversations

# Firebase TTS (tùy chọn)
FIREBASE_CRED_PATH=firebase-credentials.json
FIREBASE_STORAGE_BUCKET=your-bucket
```

---

## 4. Chạy frontend (Next.js)

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Mở `http://localhost:3000`.

### `.env.local` (frontend)

```bash
# Chỉ gốc host:port — route API tự nối /rag/stream, /rag/chat, /tts, /stt
RAG_BACKEND_URL=http://127.0.0.1:8001

MONGODB_URI=                    # tùy chọn; không có thì lưu chat trong RAM
MONGODB_DB=chatbot_rag
MONGODB_COLLECTION=conversations
```

### Tính năng UI

- **Hỏi đáp thường** / **Phân tích tình huống** (`query_mode`)
- Chat qua `POST /api/stream` → proxy SSE tới backend
- Sau khi nhận **đủ** câu trả lời: hiển thị **typewriter** (markdown qua `react-markdown`)
- Panel **passages** bên phải (full `noidung` từ meta SSE)
- **TTS** / **STT** (proxy `/api/tts`, `/api/stt`); ghi âm + waveform
- Lịch sử hội thoại: `GET/POST /api/chats`, `PATCH/DELETE /api/chats/[id]` (MongoDB hoặc RAM)
- Dark mode, sidebar pin/rename/delete

---

## 5. Streamlit (tùy chọn)

```bash
source .venv/bin/activate
pip install streamlit
streamlit run app.py
```

`http://localhost:8501` — luồng graph/RAG cũ hơn (có `analyze_complex_situation` cho chế độ tình huống), không dùng agentic API mới.

---

## Cấu trúc thư mục chính

```
project2-chatbot-rag/
├── rag_api.py                 # FastAPI entry
├── app.py                     # Streamlit UI
├── services/
│   ├── agentic_rag.py         # Tool-calling, hydrate Neo4j, rerank, SSE helpers
│   ├── utils.py               # rewrite, intent, generate
│   └── history.py             # Mongo / SQLite / JSON chat
├── retrieve/
│   ├── build_graph.py         # MySQL → Neo4j + GraphRAGRetriever
│   ├── two_stage_search.py    # Milvus client + collection
│   └── tavily_fallback.py
├── agents/
│   ├── pipeline.py            # run_pre_retrieve
│   ├── agents_manager.py      # quick, spam, sentiment
│   └── *_agent.py
├── data_processing/
│   └── seed_data_batch.py     # MySQL → Milvus
├── law-crawler/               # Crawl + MySQL docker
├── test_raptor/               # Milvus + Neo4j docker-compose
├── evaluation/                # benchmark retrieval
└── frontend/                  # Next.js 16 + Tailwind 4
    └── src/
        ├── app/api/stream/    # SSE proxy
        ├── app/api/chats/     # MongoDB chat CRUD
        ├── components/chat/   # MessageRow, PassagePanel, VoiceMessage
        └── store/chat-store.ts
```

---

## Xử lý sự cố

| Triệu chứng | Gợi ý |
|-------------|--------|
| `Collection 'phapdien_simple_tendieu' not exist` | Chạy `seed_data_batch.py` sau khi Milvus healthy |
| Milvus `Connection refused` | `docker compose up -d` trong `test_raptor`, đợi `healthz` |
| Neo4j crash / transaction log | Reset volume: stop neo4j, xóa `test_raptor/volumes/neo4j/databases` + `transactions`, `up -d`, chạy lại `build_graph.py` |
| `etcd` exited | `docker compose up -d` lại stack Milvus; có thể `docker rm` container etcd kẹt rồi tạo mới |
| API import lỗi `retrieve` | Chạy từ root repo; `build_graph.py` đã thêm `sys.path` khi chạy trực tiếp |
| Seed lỗi JSON > 65536 byte | Script tự cắt `noidung`; dùng `--resume` nếu ingest gián đoạn |
| Frontend không có câu trả lời | `RAG_BACKEND_URL` phải là `http://127.0.0.1:8001` (không thêm `/rag/chat`); kiểm tra `uvicorn` port 8001 |
| Stream không hiện status | Kiểm tra proxy `/api/stream` và CORS/network local |

---

## Đánh giá / thử nghiệm

```bash
cd evaluation
pip install -r requirements.txt
python run_evaluation.py
```

Các chiến lược retrieval benchmark nằm trong `retrieve/retrieval_strategies.py` (không phải luồng agentic mặc định).

---

## License

Dự án học thuật / đồ án — tuỳ chính sách nhóm phát triển.
# law-assistant
