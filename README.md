# Chatbot RAG — Hỏi đáp pháp luật (Pháp Điển)

Hệ thống hỏi–đáp pháp luật tiếng Việt dựa trên **Retrieval-Augmented Generation (RAG)** và **agentic tool-calling**: lớp pre-retrieve (quick / spam / sentiment), truy xuất vector (Milvus), mở rộng đồ thị (Neo4j), tùy chọn web (Tavily), rồi tổng hợp câu trả lời bằng LLM.

## Kiến trúc tổng quan

```
┌─────────────────┐     SSE / REST      ┌──────────────────┐
│  Next.js UI     │ ◄──────────────────►│  backend/rag_api.py │
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
| **backend/rag_api.py** | API RAG, stream SSE, STT |
| **frontend** | Chat UI (dark mode), markdown, passages, STT (nhập câu hỏi bằng giọng nói), lịch sử MongoDB |
| **backend/app.py** | Giao diện Streamlit (luồng cũ, tùy chọn) |

> Docker dùng cho **MySQL** (`law-crawler/`), **Milvus + Neo4j** (`infra/`). Code Python và Next.js chạy trên host (venv + `npm`).

---

## Yêu cầu

- Python ≥ 3.10
- Node.js ≥ 20 (frontend Next.js 16)
- Docker & Docker Compose
- `OPENAI_API_KEY` (embedding `text-embedding-3-small`, chat, rerank LLM tùy chọn)
- Tuỳ chọn: `TAVILY_API_KEY`, MongoDB (lịch sử chat)

---

## 1. Cài đặt Python

```bash
cd law-assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pymysql           # seed Milvus từ MySQL
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
cd infra
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
- Embed theo **block đầy đủ** (chủ đề + đề mục + chương + tên điều + nội dung, cắt ở 12.000 ký tự cho input embed), metadata JSON (có `mapc`, `noidung` tự cắt nếu vượt ~64KB Milvus)
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
uvicorn backend.rag_api:app --host 0.0.0.0 --port 8001 --reload
```

| Endpoint | Mô tả |
|----------|--------|
| `GET /health` | Health check |
| `POST /rag/chat` | Chat đồng bộ (JSON) |
| `POST /rag/stream` | Chat SSE (xem bảng sự kiện bên dưới) |
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

**Rerank:** hybrid BM25 (từ khóa) + embedding cục bộ (ngữ nghĩa, `sentence-transformers`, không gọi OpenAI) + điểm gốc Milvus; bật thêm LLM rerank: `RAG_AGENT_LLM_RERANK=1`. Kết quả `vector_search` được cache theo `(query, top_k)` trong `retrieve/.cache/` (SQLite) để tránh gọi lại OpenAI embedding cho câu hỏi lặp — tắt bằng `RAG_CACHE_ENABLED=0`.

### Sự kiện SSE (`POST /rag/stream`)

| `type` | Nội dung |
|--------|----------|
| `status` | Tiến trình agent (`text`, `iteration`, `max`) |
| `meta` | `action`, `normalized_query`, `passages`, `sources`, `iterations`, `agentic` |
| `token` | Đoạn text câu trả lời |
| `grounding` | Kiểm tra độ trung thực câu trả lời sau cùng: `grounding_score` (0..1, embedding cục bộ) + `flagged_sentences` (câu nghi bịa) |
| `done` | Kết thúc stream |
| `error` | Lỗi (`message`) |

### Biến môi trường (backend)

Copy `.env.example` thành `.env` ở thư mục gốc rồi điền giá trị thật (`.env` đã có trong `.gitignore`, không commit file này).

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

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

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
# Chỉ gốc host:port — route API tự nối /rag/stream, /rag/chat, /stt
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
- **STT** (proxy `/api/stt`): nhập câu hỏi bằng giọng nói, ghi âm + waveform
- Lịch sử hội thoại: `GET/POST /api/chats`, `PATCH/DELETE /api/chats/[id]` (MongoDB hoặc RAM)
- Dark mode, sidebar pin/rename/delete

---

## 5. Streamlit (tùy chọn)

```bash
source .venv/bin/activate
pip install streamlit
streamlit run backend/app.py
```

`http://localhost:8501` — luồng graph/RAG cũ hơn (có `analyze_complex_situation` cho chế độ tình huống), không dùng agentic API mới.

---

## Cấu trúc thư mục chính

```
law-assistant/
├── backend/
│   ├── rag_api.py              # FastAPI entry
│   └── app.py                  # Streamlit UI (luồng cũ, tùy chọn)
├── services/
│   ├── agentic_rag.py          # Tool-calling loop (run_agentic_rag), hydrate Neo4j, rerank, SSE helpers
│   ├── utils.py                # rewrite, intent, generate
│   └── history.py              # Mongo / SQLite / JSON chat
├── retrieve/
│   ├── build_graph.py          # MySQL → Neo4j + GraphRAGRetriever
│   ├── two_stage_search.py     # Milvus client + collection
│   └── tavily_fallback.py
├── agents/
│   ├── pipeline.py             # run_pre_retrieve
│   ├── agents_manager.py       # quick, spam, sentiment
│   └── *_agent.py
├── data_processing/
│   └── seed_data_batch.py      # MySQL → Milvus
├── law-crawler/                # Crawl + MySQL docker
├── infra/                      # Milvus + Neo4j docker-compose
├── evaluation/                 # benchmark retrieval
└── frontend/                   # Next.js 16 + Tailwind 4
    └── src/
        ├── app/api/stream/     # SSE proxy
        ├── app/api/chats/      # MongoDB chat CRUD
        ├── components/chat/    # MessageRow, PassagePanel, VoiceMessage
        └── store/chat-store.ts
```

---

## Xử lý sự cố

| Triệu chứng | Gợi ý |
|-------------|--------|
| `Collection 'phapdien_simple_tendieu' not exist` | Chạy `seed_data_batch.py` sau khi Milvus healthy |
| Milvus `Connection refused` | `docker compose up -d` trong `infra`, đợi `healthz` |
| Neo4j crash / transaction log | Reset volume: stop neo4j, xóa `infra/volumes/neo4j/databases` + `transactions`, `up -d`, chạy lại `build_graph.py` |
| `etcd` exited | `docker compose up -d` lại stack Milvus; có thể `docker rm` container etcd kẹt rồi tạo mới |
| API import lỗi `retrieve`/`services`/`agents` | Chạy từ root repo; `backend/rag_api.py`, `backend/app.py`, `build_graph.py` đều tự thêm `sys.path` khi chạy trực tiếp |
| Seed lỗi JSON > 65536 byte | Script tự cắt `noidung`; dùng `--resume` nếu ingest gián đoạn |
| Frontend không có câu trả lời | `RAG_BACKEND_URL` phải là `http://127.0.0.1:8001` (không thêm `/rag/chat`); kiểm tra `uvicorn` port 8001 |
| Stream không hiện status | Kiểm tra proxy `/api/stream` và CORS/network local |

---

## Đánh giá / thử nghiệm

`evaluation/run_evaluation.py` gọi thẳng `run_agentic_rag()` — đúng pipeline thật đang chạy trong `backend/rag_api.py` (không phải `retrieve/retrieval_strategies.py` độc lập trước đây).

```bash
cd evaluation
pip install -r requirements.txt
python run_evaluation.py --limit 1          # xem ước tính chi phí, KHÔNG chạy thật
python run_evaluation.py --limit 2 --yes    # chạy thử với 2 câu hỏi (tốn phí OpenAI thật)
python run_evaluation.py --yes              # chạy toàn bộ data.json
```

⚠️ Mặc định script chỉ in ước tính chi phí rồi dừng — phải thêm `--yes` mới thực sự gọi OpenAI (embedding + GPT-4o-mini, tối đa 6 vòng/câu hỏi). Dùng `--limit N` để thử rẻ trước. Kết quả gồm retrieval metrics (Precision/Recall/Hit/MRR/NDCG@k), response metrics (ROUGE-L/BLEU/Semantic Similarity) và điểm trung thực (`grounding_score`) trung bình, xuất ra `evaluation/results/agentic_evaluation_results.json`.

`retrieve/retrieval_strategies.py` (4 chiến lược retrieval độc lập) vẫn còn trong repo để tham khảo/so sánh nhưng không còn là công cụ đánh giá chính.

### Dashboard thống kê kết quả đánh giá

```bash
cd evaluation
streamlit run visualization.py
```

Mở `http://localhost:8501`, chọn file kết quả có sẵn trong `evaluation/results/` (hoặc tải file
khác lên). Hiển thị: số câu hỏi đã test, ROUGE-L/BLEU/Semantic Similarity trung bình, điểm trung
thực (`grounding`) trung bình, biểu đồ Precision/Recall/Hit/MRR/NDCG theo Top-K, và bảng chi tiết
từng câu hỏi có cảnh báo câu trả lời nghi "bịa" (grounding thấp). Tự nhận diện được cả file kết
quả cũ (so sánh nhiều chiến lược trong `retrieval_strategies.py`) lẫn file mới.

---

## Kiểm thử tự động (tests/CI)

```bash
pip install pytest
pytest tests/ -v
```

13 unit test thuần (mock embedding/MySQL, không cần OpenAI/Milvus/Neo4j sống) cho: rerank hybrid (`tests/test_hybrid_rerank.py`), kiểm tra độ trung thực (`tests/test_grounding.py`), các metric retrieval (`tests/test_retrieval_metrics.py`). Chạy tự động qua GitHub Actions (`.github/workflows/tests.yml`) ở mỗi push/PR.

---

## License

Dự án học thuật / đồ án — tuỳ chính sách nhóm phát triển.
# law-assistant
