# Chatbot RAG — Hỏi đáp pháp luật (Pháp Điển)

Hệ thống hỏi–đáp pháp luật tiếng Việt dựa trên **Retrieval-Augmented Generation (RAG)** và **agentic tool-calling**: lớp pre-retrieve (quick / spam / sentiment), truy xuất vector theo chunk (Milvus), mở rộng đồ thị (Neo4j), tùy chọn web (Tavily), rồi tổng hợp câu trả lời bằng LLM. Có tracing (Langfuse) và dashboard đánh giá (realtime + benchmark).

## Kiến trúc tổng quan

```
┌─────────────────┐     SSE / REST      ┌──────────────────┐
│  Next.js UI     │ ◄──────────────────►│  backend/rag_api.py │
│  (frontend/)    │   /api/stream       │  FastAPI :8001   │
│  MongoDB chats  │   /api/chats        └────────┬─────────┘
└────────┬────────┘                              │
         │                                       ├── agents/pipeline (pre-retrieve)
         │                                       ├── services/agentic_rag.py (+ Langfuse trace)
         │                                       └── retrieve/GraphRAGRetriever
         │                                              ├── Milvus (vector, chunk-level)
         │                                              ├── Neo4j (graph)
         │                                              └── Tavily (web, tùy chọn)
         │
┌────────┴────────┐
│ MySQL `law`     │  ← nguồn gốc từ law-crawler
└─────────────────┘

Mỗi câu hỏi qua chat UI còn được ghi vào evaluation/live_log.py → dashboard đánh giá (realtime)
```

| Thành phần | Vai trò |
|-----------|---------|
| **MySQL** | Lưu Pháp Điển đã crawl (`pddieu`, chủ đề, chương, đề mục) |
| **Milvus** | Embedding + tìm kiếm semantic theo **chunk** (`phapdien_simple_tendieu`) |
| **Neo4j** | Đồ thị điều luật, quan hệ `LIEN_QUAN` / `THAM_CHIEU`, nội dung full cho agent |
| **backend/rag_api.py** | API RAG, stream SSE, STT, ghi live log cho dashboard |
| **frontend** | Chat UI (Next.js 16, mặc định light mode), markdown, passages, STT |
| **evaluation/** | Dashboard đánh giá realtime + benchmark có gold data |
| **Langfuse** | Tracing mọi request thật (tool-calling, token/cost, latency) — tùy chọn |
| **backend/app.py** | Giao diện Streamlit (luồng cũ, tùy chọn) |

> Docker dùng cho **MySQL** (`law-crawler/`), **Milvus + Neo4j** (`infra/`). Code Python và Next.js chạy trên host (venv + `npm`). Dùng `./run.sh` để quản lý tất cả bằng 1 script duy nhất (xem mục Quick Start).

---

## Quick Start (đã setup dữ liệu rồi)

Nếu đã crawl/seed/build graph xong ít nhất 1 lần, chỉ cần:

```bash
./run.sh start       # dựng docker (nếu chưa chạy) + backend + frontend, tự mở trình duyệt
```

Các lệnh khác:

| Lệnh | Tác dụng |
|---|---|
| `./run.sh status` | Xem tiến trình app + docker đang chạy |
| `./run.sh logs [backend\|frontend\|dashboard]` | Tail log realtime (không truyền tên = tất cả) |
| `./run.sh dashboard` | Chạy dashboard đánh giá (`evaluation/visualization.py`, port 8501) |
| `./run.sh open` | Mở lại trình duyệt tới các URL đang chạy |
| `./run.sh restart` | Dừng rồi chạy lại backend/frontend/dashboard (giữ nguyên docker) |
| `./run.sh stop` | Dừng backend/frontend/dashboard (docker vẫn chạy) |
| `./run.sh down` | Dừng tất cả, kể cả docker compose |

Log ghi ở `logs/`, PID ở `.run/` (cả hai đều gitignore). Lần đầu chạy trên máy mới thì làm theo phần **Thiết lập lần đầu** bên dưới trước.

---

## Thiết lập lần đầu

### Yêu cầu

- Python ≥ 3.12 (một số pin trong `requirements.txt`, vd `numpy==2.5.4`, yêu cầu ≥3.12)
- Node.js ≥ 20 (frontend Next.js 16)
- Docker & Docker Compose
- `OPENAI_API_KEY` (embedding `text-embedding-3-small`, chat, rerank LLM tùy chọn)
- Tuỳ chọn: `TAVILY_API_KEY`, MongoDB (lịch sử chat), Langfuse Cloud (tracing)
- RAM: nên có ≥10GB khả dụng cho Docker + 2 model phân loại (sentiment/spam, ~2.5GB) chạy cùng lúc. Trên WSL2, nếu máy lag/treo khi chạy, cân nhắc tăng giới hạn RAM qua `.wslconfig` (`memory=`, `swap=`) rồi `wsl --shutdown`.

### 1. Cài đặt Python

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `torch` không nằm trong `requirements.txt` — cài riêng bản **CPU** để tránh kéo theo CUDA/nvidia-* (vài GB không cần thiết nếu không có GPU, hoặc nếu có GPU NVIDIA thì cài bản CUDA tương ứng thay vì CPU):
> ```bash
> pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
> ```

Copy `.env.example` thành `.env` ở thư mục gốc rồi điền giá trị thật (xem mục biến môi trường bên dưới).

### 2. Chuẩn bị dữ liệu

#### 2.1 Crawl Pháp Điển → MySQL

Tải dữ liệu từ [Pháp Điển](https://phapdien.moj.gov.vn/), giải nén vào `law-crawler/phap-dien/` (`chude.json`, `demuc.json`, `treeNode.json`, thư mục `demuc/`).

```bash
cd law-crawler
docker compose up -d          # MySQL + phpMyAdmin
pip install -r requirements.txt
# Lần đầu (DB trống): bỏ comment dòng db.create_tables trong main.py
python main.py
cd ..
```

MySQL mặc định: `localhost:3306`, DB `law`, user `root`, password `123456789`, phpMyAdmin `http://localhost:8081`. ~66k điều luật, crawl mất khoảng 30-60 phút tuỳ máy (chạy hoàn toàn local, không cần mạng).

#### 2.2 Seed Milvus (có chunking)

```bash
cd infra
docker compose up -d          # etcd, minio, milvus-standalone, neo4j
cd ..
```

Đợi healthy: `curl http://localhost:9091/healthz` → `OK`.

```bash
source .venv/bin/activate
cd data_processing
python seed_data_batch.py              # ingest mới (drop collection cũ)
python seed_data_batch.py --resume     # tiếp tục nếu bị gián đoạn (có checkpoint)
cd ..
```

- Collection: `phapdien_simple_tendieu`
- **Chunking**: điều luật dài được chia thành nhiều chunk theo token (`CHUNK_SIZE_TOKENS=400` mặc định, overlap tuyệt đối `CHUNK_OVERLAP_TOKENS=80` — tính theo SỐ TOKEN, không phải %) thay vì cắt bỏ phần vượt quá — mỗi chunk là 1 vector riêng, cùng `mapc` với điều gốc. Header (chủ đề/đề mục/chương/tên điều) được giữ nguyên ở mỗi chunk.
- Gửi embed theo lô nhỏ (`EMBED_API_BATCH_SIZE=100` mặc định) thay vì cả batch MySQL (200 điều, có thể >10.000 chunk) cùng lúc — tránh vượt giới hạn request OpenAI / giới hạn message gRPC của Milvus (64MB).
- Checkpoint resume lưu ở `data_processing/.seed_checkpoint.json` (doc_offset + next_id) — an toàn khi bị ngắt giữa chừng, `--resume` sẽ tiếp tục đúng chỗ.
- Lỗi embed (nếu có, sau khi đã thử lại từng chunk) ghi vào `embed_errors.csv`.
- Ước tính chi phí: ~160k vector cho 66k điều, `text-embedding-3-small` ($0.02/1M token) ≈ 1-2 USD.

#### 2.3 Build đồ thị Neo4j

```bash
python retrieve/build_graph.py
```

Neo4j Browser: `http://localhost:7474` — `neo4j` / `password123`. Build ~45 ChuDe, ~300 DeMuc, ~1.5k Chuong, 66k Dieu, ~38k cạnh `LIEN_QUAN`, ~27k cạnh `THAM_CHIEU` — mất vài chục giây tới vài phút.

#### 2.4 Chạy hệ thống

```bash
./run.sh start
```

Hoặc chạy tay từng phần — xem mục **Chạy tay (không dùng run.sh)** bên dưới.

---

## Chạy tay (không dùng run.sh)

### Backend (FastAPI)

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

### Frontend (Next.js)

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
- Panel **passages** bên phải (full `noidung` từ meta SSE), điểm xếp hạng hiển thị % (kẹp tối đa 100% — đây là điểm cộng dồn base+BM25+semantic dùng để sort, không phải xác suất)
- **STT** (proxy `/api/stt`): nhập câu hỏi bằng giọng nói, ghi âm + waveform
- Lịch sử hội thoại: `GET/POST /api/chats`, `PATCH/DELETE /api/chats/[id]` (MongoDB hoặc RAM) — gửi tin nhắn khi chưa có đoạn chat nào active sẽ tự tạo đoạn chat mới
- **Light mode mặc định**, sidebar pin/rename/delete, nút chuyển light/dark ở góc trên

---

## Luồng xử lý (backend)

**Bước 1 — Pre-retrieve** (`agents/pipeline.py` → `AgentsManager`, có `@observe()` Langfuse):

| Action | Ý nghĩa |
|--------|---------|
| `quick` | Câu hỏi FAQ — trả lời ngay, không RAG |
| `spam` | Từ chối spam |
| `escalate` | Sentiment tiêu cực — đề xuất chuyển cán bộ |
| `proceed` | Tiếp tục agentic RAG |

**Bước 2 — Khi `proceed`:** `rewrite_query_v2` + `detect_intent`. Nếu intent ngoài phạm vi → trả lời từ chối.

**Bước 3 — Agentic loop** (`services/agentic_rag.py`, tối đa 6 vòng, có `@observe()` Langfuse cho cả loop lẫn từng tool call):

| Tool | Mô tả |
|------|--------|
| `vector_search` | Milvus (chunk-level) → dedup theo `mapc` (oversample raw hit để bù việc 1 điều có nhiều chunk) → hydrate Neo4j (full nội dung) → **rerank** |
| `graph_traverse` | Mở rộng Neo4j theo `LIEN_QUAN` / `THAM_CHIEU` → rerank |
| `web_search` | Tavily (khi có `TAVILY_API_KEY`) |
| `get_full_article` | Full một điều theo `mapc` |

Passages trả về UI lấy **full text từ Neo4j** (bucket agent), không phụ thuộc snippet Milvus. MySQL là fallback khi Neo4j thiếu node.

**Rerank:** hybrid BM25 (từ khóa) + embedding cục bộ (ngữ nghĩa, `sentence-transformers`, không gọi OpenAI) + điểm gốc Milvus; bật thêm LLM rerank: `RAG_AGENT_LLM_RERANK=1`. Kết quả `vector_search` được cache theo `(query, top_k)` trong `retrieve/.cache/` (SQLite) để tránh gọi lại OpenAI embedding cho câu hỏi lặp — tắt bằng `RAG_CACHE_ENABLED=0`.

**Bước 4 — Đánh giá chất lượng câu trả lời** (`check_answer_grounding` + `check_rag_quality_extra`, embedding cục bộ, miễn phí — xem mục Đánh giá):
- `grounding_score` (faithfulness): câu trả lời có bám sát passage đã retrieve
- `answer_relevance`: câu trả lời có đúng trọng tâm câu hỏi
- `context_relevance`: passage đã retrieve có liên quan tới câu hỏi

### Sự kiện SSE (`POST /rag/stream`)

| `type` | Nội dung |
|--------|----------|
| `status` | Tiến trình agent (`text`, `iteration`, `max`) |
| `meta` | `action`, `normalized_query`, `passages`, `sources`, `iterations`, `agentic` |
| `token` | Đoạn text câu trả lời |
| `grounding` | `grounding_score`, `flagged_sentences` (câu nghi bịa), `answer_relevance`, `context_relevance` |
| `done` | Kết thúc stream |
| `error` | Lỗi (`message`) |

Mỗi request (quick/spam/escalate/rejected/proceed/error) còn được ghi vào `evaluation/results/live_chat_log.jsonl` (xem mục Đánh giá) — không ảnh hưởng tới response nếu ghi log lỗi.

---

## Biến môi trường

Copy `.env.example` thành `.env` ở thư mục gốc rồi điền giá trị thật (`.env` đã có trong `.gitignore`, không commit file này).

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_AGENT_MODEL=gpt-4o-mini

# Agentic RAG tuning
AGENT_LLM_MAX_NOIDUNG_CHARS=4000
AGENT_LLM_MAX_PASSAGE_CHARS=4000
AGENT_VECTOR_POOL_MULT=3
RAG_AGENT_LLM_RERANK=0
AGENT_GROUNDING_THRESHOLD=0.35     # ngưỡng cosine similarity đánh dấu câu "nghi bịa"
RAG_CACHE_ENABLED=1                # cache vector_search theo (query, top_k)
RAG_VECTOR_RAW_OVERSAMPLE=4        # oversample raw hit Milvus trước khi dedup theo mapc
RAG_VECTOR_RAW_LIMIT_CAP=200

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

# Chunking khi seed Milvus (data_processing/seed_data_batch.py)
CHUNK_SIZE_TOKENS=400
CHUNK_OVERLAP_TOKENS=80
EMBED_API_BATCH_SIZE=100

# Tavily (web_search) — tùy chọn
TAVILY_API_KEY=tvly-...

# Lịch sử chat Python/Streamlit — tùy chọn
MONGODB_URI=
MONGODB_DB=chatbot_rag
MONGODB_COLLECTION=conversations

# Langfuse tracing/observability — tùy chọn, để trống thì tắt, không lỗi
# Lấy tại: https://cloud.langfuse.com → Project → Settings → API Keys
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com   # hoặc https://us.cloud.langfuse.com nếu tạo project ở region US
```

---

## Tracing (Langfuse)

Khi đã điền `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`, mọi request thật (`/rag/chat`, `/rag/stream`) tự động được trace:
- 1 trace/request (`rag_chat_request` hoặc `rag_stream_request`), gộp cả bước pre-retrieve lẫn agentic loop
- Mỗi tool call (`vector_search`, `graph_traverse`, ...) là 1 span con
- Mọi lệnh gọi OpenAI (chat + embedding) tự động thành generation kèm token/cost (qua `langfuse.openai.OpenAI` — drop-in thay `openai.OpenAI`)

Xem tại `https://cloud.langfuse.com` (hoặc `https://us.cloud.langfuse.com` tuỳ region) → project → **Traces**. Không điền key thì tính năng này tắt hoàn toàn, không ảnh hưởng ứng dụng.

---

## Đánh giá

Có 2 hướng đánh giá, bổ sung cho nhau:

### A. Lịch sử câu hỏi thực tế (live, không cần gold data)

Mọi câu hỏi hỏi qua chat UI được tự động ghi vào `evaluation/results/live_chat_log.jsonl` (xem `evaluation/live_log.py`, `backend/rag_api.py`) — **không cần chạy script riêng, không cần upload file**. Dùng 3 chỉ số kiểu RAGAS tính bằng embedding cục bộ (miễn phí, không gọi OpenAI):

| Chỉ số | So sánh | Phát hiện |
|---|---|---|
| **Grounding** (faithfulness) | câu trả lời ↔ passage đã retrieve | câu trả lời có bịa không |
| **Answer Relevance** | câu hỏi ↔ câu trả lời | câu trả lời có lạc đề không |
| **Context Relevance** | câu hỏi ↔ passage đã retrieve | retrieval có tìm đúng tài liệu không |

Không tính được Precision/Recall/ROUGE-L/BLEU ở đây vì các chỉ số đó **bắt buộc phải có đáp án mẫu** (`gold_docs`/`gold_response`) — câu hỏi tự do không có đáp án mẫu nên không thể tự động chấm theo kiểu đó.

### B. Benchmark với bộ test chuẩn (cần gold data)

`evaluation/run_evaluation.py` gọi thẳng `run_agentic_rag()` — đúng pipeline thật đang chạy trong `backend/rag_api.py`.

```bash
cd evaluation
pip install -r requirements.txt
python run_evaluation.py --limit 1          # xem ước tính chi phí, KHÔNG chạy thật
python run_evaluation.py --limit 2 --yes    # chạy thử với 2 câu hỏi (tốn phí OpenAI thật)
python run_evaluation.py --yes              # chạy toàn bộ data.json
```

⚠️ Mặc định script chỉ in ước tính chi phí rồi dừng — phải thêm `--yes` mới thực sự gọi OpenAI. Dùng `--limit N` để thử rẻ trước. Kết quả gồm retrieval metrics (Precision/Recall/Hit/MRR/NDCG@k), response metrics (ROUGE-L/BLEU/Semantic Similarity) và grounding trung bình, xuất ra `evaluation/results/agentic_evaluation_results.json`. Trong lúc chạy, ghi tiến trình realtime vào `evaluation/results/_live_progress.json` — dashboard tự hiện banner "ĐANG CHẠY" và progress bar.

⚠️ **Hạn chế đã biết**: `gold_docs` trong `evaluation/data.json` lưu **OFFSET dòng MySQL**, không phải `mapc` trực tiếp. Nếu crawl lại dữ liệu (`law-crawler/main.py`), thứ tự dòng có thể đổi → offset trỏ sai điều luật → Precision/Recall/Hit/MRR/NDCG@k có thể tụt về 0 dù hệ thống vẫn trả lời đúng (kiểm chứng qua response metrics/grounding vẫn tốt). Khi gặp tình huống này, ưu tiên đọc ROUGE-L/BLEU/Semantic Similarity/grounding thay vì retrieval metrics.

`retrieve/retrieval_strategies.py` (4 chiến lược retrieval độc lập) vẫn còn trong repo để tham khảo/so sánh nhưng không còn là công cụ đánh giá chính.

### Dashboard đánh giá

```bash
./run.sh dashboard
# hoặc: cd evaluation && streamlit run visualization.py
```

Mở `http://localhost:8501`. Gồm 2 mục chính: **💬 Lịch sử câu hỏi thực tế** (mục A, hiện sẵn không cần thao tác) và **📂 So sánh với bộ test chuẩn** (mục B, chọn file có sẵn hoặc upload). Có expander **📖 Giải thích các chỉ số đánh giá & Top-K** ở đầu trang giải thích chi tiết từng chỉ số và cách đọc Top-K.

---

## Cấu trúc thư mục chính

```
LLM_Prj1/
├── run.sh                       # Quản lý toàn hệ thống: start/stop/restart/down/status/logs/dashboard
├── backend/
│   ├── rag_api.py              # FastAPI entry, ghi live log
│   └── app.py                  # Streamlit UI (luồng cũ, tùy chọn)
├── services/
│   ├── agentic_rag.py          # Tool-calling loop (run_agentic_rag), dedup, rerank, grounding/answer/context relevance, Langfuse
│   ├── utils.py                # rewrite, intent, generate (OpenAI client qua langfuse.openai)
│   ├── local_embeddings.py     # Model embedding cục bộ dùng chung (rerank + đánh giá)
│   └── history.py              # Mongo / SQLite / JSON chat
├── retrieve/
│   ├── build_graph.py          # MySQL → Neo4j + GraphRAGRetriever (dedup mapc)
│   ├── two_stage_search.py     # Milvus client + collection (OpenAI client qua langfuse.openai)
│   └── tavily_fallback.py
├── agents/
│   ├── pipeline.py             # run_pre_retrieve (@observe Langfuse)
│   ├── agents_manager.py       # quick, spam, sentiment
│   └── *_agent.py
├── data_processing/
│   └── seed_data_batch.py      # MySQL → Milvus, có chunking + checkpoint resume
├── law-crawler/                # Crawl + MySQL docker
├── infra/                      # Milvus + Neo4j docker-compose
├── evaluation/
│   ├── run_evaluation.py       # Benchmark có gold data (ghi live progress)
│   ├── visualization.py        # Dashboard Streamlit (live log + benchmark)
│   ├── live_log.py             # Ghi mọi câu hỏi thật từ backend/rag_api.py
│   ├── metric_docs.py          # Nội dung giải thích chỉ số, dùng chung CLI + dashboard
│   └── results/                # Output JSON/JSONL (gitignored phần runtime)
├── scripts/
│   └── watch_crawl.sh          # Theo dõi tiến trình crawl MySQL realtime
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
| `etcd`/`minio` exited, lỗi "file access denied" | Image minio (`cgr.dev/chainguard/minio`) chạy user nonroot (uid 65532) — nếu `infra/volumes/minio/` do Docker tự tạo thuộc root, chown lại: `docker run --rm -v "$(pwd)/infra/volumes/minio:/minio_data" alpine chown -R 65532:65532 /minio_data` |
| Seed Milvus lỗi gRPC `RESOURCE_EXHAUSTED: received message larger than max` | Batch MySQL sinh quá nhiều chunk (điều rất dài) khiến 1 lệnh insert vượt 64MB — đã fix bằng cách insert theo từng lô nhỏ (`EMBED_API_BATCH_SIZE`); nếu vẫn gặp, giảm `EMBED_API_BATCH_SIZE` |
| Seed Milvus: 1 batch OpenAI lỗi rồi rơi xuống embed từng chunk (rất chậm) | Giảm `EMBED_API_BATCH_SIZE` (mặc định 100) nếu vẫn gặp giới hạn request OpenAI |
| API import lỗi `retrieve`/`services`/`agents`/`evaluation` | Chạy từ root repo; `backend/rag_api.py`, `backend/app.py`, `build_graph.py` đều tự thêm `sys.path` khi chạy trực tiếp |
| Frontend không có câu trả lời | `RAG_BACKEND_URL` phải là `http://127.0.0.1:8001` (không thêm `/rag/chat`); kiểm tra `uvicorn` port 8001 |
| Bấm gửi/Enter không phản ứng gì trên chat UI | Thường do máy thiếu RAM (swap đầy) khiến trình duyệt/Node lag — kiểm tra `free -h`; nếu RAM/swap bình thường mà vẫn vậy, hard refresh trang (Ctrl+Shift+R) |
| Điểm % tài liệu trong panel passages > 100% | Đã fix — điểm là tổng base_score + BM25 + semantic (có thể >1.0), hiển thị giờ kẹp ở 100% |
| Máy lag/treo khi chạy đủ Docker + backend (2 model phân loại) | Tăng giới hạn RAM WSL2 qua `.wslconfig` (`memory=`, `swap=`), chạy `wsl --shutdown` rồi mở lại; hoặc tắt bớt tiến trình nặng khác (IDE extension, container không liên quan) |
| Retrieval metrics (P/R/Hit/MRR/NDCG) = 0 ở benchmark | Xem mục "Hạn chế đã biết" trong phần Đánh giá — có thể do `gold_docs` lệch sau khi crawl lại dữ liệu |

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
