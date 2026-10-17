# Kiến trúc hệ thống — Vietnamese Legal QA RAG Chatbot

> Tài liệu này phân tích chi tiết kiến trúc **hiện trạng** của mã nguồn trong repo này (được sao chép từ
> `chatbot-group14`, không giữ lịch sử git, ngày 2026-10-17). Mục tiêu: làm nền tảng kỹ thuật để đối chiếu
> với định hướng học thuật trong `Proposal.docx` và lập kế hoạch phát triển tiếp theo (xem `docs/DEVELOPMENT_PLAN.md`).

## Mục lục

1. [Tổng quan & bức tranh lớn](#1-tổng-quan--bức-tranh-lớn)
2. [Data layer — nguồn dữ liệu & MySQL](#2-data-layer--nguồn-dữ-liệu--mysql)
3. [Ingestion / Indexing pipeline](#3-ingestion--indexing-pipeline)
4. [Retrieval layer](#4-retrieval-layer)
5. [Pre-retrieve agent layer (`agents/`)](#5-pre-retrieve-agent-layer-agents)
6. [Agentic RAG core (`services/agentic_rag.py`)](#6-agentic-rag-core-servicesagentic_ragpy)
7. [Query understanding (`services/utils.py`)](#7-query-understanding-servicesutilspy)
8. [API layer (`rag_api.py` — FastAPI)](#8-api-layer-rag_apipy--fastapi)
9. [Frontend (Next.js 16 / React 19)](#9-frontend-nextjs-16--react-19)
10. [Persistence layer — lịch sử chat](#10-persistence-layer--lịch-sử-chat)
11. [Evaluation harness](#11-evaluation-harness)
12. [Legacy Streamlit app (`app.py`)](#12-legacy-streamlit-app-apppy)
13. [Hạ tầng / triển khai](#13-hạ-tầng--triển-khai)
14. [Khoảng trống kiến trúc & nợ kỹ thuật](#14-khoảng-trống-kiến-trúc--nợ-kỹ-thuật)

---

## 1. Tổng quan & bức tranh lớn

Hệ thống là một **chatbot hỏi–đáp pháp luật tiếng Việt** dựa trên Retrieval-Augmented Generation (RAG),
đi xa hơn một pipeline RAG cổ điển (retrieve → generate) bằng hai lớp bổ sung:

- **Pre-retrieve agents**: lọc câu hỏi (quick-FAQ / spam / sentiment) trước khi chạm vào pipeline RAG.
- **Agentic tool-calling loop**: thay vì retrieve 1 lần cố định, LLM chủ động gọi 4 "tools" (vector search,
  graph traversal, web search, full-article fetch) tối đa 6 vòng rồi mới tổng hợp câu trả lời.

```mermaid
flowchart TB
    subgraph Client["Next.js 16 UI (frontend/)"]
        UI[Chat UI<br/>page.tsx]
    end

    subgraph NextAPI["Next.js API routes (proxy)"]
        STREAM["/api/stream"]
        CHATS["/api/chats, /api/chats/[id]"]
        TTS_FE["/api/tts, /api/stt"]
        MONGO_FE[(MongoDB<br/>hoặc in-memory Map)]
    end

    subgraph Backend["FastAPI — rag_api.py :8001"]
        PRE["agents/pipeline.py<br/>run_pre_retrieve"]
        UTILS["services/utils.py<br/>rewrite_query_v2, detect_intent"]
        AGENTIC["services/agentic_rag.py<br/>tool-calling loop (≤6 vòng)"]
        TTS_BE["/tts (OpenAI tts-1)"]
        STT_BE["/stt (OpenAI whisper-1)"]
        HIST["services/history.py<br/>Mongo→SQLite→JSON fallback"]
    end

    subgraph DataStores["Nguồn dữ liệu"]
        MYSQL[(MySQL `law`<br/>pdchude/pddemuc/pdchuong/pddieu)]
        MILVUS[(Milvus<br/>phapdien_simple_tendieu)]
        NEO4J[(Neo4j<br/>ChuDe→DeMuc→Chuong→Dieu<br/>+ LIEN_QUAN/THAM_CHIEU)]
        TAVILY[/Tavily Web Search/]
    end

    UI -->|POST JSON| STREAM --> AGENTIC
    UI --> CHATS --> MONGO_FE
    UI --> TTS_FE --> TTS_BE
    UI --> TTS_FE --> STT_BE

    PRE --> UTILS --> AGENTIC
    AGENTIC -->|vector_search| MILVUS
    AGENTIC -->|hydrate full text| NEO4J
    AGENTIC -->|graph_traverse| NEO4J
    AGENTIC -->|web_search| TAVILY
    NEO4J -.fallback khi thiếu node.-> MYSQL
    TTS_BE --> HIST

    MYSQL -->|seed_data_batch.py| MILVUS
    MYSQL -->|build_graph.py| NEO4J
```

**Luồng chính** (khớp với state machine mô tả trong README gốc): `prompt → run_pre_retrieve → {quick | spam |
escalate | proceed}`. Khi `proceed`: `rewrite_query_v2 → detect_intent → (nếu hợp lệ) agentic loop → answer +
passages`.

---

## 2. Data layer — nguồn dữ liệu & MySQL

- **Nguồn**: Bộ Pháp Điển Việt Nam (`phapdien.moj.gov.vn`) — crawl tĩnh (HTML đã tải sẵn vào
  `law-crawler/phap-dien/` dưới dạng `chude.json`, `demuc.json`, `treeNode.json`, thư mục `demuc/*.html`).
- **`law-crawler/main.py`**: parse các file HTML trên bằng BeautifulSoup, insert theo thứ tự phân cấp
  Chủ đề → Đề mục → Chương → Điều vào MySQL qua ORM **peewee** (`law-crawler/models/models.py`), hỗ trợ
  checkpoint để resume khi crawl bị gián đoạn.
- **Schema MySQL** (`law-crawler/models/models.py`):

  | Bảng | Khóa chính | Quan hệ | Ghi chú |
  |---|---|---|---|
  | `pdchude` (PDChuDe) | `id` (char) | — | Chủ đề pháp điển |
  | `pddemuc` (PDDeMuc) | `id` (char) | `chude_id → pdchude` | Đề mục |
  | `pdchuong` (PDChuong) | `mapc` (char) | `demuc_id → pddemuc` | Chương |
  | `pddieu` (PDDieu) | `mapc` (char) | `demuc_id, chuong_id, chude_id` | **Điều luật** — đơn vị nội dung nhỏ nhất, có `noidung`, `vbqppl` (tên văn bản pháp quy), `vbqppl_link` |
  | `pdtable` (PDTable) | — | `dieu_id → pddieu` | Bảng HTML nhúng trong điều (nếu có) |
  | `pdfile` (PDFile) | — | `dieu_id → pddieu` | File đính kèm |
  | `pdmuclienquan` (PDMucLienQuan) | — | `dieu_id1, dieu_id2 → pddieu` | Cặp điều luật "liên quan" (nguồn của quan hệ `LIEN_QUAN` trong Neo4j) |

- **`law-crawler/document-crawler/main.py`**: crawl bổ sung toàn văn văn bản quy phạm pháp luật gốc
  (`vbqppl_link`) từ `pddieu`, ghi vào bảng `vbpl` — độc lập với pipeline chính, không thấy được tiêu thụ bởi
  `rag_api.py`/`agentic_rag.py`.
- **Hạ tầng**: `law-crawler/docker-compose.yml` dựng MySQL 8 (`3306`) + phpMyAdmin (`8081`). Chỉ dùng cho
  giai đoạn chuẩn bị dữ liệu, không phải runtime của chatbot.

---

## 3. Ingestion / Indexing pipeline

Hai pipeline **MySQL → Milvus** và **MySQL → Neo4j** độc lập, chạy thủ công một lần (hoặc resume) trước khi
backend hoạt động.

### 3.1. MySQL → Milvus (`data_processing/seed_data_batch.py`)

- Đọc toàn bộ `pddieu` JOIN `pdchuong/pddemuc/pdchude` qua `pymysql` (một JOIN duy nhất, load hết vào RAM).
- **Đơn vị embed hiện tại: chỉ `tendieu`** (tên điều, dòng 192: `texts = [(rec.get("tendieu") or "").strip()
  for rec in records]`) — *không* embed toàn bộ `noidung`. Đây là điểm khác biệt quan trọng so với hướng
  "chunking nội dung dài" mà `Proposal.docx` đề cập (mục 5.2) — xem thêm ở `DEVELOPMENT_PLAN.md`.
- Model embedding: OpenAI `text-embedding-3-small` (1536 chiều), batch 200 bản ghi/lần.
- Collection Milvus `phapdien_simple_tendieu`: schema gồm `id` (INT64, primary), `metadata` (JSON — chứa
  `mapc, tenchude, tendemuc, tenchuong, tendieu, noidung`), `embedding` (FLOAT_VECTOR dim=1536, index
  HNSW/COSINE, `M=16, efConstruction=200`).
- `noidung` trong metadata JSON bị cắt lùi dần (từng 4000 ký tự) cho tới khi JSON ≤ 64 000 byte (giới hạn
  Milvus) — tiềm ẩn mất nội dung với điều luật rất dài.
- Hỗ trợ `--resume` (tiếp tục từ `num_entities` hiện có, không drop collection) và fallback embed-từng-bản-ghi
  khi cả batch lỗi, ghi lỗi vào `embed_errors.csv`.
- **Lưu ý**: `data_processing/ingest.py` là một pipeline **Elasticsearch** riêng biệt (index `law_data`, custom
  Vietnamese legal synonym analyzer rất chi tiết) — không thấy được gọi từ `rag_api.py` hay `agentic_rag.py`;
  có vẻ là tàn dư từ một hướng tiếp cận cũ (`services/utils.py` cũng có `get_es()` không dùng trong luồng hiện
  hành).

### 3.2. MySQL → Neo4j (`retrieve/build_graph.py`, hàm `main()`)

- Tạo constraints/unique index cho 4 label + 1 full-text index (`dieu_fulltext` trên `Dieu.ten/noidung`,
  tạo nhưng **không thấy được dùng** ở nơi khác trong code đã đọc).
- Insert theo thứ tự: `ChuDe` → `DeMuc` (+ `CO_DE_MUC`) → `Chuong` (+ `CO_CHUONG`) → `Dieu` theo batch 500
  (+ `CO_DIEU` từ Chuong→Dieu) → `LIEN_QUAN` (hai chiều, từ bảng `pdmuclienquan`) → `THAM_CHIEU`.
- **`THAM_CHIEU` được suy ra bằng regex** `\b(\d+\.\d+\.[A-ZĐ]+\.\d+(?:\.\d+)*)\b` (mẫu mã điều luật, vd
  `1.1.LQ.1`) quét trong `noidung` của **chỉ 5 dòng đầu** (dòng 199: `LIMIT 5` — rõ ràng là code debug/thử
  nghiệm chưa chạy full dataset; cần xác nhận lại trước khi dùng cho production/đánh giá).
- **Schema đồ thị Neo4j**:

  ```
  (ChuDe)-[:CO_DE_MUC]->(DeMuc)-[:CO_CHUONG]->(Chuong)-[:CO_DIEU]->(Dieu)
  (Dieu)-[:LIEN_QUAN]->(Dieu)      // hai chiều, từ pdmuclienquan
  (Dieu)-[:THAM_CHIEU]->(Dieu)     // suy ra từ regex trong noidung (xem lưu ý trên)
  ```

- **`GraphRAGRetriever`** (lớp chính trong file này, dùng bởi Streamlit `app.py` *và* tái sử dụng (chỉ phần
  `_vector_search`, `_graph_expand`, `_rerank`) bởi `services/agentic_rag.py`):
  - `_vector_search(query, top_k)`: embed query → Milvus ANN search (COSINE, `ef=128`) → map `metadata.mapc`.
  - `_graph_expand(seed_mapcs, hop)`: Cypher `MATCH (seed)-[:LIEN_QUAN|THAM_CHIEU*1..hop]-(neighbor)`, gắn
    `base_score=1.0` cho seed / `0.5` cho neighbor, giới hạn 40 kết quả.
  - `_rerank(candidates, query, top_k)`: rerank đơn giản = `base_score + 0.05 × số từ khóa query xuất hiện
    trong (ten+noidung)` — **không dùng embedding similarity thật ở bước này**, chỉ keyword overlap.
  - `retrieve()`: gọi `decompose_query()` (Tavily fallback module) để tách truy vấn pháp lý / thời sự / tính
    toán → vector search + graph expand + rerank → fallback Tavily nếu `is_context_sufficient()` trả về
    `False` → chèn kết quả realtime/tính toán nếu có.

---

## 4. Retrieval layer

Có **3 cài đặt retrieval song song**, phục vụ 3 mục đích khác nhau — đây là điểm quan trọng cần hiểu đúng
khi đọc code:

| Module | Dùng bởi | Vai trò |
|---|---|---|
| `services/agentic_rag.py` (tool `vector_search`/`graph_traverse`) + `retrieve/build_graph.py::GraphRAGRetriever._vector_search/_graph_expand/_rerank` | **`rag_api.py` (production, `/rag/chat`, `/rag/stream`)** | Retrieval nằm trong vòng lặp tool-calling của LLM, xem mục 6 |
| `retrieve/build_graph.py::GraphRAGRetriever.retrieve()` (full, có Tavily tích hợp) | `app.py` (Streamlit, legacy) | Retrieval một lần, không agentic |
| `retrieve/two_stage_search.py::TwoStageRetriever` | `app.py` legacy + evaluation (gián tiếp) | Hybrid + LLM pairwise rerank (0-1) qua `gpt-4o-mini`, kết nối MySQL bằng SQLAlchemy engine riêng (`mysql+mysqlconnector://root:123456789@localhost:3306/law`, **hardcode** không đọc `.env`) |
| `retrieve/retrieval_strategies.py` (5 strategies: `SemanticSearchStrategy`, `HybridSearchStrategy`, `TwoStageSearchStrategy`, `TwoStageSearchV2Strategy`, `HybridRerankStrategy` + `RetrievalEngine`) | **Chỉ** `evaluation/run_evaluation.py` và `retrieve/test_retrieval.py` | Bộ chiến lược dùng để **benchmark offline**, không chạy trong đường dẫn production — xem cảnh báo ở mục 11 và 14 |

### 4.1. Web fallback (`retrieve/tavily_fallback.py`)

- `TRUSTED_DOMAINS` (dict domain → điểm tin cậy 0.85–1.0 cho `thuvienphapluat.vn`, `vbpl.vn`, `moj.gov.vn`,
  `phapdien.moj.gov.vn`, `quochoi.vn`, `chinhphu.vn`, `toaan.gov.vn`, `luatvietnam.vn`, `lawnet.vn`) +
  `UNTRUSTED_PATTERNS` (blog/forum/wiki/mạng xã hội → điểm 0.1, bị loại) + `LEGAL_KEYWORDS` (cộng điểm theo
  mật độ từ khóa pháp lý trong nội dung/tiêu đề).
- `_compute_trust_score` = `0.7×domain_score + 0.3×content_score` → `_classify_trust` chia 3 mức: `high`
  (≥0.7 → nhãn `[Web]`), `medium` (≥0.4 → `[Web - tham khảo]`), `low` (<0.4 → **bị loại hoàn toàn**).
- `is_context_sufficient(candidates, query)`: quyết định có cần fallback Tavily hay không — luôn fallback nếu
  query chứa từ khóa thời sự (`RECENT_KEYWORDS`: "2025", "mới nhất", "hiện nay"...); nếu không, yêu cầu
  điểm trung bình top-3 ≥ `SCORE_THRESHOLD=0.6` **và** ít nhất 1 kết quả overlap ≥2 từ khóa với query.
- `decompose_query(query)`: 1 lệnh gọi LLM (`gpt-4o-mini`, JSON mode qua prompt) tách câu hỏi thành
  `legal_query` / `realtime_query` / `needs_calc` / `calc_hint` — chỉ chạy khi `needs_realtime_data(query)`
  khớp từ khóa (giá, lãi suất, "hôm nay", năm 2024/2025...), nếu không trả thẳng `legal_query=query`.
- `search_tavily()` / `search_tavily_realtime()`: gọi Tavily API (`search_depth="advanced"`/`"basic"`),
  luôn thêm tiền tố `"pháp luật Việt Nam "` cho truy vấn pháp lý.

---

## 5. Pre-retrieve agent layer (`agents/`)

Tầng lọc chạy **trước** khi vào RAG, hiện thực hóa pattern "chain of responsibility" tuần tự, dừng sớm khi có
agent nào match (`AgentsManager.analyze()` trong `agents/agents_manager.py`):

```
QuickAgent.run() → nếu match FAQ → action="quick" (dừng, trả lời ngay)
  ↓ (không match)
SpamAgent.run()  → nếu is_spam → action="spam" (dừng, từ chối)
  ↓ (không match)
SentimentAgent.run() → is_negative & !is_findinfo → action="escalate"
                     → is_findinfo (dù negative hay không) → action="proceed"
  ↓ (fallback)
action="proceed", sentiment="neutral"
```

- **`QuickAgent`**: rule-based, dict tra cứu (`"xin chào"`, `"bạn là ai"`, `"cảm ơn"`...) khớp **chính xác
  hoặc prefix**, chỉ áp dụng khi câu hỏi ≤80 ký tự. Không dùng embedding/similarity (code có comment gợi ý
  nâng cấp bằng cosine similarity nhưng chưa triển khai).
- **`SpamAgent`**: hoàn toàn rule-based hiện nay — bộ phát hiện dựa trên: lặp ký tự dài (`RE_LONG_REPEAT`),
  chuỗi phụ âm vô nghĩa (`RE_CONSONANT_RUN`), tỷ lệ chữ cái thấp, tỷ lệ token "ít nguyên âm" cao, link, dấu
  câu/emoji dày đặc, từ khóa xúc phạm/jailbreak (regex word-boundary, hỗ trợ cách viết lách như `d m`).
  **Một model ML (`visolex/bartpho-spam-binary` qua HuggingFace Transformers) đã được viết nhưng bị comment
  out** (lý do ghi trong code: "Hơi chặt, đánh câu hỏi pháp lý nặng quá" — tức quá nhạy, false-positive với
  câu hỏi pháp luật hợp lệ).
- **`SentimentAgent`**: dùng model zero-shot NLI đa ngôn ngữ `joeddav/xlm-roberta-large-xnli` (không fine-tune
  riêng), 9 nhãn giả thuyết tiếng Việt tùy biến (`neutral_info, satisfied, urgent, sad, dissatisfied, angry,
  grateful_polite, hostile_abusive`), lấy top-k=2 nhãn có entailment ≥ `tau=0.2`. `NEGATIVE_LABELS =
  {sad, dissatisfied, angry, hostile_abusive}`; `is_findinfo` bật khi `neutral_info` nằm trong top-k (ngay cả
  khi đồng thời có nhãn tiêu cực) — nghĩa là câu hỏi vừa bực bội vừa cần tra cứu thông tin vẫn được `proceed`
  thay vì `escalate`.
- Kết quả tổng hợp đóng gói thành `Decision` (dataclass) + `Action` (enum: `PROCEED, QUICK_ANSWER, SPAM,
  ESCALATE`) trong `agents/types.py`, tiêu thụ trực tiếp bởi `rag_api.py`.

---

## 6. Agentic RAG core (`services/agentic_rag.py`)

Thành phần phức tạp và quan trọng nhất hệ thống — vòng lặp **OpenAI function-calling** (Chat Completions API,
`tool_choice="auto"`, `temperature=0.1`, tối đa **6 vòng lặp**, model qua `agent_chat_model()` =
`OPENAI_AGENT_MODEL` hoặc `OPENAI_MODEL` hoặc mặc định `gpt-4o-mini`).

### 6.1. Bốn tools (`RAG_AGENT_TOOLS`)

| Tool | Tham số | Hành vi (`execute_agent_tool`) |
|---|---|---|
| `vector_search` | `query` (bắt buộc), `top_k=10` | Milvus ANN (pool = `top_k × AGENT_VECTOR_POOL_MULT`, mặc định ×3, trần 40) → `fetch_dieu_records()` hydrate **toàn văn** từ Neo4j (Cypher join ChuDe/DeMuc/Chuong/Dieu theo `mapc`) → fallback MySQL (`pymysql`, JOIN `pddieu/pdchuong/pddemuc/pdchude`) cho các `mapc` Neo4j thiếu → `rerank_phapdien_hits()` |
| `graph_traverse` | `mapcs[]` (bắt buộc), `hop=2` | `GraphRAGRetriever._graph_expand()` (Cypher `LIEN_QUAN\|THAM_CHIEU*1..hop`) → rerank tương tự |
| `web_search` | `query` (bắt buộc) | `retrieve/tavily_fallback.search_tavily(query, top_k=5)`, gắn `source="web"`, giữ `trust_level`/`source_label` |
| `get_full_article` | `mapc` (bắt buộc) | `fetch_dieu_single()` (Neo4j → fallback MySQL), `score=1.0` cố định |

### 6.2. Rerank (`rerank_phapdien_hits`)

1. **Mặc định**: `GraphRAGRetriever._rerank()` — base score (ưu tiên thứ tự Milvus/Neo4j gốc) + overlap từ
   khóa (0.05/từ khớp).
2. **Tùy chọn** (`RAG_AGENT_LLM_RERANK=1`): `_apply_llm_rerank_head()` chấm lại **12 kết quả đầu** bằng 1 LLM
   call/passage (`gpt-4o-mini`, trả về số thực 0–1) — tốn kém API tuyến tính theo số passage, chỉ áp dụng cho
   head, giữ nguyên đuôi.

### 6.3. Quản lý ngữ cảnh cho LLM vs UI

- Nội dung gửi vào prompt LLM bị cắt (`truncate_for_llm`, mặc định `AGENT_LLM_MAX_NOIDUNG_CHARS=4000`,
  `AGENT_LLM_MAX_PASSAGE_CHARS=4000`) để tiết kiệm token.
- Nhưng **full hits** (không cắt) được gom vào biến `bucket` (qua `merge_passage_hits`, dedupe theo
  `mapc`/`url`, tối đa 12 phần tử mới mỗi lần gọi tool) — đây chính là nguồn cho panel "Căn cứ pháp lý" ở
  frontend (full `noidung` không bị cắt, khác với những gì LLM thực sự "nhìn thấy").
- Khi hội thoại kết thúc (LLM không gọi tool nữa, hoặc hết 6 vòng): `_dedup_results()` (trong `rag_api.py`,
  web trước, pháp điển sau, trần 20) → `_build_passages()` chuẩn hóa field cho frontend.

### 6.4. Hai entrypoint trùng lặp logic

- `run_agentic_rag_sync()` — hàm blocking, dùng bởi `POST /rag/chat` (non-stream).
- Một **bản sao gần như y hệt** của vòng lặp tool-calling được viết **inline** trong generator
  `event_stream()` của `rag_api.py::rag_stream()` (dùng bởi `POST /rag/stream`, để có thể `yield` SSE events
  giữa chừng). Hai implementation **không dùng chung hàm** — sửa logic ở một nơi (vd thêm tool mới, đổi điều
  kiện dừng) rất dễ quên đồng bộ nơi còn lại. Xem mục 14.

---

## 7. Query understanding (`services/utils.py`)

File này phục vụ **cả hai** luồng (FastAPI hiện hành *và* Streamlit legacy), nhưng chỉ 2 hàm được `rag_api.py`
dùng trực tiếp:

- **`rewrite_query_v2(prompt, history)`**: 1 LLM call (`temperature=0.0`), ép trả về **1 cụm từ khóa ngắn
  gọn** (2–10 từ, không dấu câu, không viết dạng câu hỏi) — tối ưu cho vector search chứ không phải để hiển
  thị cho người dùng. Có ràng buộc an toàn rõ ràng trong prompt: không bịa thông tin, không đoán, giữ nguyên
  prompt gốc nếu câu input chỉ là phản hồi/cảm xúc.
- **`detect_intent(prompt)`**: LLM nhị phân `Có`/`Không` để chặn câu hỏi ngoài phạm vi pháp luật; mặc định
  `Có` khi không chắc hoặc lỗi — thiên về **recall cao** (ưu tiên không từ chối nhầm) hơn precision.

Các hàm còn lại (`rewrite_query`, `rewrite_query_with_history`, `generate_response(_stream)`,
`generate_structured_response(_stream)`, `analyze_complex_situation`, `resolve_sentiment` /
`SENTIMENT_TONE_MAP`, `retrieve_parallel`) **chỉ được gọi bởi `app.py` (Streamlit, legacy)** — đây là bộ máy
"phân tích tình huống phức tạp" (tách nhiều vi phạm trong 1 tình huống, mỗi vi phạm sinh 2 query pháp lý, trả
lời có cấu trúc theo từng vi phạm) **chưa có tương đương trong `rag_api.py`/agentic flow** — `query_mode:
"situation"` ở API mới chỉ đổi system prompt (xem `agent_system_prompt(situation_mode=True)`), không chạy lại
toàn bộ `analyze_complex_situation`.

---

## 8. API layer (`rag_api.py` — FastAPI)

| Endpoint | Method | Mô tả |
|---|---|---|
| `/health` | GET | Health check đơn giản |
| `/rag/chat` | POST | Non-streaming, trả `RagChatResponse` (JSON) |
| `/rag/stream` | POST | SSE streaming (chi tiết bên dưới) |
| `/tts` | POST | Text-to-Speech |
| `/stt` | POST | Speech-to-Text |

**Pydantic models**: `ChatMessage{role, content}`, `RagChatRequest{prompt, history[], query_mode:
"normal"|"situation"}`, `RagChatResponse{answer, action, normalized_query?, sources[], passages[]}`.

### 8.1. Luồng `/rag/chat` & `/rag/stream`

Cả hai cùng chạy: `run_pre_retrieve` → (nếu `quick/spam/escalate`: trả lời tức thời, không RAG) → (nếu
`proceed`): `rewrite_query_v2` → `detect_intent` (chặn nếu "không") → vòng lặp agentic (mục 6). Khác biệt
chính: `/rag/chat` gọi `run_agentic_rag_sync()` và trả JSON một lần; `/rag/stream` chạy vòng lặp **inline**
để phát SSE theo thời gian thực.

### 8.2. SSE event schema (`/rag/stream`)

```
data: {"type":"status", "text":"🔍 Đang gọi vector_search...", "iteration":1, "max":6}
data: {"type":"meta",   "action":"proceed", "normalized_query":"...", "passages":[...], "sources":[...], "iterations":2, "agentic":true}
data: {"type":"token",  "text":"..."}          // lặp lại nhiều lần, phát theo "word chunk" ~32 ký tự
data: {"type":"done"}
data: {"type":"error",  "message":"..."}
```

- `status` phát **mỗi lần** model quyết định gọi tool (trước khi tool chạy xong), kèm số vòng hiện tại.
- `meta` phát **đúng 1 lần**, ngay trước loạt `token` đầu tiên (khi model không còn gọi tool nữa, hoặc đã hết
  6 vòng) — chứa toàn bộ `passages` cuối cùng để frontend hiển thị panel căn cứ pháp lý song song với khi
  stream text bắt đầu.
- `token` cho câu trả lời cuối: nếu lượt cuối model **đã có content** (trả lời luôn kèm tool call cuối — hiếm)
  thì chunk nội dung đó qua `_chunk_stream_text()`; nếu không, mở **một lệnh gọi streaming riêng**
  (`stream=True`, `temperature=0.2`, `max_tokens=1200`) để sinh câu trả lời tổng hợp.
- Nếu có `web_sources`, một `token` cuối cùng chứa markdown "**Nguồn tham khảo từ web:**..." được nối thêm.

### 8.3. `/tts` (text-to-speech)

- `_strip_markdown()` loại bỏ markdown trước khi đọc.
- OpenAI `tts-1` (voice mặc định `nova`, định dạng mp3, streaming response ghép byte).
- Nếu Firebase đã init (`FIREBASE_CRED_PATH` + `FIREBASE_STORAGE_BUCKET` tồn tại khi app khởi động): upload
  lên Storage, `make_public()`, trả `{url}` (JSON) + gọi `services.history.save_tts_url(chat_id, msg_idx,
  url)` để ghi `tts_url` vào đúng message trong MongoDB (xem mục 10 — đây là điểm **ngầm định** kết nối
  backend Python với dữ liệu chat của frontend Next.js, chỉ hoạt động nếu 2 runtime trỏ **cùng
  `MONGODB_URI/DB/COLLECTION`**).
- Nếu không có Firebase: trả thẳng audio bytes dạng `StreamingResponse(audio/mpeg)` (frontend tự tạo Blob URL,
  không lưu được để phát lại mà không gọi lại TTS).

### 8.4. `/stt` (speech-to-text)

OpenAI `whisper-1`, `language="vi"`, nhận file tạm `.webm` từ `UploadFile`.

---

## 9. Frontend (Next.js 16 / React 19)

- **Stack**: Next.js `16.2.3` (App Router), React `19.2.4`, TypeScript, Tailwind CSS v4 (`@tailwindcss/postcss`,
  không cần `tailwind.config`), `react-markdown` + `remark-gfm` + `remark-breaks`, `mongodb` driver trực
  tiếp (không ORM), `lucide-react` icons. `@stream-io/video-react-sdk` có trong `package.json` nhưng **không
  thấy được dùng** ở component nào đã đọc — khả năng là dependency thừa.
- **1 trang duy nhất** (`src/app/page.tsx`, ~850 dòng, toàn bộ UI chat trong 1 client component).
- **API routes** (đều là proxy mỏng tới `RAG_BACKEND_URL`, mặc định `127.0.0.1:8001`):
  - `POST /api/stream` → pipe thẳng SSE từ `/rag/stream` (không transform).
  - `POST /api/rag` → non-stream, forward tới `/rag/chat` (timeout 120s).
  - `GET/POST /api/chats`, `GET/PATCH/DELETE /api/chats/[id]` → đọc/ghi `src/lib/chats.ts`.
  - `POST /api/stt`, `POST /api/tts` → forward `multipart/form-data` / JSON tới `/stt`, `/tts`.
- **`src/lib/chats.ts`**: Mongo nếu có `MONGODB_URI`, ngược lại **in-memory `Map`** (mất dữ liệu khi restart
  server Next.js) — `ChatDoc{_id, title, pinned, messages[], createdAt, updatedAt}`.
- **State management**: tự viết, không dùng Redux/Zustand — module-level store + React 19
  `useSyncExternalStore` (`src/store/chat-store.ts`). Token stream **không** được lưu từng phần vào store;
  `page.tsx` tích lũy `fullText` cục bộ trong lúc đọc SSE, chỉ `patchChat()` **một lần** khi đã có câu trả lời
  đầy đủ (kèm hiệu ứng "gõ dần" `revealAnswerProgressively` thuần client-side, không liên quan tới tốc độ
  mạng thật).
- **Components chính**: `MessageRow` (bong bóng chat + nút TTS 3 trạng thái + nút "N tài liệu"), `PassagePanel`
  + `PassageFullModal` (accordion căn cứ pháp lý, badge Web/Pháp điển theo `source`), `VoiceMessage` /
  `LiveWaveform` (ghi âm, canvas tự vẽ — không dùng `wavesurfer.js` dù có trong dependency).
- **Hooks**: `useTTS` (gọi `/api/tts`, cache `tts_url` qua callback để không gọi lại TTS khi phát lại),
  `useSTT` (MediaRecorder API, đếm thời lượng).
- `next.config.ts` có lỗi nhỏ: vừa `export default nextConfig` vừa `module.exports = {allowedDevOrigins:...}`
  — cách viết hỗn hợp CommonJS/ESM này khiến `allowedDevOrigins` nhiều khả năng **không có hiệu lực** như ý
  định (cần xác minh khi build).
- `frontend/.env.local.example` **được README gốc nhắc tới (`cp .env.local.example .env.local`) nhưng không
  tồn tại** trong repo.

---

## 10. Persistence layer — lịch sử chat

Có **3 cài đặt lưu lịch sử chat khác nhau** trong repo — cần hiểu rõ ai dùng cái nào để tránh nhầm lẫn khi
phát triển tiếp:

1. **`services/history.py`** (Python, 479 dòng) — dùng bởi `app.py` (Streamlit) và bởi `/tts` trong
   `rag_api.py` (chỉ hàm `save_tts_url`). Tầng fallback 3 cấp: **MongoDB** (nếu `MONGODB_URI` có) → **SQLite**
   (`chat_history/conversations.sqlite`) → **JSON file** (`chat_history/conversations.json`).
2. **`services/history_sqlite.py`** (110 dòng) — bản SQLite-only đơn giản hơn, có vẻ là phiên bản cũ hơn/song
   song, không thấy được import ở `rag_api.py` hay `app.py` trong các đoạn đã đọc.
3. **`src/lib/chats.ts`** (Next.js/TypeScript) — Mongo hoặc in-memory `Map`, độc lập hoàn toàn về code với 2
   module Python trên.

**Điểm quan trọng**: (1) và (3) **được thiết kế để chia sẻ cùng một MongoDB collection** (cùng field `_id`
dạng string, cùng cấu trúc `messages[]`, cùng field `tts_url` lồng trong message) — `save_tts_url()` ở Python
update thẳng vào document mà frontend đọc, *miễn là* biến môi trường `MONGODB_URI` / `MONGODB_DB` /
`MONGODB_COLLECTION` **giống nhau** ở cả backend Python (`.env` gốc) và frontend Next.js
(`frontend/.env.local`). Đây là một ràng buộc **ngầm định, không được validate**, không bất biến (nếu một bên
thiếu `MONGODB_URI`, bên đó âm thầm rơi về SQLite/JSON hoặc in-memory, hai nơi lưu trữ **phân kỳ** mà không có
cảnh báo nào).

---

## 11. Evaluation harness

- **`evaluation/data.json`**: bộ dữ liệu đánh giá **thủ công, rất nhỏ (5 câu hỏi)**, mỗi mục có `query,
  gold_docs` (ID số nguyên tham chiếu tới MySQL — **không phải `mapc`**, cần kiểm tra lại cơ chế mapping),
  `gold_response` (câu trả lời mẫu, có cấu trúc chi tiết).
- **`evaluation/metrics/retrieval_metrics.py`**: `precision_at_k`, `recall_at_k`, `hit_at_k`, `mrr_at_k`,
  `ndcg_at_k` — đúng như Proposal mục 7.1 yêu cầu (Recall@K, Hit@K, MRR), **cộng thêm** Precision@K và nDCG@K.
- **`evaluation/metrics/response_metrics.py`**: `rouge_l_score` (rouge-score lib), `bleu_score` (nltk +
  smoothing method4), `semantic_similarity` (cosine similarity qua
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) — khớp Proposal mục 7.2 (ROUGE/BLEU +
  "LLM-based evaluation" tuy ở đây dùng embedding similarity chứ chưa phải LLM-as-judge thật sự).
- **`evaluation/run_evaluation.py`** (`RetrievalEvaluator`): chạy từng **strategy** trong
  `retrieve/retrieval_strategies.py` (5 chiến lược: `semantic_search`, `hybrid_search`, `two_stage_search`,
  `two_stage_search_v2`/`hybrid_rerank`...) × nhiều giá trị `top_k`, tính trung bình metric trên toàn bộ
  `data.json`, in bảng so sánh + tìm "strategy tốt nhất theo từng metric", xuất `results/*.json` (đã có sẵn 5
  file kết quả cũ trong repo) và biểu đồ qua `evaluation/visualization.py`.
- ⚠️ **Khoảng trống quan trọng**: harness này đánh giá các **strategy trong `retrieval_strategies.py`**
  (module benchmark độc lập, dùng `TwoStageRetriever`/Milvus/MySQL trực tiếp) — **không phải** pipeline
  agentic thực tế (`services/agentic_rag.py` với 4 tools + pre-retrieve agents) mà người dùng cuối thực sự
  trải nghiệm qua `rag_api.py`. Mọi con số Recall@K/MRR hiện có **không phản ánh** chất lượng hệ thống production.

---

## 12. Legacy Streamlit app (`app.py`)

- Giao diện Streamlit độc lập (`streamlit run app.py`, cổng `8501`), không qua FastAPI.
- Dùng trực tiếp `GraphRAGRetriever.retrieve()` (1 lần truy xuất, có Tavily fallback tích hợp sẵn trong hàm —
  **không phải** vòng lặp agentic nhiều tool như `rag_api.py`).
- Có tính năng **`analyze_complex_situation`** (phân tích tình huống nhiều vi phạm/nhiều chủ thể, sinh nhiều
  query con, trả lời có cấu trúc theo từng vi phạm) — tính năng này **chưa được port sang** luồng FastAPI mới
  (chế độ `query_mode="situation"` ở API mới chỉ đổi system prompt, đơn giản hơn nhiều).
- Lưu lịch sử qua `services/history.py` (độc lập với MongoDB của frontend Next.js, trừ khi cấu hình trùng
  `MONGODB_URI` như đã nói ở mục 10).
- README mô tả đây là "luồng cũ, tùy chọn" — về mặt kiến trúc, `app.py` và cặp `Next.js + rag_api.py` là
  **hai ứng dụng gần như tách biệt**, chỉ dùng chung tầng data (MySQL/Milvus/Neo4j) và một phần code
  (`services/utils.py`, `services/history.py`, `retrieve/build_graph.py`).

---

## 13. Hạ tầng / triển khai

- **2 Docker Compose riêng biệt** (không có compose tổng cho toàn hệ thống):
  - `law-crawler/docker-compose.yml`: MySQL 8.0 (`3306`) + phpMyAdmin (`8081`).
  - `test_raptor/docker-compose.yml`: `etcd` (Milvus dependency) + `minio` + `milvus-standalone` (v2.4.14,
    cổng `19530`/`9091`) + `neo4j:5` (cổng `7474` browser / `7687` bolt, `NEO4J_AUTH=neo4j/password123`).
- **Code ứng dụng** (FastAPI, Next.js, Streamlit) chạy **trực tiếp trên host** (venv Python + `npm run dev`),
  **không containerize** — không có `Dockerfile` cho `rag_api.py` hay `frontend/` trong repo.
- **Biến môi trường** (tổng hợp từ toàn bộ code đã đọc, không chỉ README):

  | Biến | Nơi dùng | Mặc định |
  |---|---|---|
  | `OPENAI_API_KEY` | mọi lệnh gọi OpenAI | — (bắt buộc) |
  | `OPENAI_MODEL`, `OPENAI_AGENT_MODEL` | chat/generate, agent loop | `gpt-4o-mini` |
  | `AGENT_LLM_MAX_NOIDUNG_CHARS`, `AGENT_LLM_MAX_PASSAGE_CHARS` | cắt context gửi LLM | `4000` |
  | `AGENT_VECTOR_POOL_MULT` | pool size trước rerank | `3` |
  | `RAG_AGENT_LLM_RERANK` | bật LLM rerank top-12 | `0` |
  | `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE` | fallback MySQL trong `agentic_rag.py` | `localhost/3306/root/123456789/law` |
  | `TAVILY_API_KEY` | web search | — (tắt tool nếu thiếu) |
  | `MONGODB_URI/DB/COLLECTION` | lịch sử chat (Python *và* Next.js — **phải khớp nhau**) | tắt → fallback |
  | `FIREBASE_CRED_PATH`, `FIREBASE_STORAGE_BUCKET` | upload TTS public URL | tắt → stream mp3 trực tiếp |
  | `RAG_BACKEND_URL` (frontend) | mọi route `/api/*` proxy | `http://127.0.0.1:8001` |
  | `ES_ENDPOINT`, `ES_API_KEY` | `services/utils.get_es()`, `data_processing/ingest.py` | module Elasticsearch vẫn tồn tại nhưng **ngoài luồng chính** |

  Lưu ý: `retrieve/two_stage_search.py`, `retrieve/build_graph.py` (hằng số `MYSQL_CONFIG`/`NEO4J_URI` ở đầu
  file), `data_processing/seed_data_batch.py` **hardcode** host/user/password thay vì đọc từ `.env` — chạy
  đúng miễn là dùng giá trị mặc định Docker Compose ở trên, nhưng sẽ gãy âm thầm nếu đổi mật khẩu/host.

---

## 14. Khoảng trống kiến trúc & nợ kỹ thuật

Tổng hợp để dùng trực tiếp cho phần "hạn chế / hướng phát triển" của báo cáo đồ án, xem kế hoạch khắc phục cụ
thể ở `docs/DEVELOPMENT_PLAN.md`:

1. **Trùng lặp logic agent loop** giữa `services/agentic_rag.run_agentic_rag_sync()` và vòng lặp inline trong
   `rag_api.py::rag_stream()` — rủi ro lệch hành vi stream vs non-stream theo thời gian.
2. **Embedding hiện chỉ dựa trên `tendieu`** (tên điều), không embed `noidung` đầy đủ — khác định hướng
   "chunking tài liệu dài" của Proposal; có thể bỏ lỡ các điều luật mà tên không phản ánh đúng nội dung.
3. **Evaluation harness đo sai đối tượng**: benchmark chạy trên `retrieve/retrieval_strategies.py`, không
   phải trên pipeline agentic thực tế người dùng trải nghiệm.
4. **Bộ dữ liệu đánh giá rất nhỏ** (5 câu hỏi) — không đủ ý nghĩa thống kê cho báo cáo học thuật.
5. **`insert_thamchieu()` giới hạn `LIMIT 5`** khi parse quan hệ tham chiếu từ `noidung` — cần xác nhận đây
   là code debug hay đã chạy full trên dữ liệu thật (ảnh hưởng trực tiếp độ phủ của `graph_traverse`).
6. **2 kho lưu lịch sử chat** (Python `services/history*.py` vs Next.js `lib/chats.ts`) chỉ đồng bộ **ngầm
   định** qua biến môi trường Mongo trùng nhau, không có validation/contract rõ ràng.
7. **SpamAgent** hiện thuần rule-based (model ML bị tắt vì quá nhạy) — thiếu dữ liệu nhãn để fine-tune/hiệu
   chỉnh ngưỡng đúng cách.
8. **Pipeline Elasticsearch** (`data_processing/ingest.py`, `services/utils.get_es`) và
   `@stream-io/video-react-sdk` (frontend) là mã dường như không dùng tới trong luồng hiện hành — nên xác
   nhận rồi dọn dẹp hoặc tài liệu hóa lý do giữ lại.
9. **Thiếu `frontend/.env.local.example`** dù README hướng dẫn copy từ file này.
10. **Hardcode credentials/host** ở nhiều module truy xuất (mục 13) thay vì đọc `.env` nhất quán.
11. **Không có test tự động / CI**: các file `*_test*.py` (`retrieve/test_retrieval.py`,
    `data_processing/test_search.py`, `testz/test_connect_elasticsearch.py`) là script chạy tay, không phải
    unit test (`pytest`/`unittest`); không thấy GitHub Actions hay pipeline CI nào trong repo.
12. **Tính năng "phân tích tình huống phức tạp"** (`analyze_complex_situation`) chưa được port từ `app.py`
    sang luồng FastAPI/agentic — `query_mode="situation"` ở API mới đơn giản hơn nhiều.
13. **Không container hóa** code ứng dụng (chỉ data stores được Dockerize) — khó tái lập môi trường nhất quán
    giữa các thành viên nhóm/máy chấm điểm.
