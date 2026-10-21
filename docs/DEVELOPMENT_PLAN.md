# Kế hoạch phát triển — theo định hướng `Proposal.docx`

> Kế hoạch này giả định bạn đọc đã xem `docs/ARCHITECTURE.md` (phân tích chi tiết hiện trạng) và
> `docs/PROPOSAL_SUMMARY.md` (tóm tắt đề tài gốc). Mục tiêu của tài liệu này: biến hệ thống hiện có (đã vượt
> quy mô baseline của proposal) thành một đồ án **đáp ứng đúng trọng tâm nghiên cứu** mà proposal yêu cầu —
> chunking/Top-K có hệ thống, đánh giá retrieval + generation nghiêm túc, phân tích định lượng — mà không phá
> vỡ phần hệ thống production đã chạy được.

## 0. Nguyên tắc tiếp cận

1. **Không viết lại từ đầu.** Hệ thống hiện tại (agentic RAG + graph RAG + web fallback + UI đầy đủ) là tài
   sản lớn, vượt yêu cầu baseline của proposal. Chiến lược là **lấp khoảng trống đánh giá/thực nghiệm**, không
   thay thế kiến trúc.
2. **Tách biệt "nghiên cứu" và "sản phẩm".** Các thí nghiệm chunking/embedding/ablation chạy trên một pipeline
   đánh giá độc lập (mở rộng `evaluation/`), không đụng vào đường dẫn production (`rag_api.py` /
   `services/agentic_rag.py`) cho tới khi có kết luận rõ ràng nên áp dụng thay đổi nào.
3. **Mọi kết luận phải đo được.** Mỗi thay đổi (chunk size, embedding model, Top-K, bật/tắt graph_traverse,
   bật/tắt LLM rerank...) phải có số liệu trước/sau trên cùng bộ test set để đưa vào báo cáo.
4. **Ưu tiên theo đúng thứ tự Objectives của Proposal** (mục 3): corpus → chunking → embedding/retrieval →
   generation → evaluation — rồi mới tới phần "mở rộng" (graph, agentic, web fallback) vốn đã có sẵn.

---

## 1. Bảng ánh xạ: Proposal yêu cầu gì — hệ thống hiện có gì — còn thiếu gì

| Hạng mục Proposal | Hiện trạng trong repo | Việc cần làm |
|---|---|---|
| 3. Pipeline RAG | Có, và phức tạp hơn yêu cầu (agentic, 4 tools) | Giữ nguyên; bổ sung tài liệu hóa + đánh giá đúng pipeline này |
| 3. Corpus pháp luật | Có — Bộ Pháp Điển qua MySQL (66k+ điều), phong phú hơn dataset HuggingFace đề xuất | Không cần đổi corpus; cân nhắc dùng `YuITC/Vietnamese-Legal-Documents`/`Tamir39/vietnam-tax-qa` **chỉ** để kiểm tra khả năng tổng quát hóa (tùy chọn, Phase 5) |
| 5.1 Preprocessing | Crawl + parse HTML (`law-crawler/`), chưa có bước chuẩn hóa text riêng (loại ký tự thừa, chuẩn hóa unicode...) tách bạch | Thêm module preprocessing tường minh, có thể kiểm thử độc lập |
| 5.2 Chunking | **Không có chunking nội dung** — embedding hiện tại chỉ dựa trên `tendieu` (tên điều), xem `data_processing/seed_data_batch.py:192` | **Trọng tâm Phase 2**: cài đặt chunking có cấu hình (chunk size, overlap) trên `noidung`, so sánh với cách embed hiện tại |
| 5.3 Retrieval (embedding + Top-K) | Có (Milvus HNSW/COSINE) nhưng Top-K không được thí nghiệm có hệ thống; rerank là keyword-overlap, không phải embedding similarity thuần | Phase 2–3: ablation Top-K, so sánh embedding model, tách riêng đóng góp của rerank/graph/web |
| 5.4 Generation | Có (OpenAI qua tool-calling + system prompt) | Giữ nguyên; chuẩn hóa prompt để dễ đánh giá A/B |
| 7.1 Retrieval evaluation (Recall@K, Hit@K, MRR) | Code đã có (`evaluation/metrics/retrieval_metrics.py`) nhưng **chạy sai pipeline** (benchmark `retrieval_strategies.py`, không phải `agentic_rag.py` thật) và **dataset chỉ 5 câu** | Phase 1 (mở rộng dataset) + Phase 3 (đánh giá đúng pipeline production) |
| 7.2 Answer evaluation (EM, ROUGE/BLEU, LLM-eval, human eval) | ROUGE/BLEU/semantic similarity đã có code; chưa có Exact Match, chưa có LLM-as-judge theo rubric, chưa có human eval | Phase 4 |
| 8. Phân tích ảnh hưởng chunking/Top-K | Chưa có | Phase 5 — sản phẩm đầu ra chính là báo cáo phân tích |
| Môi trường Colab/Kaggle, FAISS | Hệ thống hiện dùng Docker (Milvus/Neo4j/MySQL) trên máy cục bộ | Không cần đổi hạ tầng chính; có thể làm thêm 1 notebook minh họa "lite pipeline" bằng FAISS cho mục đích trình bày/giảng viên dễ chạy thử (tùy chọn) |

---

## 2. Lộ trình theo giai đoạn (Phase)

Giả định nhóm có thể chạy **song song nhiều phase phụ** giữa các thành viên khác nhau sau khi Phase 0 hoàn
tất. Thời lượng ước tính tính theo tuần làm việc bán thời gian (phù hợp đồ án môn học), có thể co giãn.

### Phase 0 — Chuẩn hóa môi trường & tái lập hệ thống (0.5–1 tuần)

**Mục tiêu**: mọi thành viên nhóm chạy được full stack từ repo này (`LLM_Prj1`), không phụ thuộc máy của
người tạo repo gốc.

- [ ] Dựng `law-crawler/docker-compose.yml` (MySQL) + `infra/docker-compose.yml` (Milvus/Neo4j), chạy
      `law-crawler/main.py` → `data_processing/seed_data_batch.py` → `retrieve/build_graph.py` theo đúng thứ
      tự trong README.
- [ ] Viết file `.env.example` ở root (hiện chưa có) tổng hợp **toàn bộ** biến ở `ARCHITECTURE.md` §13.
- [ ] Viết `frontend/.env.local.example` (đang thiếu — README tham chiếu nhưng không tồn tại).
- [ ] Sửa các chỗ hardcode credentials (`retrieve/two_stage_search.py`, `retrieve/build_graph.py`,
      `data_processing/seed_data_batch.py`) để đọc từ `.env` thống nhất — tránh "chạy được trên máy A nhưng
      không chạy trên máy B".
- [ ] Xác nhận lại `insert_thamchieu()` trong `retrieve/build_graph.py` (hiện giới hạn `LIMIT 5`, nhiều khả
      năng là code debug) — chạy full và đo số lượng `THAM_CHIEU` edges thực tế tạo ra.
- **Deliverable**: checklist "Getting Started" đã test trên ≥2 máy khác nhau trong nhóm; `.env.example` +
  `frontend/.env.local.example` được commit.
- **Acceptance**: `uvicorn rag_api:app` + `npm run dev` chạy được, trả lời được 1 câu hỏi end-to-end, có
  passages hiển thị.

### Phase 1 — Mở rộng bộ dữ liệu đánh giá (1–1.5 tuần)

**Mục tiêu**: có bộ test set đủ lớn và đáng tin cậy để các phase sau cho ra số liệu có ý nghĩa thống kê.

- [ ] Mở rộng `evaluation/data.json` từ 5 → **tối thiểu 50–100 câu hỏi** có gán nhãn `gold_docs` (mapc hoặc
      ID nhất quán — làm rõ và thống nhất lại việc `gold_docs` hiện là ID MySQL số nguyên, cần map sang
      `mapc` để khớp với output thật của `vector_search`/`graph_traverse`) và `gold_response`.
  - Lấy câu hỏi từ nhiều nhóm luật khác nhau (dân sự, hình sự, hành chính, lao động, giao thông...) để tránh
    thiên lệch.
  - Phân loại câu hỏi theo **độ khó truy xuất**: trực tiếp (từ khóa trùng tên điều), gián tiếp (cần suy luận
    ngữ nghĩa), đa bước (cần `graph_traverse` qua `LIEN_QUAN`/`THAM_CHIEU`), ngoài phạm vi Pháp Điển (cần
    `web_search`) — để sau này tách riêng được đóng góp của từng tool.
- [ ] (Tùy chọn, bám sát Proposal mục 4) Thử tải `YuITC/Vietnamese-Legal-Documents` hoặc
      `Tamir39/vietnam-tax-qa` từ HuggingFace, chạy thử một tập nhỏ qua pipeline hiện tại để kiểm tra khả năng
      tổng quát hóa ngoài phạm vi Bộ Pháp Điển — ghi nhận riêng, không trộn vào test set chính.
- [ ] Viết script validate schema của `evaluation/data.json` (tránh lỗi nhập liệu khi nhiều người cùng thêm
      câu hỏi).
- **Deliverable**: `evaluation/data.json` v2 (≥50 câu, có trường `difficulty`/`category`), script validate.
- **Acceptance**: ≥2 thành viên review chéo nhãn gold trước khi dùng cho Phase 3–4.

### Phase 2 — Thí nghiệm Chunking & Embedding (1.5–2 tuần)

**Mục tiêu**: trả lời trực tiếp câu hỏi trọng tâm của Proposal mục 5.2/8: chunk size/overlap và lựa chọn
embedding ảnh hưởng thế nào tới retrieval.

- [ ] Viết module ingestion **thử nghiệm**, tách biệt với `data_processing/seed_data_batch.py` production
      (ví dụ `experiments/chunking/`), hỗ trợ:
  - Nhiều chiến lược chunk trên `noidung`: không chunk (hiện tại — chỉ `tendieu`), chunk theo câu/đoạn với
    `chunk_size ∈ {200, 400, 800} token`, `overlap ∈ {0, 20%, 50%}`.
  - Ghi vào Milvus collection **riêng** theo từng cấu hình (`phapdien_chunk_{size}_{overlap}`) để không ảnh
    hưởng collection production `phapdien_simple_tendieu`.
- [ ] So sánh ≥2 embedding model: `text-embedding-3-small` (đang dùng) vs. 1 model đa ngôn ngữ mã nguồn mở
      (vd `intfloat/multilingual-e5-large` hoặc `BAAI/bge-m3`) để vừa khớp định hướng "model vừa/nhỏ" của
      Proposal mục 9, vừa test khả năng giảm chi phí API.
- [ ] Chạy `evaluation/metrics/retrieval_metrics.py` (Recall@K, Hit@K, MRR, nDCG tại K ∈ {1,3,5,10}) trên từng
      cấu hình × bộ test set Phase 1.
- **Deliverable**: bảng so sánh chunk size × overlap × embedding model × Recall/Hit/MRR; khuyến nghị cấu hình
  tốt nhất kèm phân tích đánh đổi (độ chính xác vs chi phí embedding vs độ trễ truy vấn).
- **Acceptance**: có ≥1 cấu hình chunking cho Recall@5 cao hơn baseline hiện tại (`tendieu`-only) một cách có
  ý nghĩa thống kê (hoặc kết luận ngược lại kèm giải thích, vẫn là kết quả hợp lệ cho báo cáo).

### Phase 3 — Đánh giá đúng pipeline production (1–1.5 tuần)

**Mục tiêu**: sửa khoảng trống nêu ở `ARCHITECTURE.md` §11/§14.3 — evaluation phải đo **pipeline thật**
(`services/agentic_rag.py`), không chỉ `retrieve/retrieval_strategies.py`.

- [ ] Viết adapter trong `evaluation/` gọi trực tiếp `execute_agent_tool("vector_search", ...)` và
      `run_agentic_rag_sync()` (hoặc một harness gọi thẳng `/rag/chat` qua HTTP để test cả tầng API) thay vì
      `retrieval_strategies.py`.
- [ ] Đo retrieval metrics **tách theo từng tool**: chỉ `vector_search`, `vector_search + graph_traverse`,
      `+ web_search` — để định lượng đúng câu hỏi "graph expansion và web fallback đóng góp bao nhiêu" (liên
      quan trực tiếp tới Proposal mục 8 "phân tích ảnh hưởng retrieval tới chất lượng").
- [ ] Đo ảnh hưởng của `RAG_AGENT_LLM_RERANK=0` vs `1` (có/không LLM rerank) lên Recall@K và chi phí (số LLM
      call, latency).
- [ ] Đo độ chính xác của `is_context_sufficient()`/Tavily fallback trigger: trong bao nhiêu % câu hỏi test
      set, fallback được kích hoạt đúng lúc cần (câu hỏi ngoài phạm vi Pháp Điển) và sai lúc không cần.
- **Deliverable**: báo cáo so sánh retrieval-only vs retrieval+graph vs retrieval+graph+web, có/không LLM
  rerank — trên **cùng** bộ test set Phase 1, dùng config chunking/embedding tốt nhất chọn ở Phase 2.
- **Acceptance**: số liệu Recall@K/Hit@K/MRR của pipeline thật được công bố trong báo cáo thay thế hoàn toàn
  số liệu cũ đo trên `retrieval_strategies.py`.

### Phase 4 — Đánh giá chất lượng câu trả lời (1–1.5 tuần)

**Mục tiêu**: hoàn thiện đúng Proposal mục 7.2 (EM, ROUGE/BLEU, LLM-eval, human eval, 4 khía cạnh
Correctness/Relevance/Faithfulness/Completeness).

- [ ] Thêm Exact Match cho các câu hỏi có đáp án dạng rõ ràng (mức phạt cụ thể, thời hạn cụ thể...) trong bộ
      test set Phase 1 (gắn cờ `exact_match_applicable: true/false` khi gán nhãn).
- [ ] Tận dụng `evaluation/metrics/response_metrics.py` (ROUGE-L, BLEU, semantic similarity) — đã có code,
      chỉ cần chạy trên output thật của pipeline (Phase 3) thay vì không chạy.
- [ ] Bổ sung **LLM-as-judge** thực sự (hiện tại `semantic_similarity` dùng embedding, chưa phải judge bằng
      LLM): viết prompt rubric 4 tiêu chí (Correctness, Relevance, Faithfulness/groundedness, Completeness),
      thang điểm 1–5, model chấm độc lập với model sinh câu trả lời (tránh tự chấm cho chính mình) — có thể
      tái dùng pattern chấm điểm 0–1 đã có ở `services/agentic_rag._llm_score_passage` làm tham khảo cách gọi.
- [ ] Human evaluation trên mẫu nhỏ (vd 15–20 câu), ≥2 thành viên chấm độc lập, tính độ đồng thuận (vd
      Cohen's kappa hoặc đơn giản là % khớp) để tăng độ tin cậy khi so với LLM-judge.
- **Deliverable**: bảng điểm theo 4 khía cạnh × từng cấu hình pipeline (so sánh baseline vs cấu hình tốt nhất
  từ Phase 2–3); phân tích tương quan giữa LLM-judge và human eval.
- **Acceptance**: mỗi câu trả lời trong bộ test set có đủ: ROUGE-L, BLEU, semantic similarity, LLM-judge
  4 điểm, và (với mẫu con) điểm human eval.

### Phase 5 — Phân tích tổng hợp & báo cáo (1 tuần)

- [ ] Tổng hợp toàn bộ kết quả Phase 2–4 thành báo cáo phân tích: ảnh hưởng chunk size, Top-K, embedding
      model, graph expansion, web fallback, LLM rerank — đúng cấu trúc Proposal mục 8/10.
- [ ] Dùng `evaluation/visualization.py` (đã có) để vẽ biểu đồ so sánh; mở rộng nếu thiếu loại biểu đồ cần
      (vd heatmap chunk_size × overlap, bar chart theo category câu hỏi).
- [ ] Viết phần "Hạn chế & hướng phát triển" dựa trực tiếp trên `ARCHITECTURE.md` §14 (nợ kỹ thuật) — biến
      điểm yếu kiến trúc thành nội dung học thuật hợp lệ (vd: "hệ thống hiện tại ưu tiên recall qua agentic
      tool-calling thay vì 1 lần retrieve cố định, đánh đổi với độ trễ và chi phí — phân tích trade-off này").
- **Deliverable**: báo cáo cuối (docx/pdf hoặc slide) + bộ biểu đồ + bảng số liệu đầy đủ, sẵn sàng nộp.

### Phase 6 (tùy chọn, nếu còn thời gian) — Hardening kỹ thuật

Không bắt buộc cho đồ án học thuật nhưng nên làm nếu nhóm muốn dùng sản phẩm thật hoặc tiếp tục phát triển:

- [ ] Gộp 2 bản sao logic agent loop (`run_agentic_rag_sync` vs inline trong `rag_api.rag_stream`) thành 1
      implementation dùng chung (ví dụ: `run_agentic_rag_sync` nhận thêm callback `on_event` để vừa chạy
      sync vừa phát sự kiện stream).
- [ ] Thống nhất 1 cơ chế lưu lịch sử chat duy nhất (chọn Mongo làm nguồn sự thật, bỏ SQLite/JSON fallback
      hoặc tài liệu hóa rõ khi nào dùng) thay vì 2 stack độc lập.
- [ ] Viết test tự động tối thiểu: unit test cho `agents/` (quick/spam/sentiment rule-based, dễ test nhất vì
      không gọi API ngoài), integration test cho `/health` và `/rag/chat` (mock OpenAI/Milvus/Neo4j).
- [ ] Dockerize `rag_api.py` + `frontend/` để nhóm/giảng viên chấm bài chạy bằng `docker compose up` một lệnh
      duy nhất, gộp chung với 2 compose file dữ liệu hiện có.
- [ ] Dọn dẹp code đường chết: xác nhận rồi xóa hoặc tài liệu hóa `data_processing/ingest.py`
      (Elasticsearch), `@stream-io/video-react-sdk` (frontend), `services/history_sqlite.py` nếu xác nhận
      không còn dùng.

---

## 3. Timeline tổng hợp

| Tuần | Phase | Trọng tâm |
|---|---|---|
| 1 | Phase 0 | Tái lập môi trường, chuẩn hóa `.env`, xác nhận `build_graph.py` |
| 2 | Phase 1 | Mở rộng bộ test set lên ≥50 câu, gán nhãn, review chéo |
| 3–4 | Phase 2 | Thí nghiệm chunking + embedding model trên collection Milvus riêng |
| 5 | Phase 3 | Đánh giá đúng pipeline agentic thật (tool-by-tool, rerank on/off) |
| 6 | Phase 4 | Answer evaluation: EM, ROUGE/BLEU, LLM-judge, human eval |
| 7 | Phase 5 | Phân tích tổng hợp, viết báo cáo, biểu đồ |
| 8 (nếu còn thời gian) | Phase 6 | Hardening kỹ thuật (tùy chọn) |

So với timeline gốc 5 tuần trong Proposal, kế hoạch này dài hơn (7–8 tuần) vì hệ thống đã có quy mô lớn hơn
nhiều baseline Colab/Kaggle — phần "làm pipeline RAG" (tuần 1–2 trong proposal gốc) **đã xong**, thời gian
dôi ra được dồn vào đúng phần proposal coi là trọng tâm nghiên cứu: thực nghiệm (Phase 2) và đánh giá (Phase
3–4).

## 4. Gợi ý phân công (nhóm nhiều thành viên)

Có thể chạy song song sau khi Phase 0 xong:

- **Nhánh Data/Retrieval**: Phase 1 (gán nhãn) → Phase 2 (chunking/embedding) → đóng góp input cho Phase 3.
- **Nhánh Evaluation/Generation**: xây adapter Phase 3 song song với Phase 2 (không phụ thuộc kết quả chunking
  để bắt đầu viết code), rồi chạy lại khi Phase 2 có cấu hình tốt nhất; tiếp tục Phase 4.
- **Nhánh Báo cáo/Phân tích**: bắt đầu khung báo cáo từ tuần 3 (dùng `ARCHITECTURE.md` làm phần "Hệ thống đề
  xuất"), điền số liệu dần khi Phase 2–4 ra kết quả, hoàn thiện ở Phase 5.
- **Nhánh Hardening (Phase 6)**: chỉ cần 1 người, làm song song nếu nhóm dư nhân lực, không chặn đường găng.

## 5. Rủi ro & giảm thiểu

| Rủi ro | Ảnh hưởng | Giảm thiểu |
|---|---|---|
| Chi phí OpenAI API tăng do nhiều thí nghiệm (embedding lại toàn bộ corpus nhiều lần, LLM rerank, LLM-judge) | Vượt ngân sách/API quota | Dùng tập con đại diện (vd 5-10k điều ngẫu nhiên) cho các thí nghiệm chunking thăm dò ban đầu, chỉ chạy full corpus cho cấu hình đã chọn |
| Gán nhãn `gold_docs` không nhất quán giữa người gán | Số liệu Recall/MRR không đáng tin | Hướng dẫn gán nhãn rõ ràng + review chéo bắt buộc (Phase 1) |
| `insert_thamchieu()` chạy giới hạn (`LIMIT 5`) chưa được xác nhận → graph nghèo quan hệ | Đánh giá `graph_traverse` sai thấp so với tiềm năng thật | Việc đầu tiên trong Phase 0: chạy lại full, đo số edge, nếu cần sửa code trước khi vào Phase 2–3 |
| LLM-judge thiên vị (cùng họ model với model sinh câu trả lời) | Điểm LLM-judge không khách quan | Dùng model khác (hoặc ít nhất prompt độc lập, nhiệt độ thấp) cho vai trò judge; đối chiếu với human eval |
| Thời gian Phase 2 kéo dài do re-index Milvus nhiều lần | Trễ tiến độ | Song song hóa: chạy nhiều collection thử nghiệm cùng lúc nếu tài nguyên cho phép, hoặc giảm số cấu hình xuống 3-4 thay vì full grid search |

## 6. Tiêu chí hoàn thành đồ án (Definition of Done)

- [ ] `docs/ARCHITECTURE.md` + `docs/DEVELOPMENT_PLAN.md` (tài liệu này) được review và cập nhật theo tiến độ
      thực tế (đánh dấu các mục đã hoàn thành).
- [ ] Bộ test set ≥50 câu hỏi, có review chéo, lưu tại `evaluation/data.json` (hoặc file mới nếu tách).
- [ ] Bảng kết quả Recall@K/Hit@K/MRR/nDCG cho ≥3 cấu hình chunking/embedding khác nhau.
- [ ] Bảng kết quả retrieval tách theo tool (vector-only / +graph / +web) trên pipeline production thật.
- [ ] Bảng kết quả answer evaluation đầy đủ 4 khía cạnh (Correctness/Relevance/Faithfulness/Completeness) +
      ROUGE/BLEU + (mẫu con) human eval.
- [ ] Báo cáo phân tích cuối cùng nêu rõ: cấu hình được chọn và vì sao, trade-off đã cân nhắc, hạn chế còn
      tồn tại (tham chiếu `ARCHITECTURE.md` §14), hướng phát triển tiếp theo (Phase 6 nếu chưa làm).
