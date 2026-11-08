"""Nội dung giải thích các chỉ số đánh giá — dùng chung cho run_evaluation.py
(in ra console) và visualization.py (hiển thị trong dashboard), để không lặp lại
2 nơi và luôn khớp với công thức thật trong metrics/*.py.
"""

KNOWN_LIMITATION_GOLD_DOCS = """
### ⚠️ Hạn chế đã biết: `gold_docs` dùng OFFSET dòng MySQL, không phải `mapc`

`evaluation/data.json` lưu `gold_docs` là **vị trí dòng (OFFSET)** trong bảng `pddieu`
tại thời điểm soạn bộ test, không phải mã điều luật (`mapc`) trực tiếp. Hàm
`offsets_to_mapcs()` quy đổi offset → mapc bằng `SELECT mapc FROM pddieu LIMIT 1
OFFSET N` — giả định thứ tự dòng trong bảng **không đổi** so với lúc soạn test.

**Hệ quả:** nếu crawl lại dữ liệu từ đầu (`law-crawler/main.py`), thứ tự insert các
dòng có thể khác lần trước (ví dụ thứ tự đọc file `demuc/*.html` qua `os.listdir()`
không đảm bảo ổn định) → offset trỏ sang **điều luật khác hẳn** → Precision/Recall/
Hit/MRR/NDCG@k có thể tụt về **0.0** dù hệ thống retrieval thực tế vẫn trả lời đúng
(kiểm chứng qua ROUGE-L/BLEU/Semantic Similarity/grounding vẫn ở mức tốt).

**Cách khắc phục triệt để** (chưa làm, cần người am hiểu pháp luật xác minh): sửa
`data.json` để `gold_docs` lưu trực tiếp `mapc` thay vì offset, bỏ hẳn
`offsets_to_mapcs()`. Trong lúc chưa sửa, **ưu tiên đọc response metrics +
grounding score** khi retrieval metrics về 0 bất thường ở mọi k.
"""

TOPK_GUIDE = """
### Top-K nghĩa là gì?

`retrieved_mapcs` là danh sách điều luật hệ thống trả về, lấy từ event `meta` của
`run_agentic_rag()` — tức là **thứ tự cuối cùng sau toàn bộ pipeline thật**
(vector_search → graph_traverse → hydrate Neo4j → **rerank hybrid**), không phải
top-k thô từ Milvus.

"Top-K" khi đánh giá = cắt danh sách đó ở các độ sâu khác nhau (mặc định
**k = 3, 5, 10**) rồi tính metric riêng cho từng k.

**Vì sao test nhiều k thay vì 1 con số?** Để tách biệt 2 loại lỗi khác nhau:
- **Recall@10 cao nhưng Recall@3 thấp** → hệ thống *có* tìm ra điều luật đúng,
  nhưng **rerank xếp hạng chưa tốt** (đúng mà bị đẩy xuống dưới) → cần sửa tầng rerank.
- **Recall@10 vẫn thấp** → điều luật đúng **không nằm trong tập ứng viên** ngay từ
  đầu → lỗi nằm ở embedding/retrieval tầng dưới (vector_search), rerank không cứu được.
"""

RETRIEVAL_METRIC_GUIDE = """
### Retrieval metrics (so khớp theo `mapc` — mã điều luật)

| Metric | Công thức | Đo cái gì |
|---|---|---|
| **Precision@k** | (số điều đúng trong top-k) / k | Trong k điều trả về, bao nhiêu % **chính xác** — phạt khi trả về rác/không liên quan |
| **Recall@k** | (số điều đúng trong top-k) / (tổng số điều đúng cần tìm) | Trong tất cả điều luật đúng cần có, tìm được bao nhiêu % — phạt khi **bỏ sót** |
| **Hit@k** | 1 nếu top-k có ≥1 điều đúng, ngược lại 0 | Nhị phân: câu hỏi này có "trúng" được gì không — trung bình trên cả tập = % câu hỏi tìm đúng ≥1 nguồn |
| **MRR@k** | 1/rank của điều đúng **đầu tiên** xuất hiện | Điều đúng xuất hiện **sớm cỡ nào** (rank 1 → 1.0, rank 3 → 0.33) |
| **NDCG@k** | Σ 1/log2(rank+1) cho mỗi điều đúng, chia cho giá trị lý tưởng | Chất lượng **xếp hạng tổng thể**, có trọng số — vị trí cao đóng góp nhiều hơn vị trí thấp |

Tất cả được tính trung bình trên toàn bộ tập câu hỏi.
"""

RESPONSE_METRIC_GUIDE = """
### Response metrics (so câu trả lời sinh ra vs `gold_response`)

| Metric | Cách đo | Lưu ý |
|---|---|---|
| **ROUGE-L** | Longest common subsequence giữa câu trả lời và đáp án mẫu | Nhạy với cách diễn đạt — câu đúng ý nhưng viết khác hoàn toàn vẫn bị điểm thấp |
| **BLEU** | Trùng khớp n-gram (có smoothing) | Vốn cho dịch máy; câu trả lời LLM diễn giải tự nhiên (paraphrase) dù đúng vẫn hay bị điểm thấp — chỉ tham khảo |
| **Semantic Similarity** | Cosine similarity giữa embedding cục bộ (`paraphrase-multilingual-MiniLM-L12-v2`, không tốn phí OpenAI) của câu trả lời và gold | Bù nhược điểm của ROUGE/BLEU — đo đúng **ý nghĩa** dù từ ngữ khác |
"""

GROUNDING_GUIDE = """
### Độ trung thực (grounding score) — chỉ số riêng của dự án

Tính trong `services/agentic_rag.py`, **không so với gold** mà so câu trả lời với
**chính các passage đã retrieve**:
- Mỗi câu trong câu trả lời → embedding cục bộ → so cosine similarity với các passage nguồn.
- Câu nào similarity dưới ngưỡng (`AGENT_GROUNDING_THRESHOLD`, mặc định 0.35) → bị
  đánh dấu **nghi "bịa"** (hallucination).
- `grounding_score` = điểm trung bình → đo mức độ câu trả lời **bám sát nguồn**,
  không phải đo đúng/sai so với đáp án mẫu. Phát hiện hallucination **không cần
  LLM-judge** (tốn phí), chạy bằng embedding cục bộ.

### 2 chỉ số kiểu RAGAS bổ sung — cũng không cần gold, dùng cho mục "Lịch sử câu hỏi thực tế"

Tính trong `check_rag_quality_extra()` (`services/agentic_rag.py`), cùng cơ chế
embedding cục bộ như grounding, nhưng so sánh khác cặp:

| Chỉ số | So sánh | Phát hiện lỗi gì |
|---|---|---|
| **Answer Relevance** | embedding(câu hỏi) vs embedding(câu trả lời) | Câu trả lời đúng nhưng **lạc đề/chung chung**, không bám sát câu hỏi |
| **Context Relevance** | embedding(câu hỏi) vs embedding(từng passage retrieve được) | **Retrieval tìm sai** — passage không liên quan tới câu hỏi, dù câu trả lời cuối có thể vẫn ổn nhờ LLM tự lọc |

Cả 3 (Grounding/Faithfulness + Answer Relevance + Context Relevance) hợp thành bộ
chỉ số kiểu **RAGAS** — framework đánh giá RAG phổ biến không cần đáp án mẫu,
dùng được ngay cho câu hỏi tự do người dùng hỏi qua chat UI.
"""

FULL_GUIDE_MARKDOWN = "\n---\n".join(
    [TOPK_GUIDE, RETRIEVAL_METRIC_GUIDE, RESPONSE_METRIC_GUIDE, GROUNDING_GUIDE, KNOWN_LIMITATION_GOLD_DOCS]
)

# Tooltip ngắn cho st.metric(help=...) — 1 dòng mỗi chỉ số.
METRIC_HELP = {
    "precision": "Trong k điều trả về, bao nhiêu % chính xác (phạt khi trả về điều không liên quan).",
    "recall": "Trong tất cả điều đúng cần tìm, tìm được bao nhiêu % trong top-k (phạt khi bỏ sót).",
    "hit": "1 nếu top-k có ít nhất 1 điều đúng, ngược lại 0. Trung bình = % câu hỏi tìm đúng ≥1 nguồn.",
    "mrr": "1/rank của điều đúng đầu tiên xuất hiện — đo điều đúng xuất hiện sớm cỡ nào.",
    "ndcg": "Chất lượng xếp hạng tổng thể có trọng số theo vị trí (vị trí cao đóng góp nhiều hơn).",
    "rouge_l": "Longest common subsequence với đáp án mẫu — nhạy với cách diễn đạt.",
    "bleu": "Trùng khớp n-gram với đáp án mẫu — câu trả lời paraphrase đúng ý vẫn có thể bị thấp.",
    "semantic_similarity": "Cosine similarity embedding cục bộ với đáp án mẫu — đo đúng ý nghĩa dù khác từ ngữ.",
    "grounding": "Câu trả lời bám sát passage đã retrieve tới mức nào (chống hallucination), không so với gold.",
    "answer_relevance": "Câu trả lời có đúng trọng tâm câu hỏi không (bắt lỗi lạc đề/chung chung) — không cần gold.",
    "context_relevance": "Passage đã retrieve có liên quan tới câu hỏi không (đo chất lượng retrieval) — không cần gold.",
}
