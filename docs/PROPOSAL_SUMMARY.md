# Tóm tắt `Proposal.docx` (tham khảo nhanh)

> Trích xuất lại dưới dạng text từ `Proposal.docx` (gốc đặt ở thư mục root) để tiện tra cứu/diff/grep khi lập
> kế hoạch — xem bản đầy đủ, định dạng trong file `.docx` gốc. Không chỉnh sửa nội dung, chỉ tổ chức lại cho
> dễ đọc.

## 1. Thông tin đề tài

- **Tên đề tài**: Vietnamese Legal Question Answering using RAG
- **Phương pháp chính**: Retrieval-Augmented Generation (RAG)
- **Môi trường dự kiến (ban đầu)**: Python, Google Colab / Kaggle

## 2. Motivation

LLM có thể gặp vấn đề kiến thức lỗi thời, thiếu nguồn tham chiếu, sinh thông tin không có trong tài liệu khi
trả lời câu hỏi pháp luật. RAG kết hợp khả năng sinh của LLM với tập tài liệu bên ngoài: truy xuất tài liệu
liên quan rồi dùng làm context. Đề tài hướng tới áp dụng RAG cho hỏi đáp pháp luật tiếng Việt, nghiên cứu ảnh
hưởng của retrieval và cách xây dựng context tới chất lượng câu trả lời.

## 3. Objectives

- Xây dựng pipeline RAG cho Vietnamese Legal QA.
- Xây dựng/sử dụng corpus văn bản pháp luật làm knowledge base.
- Preprocessing + chunking tài liệu pháp luật.
- Embedding + vector retrieval tìm đoạn văn bản liên quan.
- Dùng LLM sinh câu trả lời dựa trên context truy xuất.
- Đánh giá khả năng retrieval và chất lượng câu trả lời.

## 4. Dataset (dự kiến ban đầu)

- `YuITC/Vietnamese-Legal-Documents` (HuggingFace) hoặc `Tamir39/vietnam-tax-qa`.
- Dùng làm: legal documents (knowledge corpus), query/question (input), thông tin tài liệu liên quan/context
  (cơ sở đánh giá retrieval).
- Có thể giảm quy mô dataset nếu cần phù hợp giới hạn Colab/Kaggle.

## 5. Proposed Method — Pipeline dự kiến

```
Legal Documents → Document Preprocessing → Chunking → Embedding Model → Vector Database
User Question → Query Embedding → Top-K Retrieval → Relevant Contexts → LLM → Vietnamese Legal Answer
```

- **5.1 Document preprocessing**: loại bỏ thành phần không cần thiết, chuẩn hóa nội dung.
- **5.2 Chunking**: chia văn bản dài thành đoạn nhỏ hơn; thử nghiệm các cấu hình chunk size/overlap khác
  nhau để xem ảnh hưởng tới khả năng truy xuất.
- **5.3 Retrieval**: embedding model chuyển chunk + câu hỏi thành vector; chọn chunk có độ tương đồng cao làm
  context.
- **5.4 Generation**: đưa context + câu hỏi vào prompt; yêu cầu LLM chỉ dùng thông tin trong context, không
  bịa, trả lời tiếng Việt, có thể chỉ nguồn/context nếu dataset hỗ trợ.

## 6. (không có — proposal nhảy từ mục 5 sang 7)

## 7. Evaluation

### 7.1. Retrieval Evaluation

Recall@K, Hit@K, MRR — đánh giá tài liệu liên quan có xuất hiện trong top-K hay không.

### 7.2. Answer Evaluation

- Exact Match (nếu phù hợp).
- ROUGE / BLEU (nếu ground-truth answer phù hợp với các metric này).
- LLM-based evaluation (đánh giá semantic correctness).
- Human evaluation trên tập mẫu nhỏ nếu cần.
- Khía cạnh đánh giá: Correctness, Relevance, Faithfulness/groundedness, Completeness.

## 8. Expected Results

- Truy xuất đoạn văn bản pháp luật liên quan tới câu hỏi.
- Dùng context truy xuất để hỗ trợ LLM sinh câu trả lời.
- Giảm phụ thuộc câu trả lời vào kiến thức tham số của LLM.
- Hạn chế sinh thông tin không được tài liệu hỗ trợ.
- Phân tích ảnh hưởng của chunking, Top-K, retrieval tới chất lượng hệ thống.

## 9. Resource Requirements

Ưu tiên model vừa/nhỏ để chạy trên tài nguyên miễn phí: Python, Hugging Face Transformers, Sentence
Transformers (hoặc tương đương), FAISS hoặc vector DB nhẹ, Google Colab/Kaggle, GPU miễn phí nếu cần.

## 10. Expected Contribution

- Xây dựng pipeline RAG cho dữ liệu pháp luật tiếng Việt.
- Đánh giá riêng retrieval và generation.
- Phân tích ảnh hưởng chunk size, Top-K.
- Nhận xét các yếu tố ảnh hưởng chất lượng Vietnamese Legal QA.

## 11. Tentative Timeline (bản gốc, quy mô baseline Colab/Kaggle)

| Tuần | Công việc |
|---|---|
| 1 | Tìm hiểu RAG/dataset/nghiên cứu liên quan; preprocessing + xây corpus; xây embedding + retrieval |
| 2 | Tích hợp LLM, hoàn thành baseline RAG |
| 3 | Thực hiện các thí nghiệm |
| 4 | Đánh giá và phân tích kết quả |
| 5 | Hoàn thiện báo cáo và demo |

## 12. References

- Lewis, P. et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020.
- Hugging Face Datasets — Vietnamese Legal Documents.
- Các nghiên cứu liên quan Vietnamese Legal QA và RAG.

---

## Khoảng cách giữa Proposal và hệ thống hiện có

Hệ thống thực tế trong repo này **đã vượt xa quy mô baseline** mà proposal mô tả (Colab/Kaggle, FAISS, 1
pipeline RAG đơn giản): đã có agentic tool-calling, graph RAG (Neo4j), web fallback (Tavily), pre-retrieve
agents, full-stack UI (Next.js + FastAPI), TTS/STT. Tuy nhiên, về đúng **phần mà proposal nhấn mạnh làm trọng
tâm nghiên cứu** — (a) thí nghiệm chunking/Top-K có hệ thống, (b) đánh giá retrieval + generation nghiêm túc,
(c) phân tích định lượng ảnh hưởng của từng thành phần — thì hệ thống hiện tại **còn thiếu hoặc chưa khớp**
(xem mục 14 của `ARCHITECTURE.md` và toàn bộ `DEVELOPMENT_PLAN.md`). Kế hoạch phát triển tiếp theo xoay quanh
việc lấp khoảng cách này mà **không phá vỡ** hệ thống production đã có.
