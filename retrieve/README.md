# Legal Document Retrieval Strategies

Hệ thống retrieval cho tài liệu pháp luật với 4 chiến lược khác nhau.

## 📋 Các Chiến Lược

### 1. **Semantic Search** 🔍
- **Mô tả**: Tìm kiếm vector similarity thuần túy
- **Collection**: `phapdien_articles` (full text)
- **Phương pháp**: COSINE similarity trên embeddings
- **Ưu điểm**: Nhanh, đơn giản, bắt được semantic meaning
- **Nhược điểm**: Có thể miss exact keyword matches

```python
results = engine.retrieve(query, strategy="semantic_search", top_k=20)
```

### 2. **Hybrid Search** 🔀
- **Mô tả**: Kết hợp vector similarity + lexical matching
- **Collection**: `phapdien_articles`
- **Phương pháp**: 
  - Vector search để lấy candidates
  - Tính text overlap score (lexical)
  - Kết hợp: `alpha * vector_score + (1-alpha) * lexical_score`
- **Ưu điểm**: Cân bằng semantic và exact matching
- **Nhược điểm**: Phức tạp hơn, cần tune alpha

```python
results = engine.retrieve(query, strategy="hybrid_search", top_k=20, alpha=0.6)
```

### 3. **Two-Stage Search** 🎯
- **Mô tả**: Vector search trên tên điều + rerank bằng text overlap
- **Collection**: `phapdien_tendieu` (chỉ tên điều)
- **Phương pháp**:
  1. Vector search trên tên điều (nhẹ, nhanh)
  2. Lấy full context (chủ đề, đề mục, chương, nội dung)
  3. Rerank dựa trên text overlap với full context
- **Ưu điểm**: Efficient first stage, context-aware reranking
- **Nhược điểm**: Phụ thuộc vào chất lượng tên điều

```python
results = engine.retrieve(query, strategy="two_stage_search", top_k=20)
```

### 4. **Hybrid Rerank** 🤖
- **Mô tả**: Vector search + LLM reranking
- **Collection**: `phapdien_tendieu`
- **Phương pháp**:
  1. Vector search trên tên điều
  2. Lấy full context
  3. Dùng GPT-4o-mini để chấm điểm relevance (0-1)
- **Ưu điểm**: Accuracy cao nhất, LLM hiểu context tốt
- **Nhược điểm**: Chậm, tốn API cost

```python
results = engine.retrieve(query, strategy="hybrid_rerank", top_k=20)
```

## 🚀 Cài Đặt

```bash
pip install pymilvus openai sqlalchemy mysql-connector-python pandas numpy
```

## 📊 Cấu Hình Database

### MySQL
```python
engine = create_engine("mysql+mysqlconnector://root:password@localhost:3306/law")
```

### Milvus Collections
- `phapdien_articles`: Full text embeddings
- `phapdien_tendieu`: Article title embeddings

## 💻 Sử Dụng

### Cơ Bản

```python
from retrieval_strategies import RetrievalEngine

# Khởi tạo engine
engine = RetrievalEngine()

# Single query
query = "Hành vi chống đối người thi hành nhiệm vụ sẽ bị xử lý như thế nào?"
results = engine.retrieve(
    query=query,
    strategy="semantic_search",  # hoặc hybrid_search, two_stage_search, hybrid_rerank
    top_k=20
)

# In kết quả
for i, item in enumerate(results, 1):
    print(f"{i}. [{item['score']:.4f}] {item['ten']}")
    print(f"   {item['noidung'][:100]}...")
```

### Batch Processing

```python
# Multiple queries
queries = [
    "Lao động nữ trong thời kỳ thai sản có quyền lợi gì?",
    "Thời gian làm việc tối đa trong một tuần là bao nhiêu?",
]

results = engine.batch_retrieve(
    queries=queries,
    strategy="hybrid_search",
    top_k=20,
    alpha=0.6  # tham số cho hybrid_search
)

# results là dict: query -> list of documents
for query, docs in results.items():
    print(f"Query: {query}")
    print(f"Found {len(docs)} documents")
```

## 🧪 Testing

### Quick Demo
```bash
python test_retrieval.py
```

### Test Single Strategy
```bash
python test_retrieval.py single semantic_search
```

### Compare Strategies
```bash
python test_retrieval.py compare
```

### Full Test Suite
```bash
python test_retrieval.py full
```

## 📁 Cấu Trúc File

```
.
├── retrieval_strategies.py    # Core implementation
├── test_retrieval.py          # Test scripts
├── test_queries.json          # Sample queries
├── results/                   # Output directory
│   ├── semantic_search_results.json
│   ├── hybrid_search_results.csv
│   ├── two_stage_search_results.json
│   ├── hybrid_rerank_results.json
│   └── comparison_summary.json
└── README.md
```

## 📈 So Sánh Performance

| Strategy | Speed | Accuracy | Cost | Use Case |
|----------|-------|----------|------|----------|
| Semantic Search | ⚡⚡⚡ | 🎯🎯 | 💰 | General search |
| Hybrid Search | ⚡⚡ | 🎯🎯🎯 | 💰 | Balanced performance |
| Two-Stage Search | ⚡⚡ | 🎯🎯🎯 | 💰 | Title-based search |
| Hybrid Rerank | ⚡ | 🎯🎯🎯🎯 | 💰💰💰 | High precision needed |

## 🔧 Tùy Chỉnh

### Thay đổi embedding model
```python
# Trong hàm embed_batch()
resp = client.embeddings.create(
    model="text-embedding-3-large",  # hoặc model khác
    input=texts
)
```

### Thay đổi LLM reranking model
```python
engine = RetrievalEngine()
engine.strategies["hybrid_rerank"].rerank_model = "gpt-4"
```

### Điều chỉnh số candidates
```python
# Trong class initialization
HybridSearchStrategy(candidate_multiplier=3)  # mặc định là 2
```

## 📊 Export Results

```python
from retrieval_strategies import export_to_csv, export_to_json

# Export to CSV
export_to_csv(results, "output.csv")

# Export to JSON
export_to_json(results, "output.json")
```

## ⚠️ Lưu Ý

1. **API Keys**: Cần set `OPENAI_API_KEY` environment variable
2. **Database Connection**: Kiểm tra MySQL và Milvus đang chạy
3. **Collections**: Đảm bảo collections đã được tạo và có dữ liệu
4. **LLM Cost**: Strategy `hybrid_rerank` tốn nhiều API calls

## 🐛 Troubleshooting

### Lỗi kết nối Milvus
```python
connections.connect("default", host="localhost", port="19530")
# Kiểm tra Milvus đang chạy: docker ps
```

### Lỗi collection không tồn tại
```bash
# List collections trong Milvus
from pymilvus import utility
print(utility.list_collections())
```

### Lỗi MySQL connection
```python
# Test connection
import pandas as pd
df = pd.read_sql("SELECT 1", engine)
print(df)
```

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. Database connections
2. Collection names
3. API keys
4. Python dependencies

## 📝 License

MIT License