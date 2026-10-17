"""
Retrieval Strategies for Legal Document Search
===============================================

Bao gồm 4 chiến lược:
1. semantic_search: Vector similarity search thuần túy
2. hybrid_search: Vector + Lexical matching
3. two_stage_search: Vector search + Text overlap reranking
4. hybrid_rerank: Vector search + LLM reranking
"""

import re
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from pymilvus import connections, Collection
from sqlalchemy import create_engine
from openai import OpenAI


# ==================== CONFIGURATION ====================
# Kết nối OpenAI
client = OpenAI()

# Kết nối MySQL
engine = create_engine("mysql+mysqlconnector://root:123456789@localhost:3306/law")

# Kết nối Milvus
connections.connect("default", host="localhost", port="19530")
print("✓ Connected to Milvus")


# ==================== UTILITY FUNCTIONS ====================
def embed_batch(texts: List[str]) -> List[List[float]]:
    """Tạo embeddings cho danh sách texts"""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in resp.data]


def fetch_article_details(offset: int, engine) -> Dict[str, str]:
    """Lấy thông tin chi tiết của điều luật từ MySQL"""
    query = f"""
        SELECT 
            p.*,
            cd.ten AS ten_chude,
            dm.ten AS ten_demuc,
            ch.ten AS ten_chuong
        FROM pddieu p
        LEFT JOIN pdchude cd ON cd.id = p.chude_id
        LEFT JOIN pddemuc dm ON dm.id = p.demuc_id
        LEFT JOIN pdchuong ch ON ch.mapc = p.chuong_id
        LIMIT 1 OFFSET {offset - 1};
    """
    df = pd.read_sql(query, engine).fillna("")
    
    if df.empty:
        return {}
    
    row = df.iloc[0]
    return {
        "noidung": row["noidung"],
        "ten": row["ten"],
        "ten_chude": row["ten_chude"],
        "ten_demuc": row["ten_demuc"],
        "ten_chuong": row["ten_chuong"],
        "vbqppl": row["vbqppl"]
    }


def text_overlap_score(query: str, text: str) -> float:
    """Tính điểm trùng lặp từ ngữ giữa query và text"""
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    
    if not query_words:
        return 0.0
    
    overlap = query_words.intersection(text_words)
    return len(overlap) / len(query_words)


def min_max_normalize(values: List[float]) -> List[float]:
    """Chuẩn hóa danh sách giá trị về [0, 1]"""
    arr = np.array(values, dtype=float)
    
    if arr.size == 0:
        return []
    
    vmin, vmax = float(arr.min()), float(arr.max())
    
    if vmax - vmin < 1e-9:
        return [1.0 for _ in values]
    
    norm = (arr - vmin) / (vmax - vmin)
    return norm.tolist()


def format_article_block(details: Dict[str, str]) -> str:
    """Format thông tin điều luật thành block text"""
    return (
        f"Chủ đề: {details.get('ten_chude', '')}\n"
        f"Đề mục: {details.get('ten_demuc', '')}\n"
        f"Chương: {details.get('ten_chuong', '')}\n"
        f"Điều: {details.get('ten', '')}\n"
        f"{details.get('noidung', '')}\n"
        f"Căn cứ theo {details.get('vbqppl', '')}"
    )


# ==================== STRATEGY 1: SEMANTIC SEARCH ====================
class SemanticSearchStrategy:
    """
    Chiến lược 1: Tìm kiếm vector similarity thuần túy
    - Collection: phapdien_articles
    - Phương pháp: COSINE similarity
    """
    
    def __init__(self):
        self.collection_name = "phapdien_articles"
        self.collection = Collection(self.collection_name)
        self.collection.load()
        print(f"✓ Loaded collection: {self.collection_name}")
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Thực hiện semantic search"""
        # Tạo embedding cho query
        query_vec = embed_batch([query])[0]
        
        # Tìm kiếm trong Milvus
        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["metadata"]
        )
        
        if not results or not results[0]:
            return []
        
        # Xử lý kết quả
        items = []
        for hit in results[0]:
            details = fetch_article_details(hit.id, engine)
            
            items.append({
                "offset": hit.id,
                "score": float(hit.distance),
                "strategy": "semantic_search",
                **details
            })
        
        return items


# ==================== STRATEGY 2: HYBRID SEARCH ====================
class HybridSearchStrategy:
    """
    Chiến lược 2: Vector + Lexical matching
    - Collection: phapdien_articles
    - Phương pháp: Kết hợp vector similarity và text overlap
    - Score: alpha * vector_score + (1-alpha) * lexical_score
    """
    
    def __init__(self, candidate_multiplier: int = 2):
        self.collection_name = "phapdien_articles"
        self.collection = Collection(self.collection_name)
        self.collection.load()
        self.candidate_multiplier = candidate_multiplier
        print(f"✓ Loaded collection: {self.collection_name}")
    
    def search(self, query: str, top_k: int = 20, alpha: float = 0.6) -> List[Dict[str, Any]]:
        """
        Thực hiện hybrid search
        alpha: trọng số cho vector similarity (0-1)
        """
        # Lấy nhiều candidates hơn để rerank
        candidate_k = top_k * self.candidate_multiplier
        
        # Bước 1: Vector search
        query_vec = embed_batch([query])[0]
        
        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=candidate_k,
            output_fields=["metadata"]
        )
        
        if not results or not results[0]:
            return []
        
        # Bước 2: Tính lexical score cho từng candidate
        candidates = []
        vector_scores = []
        lexical_scores = []
        
        for hit in results[0]:
            details = fetch_article_details(hit.id, engine)
            
            # Vector similarity score
            vector_score = float(hit.distance)
            
            # Lexical matching score
            text = details.get("noidung", "")
            lexical_score = text_overlap_score(query, text)
            
            candidates.append({
                "offset": hit.id,
                "vector_score": vector_score,
                "lexical_score": lexical_score,
                "strategy": "hybrid_search",
                **details
            })
            
            vector_scores.append(vector_score)
            lexical_scores.append(lexical_score)
        
        # Bước 3: Chuẩn hóa và kết hợp scores
        vec_norm = min_max_normalize(vector_scores)
        lex_norm = min_max_normalize(lexical_scores)
        
        for i, cand in enumerate(candidates):
            combined = alpha * vec_norm[i] + (1.0 - alpha) * lex_norm[i]
            cand["score"] = combined
        
        # Bước 4: Sort và lấy top_k
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


# ==================== STRATEGY 3: TWO STAGE SEARCH ====================
class TwoStageSearchStrategy:
    """
    Chiến lược 3: Vector search trên tên điều + text overlap reranking
    - Collection: phapdien_tendieu (chỉ tên điều)
    - Bước 1: Vector search trên tên điều
    - Bước 2: Lấy full context (chủ đề, đề mục, chương, nội dung)
    - Bước 3: Rerank dựa trên text overlap với full context
    """
    
    def __init__(self, candidate_multiplier: int = 2):
        self.collection_name = "phapdien_tendieu"
        self.collection = Collection(self.collection_name)
        self.collection.load()
        self.candidate_multiplier = candidate_multiplier
        print(f"✓ Loaded collection: {self.collection_name}")
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Thực hiện two-stage search"""
        # Bước 1: Vector search trên tên điều
        candidate_k = top_k * self.candidate_multiplier
        query_vec = embed_batch([query])[0]
        
        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=candidate_k,
            output_fields=["metadata"]
        )
        
        if not results or not results[0]:
            return []
        
        # Bước 2: Lấy full context và tính overlap score
        candidates = []
        overlap_scores = []
        
        for hit in results[0]:
            details = fetch_article_details(hit.id, engine)
            
            # Tạo full context block
            full_block = format_article_block(details)
            
            # Tính text overlap với full context
            overlap_score = text_overlap_score(query, full_block)
            
            candidates.append({
                "offset": hit.id,
                "vector_score": float(hit.distance),
                "overlap_score": overlap_score,
                "strategy": "two_stage_search",
                "full_block": full_block,
                **details
            })
            
            overlap_scores.append(overlap_score)
        
        # Bước 3: Rerank theo overlap score
        candidates = sorted(candidates, key=lambda x: x["overlap_score"], reverse=True)
        
        # Gán lại score cuối cùng
        for cand in candidates:
            cand["score"] = cand["overlap_score"]
        
        return candidates[:top_k]

# ==================== STRATEGY 3B: TWO STAGE SEARCH V2 ====================
class TwoStageSearchV2Strategy:
    """
    Chiến lược 3B: Vector search trên tên điều + vector similarity reranking
    - Collection: phapdien_tendieu (chỉ tên điều)
    - Bước 1: Vector search trên tên điều
    - Bước 2: Lấy full context (chủ đề, đề mục, chương, nội dung)
    - Bước 3: Tạo embedding cho full context và rerank bằng vector similarity với query
    
    Khác biệt với Two-Stage Search:
    - Two-Stage Search: rerank bằng text overlap (lexical matching)
    - Two-Stage Search V2: rerank bằng vector similarity (semantic matching)
    """
    
    def __init__(self, candidate_multiplier: int = 2):
        self.collection_name = "phapdien_tendieu"
        self.collection = Collection(self.collection_name)
        self.collection.load()
        self.candidate_multiplier = candidate_multiplier
        print(f"✓ Loaded collection: {self.collection_name}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Tính cosine similarity giữa 2 vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return float(dot_product / (norm_v1 * norm_v2))
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Thực hiện two-stage search v2 với vector reranking"""
        # Bước 1: Vector search trên tên điều
        candidate_k = top_k * self.candidate_multiplier
        query_vec = embed_batch([query])[0]
        
        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=candidate_k,
            output_fields=["metadata"]
        )
        
        if not results or not results[0]:
            return []
        
        # Bước 2: Lấy full context cho tất cả candidates
        candidates = []
        full_contexts = []
        
        print(f"  → Fetching full context for {len(results[0])} candidates...")
        
        for hit in results[0]:
            details = fetch_article_details(hit.id, engine)
            
            # Tạo full context block
            full_block = format_article_block(details)
            full_contexts.append(full_block)
            
            candidates.append({
                "offset": hit.id,
                "initial_vector_score": float(hit.distance),
                "strategy": "two_stage_search_v2",
                "full_block": full_block,
                **details
            })
        
        # Bước 3: Tạo embeddings cho full contexts và rerank
        print(f"  → Creating embeddings for full contexts...")
        full_context_embeddings = embed_batch(full_contexts)
        
        print(f"  → Reranking by vector similarity...")
        rerank_scores = []
        
        for i, context_vec in enumerate(full_context_embeddings):
            # Tính cosine similarity giữa query và full context
            similarity = self._cosine_similarity(query_vec, context_vec)
            rerank_scores.append(similarity)
            candidates[i]["rerank_vector_score"] = similarity
            candidates[i]["score"] = similarity  # Score chính là rerank score
        
        # Bước 4: Sort theo rerank vector score
        candidates = sorted(candidates, key=lambda x: x["rerank_vector_score"], reverse=True)
        
        return candidates[:top_k]
    
# ==================== STRATEGY 4: HYBRID RERANK ====================
class HybridRerankStrategy:
    """
    Chiến lược 4: Vector search + LLM reranking
    - Collection: phapdien_tendieu
    - Bước 1: Vector search trên tên điều
    - Bước 2: Lấy full context
    - Bước 3: Dùng LLM (GPT-4o-mini) để chấm điểm relevance
    """
    
    def __init__(self, rerank_model: str = "gpt-4o-mini", candidate_multiplier: int = 2):
        self.collection_name = "phapdien_tendieu"
        self.collection = Collection(self.collection_name)
        self.collection.load()
        self.rerank_model = rerank_model
        self.candidate_multiplier = candidate_multiplier
        print(f"✓ Loaded collection: {self.collection_name}")
    
    def _llm_score(self, query: str, passage: str) -> float:
        """Dùng LLM để chấm điểm mức độ liên quan (0-1)"""
        system_prompt = (
            "Bạn là một hệ thống chấm điểm độ liên quan giữa câu hỏi và đoạn văn pháp luật. "
            "Đánh giá mức độ đoạn văn (passage) trả lời được câu hỏi (query). "
            "Hãy trả về MỘT số thực trong khoảng từ 0 đến 1:\n"
            "- 0: hoàn toàn không liên quan\n"
            "- 1: rất liên quan, trả lời trực tiếp câu hỏi\n"
            "Chỉ in ra đúng MỘT số, không giải thích gì thêm."
        )
        
        user_content = f"Query:\n{query}\n\nPassage:\n{passage}"
        
        try:
            resp = client.chat.completions.create(
                model=self.rerank_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=8,
            )
            
            text = resp.choices[0].message.content.strip()
            
            # Tìm số thực trong output
            match = re.search(r"([01](?:\.\d+)?)", text)
            if not match:
                return 0.0
            
            score = float(match.group(1))
            return max(0.0, min(1.0, score))  # Clamp về [0,1]
            
        except Exception as e:
            print(f"⚠ LLM scoring error: {e}")
            return 0.0
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Thực hiện hybrid rerank search"""
        # Bước 1: Vector search
        candidate_k = top_k * self.candidate_multiplier
        query_vec = embed_batch([query])[0]
        
        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=candidate_k,
            output_fields=["metadata"]
        )
        
        if not results or not results[0]:
            return []
        
        # Bước 2 & 3: Lấy context và LLM rerank
        candidates = []
        
        for i, hit in enumerate(results[0], 1):
            details = fetch_article_details(hit.id, engine)
            full_block = format_article_block(details)
            
            # LLM scoring
            print(f"  → Reranking {i}/{len(results[0])}...", end="\r")
            llm_score = self._llm_score(query, full_block)
            
            candidates.append({
                "offset": hit.id,
                "vector_score": float(hit.distance),
                "llm_score": llm_score,
                "score": llm_score,  # Dùng LLM score làm score chính
                "strategy": "hybrid_rerank",
                "full_block": full_block,
                **details
            })
        
        print()  # Newline sau khi rerank xong
        
        # Bước 4: Sort theo LLM score
        candidates = sorted(candidates, key=lambda x: x["llm_score"], reverse=True)
        return candidates[:top_k]


# ==================== RETRIEVAL ENGINE ====================
class RetrievalEngine:
    """Engine tổng hợp để chạy các chiến lược retrieval khác nhau"""
    
    def __init__(self):
        self.strategies = {
            "semantic_search": SemanticSearchStrategy(),
            "hybrid_search": HybridSearchStrategy(),
            "two_stage_search": TwoStageSearchStrategy(),
            "two_stage_search_v2": TwoStageSearchV2Strategy(),
            "hybrid_rerank": HybridRerankStrategy(),
        }
    
    def retrieve(
        self, 
        query: str, 
        strategy: str = "semantic_search",
        top_k: int = 20,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện retrieval với chiến lược được chỉ định
        
        Args:
            query: Câu hỏi/truy vấn
            strategy: Tên chiến lược (semantic_search, hybrid_search, two_stage_search, hybrid_rerank)
            top_k: Số lượng kết quả trả về
            **kwargs: Tham số bổ sung cho từng strategy (vd: alpha cho hybrid_search)
        
        Returns:
            Danh sách kết quả retrieval
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")
        
        print(f"\n🔍 Running {strategy} for query: {query[:50]}...")
        return self.strategies[strategy].search(query, top_k, **kwargs)
    
    def batch_retrieve(
        self,
        queries: List[str],
        strategy: str = "semantic_search",
        top_k: int = 20,
        **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Thực hiện retrieval cho nhiều queries
        
        Returns:
            Dict mapping từ query -> danh sách kết quả
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"📊 Batch Retrieval: {len(queries)} queries | Strategy: {strategy}")
        print(f"{'='*60}")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}]", end=" ")
            results[query] = self.retrieve(query, strategy, top_k, **kwargs)
        
        print(f"\n{'='*60}")
        print(f"✓ Completed batch retrieval")
        print(f"{'='*60}\n")
        
        return results


# ==================== EXPORT FUNCTIONS ====================
def print_results(results: Dict[str, List[Dict[str, Any]]], max_results_per_query: int = 5):
    """In kết quả retrieval ra console"""
    print(f"\n{'='*80}")
    print(f"RETRIEVAL RESULTS")
    print(f"{'='*80}")
    
    for query, items in results.items():
        print(f"\n📝 Query: {query}")
        print(f"   Found: {len(items)} documents")
        print(f"   {'-'*76}")
        
        for i, item in enumerate(items[:max_results_per_query], 1):
            score = item.get('score', 0.0)
            ten = item.get('ten', '')
            noidung = item.get('noidung', '')[:150]
            
            print(f"\n   {i}. [{score:.4f}] {ten}")
            print(f"      {noidung}...")
        
        if len(items) > max_results_per_query:
            print(f"\n      ... và {len(items) - max_results_per_query} kết quả khác")
    
    print(f"\n{'='*80}\n")


def export_to_csv(results: Dict[str, List[Dict[str, Any]]], filename: str):
    """Export kết quả retrieval ra file CSV"""
    import csv
    import os
    
    # Tạo thư mục nếu chưa tồn tại
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    total_rows = sum(len(items) for items in results.values())
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "offset", "score", "strategy", "ten", "noidung"])
        
        for query, items in results.items():
            for item in items:
                writer.writerow([
                    query,
                    item.get("offset"),
                    item.get("score", 0.0),
                    item.get("strategy"),
                    item.get("ten", ""),
                    item.get("noidung", "")[:500]  # Giới hạn độ dài
                ])
    
    print(f"✓ Exported {total_rows} rows to: {filename}")


def export_to_json(results: Dict[str, List[Dict[str, Any]]], filename: str):
    """Export kết quả retrieval ra file JSON"""
    import os
    
    # Tạo thư mục nếu chưa tồn tại
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    total_queries = len(results)
    total_docs = sum(len(items) for items in results.values())
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Exported {total_queries} queries ({total_docs} documents) to: {filename}")


# ==================== DEMO / TEST ====================
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Hành vi chống đối người thi hành nhiệm vụ sẽ bị xử lý như thế nào?",
        "Lao động nữ trong thời kỳ thai sản có quyền lợi gì?",
        "Những hành vi nào bị coi là xâm phạm an ninh quốc gia?",
    ]
    
    # Khởi tạo engine
    engine = RetrievalEngine()
    
    # Test từng strategy
    strategies = ["semantic_search", "hybrid_search", "two_stage_search", "two_stage_search_v2", "hybrid_rerank"]
    
    for strategy in strategies:
        print(f"\n\n{'#'*70}")
        print(f"# Testing Strategy: {strategy.upper()}")
        print(f"{'#'*70}")
        
        results = engine.batch_retrieve(
            queries=test_queries,
            strategy=strategy,
            top_k=5,
            alpha=0.6  # Chỉ dùng cho hybrid_search
        )
        
        # Export results
        export_to_json(results, f"results/{strategy}_results.json")
        export_to_csv(results, f"results/{strategy}_results.csv")