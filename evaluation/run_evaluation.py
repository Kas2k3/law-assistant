"""
Evaluation Framework for Retrieval Strategies
==============================================

Đánh giá 4 chiến lược retrieval trên tập data.json:
1. semantic_search
2. hybrid_search
3. two_stage_search
4. hybrid_rerank

Đánh giá cả:
- Retrieval metrics (Precision, Recall, Hit, MRR, NDCG)
- Response metrics (ROUGE-L, BLEU, Semantic Similarity)
"""

import json
import sys
import os
from typing import List, Dict, Any

# Thêm đường dẫn đến thư mục retrieve (ngang hàng với evaluation)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
retrieve_dir = os.path.join(parent_dir, "retrieve")
services_dir = os.path.join(parent_dir, "services")

# Thêm retrieve và services vào sys.path
for path in [retrieve_dir, services_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"📁 Current directory: {current_dir}")
print(f"📁 Parent directory: {parent_dir}")
print(f"📁 Retrieve directory: {retrieve_dir}")
print(f"📁 Services directory: {services_dir}")

# Kiểm tra xem có tìm thấy retrieval_strategies không
try:
    from retrieval_strategies import RetrievalEngine
    print(f"✓ Đã import RetrievalEngine thành công!")
except ImportError as e:
    print(f"\n❌ LỖI: Không tìm thấy retrieval_strategies.py")
    print(f"\nĐã tìm trong: {retrieve_dir}")
    print(f"Chi tiết lỗi: {e}")
    sys.exit(1)

# Import utils từ services
try:
    from utils import generate_response
    print(f"✓ Đã import generate_response thành công!")
except ImportError as e:
    print(f"\n❌ LỖI: Không tìm thấy services/utils.py")
    print(f"\nĐã tìm trong: {services_dir}")
    print(f"Chi tiết lỗi: {e}")
    sys.exit(1)

# Import metrics
try:
    from metrics.retrieval_metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_at_k, mrr_at_k
    from metrics.response_metrics import rouge_l_score, bleu_score, semantic_similarity
    print(f"✓ Đã import metrics thành công!")
except ImportError as e:
    print(f"\n❌ LỖI: Không tìm thấy metrics/")
    print(f"Chi tiết lỗi: {e}")
    sys.exit(1)

print()


class RetrievalEvaluator:
    """Đánh giá các chiến lược retrieval"""
    
    def __init__(self, data_file: str = "data.json"):
        self.data_file = data_file
        self.dataset = self._load_dataset()
        self.engine = RetrievalEngine()
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset từ file JSON"""
        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _extract_offsets(self, results: List[Dict[str, Any]]) -> List[int]:
        """Trích xuất danh sách offsets từ kết quả retrieval"""
        return [item.get("offset") for item in results if item.get("offset")]
    
    def evaluate_single_query(
        self, 
        query: str, 
        gold_docs: List[int],
        gold_response: str,
        strategy: str,
        top_k: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Đánh giá 1 query với 1 strategy
        
        Returns:
            Dict chứa:
            - retrieval_metrics
            - response_metrics  
            - generated_response
            - retrieved_docs
        """
        # Lọc kwargs phù hợp với từng strategy
        strategy_kwargs = {}
        
        if strategy == "hybrid_search" and "alpha" in kwargs:
            strategy_kwargs["alpha"] = kwargs["alpha"]
        
        # Lấy retrieved docs từ strategy
        results = self.engine.retrieve(
            query=query,
            strategy=strategy,
            top_k=top_k,
            **strategy_kwargs
        )
        
        retrieved_docs = self._extract_offsets(results)
        
        # Tính retrieval metrics (thêm MRR@k)
        retrieval_metrics = {
            "precision@k": precision_at_k(retrieved_docs, gold_docs, top_k),
            "recall@k": recall_at_k(retrieved_docs, gold_docs, top_k),
            "hit@k": hit_at_k(retrieved_docs, gold_docs, top_k),
            "mrr@k": mrr_at_k(retrieved_docs, gold_docs, top_k),
            "ndcg@k": ndcg_at_k(retrieved_docs, gold_docs, top_k)
        }
        
        # Generate response từ retrieved docs
        noidung_texts = [item.get("noidung", "") for item in results if item.get("noidung")]
        
        try:
            generated_response = generate_response(
                noidung_texts=noidung_texts,
                current_query=query,
                sentiment="neutral_info",  # Mặc định neutral
                chat_history=[]  # Không có history trong evaluation
            )
        except Exception as e:
            print(f"⚠ Lỗi generate response: {e}")
            generated_response = ""
        
        # Tính response metrics
        response_metrics = {}
        if generated_response and gold_response:
            try:
                response_metrics = {
                    "rouge_l": rouge_l_score(generated_response, gold_response),
                    "bleu": bleu_score(generated_response, gold_response),
                    "semantic_similarity": semantic_similarity(generated_response, gold_response)
                }
            except Exception as e:
                print(f"⚠ Lỗi tính response metrics: {e}")
                response_metrics = {
                    "rouge_l": 0.0,
                    "bleu": 0.0,
                    "semantic_similarity": 0.0
                }
        else:
            response_metrics = {
                "rouge_l": 0.0,
                "bleu": 0.0,
                "semantic_similarity": 0.0
            }
        
        return {
            "retrieval_metrics": retrieval_metrics,
            "response_metrics": response_metrics,
            "generated_response": generated_response,
            "retrieved_docs": retrieved_docs  # Lưu lại để không phải chạy lại
        }
    
    def evaluate_strategy(
        self,
        strategy: str,
        top_k: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Đánh giá 1 strategy trên toàn bộ dataset
        
        Returns:
            Dict chứa:
            - detailed_results: kết quả từng query
            - average_metrics: điểm trung bình các metrics (retrieval + response)
        """
        print(f"\n{'='*70}")
        print(f"Evaluating Strategy: {strategy.upper()}")
        print(f"Dataset size: {len(self.dataset)} queries")
        print(f"Top K: {top_k}")
        print(f"{'='*70}\n")
        
        detailed_results = []
        
        for i, item in enumerate(self.dataset, 1):
            query = item["query"]
            gold_docs = item["gold_docs"]
            gold_response = item.get("gold_response", "")
            
            print(f"[{i}/{len(self.dataset)}] Processing: {query[:60]}...", end="\r")
            
            # Đánh giá query này
            result = self.evaluate_single_query(
                query=query,
                gold_docs=gold_docs,
                gold_response=gold_response,
                strategy=strategy,
                top_k=top_k,
                **kwargs
            )
            
            detailed_results.append({
                "query": query,
                "gold_docs": gold_docs,
                "gold_response": gold_response,
                "generated_response": result["generated_response"],
                "retrieved_docs": result["retrieved_docs"],  # Lưu retrieved_docs
                "retrieval_metrics": result["retrieval_metrics"],
                "response_metrics": result["response_metrics"]
            })
        
        print()  # Newline
        
        # Tính điểm trung bình
        average_metrics = self._calculate_average_metrics(detailed_results)
        
        return {
            "strategy": strategy,
            "top_k": top_k,
            "num_queries": len(self.dataset),
            "detailed_results": detailed_results,
            "average_metrics": average_metrics
        }
    
    def _calculate_average_metrics(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """Tính điểm trung bình các metrics (retrieval + response)"""
        if not detailed_results:
            return {}
        
        retrieval_sums = {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "hit@k": 0.0,
            "mrr@k": 0.0,  # Thêm MRR
            "ndcg@k": 0.0
        }
        
        response_sums = {
            "rouge_l": 0.0,
            "bleu": 0.0,
            "semantic_similarity": 0.0
        }
        
        for result in detailed_results:
            # Retrieval metrics
            for metric_name, value in result["retrieval_metrics"].items():
                if metric_name in retrieval_sums:
                    retrieval_sums[metric_name] += value
            
            # Response metrics
            for metric_name, value in result["response_metrics"].items():
                if metric_name in response_sums:
                    response_sums[metric_name] += value
        
        n = len(detailed_results)
        
        return {
            "retrieval_metrics": {k: v / n for k, v in retrieval_sums.items()},
            "response_metrics": {k: v / n for k, v in response_sums.items()}
        }
    
    def evaluate_all_strategies(
        self,
        strategies: List[str] = None,
        top_k_values: List[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Đánh giá tất cả strategies với nhiều giá trị top_k
        
        Args:
            strategies: Danh sách strategies cần đánh giá
            top_k_values: Danh sách giá trị top_k cần test
            **kwargs: Tham số bổ sung cho strategies (vd: alpha cho hybrid_search)
        
        Returns:
            Dict chứa kết quả đầy đủ
        """
        if strategies is None:
            strategies = ["semantic_search", "hybrid_search", "two_stage_search", "hybrid_rerank"]
        
        if top_k_values is None:
            top_k_values = [3, 5, 10, 15, 20]
        
        print(f"\n{'#'*70}")
        print(f"# FULL EVALUATION")
        print(f"# Strategies: {', '.join(strategies)}")
        print(f"# Top K values: {top_k_values}")
        print(f"# Dataset: {len(self.dataset)} queries")
        print(f"{'#'*70}")
        
        all_results = {}
        
        for strategy in strategies:
            strategy_results = {}
            
            for k in top_k_values:
                key = f"k={k}"
                
                # Đánh giá strategy với top_k này
                result = self.evaluate_strategy(
                    strategy=strategy,
                    top_k=k,
                    **kwargs
                )
                
                strategy_results[key] = result
                
                # In kết quả ngắn gọn (thêm MRR)
                avg = result["average_metrics"]
                print(f"✓ {strategy} @ k={k}: "
                      f"P={avg['retrieval_metrics']['precision@k']:.4f} "
                      f"R={avg['retrieval_metrics']['recall@k']:.4f} "
                      f"H={avg['retrieval_metrics']['hit@k']:.4f} "
                      f"M={avg['retrieval_metrics']['mrr@k']:.4f} "
                      f"N={avg['retrieval_metrics']['ndcg@k']:.4f}")
            
            all_results[strategy] = strategy_results
        
        return all_results
    
    def print_comparison_table(self, results: Dict[str, Any], top_k_values: List[int]):
        """In bảng so sánh các strategies (retrieval + response metrics)"""
        print(f"\n{'='*100}")
        print("BẢNG SO SÁNH CÁC CHIẾN LƯỢC - RETRIEVAL METRICS")
        print(f"{'='*100}\n")
        
        # Retrieval Metrics (thêm MRR)
        for metric in ["precision@k", "recall@k", "hit@k", "mrr@k", "ndcg@k"]:
            print(f"📊 {metric.upper()}")
            print("-" * 100)
            
            # Header
            header = f"{'Strategy':<20}"
            for k in top_k_values:
                header += f" | TOP_{k:<5}"
            print(header)
            print("-" * 100)
            
            # Data rows
            for strategy in results.keys():
                row = f"{strategy:<20}"
                for k in top_k_values:
                    key = f"k={k}"
                    value = results[strategy][key]["average_metrics"]["retrieval_metrics"][metric]
                    row += f" | {value:<8.4f}"
                print(row)
            
            print()
        
        print(f"{'='*100}")
        print("BẢNG SO SÁNH CÁC CHIẾN LƯỢC - RESPONSE METRICS")
        print(f"{'='*100}\n")
        
        # Response Metrics
        for metric in ["rouge_l", "bleu", "semantic_similarity"]:
            print(f"📝 {metric.upper()}")
            print("-" * 100)
            
            # Header
            header = f"{'Strategy':<20}"
            for k in top_k_values:
                header += f" | TOP_{k:<5}"
            print(header)
            print("-" * 100)
            
            # Data rows
            for strategy in results.keys():
                row = f"{strategy:<20}"
                for k in top_k_values:
                    key = f"k={k}"
                    value = results[strategy][key]["average_metrics"]["response_metrics"][metric]
                    row += f" | {value:<8.4f}"
                print(row)
            
            print()
        
        print(f"{'='*100}\n")
    
    def print_best_metrics(self, results: Dict[str, Any], top_k_values: List[int]):
        """In ra strategy và top_k tốt nhất cho từng metric (retrieval + response)"""
        print(f"\n{'='*100}")
        print("🏆 CHIẾN LƯỢC TỐT NHẤT CHO TỪNG METRIC")
        print(f"{'='*100}\n")
        
        # Retrieval metrics (thêm MRR)
        print("📊 RETRIEVAL METRICS:")
        print("-" * 100)
        retrieval_metrics = ["precision@k", "recall@k", "hit@k", "mrr@k", "ndcg@k"]
        
        for metric in retrieval_metrics:
            best_value = -1
            best_strategy = ""
            best_k = 0
            
            for strategy in results.keys():
                for k in top_k_values:
                    key = f"k={k}"
                    value = results[strategy][key]["average_metrics"]["retrieval_metrics"][metric]
                    
                    if value > best_value:
                        best_value = value
                        best_strategy = strategy
                        best_k = k
            
            print(f"📌 {metric}:")
            print(f"   Strategy: {best_strategy}")
            print(f"   TOP_K: {best_k}")
            print(f"   Score: {best_value:.4f}\n")
        
        # Response metrics
        print("\n📝 RESPONSE METRICS:")
        print("-" * 100)
        response_metrics = ["rouge_l", "bleu", "semantic_similarity"]
        
        for metric in response_metrics:
            best_value = -1
            best_strategy = ""
            best_k = 0
            
            for strategy in results.keys():
                for k in top_k_values:
                    key = f"k={k}"
                    value = results[strategy][key]["average_metrics"]["response_metrics"][metric]
                    
                    if value > best_value:
                        best_value = value
                        best_strategy = strategy
                        best_k = k
            
            print(f"📌 {metric}:")
            print(f"   Strategy: {best_strategy}")
            print(f"   TOP_K: {best_k}")
            print(f"   Score: {best_value:.4f}\n")
        
        print(f"{'='*100}\n")
    
    def export_results(self, results: Dict[str, Any], output_file: str):
        """Export kết quả ra file JSON"""
        # Tạo thư mục nếu chưa tồn tại
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"✓ Đã tạo thư mục: {output_dir}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Đã lưu kết quả vào: {output_file}")


def main():
    """Main function"""
    
    # Khởi tạo evaluator
    evaluator = RetrievalEvaluator(data_file="data.json")
    
    # Các strategies cần đánh giá
    strategies = [
        "semantic_search",
        "hybrid_search",
        "two_stage_search",
        "two_stage_search_v2", 
        "hybrid_rerank"  # Comment vì chậm và tốn API cost
    ]
    
    # Các giá trị top_k
    top_k_values = [3, 5, 10, 15, 20]
    
    print(f"\n{'#'*70}")
    print(f"# FULL EVALUATION")
    print(f"# Strategies: {', '.join(strategies)}")
    print(f"# Top K values: {top_k_values}")
    print(f"# Dataset: {len(evaluator.dataset)} queries")
    print(f"{'#'*70}")
    
    all_results = {}
    
    # Đánh giá từng strategy
    for strategy in strategies:
        strategy_results = {}
        
        for k in top_k_values:
            key = f"k={k}"
            
            # Chuẩn bị kwargs cho strategy
            strategy_kwargs = {}
            if strategy == "hybrid_search":
                strategy_kwargs["alpha"] = 0.6  # Chỉ truyền alpha cho hybrid_search
            
            # Đánh giá strategy với top_k này
            result = evaluator.evaluate_strategy(
                strategy=strategy,
                top_k=k,
                **strategy_kwargs
            )
            
            strategy_results[key] = result
            
            # In kết quả ngắn gọn (thêm MRR)
            avg = result["average_metrics"]
            print(f"✓ {strategy} @ k={k}:")
            print(f"   Retrieval: P={avg['retrieval_metrics']['precision@k']:.4f} "
                  f"R={avg['retrieval_metrics']['recall@k']:.4f} "
                  f"H={avg['retrieval_metrics']['hit@k']:.4f} "
                  f"M={avg['retrieval_metrics']['mrr@k']:.4f} "
                  f"N={avg['retrieval_metrics']['ndcg@k']:.4f}")
            print(f"   Response:  ROUGE={avg['response_metrics']['rouge_l']:.4f} "
                  f"BLEU={avg['response_metrics']['bleu']:.4f} "
                  f"SemSim={avg['response_metrics']['semantic_similarity']:.4f}")
        
        all_results[strategy] = strategy_results
    
    # In bảng so sánh
    evaluator.print_comparison_table(all_results, top_k_values)
    
    # In strategy tốt nhất
    evaluator.print_best_metrics(all_results, top_k_values)
    
    # Export kết quả
    output_file = "results/retrieval_evaluation_results.json"
    evaluator.export_results(all_results, output_file)
    
    print(f"\n{'='*100}")
    print("✓ ĐÁNH GIÁ HOÀN TẤT!")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    # Cho phép chạy với tham số command line
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies")
    parser.add_argument("--strategies", nargs="+", 
                       default=["semantic_search", "hybrid_search", "two_stage_search", "two_stage_search_v2", "hybrid_rerank"],
                       help="Strategies to evaluate")
    parser.add_argument("--top-k", nargs="+", type=int,
                       default=[3, 5, 10, 15, 20],
                       help="Top K values to test")
    parser.add_argument("--data", type=str, default="data.json",
                       help="Path to data file")
    parser.add_argument("--output", type=str, 
                       default="results/retrieval_evaluation_results_2.json",
                       help="Output file path")
    parser.add_argument("--alpha", type=float, default=0.6,
                       help="Alpha parameter for hybrid_search (default: 0.6)")
    
    args = parser.parse_args()
    
    # Khởi tạo evaluator với data file được chỉ định
    evaluator = RetrievalEvaluator(data_file=args.data)
    
    print(f"\n{'#'*70}")
    print(f"# FULL EVALUATION")
    print(f"# Strategies: {', '.join(args.strategies)}")
    print(f"# Top K values: {args.top_k}")
    print(f"# Dataset: {len(evaluator.dataset)} queries")
    print(f"# Alpha (for hybrid_search): {args.alpha}")
    print(f"{'#'*70}")
    
    all_results = {}
    
    # Đánh giá từng strategy
    for strategy in args.strategies:
        strategy_results = {}
        
        for k in args.top_k:
            key = f"k={k}"
            
            # Chuẩn bị kwargs cho strategy
            strategy_kwargs = {}
            if strategy == "hybrid_search":
                strategy_kwargs["alpha"] = args.alpha
            
            # Đánh giá
            result = evaluator.evaluate_strategy(
                strategy=strategy,
                top_k=k,
                **strategy_kwargs
            )
            
            strategy_results[key] = result
            
            # In kết quả (thêm MRR)
            avg = result["average_metrics"]
            print(f"✓ {strategy} @ k={k}:")
            print(f"   Retrieval: P={avg['retrieval_metrics']['precision@k']:.4f} "
                  f"R={avg['retrieval_metrics']['recall@k']:.4f} "
                  f"H={avg['retrieval_metrics']['hit@k']:.4f} "
                  f"M={avg['retrieval_metrics']['mrr@k']:.4f} "
                  f"N={avg['retrieval_metrics']['ndcg@k']:.4f}")
            print(f"   Response:  ROUGE={avg['response_metrics']['rouge_l']:.4f} "
                  f"BLEU={avg['response_metrics']['bleu']:.4f} "
                  f"SemSim={avg['response_metrics']['semantic_similarity']:.4f}")
        
        all_results[strategy] = strategy_results
    
    # In kết quả
    evaluator.print_comparison_table(all_results, args.top_k)
    evaluator.print_best_metrics(all_results, args.top_k)
    
    # Export
    evaluator.export_results(all_results, args.output)