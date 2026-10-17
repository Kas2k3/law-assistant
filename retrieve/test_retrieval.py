"""
Test Script for Retrieval Strategies
=====================================

Script này dùng để test và so sánh các chiến lược retrieval khác nhau

Kết quả được lưu vào thư mục: results/
- results/<strategy>_results.json
- results/<strategy>_results.csv
- results/comparison_summary.json
"""

import json
from retrieval_strategies import RetrievalEngine, export_to_csv, export_to_json, print_results


def load_test_queries(filepath: str = "test_queries.json"):
    """Load câu hỏi test từ file JSON"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠ File {filepath} không tồn tại, sử dụng queries mặc định")
        return generate_default_queries()


def generate_default_queries():
    """Tạo câu hỏi test mặc định"""
    return {
        "Hình sự": [
            "Hành vi chống đối người thi hành nhiệm vụ sẽ bị xử lý như thế nào?",
            "Những hành vi nào bị coi là xâm phạm an ninh quốc gia?",
            "Tội giết người có thời hiệu truy cứu bao lâu?",
        ],
        "Lao động": [
            "Lao động nữ trong thời kỳ thai sản có quyền lợi gì?",
            "Thời gian làm việc tối đa trong một tuần là bao nhiêu?",
            "Người lao động có quyền đơn phương chấm dứt hợp đồng khi nào?",
        ],
        "Dân sự": [
            "Điều kiện để kết hôn hợp pháp là gì?",
            "Quyền sở hữu tài sản được bảo vệ như thế nào?",
            "Thời hiệu khởi kiện trong tranh chấp dân sự là bao lâu?",
        ]
    }


def run_single_strategy_test(strategy: str, queries: list, top_k: int = 20, **kwargs):
    """Test một chiến lược với danh sách queries"""
    print(f"\n{'='*70}")
    print(f"Testing Strategy: {strategy.upper()}")
    print(f"Number of queries: {len(queries)}")
    print(f"Top K: {top_k}")
    print(f"{'='*70}")
    
    engine = RetrievalEngine()
    results = engine.batch_retrieve(
        queries=queries,
        strategy=strategy,
        top_k=top_k,
        **kwargs
    )
    
    # Hiển thị kết quả
    print_results(results, max_results_per_query=3)
    
    # Lưu kết quả
    json_file = f"results/{strategy}_results.json"
    csv_file = f"results/{strategy}_results.csv"
    
    export_to_json(results, json_file)
    export_to_csv(results, csv_file)
    
    print(f"\n📁 Kết quả đã được lưu vào:")
    print(f"   - {json_file}")
    print(f"   - {csv_file}")
    
    return results


def run_comparison_test(queries: list, strategies: list, top_k: int = 20):
    """So sánh nhiều chiến lược trên cùng tập queries"""
    print(f"\n{'#'*70}")
    print(f"# COMPARISON TEST")
    print(f"# Queries: {len(queries)}")
    print(f"# Strategies: {', '.join(strategies)}")
    print(f"# Top K: {top_k}")
    print(f"{'#'*70}")
    
    all_results = {}
    
    for strategy in strategies:
        results = run_single_strategy_test(strategy, queries, top_k)
        all_results[strategy] = results
    
    # Export comparison summary
    summary_file = "results/comparison_summary.json"
    export_comparison_summary(all_results, summary_file)
    
    print(f"\n{'='*70}")
    print(f"📊 SO SÁNH KẾT QUẢ")
    print(f"{'='*70}")
    
    for strategy, results in all_results.items():
        total_docs = sum(len(items) for items in results.values())
        avg_score = sum(items[0]["score"] if items else 0.0 for items in results.values()) / len(results) if results else 0.0
        
        print(f"\n{strategy}:")
        print(f"  📝 Total queries: {len(results)}")
        print(f"  📄 Total documents: {total_docs}")
        print(f"  ⭐ Avg top-1 score: {avg_score:.4f}")
    
    print(f"\n{'='*70}")
    print(f"✓ Tất cả kết quả đã được lưu vào thư mục: results/")
    print(f"{'='*70}\n")
    
    return all_results


def export_comparison_summary(all_results: dict, filename: str):
    """Tạo summary so sánh các strategies"""
    summary = {}
    
    for strategy, results in all_results.items():
        summary[strategy] = {
            "total_queries": len(results),
            "avg_top_score": sum(
                items[0]["score"] if items else 0.0 
                for items in results.values()
            ) / len(results) if results else 0.0,
            "queries": list(results.keys())
        }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Exported comparison summary to: {filename}")


def run_full_test_suite():
    """Chạy bộ test đầy đủ cho tất cả strategies"""
    # Load queries
    queries_by_category = load_test_queries()
    
    # Flatten tất cả queries
    all_queries = []
    for category, qs in queries_by_category.items():
        all_queries.extend(qs)
    
    print(f"\n{'*'*70}")
    print(f"FULL TEST SUITE")
    print(f"Total categories: {len(queries_by_category)}")
    print(f"Total queries: {len(all_queries)}")
    print(f"{'*'*70}")
    
    # Các strategies cần test
    strategies = [
        "semantic_search",
        "hybrid_search", 
        "two_stage_search",
        "hybrid_rerank"
    ]
    
    # Test tất cả strategies
    results = run_comparison_test(
        queries=all_queries,
        strategies=strategies,
        top_k=20
    )
    
    # In summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    for strategy, strategy_results in results.items():
        print(f"\n{strategy}:")
        print(f"  - Total queries: {len(strategy_results)}")
        print(f"  - Avg documents retrieved: {sum(len(r) for r in strategy_results.values()) / len(strategy_results):.1f}")
        
        if strategy_results:
            first_result = list(strategy_results.values())[0]
            if first_result:
                print(f"  - Sample top score: {first_result[0]['score']:.4f}")
    
    print(f"\n{'='*70}")
    print("✓ Full test suite completed!")
    print(f"{'='*70}\n")


def run_custom_test(queries: list, strategy: str = "semantic_search", top_k: int = 20):
    """Test nhanh với custom queries"""
    print(f"\n{'='*70}")
    print(f"CUSTOM TEST")
    print(f"Strategy: {strategy}")
    print(f"Queries: {len(queries)}")
    print(f"Top K: {top_k}")
    print(f"{'='*70}")
    
    engine = RetrievalEngine()
    results = engine.batch_retrieve(queries, strategy, top_k)
    
    # Hiển thị kết quả
    print_results(results, max_results_per_query=3)
    
    # Lưu kết quả
    json_file = f"results/custom_{strategy}_results.json"
    csv_file = f"results/custom_{strategy}_results.csv"
    
    export_to_json(results, json_file)
    export_to_csv(results, csv_file)
    
    print(f"\n📁 Kết quả đã được lưu vào:")
    print(f"   - {json_file}")
    print(f"   - {csv_file}\n")
    
    return results


# ==================== MAIN ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "full":
            # Chạy full test suite
            run_full_test_suite()
            
        elif mode == "single":
            # Test một strategy
            strategy = sys.argv[2] if len(sys.argv) > 2 else "semantic_search"
            queries = [
                "Hành vi chống đối người thi hành nhiệm vụ sẽ bị xử lý như thế nào?",
                "Lao động nữ trong thời kỳ thai sản có quyền lợi gì?",
            ]
            run_custom_test(queries, strategy, top_k=10)
            
        elif mode == "compare":
            # So sánh strategies
            queries = [
                "Hành vi chống đối người thi hành nhiệm vụ sẽ bị xử lý như thế nào?",
            ]
            strategies = ["semantic_search", "hybrid_search", "two_stage_search"]
            run_comparison_test(queries, strategies, top_k=10)
            
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: full, single, compare")
    
    else:
        # Mặc định: chạy demo nhanh
        print("Running quick demo...")
        queries = [
            "Hành vi chống đối người thi hành nhiệm vụ sẽ bị xử lý như thế nào?",
            "Lao động nữ trong thời kỳ thai sản có quyền lợi gì?",
        ]
        
        run_custom_test(queries, strategy="semantic_search", top_k=5)