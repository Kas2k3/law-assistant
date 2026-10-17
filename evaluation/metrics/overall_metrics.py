from .retrieval_metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_at_k, mrr_at_k
from .response_metrics import rouge_l_score, bleu_score, semantic_similarity


def evaluate_dataset(dataset, k=3):
    results = []

    for item in dataset:
        retrieved_docs = item["retrieved_docs"]
        gold_docs = item["gold_docs"]
        response = item["response"]
        gold_response = item["gold_response"]

        retrieval_metrics = {
            "precision@k": precision_at_k(retrieved_docs, gold_docs, k),
            "recall@k": recall_at_k(retrieved_docs, gold_docs, k),
            "hit@k": hit_at_k(retrieved_docs, gold_docs, k),
            "mrr@k": mrr_at_k(retrieved_docs, gold_docs, k),
            "ndcg@k": ndcg_at_k(retrieved_docs, gold_docs, k),
        }

        response_metrics = {
            "rouge_l": rouge_l_score(response, gold_response),
            "bleu": bleu_score(response, gold_response),
            "semantic_similarity": semantic_similarity(response, gold_response),
        }

        results.append({
            "query": item["query"],
            "retrieval_metrics": retrieval_metrics,
            "response_metrics": response_metrics
        })

    return results