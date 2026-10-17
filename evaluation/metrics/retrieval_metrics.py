import math

def precision_at_k(retrieved_docs, gold_docs, k):
    if k == 0:
        return 0.0
    retrieved_k = retrieved_docs[:k]
    relevant = set(gold_docs)
    correct = sum(1 for doc in retrieved_k if doc in relevant)
    return correct / k


def recall_at_k(retrieved_docs, gold_docs, k):
    if not gold_docs:
        return 0.0
    retrieved_k = retrieved_docs[:k]
    relevant = set(gold_docs)
    correct = sum(1 for doc in retrieved_k if doc in relevant)
    return correct / len(relevant)


def hit_at_k(retrieved_docs, gold_docs, k):
    """
    Hit@k:
      1.0 nếu trong top-k có ÍT NHẤT 1 doc thuộc gold_docs
      0.0 nếu không có doc nào đúng.

    Khi report trên cả dataset, bạn lấy trung bình hit@k của tất cả query.
    """
    if k == 0:
        return 0.0

    relevant = set(gold_docs)
    if not relevant:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    return 1.0 if any(doc in relevant for doc in retrieved_k) else 0.0


def mrr_at_k(retrieved_docs, gold_docs, k):
    """
    MRR@k (Mean Reciprocal Rank at k):
      Trả về 1/rank của doc liên quan đầu tiên trong top-k.
      Nếu không có doc nào liên quan trong top-k, trả về 0.0.

    Ví dụ:
      - Doc liên quan ở vị trí 1 → MRR = 1/1 = 1.0
      - Doc liên quan ở vị trí 3 → MRR = 1/3 = 0.333
      - Không có doc liên quan trong top-k → MRR = 0.0

    Khi report trên cả dataset, bạn lấy trung bình MRR của tất cả query.
    """
    if k == 0:
        return 0.0

    relevant = set(gold_docs)
    if not relevant:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    for i, doc in enumerate(retrieved_k):
        if doc in relevant:
            return 1.0 / (i + 1)

    return 0.0


def ndcg_at_k(retrieved_docs, gold_docs, k):
    relevant = set(gold_docs)
    dcg = 0.0

    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant:
            dcg += 1 / math.log2(i + 2)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0