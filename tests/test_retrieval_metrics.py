"""Test các hàm metric thuần (không cần DB/OpenAI)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from metrics.retrieval_metrics import precision_at_k, recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["a", "c"], k=3) == 2 / 3
    assert precision_at_k([], ["a"], k=3) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "b", "c"], k=2) == 2 / 3


def test_recall_at_k_no_gold():
    assert recall_at_k(["a"], [], k=3) == 0.0


def test_hit_at_k():
    assert hit_at_k(["x", "a"], ["a"], k=2) == 1.0
    assert hit_at_k(["x", "y"], ["a"], k=2) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(["x", "a", "y"], ["a"], k=3) == 1 / 2
    assert mrr_at_k(["x", "y"], ["a"], k=2) == 0.0


def test_ndcg_at_k_perfect_order():
    assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == 1.0


def test_ndcg_at_k_empty_gold():
    assert ndcg_at_k(["a"], [], k=2) == 0.0
