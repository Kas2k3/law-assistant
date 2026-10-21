"""Test check_answer_grounding() — không tải model thật (monkeypatch embedder)
để chạy nhanh, xác định trong CI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import services.agentic_rag as agentic_rag


class _FakeValues(list):
    def tolist(self):
        return list(self)


class _FakeTensor(list):
    def max(self, dim):
        class _R:
            def __init__(self, vals):
                self.values = _FakeValues(vals)
        return _R([max(row) for row in self])

    def tolist(self):
        return list(self)


class _FakeModel:
    """Giả lập embedding: câu/văn bản có chung 1 'từ khoá' thì similarity cao, ngược lại thấp."""
    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            texts = [texts]
        return texts  # giữ nguyên text, so khớp ở _FakeUtil


class _FakeUtil:
    @staticmethod
    def cos_sim(a, b):
        rows = []
        for sa in a:
            row = []
            for sb in b:
                overlap = len(set(sa.lower().split()) & set(sb.lower().split()))
                row.append(1.0 if overlap >= 2 else 0.1)
            rows.append(row)
        return _FakeTensor(rows)


def test_grounding_no_sentences_returns_none(monkeypatch):
    result = agentic_rag.check_answer_grounding("", [{"noidung": "abc"}])
    assert result["grounding_score"] is None


def test_grounding_no_passages_returns_none():
    result = agentic_rag.check_answer_grounding("Cau tra loi nao do.", [])
    assert result["grounding_score"] is None


def test_grounding_flags_unrelated_sentence(monkeypatch):
    monkeypatch.setattr("services.local_embeddings.get_local_embedder", lambda: _FakeModel())
    monkeypatch.setattr(agentic_rag, "_SENTENCE_SPLIT_PATTERN", agentic_rag._SENTENCE_SPLIT_PATTERN)

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "util", _FakeUtil(), raising=False)

    passages = [{"noidung": "quy dinh an ninh quoc gia viet nam"}]
    answer = "Quy dinh an ninh quoc gia duoc neu ro. Hom nay troi dep va toi di choi cong vien."

    result = agentic_rag.check_answer_grounding(answer, passages)
    assert result["grounding_score"] is not None
    assert any("troi dep" in s for s in result["flagged_sentences"])
