from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

# load embedding model 1 lần
_embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def rouge_l_score(response, gold_response):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(gold_response, response)
    return scores["rougeL"].fmeasure


def bleu_score(response, gold_response):
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        [gold_response.split()],
        response.split(),
        smoothing_function=smoothie
    )


def semantic_similarity(response, gold_response):
    emb1 = _embedding_model.encode(response, convert_to_tensor=True)
    emb2 = _embedding_model.encode(gold_response, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()
