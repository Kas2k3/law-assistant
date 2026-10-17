import os
from dotenv import load_dotenv  # thêm dòng này

load_dotenv()  # load trước khi getenv
from urllib.parse import urlparse
from tavily import TavilyClient

TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
SCORE_THRESHOLD = 0.6
MAX_WEB_RESULTS = 5

client = TavilyClient(api_key=TAVILY_API_KEY)

# ── Trust config ──────────────────────────────────────────────
TRUSTED_DOMAINS = {
    "thuvienphapluat.vn":  1.0,
    "vbpl.vn":             1.0,
    "moj.gov.vn":          1.0,
    "phapdien.moj.gov.vn": 1.0,
    "quochoi.vn":          0.95,
    "chinhphu.vn":         0.9,
    "toaan.gov.vn":        0.9,
    "luatvietnam.vn":      0.85,
    "lawnet.vn":           0.85,
}

UNTRUSTED_PATTERNS = [
    "blogspot", "wordpress", "facebook", "tiktok",
    "youtube", "forum", "wiki", "123doc", "tailieu",
    "slideshare", "scribd",
]

LEGAL_KEYWORDS = [
    "điều", "khoản", "nghị định", "thông tư", "luật",
    "bộ luật", "quyết định", "quy định", "xử phạt",
    "điều khoản", "văn bản", "pháp luật", "hành chính",
]


# ── Trust scoring ─────────────────────────────────────────────
def _compute_trust_score(result: dict) -> float:
    url     = result.get("url", "")
    content = (result.get("noidung") or "").lower()
    title   = (result.get("ten") or "").lower()
    domain  = urlparse(url).netloc.replace("www.", "")

    domain_score = 0.4
    for trusted, score in TRUSTED_DOMAINS.items():
        if trusted in domain:
            domain_score = score
            break
    for pattern in UNTRUSTED_PATTERNS:
        if pattern in domain:
            domain_score = 0.1
            break

    keyword_hits  = sum(1 for kw in LEGAL_KEYWORDS if kw in content or kw in title)
    content_score = min(keyword_hits / 5, 1.0)

    return round(domain_score * 0.7 + content_score * 0.3, 3)


def _classify_trust(score: float) -> tuple[str, str]:
    """Trả về (trust_level, source_label)."""
    if score >= 0.7:
        return "high",   "[Web]"
    elif score >= 0.4:
        return "medium", "[Web - tham khảo]"
    else:
        return "low",    "[Web - không tin cậy]"


def _filter_and_score(results: list[dict]) -> list[dict]:
    scored = []
    for r in results:
        score        = _compute_trust_score(r)
        level, label = _classify_trust(score)

        if level == "low":
            print(f"[Trust] Loại bỏ ({score}): {r.get('url','')}")
            continue

        r["trust_score"]  = score
        r["trust_level"]  = level
        r["source_label"] = label
        r["source"]       = "web"
        scored.append(r)

    scored.sort(key=lambda x: x["trust_score"], reverse=True)
    print(f"[Trust] Còn lại: {len(scored)}/{len(results)} kết quả sau lọc")
    return scored


# ── Public API ────────────────────────────────────────────────
RECENT_KEYWORDS = [
    "2025", "2024", "mới nhất", "hiện tại", "hiện nay",
    "gần đây", "vừa ban hành", "mới ban hành", "năm nay",
]

def is_context_sufficient(
    candidates: list[dict],
    query: str = "",                  # ← tham số mới
    threshold: float = SCORE_THRESHOLD,
) -> bool:

    # Ưu tiên 1: query chứa từ khoá thời sự → luôn fallback
    q = query.lower()
    if any(kw in q for kw in RECENT_KEYWORDS):
        print(f"[Sufficient] Phát hiện từ khoá thời sự trong query → buộc fallback Tavily")
        return False

    # Ưu tiên 2: không có kết quả
    if not candidates:
        return False

    # Ưu tiên 3: score trung bình top-3
    top3 = [c.get("score", 0) for c in candidates[:3]]
    avg  = sum(top3) / len(top3)

    if avg >= threshold and len(candidates) >= 3:
        # Kiểm tra relevance thực sự — ít nhất 1 kết quả khớp 2+ từ trong query
        query_tokens = set(q.split())
        has_relevant = any(
            len(query_tokens & set(
                ((c.get("ten") or "") + " " + (c.get("noidung") or "")).lower().split()
            )) >= 2
            for c in candidates[:5]
        )
        if not has_relevant:
            print(f"[Sufficient] Score đủ ({avg:.2f}) nhưng nội dung lạc chủ đề → fallback")
            return False
        return True

    return False


def search_tavily(query: str, top_k: int = MAX_WEB_RESULTS) -> list[dict]:
    try:
        response = client.search(
            query=f"pháp luật Việt Nam {query}",
            search_depth="advanced",
            max_results=top_k,
            include_answer=True,
        )
    except Exception as e:
        print(f"[Tavily] Lỗi search: {e}")
        return []

    raw = []

    # Answer tổng hợp — chưa có URL nên trust mặc định medium
    if response.get("answer"):
        raw.append({
            "mapc": None, "ten": "Tổng hợp từ web",
            "noidung": response["answer"],
            "tenchuong": None, "tendemuc": None, "tenchude": None,
            "passage": response["answer"],
            "score": 0.9, "url": "",
            "trust_score": 0.55, "trust_level": "medium",
            "source_label": "[Web - tham khảo]", "source": "web",
        })

    for r in response.get("results", []):
        passage = "\n".join(filter(None, [r.get("title",""), r.get("content","")[:800]]))
        raw.append({
            "mapc": None, "ten": r.get("title",""),
            "noidung": r.get("content","")[:800],
            "tenchuong": None, "tendemuc": None, "tenchude": None,
            "passage": passage,
            "score": r.get("score", 0.5),
            "url":   r.get("url",""),
        })

    filtered = _filter_and_score(raw)
    print(f"[Tavily] Trả về {len(filtered)} kết quả sau trust filter")
    return filtered
# ── Query Decomposer ──────────────────────────────────────────
from openai import OpenAI
import json as _json

_openai_client = OpenAI()

REALTIME_KEYWORDS = [
    "hôm nay", "hiện tại", "hiện nay", "bây giờ", "mới nhất",
    "tháng này", "năm nay", "đang", "gần đây",
    "giá", "tỷ giá", "lãi suất", "chỉ số", "thống kê",
    "2025", "2024",
]

def needs_realtime_data(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in REALTIME_KEYWORDS)


def decompose_query(query: str) -> dict:
    """
    Phân tích câu hỏi phức hợp thành sub-queries theo loại.
    Trả về dict: { legal_query, realtime_query, needs_calc, calc_hint }
    """
    if not needs_realtime_data(query):
        return {
            "legal_query":    query,
            "realtime_query": None,
            "needs_calc":     False,
            "calc_hint":      None,
        }

    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Bạn phân tích câu hỏi và tách thành các phần. "
                    "Chỉ trả về JSON hợp lệ, không giải thích.\n"
                    "Schema:\n"
                    "{\n"
                    '  "legal_query": "phần liên quan pháp luật (string hoặc null)",\n'
                    '  "realtime_query": "phần cần dữ liệu thực tế như giá, năm mới... (string hoặc null)",\n'
                    '  "needs_calc": true/false,\n'
                    '  "calc_hint": "mô tả phép tính cần làm (string hoặc null)"\n'
                    "}"
                )},
                {"role": "user", "content": f"Câu hỏi: {query}"}
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return _json.loads(raw)
    except Exception as e:
        print(f"[Decompose] Lỗi: {e} → dùng query gốc")
        return {
            "legal_query":    query,
            "realtime_query": None,
            "needs_calc":     False,
            "calc_hint":      None,
        }


def search_tavily_realtime(realtime_query: str) -> list[dict]:
    """
    Search Tavily cho dữ liệu thực tế (giá, số liệu, tin tức mới...).
    """
    try:
        response = client.search(
            query=realtime_query,
            search_depth="basic",
            max_results=3,
            include_answer=True,
        )
    except Exception as e:
        print(f"[Tavily Realtime] Lỗi: {e}")
        return []

    raw = []

    if response.get("answer"):
        raw.append({
            "mapc": None, "ten": f"Dữ liệu thực tế: {realtime_query}",
            "noidung": response["answer"],
            "tenchuong": None, "tendemuc": None, "tenchude": None,
            "passage": f"[Dữ liệu thực tế]\n{response['answer']}",
            "score": 0.95, "url": "",
            "trust_score": 0.6, "trust_level": "medium",
            "source_label": "[Dữ liệu thực tế - tham khảo]",
            "source": "web_realtime",
        })

    for r in response.get("results", []):
        passage = "\n".join(filter(None, [r.get("title", ""), r.get("content", "")[:500]]))
        raw.append({
            "mapc": None, "ten": r.get("title", ""),
            "noidung": r.get("content", "")[:500],
            "tenchuong": None, "tendemuc": None, "tenchude": None,
            "passage": passage,
            "score": r.get("score", 0.5),
            "url": r.get("url", ""),
        })

    filtered = _filter_and_score(raw)
    print(f"[Tavily Realtime] {len(filtered)} kết quả sau lọc")
    return filtered