from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable
from .base_agent import BaseAgent
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dataclasses import dataclass
import re
import sys

# ==== Regex & hằng số cốt lõi ====

# Ký tự lặp: "zzzzzz", "aaaaa", "???!!!???"
RE_LONG_REPEAT = re.compile(r"(.)\1{4,}", re.UNICODE)

# Cụm toàn phụ âm dài (Latin) -> có xu hướng vô nghĩa: "skdjfhg", "rtzpthk"
RE_CONSONANT_RUN = re.compile(r"\b[b-df-hj-np-tv-z]{6,}\b", re.IGNORECASE)

# Emoji & ký tự biểu tượng (dải Unicode phổ biến)
RE_EMOJI = re.compile(
    r"[\U0001F300-\U0001F6FF\U0001F900-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF]",
    re.UNICODE,
)

# Link cơ bản
RE_LINK = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)

# Dấu câu dày đặc
RE_PUNCT_HEAVY = re.compile(r"[!?.,@#$%^&*()_+\-=]{6,}")

# Token chữ cái (Latin + tiếng Việt có dấu)
RE_LETTERS = re.compile(r"[A-Za-zÀ-ỹ]", re.UNICODE)

# Token từ Latin/Việt để đếm nguyên âm
RE_WORD = re.compile(r"[A-Za-zÀ-ỹ]+", re.UNICODE)

# Nguyên âm (Latin + tiếng Việt)
VOWELS = "aeiouyàáạảãăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơớờởỡợùúụủũưứừửữựỳýỵỷỹ"


OFFENSIVE_WORDS = {
    # thêm/bớt tùy môi trường của bạn
    "dm",
    "đm",
    "ngu",
    "óc chó",
    "fuck",
    "shit",
}

'''
“Prompt jailbreak” là các kỹ thuật hoặc chuỗi lệnh mà người dùng cố ý nhập vào để:

- Bắt mô hình bỏ qua hướng dẫn hoặc chính sách an toàn,

- Thay đổi hành vi (ví dụ như buộc mô hình “đóng vai” khác hoặc “chạy mã nguy hiểm”),

- Tiết lộ thông tin nội bộ (như prompt ẩn, dữ liệu cấu hình).
'''

JAILBREAK_PHRASES = {
    "ignore all previous instructions",
    "disable safety",
    "run shell command",
    "execute python",
    "prompt injection",
}

def _build_patterns(words: Iterable[str]) -> list[re.Pattern]:
    """
    Tạo regex an toàn cho từng từ/cụm:
    - Đơn từ: dùng \b...\b (ranh giới từ).
    - Cụm nhiều từ: thay ' ' -> \s+ (1+ khoảng trắng).
    - Viết tắt 2 ký tự kiểu 'dm' / 'đm': cho phép 0–1 khoảng trắng giữa 2 ký tự, vẫn giữ \b hai bên.
    Lưu ý: dùng re.IGNORECASE và re.UNICODE để hỗ trợ tiếng Việt.
    """
    patterns = []
    for w in words:
        w_norm = w.strip()
        if not w_norm:
            continue

        # Cụm nhiều từ (vd: "óc chó", "ignore previous")
        if " " in w_norm:
            # "óc chó" -> r"\bóc\s+chó\b"
            part = r"\s+".join(map(re.escape, w_norm.split()))
            pat = rf"\b{part}\b"

        # Viết tắt 2 ký tự (vd: "dm", "đm") -> cho phép 0 hoặc 1 space ở giữa
        elif len(w_norm) == 2 and w_norm.isalpha():
            # "dm" -> r"\bd\s{0,1}m\b"
            pat = rf"\b{re.escape(w_norm[0])}\s{{0,1}}{re.escape(w_norm[1])}\b"

        else:
            # Đơn từ bình thường -> r"\bword\b"
            pat = rf"\b{re.escape(w_norm)}\b"

        patterns.append(re.compile(pat, re.IGNORECASE | re.UNICODE))
    return patterns

# Khởi tạo một lần (nếu bạn dùng OOP, hãy chuyển vào __init__)
_OFFENSIVE_PATTERNS = _build_patterns(OFFENSIVE_WORDS)
_JAILBREAK_PATTERNS = _build_patterns(JAILBREAK_PHRASES)


class SpamAgent(BaseAgent):
    """Phát hiện nội dung spam bằng model nhỏ -> Hơi chặt, đánh câu hỏi pháp lý nặng quá"""
    '''
    def __init__(self):
        super().__init__("spam")
        model_name = "visolex/bartpho-spam-binary"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.threshold = 0.99

    def run(self, prompt: str, history: Optional[List[Any]] = None) -> Dict[str, Any]:
        with torch.no_grad():
            x = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            p = torch.softmax(self.model(**x).logits, dim=-1)[0, 1].item()
        return {
            "type": "spam",
            "score": p,
            "is_spam": p >= self.threshold,
        }
    '''


    """
    Spam detector tối giản bằng rule base để test câu hỏi người dùng.
    - Không xét đến regex nội dung chính theo domain (ví dụ: pháp luật).
    - Chỉ dựa trên tín hiệu spam chung: lặp ký tự, vô nghĩa, link, xúc phạm, jailbreak, emoji/dấu câu quá nhiều, lặp lại.
    """
    def __init__(
            self,
            low_alpha_ratio_thresh: float = 0.4,
            low_vowel_token_frac_thresh: float = 0.5,
            max_repeat_recent: int = 3,
        ):
            super().__init__("spam")
            self.low_alpha_ratio_thresh = low_alpha_ratio_thresh
            self.low_vowel_token_frac_thresh = low_vowel_token_frac_thresh
            self.max_repeat_recent = max_repeat_recent

    # ===== Helper =====
    @staticmethod
    def normalize(text: str) -> str:
        return (text or "").strip().lower()

    def has_long_repeat(self, text: str) -> bool:
        return bool(RE_LONG_REPEAT.search(text))

    def has_link(self, text: str) -> bool:
        return bool(RE_LINK.search(text))

    def heavy_punct(self, text: str) -> bool:
        return bool(RE_PUNCT_HEAVY.search(text))

    def many_emojis(self, text: str, min_count: int = 4) -> bool:
        return len(RE_EMOJI.findall(text)) >= min_count

    def low_alpha_ratio(self, text: str) -> bool:
        letters = RE_LETTERS.findall(text)
        return (len(letters) / max(len(text), 1)) < self.low_alpha_ratio_thresh

    def has_consonant_run(self, text: str) -> bool:
        return bool(RE_CONSONANT_RUN.search(text))

    def too_many_low_vowel_tokens(self, text: str) -> bool:
        tokens = RE_WORD.findall(text)
        if not tokens:
            return False
        bad = 0
        for w in tokens:
            # token dài >=6 mà có <=1 nguyên âm => nghi vô nghĩa
            v = re.findall(f"[{VOWELS}]", w, re.IGNORECASE)
            if len(w) >= 6 and len(v) <= 1:
                bad += 1
        return (bad / len(tokens)) >= self.low_vowel_token_frac_thresh

    def offensive_or_jailbreak(self, text: str) -> Optional[str]:
        t = self.normalize(text)
        # for w in OFFENSIVE_WORDS:
        #     if w in t:
        #         return "offensive"
        # for p in JAILBREAK_PHRASES:
        #     if p in t:
        #         return "jailbreak"
        # return None
        # Offensive trước
        for p in _OFFENSIVE_PATTERNS:
            m = p.search(t)
            if m:
                # Debug nếu cần:
                # print("Matched offensive:", p.pattern, "at", m.span(), "=>", m.group())
                return "offensive"
        # Jailbreak sau
        for p in _JAILBREAK_PATTERNS:
            m = p.search(t)
            if m:
                # print("Matched jailbreak:", p.pattern, "at", m.span(), "=>", m.group())
                return "jailbreak"
        return None

    def all_symbols(self, text: str) -> bool:
        # hầu hết là ký tự đặc biệt/dấu câu
        letters = RE_LETTERS.findall(text)
        return len(letters) <= 1 and any(ch for ch in text if not ch.isalnum())

    # def repeated_message(self, text: str, history: List[str]) -> bool:
    #     if not history:
    #         return False
    #     t = self.normalize(text)
    #     # kiểm tra xuất hiện trong N tin gần nhất
    #     recent = [self.normalize(x) for x in history[-self.max_repeat_recent:]]
    #     return t in recent

    def looks_meaningless(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True     # rỗng -> vô nghĩa trong bối cảnh hỏi

        if self.has_long_repeat(t):
            return True
        if self.has_consonant_run(t):
            return True
        if self.low_alpha_ratio(t):
            return True
        if self.too_many_low_vowel_tokens(t):
            return True
        if self.all_symbols(t):
            return True
        return False

    def run(self, prompt: str, history: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Phát hiện spam, giữ nguyên output format."""
        history = history or []
        reasons: List[str] = []
        details: Dict[str, Any] = {}

        # 1) Vô nghĩa/ngẫu nhiên
        if self.looks_meaningless(prompt):
            reasons.append("meaningless_text")
            details["meaningless"] = True

        # 2) Lặp ký tự
        if self.has_long_repeat(prompt):
            reasons.append("long_repeat")
            details["long_repeat"] = True

        # 3) Link/quảng cáo
        if self.has_link(prompt):
            reasons.append("contains_link")
            details["contains_link"] = True

        # 4) Dấu câu/emoji dày đặc
        if self.heavy_punct(prompt):
            reasons.append("heavy_punct")
            details["heavy_punct"] = True

        if self.many_emojis(prompt):
            reasons.append("many_emojis")
            details["many_emojis"] = True

        # 5) Xúc phạm / jailbreak
        abuse = self.offensive_or_jailbreak(prompt)
        if abuse:
            reasons.append(abuse)
            details[abuse] = True

        # 6) Lặp lại nhiều lần cùng nội dung
        # if self.repeated_message(prompt, history):
        #     reasons.append("repeated_message")
        #     details["repeated_message"] = True

        # Tính điểm thô (số trigger)
        score = len(set(reasons))
        is_spam = score > 0

        # Giữ nguyên cấu trúc output
        return {
            "type": "spam",
            "is_spam": is_spam,
            "score": score,
            "reasons": sorted(set(reasons)),
            "details": details,
        }
