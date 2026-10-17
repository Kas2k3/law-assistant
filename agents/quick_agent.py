from typing import Any, Dict, List, Optional
import re
from .base_agent import BaseAgent

# điều chỉnh thành so sánh cosin của embed để nâng cao chất lượng kiểm tra match
class QuickAgent(BaseAgent):
    """Nhận diện câu hỏi ngắn hoặc chào hỏi."""

    def __init__(self):
        super().__init__("quick")
        self.quick_map = {
            "xin chào": (
                "Xin chào bạn 👋! Mình là trợ lý pháp lý AI — có thể giúp bạn giải đáp các thắc mắc liên quan đến pháp luật thông qua tra cứu các quy định, nghị định, "
                "thông tư và các văn bản pháp luật Việt Nam. Bạn muốn tìm hiểu về lĩnh vực nào hôm nay?"
            ),
            "chào bạn": (
                "Chào bạn! Rất vui được hỗ trợ bạn 💬. Mình có thể giúp bạn tra cứu, tóm tắt hoặc giải thích "
                "nội dung của các văn bản pháp luật, cũng như đưa ra hướng dẫn về thủ tục hành chính cụ thể."
            ),
            "hello": (
                "Hello 👋! Mình là trợ lý pháp lý AI hỗ trợ tra cứu văn bản luật Việt Nam. "
                "Bạn có thể hỏi mình về quy định, mức xử phạt, điều kiện pháp lý hoặc hướng dẫn thủ tục nhé."
            ),
            "hi": (
                "Hi! 👋 Mình là trợ lý pháp lý hỗ trợ tìm kiếm và giải thích luật. "
                "Bạn có thể hỏi mình về các quy định, điều luật cụ thể, hoặc quy trình pháp lý mà bạn đang thắc mắc."
            ),
            "okay": (
                "Rất vui được hỗ trợ bạn. "
                "Bạn có thể hỏi mình về các quy định, điều luật cụ thể, hoặc quy trình pháp lý mà bạn đang thắc mắc."
            ),
            "ok": (
                "Rất vui được hỗ trợ bạn. "
                "Bạn có thể hỏi mình về các quy định, điều luật cụ thể, hoặc quy trình pháp lý mà bạn đang thắc mắc."
            ),
            "oke": (
                "Rất vui được hỗ trợ bạn. "
                "Bạn có thể hỏi mình về các quy định, điều luật cụ thể, hoặc quy trình pháp lý mà bạn đang thắc mắc."
            ),
            "bạn là ai": (
                "Mình là một **trợ lý pháp lý AI** được thiết kế để hỗ trợ bạn trong việc tìm hiểu và tra cứu "
                "các quy định pháp luật Việt Nam. Mình có thể tóm tắt điều luật, giải thích nội dung văn bản, "
                "hoặc cung cấp căn cứ pháp lý cho câu hỏi của bạn."
            ),
            "bạn có thể làm gì": (
                "Mình có thể giúp bạn:\n"
                "- 📜 Tra cứu văn bản pháp luật Việt Nam (Luật, Nghị định, Thông tư...)\n"
                "- 💡 Giải thích điều khoản, khái niệm hoặc quy định cụ thể\n"
                "- ⚖️ Gợi ý căn cứ pháp lý liên quan đến tình huống bạn đang hỏi\n"
                "- 🧭 Hướng dẫn quy trình, thủ tục hành chính hoặc xử phạt hành chính\n\n"
                "Bạn muốn mình bắt đầu với chủ đề nào?"
            ),
            "bạn giúp được gì": (
                "Mình có thể giúp bạn tìm kiếm các quy định, hướng dẫn hoặc mức xử phạt liên quan "
                "đến các hành vi cụ thể trong luật Việt Nam. Ngoài ra, mình cũng có thể cung cấp trích dẫn điều luật "
                "và giải thích nội dung theo cách dễ hiểu nhất."
            ),
            "giới thiệu": (
                "Mình là trợ lý pháp lý AI — được huấn luyện để hiểu và giải thích các văn bản pháp luật Việt Nam. "
                "Mục tiêu của mình là giúp người dùng dễ dàng tiếp cận thông tin pháp luật một cách nhanh chóng, "
                "chính xác và dễ hiểu 💬."
            ),
            "bắt đầu": (
                "Bạn có thể nhập câu hỏi pháp luật mà bạn đang quan tâm, ví dụ:\n"
                "- *“Hành vi chống người thi hành công vụ bị xử phạt thế nào?”*\n"
                "- *“Thủ tục đăng ký kết hôn cần giấy tờ gì?”*\n\n"
                "Mình sẽ tìm và trả lời dựa trên các văn bản pháp luật hiện hành nhé ⚖️."
            ),
            "bắt đầu như thế nào": (
                "Đơn giản thôi! Bạn chỉ cần gõ câu hỏi pháp luật mà bạn muốn tìm hiểu, "
                "mình sẽ tra cứu các văn bản liên quan và tóm tắt lại câu trả lời cho bạn."
            ),
            "cảm ơn": (
                "Không có gì đâu 😊! Hỗ trợ bạn tra cứu pháp luật là nhiệm vụ của mình. "
                "Nếu bạn cần tìm hiểu thêm điều luật khác, cứ hỏi mình nhé!"
            ),
            "tạm biệt": (
                "Tạm biệt bạn! 👋 Hy vọng những thông tin mình cung cấp đã giúp ích cho bạn. "
                "Nếu có thắc mắc khác về pháp luật, cứ quay lại hỏi mình bất cứ lúc nào nhé!"
            ),
        }
        self.max_len = 80

    def _normalize(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text
    
    def _quick_answer_lookup(self, prompt_norm: str) -> Optional[str]:
        """Khớp theo luật: chính xác hoặc bắt đầu bằng key."""
        for k, v in self.quick_map.items():
            if prompt_norm == k or prompt_norm.startswith(k):
                return v
        # loại bỏ vài pattern dạng câu hỏi pháp luật
        if re.search(r"\b(luật\s+gì|điều\s+\d+)\b", prompt_norm):
            return None
        return None    
    

    def _quick_question_check(self, prompt: str) -> bool:
        """Kiểm tra xem có phải câu hỏi nhanh không."""
        prompt_norm = self._normalize(prompt)
        if len(prompt_norm) <= self.max_len and self._quick_answer_lookup(prompt_norm):
            return True
        return False

    def _quick_answer(self, prompt: str) -> str:
        """Trả về quick answer (đã check trước)."""
        return self._quick_answer_lookup(self._normalize(prompt)) or ""

    def run(self, prompt: str, history: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Kiểm tra và trả về nếu là quick question."""
        prompt_norm = self._normalize(prompt)
        is_quick = self._quick_question_check(prompt)
        answer = self._quick_answer(prompt) if is_quick else ""
        return {
            "type": "quick",
            "value": is_quick,
            "answer": answer,
            "normalized_prompt": prompt_norm,
        }