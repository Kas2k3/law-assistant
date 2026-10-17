from typing import Any, Dict, List, Optional
from .quick_agent import QuickAgent
from .spam_agent import SpamAgent
from .sentiment_agent import SentimentAgent

class AgentsManager:
    """Quản lý và điều phối các agent pre-retrieve."""

    def __init__(self):
        self.agents = [
            QuickAgent(),
            SpamAgent(),
            SentimentAgent(),
        ]

    def analyze(self, prompt: str, history: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Chạy toàn bộ agent và quyết định action."""
        results = {}

        for agent in self.agents:
            result = agent.run(prompt, history)
            results[agent.name] = result

            if agent.name == "quick" and result.get("value"):
                return {
                    "action": "quick",
                    "reason": "quick_match",
                    "answer_text": result.get("answer", ""),
                    "results": results,
                }

            if agent.name == "spam" and result.get("is_spam"):
                reasons = result.get("reasons") or []
                # Join trả về chuỗi, ban đầu đang là list
                if isinstance(reasons, (list, tuple)):
                    reason = ", ".join(map(str, reasons))
                else:
                    reason = str(reasons)

                return {
                    "action": "spam",
                    "reason": reason,
                    "results": results,
                }

            if agent.name == "sentiment" and result.get("is_negative"):
                labels = result.get("sentiment_type") or []
                # Đảm bảo là list chuỗi và bỏ phần tử rỗng/None
                labels = [str(x).strip() for x in labels if x]
                reason = ", ".join(labels)

                if result.get("is_findinfo"):
                    return {
                        "action": "proceed",
                        "sentiment": reason,
                        "results": results
                    }
                else:
                    return {
                        "action": "escalate",
                        "reason": reason,  # ví dụ: "anger, frustration"
                        "results": results,
                    }
                
            if agent.name == "sentiment" and not result.get("is_negative"):
                labels = result.get("sentiment_type") or []
                # Đảm bảo là list chuỗi và bỏ phần tử rỗng/None
                labels = [str(x).strip() for x in labels if x]
                reason = ", ".join(labels)

                if result.get("is_findinfo"):
                    return {
                        "action": "proceed", 
                        "sentiment": reason, 
                        "results": results}
                # else:
                #     return {
                #         "action": "quick",
                #         "reason": "no_need_info",
                #         "answer_text": "Nếu bạn cần thêm thông tin gì mình sẵn sàng giúp bạn giải đáp nha!",
                #         "results": results,
                #     }
                
        return {"action": "proceed", "sentiment": "neutral", "results": results}
