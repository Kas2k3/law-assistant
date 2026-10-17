from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Literal

class Action(str, Enum):
    PROCEED = "proceed"          # đi tiếp luồng retrieve cũ
    QUICK_ANSWER = "quick_answer" # trả lời ngay, không retrieve
    SPAM = "spam"                # chặn
    ESCALATE = "escalate"        # chuyển human

@dataclass
class Decision:
    action: Action
    reason: str = ""
    answer_text: Optional[str] = None
    normalized_prompt: Optional[str] = None
    sentiment: Optional[Dict[str, Any]] = None
