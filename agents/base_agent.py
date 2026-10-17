from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseAgent(ABC):
    """Lớp cơ sở cho mọi agent trong pre-retrieve."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, prompt: str, history: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Trả về kết quả phân tích dạng dict."""
        pass
