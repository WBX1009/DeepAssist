from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

class BaseLLM(ABC):
    """
    大语言模型抽象基类。
    未来无论是换 OpenAI、智谱还是本地 VLLM，只需实现此接口即可，上层无需改动。
    """
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Any:
        """非流式对话，支持传入工具集合"""
        pass
        
    @abstractmethod
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        """流式对话输出（用于 SSE）"""
        pass
