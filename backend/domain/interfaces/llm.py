from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator

class BaseLLM(ABC):
    """
    大语言模型抽象基类。
    未来无论是换 OpenAI、智谱还是本地 VLLM，只需实现此接口即可，上层无需改动。
    """
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], tools: List[Any] = None) -> Any:
        """非流式对话，支持传入工具集合"""
        pass
        
    @abstractmethod
    def chat_stream(self, messages: List[Dict[str, Any]]) -> Iterator[str]:
        """流式对话输出（用于 SSE）"""
        pass