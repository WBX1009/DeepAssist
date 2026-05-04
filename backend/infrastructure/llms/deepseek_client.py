from typing import Any, Dict, Iterator, List, Optional
from openai import OpenAI

from backend.domain.interfaces.llm import BaseLLM
from backend.common.config import settings
from backend.common.logger import get_logger

logger = get_logger(__name__)

class DeepSeekClient(BaseLLM):
    """
    DeepSeek 官方大模型客户端实现类。
    完全遵守 domain/interfaces/llm.py 的契约。
    """
    def __init__(self, model_name: str = settings.LLM_CHAT_MODEL):
        """
        :param model_name: 默认使用 deepseek-chat，如果是跑 Agent 推理节点，可传入 deepseek-reasoner
        """
        self.model_name = model_name
        
        # 使用配置项初始化 OpenAI 客户端，劫持 Base URL 指向 DeepSeek
        if not settings.DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY 未配置，调用时将报错！")
            
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        logger.info(f"✅ DeepSeekClient 初始化完成，当前指定模型: {self.model_name}")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Any:
        """
        非流式对话，支持传入工具 (Tool Calling / Function Calling)
        """
        try:
            resolved_model = self._resolve_model_name(model_name)
            if tools and resolved_model == settings.LLM_REASONER_MODEL:
                logger.warning("DeepSeek reasoner does not support tool calls; falling back to %s", settings.LLM_CHAT_MODEL)
                resolved_model = settings.LLM_CHAT_MODEL
            kwargs = {
                "model": resolved_model,
                "messages": messages,
                "temperature": self._resolve_temperature(temperature),
            }
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
            # 如果存在工具，则绑定工具（注：deepseek-reasoner 暂不支持 tool_calls，这里由业务层控制）
            if tools:
                kwargs["tools"] = tools
                
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message
            
        except Exception as e:
            logger.error(f"DeepSeek 接口调用失败: {str(e)}")
            raise e

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        try:
            kwargs = {
                "model": self._resolve_model_name(model_name),
                "messages": messages,
                "stream": True,
                "temperature": self._resolve_temperature(temperature),
            }
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
            response = self.client.chat.completions.create(**kwargs)
            
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
                
        except Exception as e:
            logger.error(f"DeepSeek 流式输出失败: {str(e)}")
            raise e

    def _resolve_model_name(self, model_name: Optional[str]) -> str:
        requested = (model_name or self.model_name or "").strip()
        normalized = requested.lower()
        if normalized in {"deepseek-chat", "deepseek chat", "chat"}:
            return settings.LLM_CHAT_MODEL
        if normalized in {"deepseek-reasoner", "deepseek reasoner", "reasoner"}:
            return settings.LLM_REASONER_MODEL
        if "glm" in normalized or "智谱" in requested:
            logger.warning("Zhipu model [%s] is not configured; falling back to %s", requested, self.model_name)
            return self.model_name
        return requested or self.model_name

    def _resolve_temperature(self, temperature: Optional[float]) -> float:
        if temperature is None:
            return 0.7
        return max(0.0, min(2.0, float(temperature)))
