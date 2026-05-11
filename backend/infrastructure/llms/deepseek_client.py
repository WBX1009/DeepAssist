from typing import Any, Dict, Iterator, List, Optional
from openai import OpenAI

from backend.domain.interfaces.llm import BaseLLM
from backend.common.config import settings
from backend.common.logger import get_logger

logger = get_logger(__name__)

class DeepSeekClient(BaseLLM):
    def __init__(self, model_name: str = settings.LLM_CHAT_MODEL):
        self.model_name = model_name
        if not settings.DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY 未配置，调用时将报错！")

        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        logger.info(f"✅ DeepSeekClient 初始化完成，当前指定模型: {self.model_name}")

    def chat(self, messages, tools=None, model_name=None, temperature=None, top_p=None) -> Any:
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
            if tools:
                kwargs["tools"] = tools

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message
        except Exception as e:
            logger.error(f"DeepSeek 接口调用失败: {str(e)}")
            raise e

    def chat_stream(self, messages, model_name=None, temperature=None, top_p=None) -> Iterator[Any]:
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
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                
                # 🚀 修复点：拦截并吐出深度思考过程
                if getattr(delta, 'reasoning_content', None):
                    yield {"type": "reasoning", "content": delta.reasoning_content}
                
                if getattr(delta, 'content', None):
                    yield {"type": "content", "content": delta.content}

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
            return self.model_name
        return requested or self.model_name

    def _resolve_temperature(self, temperature: Optional[float]) -> float:
        if temperature is None:
            return 0.7
        return max(0.0, min(2.0, float(temperature)))