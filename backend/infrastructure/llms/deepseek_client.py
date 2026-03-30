import os
from typing import List, Dict, Any, Iterator
from openai import OpenAI

from backend.domain.interfaces.llm import BaseLLM
from backend.core.config import settings
from backend.core.logger import get_logger

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

    def chat(self, messages: List[Dict[str, Any]], tools: List[Any] = None) -> Any:
        """
        非流式对话，支持传入工具 (Tool Calling / Function Calling)
        """
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
            }
            # 如果存在工具，则绑定工具（注：deepseek-reasoner 暂不支持 tool_calls，这里由业务层控制）
            if tools:
                kwargs["tools"] = tools
                
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message
            
        except Exception as e:
            logger.error(f"DeepSeek 接口调用失败: {str(e)}")
            raise e

    def chat_stream(self, messages: List[Dict[str, Any]]) -> Iterator[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, stream=True, temperature=0.7
            )
            
            is_thinking = False
            for chunk in response:
                delta = chunk.choices[0].delta
                
                # 处理思维链内容 (Reasoner专属)
                reasoning = getattr(delta, 'reasoning_content', None)
                if reasoning:
                    if not is_thinking:
                        yield "\n```thought\n"  # 思考开始标记 (前端可以用 markdown 渲染成折叠框)
                        is_thinking = True
                    yield reasoning
                
                # 处理正式回复内容
                if delta.content:
                    if is_thinking:
                        yield "\n```\n\n"  # 思考结束标记
                        is_thinking = False
                    yield delta.content
                    
            if is_thinking: # 兜底，防止流意外中断没有闭合标记
                yield "\n```\n\n"
                
        except Exception as e:
            logger.error(f"DeepSeek 流式输出失败: {str(e)}")
            yield f"\n[模型响应中断: {str(e)}]"