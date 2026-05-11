import json
from functools import lru_cache
from typing import Iterable, Optional, Set

from backend.common.logger import get_logger
from backend.domain.entities.intent import IntentDecision, IntentType
from backend.domain.interfaces.llm import BaseLLM

logger = get_logger(__name__)

class IntentRouter:
    """LLM-based deterministic intent router for workflow selection."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    @lru_cache(maxsize=32)
    def _analyze_intent(self, query: str) -> dict:
        """🌟 核心改造：统一使用 LLM 进行意图识别，加 lru_cache 防止同一次请求被多次重复分析"""
        default_result = {
            "intent": "chat",
            "confidence": 0.4,
            "reason": "fallback to default chat",
            "is_tool_inventory": False,
            "is_kb_catalog": False
        }
        if not query.strip():
            return default_result

        system_prompt = (
            "你是一个底层的意图路由引擎。你的任务是分析用户的输入，并将其精确分类。\n"
            "请输出纯JSON对象，必须包含以下5个字段：\n"
            "1. intent (字符串): 必须是 'chat', 'rag', 'agent' 之一。\n"
            "   - agent: 用户指令需要调用外部工具、执行代码、操作文件、查天气、查数据库等复杂动作。\n"
            "   - rag: 用户明确需要检索知识库、文档、参考资料以获取长文本支撑的专业问答。\n"
            "   - chat: 常规问候、闲聊、基础常识，无需外部检索和工具。\n"
            "   注意：如果用户明确指定“不要查资料”或包含拒绝查库的负面指令，绝对不能输出 'rag'。\n"
            "2. confidence (浮点数): 0.0 到 1.0 之间的置信度。\n"
            "3. reason (字符串): 做出该判断的简短理由。\n"
            "4. is_tool_inventory (布尔值): 用户是否在专门询问“系统具备哪些工具/能调用什么工具”。\n"
            "5. is_kb_catalog (布尔值): 用户是否在专门询问“系统连接了哪些知识库/文档列表/集合状态”。\n\n"
            "请直接返回 JSON 格式，不要包含 ```json 等 Markdown 标记及任何其他说明文字。"
        )

        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        try:
            # 使用 temperature=0.0 保证分类的确定性和稳定性
            response = self.llm.chat(messages, temperature=0.0)
            content = response.content.strip()

            # 清洗大模型可能附带的 Markdown 标记
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            result = json.loads(content)
            
            # 安全兜底校验
            if result.get("intent") not in {"chat", "rag", "agent"}:
                result["intent"] = "chat"
                
            return result
        except Exception as exc:
            logger.error(f"LLM intent routing failed: {exc}")
            return default_result

    def route(
        self,
        query: str,
        allowed_intents: Optional[Iterable[IntentType]] = None,
    ) -> IntentDecision:
        allowed: Set[IntentType] = set(allowed_intents or IntentType)
        
        # 将会命中同一次请求的 lru_cache
        analysis = self._analyze_intent(query)
        
        intent_str = analysis.get("intent", "chat")
        intent_val = IntentType.CHAT
        if intent_str == "rag":
            intent_val = IntentType.RAG
        elif intent_str == "agent":
            intent_val = IntentType.AGENT
            
        decision = IntentDecision(
            intent=intent_val,
            confidence=float(analysis.get("confidence", 0.6)),
            reason=str(analysis.get("reason", "LLM based intent routing")),
            signals=[f"llm_routed_{intent_str}"]
        )

        return self._coerce(decision, allowed)

    def is_tool_inventory_query(self, query: str) -> bool:
        analysis = self._analyze_intent(query)
        return bool(analysis.get("is_tool_inventory", False))

    def is_kb_catalog_query(self, query: str) -> bool:
        analysis = self._analyze_intent(query)
        return bool(analysis.get("is_kb_catalog", False))

    def _coerce(self, decision: IntentDecision, allowed: Set[IntentType]) -> IntentDecision:
        if decision.intent in allowed:
            return decision

        fallback = IntentType.CHAT if IntentType.CHAT in allowed else next(iter(allowed))
        return IntentDecision(
            intent=fallback,
            confidence=max(0.3, decision.confidence - 0.3),
            reason=f"{decision.reason}; coerced to {fallback.value}",
            signals=decision.signals,
        )