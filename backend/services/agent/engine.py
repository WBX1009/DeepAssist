import json
import inspect
from typing import List, Dict, Any, Callable, Iterator
from backend.domain.interfaces.llm import BaseLLM
from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

class AgentEngine:
    """
    原生解耦版 Agent 中枢循环引擎。
    已修复：大模型在调用工具前的中间发言被静默吞噬的问题。
    """
    def __init__(self, llm: BaseLLM, tools: List[Callable]):
        self.llm = llm
        self.tools_map = {tool.__name__: tool for tool in tools}
        self.openai_tools = self._build_tools_schema(tools)

    def _build_tools_schema(self, tools: List[Callable]) -> List[Dict[str, Any]]:
        schemas =[]
        for tool in tools:
            sig = inspect.signature(tool)
            properties = {}
            required =[]
            
            for name, param in sig.parameters.items():
                # 根据类型提示(Type Hint)动态推断 JSON Schema 类型
                param_type = "string"
                if param.annotation == int: param_type = "integer"
                elif param.annotation == float: param_type = "number"
                elif param.annotation == bool: param_type = "boolean"
                
                properties[name] = {"type": param_type, "description": f"参数 {name}"}
                
                # 如果没有默认值，说明是必填项
                if param.default == inspect.Parameter.empty:
                    required.append(name)
                    
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": tool.__doc__ or f"执行 {tool.__name__} 工具",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        return schemas

    def stream_run(self, messages: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        current_messages = messages.copy()
        step = 0
        
        while step < settings.MAX_AGENT_STEPS:
            step += 1
            yield {"type": "status", "content": f"🔄 开启第 {step} 轮推理..."}
            
            response_msg = self.llm.chat(messages=current_messages, tools=self.openai_tools)
            safe_content = response_msg.content or ""
            
            if getattr(response_msg, "reasoning_content", None):
                yield {"type": "reasoning", "content": response_msg.reasoning_content}
                
            msg_dict = {"role": "assistant", "content": safe_content}
            
            if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
                msg_dict["tool_calls"] =[
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in response_msg.tool_calls
                ]
            current_messages.append(msg_dict)
            
            # ==========================================
            # 🟢 修复漏洞 2：防止 LLM 决定调用工具前的发言被吞噬
            # ==========================================
            if msg_dict.get("tool_calls"):
                if safe_content.strip():
                    # LLM 在调用工具前说了一句话，必须抛给前端展示
                    yield {"type": "status", "content": f"🗣️ {safe_content}"}
            else:
                # 没有任何工具调用，这是最终结论！
                yield {"type": "final_answer", "content": safe_content}
                new_msgs = current_messages[len(messages):]
                yield {"type": "finish", "new_messages": new_msgs}
                return
                
            # 执行工具逻辑 (Execute Phase)
            for tool_call in msg_dict["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args_str = tool_call["function"]["arguments"]
                tool_id = tool_call["id"]
                
                yield {"type": "tool_call", "name": tool_name, "args": tool_args_str}
                
                try:
                    args = json.loads(tool_args_str)
                    target_func = self.tools_map.get(tool_name)
                    observation = target_func(**args) if target_func else f"错误: 找不到可用工具 {tool_name}"
                except Exception as e:
                    observation = f"工具执行异常: {str(e)}"
                    logger.error(observation)
                    
                yield {"type": "tool_result", "name": tool_name, "content": str(observation)}
                
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": str(observation)
                })
                
        yield {"type": "error", "content": f"达到最大步数 ({settings.MAX_AGENT_STEPS}) 限制，强制终止。"}
        new_msgs = current_messages[len(messages):]
        yield {"type": "finish", "new_messages": new_msgs}