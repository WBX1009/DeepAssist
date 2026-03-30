from typing import Iterator
from backend.services.agent.engine import AgentEngine
from backend.services.session.manager import SessionManager
from backend.services.streaming.sse_manager import SSEManager
from backend.services.agent.prompt import PromptManager
from backend.core.logger import get_logger

logger = get_logger(__name__)

class AgentApplication:
    def __init__(self, agent_engine: AgentEngine, session_manager: SessionManager):
        self.engine = agent_engine
        self.session_mgr = session_manager

    def stream_agent_task(self, session_id: str, query: str, use_user_memory: bool = False) -> Iterator[str]:
        try:
            logger.info(f"🤖 进入 Agent 模式：多步推理 [{session_id}]")
            
            messages = self.session_mgr.get_chat_context(session_id, max_rounds=10)
            system_prompt = PromptManager.AGENT_SYSTEM_PROMPT
            
            if use_user_memory:
                user_profile = "\n[用户长期记忆/偏好]: 熟悉 Python 和系统架构，请用专业术语回答，尽量给出代码示例。"
                system_prompt += user_profile
                
            messages.insert(0, {"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})
            
            final_answer_accumulated = ""
            
            # ⚡ 消费 Engine 产生的流式事件
            for event in self.engine.stream_run(messages):
                event_type = event.get("type")
                
                if event_type == "status":
                    # 轮次状态
                    yield SSEManager.format_chunk(f"\n> *{event['content']}*\n")
                    
                elif event_type == "reasoning":
                    # DeepSeek 独有的思考过程
                    yield SSEManager.format_chunk(f"\n```thought\n{event['content']}\n```\n")
                    
                elif event_type == "tool_call":
                    # 工具调用
                    name = event.get("name")
                    args = event.get("args")
                    yield SSEManager.format_chunk(f"\n🛠️ **调用工具**: `{name}`\n> 参数: `{args}`\n\n")
                    
                elif event_type == "tool_result":
                    # 截断部分过长的工具结果，避免刷屏
                    obs = event.get("content", "")[:200].replace('\n', ' ')
                    yield SSEManager.format_chunk(f"✅ **观察结果**: {obs}...\n")
                    
                elif event_type == "final_answer":
                    # 最终的回答
                    final_answer_accumulated = event.get("content", "")
                    chunk_size = 10
                    yield SSEManager.format_chunk("\n💡 **最终结论**:\n")
                    for i in range(0, len(final_answer_accumulated), chunk_size):
                        yield SSEManager.format_chunk(final_answer_accumulated[i:i+chunk_size])
                        
                elif event_type == "error":
                    yield SSEManager.format_error(event.get("content"))                 

                elif event_type == "finish":
                    # 🚀 【新增】：捕获引擎吐出的“全量中间消息”，完美落库！
                    new_messages = event.get("new_messages",[])
                    
                    # 拼装完整的一轮完整交互：用户的提问 + Agent产生的所有中间步骤和最终回复
                    full_interaction = [{"role": "user", "content": query}] + new_messages
                    
                    self.session_mgr.add_messages(session_id, full_interaction)
                    logger.info(f"✅ 成功将 Agent 多轮推理轨迹 ({len(full_interaction)} 条记录) 存入数据库。")
  
        except Exception as e:
            logger.error(f"Agent 运行异常: {e}")
            yield SSEManager.format_error(str(e))
            # 落库保存
            self.session_mgr.save_interaction(session_id, query, final_answer_accumulated)
            yield SSEManager.format_end()