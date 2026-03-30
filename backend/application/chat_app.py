from typing import Iterator
from backend.domain.interfaces.llm import BaseLLM
from backend.services.session.manager import SessionManager
from backend.services.rag.fusion import HybridRetriever
from backend.services.streaming.sse_manager import SSEManager
from backend.services.agent.prompt import PromptManager
from backend.core.logger import get_logger

logger = get_logger(__name__)

class ChatApplication:
    """
    对话应用编排器。
    严格执行显式路由策略：
    - Quick (快速开始): 纯 LLM，0记忆，不落库。
    - RAG (知识问答): 混合检索，极短记忆(仅用于指代消解)，严格落库。
    """
    def __init__(self, llm: BaseLLM, session_manager: SessionManager, retriever: HybridRetriever = None):
        self.llm = llm
        self.session_mgr = session_manager
        self.retriever = retriever

    def _build_rag_prompt(self, query: str, collection_name: str) -> str:
        """RAG 模式专属：去检索引擎捞数据，并拼接严格的 Prompt"""
        if not self.retriever:
            logger.warning("未注入 Retriever，RAG 模式退化为普通提问")
            return query
            
        docs = self.retriever.retrieve(collection_name, query)
        if not docs:
            return query  # 没查到相关片段
            
        # 组装背景知识
        context_str = "\n\n".join([f"【参考片段 {i+1}】\n{d.content}" for i, d in enumerate(docs)])
        
        # 将用户问题与检索到的上下文融合
        prompt = (
            f"=== 参考资料 ===\n{context_str}\n\n"
            f"=== 用户问题 ===\n{query}"
        )
        logger.info(f"RAG Prompt 构建完成，包含 {len(docs)} 个参考片段")
        return prompt

    def stream_chat(self, session_id: str, query: str, mode: str, collection_name: str = "tech_docs_kb") -> Iterator[str]:
        """
        核心流式生成器。
        根据 mode (quick / rag) 执行完全不同的策略。
        """
        try:
            # ==========================================
            # ⚡ 策略 1：快速开始 (Quick Mode)
            # ==========================================
            if mode == "quick":
                logger.info("🚀 进入 Quick 模式：纯LLM，无记忆，无检索")
                messages =[
                    {"role": "system", "content": PromptManager.CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ]
                
                # 直接调用大模型，生成流式响应
                for chunk in self.llm.chat_stream(messages):
                    yield SSEManager.format_chunk(chunk)
                yield SSEManager.format_end()
                
                # 🚫 核心约束：绝对不写入数据库，无痕退出！
                return

            # ==========================================
            # 📚 策略 2：知识问答 (RAG Mode)
            # ==========================================
            elif mode == "rag":
                logger.info(f"📚 进入 RAG 模式：知识库检索，极短记忆[{session_id}]")
                
                # 🛡️ 【关键拦截】：如果底层组件缺失，直接拒绝服务并明确提示
                if self.retriever is None:
                    error_msg = "系统提示：核心检索组件（如向量模型或数据库）未正确加载。请联系管理员检查服务器配置后重试。"
                    logger.error(f"拦截 RAG 请求：{error_msg}")
                    yield SSEManager.format_error(error_msg)
                    return
                
                messages = self.session_mgr.get_chat_context(session_id, max_rounds=2)
                messages.insert(0, {"role": "system", "content": PromptManager.RAG_SYSTEM_PROMPT})
                
                final_query = self._build_rag_prompt(query, collection_name)
                messages.append({"role": "user", "content": final_query})
                
                full_response = ""
                for chunk in self.llm.chat_stream(messages):
                    full_response += chunk
                    yield SSEManager.format_chunk(chunk)
                    
                self.session_mgr.save_interaction(session_id, query, full_response)
                yield SSEManager.format_end()
                
            else:
                yield SSEManager.format_error(f"未知的运行模式: {mode}")

        except Exception as e:
            logger.error(f"流式对话异常: {e}")
            yield SSEManager.format_error(str(e))