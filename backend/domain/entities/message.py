from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Message(BaseModel):
    """
    标准的对话消息结构。
    这个模型将用于 API 的请求/响应体以及内部会话管理。
    """
    role: str = Field(..., description="消息发送者角色: 'user', 'assistant', 'system', 'tool'")
    content: str = Field(..., description="消息的正文内容")
    name: Optional[str] = Field(default=None, description="当 role 是 'tool' 时，表示工具的名称")
    tool_call_id: Optional[str] = Field(default=None, description="当 role 是 'tool' 时，对应工具调用的唯一ID")

class ChatSession(BaseModel):
    """
    一个完整的会话实体，用于前端展示历史聊天列表。
    """
    session_id: str
    title: str = Field(default="新聊天", description="会话标题（可以由第一句用户提问生成）")
    messages: List[Message] = Field(default_factory=list, description="会话中的所有消息")
    
class ToolCall(BaseModel):
    """用于表示大模型请求调用工具的结构"""
    id: str
    type: str = "function"
    function: Dict[str, Any]

class AIMessage(Message):
    """扩展标准Message，以支持大模型的工具调用输出"""
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="大模型请求的工具调用列表")