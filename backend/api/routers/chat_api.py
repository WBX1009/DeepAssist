from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.application.chat_app import ChatApplication
from backend.api.dependencies import get_chat_app

router = APIRouter(prefix="/api/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    session_id: str
    query: str
    mode: str  # "quick" 或 "rag"

@router.post("/stream")
async def stream_chat_endpoint(req: ChatRequest, app: ChatApplication = Depends(get_chat_app)):
    """处理快速开始和知识问答的流式请求"""
    
    # 调用应用层的流式生成器
    generator = app.stream_chat(
        session_id=req.session_id,
        query=req.query,
        mode=req.mode
    )
    
    # 封装为 Server-Sent Events 流返回给前端
    return StreamingResponse(generator, media_type="text/event-stream")

@router.get("/sessions")
def get_all_chat_sessions(app: ChatApplication = Depends(get_chat_app)):
    """获取历史会话列表"""
    sessions = app.session_mgr.list_sessions()
    return {"status": "success", "data": sessions}
    
@router.get("/history/{session_id}")
def get_chat_history(session_id: str, app: ChatApplication = Depends(get_chat_app)):
    """获取单次会话的完整历史记录（用于前端恢复界面）"""
    # 提取多一点历史用于前端渲染展示
    history = app.session_mgr.get_chat_context(session_id, max_rounds=50) 
    return {"status": "success", "data": history}