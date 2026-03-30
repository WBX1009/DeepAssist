from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.application.agent_app import AgentApplication
from backend.api.dependencies import get_agent_app

router = APIRouter(prefix="/api/agent", tags=["Agent"])

class AgentRequest(BaseModel):
    session_id: str
    query: str
    use_user_memory: bool = False  # 是否激活长期画像记忆

@router.post("/stream")
async def stream_agent_endpoint(req: AgentRequest, app: AgentApplication = Depends(get_agent_app)):
    """处理复杂智能体工具调用的流式请求"""
    
    generator = app.stream_agent_task(
        session_id=req.session_id,
        query=req.query,
        use_user_memory=req.use_user_memory
    )
    
    return StreamingResponse(generator, media_type="text/event-stream")