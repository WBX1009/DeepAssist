from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api.dependencies import get_agent_app
from backend.application.agent_app import AgentApplication

router = APIRouter(prefix="/api/agent", tags=["Agent"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class AgentRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    use_user_memory: bool = False
    model_name: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    history_rounds: int = Field(default=10, ge=1, le=50)


@router.post("/stream")
async def stream_agent(
    request: AgentRequest,
    app: AgentApplication = Depends(get_agent_app),
) -> StreamingResponse:
    generator = app.stream_agent_task(
        session_id=request.session_id,
        query=request.query,
        use_user_memory=request.use_user_memory,
        model_name=request.model_name,
        temperature=request.temperature,
        top_p=request.top_p,
        history_rounds=request.history_rounds,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
