from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api.dependencies import get_chat_app
from backend.application.chat_app import ChatApplication

router = APIRouter(prefix="/api/chat", tags=["Chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    mode: str = Field(default="quick", description="quick, rag, or auto")
    collection_name: str = Field(default="tech_docs_kb")
    model_name: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    history_rounds: int = Field(default=10, ge=1, le=50)
    use_user_memory: bool = False


@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    app: ChatApplication = Depends(get_chat_app),
) -> StreamingResponse:
    generator = app.stream_chat(
        session_id=request.session_id,
        query=request.query,
        mode=request.mode,
        collection_name=request.collection_name,
        model_name=request.model_name,
        temperature=request.temperature,
        top_p=request.top_p,
        history_rounds=request.history_rounds,
        use_user_memory=request.use_user_memory,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/sessions")
def list_sessions(app: ChatApplication = Depends(get_chat_app)) -> dict:
    return {"status": "success", "data": app.list_sessions()}


@router.get("/history/{session_id}")
def get_history(
    session_id: str,
    app: ChatApplication = Depends(get_chat_app),
) -> dict:
    return {"status": "success", "data": app.get_history(session_id, max_rounds=50)}


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    app: ChatApplication = Depends(get_chat_app),
) -> dict:
    if app.delete_session(session_id):
        return {"status": "success", "message": f"Session {session_id} deleted."}
    return {"status": "error", "message": "Failed to delete session."}
