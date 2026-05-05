from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class StreamEvent(BaseModel):
    """Transport-neutral event emitted by application workflows."""

    event: str = Field(..., description="State machine event name")
    content: Optional[str] = Field(default=None, description="Text payload for deltas or results")
    message: Optional[str] = Field(default=None, description="Human-readable status or error message")
    name: Optional[str] = Field(default=None, description="Tool or step name")
    args: Optional[Any] = Field(default=None, description="Tool call arguments")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional structured payload")

    def to_payload(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    @classmethod
    def message_delta(cls, content: str) -> "StreamEvent":
        return cls(event="message_delta", content=content)

    @classmethod
    def status(cls, message: str) -> "StreamEvent":
        return cls(event="status", message=message)

    @classmethod
    def reasoning(cls, content: str) -> "StreamEvent":
        return cls(event="reasoning", content=content)

    @classmethod
    def tool_call(
        cls,
        name: str,
        args: Any,
        tool_call_id: Optional[str] = None,
    ) -> "StreamEvent":
        data = {}
        if tool_call_id:
            data["tool_call_id"] = tool_call_id
        return cls(event="tool_call", name=name, args=args, data=data)

    @classmethod
    def tool_result(
        cls,
        name: str,
        content: str,
        tool_call_id: Optional[str] = None,
        success: Optional[bool] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "StreamEvent":
        data: Dict[str, Any] = {}
        if tool_call_id:
            data["tool_call_id"] = tool_call_id
        if success is not None:
            data["success"] = success
        if error:
            data["error"] = error
        if metadata:
            data["metadata"] = metadata
        return cls(event="tool_result", name=name, content=content, data=data)

    @classmethod
    def retrieval_trace(cls, data: Dict[str, Any]) -> "StreamEvent":
        return cls(event="retrieval_trace", data=data)

    @classmethod
    def citation_trace(cls, data: Dict[str, Any]) -> "StreamEvent":
        return cls(event="citation_trace", data=data)

    @classmethod
    def context_window_trace(cls, data: Dict[str, Any]) -> "StreamEvent":
        return cls(event="context_window_trace", data=data)

    @classmethod
    def answer_guard(cls, data: Dict[str, Any]) -> "StreamEvent":
        return cls(event="answer_guard", data=data)

    @classmethod
    def plan_assessment(cls, data: Dict[str, Any]) -> "StreamEvent":
        return cls(event="plan_assessment", data=data)

    @classmethod
    def failure_recovery(cls, message: str, data: Dict[str, Any]) -> "StreamEvent":
        return cls(event="failure_recovery", message=message, data=data)

    @classmethod
    def self_correction(cls, message: str, data: Dict[str, Any]) -> "StreamEvent":
        return cls(event="self_correction", message=message, data=data)

    @classmethod
    def final_answer(cls, content: str) -> "StreamEvent":
        return cls(event="final_answer", content=content)

    @classmethod
    def error(cls, message: str) -> "StreamEvent":
        return cls(event="error", message=message)

    @classmethod
    def done(cls) -> "StreamEvent":
        return cls(event="done")
