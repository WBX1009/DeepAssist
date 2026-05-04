import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, get_origin

from backend.domain.entities.tooling import ToolCall, ToolResult


@dataclass(frozen=True)
class ToolSpec:
    """Registered tool contract exposed to the LLM."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]
    signature: inspect.Signature

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolPolicy:
    """Small governance layer for tool execution."""

    allowed_tools: Optional[Set[str]] = None
    max_result_chars: int = 4000

    def can_execute(self, tool_name: str) -> bool:
        return self.allowed_tools is None or tool_name in self.allowed_tools

    def apply_result_limits(self, result: ToolResult) -> ToolResult:
        if len(result.content) <= self.max_result_chars:
            return result

        truncated = result.content[: self.max_result_chars]
        metadata = {
            **result.metadata,
            "truncated": True,
            "original_length": len(result.content),
        }
        return result.model_copy(
            update={
                "content": f"{truncated}\n...[truncated]",
                "metadata": metadata,
            }
        )


class ToolRegistry:
    """Registry that turns Python callables into governed agent tools."""

    def __init__(self, policy: Optional[ToolPolicy] = None):
        self.policy = policy or ToolPolicy()
        self._tools: Dict[str, ToolSpec] = {}

    @classmethod
    def from_callables(
        cls,
        tools: Iterable[Callable[..., Any]],
        policy: Optional[ToolPolicy] = None,
    ) -> "ToolRegistry":
        registry = cls(policy=policy)
        for tool in tools:
            registry.register_callable(tool)
        return registry

    def register_callable(
        self,
        handler: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        tool_name = name or handler.__name__
        signature = inspect.signature(handler)
        self._tools[tool_name] = ToolSpec(
            name=tool_name,
            description=description or inspect.getdoc(handler) or f"Execute {tool_name}",
            parameters=self._build_parameters_schema(signature),
            handler=handler,
            signature=signature,
        )

    def openai_schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        spec = self._tools.get(tool_call.name)
        if not spec:
            return ToolResult.unknown_tool(tool_call, self._available_tool_names())

        if not self.policy.can_execute(tool_call.name):
            return ToolResult.execution_failed(
                tool_call,
                f"Tool is blocked by policy: {tool_call.name}",
            ).model_copy(update={"metadata": {"error_type": "tool_blocked"}})

        validation_error = self._validate_arguments(spec, tool_call.args)
        if validation_error:
            return ToolResult.invalid_arguments(tool_call, validation_error)

        try:
            raw_result = spec.handler(**tool_call.args)
            if isinstance(raw_result, ToolResult):
                result = raw_result
            else:
                result = ToolResult.ok(tool_call, str(raw_result))
        except Exception as exc:
            result = ToolResult.execution_failed(tool_call, str(exc))

        return self.policy.apply_result_limits(result)

    def _available_tool_names(self) -> str:
        return ", ".join(sorted(self._tools)) or "<none>"

    def _validate_arguments(self, spec: ToolSpec, args: Dict[str, Any]) -> Optional[str]:
        try:
            spec.signature.bind(**args)
        except TypeError as exc:
            return str(exc)

        errors: List[str] = []
        for name, value in args.items():
            parameter = spec.signature.parameters.get(name)
            if parameter is None:
                continue

            expected = parameter.annotation
            if expected == inspect.Parameter.empty:
                continue

            if not self._matches_annotation(value, expected):
                expected_name = getattr(expected, "__name__", str(expected))
                errors.append(
                    f"{name} expected {expected_name}, got {type(value).__name__}"
                )

        return "; ".join(errors) if errors else None

    def _build_parameters_schema(self, signature: inspect.Signature) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for name, parameter in signature.parameters.items():
            properties[name] = self._parameter_to_schema(name, parameter)
            if parameter.default == inspect.Parameter.empty:
                required.append(name)

        return {"type": "object", "properties": properties, "required": required}

    def _parameter_to_schema(self, name: str, parameter: inspect.Parameter) -> Dict[str, Any]:
        return {
            "type": self._json_type_for_annotation(parameter.annotation),
            "description": f"Parameter {name}",
        }

    def _json_type_for_annotation(self, annotation: Any) -> str:
        origin = get_origin(annotation)
        if annotation == inspect.Parameter.empty:
            return "string"
        if annotation == bool:
            return "boolean"
        if annotation == int:
            return "integer"
        if annotation == float:
            return "number"
        if annotation == str:
            return "string"
        if annotation == list or origin == list:
            return "array"
        if annotation == dict or origin == dict:
            return "object"
        return "string"

    def _matches_annotation(self, value: Any, annotation: Any) -> bool:
        origin = get_origin(annotation)
        if annotation == bool:
            return isinstance(value, bool)
        if annotation == int:
            return isinstance(value, int) and not isinstance(value, bool)
        if annotation == float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if annotation == str:
            return isinstance(value, str)
        if annotation == list or origin == list:
            return isinstance(value, list)
        if annotation == dict or origin == dict:
            return isinstance(value, dict)
        return True
