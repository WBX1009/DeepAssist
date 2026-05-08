import inspect
from difflib import get_close_matches
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
    allowed_categories: Optional[Set[str]] = None
    tool_categories: Optional[Dict[str, str]] = None
    max_result_chars: int = 4000

    def can_execute(self, tool_name: str) -> bool:
        # Check specific tool whitelist first
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return False
        # If category-based policy is active, verify category
        if self.allowed_categories is not None and self.tool_categories:
            category = self.tool_categories.get(tool_name)
            if category is None or category not in self.allowed_categories:
                return False
        return True

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

    def describe_tools(self) -> str:
        if not self._tools:
            return "No tools are currently registered."

        lines: List[str] = []
        for spec in sorted(self._tools.values(), key=lambda item: item.name):
            properties = spec.parameters.get("properties", {})
            required_args = set(spec.parameters.get("required", []))
            rendered_args: List[str] = []
            for name, schema in properties.items():
                json_type = schema.get("type", "string")
                marker = "*" if name in required_args else "?"
                rendered_args.append(f"{name}:{json_type}{marker}")

            args_text = ", ".join(rendered_args) if rendered_args else "no arguments"
            lines.append(f"- {spec.name}({args_text}): {spec.description}")

        return "\n".join(lines)

    def list_tool_specs(self) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for spec in sorted(self._tools.values(), key=lambda item: item.name):
            properties = spec.parameters.get("properties", {})
            required_args = list(spec.parameters.get("required", []))
            specs.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "required_args": required_args,
                    "parameters": [
                        {
                            "name": name,
                            "type": schema.get("type", "string"),
                            "required": name in required_args,
                            "description": schema.get("description", ""),
                        }
                        for name, schema in properties.items()
                    ],
                }
            )
        return specs

    def execute(self, tool_call: ToolCall) -> ToolResult:
        spec = self._tools.get(tool_call.name)
        if not spec:
            return ToolResult.unknown_tool(
                tool_call,
                self._available_tool_names(),
                metadata=self._unknown_tool_metadata(tool_call),
            )

        if not self.policy.can_execute(tool_call.name):
            return ToolResult.execution_failed(
                tool_call,
                f"Tool is blocked by policy: {tool_call.name}",
                metadata={
                    "error_type": "tool_blocked",
                    "retryable": False,
                    "repair_strategy": "fallback_answer",
                    "diagnosis": "The requested tool is blocked by execution policy.",
                },
            )

        validation_error = self._validate_arguments(spec, tool_call.args)
        if validation_error:
            return ToolResult.invalid_arguments(
                tool_call,
                validation_error,
                metadata=self._invalid_argument_metadata(spec, tool_call, validation_error),
            )

        try:
            raw_result = spec.handler(**tool_call.args)
            if isinstance(raw_result, ToolResult):
                result = raw_result
            else:
                result = ToolResult.ok(tool_call, str(raw_result))
        except Exception as exc:
            result = ToolResult.execution_failed(
                tool_call,
                str(exc),
                metadata=self._execution_failure_metadata(spec, tool_call, exc),
            )

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

    def _unknown_tool_metadata(self, tool_call: ToolCall) -> Dict[str, Any]:
        tool_names = sorted(self._tools)
        matches = get_close_matches(tool_call.name, tool_names, n=1, cutoff=0.5)
        metadata: Dict[str, Any] = {
            "error_type": "unknown_tool",
            "retryable": bool(matches),
            "repair_strategy": "switch_tool" if matches else "fallback_answer",
            "available_tools": tool_names,
            "diagnosis": "The tool name does not match any registered tool.",
        }
        if matches:
            metadata["suggested_tool"] = matches[0]
        return metadata

    def _invalid_argument_metadata(
        self,
        spec: ToolSpec,
        tool_call: ToolCall,
        validation_error: str,
    ) -> Dict[str, Any]:
        required_args = [
            name
            for name, parameter in spec.signature.parameters.items()
            if parameter.default == inspect.Parameter.empty
        ]
        provided_args = set(tool_call.args)
        expected_args = set(spec.signature.parameters)
        missing_required_args = [name for name in required_args if name not in provided_args]
        unexpected_args = sorted(provided_args - expected_args)
        argument_type_errors = self._argument_type_errors(spec, tool_call.args)
        return {
            "error_type": "invalid_arguments",
            "retryable": True,
            "repair_strategy": "fix_arguments",
            "diagnosis": validation_error,
            "required_args": required_args,
            "missing_required_args": missing_required_args,
            "unexpected_args": unexpected_args,
            "argument_type_errors": argument_type_errors,
            "allowed_tools": sorted(self._tools),
        }

    def _execution_failure_metadata(
        self,
        spec: ToolSpec,
        tool_call: ToolCall,
        exc: Exception,
    ) -> Dict[str, Any]:
        error_text = str(exc).lower()
        diagnosis = "The tool raised an exception during execution."
        repair_strategy = "adjust_arguments_or_fallback"
        retryable = True

        if "missing" in error_text or "not found" in error_text:
            diagnosis = "The tool could not find the requested target or input."
            repair_strategy = "adjust_arguments"
        elif "permission" in error_text or "denied" in error_text:
            diagnosis = "The tool lacks permission to access the requested resource."
            repair_strategy = "fallback_answer"
            retryable = False
        elif "timeout" in error_text:
            diagnosis = "The tool timed out before producing a result."
            repair_strategy = "retry_or_fallback"
        elif "json" in error_text or "decode" in error_text:
            diagnosis = "The tool received malformed structured input."
            repair_strategy = "fix_arguments"

        return {
            "error_type": "tool_execution_failed",
            "retryable": retryable,
            "repair_strategy": repair_strategy,
            "diagnosis": diagnosis,
            "exception_type": type(exc).__name__,
            "tool_name": spec.name,
            "allowed_tools": sorted(self._tools),
            "provided_args": {key: type(value).__name__ for key, value in tool_call.args.items()},
        }

    def _argument_type_errors(self, spec: ToolSpec, args: Dict[str, Any]) -> List[str]:
        issues: List[str] = []
        for name, value in args.items():
            parameter = spec.signature.parameters.get(name)
            if parameter is None:
                continue
            expected = parameter.annotation
            if expected == inspect.Parameter.empty:
                continue
            if self._matches_annotation(value, expected):
                continue
            expected_name = getattr(expected, "__name__", str(expected))
            issues.append(f"{name} expected {expected_name}, got {type(value).__name__}")
        return issues
