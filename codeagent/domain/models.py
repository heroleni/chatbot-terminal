"""Domain model.

Pure types: they import no provider SDK (Gemini, OpenAI, Anthropic) and no OS
module. This is the shared language every adapter speaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


def object_schema(properties: dict | None = None, required: list | None = None) -> dict:
    """JSON Schema for a tool's parameters (shared by the three providers)."""
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = list(required)
    return schema


@dataclass(frozen=True)
class ToolSpec:
    """Description of a tool the model is allowed to call."""
    name: str
    description: str
    parameters: dict = field(default_factory=object_schema)


@dataclass
class ToolCall:
    """The model asks to execute a tool."""
    name: str
    args: dict
    id: str = ""


@dataclass
class ToolResult:
    """Outcome of a ToolCall, handed back to the model."""
    call: ToolCall
    output: str


@dataclass
class Message:
    """Normalized conversation message.

    provider_data holds the native payload of the provider that produced it (for
    example Gemini's thought signatures). Each adapter reads only its own key.
    It is intentionally NOT persisted: it is provider-specific and short-lived.
    """
    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    provider_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", text=text)

    # ---- serialization (used by the session repository) --------------------
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "text": self.text,
            "tool_calls": [{"name": c.name, "args": c.args, "id": c.id} for c in self.tool_calls],
            "tool_results": [
                {"name": r.call.name, "args": r.call.args, "id": r.call.id, "output": r.output}
                for r in self.tool_results
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        calls = [ToolCall(c["name"], c.get("args", {}), c.get("id", ""))
                 for c in data.get("tool_calls", [])]
        results = [
            ToolResult(ToolCall(r["name"], r.get("args", {}), r.get("id", "")), r.get("output", ""))
            for r in data.get("tool_results", [])
        ]
        return cls(role=data["role"], text=data.get("text", ""),
                   tool_calls=calls, tool_results=results)


@dataclass
class LLMResponse:
    message: Message


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str


@dataclass(frozen=True)
class Chunk:
    """A fragment retrieved from a knowledge source (RAG), with its source metadata."""
    source: str
    position: int
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class SessionInfo:
    """Summary of a persisted chat session, as listed by /chats."""
    id: str
    title: str
    provider: str
    model: str
    message_count: int
    updated_at: str
