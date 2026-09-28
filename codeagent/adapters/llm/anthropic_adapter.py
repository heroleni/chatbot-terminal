"""Adapter: LLMPort -> official Anthropic SDK (Messages API)."""
from __future__ import annotations

import os
from typing import Sequence

import anthropic

from codeagent.adapters.llm.retry import with_backoff
from codeagent.domain.errors import LLMError, hint_for
from codeagent.domain.models import LLMResponse, Message, ToolCall, ToolSpec

DEFAULT_MODEL = "claude-sonnet-5-5"


def to_anthropic_messages(messages: Sequence[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            raw = m.provider_data.get("anthropic")
            if raw is not None:  # keep native blocks (e.g. reasoning blocks)
                out.append({"role": "assistant", "content": raw})
                continue
            blocks: list[dict] = [{"type": "text", "text": m.text}] if m.text else []
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                       for c in m.tool_calls]
            out.append({"role": "assistant", "content": blocks})
        else:  # tool: every result travels inside a single user message
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r.call.id, "content": r.output}
                for r in m.tool_results]})
    return out


def to_anthropic_tools(tools: Sequence[ToolSpec]) -> list[dict]:
    return [{"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools]


def parse_response(resp) -> Message:
    texts, calls = [], []
    for block in resp.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(name=block.name, args=dict(block.input), id=block.id))
    return Message(role="assistant", text="\n".join(texts), tool_calls=calls,
                   provider_data={"anthropic": resp.content})


class AnthropicAdapter:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 max_tokens: int = 4096, client=None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self._max_tokens = max_tokens
        if client is None:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("Missing environment variable ANTHROPIC_API_KEY")
            client = anthropic.Anthropic(api_key=key)
        self._client = client

    def generate(self, system: str, messages: Sequence[Message],
                 tools: Sequence[ToolSpec]) -> LLMResponse:
        kwargs: dict = {"model": self.model, "max_tokens": self._max_tokens,
                        "system": system, "messages": to_anthropic_messages(messages)}
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        try:
            resp = with_backoff(
                lambda: self._client.messages.create(**kwargs),
                lambda e: getattr(e, "status_code", None),
                anthropic.APIError,
            )
        except anthropic.APIError as e:
            code = getattr(e, "status_code", None)
            raise LLMError(str(e), code=code, hint=hint_for(code)) from e
        return LLMResponse(parse_response(resp))
