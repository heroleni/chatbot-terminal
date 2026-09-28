"""Adapter: LLMPort -> official OpenAI SDK (Chat Completions).

With base_url (or the OPENAI_BASE_URL variable) the very same adapter also serves
any OpenAI-compatible server, for example OpenRouter or a local Ollama.
"""
from __future__ import annotations

import json
import os
from typing import Sequence

import openai

from codeagent.adapters.llm.retry import with_backoff
from codeagent.domain.errors import LLMError, hint_for
from codeagent.domain.models import LLMResponse, Message, ToolCall, ToolSpec


def to_openai_messages(system: str, messages: Sequence[Message]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            msg: dict = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                    for c in m.tool_calls]
            out.append(msg)
        else:  # tool: one message per result
            for r in m.tool_results:
                out.append({"role": "tool", "tool_call_id": r.call.id, "content": r.output})
    return out


def to_openai_tools(tools: Sequence[ToolSpec]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools]


def parse_response(resp) -> Message:
    msg = resp.choices[0].message
    calls = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(name=tc.function.name, args=args, id=tc.id))
    return Message(role="assistant", text=msg.content or "", tool_calls=calls)


class OpenAIAdapter:
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, client=None):
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise LLMError("No model configured: pass --model or set OPENAI_MODEL in your .env "
                           "(check the model names available to your account).")
        if client is None:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key and not (base_url or os.environ.get("OPENAI_BASE_URL")):
                raise LLMError("Missing environment variable OPENAI_API_KEY")
            client = openai.OpenAI(api_key=key or "no-key", base_url=base_url)
        self._client = client

    def generate(self, system: str, messages: Sequence[Message],
                 tools: Sequence[ToolSpec]) -> LLMResponse:
        kwargs: dict = {"model": self.model, "messages": to_openai_messages(system, messages)}
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            # Reasoning models (gpt-6-*) only support function calling on the
            # Chat Completions endpoint when reasoning effort is disabled.
            if self.model.startswith("gpt-6"):
                kwargs["reasoning_effort"] = "none"
        try:
            resp = with_backoff(
                lambda: self._client.chat.completions.create(**kwargs),
                lambda e: getattr(e, "status_code", None),
                openai.OpenAIError,
            )
        except openai.OpenAIError as e:
            code = getattr(e, "status_code", None)
            raise LLMError(str(e), code=code, hint=hint_for(code)) from e
        return LLMResponse(parse_response(resp))
