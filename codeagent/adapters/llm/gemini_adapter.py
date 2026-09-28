"""Adapter: LLMPort -> SDK google-genai (Gemini)."""
from __future__ import annotations

import os
import random
import time
from typing import Sequence

from google import genai
from google.genai import errors, types

from codeagent.domain.errors import RETRYABLE_CODES, LLMError, hint_for
from codeagent.domain.models import LLMResponse, Message, ToolCall, ToolSpec

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 4


def to_contents(messages: Sequence[Message]) -> list:
    out = []
    for m in messages:
        if m.role == "user":
            out.append(types.Content(role="user", parts=[types.Part(text=m.text)]))
        elif m.role == "assistant":
            raw = m.provider_data.get("gemini")
            if raw is not None:  # keep Gemini 3 thought signatures
                out.append(raw)
                continue
            parts = [types.Part(text=m.text)] if m.text else []
            parts += [types.Part(function_call=types.FunctionCall(name=c.name, args=c.args))
                      for c in m.tool_calls]
            out.append(types.Content(role="model", parts=parts))
        else:  # tool
            parts = [types.Part.from_function_response(name=r.call.name, response={"result": r.output})
                     for r in m.tool_results]
            out.append(types.Content(role="user", parts=parts))
    return out


def to_declarations(tools: Sequence[ToolSpec]) -> list[dict]:
    decls = []
    for t in tools:
        d = {"name": t.name, "description": t.description}
        if t.parameters.get("properties"):  # Gemini rejects object schemas with no properties
            d["parameters"] = t.parameters
        decls.append(d)
    return decls


def parse_response(resp) -> Message:
    content = resp.candidates[0].content if resp.candidates else None
    if content is None:
        raise LLMError("Empty response, or blocked by the provider")
    texts, calls = [], []
    for p in content.parts or []:
        if p.text and not getattr(p, "thought", False):
            texts.append(p.text)
        if p.function_call:
            fc = p.function_call
            calls.append(ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id or ""))
    return Message(role="assistant", text="\n".join(texts), tool_calls=calls,
                   provider_data={"gemini": content})


class GeminiAdapter:
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None, client=None):
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        if client is None:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise LLMError("Missing environment variable GEMINI_API_KEY")
            # The key prefix (AIza / AQ.) is not validated here: Google defines it.
            client = genai.Client(api_key=key)
        self._client = client

    def generate(self, system: str, messages: Sequence[Message],
                 tools: Sequence[ToolSpec]) -> LLMResponse:
        # thinking_level only exists on Gemini 3; on 2.5 the parameter is thinking_budget.
        thinking = types.ThinkingConfig(thinking_level="low") if self.model.startswith("gemini-3") else None
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=to_declarations(tools))] if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=thinking,
        )
        contents = to_contents(messages)
        delay = 1.0
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config)
            except errors.APIError as e:
                if e.code in RETRYABLE_CODES and attempt < MAX_RETRIES - 1:
                    # Transient provider overload: exponential backoff with jitter.
                    time.sleep(delay + random.uniform(0, 0.5))
                    delay *= 2
                    continue
                raise LLMError(str(e.message), code=e.code, hint=hint_for(e.code)) from e
            return LLMResponse(parse_response(resp))
