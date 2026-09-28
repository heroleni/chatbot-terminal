"""Adapter factory. Lazy imports: you only need to install the SDK you actually use."""
from __future__ import annotations

import importlib

from codeagent.domain.ports import LLMPort

_PROVIDERS = {
    "gemini": ("codeagent.adapters.llm.gemini_adapter", "GeminiAdapter"),
    "openai": ("codeagent.adapters.llm.openai_adapter", "OpenAIAdapter"),
    "anthropic": ("codeagent.adapters.llm.anthropic_adapter", "AnthropicAdapter"),
}

PROVIDER_NAMES = tuple(_PROVIDERS)


def create_llm(provider: str, model: str | None = None) -> LLMPort:
    try:
        module_name, class_name = _PROVIDERS[provider.lower()]
    except KeyError:
        raise ValueError(f"Unknown provider: '{provider}'. Options: {', '.join(_PROVIDERS)}") from None
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise RuntimeError(f"The SDK for '{provider}' is not installed ({e.name}). "
                           f"Run: pip install -r requirements.txt") from e
    return getattr(module, class_name)(model=model)
