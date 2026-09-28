"""Domain-level errors. Providers' exceptions are translated into these."""
from __future__ import annotations


class LLMError(Exception):
    """An LLM provider failure, translated into a type owned by the domain."""

    def __init__(self, message: str, code: int | None = None, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.hint = hint


class ConfigurationError(Exception):
    """A required environment variable or setting is missing or invalid."""


#: Status codes that are worth retrying with exponential backoff.
RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})


def hint_for(code: int | None) -> str:
    """A readable, actionable hint shown in the terminal next to the error."""
    if code in (400, 404):
        return ("Check the model name for this provider and update the SDK "
                "(pip install -U -r requirements.txt).")
    if code in (401, 403):
        return ("Check the API key in your .env file: it may be missing, expired "
                "or lacking permission for this model.")
    if code == 429:
        return "Rate limit or quota exceeded. Wait a moment, switch model, or use /provider."
    if code in (500, 502, 503, 504):
        return ("The provider is overloaded; this is not your key or your code. "
                "Retries were already attempted. Try a stable model "
                "(GEMINI_MODEL=gemini-2.5-flash) or switch with /provider.")
    return ""
