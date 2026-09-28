"""Shared backoff helper for every provider adapter.

Transient provider failures (429, 500, 502, 503, 504) are recoverable: they must
not terminate the application. Each adapter retries with exponential backoff and
jitter, and only then raises an LLMError carrying a readable hint.
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from codeagent.domain.errors import RETRYABLE_CODES

T = TypeVar("T")

MAX_RETRIES = 4
BASE_DELAY = 1.0


def with_backoff(call: Callable[[], T], code_of: Callable[[Exception], int | None],
                 exception_type: type[Exception], sleep: Callable[[float], None] = time.sleep) -> T:
    """Run `call`, retrying transient provider errors with exponential backoff."""
    delay = BASE_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except exception_type as e:
            if code_of(e) in RETRYABLE_CODES and attempt < MAX_RETRIES - 1:
                sleep(delay + random.uniform(0, 0.5))
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")
