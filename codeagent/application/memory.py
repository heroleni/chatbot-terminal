"""Rolling conversation memory.

Acceptance criteria: keep a rolling window of the 10 most recent user/assistant
messages; when it exceeds 10, drop the oldest. Every model call receives the
system instructions plus that window only.

Tool messages are not part of the 10-message count: they are intermediate
machinery belonging to an assistant turn. They are kept inside the window when
they sit between the counted messages, because every provider SDK rejects a
tool result that is not preceded by the tool call that produced it.
"""
from __future__ import annotations

from codeagent.domain.models import Message

DEFAULT_WINDOW = 10


class ConversationMemory:
    def __init__(self, max_messages: int = DEFAULT_WINDOW):
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages
        self._messages: list[Message] = []

    # ---- mutation ----------------------------------------------------------
    def append(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    def extend(self, messages: list[Message]) -> None:
        for m in messages:
            self.append(m)

    def replace(self, messages: list[Message]) -> None:
        """Rebuild the active window from a resumed session."""
        self._messages = list(messages)
        self._trim()

    def clear(self) -> None:
        self._messages = []

    def rollback(self, size: int) -> None:
        """Drop everything appended after the window had `size` messages."""
        del self._messages[size:]

    # ---- reading -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def counted(self) -> list[Message]:
        """Only the user/assistant messages, i.e. the ones the window counts."""
        return [m for m in self._messages if m.role in ("user", "assistant")]

    def window(self) -> list[Message]:
        """Exactly what is sent to the provider on every call."""
        return list(self._messages)

    # ---- internals ---------------------------------------------------------
    def _trim(self) -> None:
        """Drop the oldest counted messages until at most max_messages remain."""
        while True:
            indices = [i for i, m in enumerate(self._messages) if m.role in ("user", "assistant")]
            if len(indices) <= self.max_messages:
                break
            # Remove the oldest counted message and any tool messages that
            # preceded the next counted one (they would be left orphaned).
            start = indices[0]
            end = indices[1] if len(indices) > 1 else len(self._messages)
            del self._messages[start:end]

        # A provider conversation must not begin with an orphan tool result.
        while self._messages and self._messages[0].role == "tool":
            self._messages.pop(0)
