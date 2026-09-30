"""Use case: persistent chat sessions.

Pairs the in-memory rolling window (ConversationMemory) with a durable
SessionRepositoryPort, so chats survive application restarts:

  - start()           -> opens a brand new session
  - record()          -> persists one message and keeps the window in sync
  - resume(id)        -> rebuilds the active window from that session's last 10
                         user/assistant messages and keeps writing into it
  - list_sessions()   -> what /chats shows
"""
from __future__ import annotations

from codeagent.domain.models import Message, SessionInfo
from codeagent.domain.ports import SessionRepositoryPort

TITLE_MAX_CHARS = 60


def make_title(messages: list[Message]) -> str:
    """Use the first user message as a human-readable session title."""
    for m in messages:
        if m.role == "user" and m.text.strip():
            text = " ".join(m.text.split())
            return text[:TITLE_MAX_CHARS] + ("..." if len(text) > TITLE_MAX_CHARS else "")
    return "(empty session)"


class SessionService:
    def __init__(self, repository: SessionRepositoryPort, provider: str, model: str):
        self._repo = repository
        self._provider = provider
        self._model = model
        self.session_id: str = ""

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> str:
        self.session_id = self._repo.create(self._provider, self._model)
        return self.session_id

    def resume(self, session_id: str) -> list[Message] | None:
        """Return the full stored history, or None if the id does not exist."""
        history = self._repo.load(session_id)
        if history is None:
            return None
        self.session_id = session_id
        return history

    def set_provider(self, provider: str, model: str) -> None:
        """Remember the active provider/model so /chats reflects reality."""
        self._provider, self._model = provider, model
        if self.session_id:
            self._repo.touch(self.session_id, provider, model)

    # ---- writing -----------------------------------------------------------
    def record(self, message: Message) -> None:
        if not self.session_id:
            self.start()
        self._repo.append(self.session_id, message)
        if message.role == "user":
            # Keep the listing title and the updated_at stamp fresh.
            stored = self._repo.load(self.session_id) or []
            self._repo.touch(self.session_id, self._provider, self._model, make_title(stored))

    # ---- reading -----------------------------------------------------------
    def list_sessions(self, limit: int = 20) -> list[SessionInfo]:
        return self._repo.list_sessions(limit)
