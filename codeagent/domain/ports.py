"""Ports: the contracts the core needs from the outside world.

The core (application/) knows only these interfaces. The adapters (adapters/)
implement them: Gemini, OpenAI, Anthropic, SQLite, the disk, the terminal.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from codeagent.domain.models import Chunk, LLMResponse, Message, SessionInfo, SkillInfo, ToolSpec


# ---- Driven ports (the core calls them) ------------------------------------
class LLMPort(Protocol):
    """Any language model. This is the contract behind the Adapter pattern."""
    name: str

    def generate(self, system: str, messages: Sequence[Message],
                 tools: Sequence[ToolSpec]) -> LLMResponse: ...


class RetrieverPort(Protocol):
    """Any retrieval mechanism (BM25, embeddings, a vector database...)."""

    def search(self, query: str, k: int = 4) -> list[Chunk]: ...


class SkillRepositoryPort(Protocol):
    """Where skills live (folders, a database, a remote service...)."""

    def list_skills(self) -> list[SkillInfo]: ...

    def load(self, name: str) -> str | None: ...


class SessionRepositoryPort(Protocol):
    """Durable storage for chat sessions, so they survive application restarts."""

    def create(self, provider: str, model: str) -> str:
        """Create an empty session and return its stable id."""

    def append(self, session_id: str, message: Message) -> None:
        """Persist one message at the end of the session."""

    def load(self, session_id: str) -> list[Message] | None:
        """Full message history, or None when the session does not exist."""

    def list_sessions(self, limit: int = 20) -> list[SessionInfo]: ...

    def touch(self, session_id: str, provider: str, model: str, title: str = "") -> None:
        """Update the session's provider/model/title and its updated_at stamp."""


class WorkspacePort(Protocol):
    """Access to the working directory (files)."""

    def list_dir(self, path: str = ".") -> list[str]: ...

    def read_text(self, path: str) -> str: ...

    def write_text(self, path: str, content: str) -> None: ...


class ShellPort(Protocol):
    def run(self, command: str) -> str: ...


class WebPort(Protocol):
    """Internet access: search and read pages."""

    def search(self, query: str, max_results: int = 5) -> str: ...

    def fetch(self, url: str) -> str: ...


class ConfirmationPort(Protocol):
    """Ask the user for permission before actions with side effects."""

    def confirm(self, message: str) -> bool: ...


class OutputPort(Protocol):
    """Show the user what the agent is doing."""

    def show_text(self, text: str) -> None: ...

    def show_tool_call(self, name: str, args: dict) -> None: ...

    def show_error(self, message: str) -> None: ...
