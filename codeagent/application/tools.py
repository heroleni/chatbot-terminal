"""Tools (Python-side skills).

They depend on ports only: none of them imports os, subprocess or a provider SDK.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable

from codeagent.domain.models import ToolSpec, object_schema
from codeagent.domain.ports import ConfirmationPort, ShellPort, WebPort, WorkspacePort


class Tool(ABC):
    spec: ToolSpec

    @abstractmethod
    def run(self, **args: Any) -> str: ...


class FunctionTool(Tool):
    """Turns any Python function into a tool. The shortest way to add one."""

    def __init__(self, name: str, description: str, fn: Callable[..., Any],
                 properties: dict | None = None, required: list | None = None):
        self.spec = ToolSpec(name, description, object_schema(properties, required))
        self._fn = fn

    def run(self, **args: Any) -> str:
        return str(self._fn(**args))


class ListDirTool(Tool):
    def __init__(self, workspace: WorkspacePort):
        self.spec = ToolSpec(
            "list_dir", "List the files and folders of a directory.",
            object_schema({"path": {"type": "string", "description": "Relative path, defaults to '.'"}}),
        )
        self._ws = workspace

    def run(self, path: str = ".") -> str:
        return "\n".join(self._ws.list_dir(path)) or "(empty folder)"


class ReadFileTool(Tool):
    def __init__(self, workspace: WorkspacePort):
        self.spec = ToolSpec(
            "read_file", "Read the contents of a text file.",
            object_schema({"path": {"type": "string"}}, ["path"]),
        )
        self._ws = workspace

    def run(self, path: str) -> str:
        return self._ws.read_text(path)


class WriteFileTool(Tool):
    def __init__(self, workspace: WorkspacePort, confirmer: ConfirmationPort):
        self.spec = ToolSpec(
            "write_file", "Create or overwrite a file with the given content.",
            object_schema({"path": {"type": "string"}, "content": {"type": "string"}},
                          ["path", "content"]),
        )
        self._ws, self._confirm = workspace, confirmer

    def run(self, path: str, content: str) -> str:
        if not self._confirm.confirm(f"Write {path} ({len(content)} characters)"):
            return "The user declined the operation."
        self._ws.write_text(path, content)
        return f"Wrote {path}"


class RunBashTool(Tool):
    def __init__(self, shell: ShellPort, confirmer: ConfirmationPort):
        self.spec = ToolSpec(
            "run_bash", "Run a shell command and return stdout/stderr.",
            object_schema({"command": {"type": "string"}}, ["command"]),
        )
        self._shell, self._confirm = shell, confirmer

    def run(self, command: str) -> str:
        if not self._confirm.confirm(f"Run: {command}"):
            return "The user declined the operation."
        return self._shell.run(command)


class WebSearchTool(Tool):
    def __init__(self, web: WebPort):
        self.spec = ToolSpec(
            "search_web",
            "Search the internet (DuckDuckGo). Use it when the user asks to research, look up "
            "current information, or back an answer with sources. Returns title, link and summary.",
            object_schema({
                "query": {"type": "string", "description": "Search query, specific and concise"},
                "max_results": {"type": "integer", "description": "Number of results (1-10, default 5)"},
            }, ["query"]),
        )
        self._web = web

    def run(self, query: str, max_results: Any = None) -> str:
        n = max(1, min(10, int(max_results or 5)))
        return self._web.search(query, n)


class FetchUrlTool(Tool):
    def __init__(self, web: WebPort):
        self.spec = ToolSpec(
            "fetch_url_content",
            "Download a URL and extract its main text. Use it when the user gives a direct "
            "link, or to read a search_web result in depth.",
            object_schema({"url": {"type": "string", "description": "Full URL (https://...)"}}, ["url"]),
        )
        self._web = web

    def run(self, url: str) -> str:
        return self._web.fetch(url)


def default_tools(workspace: WorkspacePort, shell: ShellPort, confirmer: ConfirmationPort,
                  now: Callable[[], datetime] = datetime.now) -> list[Tool]:
    """The built-in Python tools, all of them behind ports."""
    return [
        FunctionTool("current_time", "Return the current date and time of the user's computer.",
                     lambda: now().strftime("%Y-%m-%d %H:%M:%S")),
        ListDirTool(workspace),
        ReadFileTool(workspace),
        WriteFileTool(workspace, confirmer),
        RunBashTool(shell, confirmer),
        FunctionTool("add", "Add two numbers.", lambda a, b: a + b,
                     {"a": {"type": "number"}, "b": {"type": "number"}}, ["a", "b"]),
    ]
