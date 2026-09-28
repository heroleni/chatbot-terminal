"""Disk and shell adapters (they implement WorkspacePort and ShellPort)."""
from __future__ import annotations

import os
import subprocess


class LocalWorkspace:
    def __init__(self, root: str | None = None, max_read_chars: int = 50_000):
        self.root = os.path.realpath(root or os.getcwd())
        self._max = max_read_chars

    def _resolve(self, path: str) -> str:
        full = os.path.realpath(os.path.join(self.root, path or "."))
        try:
            inside = os.path.commonpath([self.root, full]) == self.root
        except ValueError:  # e.g. a different drive on Windows
            inside = False
        if not inside:
            raise PermissionError("Path outside the working directory")
        return full

    def list_dir(self, path: str = ".") -> list[str]:
        full = self._resolve(path)
        return sorted(n + ("/" if os.path.isdir(os.path.join(full, n)) else "")
                      for n in os.listdir(full))

    def read_text(self, path: str) -> str:
        with open(self._resolve(path), encoding="utf-8", errors="replace") as f:
            return f.read(self._max)

    def write_text(self, path: str, content: str) -> None:
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)


class LocalShell:
    def __init__(self, cwd: str | None = None, timeout: int = 120, max_chars: int = 20_000):
        self._cwd, self._timeout, self._max = cwd or os.getcwd(), timeout, max_chars

    def run(self, command: str) -> str:
        try:
            r = subprocess.run(command, shell=True, cwd=self._cwd, capture_output=True,
                               text=True, errors="replace", timeout=self._timeout)
        except subprocess.TimeoutExpired:
            return f"The command exceeded the {self._timeout}s limit and was cancelled."
        return (r.stdout + r.stderr)[-self._max:] or "(no output)"
