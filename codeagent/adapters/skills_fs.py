"""Filesystem skill repository: skills/<name>/SKILL.md

Same format as Agent Skills: a YAML front-matter header with name and description.
"""
from __future__ import annotations

import os

from codeagent.domain.models import SkillInfo


def parse_frontmatter(text: str) -> tuple[dict, str]:
    meta: dict = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip()
            body = parts[2].strip()
    return meta, body


class FileSystemSkillRepository:
    def __init__(self, skills_dir: str):
        self._dir = skills_dir
        self.skipped: list[str] = []  # folders without SKILL.md, reported to the user

    def _scan(self) -> dict[str, tuple[str, str]]:
        found: dict[str, tuple[str, str]] = {}
        self.skipped = []
        if not os.path.isdir(self._dir):
            return found
        for folder in sorted(os.listdir(self._dir)):
            folder_path = os.path.join(self._dir, folder)
            if not os.path.isdir(folder_path):
                continue
            path = os.path.join(folder_path, "SKILL.md")
            if not os.path.isfile(path):
                self.skipped.append(folder)
                continue
            with open(path, encoding="utf-8-sig") as f:
                meta, body = parse_frontmatter(f.read())
            found[meta.get("name", folder)] = (meta.get("description", "(no description)"), body)
        return found

    def list_skills(self) -> list[SkillInfo]:
        return [SkillInfo(n, d) for n, (d, _) in self._scan().items()]

    def load(self, name: str) -> str | None:
        entry = self._scan().get(name)
        return entry[1] if entry else None
