"""Agentic RAG: retrieval is exposed as tools and the MODEL decides when, where
and how many times to retrieve.

Knowledge sources the agent can route to (the router is the LLM itself):
  - use_skill        -> procedural knowledge (skills/ folders)
  - search_knowledge -> project documents embedded in the vector store
  - search_code      -> code and notes in the workspace
  - read_file / list_dir -> direct reading once it knows the exact file
  - search_web / fetch_url_content -> the internet
"""
from __future__ import annotations

from typing import Any

from codeagent.application.tools import Tool
from codeagent.domain.models import Chunk, ToolSpec, object_schema
from codeagent.domain.ports import RetrieverPort, SkillRepositoryPort

RAG_POLICY = """\
Retrieval policy (Agentic RAG):
1. Decide whether you need external information. If the question is general or conversational, answer directly without searching.
2. Pick the right source: use_skill for procedures and ways of working; search_knowledge for the project's document knowledge base; search_code for code and notes in the workspace; read_file when you already know the exact file. If the user asks you to research or look something up (or asks for sources), use search_web. If they give you a direct link, use fetch_url_content. After search_web, read the 1-2 most promising results with fetch_url_content when the summary is not enough.
3. Judge what you retrieved. If it is insufficient or irrelevant, rephrase the query (other terms, synonyms, a concrete identifier) or switch source. At most 3 searches per question.
4. Answer ONLY with what the retrieved sources support, and cite them as [source: path] or, for the internet, [source: URL].
5. If the retrieved context does not support an answer, say explicitly that there is not enough information in the knowledge base. Never invent facts, file names, figures or citations to fill the gap.
"""


def format_chunks(chunks: list[Chunk]) -> str:
    """Render chunks with their source metadata, which the model must cite."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] source: {c.source} (chunk {c.position}, relevance {c.score:.2f})"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


class UseSkillTool(Tool):
    """Loads a skill's full instructions on demand."""

    def __init__(self, skills: SkillRepositoryPort):
        self.spec = ToolSpec(
            "use_skill",
            "Load the instructions of a skill from the skills/ folder by name. "
            "Use it when the task matches the description of an available skill.",
            object_schema({"name": {"type": "string", "description": "Skill name"}}, ["name"]),
        )
        self._skills = skills

    def run(self, name: str) -> str:
        body = self._skills.load(name)
        if body is None:
            available = ", ".join(s.name for s in self._skills.list_skills()) or "none"
            return f"There is no skill named '{name}'. Available: {available}"
        return body


class RetrievalTool(Tool):
    """Generic search tool over any RetrieverPort."""

    def __init__(self, name: str, description: str, retriever: RetrieverPort, default_k: int = 4):
        self.spec = ToolSpec(
            name, description,
            object_schema({
                "query": {"type": "string", "description": "Query using the most specific keywords"},
                "k": {"type": "integer", "description": f"Number of chunks (1-8, default {default_k})"},
            }, ["query"]),
        )
        self._retriever, self._default_k = retriever, default_k

    def run(self, query: str, k: Any = None) -> str:
        k = max(1, min(8, int(k or self._default_k)))
        try:
            chunks = self._retriever.search(query, k)
        except Exception as e:
            # A retrieval failure is recoverable: tell the model, do not crash.
            return (f"The knowledge base could not be reached ({e}). "
                    "Say you cannot verify this right now; do not invent an answer.")
        if not chunks:
            return ("No results for that query. Rephrase with other terms or synonyms, or try "
                    "another source. If nothing is found, state that the knowledge base does "
                    "not contain enough information to answer.")
        return format_chunks(chunks)
