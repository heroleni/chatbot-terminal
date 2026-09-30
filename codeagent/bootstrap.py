"""Composition root: wires the concrete adapters into the application core."""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from codeagent.adapters.cli import TerminalUI, describe_provider, run_repl
from codeagent.adapters.llm.factory import PROVIDER_NAMES, create_llm
from codeagent.adapters.retrieval.bm25 import Bm25Retriever
from codeagent.adapters.sessions_sqlite import SqliteSessionRepository
from codeagent.adapters.skills_fs import FileSystemSkillRepository
from codeagent.adapters.web import WebClient
from codeagent.adapters.workspace import LocalShell, LocalWorkspace
from codeagent.application.agent_service import AgentService
from codeagent.application.memory import ConversationMemory
from codeagent.application.rag import RAG_POLICY, RetrievalTool, UseSkillTool
from codeagent.application.session_service import SessionService
from codeagent.application.tools import FetchUrlTool, WebSearchTool, default_tools
from codeagent.domain.errors import LLMError

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(PROJECT_DIR, "data", "chats.db")

CODE_SUFFIXES = (".py", ".md", ".txt", ".js", ".ts", ".json", ".toml", ".yaml", ".yml",
                 ".html", ".css")


def build_system_prompt(root: str) -> str:
    return (
        "You are a coding agent working in the user's terminal. "
        f"Working directory: {root}. "
        "Explore before editing, make small changes, and verify by running tests or the "
        "code itself whenever possible. Always answer in English.\n\n"
        + RAG_POLICY
    )


def build_vector_tool(ui: TerminalUI) -> RetrievalTool | None:
    """Semantic retrieval over Supabase + pgvector.

    Missing credentials are a recoverable configuration error: the agent still
    starts, with a readable warning, and keeps BM25 and the other tools.
    """
    try:
        from rag.gemini_embeddings import GeminiEmbeddingAdapter
        from rag.supabase_vector import SupabaseVectorRetriever

        retriever = SupabaseVectorRetriever(
            embedding_adapter=GeminiEmbeddingAdapter(),
            threshold=float(os.getenv("RAG_MIN_SCORE", "0.35")),
        )
    except Exception as e:
        ui.show_error(
            f"Vector knowledge base disabled: {e}\n"
            "   Set SUPABASE_URL, SUPABASE_KEY and GEMINI_API_KEY in .env and run "
            "`python -m rag.ingest` to enable search_knowledge."
        )
        return None

    return RetrievalTool(
        "search_knowledge",
        "Search the document knowledge base (Supabase + pgvector) using embeddings. "
        "Returns the most relevant chunks together with their source metadata.",
        retriever,
        default_k=int(os.getenv("RAG_TOP_K", "5")),
    )


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        prog="codeagent",
        description="Multi-provider terminal chatbot with memory, saved chats and RAG",
    )
    parser.add_argument("--provider", choices=PROVIDER_NAMES,
                        default=os.environ.get("LLM_PROVIDER", "openai"),
                        help="Active LLM provider (also switchable at runtime with /provider)")
    parser.add_argument("--model", default=None, help="Model name for the chosen provider")
    parser.add_argument("--max-steps", type=int, default=12,
                        help="Maximum tool-calling steps per user message")
    parser.add_argument("--memory-window", type=int, default=10,
                        help="How many recent user/assistant messages are sent to the model")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite file holding the saved chats")
    parser.add_argument("--resume", default=None, metavar="SESSION_ID",
                        help="Resume a saved session on startup")
    parser.add_argument("prompt", nargs="*", help="If omitted, the interactive REPL opens")

    args = parser.parse_args(argv)
    root = os.getcwd()
    ui = TerminalUI()

    try:
        llm = create_llm(args.provider, args.model)
    except (LLMError, ValueError, RuntimeError) as e:
        raise SystemExit(f"[error] {e}")

    skills = FileSystemSkillRepository(os.path.join(PROJECT_DIR, "skills"))

    tools = default_tools(LocalWorkspace(root), LocalShell(root), ui)
    tools.append(UseSkillTool(skills))

    web = WebClient()
    tools += [WebSearchTool(web), FetchUrlTool(web)]

    vector_tool = build_vector_tool(ui)
    if vector_tool is not None:
        tools.append(vector_tool)

    tools.append(RetrievalTool(
        "search_code",
        "Search the workspace code and notes by keyword (function names, errors, "
        "concepts). Returns fragments with their file path.",
        Bm25Retriever(root, suffixes=CODE_SUFFIXES),
    ))

    sessions = SessionService(SqliteSessionRepository(args.db), llm.name,
                              getattr(llm, "model", ""))

    service = AgentService(
        llm, tools, skills, ui, build_system_prompt(root), args.max_steps,
        memory=ConversationMemory(args.memory_window), sessions=sessions,
    )

    if args.resume:
        if not service.resume(args.resume):
            raise SystemExit(f"[error] No saved session with id '{args.resume}'.")
    else:
        sessions.start()

    if args.prompt:
        service.ask(" ".join(args.prompt))
        return

    banner = [
        f"Terminal chatbot ready in {root}.",
        describe_provider(service),
        f"Memory window: {args.memory_window} messages | Saved chats: {args.db}",
        "Tools: " + ", ".join(t.spec.name for t in tools),
        "Skills: " + (", ".join(s.name for s in skills.list_skills()) or "none"),
    ]
    if skills.skipped:
        banner.append("[warning] Skill folders without SKILL.md (ignored): "
                      + ", ".join(skills.skipped))

    run_repl(service, ui, banner)
