"""Terminal adapter: driving adapter (input) and output adapter at once.

Implements OutputPort and ConfirmationPort, and holds the REPL plus the
slash-command dispatcher (/provider, /memory, /chats, /resume, /clear).
"""
from __future__ import annotations

from codeagent.adapters.llm.factory import PROVIDER_NAMES, create_llm
from codeagent.application.agent_service import AgentService
from codeagent.domain.errors import LLMError

HELP = """\
Commands:
  /help                     Show this help.
  /provider                 Show the active provider and the available ones.
  /provider <name> [model]  Switch provider at runtime (openai, anthropic, gemini).
  /memory                   Inspect the active 10-message window.
  /clear                    Clear the active window (the saved chat is kept).
  /chats                    List saved sessions by id.
  /resume <session-id>      Continue a saved session.
  /new                      Start a new saved session.
  /exit                     Quit (also: exit, quit).
"""


class TerminalUI:
    def show_text(self, text: str) -> None:
        print(f"\n{text}")

    def show_tool_call(self, name: str, args: dict) -> None:
        print(f"\n[tool] {name}({args})")

    def show_error(self, message: str) -> None:
        print(f"\n[error] {message}")

    def confirm(self, message: str) -> bool:
        answer = input(f"\n[confirm] {message}\nAllow? [y/N] ").strip().lower()
        return answer in {"y", "yes"}


def describe_provider(service: AgentService) -> str:
    llm = service.llm
    session = service.sessions.session_id if service.sessions else "-"
    return (f"Provider: {llm.name} | Model: {getattr(llm, 'model', '?')} | "
            f"Session: {session}")


def _cmd_provider(service: AgentService, ui: TerminalUI, args: list[str]) -> None:
    if not args:
        ui.show_text(describe_provider(service)
                     + f"\nAvailable: {', '.join(PROVIDER_NAMES)}")
        return
    name = args[0].lower()
    if name not in PROVIDER_NAMES:
        ui.show_error(f"Unknown provider '{name}'. Available: {', '.join(PROVIDER_NAMES)}")
        return
    model = args[1] if len(args) > 1 else None
    try:
        service.update_llm(create_llm(name, model))
    except (LLMError, ValueError, RuntimeError) as e:
        # A failed switch must not kill the session: the old provider stays active.
        ui.show_error(f"Could not switch to '{name}': {e}")
        return
    ui.show_text(f"Switched provider. {describe_provider(service)}")


def _cmd_memory(service: AgentService, ui: TerminalUI) -> None:
    counted = service.memory.counted()
    if not counted:
        ui.show_text("The active window is empty.")
        return
    lines = [f"Active window: {len(counted)}/{service.memory.max_messages} "
             f"user/assistant messages."]
    for i, m in enumerate(counted, 1):
        text = " ".join(m.text.split()) or "(tool call only)"
        lines.append(f"  {i:>2}. {m.role:<9} {text[:90]}{'...' if len(text) > 90 else ''}")
    ui.show_text("\n".join(lines))


def _cmd_chats(service: AgentService, ui: TerminalUI) -> None:
    if service.sessions is None:
        ui.show_error("Session persistence is not enabled.")
        return
    sessions = service.sessions.list_sessions()
    if not sessions:
        ui.show_text("No saved sessions yet.")
        return
    lines = ["Saved sessions (most recent first):"]
    for s in sessions:
        marker = "*" if s.id == service.sessions.session_id else " "
        lines.append(f" {marker} {s.id}  [{s.message_count:>3} msgs] "
                     f"{s.provider}/{s.model}  {s.title}")
    lines.append("\nUse /resume <session-id> to continue one.")
    ui.show_text("\n".join(lines))


def _cmd_resume(service: AgentService, ui: TerminalUI, args: list[str]) -> None:
    if not args:
        ui.show_error("Usage: /resume <session-id>   (see /chats)")
        return
    if service.resume(args[0]):
        ui.show_text(f"Resumed session {args[0]}. {describe_provider(service)}")
        _cmd_memory(service, ui)
    else:
        ui.show_error(f"No saved session with id '{args[0]}'. Use /chats to list them.")


def handle_command(service: AgentService, ui: TerminalUI, line: str) -> bool:
    """Run a slash command. Returns False when the user asked to exit."""
    parts = line[1:].split()
    name = parts[0].lower() if parts else ""
    args = parts[1:]

    if name in {"exit", "quit", "q"}:
        return False
    if name == "help":
        ui.show_text(HELP)
    elif name == "provider":
        _cmd_provider(service, ui, args)
    elif name == "memory":
        _cmd_memory(service, ui)
    elif name == "clear":
        service.clear_memory()
        ui.show_text("Active window cleared. The saved chat is untouched.")
    elif name == "chats":
        _cmd_chats(service, ui)
    elif name == "resume":
        _cmd_resume(service, ui, args)
    elif name == "new":
        if service.sessions is None:
            ui.show_error("Session persistence is not enabled.")
        else:
            service.clear_memory()
            ui.show_text(f"Started a new session: {service.sessions.start()}")
    else:
        ui.show_error(f"Unknown command '/{name}'. Type /help.")
    return True


def run_repl(service: AgentService, ui: TerminalUI, banner: list[str]) -> None:
    for line in banner:
        print(line)
    print("Type /help for commands, or /exit to quit.")
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        if user.startswith("/"):
            if not handle_command(service, ui, user):
                break
            continue
        service.ask(user)
