"""Core use case: the agent loop.

It knows nothing about Gemini, OpenAI or Anthropic, nor about the disk or the
terminal. It only talks to ports. Switching provider therefore changes one
reference and leaves the conversation logic untouched.
"""
from __future__ import annotations

from codeagent.application.memory import ConversationMemory
from codeagent.application.session_service import SessionService
from codeagent.application.tools import Tool
from codeagent.domain.errors import LLMError
from codeagent.domain.models import Message, ToolCall, ToolResult
from codeagent.domain.ports import LLMPort, OutputPort, SkillRepositoryPort


class AgentService:
    def __init__(self, llm: LLMPort, tools: list[Tool], skills: SkillRepositoryPort,
                 output: OutputPort, system_prompt: str, max_steps: int = 12,
                 memory: ConversationMemory | None = None,
                 sessions: SessionService | None = None):
        self._llm = llm
        self._tools = {t.spec.name: t for t in tools}
        self._skills = skills
        self._out = output
        self._base_prompt = system_prompt
        self._max_steps = max_steps
        # `is None`, not `or`: ConversationMemory defines __len__, so an empty
        # window is falsy and `or` would silently discard the injected instance.
        self.memory = ConversationMemory() if memory is None else memory
        self.sessions = sessions

    # ---- provider switching -----------------------------------------------
    @property
    def llm(self) -> LLMPort:
        return self._llm

    def update_llm(self, new_llm: LLMPort) -> None:
        """Swap the LLM port at runtime. The conversation logic does not change."""
        self._llm = new_llm
        if self.sessions is not None:
            self.sessions.set_provider(new_llm.name, getattr(new_llm, "model", ""))

    # ---- memory ------------------------------------------------------------
    def clear_memory(self) -> None:
        """Clear the active window. The persisted session is left untouched."""
        self.memory.clear()

    def resume(self, session_id: str) -> bool:
        """Rebuild the active window from a stored session's last messages."""
        if self.sessions is None:
            return False
        history = self.sessions.resume(session_id)
        if history is None:
            return False
        self.memory.replace(history)
        return True

    def _record(self, message: Message) -> None:
        """Append to the rolling window and persist to the session store."""
        self.memory.append(message)
        if self.sessions is not None:
            self.sessions.record(message)

    def _system(self) -> str:
        text = self._base_prompt
        infos = self._skills.list_skills()
        if infos:
            lines = "\n".join(f"- {s.name}: {s.description}" for s in infos)
            text += ("\n\nAvailable skills (load them with the use_skill tool when they "
                     f"apply to the task):\n{lines}")
        return text

    def _run_tool(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return f"Unknown tool: {call.name}"
        try:
            return tool.run(**call.args) or "(no output)"
        except Exception as e:  # the error goes back to the model so it can correct itself
            return f"Error: {e}"

    def ask(self, text: str) -> str:
        """Process one user message and return the final assistant text."""
        checkpoint = len(self.memory)
        self._record(Message.user(text))
        system = self._system()
        specs = [t.spec for t in self._tools.values()]
        final = ""

        for _ in range(self._max_steps):
            try:
                # Only the system instructions plus the rolling window are sent.
                reply = self._llm.generate(system, self.memory.window(), specs).message
            except LLMError as e:
                self.memory.rollback(checkpoint)  # leave the window consistent for a retry
                self._out.show_error(f"Model error ({e.code or 'no code'}): {e}"
                                     + (f"\n   {e.hint}" if e.hint else ""))
                return ""

            self._record(reply)
            if reply.text.strip():
                self._out.show_text(reply.text)
                final = reply.text
            if not reply.tool_calls:
                return final

            results = []
            for call in reply.tool_calls:
                self._out.show_tool_call(call.name, call.args)
                results.append(ToolResult(call, self._run_tool(call)))
            self._record(Message(role="tool", tool_results=results))

        self._out.show_error(f"Reached the {self._max_steps}-step limit without a final answer.")
        return final
