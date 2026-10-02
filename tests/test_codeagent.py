"""Network-free test suite: it uses a scripted fake LLM.

Run with:  python -m unittest -v
"""
import os
import tempfile
import unittest
from types import SimpleNamespace

from codeagent.adapters.retrieval.bm25 import Bm25Retriever
from codeagent.adapters.skills_fs import FileSystemSkillRepository
from codeagent.adapters.workspace import LocalWorkspace
from codeagent.adapters.sessions_sqlite import SqliteSessionRepository
from codeagent.application.agent_service import AgentService
from codeagent.application.memory import ConversationMemory
from codeagent.application.session_service import SessionService
from codeagent.application.rag import RetrievalTool, UseSkillTool
from codeagent.application.tools import FunctionTool, WriteFileTool
from codeagent.domain.errors import LLMError
from codeagent.domain.models import (LLMResponse, Message, SkillInfo, ToolCall, ToolResult,
                                     ToolSpec, object_schema)


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class ScriptedLLM:
    name = "fake"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def generate(self, system, messages, tools):
        self.calls.append((system, list(messages), list(tools)))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(r)


class FakeUI:
    def __init__(self, answer=True):
        self.texts, self.calls, self.errors, self.answer = [], [], [], answer

    def show_text(self, t): self.texts.append(t)
    def show_tool_call(self, n, a): self.calls.append((n, a))
    def show_error(self, m): self.errors.append(m)
    def confirm(self, m): return self.answer


class NoSkills:
    def list_skills(self): return []
    def load(self, name): return None


def assistant(text="", calls=()):
    return Message(role="assistant", text=text, tool_calls=list(calls))


ADD = FunctionTool("add", "Add", lambda a, b: a + b,
                     {"a": {"type": "number"}, "b": {"type": "number"}}, ["a", "b"])


def make_service(llm, ui, tools=(ADD,), skills=None, max_steps=12, window=10, sessions=None):
    return AgentService(llm, list(tools), skills or NoSkills(), ui, "base", max_steps,
                        memory=ConversationMemory(window), sessions=sessions)


class AgentLoopTests(unittest.TestCase):
    def test_tool_loop_and_history(self):
        llm = ScriptedLLM([assistant(calls=[ToolCall("add", {"a": 2, "b": 3}, "c1")]),
                           assistant("The result is 5")])
        ui = FakeUI()
        svc = make_service(llm, ui)
        self.assertEqual(svc.ask("add 2+3"), "The result is 5")
        self.assertEqual([m.role for m in svc.memory.messages],
                         ["user", "assistant", "tool", "assistant"])
        self.assertEqual(svc.memory.messages[2].tool_results[0].output, "5")
        self.assertEqual(ui.calls, [("add", {"a": 2, "b": 3})])

    def test_unknown_tool_and_tool_exception_go_back_to_model(self):
        llm = ScriptedLLM([assistant(calls=[ToolCall("nope", {}, "1"), ToolCall("add", {"a": 1}, "2")]),
                           assistant("done")])
        svc = make_service(llm, FakeUI())
        svc.ask("x")
        outs = [r.output for r in svc.memory.messages[2].tool_results]
        self.assertIn("Unknown tool", outs[0])
        self.assertTrue(outs[1].startswith("Error"))

    def test_max_steps_guard(self):
        loop = [assistant(calls=[ToolCall("add", {"a": 1, "b": 1}, str(i))]) for i in range(5)]
        ui = FakeUI()
        make_service(ScriptedLLM(loop), ui, max_steps=3).ask("x")
        self.assertTrue(any("step limit" in e for e in ui.errors))

    def test_llm_error_rolls_back_memory_and_does_not_crash(self):
        ui = FakeUI()
        svc = make_service(ScriptedLLM([LLMError("boom", code=401, hint="check the key"),
                                        assistant("recovered")]), ui)
        self.assertEqual(svc.ask("hi"), "")
        self.assertEqual(svc.memory.messages, [])  # window stays consistent for a retry
        self.assertIn("check the key", ui.errors[0])
        self.assertEqual(svc.ask("hi again"), "recovered")  # app keeps running

    def test_skills_listed_in_system_prompt(self):
        class Skills(NoSkills):
            def list_skills(self): return [SkillInfo("explain-code", "Explains code")]
        llm = ScriptedLLM([assistant("ok")])
        make_service(llm, FakeUI(), skills=Skills()).ask("hi")
        self.assertIn("explain-code: Explains code", llm.calls[0][0])

    def test_write_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            ws = LocalWorkspace(d)
            self.assertIn("declined", WriteFileTool(ws, FakeUI(answer=False)).run("a.txt", "x"))
            self.assertFalse(os.path.exists(os.path.join(d, "a.txt")))
            WriteFileTool(ws, FakeUI(answer=True)).run("sub/a.txt", "x")
            self.assertTrue(os.path.exists(os.path.join(d, "sub", "a.txt")))


class AdapterAndInfraTests(unittest.TestCase):
    def test_workspace_blocks_escape(self):
        with tempfile.TemporaryDirectory() as d:
            ws = LocalWorkspace(d)
            for bad in ("../x", "/etc/passwd"):
                with self.assertRaises(PermissionError):
                    ws.read_text(bad)

    def test_skills_repo_crlf_and_missing_skill_md(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "ok"))
            os.makedirs(os.path.join(d, "bad"))
            with open(os.path.join(d, "ok", "SKILL.md"), "w", newline="", encoding="utf-8") as f:
                f.write("---\r\nname: my-skill\r\ndescription: Does a thing\r\n---\r\n\r\nBody\r\n")
            with open(os.path.join(d, "bad", "READ.md"), "w", encoding="utf-8") as f:
                f.write("x")
            repo = FileSystemSkillRepository(d)
            self.assertEqual(repo.list_skills(), [SkillInfo("my-skill", "Does a thing")])
            self.assertEqual(repo.skipped, ["bad"])
            self.assertEqual(repo.load("my-skill"), "Body")
            self.assertIn("no skill named", UseSkillTool(repo).run("other"))

    def test_bm25_ranks_relevant_doc_and_reindexes(self):
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "a.md"), "Hexagonal architecture uses ports and adapters.")
            write(os.path.join(d, "b.md"), "Potato omelette recipe with onion.")
            r = Bm25Retriever(d)
            top = r.search("what are hexagonal ports?", 2)
            self.assertEqual(top[0].source, "a.md")
            self.assertEqual(r.search("zzzz nonexistent"), [])
            write(os.path.join(d, "c.md"), "Embeddings capture semantics.")
            self.assertEqual(r.search("embeddings")[0].source, "c.md")

    def test_retrieval_tool_formats_citations_and_no_results(self):
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "a.md"), "BM25 scores fragments by term frequency.")
            tool = RetrievalTool("search_knowledge", "d", Bm25Retriever(d))
            self.assertIn("source: a.md", tool.run("bm25"))
            self.assertIn("not contain enough information", tool.run("nonexistent"))
            self.assertIn("source: a.md", tool.run("bm25", 4.0))  # Gemini sends floats

    def test_factory_unknown_provider(self):
        from codeagent.adapters.llm.factory import create_llm
        with self.assertRaises(ValueError):
            create_llm("nope")

    # ---- format conversion (no network) ----
    def _history(self):
        call = ToolCall("add", {"a": 1, "b": 2}, "call_1")
        return [Message.user("hi"),
                Message(role="assistant", text="working", tool_calls=[call]),
                Message(role="tool", tool_results=[ToolResult(call, "3")]),
                Message(role="assistant", text="3")]

    def test_openai_format(self):
        from codeagent.adapters.llm import openai_adapter as oa
        msgs = oa.to_openai_messages("SYS", self._history())
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["arguments"], '{"a": 1, "b": 2}')
        self.assertEqual(msgs[3], {"role": "tool", "tool_call_id": "call_1", "content": "3"})
        resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, tool_calls=[SimpleNamespace(id="x", function=SimpleNamespace(
                name="add", arguments='{"a": 1}'))]))])
        parsed = oa.parse_response(resp)
        self.assertEqual(parsed.tool_calls[0].args, {"a": 1})
        spec = ToolSpec("t", "d", object_schema({"q": {"type": "string"}}, ["q"]))
        self.assertEqual(oa.to_openai_tools([spec])[0]["function"]["parameters"]["required"], ["q"])

    def test_anthropic_format(self):
        from codeagent.adapters.llm import anthropic_adapter as aa
        msgs = aa.to_anthropic_messages(self._history())
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant"])
        self.assertEqual(msgs[1]["content"][1]["type"], "tool_use")
        self.assertEqual(msgs[2]["content"][0],
                         {"type": "tool_result", "tool_use_id": "call_1", "content": "3"})
        resp = SimpleNamespace(content=[
            SimpleNamespace(type="text", text="hi"),
            SimpleNamespace(type="tool_use", id="tu1", name="add", input={"a": 1, "b": 2})])
        parsed = aa.parse_response(resp)
        self.assertEqual((parsed.text, parsed.tool_calls[0].id), ("hi", "tu1"))
        self.assertIn("anthropic", parsed.provider_data)
        self.assertEqual(aa.to_anthropic_messages([parsed])[0]["content"], resp.content)

    def test_gemini_format(self):
        from google.genai import types
        from codeagent.adapters.llm import gemini_adapter as ga
        contents = ga.to_contents(self._history())
        self.assertEqual([c.role for c in contents], ["user", "model", "user", "model"])
        self.assertEqual(contents[1].parts[1].function_call.name, "add")
        self.assertEqual(contents[2].parts[0].function_response.response, {"result": "3"})
        decls = ga.to_declarations([ToolSpec("now", "d"), ToolSpec("t", "d", object_schema({"q": {"type": "string"}}))])
        self.assertNotIn("parameters", decls[0])
        self.assertIn("parameters", decls[1])
        native = types.Content(role="model", parts=[
            types.Part(text="ok"),
            types.Part(function_call=types.FunctionCall(name="add", args={"a": 1}))])
        parsed = ga.parse_response(SimpleNamespace(candidates=[SimpleNamespace(content=native)]))
        self.assertEqual(parsed.tool_calls[0].name, "add")
        self.assertIs(ga.to_contents([parsed])[0], native)  # native content preserved
        with self.assertRaises(LLMError):
            ga.parse_response(SimpleNamespace(candidates=[]))



class WebToolsTests(unittest.TestCase):
    def test_web_tools_delegate_to_port_and_clamp(self):
        from codeagent.application.tools import FetchUrlTool, WebSearchTool

        class FakeWeb:
            def search(self, q, n): return f"search:{q}:{n}"
            def fetch(self, u): return f"fetch:{u}"

        web = FakeWeb()
        self.assertEqual(WebSearchTool(web).run("hi"), "search:hi:5")
        self.assertEqual(WebSearchTool(web).run("hi", 99), "search:hi:10")
        self.assertEqual(FetchUrlTool(web).run("https://x.com"), "fetch:https://x.com")

    def test_fetch_rejects_bad_url(self):
        from codeagent.adapters.web import WebClient
        self.assertIn("Invalid URL", WebClient().fetch("ftp://algo"))


if __name__ == "__main__":
    unittest.main()


class MemoryWindowTests(unittest.TestCase):
    """The rolling 10-message window is an explicit acceptance criterion."""

    def test_keeps_only_the_ten_most_recent_counted_messages(self):
        mem = ConversationMemory(10)
        for i in range(14):
            mem.append(Message.user(f"m{i}"))
        counted = mem.counted()
        self.assertEqual(len(counted), 10)
        self.assertEqual(counted[0].text, "m4")   # the oldest ones were dropped
        self.assertEqual(counted[-1].text, "m13")

    def test_tool_messages_do_not_consume_the_window_budget(self):
        mem = ConversationMemory(10)
        call = ToolCall("add", {"a": 1}, "c1")
        for i in range(6):
            mem.append(Message.user(f"q{i}"))
            mem.append(Message(role="assistant", tool_calls=[call]))
            mem.append(Message(role="tool", tool_results=[ToolResult(call, "ok")]))
        self.assertEqual(len(mem.counted()), 10)
        # A window must never start with an orphan tool result.
        self.assertNotEqual(mem.window()[0].role, "tool")

    def test_only_the_window_is_sent_to_the_provider(self):
        llm = ScriptedLLM([assistant("ok")])
        svc = make_service(llm, FakeUI(), window=4)
        for i in range(6):
            svc.memory.append(Message.user(f"old{i}"))
        llm.replies = [assistant("ok")]
        svc.ask("newest")
        sent = llm.calls[0][1]
        self.assertLessEqual(len([m for m in sent if m.role in ("user", "assistant")]), 4)
        self.assertEqual(sent[-1].text, "newest")

    def test_clear_empties_the_window(self):
        mem = ConversationMemory(10)
        mem.append(Message.user("hi"))
        mem.clear()
        self.assertEqual(mem.messages, [])


class SessionPersistenceTests(unittest.TestCase):
    """Saved chats must survive an application restart."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "chats.db")

    def test_sessions_survive_a_restart_with_tool_structure(self):
        repo = SqliteSessionRepository(self.db)
        svc = SessionService(repo, "openai", "gpt-4o-mini")
        sid = svc.start()
        call = ToolCall("search_knowledge", {"query": "chunking"}, "t1")
        svc.record(Message.user("How does chunking work?"))
        svc.record(Message(role="assistant", text="Looking.", tool_calls=[call],
                           provider_data={"gemini": object()}))
        svc.record(Message(role="tool", tool_results=[ToolResult(call, "700 tokens")]))

        reopened = SqliteSessionRepository(self.db)  # simulates a restart
        history = reopened.load(sid)
        self.assertEqual([m.role for m in history], ["user", "assistant", "tool"])
        self.assertEqual(history[1].tool_calls[0].name, "search_knowledge")
        self.assertEqual(history[2].tool_results[0].output, "700 tokens")
        # provider_data is deliberately not persisted: it is vendor-specific.
        self.assertEqual(history[1].provider_data, {})

    def test_list_sessions_titles_by_first_user_message(self):
        repo = SqliteSessionRepository(self.db)
        svc = SessionService(repo, "anthropic", "claude-sonnet-5-5")
        sid = svc.start()
        svc.record(Message.user("Explain the retrieval threshold"))
        listed = repo.list_sessions()
        self.assertEqual(listed[0].id, sid)
        self.assertEqual(listed[0].title, "Explain the retrieval threshold")
        self.assertEqual(listed[0].provider, "anthropic")

    def test_unknown_session_id_returns_none(self):
        self.assertIsNone(SqliteSessionRepository(self.db).load("does-not-exist"))

    def test_resume_rebuilds_the_window_from_the_last_messages(self):
        repo = SqliteSessionRepository(self.db)
        sessions = SessionService(repo, "openai", "gpt-4o-mini")
        sid = sessions.start()
        for i in range(14):
            sessions.record(Message.user(f"m{i}"))

        svc = make_service(ScriptedLLM([]), FakeUI(), window=10,
                           sessions=SessionService(repo, "openai", "gpt-4o-mini"))
        self.assertTrue(svc.resume(sid))
        self.assertEqual(len(svc.memory.counted()), 10)
        self.assertEqual(svc.memory.counted()[-1].text, "m13")
        self.assertFalse(svc.resume("nope"))

    def test_new_messages_are_saved_back_into_the_resumed_session(self):
        repo = SqliteSessionRepository(self.db)
        sid = SessionService(repo, "openai", "gpt-4o-mini").start()
        svc = make_service(ScriptedLLM([assistant("answer")]), FakeUI(),
                           sessions=SessionService(repo, "openai", "gpt-4o-mini"))
        svc.resume(sid)
        svc.ask("a question")
        roles = [m.role for m in repo.load(sid)]
        self.assertEqual(roles, ["user", "assistant"])
