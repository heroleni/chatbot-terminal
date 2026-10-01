# Conversation Memory and Persistent Sessions

## Two kinds of memory

A chatbot needs two distinct forms of memory, and conflating them produces a
system that is either expensive or forgetful.

The first is the active context: the messages actually sent to the model on each
call. This is bounded, because every token costs money and latency, and because
context windows are finite. The second is the durable record: everything that
was ever said, kept on disk so a conversation survives closing the terminal.
These have opposite requirements. The active context must be small and recent;
the durable record must be complete and permanent.

Keeping them separate is what allows a bounded window over an unbounded history.

## The rolling window

This project keeps a rolling window of the ten most recent user and assistant
messages. When an eleventh arrives, the oldest is dropped. Every model call
receives the system instructions plus that window, and nothing else.

Tool messages complicate the counting. A single user question can produce an
assistant message requesting three tool calls, a tool message carrying their
results, and a further assistant message with the final answer. Counting tool
messages against the ten-message budget would mean a question that triggered
heavy tool use could evict the user's own question from the window.

The window therefore counts only user and assistant messages. Tool messages are
retained when they sit between counted messages, because every provider SDK
rejects a tool result that is not preceded by the tool call that produced it. An
orphaned tool result is not merely untidy; it is an API error.

Trimming has one further constraint. After dropping an old message, the window
must not begin with a tool message, for the same reason. The trimming logic
removes an evicted message together with any tool messages that followed it
before the next counted message, and then drops any leading tool messages that
remain.

## Rollback on provider failure

When a model call fails, the window is already holding the user's message and
possibly a partial exchange. Leaving that in place corrupts the next attempt:
the model would see a question it never answered, or an assistant turn truncated
mid-tool-call.

The conversation loop records the window's length before processing a message
and rolls back to that checkpoint when a provider error surfaces. The user sees
an error, the window returns to its prior state, and retrying behaves exactly as
if the failed attempt had never happened.

## Durable sessions

Sessions are persisted in SQLite. Two tables carry the data: one holding session
metadata (a stable identifier, a title, the provider and model, creation and
update timestamps) and one holding messages (the session they belong to, their
position in order, their role and a JSON payload).

Storing each message as JSON rather than as plain text preserves tool calls and
tool results across a restart, so a resumed conversation retains the structure
of what happened and not merely the prose.

One field is deliberately not persisted. Native provider payloads, such as
Gemini thought signatures or Anthropic content blocks, are vendor-specific
objects that are frequently not serializable and that expire with the model
version that produced them. A session resumed under a different provider than
the one that created it would be corrupted by replaying them. Persisting plain
text and tool structure means a session started on one provider can be continued
on another.

Session identifiers combine the creation date with a short random suffix, giving
something short enough to type after a resume command and stable enough to
reference later. Titles are derived from the first user message, so listing
saved sessions shows what each one was about rather than a column of opaque
identifiers.

## Resuming

Resuming loads the session's full stored history and rebuilds the active window
from it, applying the same trimming rules. The user gets back the ten most
recent exchanges as live context while the complete history stays on disk. New
messages are appended to the same session.

The distinction between clearing memory and starting a new session is worth
making explicit. Clearing empties the active window while leaving the stored
session intact, which is what someone wants when changing subject within one
conversation. Starting a new session opens a fresh record, leaving the previous
one listed and resumable. Neither operation deletes anything.

## Why this lives in the application layer

The rolling window and the session service contain no SQL and no file paths.
They depend on a repository port that declares what storage must provide:
create, append, load, list and touch. SQLite is one implementation of that port.
A JSON file implementation would satisfy the same contract and require no change
to the conversation logic, which is the practical test of whether the boundary
was drawn in the right place.

This also makes the behaviour testable without a database. The window's trimming
rules, the rollback on failure and the resume path can all be exercised against
an in-memory fake, which means the tests run in milliseconds and in any
environment.
