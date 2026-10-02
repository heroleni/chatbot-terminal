# AI-Assisted Development Trace

This folder documents how AI coding assistants were used while building this
project, as required by the challenge.

## Assistants and models used

| Tool / agent | Model | What it was used for |
|---|---|---|
| Claude (claude.ai web chat) | `claude-opus-5` | Project audit against the acceptance criteria; designing and implementing the rolling memory window, the SQLite session repository and the terminal command dispatcher; diagnosing the Gemini 503 failure; translating the codebase to English; writing the knowledge-base documents and the two custom skills. |
| Google Gemini | `gemini-2.5-flash` | Used *by* the application at runtime as a chat provider and for RAG embeddings (`gemini-embedding-001`). Not a coding assistant. |

> Fill in any additional assistant you used (GitHub Copilot, Cursor, ChatGPT,
> Claude Code...) with the model name and what it contributed. Every row here
> must correspond to entries in `prompts.jsonl`.

## What was written with assistance vs. by hand

- **Assisted:** the hexagonal skeleton (ports, adapters, composition root), the
  three provider adapters, the BM25 retriever, the memory window and session
  layer, the SQL schema for pgvector, and the documentation.
- **Reviewed and adjusted by hand:** provider model names, environment
  configuration, Supabase project setup, and all credentials.

Every AI-produced file was read and executed before being committed. The test
suite in `tests/` runs without network access and is the main verification gate.

## Log format

`prompts.jsonl` is one JSON object per line, with these fields:

| Field | Meaning |
|---|---|
| `timestamp` | ISO-8601 UTC timestamp of when the prompt was sent |
| `agent` | The tool the prompt was sent to (e.g. `claude.ai`) |
| `model` | The model that answered |
| `prompt` | The prompt text that was sent |
| `purpose` | Why the prompt was sent — what it was meant to achieve |

## Secrets policy

No API keys, tokens, passwords, connection strings or Supabase URLs appear in
this log. Prompts that originally contained a credential were redacted to
`<REDACTED>` before being recorded. The `.env` file is listed in `.gitignore`
and is never committed; `.env.example` carries placeholder values only.
