# Multi-Provider Terminal Chatbot

A terminal chatbot that runs on **OpenAI, Anthropic or Gemini** behind a single
provider contract, keeps a rolling 10-message memory, persists chat sessions in
SQLite, and answers from a RAG knowledge base stored in Supabase (PostgreSQL +
pgvector).

The architecture is hexagonal: `domain/` holds pure types and ports,
`application/` holds the use cases, and `adapters/` holds every SDK, the disk,
the database and the terminal. The conversation logic lives in
`application/agent_service.py` and is **not** duplicated per provider.

---

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
```

Then open `.env` and fill in your own credentials. Every required variable is
listed in `.env.example`; no real key is ever committed to this repository.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | Official OpenAI SDK (default `gpt-6-luna`) |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Official Anthropic SDK |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Gemini chat + RAG embeddings |
| `LLM_PROVIDER` | Provider used at startup |
| `SUPABASE_URL` / `SUPABASE_KEY` | Hosted vector store |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | Embedding configuration |
| `RAG_CHUNK_TOKENS` / `RAG_OVERLAP_TOKENS` | Chunking (default 700 / 100 = 14.3 % overlap) |
| `RAG_TOP_K` / `RAG_MIN_SCORE` | Retrieval tuning |

Credentials are read from environment variables only. Nothing is hardcoded.

## 2. Deployed services

The terminal client runs locally; every service it consumes is deployed and
reachable over the network. No dependency runs on localhost.

| Service | Role | URL |
| --- | --- | --- |
| OpenAI API | Provider (official SDK) | `https://api.openai.com` |
| Anthropic API | Provider (official SDK) | `https://api.anthropic.com` |
| Google Gemini API | Provider + embeddings | `https://generativelanguage.googleapis.com` |
| Supabase (PostgreSQL + pgvector) | Vector store | `https://<your-project>.supabase.co` |

Set your Supabase project URL in `.env` as `SUPABASE_URL`. Before the first
ingestion, run `data/rag.sql` once in the Supabase SQL editor: it enables the
`vector` extension and creates the chunk table and the similarity-search
function.

## 3. Ingest the knowledge base

Three Markdown documents ship in `data/documents/`. Ingestion loads them,
chunks them with overlap, embeds each chunk and stores the vectors in Supabase:

```bash
python -m rag.ingest --documents data/documents
```

Expected output: 2 chunks per document, 6 in total. Re-running is safe — each
source is replaced rather than duplicated. PDF, Markdown and plain text are
supported; any other extension is skipped with a readable message.

## 4. Run the chatbot

```bash
python main.py                                   # interactive, provider from .env
python main.py --provider anthropic              # pick a provider at startup
python main.py --provider openai --model gpt-6-luna
python main.py --resume <session-id>             # reopen a saved chat
python main.py "explain codeagent/bootstrap.py"  # single question, no REPL
```

The banner shows the active provider and model. The session accepts messages
until you exit.

## 5. Terminal commands

| Command | Effect |
| --- | --- |
| `/help` | List the commands |
| `/provider` | Show the active provider and the available ones |
| `/provider <name> [model]` | Switch provider at runtime without losing the conversation |
| `/memory` | Inspect the active 10-message window |
| `/clear` | Clear the active window (the saved chat is kept) |
| `/chats` | List saved sessions by stable id |
| `/resume <session-id>` | Continue a saved session |
| `/new` | Start a new saved session |
| `/exit` | Quit (also `exit`, `quit`) |

### Memory and saved chats

The active window holds the **10 most recent user/assistant messages**; when an
eleventh arrives the oldest is dropped. Every model call sends the system
instructions plus that window only. Tool messages are not counted, but they stay
attached to the assistant turn that produced them, because every provider SDK
rejects a tool result whose tool call is missing.

Complete sessions are written to SQLite at `data/chats.db` and survive restarts.
`/resume <session-id>` rebuilds the active window from that session's last 10
messages and keeps saving new messages back to it.

## 6. Testing

```bash
python -m unittest -v
```

25 tests run with no network access: a scripted fake LLM drives the agent loop,
and the adapters are checked against their ports.

### Testing the provider switch

```text
> /provider
> /provider openai
> who are you?
> /provider anthropic
> what did I just ask you?
```

The second provider answers from the same window, which shows the conversation
logic is shared.

### Testing the RAG flow

After ingestion, ask something the documents cover and something they do not:

```text
> which providers does this project support?
> what is the capital of Mongolia?
```

The first answer cites its sources as `[source: documents/<file>]`. The second
states there is not enough information in the knowledge base instead of
inventing an answer. `/memory` after each turn shows the window staying capped.

### Testing the saved chats

```text
> /chats                     # note a session id
> /exit
$ python main.py             # restart the process
> /chats                     # the session is still listed
> /resume <session-id>       # the conversation continues
```

## 7. Custom skills

Two reusable skills live in `skills/`, each documenting its purpose, expected
input, expected output and how it is invoked:

- `knowledge-lookup` — answer strictly from the RAG knowledge base with sources.
- `explain-code` — explain a file or symbol from the workspace.

The model loads a skill on demand through the `use_skill` tool; both are usable
in a live demo.

## 8. Error handling

Recoverable failures print a readable message and return you to the prompt
instead of crashing: a missing API key, a provider outage, an empty retrieval
result, an unsupported document, an unknown command, or a bad session id.
Provider overload (HTTP 429/5xx) is retried with exponential backoff before the
error surfaces.

## 9. Project structure

```
codeagent/
  domain/       models, ports, errors        (no SDK imports)
  application/  agent loop, memory, sessions, RAG tools
  adapters/     openai / anthropic / gemini, SQLite, disk, web, terminal
rag/            chunker, embeddings, Supabase vector store, ingestion CLI
data/documents/ the three knowledge-base documents
data/rag.sql    Supabase schema
skills/         the two custom skills
coding-assistance/  AI-assisted development trace
tests/          unit tests
```
