# Provider Architecture: One Contract, Three SDKs

## Why a shared contract exists

A terminal chatbot that speaks to more than one language model provider faces a
structural problem before it faces any product problem. Each vendor ships its
own SDK, its own message shape, its own tool-calling encoding and its own error
taxonomy. The naive solution is a conditional: if the provider is OpenAI do one
thing, if it is Anthropic do another. That conditional multiplies. It appears in
the conversation loop, in the retry logic, in the history serializer and in the
tests. Every new provider doubles the surface area that can break, and the
conversation logic becomes impossible to reason about independently.

The alternative is a single contract that the conversation logic depends on, and
one small adapter per vendor that translates between that contract and the
vendor's SDK. This is the Adapter pattern, and in a hexagonal architecture the
contract is called a port.

## The LLMPort contract

The port in this project is deliberately small. It declares a name, and a single
generate method that receives the system instructions, a sequence of normalized
messages and a sequence of tool specifications, and returns a normalized
response. Nothing else. There is no streaming, no temperature, no token budget
in the contract, because those are vendor concerns that each adapter can resolve
from configuration without leaking into the core.

Smallness matters. A port with twenty methods is a port that every adapter
implements badly. A port with one method is a port that a new vendor can satisfy
in roughly eighty lines, which is the actual measure of whether the abstraction
is working.

## Normalized messages

The domain defines its own Message type with a role, text, a list of tool calls
and a list of tool results. The role is restricted to user, assistant or tool.
This type imports nothing from any SDK, which is the property that makes it
testable: a fake model that returns scripted Message objects can exercise the
entire conversation loop without a network connection or an API key.

One field deserves explanation. Messages carry a provider_data dictionary that
holds the native payload of whichever vendor produced them. Gemini 3 returns
thought signatures that must be echoed back verbatim on the next turn or the
model loses its reasoning chain. Anthropic returns content blocks that behave
the same way. Rather than modelling every vendor's internal representation in
the domain, the adapter stashes the raw object under its own key and reads only
that key back. Each adapter ignores keys it did not write.

This field is intentionally excluded from persistence. Native payloads are
vendor objects, often not JSON-serializable, and they expire with the model
version that produced them. A resumed session therefore rebuilds plain text and
tool structure, which every provider accepts, rather than attempting to replay
another vendor's internal state.

## Tool calling across three encodings

Tool calling is where the three SDKs diverge most sharply, and it is worth
spelling out because the differences are easy to get wrong.

OpenAI expects tools as a list of objects with a type of function and a nested
function object holding name, description and a JSON Schema for parameters. Tool
results come back as separate messages with a tool role, each carrying the
tool_call_id it answers. The arguments arrive as a JSON string that must be
parsed, and malformed JSON is a real failure mode worth defending against.

Anthropic expects tools with name, description and an input_schema field. Tool
calls appear as tool_use blocks inside the assistant's content array, and every
tool result for a turn travels inside a single user message as a list of
tool_result blocks keyed by tool_use_id. Splitting those results across several
messages is rejected by the API.

Gemini expects function declarations with name, description and parameters, but
it rejects an object schema that declares no properties. A tool that takes no
arguments must therefore omit the parameters key entirely rather than sending an
empty object. Results are returned as function response parts.

All three encodings are derived from the same domain ToolSpec, which carries a
name, a description and a plain JSON Schema dictionary. The translation lives
entirely inside each adapter.

## Error handling and recoverable failures

Each SDK raises its own exception hierarchy. The adapters catch those and raise
a single domain LLMError carrying the HTTP status code and a human-readable
hint. The conversation loop catches only LLMError, which means it does not
import any SDK and does not need to know which vendor is active.

Status codes fall into two classes. Permanent failures (400, 401, 403, 404) mean
a bad key, a bad model name or a stale SDK, and retrying them is pointless. The
hint tells the user which of those to check. Transient failures (429, 500, 502,
503, 504) mean the provider is rate limiting or overloaded, and these are
genuinely recoverable: the correct response is exponential backoff with jitter,
retrying a small number of times before surfacing an error.

A 503 carrying the message that a model is experiencing high demand is almost
always a preview or newly released model rather than a problem with the caller's
code or credentials. Switching to a generally available model usually resolves
it immediately, which is why the error hint says so explicitly instead of
sending the user to check their API key.

The important property is that none of these failures terminate the
application. The conversation window is rolled back to its state before the
failed turn, so the history stays consistent, an error is printed, and the user
gets their prompt back and can retry or switch provider.

## Switching provider at runtime

Because the conversation loop holds a reference to an LLMPort and nothing else,
switching provider is a single assignment. The terminal command constructs a new
adapter through the factory and hands it to the agent, which replaces its
reference and records the new provider and model on the active session.

A failed switch must not destroy the session. If constructing the new adapter
raises, because the key for that vendor is missing or the model name is wrong,
the error is reported and the previous adapter stays active. The user loses
nothing.

The factory itself imports lazily. Only the SDK actually being used is imported,
so a user who has configured only one vendor is not forced to install the other
two. A missing SDK produces a message naming the package and the install
command, rather than an import traceback.
