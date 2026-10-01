---
name: explain-code
description: Explain a source file clearly for a reader who did not write it — what it does, how it is structured, and what could break. Use it when the user asks to explain, review, understand or document a file in the workspace.
---

# Explain Code

## Purpose

Produce a readable explanation of one source file: what it is for, how it is
organised, and where its risks are. The audience is a competent developer who
has never seen this file before.

## Expected input

A path to a file in the working directory, or a description precise enough to
locate one. Examples:

- "Explain codeagent/application/memory.py"
- "What does the session repository actually do?"

## Expected output

An explanation in English with four parts:

1. **Summary** — one sentence on what the file is responsible for.
2. **Structure** — the main classes and functions in the order they matter,
   each with what it does and why it exists.
3. **Flow** — how a typical call moves through the file, when there is one.
4. **Risks** — at most three concrete observations about what could break,
   what is unhandled, or what would be worth improving. Omit this section
   entirely rather than padding it with generic advice.

## How it is invoked

The model calls `use_skill` with `name: explain-code`. In the terminal the user
asks for an explanation of a file and the agent routes here.

## Procedure

1. If the exact path is unknown, call `list_dir` to locate it, or `search_code`
   when only a concept or symbol is known.
2. Read the file with `read_file`. Read it fully before writing anything.
3. If the file calls into another module whose behaviour changes the
   explanation, read that module too. Do not guess what an import does.
4. Write the four sections above. Prefer naming the real function and class
   names over paraphrases, so the reader can follow along in the file.
5. Keep the explanation proportional: a 40-line file does not need 400 words.

## Constraints

- Never modify the file. This skill is read-only unless the user separately
  asks for a change.
- Do not describe code you did not read. If a section is unclear, say which part
  and why, rather than inventing a rationale for it.
- Report what the code does, not what its name suggests it should do. Where the
  two disagree, that disagreement is the most useful thing in the explanation.
