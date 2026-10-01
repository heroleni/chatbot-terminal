---
name: knowledge-lookup
description: Answer a question strictly from the document knowledge base, citing every source and refusing to answer when the retrieved context is insufficient. Use it whenever the user asks about the project's documentation, architecture, RAG pipeline, memory or providers.
---

# Knowledge Lookup

## Purpose

Answer questions from the ingested document knowledge base (Supabase + pgvector)
with grounded, attributable answers. This skill exists to make the difference
between "the model recalled something" and "the knowledge base says this"
visible to the reader.

## Expected input

A natural-language question about the project's documents. Examples:

- "What chunk size and overlap does the pipeline use, and why?"
- "How does the system avoid hallucinating when retrieval finds nothing?"
- "Why is provider_data excluded from persisted sessions?"

## Expected output

A short answer in English, where every factual claim is followed by a citation
in the form `[source: <path> #<chunk>]`, and a `Sources` list at the end. If the
retrieved context does not support an answer, the output is an explicit
statement that the knowledge base does not contain enough information, with no
answer attempted.

## How it is invoked

The model calls `use_skill` with `name: knowledge-lookup`, which loads these
instructions. In the terminal the user simply asks a documentation question; the
agent routes to this skill on its own.

## Procedure

1. Call `search_knowledge` with the most specific keywords from the question.
   Prefer concrete nouns and identifiers over full sentences.
2. Read the returned chunks. Each one arrives with its source path, chunk
   position and similarity score.
3. Judge the results before using them:
   - If the chunks clearly address the question, continue to step 4.
   - If they are weakly related, call `search_knowledge` once more with
     different wording or synonyms. Stop after three searches total.
   - If nothing relevant comes back, go to step 5.
4. Write the answer using only what the chunks state. Cite after each claim.
   Do not merge in background knowledge, and do not smooth over a gap in the
   sources with a plausible-sounding sentence.
5. When the context is insufficient, say so plainly:
   "The knowledge base does not contain enough information to answer that."
   Then name what was searched for. Do not guess, and do not invent a source.

## Constraints

- Never cite a document, path or chunk number that did not appear in the
  retrieval results.
- Never present a training-data fact as if it came from the knowledge base.
- Low-similarity chunks are filtered out before they reach you. If you received
  nothing, that is information: the corpus does not cover the question.
