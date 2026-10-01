"""Pure-Python BM25 lexical retriever (no dependencies). Implements RetrieverPort.

Strengths: zero install, works offline, finds exact identifiers (function names,
error strings) and is explainable. Limit: it does not understand synonyms, which
is why the semantic adapter (Supabase + pgvector embeddings) exists behind the
very same RetrieverPort interface.
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter

from codeagent.domain.models import Chunk

STOPWORDS = set("""
a al algo ante con como cual cuando de del desde donde e el ella en entre es esta este
esto estos fue ha han hay la las le les lo los me mi muy no o para pero por que se si
sin sobre su sus te tu un una uno y ya
the a an and are as at be by for from has have in is it its of on or that this to was were with
""".split())

EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", "venv", "env", ".idea", ".vscode",
                "dist", "build", ".agent", ".agents", ".mypy_cache", ".pytest_cache"}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return [t for t in re.findall(r"[a-z0-9_]+", text) if len(t) > 1 and t not in STOPWORDS]


def split_chunks(text: str, size: int = 900) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        while len(p) > size:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:size])
            p = p[size:]
        if not p:
            continue
        if cur and len(cur) + 2 + len(p) > size:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


class Bm25Retriever:
    def __init__(self, root: str, suffixes: tuple[str, ...] = (".md", ".txt"),
                 chunk_chars: int = 900, k1: float = 1.5, b: float = 0.75,
                 max_file_bytes: int = 200_000):
        self.root = os.path.abspath(root)
        self._suffixes = suffixes
        self._chunk_chars, self._k1, self._b = chunk_chars, k1, b
        self._max_bytes = max_file_bytes
        self._signature: tuple = ()
        self._docs: list[tuple[Chunk, Counter, int]] = []
        self._df: Counter = Counter()
        self._avgdl = 1.0

    def _files(self) -> list[str]:
        paths = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".venv")]
            for fn in filenames:
                if fn.lower().endswith(self._suffixes):
                    full = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(full) <= self._max_bytes:
                            paths.append(full)
                    except OSError:
                        pass
        return sorted(paths)

    def _ensure_index(self) -> None:
        files = self._files()
        sig = tuple((p, os.stat(p).st_mtime_ns) for p in files if os.path.exists(p))
        if sig == self._signature:
            return  # the index is still valid; it is rebuilt when files change
        self._signature, self._docs, self._df = sig, [], Counter()
        for path in files:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            rel = os.path.relpath(path, self.root).replace(os.sep, "/")
            for i, piece in enumerate(split_chunks(text, self._chunk_chars), 1):
                tf = Counter(tokenize(piece))
                self._docs.append((Chunk(rel, i, piece), tf, sum(tf.values())))
                self._df.update(tf.keys())
        total = sum(n for _, _, n in self._docs)
        self._avgdl = (total / len(self._docs)) if self._docs else 1.0

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        self._ensure_index()
        q_terms = set(tokenize(query))
        n_docs = len(self._docs)
        scored: list[Chunk] = []
        for chunk, tf, dl in self._docs:
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = f + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                score += idf * f * (self._k1 + 1) / denom
            if score > 0:
                scored.append(Chunk(chunk.source, chunk.position, chunk.text, score))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]
