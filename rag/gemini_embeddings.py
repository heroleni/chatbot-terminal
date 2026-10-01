"""Gemini embedding adapter used independently from the active chat provider."""
from __future__ import annotations

import os
from math import sqrt
from typing import Sequence

from google import genai
from google.genai import types


class GeminiEmbeddingAdapter:
    def __init__(self, model: str | None = None, dimensions: int | None = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for RAG embeddings.")

        self.model = model or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.dimensions = int(dimensions or os.getenv("EMBEDDING_DIMENSIONS", "768"))
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _normalize(values: Sequence[float]) -> list[float]:
        norm = sqrt(sum(float(x) * float(x) for x in values))
        if norm == 0:
            return [float(x) for x in values]
        return [float(x) / norm for x in values]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        result = self.client.models.embed_content(
            model=self.model,
            contents=list(texts),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.dimensions,
            ),
        )
        embeddings = [e.values for e in (result.embeddings or [])]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Gemini returned {len(embeddings)} embeddings for {len(texts)} texts."
            )
        return [self._normalize(values) for values in embeddings]

    def embed_query(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.dimensions,
            ),
        )
        if not result.embeddings:
            raise RuntimeError("Gemini returned no embedding for the query.")
        return self._normalize(result.embeddings[0].values)
