"""Supabase + pgvector semantic retriever."""
from __future__ import annotations

import os
from typing import Any

from supabase import Client, create_client

from codeagent.domain.models import Chunk


class SupabaseVectorRetriever:
    def __init__(
        self,
        embedding_adapter: Any,
        url: str | None = None,
        key: str | None = None,
        table: str = "document_chunks",
        rpc_name: str = "match_document_chunks",
        threshold: float = 0.35,
    ):
        url = url or os.getenv("SUPABASE_URL")
        key = key or os.getenv("SUPABASE_KEY")
        if not url:
            raise RuntimeError("SUPABASE_URL is required for the RAG vector store.")
        if not key:
            raise RuntimeError("SUPABASE_KEY is required for the RAG vector store.")

        self.client: Client = create_client(url, key)
        self.embedding = embedding_adapter
        self.table = table
        self.rpc_name = rpc_name
        self.threshold = threshold

    def upsert_chunks(self, chunks: list[Any], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return 0

        rows = []
        for chunk, vector in zip(chunks, embeddings):
            rows.append(
                {
                    "source": chunk.source,
                    "chunk_index": chunk.position,
                    "content": chunk.text,
                    "metadata": chunk.metadata,
                    "embedding": vector,
                }
            )

        self.client.table(self.table).upsert(
            rows,
            on_conflict="source,chunk_index",
        ).execute()
        return len(rows)

    def delete_source(self, source: str) -> None:
        self.client.table(self.table).delete().eq("source", source).execute()

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        vector = self.embedding.embed_query(query)
        response = self.client.rpc(
            self.rpc_name,
            {
                "query_embedding": vector,
                "match_threshold": self.threshold,
                "match_count": max(1, min(8, k)),
            },
        ).execute()

        rows = response.data or []
        return [
            Chunk(
                source=str(row.get("source", "unknown")),
                position=int(row.get("chunk_index", row.get("position", 0))),
                text=str(row.get("content", "")),
                score=float(row.get("similarity", 0.0)),
            )
            for row in rows
        ]
