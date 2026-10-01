"""CLI ingestion: local documents -> chunks -> Gemini embeddings -> Supabase."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from rag.gemini_embeddings import GeminiEmbeddingAdapter
from rag.supabase_vector import SupabaseVectorRetriever
from rag.chunker import TokenChunker, read_document

load_dotenv()

SUPPORTED = {".pdf", ".md", ".txt"}


def iter_documents(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def ingest(root: Path) -> int:
    chunker = TokenChunker(
        chunk_tokens=int(os.getenv("RAG_CHUNK_TOKENS", "700")),
        overlap_tokens=int(os.getenv("RAG_OVERLAP_TOKENS", "100")),
    )
    embeddings = GeminiEmbeddingAdapter()
    retriever = SupabaseVectorRetriever(embeddings)

    total = 0
    documents = list(iter_documents(root))
    if len(documents) < 3:
        raise RuntimeError(
            f"RAG requires at least 3 local documents; found {len(documents)} in {root}."
        )

    for path in documents:
        source = path.relative_to(root.parent).as_posix()
        try:
            text = read_document(path)
            chunks = chunker.split(
                text,
                source=source,
                metadata={
                    "filename": path.name,
                    "file_type": path.suffix.lower().lstrip("."),
                },
            )
            if not chunks:
                print(f"[skip] Empty document: {source}")
                continue

            vectors = embeddings.embed_documents([c.text for c in chunks])
            retriever.delete_source(source)
            retriever.upsert_chunks(chunks, vectors)
            total += len(chunks)
            print(f"[ok] {source}: {len(chunks)} chunks")
        except Exception as exc:
            print(f"[error] {source}: {exc}")

    print(f"Ingestion complete: {total} chunks stored in Supabase.")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local documents into Supabase pgvector.")
    parser.add_argument(
        "--documents",
        default=os.path.join("data", "documents"),
        help="Directory containing PDF, Markdown or text documents.",
    )
    args = parser.parse_args()
    ingest(Path(args.documents).resolve())


if __name__ == "__main__":
    main()
