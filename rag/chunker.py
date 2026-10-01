"""Token-based document chunking with overlap for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class _WordEncoding:
    """Offline fallback used when tiktoken cannot download its vocabulary.

    tiktoken fetches cl100k_base over the network on first use, so an offline or
    firewalled machine would otherwise fail at import time. Splitting on
    whitespace is coarser but keeps ingestion working; chunk and overlap sizes
    then count words instead of tokens.
    """

    @staticmethod
    def encode(text: str, **_: object) -> list[str]:
        return text.split(" ")

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return " ".join(tokens)


def get_encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return _WordEncoding()



@dataclass(frozen=True)
class DocumentChunk:
    source: str
    position: int
    text: str
    metadata: dict


class TokenChunker:
    def __init__(self, chunk_tokens: int = 700, overlap_tokens: int = 100):
        if chunk_tokens < 100:
            raise ValueError("chunk_tokens must be at least 100")
        if overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = get_encoding()

    def split(self, text: str, source: str, metadata: dict | None = None) -> list[DocumentChunk]:
        text = text.strip()
        if not text:
            return []

        tokens = self.encoding.encode(text, disallowed_special=())
        chunks: list[DocumentChunk] = []
        start = 0
        position = 1
        step = self.chunk_tokens - self.overlap_tokens

        while start < len(tokens):
            end = min(start + self.chunk_tokens, len(tokens))
            piece = self.encoding.decode(tokens[start:end]).strip()
            if piece:
                chunks.append(
                    DocumentChunk(
                        source=source,
                        position=position,
                        text=piece,
                        metadata={**(metadata or {}), "token_start": start, "token_end": end},
                    )
                )
                position += 1
            if end >= len(tokens):
                break
            start += step

        return chunks


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ValueError(
                "Reading PDF documents requires pypdf: pip install -r requirements.txt"
            ) from e

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    raise ValueError(
        f"Unsupported document type: {suffix}. Supported: .md, .txt, .pdf"
    )
