-- Run this once in the Supabase SQL Editor.
-- Embedding dimension must match EMBEDDING_DIMENSIONS in .env.

create extension if not exists vector with schema extensions;

create table if not exists public.document_chunks (
    id bigint generated always as identity primary key,
    source text not null,
    chunk_index integer not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding extensions.vector(768) not null,
    created_at timestamptz not null default now(),
    unique (source, chunk_index)
);

create index if not exists document_chunks_embedding_hnsw
on public.document_chunks
using hnsw (embedding vector_cosine_ops);

create index if not exists document_chunks_source_idx
on public.document_chunks (source);

create or replace function public.match_document_chunks (
    query_embedding extensions.vector(768),
    match_threshold float,
    match_count int
)
returns table (
    id bigint,
    source text,
    chunk_index integer,
    content text,
    metadata jsonb,
    similarity float
)
language sql
stable
as $$
    select
        dc.id,
        dc.source,
        dc.chunk_index,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) as similarity
    from public.document_chunks dc
    where 1 - (dc.embedding <=> query_embedding) >= match_threshold
    order by dc.embedding <=> query_embedding
    limit least(greatest(match_count, 1), 8);
$$;

-- For a private/local CLI you can use SUPABASE_KEY from your secure environment.
-- Do NOT commit a service-role/secret key to the repository.
