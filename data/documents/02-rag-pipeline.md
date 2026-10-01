# The RAG Pipeline: Chunking, Embeddings and Retrieval

## What retrieval-augmented generation solves

A language model answers from its training data, which is fixed, general and
silently out of date. Retrieval-augmented generation attaches a second source of
truth: a corpus the operator controls. Before generating, the system searches
that corpus, selects the passages most relevant to the question, and places them
in the model's context alongside the question itself. The model then answers
from material it can actually see.

The benefit is not only freshness. It is attribution. Because each retrieved
passage carries the document it came from, the answer can cite its sources, and
a reader can verify a claim instead of trusting it.

## Chunking and why overlap exists

Documents cannot be embedded whole. Embedding models accept a bounded input, and
more importantly, a single vector averaged over a long document represents
nothing in particular: the specific passage that answers a question is diluted
by everything else in the file. Documents are therefore split into chunks, and
each chunk is embedded separately.

Chunk size is a trade-off with two failure modes at the extremes. Chunks that
are too small lose the context that makes them interpretable; a sentence
referring to "this approach" is useless without the paragraph that named the
approach. Chunks that are too large reintroduce the dilution problem and waste
context window on irrelevant text. A practical starting range is 500 to 800
tokens, which is roughly one to two dense paragraphs of technical prose, and the
range should be adjusted to the structure of the actual documents.

This project chunks at 700 tokens using the cl100k_base tokenizer, counting
real tokens rather than characters or words, because token counts are what the
embedding model and the context window are actually measured in.

Overlap addresses a different problem. If chunks are cut at hard boundaries, a
passage that straddles a boundary is split across two chunks and fully present
in neither. A question whose answer spans the cut retrieves half of it. Overlap
means each chunk repeats the final portion of the previous one, so any
contiguous span shorter than the overlap appears intact in at least one chunk.

The standard range is 10 to 20 percent of chunk size. This project uses 100
tokens of overlap on 700-token chunks, which is approximately 14.3 percent. The
cost is storage and some duplicate retrieval; the benefit is that boundary
passages stop disappearing.

A consequence worth stating plainly: if the source documents are shorter than
the chunk size, chunking and overlap never actually execute. Each document
becomes exactly one chunk, and the pipeline looks correct while never exercising
the behaviour it claims. Documents used to demonstrate a chunking pipeline must
be long enough to be split.

## Embeddings and vector similarity

An embedding model maps text to a dense vector such that semantically similar
texts land near each other. This is what distinguishes vector search from
keyword search: a query about "switching between model vendors" can retrieve a
passage about "provider adapters" even though the two share almost no words.

This project embeds with the Gemini embedding model at 768 dimensions, and
embeds documents and queries with different task types. Using
RETRIEVAL_DOCUMENT when indexing and RETRIEVAL_QUERY when searching is not
cosmetic: the model produces asymmetric representations tuned for each role, and
mixing them measurably degrades retrieval quality.

Vectors are normalized to unit length before storage. With normalized vectors,
cosine similarity reduces to a dot product, and the distance operator the
database exposes becomes directly interpretable as a similarity score.

## Storage in Supabase with pgvector

Embeddings are stored in PostgreSQL through the pgvector extension, hosted on
Supabase. The table holds the source path, the chunk index within that source,
the chunk text, a JSON metadata column and the embedding vector itself. A unique
constraint on source and chunk index makes ingestion idempotent: re-ingesting a
document replaces its chunks rather than accumulating duplicates.

An HNSW index on the embedding column using cosine distance keeps search fast as
the corpus grows. HNSW is an approximate nearest-neighbour index, trading a
small amount of recall for a large speed gain, which is the right trade for
interactive retrieval.

Search runs through a SQL function that takes a query embedding, a similarity
threshold and a result count, and returns the matching rows ordered by distance
with their similarity computed as one minus the cosine distance. Enforcing the
threshold in SQL rather than in application code means the database never ships
rows that will be discarded.

The threshold matters more than it appears. Vector search always returns a
nearest neighbour, even for a question the corpus cannot answer at all. Without
a minimum similarity, the system confidently retrieves its least-irrelevant
chunk and the model dutifully builds an answer on it. The threshold is what
allows the honest response that the knowledge base does not contain enough
information.

## Retrieval with source metadata

Every retrieved chunk is formatted with its source path, its position within
that source and its similarity score before being handed to the model. The
system prompt instructs the model to cite those sources in its answer and to say
explicitly when the retrieved context does not support a conclusion.

This is the anti-hallucination control, and it has two halves. The threshold
removes weak matches so the model is not handed plausible-looking noise. The
instruction tells the model what to do when it receives nothing useful. Either
half alone is insufficient: a model given irrelevant chunks will use them, and a
model given no instruction will fill the gap from its training data without
signalling that it did so.

## Agentic retrieval

A conventional RAG pipeline retrieves once, on every question, whether or not
retrieval is needed. This project instead exposes retrieval as tools the model
chooses to call, which is the agentic variant.

The model decides whether the question needs external information at all,
which source to consult, and whether a disappointing first result warrants
rephrasing the query or switching sources. Conversational turns skip retrieval
entirely. Questions about project documentation go to the vector store.
Questions about code in the working directory go to the BM25 lexical index,
which finds exact identifiers that embeddings blur. Questions about a specific
known file go to a direct file read.

Both retrievers implement the same port, so the agent treats them
interchangeably and a third could be added without touching the conversation
logic.
