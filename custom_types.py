from typing import TypedDict


class ProcessedChunk(TypedDict):
    """Represents a single processed text chunk with its metadata."""
    doc_id: str
    chunk_index: int
    title: str
    content: str
    metadata: dict


class QueryResult(TypedDict):
    """Represents the result of a RAG query."""
    answer: str
    source_nodes: list
    query: str