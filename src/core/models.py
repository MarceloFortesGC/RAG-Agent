"""Data models for RAG Core (query, chunks, result)."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Input for a search request."""

    query: str = Field(..., description="Search query text")
    project_id: Optional[str] = Field(default=None, description="Project ID for single-project search")
    tag: Optional[str] = Field(default=None, description="Tag for multi-project search")
    match_count: Optional[int] = Field(default=10, description="Number of results to return")
    search_type: Optional[str] = Field(
        default="hybrid",
        description="Search strategy: semantic, text, or hybrid",
    )


class SearchChunk(BaseModel):
    """Single chunk from search (content + score + metadata)."""

    chunk_id: str = Field(..., description="Chunk ID (project_id:doc_slug:index)")
    content: str = Field(..., description="Chunk text content")
    similarity: float = Field(..., description="Relevance score (0-1)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    document_title: str = Field(..., description="Document title from metadata")
    document_source: str = Field(..., description="Document path/source from metadata")


class SearchResult(BaseModel):
    """Result of a search (query + chunks + projects involved)."""

    query: str = Field(..., description="Original query")
    chunks: list[SearchChunk] = Field(default_factory=list, description="Search result chunks")
    projects_involved: list[str] = Field(
        default_factory=list,
        description="Project IDs involved (for multi-project search)",
    )
