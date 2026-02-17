"""
MCP server (stdio): exposes search_knowledge_base tool only.
Uses RAG Core; no LLM, no session state.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.core.deps import RAGDependencies
from src.core.models import SearchQuery, SearchResult
from src.core.rag_core import RAGCore

load_dotenv(override=True)


@dataclass
class AppContext:
    """Lifespan context: RAG Core (single instance)."""

    rag_core: RAGCore


_rag_core: RAGCore | None = None


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Create RAGDependencies and RAGCore once at startup."""
    global _rag_core
    deps = RAGDependencies()
    deps.initialize()
    _rag_core = RAGCore(deps)
    yield AppContext(rag_core=_rag_core)
    _rag_core = None


mcp = FastMCP(
    "RAG Knowledge Base",
    json_response=True,
    lifespan=app_lifespan,
)


@mcp.tool()
async def search_knowledge_base(
    query: str,
    project_id: str | None = None,
    tag: str | None = None,
    match_count: int | None = 10,
    search_type: str | None = "hybrid",
) -> SearchResult:
    """
    Search the knowledge base by project or by tag (multi-project).

    Args:
        query: Search query text
        project_id: Project ID for single-project search (optional)
        tag: Tag to search across projects (e.g. "flutter") (optional)
        match_count: Number of results (default 10)
        search_type: "semantic", "text", or "hybrid" (default "hybrid")

    Returns:
        SearchResult with query, chunks, and projects_involved
    """
    if _rag_core is None:
        return SearchResult(query=query, chunks=[], projects_involved=[])
    q = SearchQuery(
        query=query,
        project_id=project_id,
        tag=tag,
        match_count=match_count,
        search_type=search_type,
    )
    return await _rag_core.search(q)


if __name__ == "__main__":
    mcp.run(transport="stdio")
