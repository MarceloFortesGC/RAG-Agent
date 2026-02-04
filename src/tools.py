"""Search tools for RAG Agent (Chroma)."""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from pydantic_ai import RunContext
from pydantic import BaseModel, Field

from src.dependencies import AgentDependencies
from src.projects import get_projects_by_tag

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Model for search results (Chroma: chunk_id, content from documents, metadata)."""

    chunk_id: str = Field(..., description="Chunk ID (project_id:doc_slug:index)")
    document_id: str = Field(default="", description="Legacy; use chunk_id")
    content: str = Field(..., description="Chunk text content")
    similarity: float = Field(..., description="Relevance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    document_title: str = Field(..., description="Document title from metadata")
    document_source: str = Field(..., description="Document path/source from metadata")


async def semantic_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    project_id: str,
    match_count: Optional[int] = None,
) -> List[SearchResult]:
    """
    Semantic search in Chroma filtered by project_id (always with where).

    Args:
        ctx: Agent runtime context with dependencies
        query: Search query text
        project_id: Project ID to filter (required)
        match_count: Number of results to return (default: 10)

    Returns:
        List of search results ordered by similarity
    """
    try:
        deps = ctx.deps
        if not deps.chroma_collection:
            await deps.initialize()

        if match_count is None:
            match_count = deps.settings.default_match_count
        match_count = min(match_count, deps.settings.max_match_count)

        query_embedding = await deps.get_embedding(query)

        result = await asyncio.to_thread(
            deps.chroma_collection.query,
            query_embeddings=[query_embedding],
            n_results=match_count,
            where={"project_id": project_id},
            include=["documents", "metadatas", "distances"],
        )

        ids_list = result.get("ids", [[]])
        docs_list = result.get("documents", [[]])
        metas_list = result.get("metadatas", [[]])
        dists_list = result.get("distances", [[]])

        ids = ids_list[0] if ids_list else []
        docs = docs_list[0] if docs_list else []
        metas = metas_list[0] if metas_list else []
        dists = dists_list[0] if dists_list else []

        # Chroma returns distances (L2: lower = more similar). Map to similarity in (0, 1].
        def dist_to_similarity(d: float) -> float:
            if d is None:
                return 0.0
            return 1.0 / (1.0 + d)

        search_results = []
        for i, chunk_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            content = docs[i] if i < len(docs) else ""
            dist = dists[i] if i < len(dists) else 0.0
            search_results.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    document_id="",
                    content=content,
                    similarity=dist_to_similarity(dist),
                    metadata=meta,
                    document_title=meta.get("document_title", ""),
                    document_source=meta.get("document_path", ""),
                )
            )

        logger.info(
            f"Busca semântica: query={query}, project_id={project_id}, resultados={len(search_results)}"
        )
        return search_results

    except Exception as e:
        logger.exception(f"Busca semântica falhou: query={query}, erro={str(e)}")
        return []


async def text_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    project_id: str,
    match_count: Optional[int] = None,
) -> List[SearchResult]:
    """
    Full-text search (not implemented with Chroma; returns empty).

    Args:
        ctx: Agent runtime context
        query: Search query text
        project_id: Project ID to filter
        match_count: Number of results

    Returns:
        Empty list (Chroma semantic-only)
    """
    return []


def reciprocal_rank_fusion(
    search_results_list: List[List[SearchResult]],
    k: int = 60,
) -> List[SearchResult]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        search_results_list: List of ranked result lists
        k: RRF constant (default: 60)

    Returns:
        Unified list sorted by combined RRF score
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, SearchResult] = {}

    for results in search_results_list:
        for rank, result in enumerate(results):
            chunk_id = result.chunk_id
            rrf_score = 1.0 / (k + rank)
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += rrf_score
            else:
                rrf_scores[chunk_id] = rrf_score
                chunk_map[chunk_id] = result

    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    merged_results = []
    for chunk_id, rrf_score in sorted_chunks:
        result = chunk_map[chunk_id]
        merged_results.append(
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                similarity=rrf_score,
                metadata=result.metadata,
                document_title=result.document_title,
                document_source=result.document_source,
            )
        )

    logger.info(
        f"RRF mesclou {len(search_results_list)} listas em {len(merged_results)} resultados"
    )
    return merged_results


async def hybrid_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    project_id: str,
    match_count: Optional[int] = None,
    text_weight: Optional[float] = None,
) -> List[SearchResult]:
    """
    Hybrid search: semantic only (Chroma). Text search not implemented.

    Args:
        ctx: Agent runtime context
        query: Search query text
        project_id: Project ID to filter (required)
        match_count: Number of results
        text_weight: Unused (Chroma semantic-only)

    Returns:
        List of search results from semantic search
    """
    return await semantic_search(ctx, query, project_id, match_count)


async def multi_project_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    tag: str,
    match_count_per_project: int = 5,
) -> List[SearchResult]:
    """
    Semantic search across all projects that have the given tag.

    Runs one query per project with where project_id, then merges and
    deduplicates by chunk_id, sorted by similarity.

    Args:
        ctx: Agent runtime context
        query: Search query text
        tag: Tag to select projects (e.g. "flutter")
        match_count_per_project: Max results per project

    Returns:
        Merged list of search results (deduped by chunk_id, sorted by similarity)
    """
    projects = get_projects_by_tag(tag)
    if not projects:
        logger.warning(f"Nenhum projeto com tag '{tag}'")
        return []

    all_results: List[SearchResult] = []
    seen_ids: set[str] = set()
    for p in projects:
        project_id = p["project_id"]
        results = await semantic_search(
            ctx, query, project_id=project_id, match_count=match_count_per_project
        )
        for r in results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                all_results.append(r)

    all_results.sort(key=lambda x: x.similarity, reverse=True)
    logger.info(
        f"Busca multi-projeto: tag={tag}, projetos={len(projects)}, resultados={len(all_results)}"
    )
    return all_results


def build_rag_context(
    results: List[SearchResult],
    max_tokens: Optional[int] = None,
) -> str:
    """
    Build context string from search results for RAG prompt.

    Args:
        results: Search results (content will be concatenated)
        max_tokens: Optional token limit (rough: 4 chars per token)

    Returns:
        Context string (documents joined by double newline)
    """
    parts = [r.content for r in results if r.content]
    context = "\n\n".join(parts)
    if max_tokens and max_tokens > 0:
        rough_chars = max_tokens * 4
        if len(context) > rough_chars:
            context = context[:rough_chars] + "\n\n[... truncado]"
    return context
