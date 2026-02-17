"""Search tools for RAG Agent (Chroma). Uses RAGDependencies from core."""

import asyncio
import logging
from typing import Dict, List, Optional

from src.core.deps import RAGDependencies
from src.core.models import SearchChunk
from src.projects import get_projects_by_tag

logger = logging.getLogger(__name__)


def _dist_to_similarity(d: float) -> float:
    if d is None:
        return 0.0
    return 1.0 / (1.0 + d)


async def semantic_search(
    deps: RAGDependencies,
    query: str,
    project_id: str,
    match_count: Optional[int] = None,
) -> List[SearchChunk]:
    """
    Semantic search in Chroma filtered by project_id (always with where).

    Args:
        deps: RAG dependencies (Chroma + embedding)
        query: Search query text
        project_id: Project ID to filter (required)
        match_count: Number of results to return (default from settings)

    Returns:
        List of search chunks ordered by similarity
    """
    try:
        deps.initialize()
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

        chunks: List[SearchChunk] = []
        for i, chunk_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            content = docs[i] if i < len(docs) else ""
            dist = dists[i] if i < len(dists) else 0.0
            chunks.append(
                SearchChunk(
                    chunk_id=str(chunk_id),
                    content=content,
                    similarity=_dist_to_similarity(dist),
                    metadata=meta,
                    document_title=meta.get("document_title", ""),
                    document_source=meta.get("document_path", ""),
                )
            )

        logger.info(
            "Busca semântica: query=%s, project_id=%s, resultados=%s",
            query[:50], project_id, len(chunks),
        )
        return chunks

    except Exception as e:
        logger.exception("Busca semântica falhou: query=%s, erro=%s", query[:50], str(e))
        return []


async def text_search(
    deps: RAGDependencies,
    query: str,
    project_id: str,
    match_count: Optional[int] = None,
) -> List[SearchChunk]:
    """
    Full-text search (not implemented with Chroma; returns empty).

    Args:
        deps: RAG dependencies
        query: Search query text
        project_id: Project ID to filter
        match_count: Number of results

    Returns:
        Empty list (Chroma semantic-only)
    """
    return []


def reciprocal_rank_fusion(
    search_results_list: List[List[SearchChunk]],
    k: int = 60,
) -> List[SearchChunk]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        search_results_list: List of ranked result lists
        k: RRF constant (default: 60)

    Returns:
        Unified list sorted by combined RRF score
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, SearchChunk] = {}

    for results in search_results_list:
        for rank, result in enumerate(results):
            rrf_score = 1.0 / (k + rank)
            if result.chunk_id in rrf_scores:
                rrf_scores[result.chunk_id] += rrf_score
            else:
                rrf_scores[result.chunk_id] = rrf_score
                chunk_map[result.chunk_id] = result

    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    merged: List[SearchChunk] = []
    for chunk_id, rrf_score in sorted_chunks:
        result = chunk_map[chunk_id]
        merged.append(
            SearchChunk(
                chunk_id=result.chunk_id,
                content=result.content,
                similarity=rrf_score,
                metadata=result.metadata,
                document_title=result.document_title,
                document_source=result.document_source,
            )
        )

    logger.info("RRF mesclou %s listas em %s resultados", len(search_results_list), len(merged))
    return merged


async def hybrid_search(
    deps: RAGDependencies,
    query: str,
    project_id: str,
    match_count: Optional[int] = None,
    text_weight: Optional[float] = None,
) -> List[SearchChunk]:
    """
    Hybrid search: semantic only (Chroma). Text search not implemented.

    Args:
        deps: RAG dependencies
        query: Search query text
        project_id: Project ID to filter (required)
        match_count: Number of results
        text_weight: Unused (Chroma semantic-only)

    Returns:
        List of search chunks from semantic search
    """
    return await semantic_search(deps, query, project_id, match_count)


async def multi_project_search(
    deps: RAGDependencies,
    query: str,
    tag: str,
    match_count_per_project: int = 5,
) -> List[SearchChunk]:
    """
    Semantic search across all projects that have the given tag.

    Args:
        deps: RAG dependencies
        query: Search query text
        tag: Tag to select projects (e.g. "flutter")
        match_count_per_project: Max results per project

    Returns:
        Merged list (deduped by chunk_id, sorted by similarity)
    """
    projects = get_projects_by_tag(tag)
    if not projects:
        logger.warning("Nenhum projeto com tag '%s'", tag)
        return []

    all_results: List[SearchChunk] = []
    seen_ids: set[str] = set()
    for p in projects:
        project_id = p["project_id"]
        results = await semantic_search(
            deps, query, project_id=project_id, match_count=match_count_per_project
        )
        for r in results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                all_results.append(r)

    all_results.sort(key=lambda x: x.similarity, reverse=True)
    logger.info(
        "Busca multi-projeto: tag=%s, projetos=%s, resultados=%s",
        tag, len(projects), len(all_results),
    )
    return all_results


def build_rag_context(
    chunks: List[SearchChunk],
    max_tokens: Optional[int] = None,
) -> str:
    """
    Build context string from search chunks for RAG prompt.

    Args:
        chunks: Search chunks (content will be concatenated)
        max_tokens: Optional token limit (rough: 4 chars per token)

    Returns:
        Context string (documents joined by double newline)
    """
    parts = [c.content for c in chunks if c.content]
    context = "\n\n".join(parts)
    if max_tokens and max_tokens > 0:
        rough_chars = max_tokens * 4
        if len(context) > rough_chars:
            context = context[:rough_chars] + "\n\n[... truncado]"
    return context
