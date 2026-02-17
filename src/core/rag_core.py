"""RAG Core: search orchestration (no LLM, no I/O creation)."""

from src.core.deps import RAGDependencies
from src.core.models import SearchQuery, SearchResult
from src.projects import PROJECTS, get_projects_by_tag
from src.tools import hybrid_search, multi_project_search, semantic_search, text_search


class RAGCore:
    """Single entry point for search. Uses injected RAGDependencies."""

    def __init__(self, deps: RAGDependencies) -> None:
        self._deps = deps

    async def search(self, q: SearchQuery) -> SearchResult:
        """
        Run search: validate project_id or tag, choose strategy, return structured result.
        """
        match_count = q.match_count or self._deps.settings.default_match_count
        match_count = min(match_count, self._deps.settings.max_match_count)

        if q.tag:
            projects = get_projects_by_tag(q.tag)
            if not projects:
                return SearchResult(query=q.query, chunks=[], projects_involved=[])
            chunks = await multi_project_search(
                self._deps,
                query=q.query,
                tag=q.tag,
                match_count_per_project=match_count,
            )
            project_ids = list({str(c.metadata["project_id"]) for c in chunks if c.metadata.get("project_id")})
            return SearchResult(query=q.query, chunks=chunks, projects_involved=project_ids)

        project_id = q.project_id
        if not project_id and PROJECTS:
            project_id = next(iter(PROJECTS.values()))["project_id"]
        if not project_id:
            return SearchResult(query=q.query, chunks=[], projects_involved=[])

        search_type = (q.search_type or "hybrid").lower()
        if search_type == "semantic":
            chunks = await semantic_search(
                self._deps, q.query, project_id=project_id, match_count=match_count
            )
        elif search_type == "text":
            chunks = await text_search(
                self._deps, q.query, project_id=project_id, match_count=match_count
            )
        else:
            chunks = await hybrid_search(
                self._deps, q.query, project_id=project_id, match_count=match_count
            )

        return SearchResult(query=q.query, chunks=chunks, projects_involved=[project_id])
