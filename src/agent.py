"""Main RAG agent implementation with shared state (RAGCore from CLI)."""

from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps

from src.core.models import SearchQuery, SearchResult
from src.core.rag_core import RAGCore
from src.projects import PROJECTS
from src.prompts import MAIN_SYSTEM_PROMPT
from src.providers import get_llm_model
from src.tools import build_rag_context


class RAGState(BaseModel):
    """State for the RAG agent; holds RAGCore (injected by CLI)."""

    model_config = {"arbitrary_types_allowed": True}

    rag_core: Optional[RAGCore] = None


rag_agent = Agent(
    get_llm_model(),
    deps_type=StateDeps[RAGState],
    system_prompt=MAIN_SYSTEM_PROMPT,
)


def _format_projects_involved(project_ids: list[str]) -> str:
    """Format 'Projetos envolvidos' line from project ids (name + tags)."""
    lines = []
    for pid in project_ids:
        for p in PROJECTS.values():
            if p.get("project_id") == pid:
                tags = ", ".join(p.get("tags", []))
                lines.append(f"- {p.get('name', pid)} ({tags})")
                break
    return "\n".join(lines) if lines else ""


def _format_search_result_for_llm(result: SearchResult) -> str:
    """Turn SearchResult into the string the LLM expects (no I/O)."""
    if not result.chunks:
        return "Nenhuma informação relevante encontrada na base de conhecimento."
    if len(result.projects_involved) > 1:
        projects_line = _format_projects_involved(result.projects_involved)
        context = build_rag_context(result.chunks)
        return (
            "Você está analisando múltiplos projetos.\n\n"
            "Projetos envolvidos:\n"
            f"{projects_line}\n\n"
            "Use os trechos abaixo para responder:\n\n"
            f"{context}"
        )
    parts = [f"Encontrados {len(result.chunks)} trechos relevantes:\n"]
    for c in result.chunks:
        parts.append(f"\n--- {c.document_title} (relevância: {c.similarity:.2f}) ---")
        parts.append(c.content)
    return "\n".join(parts)


@rag_agent.tool
async def search_knowledge_base(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    project_id: Optional[str] = None,
    tag: Optional[str] = None,
    match_count: Optional[int] = 5,
    search_type: Optional[str] = "hybrid",
) -> str:
    """
    Search the knowledge base (by project or by tag for multi-project).

    Args:
        ctx: Agent runtime context (state must contain rag_core)
        query: Search query text
        project_id: Project ID to search in (single project; default: first in projects.json)
        tag: If set, search across all projects with this tag (e.g. "flutter")
        match_count: Number of results (default: 5)
        search_type: "semantic" or "text" or "hybrid" (default: hybrid)

    Returns:
        Formatted string for the LLM (with "Projetos envolvidos" when tag is used)
    """
    try:
        rag_core = ctx.deps.state.rag_core if ctx.deps and ctx.deps.state else None
        if rag_core is None:
            return "Erro: RAG Core não configurado. Execute pelo CLI ou configure o estado."

        q = SearchQuery(
            query=query,
            project_id=project_id,
            tag=tag,
            match_count=match_count,
            search_type=search_type,
        )
        result = await rag_core.search(q)
        return _format_search_result_for_llm(result)
    except Exception as e:
        return f"Erro ao buscar na base de conhecimento: {str(e)}"
