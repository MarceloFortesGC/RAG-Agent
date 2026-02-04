"""Main RAG agent implementation with shared state (Chroma, projects)."""

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from typing import Optional

from pydantic_ai.ag_ui import StateDeps

from src.providers import get_llm_model
from src.dependencies import AgentDependencies
from src.prompts import MAIN_SYSTEM_PROMPT
from src.tools import (
    semantic_search,
    hybrid_search,
    text_search,
    multi_project_search,
    build_rag_context,
)
from src.projects import PROJECTS


class RAGState(BaseModel):
    """Minimal shared state for the RAG agent."""
    pass


# Create the RAG agent with AGUI support
rag_agent = Agent(
    get_llm_model(),
    deps_type=StateDeps[RAGState],
    system_prompt=MAIN_SYSTEM_PROMPT
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
        ctx: Agent runtime context
        query: Search query text
        project_id: Project ID to search in (single project; default: first in projects.json)
        tag: If set, search across all projects with this tag (e.g. "flutter")
        match_count: Number of results (default: 5)
        search_type: "semantic" or "text" or "hybrid" (default: hybrid)

    Returns:
        Formatted string for the LLM (with "Projetos envolvidos" when tag is used)
    """
    try:
        agent_deps = AgentDependencies()
        await agent_deps.initialize()

        class DepsWrapper:
            def __init__(self, deps):
                self.deps = deps

        deps_ctx = DepsWrapper(agent_deps)

        if tag:
            # Multi-project: search by tag, enrich prompt with projects involved
            results = await multi_project_search(
                ctx=deps_ctx,
                query=query,
                tag=tag,
                match_count_per_project=match_count,
            )
            await agent_deps.cleanup()
            if not results:
                return "Nenhuma informação relevante encontrada na base de conhecimento."
            project_ids = list({r.metadata.get("project_id") for r in results if r.metadata.get("project_id")})
            projects_line = _format_projects_involved(project_ids)
            context = build_rag_context(results)
            return (
                "Você está analisando múltiplos projetos.\n\n"
                "Projetos envolvidos:\n"
                f"{projects_line}\n\n"
                "Use os trechos abaixo para responder:\n\n"
                f"{context}"
            )

        # Single project
        if not project_id and PROJECTS:
            project_id = next(iter(PROJECTS.values()))["project_id"]
        if not project_id:
            return "Nenhum projeto configurado em projects.json."

        if search_type == "hybrid":
            results = await hybrid_search(
                ctx=deps_ctx,
                query=query,
                project_id=project_id,
                match_count=match_count,
            )
        elif search_type == "semantic":
            results = await semantic_search(
                ctx=deps_ctx,
                query=query,
                project_id=project_id,
                match_count=match_count,
            )
        else:
            results = await text_search(
                ctx=deps_ctx,
                query=query,
                project_id=project_id,
                match_count=match_count,
            )

        await agent_deps.cleanup()

        if not results:
            return "Nenhuma informação relevante encontrada na base de conhecimento."

        response_parts = [f"Encontrados {len(results)} trechos relevantes:\n"]
        for i, result in enumerate(results, 1):
            response_parts.append(
                f"\n--- {result.document_title} (relevância: {result.similarity:.2f}) ---"
            )
            response_parts.append(result.content)
        return "\n".join(response_parts)

    except Exception as e:
        return f"Erro ao buscar na base de conhecimento: {str(e)}"
