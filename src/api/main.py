"""
REST API: POST /search, GET /health.
Uses RAG Core only; no LLM. RAGDependencies and RAGCore initialized at startup.
"""

from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI

from src.core.deps import RAGDependencies
from src.core.models import SearchQuery, SearchResult
from src.core.rag_core import RAGCore

load_dotenv(override=True)

_rag_core: RAGCore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG Core once at startup."""
    global _rag_core
    deps = RAGDependencies()
    deps.initialize()
    _rag_core = RAGCore(deps)
    yield
    _rag_core = None


app = FastAPI(title="RAG Knowledge Base API", lifespan=lifespan)


def get_rag_core() -> RAGCore:
    if _rag_core is None:
        raise RuntimeError("RAG Core not initialized")
    return _rag_core


@app.get("/health")
async def health() -> dict[str, str]:
    """Readiness: config and Chroma available."""
    get_rag_core()
    return {"status": "ok"}


@app.post("/search", response_model=SearchResult)
async def search(body: dict[str, Any]) -> SearchResult:
    """
    Search the knowledge base.
    Body: query (required), project_id?, tag?, match_count?, search_type?
    """
    q = SearchQuery(
        query=body.get("query", ""),
        project_id=body.get("project_id"),
        tag=body.get("tag"),
        match_count=body.get("match_count"),
        search_type=body.get("search_type"),
    )
    return await get_rag_core().search(q)
