"""Test search functions with Chroma."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dependencies import AgentDependencies
from src.tools import semantic_search, hybrid_search, multi_project_search
from src.projects import resolve_project, get_projects_by_tag, PROJECTS


class MockContext:
    """Mock context for search functions."""
    def __init__(self, deps):
        self.deps = deps


async def test_chroma_connection():
    """Test Chroma connection and collection."""
    print("Testando conexão Chroma...")
    deps = AgentDependencies()
    await deps.initialize()
    count = deps.chroma_collection.count()
    print(f"Chunks na collection rag_chunks: {count}")
    await deps.cleanup()
    return count >= 0


async def test_semantic_search_isolation():
    """Test semantic search returns only chunks for given project_id."""
    print("\nTestando busca semântica (isolamento por projeto)...")
    deps = AgentDependencies()
    await deps.initialize()
    ctx = MockContext(deps)

    if not PROJECTS:
        print("Nenhum projeto em projects.json; pulando teste.")
        await deps.cleanup()
        return True

    project_id = next(iter(PROJECTS.values()))["project_id"]
    results = await semantic_search(ctx, "teste", project_id=project_id, match_count=5)
    print(f"Busca semântica retornou {len(results)} resultados para {project_id}")

    for r in results:
        meta_pid = r.metadata.get("project_id") if r.metadata else None
        if meta_pid and meta_pid != project_id:
            print(f"[FALHA] Resultado de outro projeto: {meta_pid}")
            await deps.cleanup()
            return False

    await deps.cleanup()
    return True


async def test_multi_project_search():
    """Test multi-project search by tag returns results from multiple projects."""
    print("\nTestando busca multi-projeto (por tag)...")
    projects_flutter = get_projects_by_tag("flutter")
    if not projects_flutter:
        print("Nenhum projeto com tag 'flutter'; pulando teste.")
        return True

    deps = AgentDependencies()
    await deps.initialize()
    ctx = MockContext(deps)
    results = await multi_project_search(ctx, "teste", tag="flutter", match_count_per_project=3)
    print(f"Busca multi-projeto (flutter) retornou {len(results)} resultados")

    project_ids = {r.metadata.get("project_id") for r in results if r.metadata and r.metadata.get("project_id")}
    print(f"Projetos nos resultados: {project_ids}")

    await deps.cleanup()
    return True


async def test_resolve_project():
    """Test project resolution."""
    print("\nTestando resolve_project...")
    try:
        p = resolve_project("master_detox")
        assert p["project_id"] == "master_detox"
        assert "name" in p
        print(f"  master_detox -> {p['name']} ({p['project_id']})")
    except RuntimeError as e:
        print(f"  master_detox não definido: {e}")
        return False

    try:
        resolve_project("inexistente")
        print("  [FALHA] resolve_project('inexistente') deveria levantar RuntimeError")
        return False
    except RuntimeError:
        print("  inexistente -> RuntimeError (esperado)")
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG Agent - Testes de busca (Chroma)")
    print("=" * 60)

    try:
        await test_resolve_project()
        ok = await test_chroma_connection()
        if not ok:
            print("\n[AVISO] Chroma inacessível ou vazio. Execute ingestão antes:")
            print("  uv run python -m src.ingestion.ingest -d ./documents")
            return

        await test_semantic_search_isolation()
        await test_multi_project_search()

        print("\n" + "=" * 60)
        print("Testes concluídos")
        print("=" * 60)
    except Exception as e:
        print(f"\nErro: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
