"""Chroma: verificar collection e contagem. Use: uv run python test_scripts/check_chroma.py"""

import asyncio
from src.dependencies import AgentDependencies
from src.projects import PROJECTS


async def main():
    deps = AgentDependencies()
    await deps.initialize()
    print("Chroma path:", deps.settings.chroma_path)
    print("rag_chunks count:", deps.chroma_collection.count())
    print("Projetos:", list(PROJECTS.keys()))
    await deps.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
