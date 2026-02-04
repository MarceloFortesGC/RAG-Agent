"""Check Chroma vector store and projects configuration."""

import asyncio
import sys
from pathlib import Path

# Allow running from project root or from test_scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dependencies import AgentDependencies
from src.projects import PROJECTS


async def main():
    deps = AgentDependencies()
    await deps.initialize()

    print("=" * 60)
    print("Chroma RAG - Configuração")
    print("=" * 60)

    print(f"\nChroma path: {deps.settings.chroma_path}")
    count = deps.chroma_collection.count()
    print(f"Collection rag_chunks: {count} chunks")

    print("\nProjetos (projects.json):")
    for key, p in PROJECTS.items():
        print(f"  - {key}: {p.get('name')} ({p.get('project_id')})")

    print("\n" + "=" * 60)
    await deps.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
