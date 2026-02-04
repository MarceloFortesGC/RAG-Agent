"""Chroma: verificar configuração. Use: uv run python test_scripts/check_chroma.py"""

import asyncio
from src.dependencies import AgentDependencies


async def main():
    deps = AgentDependencies()
    await deps.initialize()
    print("Chroma path:", deps.settings.chroma_path)
    print("rag_chunks count:", deps.chroma_collection.count())
    await deps.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
