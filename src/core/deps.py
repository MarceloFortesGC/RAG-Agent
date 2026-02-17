"""Shared dependencies for RAG Core (Chroma + embedding client)."""

from typing import Any

import chromadb
import openai

from src.settings import Settings, load_settings


class RAGDependencies:
    """
    Single place for Chroma and embedding client.
    Caller must ensure load_dotenv() was called before if using .env.
    """

    def __init__(self) -> None:
        self._settings: Settings | None = None
        self._chroma_client: Any = None
        self._chroma_collection: Any = None
        self._embedding_client: openai.AsyncOpenAI | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Load settings and create Chroma + embedding client once."""
        if self._initialized:
            return
        self._settings = load_settings()
        self._chroma_client = chromadb.PersistentClient(path=self._settings.chroma_path)
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name="rag_chunks",
            metadata={"description": "Chunks de todos os projetos"},
        )
        self._embedding_client = openai.AsyncOpenAI(
            api_key=self._settings.embedding_api_key,
            base_url=self._settings.embedding_base_url,
        )
        self._initialized = True

    @property
    def settings(self) -> Settings:
        if not self._initialized:
            self.initialize()
        assert self._settings is not None
        return self._settings

    @property
    def chroma_collection(self) -> Any:
        if not self._initialized:
            self.initialize()
        assert self._chroma_collection is not None
        return self._chroma_collection

    @property
    def embedding_client(self) -> openai.AsyncOpenAI:
        if not self._initialized:
            self.initialize()
        assert self._embedding_client is not None
        return self._embedding_client

    async def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        response = await self.embedding_client.embeddings.create(
            model=self.settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding
