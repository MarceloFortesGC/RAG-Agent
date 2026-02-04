"""Dependencies for RAG Agent."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging
import openai
import chromadb
from src.settings import load_settings

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """Dependencies injected into the agent context."""

    # Core dependencies
    openai_client: Optional[openai.AsyncOpenAI] = None
    settings: Optional[Any] = None
    chroma_client: Optional[Any] = None
    chroma_collection: Optional[Any] = None

    # Session context
    session_id: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    query_history: list = field(default_factory=list)

    async def initialize(self) -> None:
        """
        Initialize external connections (OpenAI embeddings, Chroma).

        Raises:
            ValueError: If settings cannot be loaded
        """
        if not self.settings:
            self.settings = load_settings()
            logger.info("Configurações carregadas")

        # Chroma: one collection for all projects
        if not self.chroma_client:
            self.chroma_client = chromadb.PersistentClient(path=self.settings.chroma_path)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="rag_chunks",
                metadata={"description": "Chunks de todos os projetos"},
            )
            logger.info(f"Chroma: {self.settings.chroma_path}, collection=rag_chunks")

        # OpenAI client for embeddings
        if not self.openai_client:
            self.openai_client = openai.AsyncOpenAI(
                api_key=self.settings.embedding_api_key,
                base_url=self.settings.embedding_base_url,
            )
            logger.info(
                f"Cliente OpenAI: model={self.settings.embedding_model}, "
                f"dimension={self.settings.embedding_dimension}"
            )

    async def cleanup(self) -> None:
        """Clean up external connections."""
        # Chroma PersistentClient does not require explicit close
        pass

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        if not self.openai_client:
            await self.initialize()

        response = await self.openai_client.embeddings.create(
            model=self.settings.embedding_model, input=text
        )
        return response.data[0].embedding

    def set_user_preference(self, key: str, value: Any) -> None:
        """
        Set a user preference for the session.

        Args:
            key: Preference key
            value: Preference value
        """
        self.user_preferences[key] = value

    def add_to_history(self, query: str) -> None:
        """
        Add a query to the search history.

        Args:
            query: Search query to add to history
        """
        self.query_history.append(query)
        if len(self.query_history) > 10:
            self.query_history.pop(0)
