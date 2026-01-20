"""Settings configuration for MongoDB RAG Agent."""

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv
from typing import Optional
import os

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # MongoDB Configuration
    mongodb_uri: str = Field(..., description="MongoDB Atlas connection string")

    mongodb_database: str = Field(default="rag_db", description="MongoDB database name")

    mongodb_collection_documents: str = Field(
        default="documents", description="Collection for source documents"
    )

    mongodb_collection_chunks: str = Field(
        default="chunks", description="Collection for document chunks with embeddings"
    )

    mongodb_vector_index: str = Field(
        default="vector_index",
        description="Vector search index name (must be created in Atlas UI)",
    )

    mongodb_text_index: str = Field(
        default="text_index",
        description="Full-text search index name (must be created in Atlas UI)",
    )

    # LLM Configuration (OpenAI-compatible)
    llm_provider: str = Field(
        default="openrouter",
        description="LLM provider (openai, anthropic, gemini, ollama, etc.)",
    )

    llm_api_key: str = Field(..., description="API key for the LLM provider")

    llm_model: str = Field(
        default="anthropic/claude-haiku-4.5",
        description="Model to use for search and summarization",
    )

    llm_base_url: Optional[str] = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the LLM API (for OpenAI-compatible providers)",
    )

    # Embedding Configuration
    embedding_provider: str = Field(default="openai", description="Embedding provider")

    embedding_api_key: str = Field(..., description="API key for embedding provider")

    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model to use"
    )

    embedding_base_url: Optional[str] = Field(
        default="https://api.openai.com/v1", description="Base URL for embedding API"
    )

    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension (1536 for text-embedding-3-small)",
    )

    # Search Configuration
    default_match_count: int = Field(
        default=10, description="Default number of search results to return"
    )

    max_match_count: int = Field(
        default=50, description="Maximum number of search results allowed"
    )

    default_text_weight: float = Field(
        default=0.3, description="Default text weight for hybrid search (0-1)"
    )


def validate_environment_variables() -> tuple[bool, list[str]]:
    """
    Valida todas as variáveis de ambiente obrigatórias.
    
    Returns:
        Tuple de (sucesso: bool, erros: list[str])
    """
    erros = []
    
    # Variáveis obrigatórias
    variaveis_obrigatorias = {
        "MONGODB_URI": "URI de conexão do MongoDB Atlas",
        "LLM_API_KEY": "Chave API do provedor LLM",
        "EMBEDDING_API_KEY": "Chave API do provedor de embeddings"
    }
    
    # Verificar presença e valores não vazios
    for var_name, descricao in variaveis_obrigatorias.items():
        valor = os.getenv(var_name)
        if not valor or not valor.strip():
            erros.append(f"{var_name} ({descricao}) não está definida ou está vazia")
    
    # Validar formato do MongoDB URI
    mongodb_uri = os.getenv("MONGODB_URI", "").strip()
    if mongodb_uri:
        if not (mongodb_uri.startswith("mongodb://") or mongodb_uri.startswith("mongodb+srv://")):
            erros.append("MONGODB_URI deve começar com 'mongodb://' ou 'mongodb+srv://'")
    
    # Validar que API keys não são apenas espaços
    llm_key = os.getenv("LLM_API_KEY", "").strip()
    if llm_key and len(llm_key) < 10:
        erros.append("LLM_API_KEY parece ser inválida (muito curta)")
    
    embedding_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    if embedding_key and len(embedding_key) < 10:
        erros.append("EMBEDDING_API_KEY parece ser inválida (muito curta)")
    
    return (len(erros) == 0, erros)


def load_settings() -> Settings:
    """Load settings with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Falha ao carregar configurações: {e}"
        if "mongodb_uri" in str(e).lower():
            error_msg += "\nCertifique-se de definir MONGODB_URI no seu arquivo .env"
        if "llm_api_key" in str(e).lower():
            error_msg += "\nCertifique-se de definir LLM_API_KEY no seu arquivo .env"
        if "embedding_api_key" in str(e).lower():
            error_msg += "\nCertifique-se de definir EMBEDDING_API_KEY no seu arquivo .env"
        raise ValueError(error_msg) from e
