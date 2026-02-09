"""Settings configuration for RAG Agent."""

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

    # Chroma (vector store)
    chroma_path: str = Field(
        default="./chroma_data",
        description="Path for Chroma persistent storage",
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

    # Custo LLM (exibição em BRL)
    usd_to_brl_rate: float = Field(
        default=5.5,
        description="Taxa USD/BRL fallback quando API de câmbio não disponível (USD_TO_BRL_RATE)",
    )

    currency_api_key: Optional[str] = Field(
        default=None,
        description="API key Free Currency API para obter taxa USD/BRL em tempo real (CURRENCY_API_KEY)",
    )


def validate_environment_variables() -> tuple[bool, list[str]]:
    """
    Valida todas as variáveis de ambiente obrigatórias.

    Returns:
        Tuple de (sucesso: bool, erros: list[str])
    """
    erros = []

    variaveis_obrigatorias = {
        "LLM_API_KEY": "Chave API do provedor LLM",
        "EMBEDDING_API_KEY": "Chave API do provedor de embeddings",
    }

    for var_name, descricao in variaveis_obrigatorias.items():
        valor = os.getenv(var_name)
        if not valor or not valor.strip():
            erros.append(f"{var_name} ({descricao}) não está definida ou está vazia")

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
        if "llm_api_key" in str(e).lower():
            error_msg += "\nCertifique-se de definir LLM_API_KEY no seu arquivo .env"
        if "embedding_api_key" in str(e).lower():
            error_msg += "\nCertifique-se de definir EMBEDDING_API_KEY no seu arquivo .env"
        raise ValueError(error_msg) from e
