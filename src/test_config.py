"""Configuration validation script for MongoDB RAG Agent."""

import sys
from src.settings import load_settings
from src.providers import get_model_info


def mask_credential(value: str) -> str:
    """Mask credentials for safe display."""
    if not value or len(value) < 8:
        return "***"
    return value[:4] + "..." + value[-4:]


def validate_config() -> bool:
    """
    Validate configuration and display settings.

    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        print("=" * 60)
        print("MongoDB RAG Agent - Validação de Configuração")
        print("=" * 60)
        print()

        # Load settings
        print("[1/4] Carregando configurações...")
        settings = load_settings()
        print("[OK] Configurações carregadas com sucesso")
        print()

        # Validate MongoDB configuration
        print("[2/4] Validando configuração do MongoDB...")
        print(f"  MongoDB URI: {mask_credential(settings.mongodb_uri)}")
        print(f"  Database: {settings.mongodb_database}")
        print(f"  Coleção de Documentos: {settings.mongodb_collection_documents}")
        print(f"  Coleção de Chunks: {settings.mongodb_collection_chunks}")
        print(f"  Índice Vetorial: {settings.mongodb_vector_index}")
        print(f"  Índice de Texto: {settings.mongodb_text_index}")
        print("[OK] Configuração do MongoDB presente")
        print()

        # Validate LLM configuration
        print("[3/4] Validando configuração do LLM...")
        model_info = get_model_info()
        print(f"  Provedor: {model_info['llm_provider']}")
        print(f"  Modelo: {model_info['llm_model']}")
        print(f"  URL Base: {model_info['llm_base_url']}")
        print(f"  Chave API: {mask_credential(settings.llm_api_key)}")
        print("[OK] Configuração do LLM presente")
        print()

        # Validate Embedding configuration
        print("[4/4] Validando configuração de Embeddings...")
        print(f"  Provedor: {settings.embedding_provider}")
        print(f"  Modelo: {settings.embedding_model}")
        print(f"  Dimensão: {settings.embedding_dimension}")
        print(f"  Chave API: {mask_credential(settings.embedding_api_key)}")
        print("[OK] Configuração de Embeddings presente")
        print()

        # Success summary
        print("=" * 60)
        print("[OK] TODAS AS VERIFICAÇÕES DE CONFIGURAÇÃO PASSARAM")
        print("=" * 60)
        print()
        print("Próximos passos:")
        print("1. Adicione documentos na pasta ./documents/")
        print("2. Execute a ingestão: uv run python -m src.ingestion.ingest -d ./documents")
        print("3. Crie os índices de busca no MongoDB Atlas (após a ingestão concluir)")
        print("   Veja README.md para instruções de criação de índices")
        print()

        return True

    except ValueError as e:
        print()
        print("=" * 60)
        print("[FALHA] VALIDAÇÃO DE CONFIGURAÇÃO FALHOU")
        print("=" * 60)
        print()
        print(f"Erro: {e}")
        print()
        print("Por favor, verifique seu arquivo .env e certifique-se de que todas as variáveis necessárias estão definidas.")
        print("Veja .env.example para as variáveis necessárias.")
        print()
        return False

    except Exception as e:
        print()
        print("=" * 60)
        print("[FALHA] ERRO INESPERADO")
        print("=" * 60)
        print()
        print(f"Erro: {e}")
        print()
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate_config()
    sys.exit(0 if success else 1)
