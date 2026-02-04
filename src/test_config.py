"""Configuration validation script for RAG Agent."""

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
        print("RAG Agent - Validação de Configuração")
        print("=" * 60)
        print()

        print("[1/3] Carregando configurações...")
        settings = load_settings()
        print("[OK] Configurações carregadas")
        print()

        print("[2/3] Validando LLM...")
        model_info = get_model_info()
        print(f"  Provedor: {model_info['llm_provider']}")
        print(f"  Modelo: {model_info['llm_model']}")
        print(f"  Chave API: {mask_credential(settings.llm_api_key)}")
        print("[OK] LLM configurado")
        print()

        print("[3/3] Validando Embeddings...")
        print(f"  Modelo: {settings.embedding_model}")
        print(f"  Dimensão: {settings.embedding_dimension}")
        print(f"  Chave API: {mask_credential(settings.embedding_api_key)}")
        print("[OK] Embeddings configurado")
        print()

        print("=" * 60)
        print("[OK] TODAS AS VERIFICAÇÕES PASSARAM")
        print("=" * 60)
        print()
        print("Próximos passos:")
        print("1. Adicione documentos em ./documents/<project_key>/")
        print("2. Execute: uv run python -m src.ingestion.ingest -d ./documents")
        print("3. Execute o CLI: uv run python -m src.cli")
        print()

        return True

    except ValueError as e:
        print()
        print("=" * 60)
        print("[FALHA] VALIDAÇÃO FALHOU")
        print("=" * 60)
        print()
        print(f"Erro: {e}")
        print()
        print("Verifique o arquivo .env. Veja .env.example.")
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
