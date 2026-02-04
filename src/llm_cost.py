"""Estimativa de custo por uso de tokens do LLM (OpenRouter/OpenAI-compatible)."""

from typing import Optional

# Preços em USD por 1M tokens (input, output). Fontes: OpenRouter/fornecedores.
# Modelos não listados exibem apenas contagem de tokens.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "anthropic/claude-4.5-haiku-20251001": (1.0, 5.0),
    "anthropic/claude-3-5-haiku": (0.80, 4.0),
    "anthropic/claude-3-5-sonnet": (3.0, 15.0),
    "anthropic/claude-3-opus": (15.0, 75.0),
    "openai/gpt-4o": (2.5, 10.0),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4-turbo": (10.0, 30.0),
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "google/gemini-2.5-flash-preview": (0.15, 0.60),
    "google/gemini-2.5-pro-preview": (1.25, 10.0),
}


def _find_pricing(model_id: str) -> Optional[tuple[float, float]]:
    """Retorna (input_usd_per_1m, output_usd_per_1m) ou None se modelo desconhecido."""
    if not model_id or not model_id.strip():
        return None
    key = model_id.strip().lower()
    if key in _MODEL_PRICING:
        return _MODEL_PRICING[key]
    # Tenta com prefixo de provedor (ex: gpt-4o -> openai/gpt-4o)
    for prefix in ("openai/", "anthropic/", "google/"):
        candidate = prefix + key
        if candidate in _MODEL_PRICING:
            return _MODEL_PRICING[candidate]
    return None


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    model_id: str,
) -> Optional[float]:
    """
    Estima custo em USD para o uso de tokens dado o modelo.

    Args:
        input_tokens: Tokens de entrada (prompt).
        output_tokens: Tokens de saída (resposta).
        model_id: ID do modelo (ex: anthropic/claude-haiku-4.5).

    Returns:
        Custo estimado em USD ou None se modelo não tiver preço configurado.
    """
    prices = _find_pricing(model_id)
    if prices is None:
        return None
    input_per_1m, output_per_1m = prices
    cost = (input_tokens / 1_000_000.0) * input_per_1m + (
        output_tokens / 1_000_000.0
    ) * output_per_1m
    return round(cost, 6)


def format_usage_and_cost(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    model_id: str,
    usd_to_brl: Optional[float] = None,
) -> str:
    """
    Formata linha de uso de tokens e custo estimado para exibição no CLI.

    Args:
        input_tokens: Tokens de entrada.
        output_tokens: Tokens de saída.
        total_tokens: Total de tokens (input + output).
        model_id: ID do modelo usado.
        usd_to_brl: Se informado, converte custo para BRL e exibe em R$.

    Returns:
        String formatada (ex: "Tokens: 150 in / 80 out (230 total) | Custo est.: R$ 0,00").
    """
    parts = [
        f"Tokens: {input_tokens} in / {output_tokens} out ({total_tokens} total)"
    ]
    cost_usd = estimate_cost_usd(input_tokens, output_tokens, model_id)
    if cost_usd is not None:
        if usd_to_brl is not None and usd_to_brl > 0:
            cost_brl = cost_usd * usd_to_brl
            if cost_brl < 0.01 and cost_brl > 0:
                parts.append(f"Custo est.: R$ {cost_brl:.4f}")
            else:
                parts.append(f"Custo est.: R$ {cost_brl:.2f}")
        else:
            if cost_usd < 0.0001 and cost_usd > 0:
                parts.append(f"Custo est.: ~${cost_usd:.6f}")
            else:
                parts.append(f"Custo est.: ${cost_usd:.4f}")
    parts.append(f"Modelo: {model_id}")
    return " | ".join(parts)
