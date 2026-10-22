"""Camada do modelo: o único módulo que conhece o provedor (Groq).

Trocar de provedor muda só este arquivo (D7). O resto do agente depende
apenas de langchain-core.
"""

import random
import time
from typing import Any, Callable, Sequence

import groq  # dependência transitiva do langchain-groq (exceções e listagem de modelos)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

REQUEST_TIMEOUT = 30.0  # segundos por requisição (o padrão do SDK é 60)


class DailyQuotaError(RuntimeError):
    """Limite diário esgotado: tentar de novo agora não adianta."""


def build_llm(model: str, api_key: str, temperature: float | None = None) -> BaseChatModel:
    """Cria o modelo de chat do Groq (Factory).

    - max_retries=0 desliga o retry do SDK (padrão: 2): o único ponto de retry é
      invoke_with_retry (D9).
    - timeout explícito de REQUEST_TIMEOUT por requisição.
    - temperature=None não passa o parâmetro: vale o padrão do ChatGroq (0.7).
      O pacote sempre envia uma temperatura; não há como usar o padrão da API.
    """
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "max_retries": 0, "timeout": REQUEST_TIMEOUT}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatGroq(**kwargs)


def invoke_with_retry(
    runnable: Runnable,
    messages: Sequence[BaseMessage],
    max_attempts: int = 3,
    *,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[AIMessage, int]:
    """Chama runnable.invoke(messages) com backoff exponencial + jitter, sempre limitado.

    Retorna (resposta, tentativas). Usa o tempo sugerido pela API (retry-after)
    quando houver; se ele passar de max_delay, levanta na hora em vez de esperar.
    Espera máxima por chamada: (max_attempts - 1) * (max_delay + base_delay),
    ~62 s no padrão. Erro não retentável (ex.: 400) ou a última falha são
    levantados como vieram; limite diário esgotado vira DailyQuotaError na hora.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts deve ser >= 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return runnable.invoke(messages), attempt
        except Exception as exc:
            if _is_daily_quota(exc):
                raise DailyQuotaError(
                    "Limite diário do Groq esgotado (RPD/TPD, vale para a organização inteira). "
                    "Confira a página Limits do console Groq e tente mais tarde."
                ) from exc
            if attempt == max_attempts or not _is_retryable(exc):
                raise
            delay = _suggested_delay(exc)
            if delay is None:
                delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            elif delay > max_delay:
                raise  # esperar tanto travaria a execução: falha rápido
            sleep(delay + random.uniform(0, base_delay))  # jitter evita rajadas sincronizadas


def _is_retryable(exc: BaseException) -> bool:
    """Só 429 por minuto, 5xx, falha de conexão e chamada malformada; nunca outro 400 (D9).

    Único lugar a ajustar quando o tipo real da exceção mudar. O ChatGroq não
    converte as exceções do SDK: chegam groq.RateLimitError (429),
    groq.InternalServerError (5xx) e groq.APIConnectionError (inclui timeout).
    """
    if isinstance(exc, groq.RateLimitError):
        return not _is_daily_quota(exc)
    return isinstance(exc, (groq.InternalServerError, groq.APIConnectionError)) or _is_tool_use_failed(exc)


def _error_body(exc: BaseException) -> dict:
    """O objeto "error" do JSON da resposta de erro ({} se não houver)."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return {}
    error = body.get("error", body)
    return error if isinstance(error, dict) else {}


def _is_tool_use_failed(exc: BaseException) -> bool:
    """400 de chamada de ferramenta malformada? O Groq recomenda tentar de novo.

    [A VERIFICAR na API real: code "tool_use_failed"; a doc só cita failed_generation]
    """
    if not isinstance(exc, groq.BadRequestError):
        return False
    error = _error_body(exc)
    return error.get("code") == "tool_use_failed" or "failed_generation" in error


def _is_daily_quota(exc: BaseException) -> bool:
    """429 de limite diário (RPD ou TPD)? [A VERIFICAR na API real: "per day" na mensagem]."""
    if not isinstance(exc, groq.RateLimitError):
        return False
    return "per day" in str(_error_body(exc).get("message", exc)).lower()


def _suggested_delay(exc: BaseException) -> float | None:
    """Segundos sugeridos pela API no header retry-after (só vem em 429), se houver."""
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def text_of(content: str | list) -> str:
    """Texto de ai.content, que pode ser str (o Groq devolve str) ou lista de blocos.

    Concatena só os blocos de texto; ignora raciocínio e outros blocos.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def list_models(api_key: str) -> list[str]:
    """Modelos ativos para a chave; útil quando o modelo não é encontrado.

    A API não diz quais aceitam ferramentas: confira na página Tool Use da documentação.
    """
    client = groq.Groq(api_key=api_key)
    return sorted(model.id for model in client.models.list().data if getattr(model, "active", True))
