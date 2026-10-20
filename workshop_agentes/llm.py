"""Camada do modelo: o único módulo que conhece o provedor (Gemini).

Trocar de provedor muda só este arquivo (D7). O resto do agente depende
apenas de langchain-core.
"""

import random
import time
from typing import Any, Callable, Sequence

from google import genai  # dependência transitiva do langchain-google-genai (exceção da seção 3.1)
from langchain_core.exceptions import ModelAPIError, ModelRateLimitError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI


class DailyQuotaError(RuntimeError):
    """Cota diária esgotada: tentar de novo hoje não adianta."""


def build_llm(model: str, api_key: str, temperature: float | None = None) -> BaseChatModel:
    """Cria o modelo de chat do Gemini (Factory).

    - max_retries=1 desliga o retry do pacote (0 significaria "padrão do SDK",
      com 5 retries): o único ponto de retry é invoke_with_retry (D9, opção b).
    - temperature=None não passa o parâmetro: o Gemini 3 recomenda o padrão.
    """
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "max_retries": 1}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatGoogleGenerativeAI(**kwargs)


def invoke_with_retry(
    runnable: Runnable,
    messages: Sequence[BaseMessage],
    max_attempts: int = 5,
    *,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[AIMessage, int]:
    """Chama runnable.invoke(messages) com backoff exponencial + jitter.

    Retorna (resposta, tentativas). Respeita o tempo sugerido pela API quando
    houver. Erro não retentável (ex.: 400) ou a última falha são levantados como
    vieram; cota diária esgotada vira DailyQuotaError na hora (sem esperar).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts deve ser >= 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return runnable.invoke(messages), attempt
        except Exception as exc:
            if _is_daily_quota(exc):
                raise DailyQuotaError(
                    "Cota diária do Gemini esgotada (renova à meia-noite do Pacífico). "
                    "Tente mais tarde ou use uma chave de outro projeto."
                ) from exc
            if attempt == max_attempts or not _is_retryable(exc):
                raise
            delay = _suggested_delay(exc)
            if delay is None:
                delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            sleep(delay + random.uniform(0, base_delay))  # jitter evita rajadas sincronizadas


def _is_retryable(exc: BaseException) -> bool:
    """Só 429 por minuto e 5xx valem nova tentativa; nunca 400 (D9).

    Único lugar a ajustar quando o tipo real da exceção mudar. O pacote
    converte 429 em GoogleRateLimitError e 5xx em GoogleAPIError, subclasses
    das classes genéricas do langchain-core usadas aqui.
    """
    return isinstance(exc, (ModelRateLimitError, ModelAPIError)) and not _is_daily_quota(exc)


def _error_details(exc: BaseException) -> list[dict]:
    """Lista error.details (formato google.rpc) da exceção ou de suas causas.

    O GoogleRateLimitError guarda o ClientError original em __cause__, e é ele
    que tem o JSON do erro em .details.
    """
    current: BaseException | None = exc
    while current is not None:
        payload = getattr(current, "details", None)
        if isinstance(payload, dict):
            error = payload.get("error", payload)
            return [d for d in error.get("details") or [] if isinstance(d, dict)]
        current = current.__cause__
    return []


def _is_daily_quota(exc: BaseException) -> bool:
    """429 de cota diária? [A VERIFICAR na API real: quotaId com "PerDay"]."""
    for detail in _error_details(exc):
        for violation in detail.get("violations") or []:
            if "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


def _suggested_delay(exc: BaseException) -> float | None:
    """Segundos sugeridos pela API (RetryInfo.retryDelay, ex.: "7s"), se houver."""
    for detail in _error_details(exc):
        delay = detail.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def text_of(content: str | list) -> str:
    """Texto de ai.content, que pode ser str ou lista de blocos (Gemini 3).

    Concatena só os blocos de texto; ignora raciocínio, assinaturas etc.
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


def list_generate_models(api_key: str) -> list[str]:
    """Modelos que aceitam generateContent; útil quando o modelo não é encontrado."""
    client = genai.Client(api_key=api_key)
    return sorted(
        model.name.removeprefix("models/")
        for model in client.models.list()
        if model.name and "generateContent" in (model.supported_actions or [])
    )
