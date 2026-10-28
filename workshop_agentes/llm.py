"""Camada do modelo: o único módulo que importa o pacote do provedor (Groq).

Trocar de provedor muda este arquivo (D7) e os valores de config.py (DEFAULT_MODEL e os
nomes GROQ_* dos segredos). O resto do agente depende apenas de langchain-core.
"""

import random
import re
import time
from typing import Any, Callable, Sequence

import groq  # dependência transitiva do langchain-groq (exceções e listagem de modelos)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

REQUEST_TIMEOUT = 30.0  # segundos por requisição (o padrão do SDK é 60)
# API REAL (2026-10-21, qwen3.8-27b): sem limite ou com 2048, a resposta dentro do finish
# veio cortada em ~1000 caracteres, sem aviso (finish_reason continuou "tool_calls");
# com 4000 e 8192 veio completa (a de 8192 com 2901 caracteres). O máximo do modelo é 16384.
MAX_OUTPUT_TOKENS = 8192
# 429 por minuto (RPM, TPM, OTPM...): espera somada máxima por chamada. A cota se recompõe aos
# poucos e volta inteira em ~1 min (API REAL 2026-10-22: "OTPM: Limit 1000, Used 253, Requested 816.
# Please try again in 4.14s"; o Groq reserva uma estimativa da saída antes de gerar).
RATE_LIMIT_BUDGET = 60.0


class DailyQuotaError(RuntimeError):
    """Limite diário esgotado: tentar de novo agora não adianta."""


def build_llm(model: str, api_key: str, temperature: float | None = None) -> BaseChatModel:
    """Cria o modelo de chat do Groq (Factory).

    - max_retries=0 desliga o retry do SDK (padrão: 2): o único ponto de retry é
      invoke_with_retry (D9).
    - timeout explícito de REQUEST_TIMEOUT por requisição.
    - max_tokens=MAX_OUTPUT_TOKENS: limite de saída sempre enviado (ver a constante).
    - temperature=None não passa o parâmetro: vale o padrão do ChatGroq (0.7).
      O pacote sempre envia uma temperatura; não há como usar o padrão da API.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "max_retries": 0,
        "timeout": REQUEST_TIMEOUT,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
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
    rate_limit_budget: float = RATE_LIMIT_BUDGET,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[BaseException, float], None] | None = None,
) -> tuple[AIMessage, int]:
    """Chama runnable.invoke(messages) com backoff exponencial + jitter, sempre limitado.

    Retorna (resposta, tentativas). Dois limites:
    - 429 por minuto: retenta enquanto a espera somada couber em rate_limit_budget (~1 min, o
      tempo de a cota se recompor). Cada espera é o retry-after da API ou o backoff, o que for
      maior; o backoff dobra a cada 429, então são no máximo ~6 tentativas. Se a próxima espera
      não couber, levanta na hora em vez de travar.
    - 5xx, falha de conexão e chamada malformada: até max_attempts falhas desse tipo, cada espera
      até max_delay (~62 s somados no padrão).
    Erro não retentável (ex.: 400) ou a última falha são levantados como vieram; limite diário
    esgotado vira DailyQuotaError na hora.
    on_retry(exc, espera), se dado, é chamado antes de cada nova tentativa (tracing).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts deve ser >= 1")
    attempt = rate_limited = other_failures = 0
    waited = 0.0
    while True:
        attempt += 1
        try:
            return runnable.invoke(messages), attempt
        except Exception as exc:
            if _is_daily_quota(exc):
                raise DailyQuotaError(
                    "Limite diário do Groq esgotado (RPD/TPD, vale para a organização inteira). "
                    "Confira a página Limits do console Groq e tente mais tarde."
                ) from exc
            if not _is_retryable(exc):
                raise
            suggested = _suggested_delay(exc)
            jitter = random.uniform(0, base_delay)  # evita rajadas sincronizadas
            if isinstance(exc, groq.RateLimitError):  # por minuto: o diário já saiu acima
                rate_limited += 1
                delay = max(suggested or 0.0, base_delay * 2 ** (rate_limited - 1)) + jitter
                if waited + delay > rate_limit_budget:
                    raise  # a cota não volta dentro do orçamento: falha em vez de travar
            else:
                other_failures += 1
                if other_failures == max_attempts:
                    raise
                if suggested is None:
                    suggested = min(max_delay, base_delay * 2 ** (other_failures - 1))
                elif suggested > max_delay:
                    raise  # esperar tanto travaria a execução: falha rápido
                delay = suggested + jitter
            waited += delay
            if on_retry is not None:
                on_retry(exc, delay)
            sleep(delay)


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

    Na API real (2026-10-21): code "tool_use_failed" e failed_generation no erro.
    """
    if not isinstance(exc, groq.BadRequestError):
        return False
    error = _error_body(exc)
    return error.get("code") == "tool_use_failed" or "failed_generation" in error


def rate_limit_window(exc: BaseException) -> str | None:
    """Limite estourado num 429: "RPM", "RPD", "TPM", "TPD", "ITPM" ou "OTPM" (None se não for 429).

    A mensagem real nomeia a janela, ex.: "on requests per minute (RPM)" e "on output
    tokens per minute (OTPM)" (vistos em 2026-10-21; o OTPM não vem nos cabeçalhos).
    Seguro de exibir: a mensagem inteira tem o id da organização.
    """
    if not isinstance(exc, groq.RateLimitError):
        return None
    match = re.search(r"\(([IO]?TP[MD]|RP[MD])\)", str(_error_body(exc).get("message", exc)))
    return match.group(1) if match else None


def rate_limit_usage(exc: BaseException) -> str | None:
    """Os números de um 429, ex.: "Limit 1000, Used 253, Requested 816" (None se não vier nenhum).

    Cada número é lido separadamente: em 2026-10-22, 3 de 5 429 OTPM reais vieram sem o trecho
    completo. Seguro de exibir: só números. Requested é a estimativa que o Groq reserva antes
    de gerar, não o tamanho real da resposta.
    """
    if not isinstance(exc, groq.RateLimitError):
        return None
    message = str(_error_body(exc).get("message", exc))
    found = [m.group(0) for key in ("Limit", "Used", "Requested") if (m := re.search(rf"\b{key} \d+", message))]
    return ", ".join(found) or None


def error_label(exc: BaseException) -> str:
    """Rótulo curto e seguro de exibir: tipo + limite, código ou status (nunca a mensagem).

    Num 429 vêm também os números, ex.: "RateLimitError OTPM: Limit 1000, Used 253, Requested 816".
    """
    detail = rate_limit_window(exc) or _error_body(exc).get("code") or getattr(exc, "status_code", None)
    label = f"{type(exc).__name__} {detail}" if detail else type(exc).__name__
    usage = rate_limit_usage(exc)
    return f"{label}: {usage}" if usage else label


def explain_error(exc: BaseException, api_key: str | None = None) -> str:
    """O que fazer diante de uma falha na chamada, em português; nunca a mensagem do erro (tem dados da conta).

    Com api_key, um modelo não encontrado vem com os modelos que aceitam ferramentas (list_models).
    """
    if isinstance(exc, DailyQuotaError):
        return str(exc)  # mensagem nossa, sem dados da conta
    if isinstance(exc, groq.RateLimitError):  # o invoke_with_retry já esperou o que cabia no orçamento
        return f"Limite por minuto atingido ({rate_limit_window(exc) or 'limite'}): espere cerca de 1 minuto e rode de novo."
    if isinstance(exc, groq.AuthenticationError):
        return "Chave recusada: confira o secret GROQ_API_KEY e se o notebook tem acesso a ele."
    if isinstance(exc, groq.NotFoundError):
        hint = "Modelo não encontrado para a sua chave: defina o secret GROQ_MODEL com um modelo ativo."
        if api_key:
            try:
                hint += " Modelos que aceitam ferramentas: " + ", ".join(list_models(api_key))
            except Exception as list_exc:
                hint += f" (não foi possível listar os modelos: {type(list_exc).__name__})"
        return hint
    if isinstance(exc, groq.APIConnectionError):
        return "Sem conexão com a API do Groq (rede ou tempo esgotado): confira a internet e tente de novo."
    return "Falha inesperada na chamada ao modelo."


def _is_daily_quota(exc: BaseException) -> bool:
    """429 de limite diário (RPD ou TPD)? [A VERIFICAR: 429 diário ainda não observado]"""
    return rate_limit_window(exc) in ("RPD", "TPD")


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
    """Modelos ativos que aceitam ferramentas; útil quando o modelo não é encontrado.

    Quem aceita ferramentas tem "tools" em supported_features (API real, 2026-10-22); o campo
    falta em alguns modelos (ex.: os de áudio) e não está nos tipos do SDK. Se nenhum modelo
    trouxer o campo, devolve todos os ativos.
    """
    client = groq.Groq(api_key=api_key, max_retries=0, timeout=REQUEST_TIMEOUT)  # mesma política do resto
    models = [model for model in client.models.list().data if getattr(model, "active", True)]
    if any(getattr(model, "supported_features", None) for model in models):
        models = [model for model in models if "tools" in (getattr(model, "supported_features", None) or [])]
    return sorted(model.id for model in models)
