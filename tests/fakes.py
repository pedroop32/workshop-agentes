"""Dublês de teste: nada aqui acessa rede, chave ou cota.

Os erros usam as classes reais do SDK groq (as que o ChatGroq deixa passar),
com uma httpx.Response falsa: status, headers e o JSON de erro da API.
"""

import itertools
from typing import Any

import groq
import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

# --- Modelo de chat falso --------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """Modelo falso: devolve, em ordem, AIMessages pré-definidos (ou levanta exceções).

    GenericFakeChatModel não serve: não implementa bind_tools. Guarda uma cópia
    das mensagens recebidas em cada chamada, para os testes inspecionarem o que
    o "modelo" viu.
    """

    script: list[Any] = Field(default_factory=list)
    received: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] | None = None
    tool_choice: Any = None
    bind_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received.append(list(messages))
        if not self.script:
            raise RuntimeError("roteiro do modelo falso acabou")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_tools = list(tools)
        self.tool_choice = tool_choice
        self.bind_calls += 1
        return self


USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
PLAN_ARGS = {
    "previous_step": "Nothing done yet.",
    "plan": "1. Compare. 2. Conclude.",
    "next_step": "Compare both structures.",
    "external": "Vou comparar as duas estruturas.",
}
_ids = itertools.count(1)


def call(name: str, args: Any = None, id: str | None = "auto") -> dict:
    """Uma tool_call no formato do langchain-core (id único por padrão)."""
    return {
        "name": name,
        "args": {} if args is None else args,
        "id": f"call_{next(_ids)}" if id == "auto" else id,
        "type": "tool_call",
    }


def ai(*calls: dict, content: Any = "", usage: dict | None = USAGE, finish_reason: str = "stop") -> AIMessage:
    """Resposta do modelo com tool_calls, usage_metadata e finish_reason."""
    return AIMessage(
        content=content,
        tool_calls=list(calls),
        usage_metadata=dict(usage) if usage is not None else None,
        response_metadata={"finish_reason": finish_reason},
    )


# --- Runnable que falha (testes de retry) ----------------------------------------


class FlakyRunnable:
    """Runnable falso: levanta as exceções dadas, em ordem, e depois responde."""

    def __init__(self, errors: list[Exception], response: AIMessage | None = None) -> None:
        self.errors = list(errors)
        self.response = response if response is not None else AIMessage(content="ok")
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.response


_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status_error(cls: type[groq.APIStatusError], code: int, error: dict, headers: dict | None = None):
    """Exceção do SDK como ele a monta: mensagem "Error code: N - {json}" e body = JSON decodificado."""
    body = {"error": error}
    response = httpx.Response(code, headers=headers or {}, json=body, request=_REQUEST)
    return cls(f"Error code: {code} - {body}", response=response, body=body)


def rate_limit_error(
    per_day: bool = False, retry_after: str | None = "7", window: str | None = None
) -> groq.RateLimitError:
    """429 como o SDK levanta, no formato visto na API real (2026-10-21).

    [A VERIFICAR: texto do 429 diário, ainda não observado]
    """
    window = window or ("requests per day (RPD)" if per_day else "requests per minute (RPM)")
    error = {
        "message": f"Rate limit reached for model `fake` in organization `org_fake` service tier `on_demand` "
        f"on {window}: Limit 3, Used 3, Requested 1. Please try again in 7s.",
        "type": "requests",
        "code": "rate_limit_exceeded",
    }
    headers = {"retry-after": retry_after} if retry_after is not None else None
    return _status_error(groq.RateLimitError, 429, error, headers)


def server_error(code: int = 503) -> groq.InternalServerError:
    """5xx como o SDK levanta."""
    return _status_error(groq.InternalServerError, code, {"message": "fake unavailable", "type": "internal_server_error"})


def bad_request_error() -> groq.BadRequestError:
    """400 comum como o SDK levanta (nunca deve ser retentado)."""
    return _status_error(groq.BadRequestError, 400, {"message": "fake invalid", "type": "invalid_request_error"})


def tool_use_failed_error() -> groq.BadRequestError:
    """400 de chamada de ferramenta malformada, no formato visto na API real (2026-10-21)."""
    error = {
        "message": "Failed to parse tool call arguments as JSON",
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": '{"name": "finish", "arguments": {"answer": }',
    }
    return _status_error(groq.BadRequestError, 400, error)


def connection_error() -> groq.APITimeoutError:
    """Timeout como o SDK levanta (subclasse de APIConnectionError)."""
    return groq.APITimeoutError(request=_REQUEST)
