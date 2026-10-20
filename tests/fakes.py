"""Dublês de teste: nada aqui acessa rede, chave ou cota.

Os erros usam as classes reais do pacote (com o ClientError original em
__cause__), no formato google.rpc que a API devolve.
"""

import itertools
from typing import Any

from google.genai.errors import ClientError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleInvalidRequestError,
    GoogleRateLimitError,
)
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


def ai(*calls: dict, content: Any = "", usage: dict | None = USAGE, finish_reason: str = "STOP") -> AIMessage:
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


def _rpc_error(code: int, status: str, details: list[dict]) -> dict:
    return {"error": {"code": code, "message": f"fake {status}", "status": status, "details": details}}


def rate_limit_error(per_day: bool = False, retry_delay: str | None = "7s") -> GoogleRateLimitError:
    """429 como o pacote levanta. [A VERIFICAR na API real: formato do quotaId]"""
    window = "PerDay" if per_day else "PerMinute"
    details: list[dict] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": f"GenerateRequests{window}PerProjectPerModel-FreeTier"}],
        }
    ]
    if retry_delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    cause = ClientError(429, _rpc_error(429, "RESOURCE_EXHAUSTED", details))
    exc = GoogleRateLimitError(f"Error calling model 'fake' (RESOURCE_EXHAUSTED): {cause}")
    exc.__cause__ = cause
    return exc


def server_error(code: int = 503) -> GoogleAPIError:
    """5xx como o pacote levanta."""
    return GoogleAPIError(code=code, response_json=_rpc_error(code, "UNAVAILABLE", []))


def bad_request_error() -> GoogleInvalidRequestError:
    """400 como o pacote levanta (nunca deve ser retentado)."""
    cause = ClientError(400, _rpc_error(400, "INVALID_ARGUMENT", []))
    exc = GoogleInvalidRequestError(f"Error calling model 'fake' (INVALID_ARGUMENT): {cause}")
    exc.__cause__ = cause
    return exc
