"""Dublês de teste: nada aqui acessa rede, chave ou cota.

Os erros usam as classes reais do pacote (com o ClientError original em
__cause__), no formato google.rpc que a API devolve.
"""

from google.genai.errors import ClientError
from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleInvalidRequestError,
    GoogleRateLimitError,
)


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
