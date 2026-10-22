"""Testes offline de llm.py: retry (com limites), text_of, build_llm e listagem de modelos."""

from types import SimpleNamespace

import groq
import pytest
from fakes import (
    FlakyRunnable,
    bad_request_error,
    connection_error,
    rate_limit_error,
    server_error,
    tool_use_failed_error,
)
from langchain_core.language_models import BaseChatModel

from workshop_agentes import llm
from workshop_agentes.llm import DailyQuotaError, build_llm, invoke_with_retry, list_models, text_of


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# --- invoke_with_retry ---------------------------------------------------------


def test_retryable_errors_then_success_returns_attempts():
    sleep = SleepRecorder()
    runnable = FlakyRunnable([rate_limit_error(), server_error(503)])
    response, attempts = invoke_with_retry(runnable, [], sleep=sleep)
    assert response is runnable.response
    assert attempts == 3  # N=2 falhas + 1 sucesso
    assert len(sleep.delays) == 2


def test_rate_limit_respects_suggested_delay():
    sleep = SleepRecorder()
    invoke_with_retry(FlakyRunnable([rate_limit_error(retry_after="7")]), [], sleep=sleep, base_delay=1.0)
    assert 7.0 <= sleep.delays[0] <= 8.0  # retry-after + jitter < base_delay


def test_suggested_delay_above_max_fails_fast():
    sleep = SleepRecorder()
    original = rate_limit_error(retry_after="45")
    runnable = FlakyRunnable([original])
    with pytest.raises(groq.RateLimitError) as info:
        invoke_with_retry(runnable, [], sleep=sleep, max_delay=30.0)
    assert info.value is original
    assert runnable.calls == 1
    assert sleep.delays == []  # não espera 45 s


def test_default_bounds_cap_total_wait():
    sleep = SleepRecorder()
    runnable = FlakyRunnable([rate_limit_error(retry_after="30") for _ in range(5)])
    with pytest.raises(groq.RateLimitError):
        invoke_with_retry(runnable, [], sleep=sleep)
    assert runnable.calls == 3  # max_attempts padrão
    assert sum(sleep.delays) <= 2 * (30.0 + 1.0)  # pior caso documentado (~62 s)


def test_backoff_grows_exponentially_without_suggestion():
    sleep = SleepRecorder()
    invoke_with_retry(FlakyRunnable([server_error()] * 3), [], max_attempts=4, sleep=sleep, base_delay=1.0)
    for delay, expected in zip(sleep.delays, [1.0, 2.0, 4.0], strict=True):
        assert expected <= delay <= expected + 1.0


def test_backoff_never_exceeds_max_delay():
    sleep = SleepRecorder()
    invoke_with_retry(FlakyRunnable([server_error()] * 5), [], max_attempts=6, sleep=sleep, max_delay=3.0)
    assert all(delay <= 3.0 + 1.0 for delay in sleep.delays)


def test_tool_use_failed_and_connection_errors_are_retried():
    runnable = FlakyRunnable([tool_use_failed_error(), connection_error()])
    _, attempts = invoke_with_retry(runnable, [], sleep=SleepRecorder())
    assert attempts == 3


def test_bad_request_raises_on_first_attempt():
    sleep = SleepRecorder()
    runnable = FlakyRunnable([bad_request_error()])
    with pytest.raises(groq.BadRequestError):
        invoke_with_retry(runnable, [], sleep=sleep)
    assert runnable.calls == 1  # attempts == 1
    assert sleep.delays == []


def test_exhausted_attempts_raise_original_exception():
    errors = [server_error() for _ in range(3)]
    runnable = FlakyRunnable(errors)
    with pytest.raises(groq.InternalServerError) as info:
        invoke_with_retry(runnable, [], max_attempts=3, sleep=SleepRecorder())
    assert info.value is errors[-1]
    assert runnable.calls == 3


def test_daily_quota_fails_fast_with_clear_message():
    sleep = SleepRecorder()
    original = rate_limit_error(per_day=True)
    runnable = FlakyRunnable([original])
    with pytest.raises(DailyQuotaError, match="Limite diário") as info:
        invoke_with_retry(runnable, [], sleep=sleep)
    assert runnable.calls == 1
    assert sleep.delays == []
    assert info.value.__cause__ is original


def test_invalid_max_attempts():
    with pytest.raises(ValueError):
        invoke_with_retry(FlakyRunnable([]), [], max_attempts=0)


def test_is_retryable_classification():
    assert llm._is_retryable(rate_limit_error())
    assert llm._is_retryable(server_error(500))
    assert llm._is_retryable(connection_error())
    assert llm._is_retryable(tool_use_failed_error())
    assert not llm._is_retryable(rate_limit_error(per_day=True))
    assert not llm._is_retryable(bad_request_error())
    assert not llm._is_retryable(ValueError("bug local"))


def test_rate_limit_window_names_the_limit():
    assert llm.rate_limit_window(rate_limit_error()) == "RPM"
    assert llm.rate_limit_window(rate_limit_error(per_day=True)) == "RPD"
    assert llm.rate_limit_window(rate_limit_error(window="output tokens per minute (OTPM)")) == "OTPM"
    assert llm.rate_limit_window(server_error()) is None
    assert llm.rate_limit_window(ValueError("bug local")) is None


def test_otpm_is_retried_not_daily():
    otpm = rate_limit_error(window="output tokens per minute (OTPM)")
    assert llm._is_retryable(otpm) and not llm._is_daily_quota(otpm)


def test_error_label_is_short_and_safe():
    assert llm.error_label(rate_limit_error(window="output tokens per minute (OTPM)")) == "RateLimitError OTPM"
    assert llm.error_label(tool_use_failed_error()) == "BadRequestError tool_use_failed"
    assert llm.error_label(server_error(503)) == "InternalServerError 503"
    assert llm.error_label(connection_error()) == "APITimeoutError"
    assert "org_fake" not in llm.error_label(rate_limit_error())


def test_on_retry_reports_each_retry_with_its_wait():
    seen: list[tuple[str, float]] = []
    sleep = SleepRecorder()
    runnable = FlakyRunnable([rate_limit_error(retry_after="7"), server_error()])
    invoke_with_retry(runnable, [], sleep=sleep, on_retry=lambda exc, delay: seen.append((type(exc).__name__, delay)))
    assert [name for name, _ in seen] == ["RateLimitError", "InternalServerError"]
    assert [delay for _, delay in seen] == sleep.delays  # a espera informada é a mesma usada


def test_suggested_delay_reads_retry_after_header():
    assert llm._suggested_delay(rate_limit_error(retry_after="2.5")) == 2.5
    assert llm._suggested_delay(rate_limit_error(retry_after=None)) is None
    assert llm._suggested_delay(ValueError("sem resposta")) is None


# --- text_of -------------------------------------------------------------------


def test_text_of_plain_string():
    assert text_of("olá") == "olá"


def test_text_of_blocks_keeps_only_text():
    content = [
        {"type": "text", "text": "Fila: ", "extras": {"signature": "abc=="}},
        {"type": "reasoning", "reasoning": "pensamento interno"},
        "use ",
        {"type": "text", "text": "array dinâmico."},
    ]
    assert text_of(content) == "Fila: use array dinâmico."


def test_text_of_empty_list():
    assert text_of([]) == ""


# --- build_llm (offline: construir não acessa a rede) ----------------------------


@pytest.fixture
def no_env_keys(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_build_llm_defaults(no_env_keys):
    model = build_llm("openai/gpt-oss-120b", "chave-falsa")
    assert isinstance(model, BaseChatModel)
    assert model.max_retries == 0  # retry só em invoke_with_retry (D9)
    assert model.request_timeout == llm.REQUEST_TIMEOUT
    assert model.max_tokens == llm.MAX_OUTPUT_TOKENS  # sem ele o Groq cortou respostas no finish
    assert model.temperature == 0.7  # não passada: padrão do ChatGroq
    assert "chave-falsa" not in repr(model)


def test_build_llm_explicit_temperature(no_env_keys):
    assert build_llm("openai/gpt-oss-120b", "chave-falsa", temperature=0.5).temperature == 0.5


# --- list_models -----------------------------------------------------------------


def test_list_models_keeps_active_sorted(monkeypatch):
    models = [
        SimpleNamespace(id="openai/gpt-oss-120b", active=True),
        SimpleNamespace(id="modelo-desativado", active=False),
        SimpleNamespace(id="llama-3.3-70b-versatile", active=True),
        SimpleNamespace(id="sem-flag"),
    ]
    fake_client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=models)))
    created: dict = {}
    monkeypatch.setattr(llm.groq, "Groq", lambda **kwargs: created.update(kwargs) or fake_client)
    assert list_models("chave-falsa") == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "sem-flag"]
    assert created["max_retries"] == 0 and created["timeout"] == llm.REQUEST_TIMEOUT  # mesma política do resto
