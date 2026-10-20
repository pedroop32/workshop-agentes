"""Testes offline de llm.py: retry, text_of, build_llm e listagem de modelos."""

from types import SimpleNamespace

import pytest
from fakes import FlakyRunnable, bad_request_error, rate_limit_error, server_error
from langchain_core.language_models import BaseChatModel

from workshop_agentes import llm
from workshop_agentes.llm import DailyQuotaError, build_llm, invoke_with_retry, text_of


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
    invoke_with_retry(FlakyRunnable([rate_limit_error(retry_delay="7s")]), [], sleep=sleep, base_delay=1.0)
    assert 7.0 <= sleep.delays[0] <= 8.0  # sugestão da API + jitter < base_delay


def test_backoff_grows_exponentially_without_suggestion():
    sleep = SleepRecorder()
    invoke_with_retry(FlakyRunnable([server_error()] * 3), [], sleep=sleep, base_delay=1.0)
    for delay, expected in zip(sleep.delays, [1.0, 2.0, 4.0]):
        assert expected <= delay <= expected + 1.0


def test_bad_request_raises_on_first_attempt():
    sleep = SleepRecorder()
    runnable = FlakyRunnable([bad_request_error()])
    with pytest.raises(type(bad_request_error())):
        invoke_with_retry(runnable, [], sleep=sleep)
    assert runnable.calls == 1  # attempts == 1
    assert sleep.delays == []


def test_exhausted_attempts_raise_original_exception():
    errors = [server_error() for _ in range(3)]
    runnable = FlakyRunnable(errors)
    with pytest.raises(type(errors[0])) as info:
        invoke_with_retry(runnable, [], max_attempts=3, sleep=SleepRecorder())
    assert info.value is errors[-1]
    assert runnable.calls == 3


def test_daily_quota_fails_fast_with_clear_message():
    sleep = SleepRecorder()
    original = rate_limit_error(per_day=True)
    runnable = FlakyRunnable([original])
    with pytest.raises(DailyQuotaError, match="Cota diária") as info:
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
    assert not llm._is_retryable(rate_limit_error(per_day=True))
    assert not llm._is_retryable(bad_request_error())
    assert not llm._is_retryable(ValueError("bug local"))


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
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_build_llm_defaults(no_env_keys):
    model = build_llm("gemini-3.8-flash", "chave-falsa")
    assert isinstance(model, BaseChatModel)
    assert model.max_retries == 1  # retry só em invoke_with_retry (D9)
    assert model.temperature is None  # não passado: padrão do modelo
    assert "chave-falsa" not in repr(model)


def test_build_llm_explicit_temperature(no_env_keys):
    assert build_llm("gemini-3.8-flash", "chave-falsa", temperature=0.5).temperature == 0.5


# --- list_generate_models --------------------------------------------------------


def test_list_generate_models_filters_by_action(monkeypatch):
    models = [
        SimpleNamespace(name="models/gemini-b", supported_actions=["generateContent", "countTokens"]),
        SimpleNamespace(name="models/embedding", supported_actions=["embedContent"]),
        SimpleNamespace(name="models/gemini-a", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/sem-acoes", supported_actions=None),
    ]
    fake_client = SimpleNamespace(models=SimpleNamespace(list=lambda: models))
    monkeypatch.setattr(llm.genai, "Client", lambda api_key: fake_client)
    assert llm.list_generate_models("chave-falsa") == ["gemini-a", "gemini-b"]
