"""Testes offline de agent/config.py."""

from workshop_agentes.agent.config import DEFAULT_MODEL, SYSTEM_PROMPT, AgentConfig, get_secret


def test_get_secret_reads_environment(monkeypatch):
    monkeypatch.setenv("WORKSHOP_TESTE_SEGREDO", "valor")
    assert get_secret("WORKSHOP_TESTE_SEGREDO") == "valor"


def test_get_secret_default_when_missing(monkeypatch):
    monkeypatch.delenv("WORKSHOP_TESTE_AUSENTE", raising=False)
    assert get_secret("WORKSHOP_TESTE_AUSENTE", "padrao") == "padrao"
    assert get_secret("WORKSHOP_TESTE_AUSENTE") is None


def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.max_steps == 8
    assert cfg.temperature is None
    assert cfg.verbose is True
    assert cfg.system_prompt == SYSTEM_PROMPT
    assert "update_plan" in SYSTEM_PROMPT and "plain text" in SYSTEM_PROMPT
    assert "finish" not in SYSTEM_PROMPT  # a resposta final vem em texto (no_tool_call)


def test_default_model_is_groq_qwen():
    assert DEFAULT_MODEL == "qwen/qwen3.8-27b"
