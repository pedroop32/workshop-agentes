"""Testes offline de tracing/tracer.py e tracing/report.py."""

import pytest

from workshop_agentes.tracing.report import (
    APPROXIMATION_NOTE,
    FAILED_ATTEMPTS_NOTE,
    attempts_of,
    format_report,
    totals,
    usage_of,
)
from workshop_agentes.tracing.tracer import Span, Tracer


def traced_run(tracer: Tracer) -> None:
    """run -> 2 passos; cada passo com um llm_call e ferramentas."""
    plan = {"previous_step": "nada ainda", "plan": "comparar e concluir", "next_step": "comparar"}
    with tracer.span("run", "agent", section="demo"):
        with tracer.span("step", "step_0") as step:
            step.attrs["usage"] = {"total_tokens": 999}  # não pode entrar nos totais
            with tracer.span("llm_call", "model") as s:
                s.attrs.update(usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                               finish_reason="STOP", attempts=1)
            with tracer.span("tool_call", "update_plan") as s:
                s.attrs.update(args=plan, result_chars=16)
        with tracer.span("step", "step_1"):
            with tracer.span("llm_call", "model") as s:
                s.attrs.update(usage={"input_tokens": 150}, finish_reason="STOP", attempts=2)  # campos ausentes
            with tracer.span("tool_call", "finish") as s:
                s.attrs.update(args={"answer": "x"}, result_chars=15)


def test_nested_spans_have_parent_ids():
    tracer = Tracer()
    traced_run(tracer)
    by_name = {(s.kind, s.name, s.parent_id) for s in tracer.spans}
    run, step0 = tracer.spans[0], tracer.spans[1]
    assert run.parent_id is None and run.attrs["section"] == "demo"
    assert step0.parent_id == run.id
    assert ("llm_call", "model", step0.id) in by_name
    assert ("tool_call", "update_plan", step0.id) in by_name
    assert all(s.t_end is not None and s.latency >= 0 for s in tracer.spans)


def test_open_span_has_no_latency():
    tracer = Tracer()
    with tracer.span("run", "agent") as s:
        assert s.t_end is None and s.latency is None
    assert s.latency is not None


def test_exception_closes_spans_records_error_and_reraises():
    tracer = Tracer()
    with pytest.raises(ValueError):
        with tracer.span("step", "step_0"):
            with tracer.span("llm_call", "model"):
                raise ValueError("falha")
    assert all(s.t_end is not None for s in tracer.spans)
    assert [s.attrs["error"] for s in tracer.spans] == ["ValueError", "ValueError"]
    with tracer.span("run", "depois") as after:  # pilha consistente: não herda pai antigo
        pass
    assert after.parent_id is None


def test_reset_creates_new_list_and_keeps_old_one_intact():
    tracer = Tracer()
    traced_run(tracer)
    old = tracer.spans
    count = len(old)
    tracer.reset()
    with tracer.span("run", "nova"):
        pass
    assert len(old) == count
    assert tracer.spans is not old and len(tracer.spans) == 1
    assert tracer.spans[0].id == 0


def test_usage_of_missing_fields_are_zero():
    span = Span(0, None, "llm_call", "model", 0.0, 1.0, {"usage": None})
    assert usage_of(span) == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_totals_use_only_leaves():
    tracer = Tracer()
    traced_run(tracer)
    total = totals(tracer.spans)
    leaves = [s for s in tracer.spans if s.kind in ("llm_call", "tool_call")]
    assert total["input_tokens"] == 250
    assert total["output_tokens"] == 20
    assert total["total_tokens"] == 120  # o 999 do step não entra (sem dupla contagem)
    assert total["latency"] == pytest.approx(sum(s.latency for s in leaves))
    assert total["llm_calls"] == 2 and total["tool_calls"] == 2


def test_report_has_steps_functions_totals_and_note():
    tracer = Tracer()
    traced_run(tracer)
    report = format_report(tracer.spans)
    assert "Passo 1 | tokens in=100 out=20 total=120" in report
    assert "finish_reason=STOP | tentativas=1" in report
    assert "Passo 2 | tokens in=150 out=0 total=0" in report
    assert "  plan: comparar e concluir" in report
    assert "  previous_step: nada ainda" in report
    assert "  next_step: comparar" in report
    assert "model (llm_call): chamadas=2 tokens=120" in report
    assert "finish (tool_call): chamadas=1" in report
    assert "tokens: in=250 out=20 total=120" in report
    assert APPROXIMATION_NOTE in report


def test_report_shows_forced_tool_and_plan_of_any_tool():
    tracer = Tracer()
    with tracer.span("run", "agent"), tracer.span("step", "step_0"):
        with tracer.span("llm_call", "model") as s:
            s.attrs.update(attempts=1, forced_tool="update_plan")
        with tracer.span("tool_call", "outro_planejador") as s:  # o relatório não testa o nome
            s.attrs["args"] = {"plan": "só o plano"}
        with tracer.span("tool_call", "search") as s:
            s.attrs["args"] = {"query": "fila"}
    report = format_report(tracer.spans)
    assert "  tool_choice forçado: update_plan" in report
    assert "  plan: só o plano" in report and "  previous_step: -" in report
    assert report.count("  plan:") == 1  # search não tem campos do plano


def test_report_flags_text_with_calls_empty_reply_and_tool_errors():
    tracer = Tracer()
    with tracer.span("run", "agent"), tracer.span("step", "step_0"):
        with tracer.span("llm_call", "model") as s:
            s.attrs.update(attempts=1, text_chars=23, empty_reply=True)
        with tracer.span("tool_call", "update_plan") as s:
            s.attrs.update(args={"plan": "x"}, error="ValidationError")
    report = format_report(tracer.spans)
    assert "  texto junto com chamada (não é a resposta): 23 caracteres" in report
    assert "  resposta vazia: o agente pediu a resposta de novo" in report
    assert "  erro em update_plan: ValidationError" in report


def test_report_shows_retries():
    tracer = Tracer()
    with tracer.span("run", "agent"), tracer.span("step", "step_0"), tracer.span("llm_call", "model") as s:
        s.attrs.update(attempts=2, retries=["RateLimitError OTPM (espera 2.3s)"])
    assert "  retentado: RateLimitError OTPM (espera 2.3s)" in format_report(tracer.spans)


def test_failed_call_counts_its_real_attempts():
    tracer = Tracer()
    with pytest.raises(RuntimeError):
        with tracer.span("run", "agent"), tracer.span("step", "step_0"), tracer.span("llm_call", "model") as s:
            s.attrs["retries"] = ["RateLimitError OTPM (espera 1.0s)", "RateLimitError OTPM (espera 2.0s)"]
            raise RuntimeError("x")
    llm_span = next(s for s in tracer.spans if s.kind == "llm_call")
    assert attempts_of(llm_span) == 3  # 2 retries + a última, que falhou (antes aparecia 0)
    report = format_report(tracer.spans)
    assert "tentativas=3" in report and FAILED_ATTEMPTS_NOTE in report


def test_report_shows_stop_reason():
    tracer = Tracer()
    with tracer.span("run", "agent") as run, tracer.span("step", "step_0"), tracer.span("llm_call", "model") as s:
        s.attrs.update(attempts=1, finish_reason="stop")
        run.attrs.update(reason="no_tool_call", finish_reason="stop")
    report = format_report(tracer.spans)
    assert "Motivo de parada: no_tool_call | finish_reason da última resposta: stop" in report
    assert FAILED_ATTEMPTS_NOTE not in report  # sem falhas, sem a nota


def test_report_shows_errors():
    tracer = Tracer()
    with pytest.raises(RuntimeError):
        with tracer.span("run", "agent"), tracer.span("step", "step_0"), tracer.span("llm_call", "model"):
            raise RuntimeError("x")
    assert "erro em model: RuntimeError" in format_report(tracer.spans)
