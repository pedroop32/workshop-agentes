"""Testes offline dos evals (workshop_agentes/evals), com o modelo falso: nenhuma chamada à API."""

import json
from functools import partial

import pytest
from fakes import PLAN_ARGS, ScriptedChatModel, ai, bad_request_error, call, rate_limit_error
from langchain_core.tools import tool

from workshop_agentes import llm
from workshop_agentes.agent.config import AgentConfig
from workshop_agentes.agent.loop import Agent, AgentResult
from workshop_agentes.agent.memory import WorkingMemory
from workshop_agentes.evals import __main__ as cli
from workshop_agentes.evals import judge as judge_module
from workshop_agentes.evals.cases import EvalCase, load_golden_set
from workshop_agentes.evals.checks import evaluate, normalize, run_metrics
from workshop_agentes.evals.judge import JUDGE_CHECK, Judge
from workshop_agentes.evals.report import format_case, format_summary, summarize
from workshop_agentes.evals.runner import CaseResult, load_results, rescore, run_suite, save_results, score
from workshop_agentes.tools.core import update_plan
from workshop_agentes.tools.registry import ToolRegistry
from workshop_agentes.tracing.tracer import Span, Tracer


def make_agent(script, extra_tools=()):
    model = ScriptedChatModel(script=list(script))
    registry = ToolRegistry()
    registry.register(update_plan)
    for extra in extra_tools:
        registry.register(extra)
    cfg = AgentConfig(verbose=False)
    return Agent(model, registry, WorkingMemory(cfg.system_prompt), Tracer(), cfg), model


def plan_then(answer: str) -> list:
    """O run típico: o plano forçado e a resposta em texto (2 chamadas)."""
    return [ai(call("update_plan", PLAN_ARGS), finish_reason="tool_calls"), ai(content=answer)]


def result_of(answer, reason="no_tool_call", steps=2, error=None) -> AgentResult:
    return AgentResult(answer, steps, reason, [], [], error)


def checks_by_name(outcome: CaseResult) -> dict:
    return {c.name: c for c in outcome.checks}


def write_golden(tmp_path, cases) -> str:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return str(path)


# --- Golden set -------------------------------------------------------------------


def test_golden_set_loads_with_unique_ids():
    cases = load_golden_set()
    assert len(cases) >= 5
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.task for c in cases)


def test_unknown_field_is_rejected(tmp_path):
    path = write_golden(tmp_path, [{"id": "a", "task": "t", "must_includes": ["x"]}])
    with pytest.raises(ValueError, match="must_includes"):
        load_golden_set(path)


def test_list_field_given_as_string_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must_include deve ser uma lista"):
        load_golden_set(write_golden(tmp_path, [{"id": "a", "task": "t", "must_include": "391"}]))


def test_duplicate_id_and_missing_task_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="repetido"):
        load_golden_set(write_golden(tmp_path, [{"id": "a", "task": "t"}, {"id": "a", "task": "u"}]))
    with pytest.raises(ValueError, match="obrigatórios"):
        load_golden_set(write_golden(tmp_path, [{"id": "a"}]))


# --- Checks -----------------------------------------------------------------------


def test_must_include_ignores_case_and_accents_and_accepts_alternatives():
    case = EvalCase("c", "t", must_include=["array dinamico", ["fila", "queue"]])
    checks = evaluate(case, result_of("Use um ARRAY DINÂMICO como Queue."))
    assert all(c.passed for c in checks)
    assert normalize("Dinâmico") == "dinamico"


def test_must_not_include_and_max_steps():
    case = EvalCase("c", "t", must_not_include=["4."], max_steps=2)
    by_name = {c.name: c for c in evaluate(case, result_of("1. a\n4. d", steps=3))}
    assert by_name['não contém "4."'].passed is False
    assert by_name["passos <= 2"].passed is False and by_name["passos <= 2"].detail == "3"


@pytest.mark.parametrize(
    "result",
    [
        result_of(None, reason="error", error="BadRequestError 400"),
        result_of(None, reason="max_steps"),
        result_of("resposta cortada", reason="length"),
        result_of("   "),
    ],
)
def test_completion_fails_without_a_complete_final_answer(result):
    completed = evaluate(EvalCase("c", "t"), result)[0]
    assert completed.name == "concluiu" and completed.passed is False


# --- Runner, com o agente de verdade e o modelo falso ------------------------------


def test_passing_case_reports_metrics_from_tracing():
    agent, _ = make_agent(plan_then("17 × 23 = 391"))
    case = EvalCase("aritmetica", "Quanto é 17 × 23?", must_include=["391"], max_steps=3, expected_tools=["update_plan"])
    [outcome] = run_suite(agent, [case], log=None)
    assert outcome.status == "pass"
    metrics = run_metrics(outcome.result)
    assert metrics["total_tokens"] == 240 and metrics["llm_calls"] == 2  # 2 x USAGE do fakes.py
    assert metrics["tools_used"] == ["update_plan"] and metrics["tool_errors"] == []
    assert metrics["run_latency"] is not None and metrics["retries"] == 0
    run_span = next(s for s in outcome.result.spans if s.kind == "run")
    assert run_span.attrs["section"] == "eval:aritmetica"


def test_failing_check_fails_the_case():
    agent, _ = make_agent(plan_then("É 400."))
    [outcome] = run_suite(agent, [EvalCase("a", "t", must_include=["391"])], log=None)
    assert outcome.status == "fail"
    assert checks_by_name(outcome)['contém "391"'].passed is False


def test_tool_error_fails_the_case():
    missing_plan = {k: v for k, v in PLAN_ARGS.items() if k != "plan"}
    agent, _ = make_agent([ai(call("update_plan", missing_plan)), ai(content="ok")])
    [outcome] = run_suite(agent, [EvalCase("a", "t")], log=None)
    check = checks_by_name(outcome)["ferramentas sem erro"]
    assert outcome.status == "fail" and check.detail == "update_plan: ValidationError"


def test_case_expecting_unregistered_tool_is_skipped_without_calling_the_model():
    agent, model = make_agent([])
    [outcome] = run_suite(agent, [EvalCase("rag", "t", expected_tools=["rag_search"])], log=None)
    assert outcome.status == "skip" and "rag_search" in outcome.note
    assert model.received == []  # nenhuma chamada


def test_tool_checks_work_for_any_registered_tool():
    @tool
    def search(query: str) -> dict:
        """Busca de teste."""
        return {"result": query}

    script = [ai(call("update_plan", PLAN_ARGS)), ai(call("search", {"query": "fila"})), ai(content="Resposta.")]
    agent, _ = make_agent(script, extra_tools=[search])
    [outcome] = run_suite(agent, [EvalCase("a", "t", expected_tools=["search"], forbidden_tools=["update_plan"])], log=None)
    by_name = checks_by_name(outcome)
    assert by_name["chamou search"].passed is True
    assert by_name["não chamou update_plan"].passed is False and outcome.status == "fail"


def test_fatal_error_stops_the_suite_with_a_single_call():
    agent, model = make_agent([rate_limit_error(per_day=True)])  # vira DailyQuotaError na hora
    results = run_suite(agent, [EvalCase("a", "t"), EvalCase("b", "t"), EvalCase("c", "t")], log=None)
    assert [r.status for r in results] == ["fail", "not_run", "not_run"]
    assert "DailyQuotaError" in results[1].note
    assert len(model.received) == 1


def test_non_fatal_error_does_not_stop_the_suite():
    agent, _ = make_agent([bad_request_error(), *plan_then("391")])
    results = run_suite(agent, [EvalCase("a", "t"), EvalCase("b", "t", must_include=["391"])], log=None)
    assert [r.status for r in results] == ["fail", "pass"]
    assert results[0].result.error == "BadRequestError 400"


def test_progress_is_logged_per_case():
    agent, _ = make_agent(plan_then("391"))
    lines = []
    run_suite(agent, [EvalCase("a", "t"), EvalCase("b", "t", expected_tools=["rag_search"])], log=lines.append)
    assert lines[0] == "[1/2] a: rodando..."
    assert lines[1].startswith("[1/2] a: PASSOU (") and "2 passos, no_tool_call" in lines[1]
    assert lines[2] == "[2/2] b: PULADO (ferramenta não registrada: rag_search)"


# --- Juiz -------------------------------------------------------------------------


def make_judge(*replies):
    model = ScriptedChatModel(script=list(replies))
    return Judge(model), model


def test_judge_reads_the_forced_verdict_and_counts_tokens():
    judge, model = make_judge(ai(call("Verdict", {"passed": False, "reason": "Sem recomendação."})))
    outcome = score(EvalCase("a", "Compare.", rubric="Ends with a recommendation."), result_of("Ambas servem."), judge)
    check = checks_by_name(outcome)[JUDGE_CHECK]
    assert check.passed is False and check.detail == "Sem recomendação."
    assert outcome.status == "fail" and outcome.judge_tokens == 120
    assert model.tool_choices == ["Verdict"]
    prompt = model.received[0][-1].content
    assert "Ends with a recommendation." in prompt and "Ambas servem." in prompt


@pytest.mark.parametrize("verbose", [True, False])
def test_judge_announces_retries_like_the_agent(monkeypatch, capsys, verbose):
    monkeypatch.setattr(judge_module, "invoke_with_retry", partial(llm.invoke_with_retry, sleep=lambda seconds: None))
    otpm = rate_limit_error(window="output tokens per minute (OTPM)")
    model = ScriptedChatModel(script=[otpm, ai(call("Verdict", {"passed": True, "reason": "ok"}))])
    check, _ = Judge(model, verbose=verbose).grade(EvalCase("a", "t", rubric="r"), "resposta")
    assert check.passed is True
    out = capsys.readouterr().out
    assert ("[juiz retentando em" in out and "RateLimitError OTPM" in out) is verbose


@pytest.mark.parametrize("reply", [bad_request_error(), ai(content="sem veredito")])
def test_judge_failure_is_not_evaluated_and_does_not_fail_the_case(reply):
    judge, _ = make_judge(reply)
    outcome = score(EvalCase("a", "t", rubric="r"), result_of("ok"), judge)
    check = checks_by_name(outcome)[JUDGE_CHECK]
    assert check.passed is None and check.detail.startswith("juiz falhou")
    assert outcome.status == "pass"


def test_rubric_without_judge_or_answer_makes_no_call():
    judge, model = make_judge()
    assert checks_by_name(score(EvalCase("a", "t", rubric="r"), result_of("ok")))[JUDGE_CHECK].passed is None
    outcome = score(EvalCase("a", "t", rubric="r"), result_of(None, reason="error"), judge)
    assert checks_by_name(outcome)[JUDGE_CHECK].detail == "não avaliado: sem resposta"
    assert model.received == []


# --- Relatório --------------------------------------------------------------------


def suite_results():
    agent, _ = make_agent([*plan_then("391"), *plan_then("errado")])
    cases = [
        EvalCase("ok", "t", must_include=["391"]),
        EvalCase("ruim", "t", must_include=["391"]),
        EvalCase("rag", "t", expected_tools=["rag_search"]),
    ]
    return run_suite(agent, cases, log=None)


def test_summary_rates_are_over_executed_cases():
    s = summarize(suite_results())
    assert (s["cases"], s["executed"], s["passed"], s["failed"], s["skipped"]) == (3, 2, 1, 1, 1)
    assert s["success_rate"] == 0.5 and s["completion_rate"] == 1.0
    assert s["total_tokens"] == 480 and s["tokens_mean"] == 240 and s["llm_calls"] == 4
    assert s["reasons"] == {"no_tool_call": 2} and s["failed_checks"] == {'contém "391"': 1}


def test_reports_show_status_checks_and_trace():
    results = suite_results()
    text = format_case(results[1])
    assert text.startswith("[FALHOU] ruim: t") and '✗ contém "391"' in text and "resposta: errado" in text
    assert "=== Relatório de tracing ===" in format_case(results[0], trace=True)
    assert "[PULADO] rag" in format_case(results[2])
    summary = format_summary(results)
    assert "Taxa de sucesso: 50% (1/2)" in summary and 'Checks que falharam: contém "391" (1)' in summary


def test_summary_without_executed_cases():
    results = [CaseResult(EvalCase("a", "t"), "skip", note="x")]
    assert "Nenhum caso executado." in format_summary(results)


# --- JSON e rescore ---------------------------------------------------------------


def test_results_round_trip_and_rescore_without_model(tmp_path):
    path = tmp_path / "runs.json"
    save_results(suite_results(), path, model="fake")
    meta, saved = load_results(path)
    assert meta == {"model": "fake"} and set(saved) == {"ok", "ruim", "rag"}
    new_cases = [EvalCase("ok", "t", must_include=["391"]), EvalCase("ruim", "t", must_include=["errado"]),
                 EvalCase("rag", "t"), EvalCase("novo", "t")]
    results = rescore(new_cases, saved)
    assert [r.status for r in results] == ["pass", "pass", "skip", "skip"]
    assert all(isinstance(s, Span) for s in results[0].result.spans)
    assert run_metrics(results[0].result)["total_tokens"] == 240  # métricas relidas do tracing salvo


# --- Linha de comando -------------------------------------------------------------


@pytest.fixture
def no_api(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("não devia criar o modelo (chamada à API)")

    monkeypatch.setattr(cli, "build_llm", fail)


def test_cli_dry_run_makes_no_api_call(no_api, capsys):
    assert cli.main(["--dry-run", "--judge"]) == 0
    out = capsys.readouterr().out
    assert "roda  fila-estruturas + juiz" in out and "Nenhuma foi feita." in out


def test_cli_unknown_case_id(no_api, capsys):
    assert cli.main(["--only", "nao-existe"]) == 2
    assert "nao-existe" in capsys.readouterr().out


def test_cli_without_key_fails_clearly(no_api, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_secret", lambda name, default=None: default)
    assert cli.main(["--only", "aritmetica"]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().out


def test_cli_runs_writes_reports_and_rescores_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_secret", lambda name, default=None: "fake" if name == "GROQ_API_KEY" else default)
    monkeypatch.setattr(cli, "build_llm", lambda *args: ScriptedChatModel(script=plan_then("391")))
    assert cli.main(["--only", "aritmetica", "--out", str(tmp_path), "--quiet"]) == 0
    [report] = tmp_path.glob("*.txt")
    [runs] = tmp_path.glob("*.json")
    assert "[PASSOU] aritmetica" in report.read_text(encoding="utf-8")
    assert "Taxa de sucesso: 100% (1/1)" in capsys.readouterr().out

    def fail(*args, **kwargs):
        raise AssertionError("o rescore sem --judge não chama a API")

    monkeypatch.setattr(cli, "build_llm", fail)
    out_dir = tmp_path / "rescore"
    assert cli.main(["--rescore", str(runs), "--only", "aritmetica", "--out", str(out_dir)]) == 0
    assert "reavaliado de" in capsys.readouterr().out
    assert len(list(out_dir.glob("*.json"))) == 1
