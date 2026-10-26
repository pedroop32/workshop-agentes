"""Runner dos evals: roda os casos no agente, aplica os checks e salva/relê os resultados em JSON.

Um único Agent serve todos os casos (run() reinicia memória e tracer, D8). Para não gastar API:
caso que depende de ferramenta não registrada é pulado sem chamar o modelo, e um erro que se
repetiria em todos os casos (chave, modelo, cota diária) interrompe a suíte.
"""

import json
from collections.abc import Callable, Collection, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from workshop_agentes.agent.loop import Agent, AgentResult
from workshop_agentes.evals.cases import EvalCase
from workshop_agentes.evals.checks import CheckResult, evaluate, run_metrics
from workshop_agentes.evals.judge import JUDGE_CHECK, Judge
from workshop_agentes.tracing.tracer import Span

# 1º termo do AgentResult.error (o rótulo do error_label) dos erros que se repetiriam em todo caso.
FATAL_ERRORS = ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "DailyQuotaError")
STATUS_LABELS = {"pass": "PASSOU", "fail": "FALHOU", "skip": "PULADO", "not_run": "NÃO EXECUTADO"}
RESULT_FIELDS = ("answer", "steps", "reason", "error", "finish_reason")  # o AgentResult salvo, sem mensagens


@dataclass(slots=True)
class CaseResult:
    """Resultado de um caso.

    status: "pass" (nenhum check reprovado), "fail", "skip" (ferramenta esperada não registrada)
      ou "not_run" (suíte interrompida por erro fatal); note explica skip e not_run;
    result: o AgentResult do run (None se não rodou; relido de JSON, vem sem as mensagens);
    judge_tokens: tokens gastos pelo juiz neste caso.
    """

    case: EvalCase
    status: str
    checks: list[CheckResult] = field(default_factory=list)
    result: AgentResult | None = None
    note: str | None = None
    judge_tokens: int = 0


def missing_tools(case: EvalCase, registered: Collection[str]) -> list[str]:
    """expected_tools do caso que não estão registradas (o caso é pulado)."""
    return [name for name in case.expected_tools if name not in registered]


def score(case: EvalCase, result: AgentResult, judge: Judge | None = None) -> CaseResult:
    """Checks determinísticos + rubrica; o juiz só é chamado se existir, houver rubrica e resposta."""
    checks = evaluate(case, result)
    judge_tokens = 0
    if case.rubric:
        if judge is None:
            checks.append(CheckResult(JUDGE_CHECK, None, "não avaliado: rode com --judge"))
        elif not (result.answer or "").strip():
            checks.append(CheckResult(JUDGE_CHECK, None, "não avaliado: sem resposta"))
        else:
            check, judge_tokens = judge.grade(case, result.answer)
            checks.append(check)
    status = "fail" if any(c.passed is False for c in checks) else "pass"
    return CaseResult(case, status, checks, result, judge_tokens=judge_tokens)


def run_suite(
    agent: Agent,
    cases: Sequence[EvalCase],
    judge: Judge | None = None,
    log: Callable[[str], None] | None = print,
) -> list[CaseResult]:
    """Roda os casos em ordem e devolve um CaseResult por caso; log recebe o progresso (None = silêncio)."""
    registered = {tool.name for tool in agent.registry.tools}
    results: list[CaseResult] = []
    fatal = None
    for number, case in enumerate(cases, start=1):
        prefix = f"[{number}/{len(cases)}] {case.id}"
        missing = missing_tools(case, registered)
        if fatal is not None:
            outcome = CaseResult(case, "not_run", note=fatal)
        elif missing:
            outcome = CaseResult(case, "skip", note=f"ferramenta não registrada: {', '.join(missing)}")
        else:
            if log:
                log(f"{prefix}: rodando...")
            outcome = score(case, agent.run(case.task, section=f"eval:{case.id}"), judge)
            error = outcome.result.error
            if error and error.split()[0] in FATAL_ERRORS:
                fatal = f"suíte interrompida: {error} em {case.id} (se repetiria em todos os casos)"
        results.append(outcome)
        if log:
            log(f"{prefix}: {STATUS_LABELS[outcome.status]}{_progress_detail(outcome)}")
    return results


def rescore(
    cases: Sequence[EvalCase], saved: dict[str, dict[str, Any]], judge: Judge | None = None
) -> list[CaseResult]:
    """Reavalia runs salvos (load_results) com os casos atuais, sem rodar o agente.

    O juiz só é chamado se for dado. Caso sem run salvo mantém o status salvo (skip/not_run) ou,
    se o caso é novo, fica "skip".
    """
    results = []
    for case in cases:
        item = saved.get(case.id)
        if item is None:
            results.append(CaseResult(case, "skip", note="sem run salvo para este caso"))
        elif item.get("result") is None:
            results.append(CaseResult(case, item["status"], note=item.get("note")))
        else:
            results.append(score(case, _result_from_dict(item["result"]), judge))
    return results


def save_results(results: Sequence[CaseResult], path: str | Path, **meta: Any) -> None:
    """Grava a suíte em JSON: meta (modelo, data...) e, por caso, status, checks e o run (sem mensagens)."""
    data = {**meta, "cases": [_case_to_dict(r) for r in results]}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_results(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Lê um JSON de save_results: (meta, {id do caso: dados salvos}) para o rescore."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    saved = {item["id"]: item for item in data.pop("cases")}
    return data, saved


def _case_to_dict(r: CaseResult) -> dict[str, Any]:
    run = None
    if r.result is not None:
        run = {key: getattr(r.result, key) for key in RESULT_FIELDS}
        run["spans"] = [asdict(s) for s in r.result.spans]
    return {
        "id": r.case.id,
        "status": r.status,
        "note": r.note,
        "judge_tokens": r.judge_tokens,
        "checks": [asdict(c) for c in r.checks],
        "result": run,
    }


def _result_from_dict(data: dict[str, Any]) -> AgentResult:
    """AgentResult de volta, com os spans (Span(**d)) e sem as mensagens, que não são salvas."""
    spans = [Span(**s) for s in data["spans"]]
    return AgentResult(spans=spans, messages=[], **{key: data.get(key) for key in RESULT_FIELDS})


def _progress_detail(outcome: CaseResult) -> str:
    if outcome.result is None:
        return f" ({outcome.note})"
    latency = run_metrics(outcome.result)["run_latency"]
    return f" ({latency or 0:.2f}s, {outcome.result.steps} passos, {outcome.result.reason})"
