"""Relatório dos evals em texto: um bloco por caso e o resumo da suíte (f-strings, como o do tracing)."""

from collections import Counter
from collections.abc import Sequence
from statistics import mean, median
from typing import Any

from workshop_agentes.evals.checks import is_completed, run_metrics
from workshop_agentes.evals.judge import JUDGE_CHECK
from workshop_agentes.evals.runner import STATUS_LABELS, CaseResult
from workshop_agentes.tracing.report import TOKEN_KEYS, format_report

CHECK_MARKS = {True: "✓", False: "✗", None: "–"}
ANSWER_PREVIEW = 300  # caracteres da resposta no terminal; o arquivo .txt tem a resposta inteira


def format_case(r: CaseResult, trace: bool = False) -> str:
    """Status, métricas do tracing, cada check e a resposta.

    trace=True (o arquivo .txt) mostra a resposta inteira e o relatório de tracing do run.
    """
    lines = [f"[{STATUS_LABELS[r.status]}] {r.case.id}: {r.case.task}"]
    if r.note:
        lines.append(f"  {r.note}")
    if r.result is None:
        return "\n".join(lines)
    res, m = r.result, run_metrics(r.result)
    latency = f"{m['run_latency']:.2f}s" if m["run_latency"] is not None else "-"
    lines.append(
        f"  run: {res.reason} | passos={res.steps} | finish_reason={res.finish_reason} | latência {latency}"
        f" | tokens in={m['input_tokens']} out={m['output_tokens']} total={m['total_tokens']}"
    )
    lines.append(
        f"  chamadas ao modelo={m['llm_calls']} | retentativas={m['retries']}"
        f" | ferramentas: {', '.join(m['tools_used']) or '-'}"
    )
    lines.extend(f"  {CHECK_MARKS[c.passed]} {c.name}" + (f": {c.detail}" if c.detail else "") for c in r.checks)
    answer = res.answer or "-"
    if not trace and len(answer) > ANSWER_PREVIEW:
        answer = answer[:ANSWER_PREVIEW] + "…"
    lines.append("  resposta: " + answer.replace("\n", "\n    "))
    if trace:
        lines.append(format_report(res.spans))
    return "\n".join(lines)


def summarize(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Números da suíte; as taxas são sobre os casos executados (pulados e não executados ficam fora)."""
    executed = [r for r in results if r.result is not None]
    metrics = [run_metrics(r.result) for r in executed]
    latencies = [m["run_latency"] for m in metrics if m["run_latency"] is not None]
    status = Counter(r.status for r in results)
    judged = [c for r in executed for c in r.checks if c.name == JUDGE_CHECK and c.passed is not None]
    summary: dict[str, Any] = {
        "cases": len(results),
        "executed": len(executed),
        "passed": status["pass"],
        "failed": status["fail"],
        "skipped": status["skip"],
        "not_run": status["not_run"],
        "completed": sum(is_completed(r.result) for r in executed),
        "model_errors": sum(r.result.reason == "error" for r in executed),
        "tool_errors": sum(len(m["tool_errors"]) for m in metrics),
        "reasons": dict(Counter(r.result.reason for r in executed)),
        "latency_mean": mean(latencies) if latencies else None,
        "latency_median": median(latencies) if latencies else None,
        "latency_max": max(latencies) if latencies else None,
        "llm_calls": sum(m["llm_calls"] for m in metrics),
        "retries": sum(m["retries"] for m in metrics),
        "judged": len(judged),
        "judge_passed": sum(bool(c.passed) for c in judged),
        "judge_tokens": sum(r.judge_tokens for r in results),
        "failed_checks": dict(Counter(c.name for r in executed for c in r.checks if c.passed is False).most_common()),
    }
    summary.update({key: sum(m[key] for m in metrics) for key in TOKEN_KEYS})
    summary["success_rate"] = summary["passed"] / len(executed) if executed else None
    summary["completion_rate"] = summary["completed"] / len(executed) if executed else None
    summary["tokens_mean"] = summary["total_tokens"] / len(executed) if executed else None
    return summary


def format_summary(results: Sequence[CaseResult]) -> str:
    """Resumo da suíte: contagens, taxas, erros, latência, tokens, juiz e os checks que falharam."""
    s = summarize(results)
    lines = [
        "=== Resumo dos evals ===",
        f"Casos: {s['cases']} | executados={s['executed']} | passaram={s['passed']} | falharam={s['failed']}"
        f" | pulados={s['skipped']} | não executados={s['not_run']}",
    ]
    if not s["executed"]:
        lines.append("Nenhum caso executado.")
        return "\n".join(lines)
    lines += [
        f"Taxa de sucesso: {s['success_rate']:.0%} ({s['passed']}/{s['executed']})"
        f" | conclusão: {s['completion_rate']:.0%} ({s['completed']}/{s['executed']})"
        f" | runs com erro do modelo: {s['model_errors']} | erros de ferramenta: {s['tool_errors']}",
        "Motivos de parada: " + ", ".join(f"{reason}={n}" for reason, n in s["reasons"].items()),
    ]
    if s["latency_mean"] is not None:
        lines.append(
            f"Latência por caso: média {s['latency_mean']:.2f}s | mediana {s['latency_median']:.2f}s"
            f" | máx {s['latency_max']:.2f}s"
        )
    lines += [
        f"Tokens: total={s['total_tokens']} (in={s['input_tokens']} out={s['output_tokens']})"
        f" | média por caso={s['tokens_mean']:.0f}",
        f"Chamadas ao modelo: {s['llm_calls']} | retentativas: {s['retries']}",
    ]
    if s["judged"] or s["judge_tokens"]:
        lines.append(f"Juiz: {s['judged']} avaliados, {s['judge_passed']} aprovados | tokens do juiz={s['judge_tokens']}")
    if s["failed_checks"]:
        lines.append("Checks que falharam: " + "; ".join(f"{name} ({n})" for name, n in s["failed_checks"].items()))
    return "\n".join(lines)
