"""Relatório de tracing em texto (f-strings, sem pandas)."""

from collections import defaultdict
from collections.abc import Sequence

from workshop_agentes.tracing.tracer import Span

LEAF_KINDS = ("llm_call", "tool_call")
TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")
PLAN_FIELDS = ("previous_step", "plan", "next_step")

APPROXIMATION_NOTE = (
    "Nota: o input_tokens de cada chamada inclui todo o histórico reenviado (inclusive o que o "
    "modelo gerou antes); a variação entre chamadas é só uma aproximação do custo de cada passo."
)
FAILED_ATTEMPTS_NOTE = "Nota: tentativas que falharam não entram nos tokens (a API não devolve usage em erros)."


def usage_of(span: Span) -> dict[str, int]:
    """Tokens de um span; campos ausentes (ou usage None) contam como 0."""
    usage = span.attrs.get("usage") or {}
    return {key: int(usage.get(key) or 0) for key in TOKEN_KEYS}


def totals(spans: Sequence[Span]) -> dict[str, float]:
    """Totais somando só as folhas (llm_call e tool_call); somar todos contaria em dobro."""
    leaves = [s for s in spans if s.kind in LEAF_KINDS]
    result: dict[str, float] = {key: sum(usage_of(s)[key] for s in leaves) for key in TOKEN_KEYS}
    result["latency"] = sum(s.latency or 0.0 for s in leaves)
    result["llm_calls"] = sum(s.kind == "llm_call" for s in leaves)
    result["tool_calls"] = sum(s.kind == "tool_call" for s in leaves)
    return result


def attempts_of(span: Span) -> int:
    """Tentativas de um llm_call; numa chamada que falhou, são os retries + a última."""
    if "attempts" in span.attrs:
        return int(span.attrs["attempts"])
    return len(span.attrs.get("retries") or []) + 1 if "error" in span.attrs else 0


def format_report(spans: Sequence[Span]) -> str:
    """Resumo por passo, por função e total: tokens, latência, finish_reason, plano e motivo de parada."""
    children: dict[int | None, list[Span]] = defaultdict(list)
    for s in spans:
        children[s.parent_id].append(s)

    lines = ["=== Relatório de tracing ==="]
    for number, step in enumerate((s for s in spans if s.kind == "step"), start=1):
        lines.extend(_step_lines(number, step, children[step.id]))

    lines.append("--- Por função ---")
    for (kind, name), group in _by_function(spans).items():
        tokens = sum(usage_of(s)["total_tokens"] for s in group)
        latency = sum(s.latency or 0.0 for s in group)
        lines.append(f"{name} ({kind}): chamadas={len(group)} tokens={tokens} latência={latency:.2f}s")

    total = totals(spans)
    run = next((s for s in spans if s.kind == "run"), None)
    run_latency = f" | duração do run={run.latency:.2f}s" if run and run.latency is not None else ""
    lines.append("--- Total (soma das folhas) ---")
    lines.append(
        f"tokens: in={total['input_tokens']} out={total['output_tokens']} total={total['total_tokens']}"
        f" | latência das folhas={total['latency']:.2f}s{run_latency}"
    )
    if run and "reason" in run.attrs:
        lines.append(
            f"Motivo de parada: {run.attrs['reason']} | finish_reason da última resposta: {run.attrs.get('finish_reason')}"
        )
    lines.append(APPROXIMATION_NOTE)
    if any(s.kind == "llm_call" and (s.attrs.get("retries") or "error" in s.attrs) for s in spans):
        lines.append(FAILED_ATTEMPTS_NOTE)
    return "\n".join(lines)


def _step_lines(number: int, step: Span, kids: list[Span]) -> list[str]:
    llm_calls = [s for s in kids if s.kind == "llm_call"]
    tool_calls = [s for s in kids if s.kind == "tool_call"]
    usage = {key: sum(usage_of(s)[key] for s in llm_calls) for key in TOKEN_KEYS}
    finish = ", ".join(str(s.attrs.get("finish_reason")) for s in llm_calls) or "-"
    attempts = sum(attempts_of(s) for s in llm_calls)
    latency = f"{step.latency:.2f}s" if step.latency is not None else "aberto"

    lines = [
        f"Passo {number} | tokens in={usage['input_tokens']} out={usage['output_tokens']} "
        f"total={usage['total_tokens']} | latência {latency} | finish_reason={finish} | tentativas={attempts}",
        f"  ferramentas: {', '.join(s.name for s in tool_calls) or '-'}",
    ]
    for s in llm_calls:
        if s.attrs.get("retries"):
            lines.append(f"  retentado: {'; '.join(s.attrs['retries'])}")
    for s in kids:
        if "error" in s.attrs:
            lines.append(f"  erro em {s.name}: {s.attrs['error']}")
    for s in tool_calls:
        if s.name == "update_plan":
            args = s.attrs.get("args")
            args = args if isinstance(args, dict) else {}
            lines.extend(f"  {key}: {args.get(key, '-')}" for key in PLAN_FIELDS)
    return lines


def _by_function(spans: Sequence[Span]) -> dict[tuple[str, str], list[Span]]:
    groups: dict[tuple[str, str], list[Span]] = defaultdict(list)
    for s in spans:
        if s.kind in LEAF_KINDS:
            groups[(s.kind, s.name)].append(s)
    return groups
