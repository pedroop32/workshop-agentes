"""Checks determinísticos de um run: conclusão, erros, termos esperados, passos e ferramentas.

Leem só o AgentResult (resposta, motivo de parada e spans do tracing), nunca as mensagens: um
run salvo em JSON é reavaliado sem chamar a API. Os checks de ferramenta usam os nomes dos spans
tool_call, então valem para qualquer ferramenta registrada no futuro.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from workshop_agentes.agent.loop import AgentResult
from workshop_agentes.evals.cases import EvalCase
from workshop_agentes.tracing.report import totals

COMPLETED_REASONS = ("no_tool_call", "finish")  # os motivos de parada com resposta final
# Nota de cada tentativa retentada, como o loop a grava em attrs["retries"] (Agent._retry_recorder)
# e o juiz a repete: "<rótulo do error_label> (espera 3.1s)". Lida daqui, vale também para runs salvos.
RETRY_WAIT = re.compile(r"\(espera (\d+(?:\.\d+)?)s\)$")


@dataclass(slots=True)
class CheckResult:
    """Resultado de um check; passed=None = não avaliado (ex.: rubrica sem --judge)."""

    name: str
    passed: bool | None
    detail: str = ""


def rate_limit_wait(retries: Sequence[str]) -> float:
    """Segundos de espera antes de retentar um 429, somados (cada nota traz a espera arredondada a 0,1 s)."""
    waits = (RETRY_WAIT.search(note) for note in retries if note.startswith("RateLimitError"))
    return sum(float(match.group(1)) for match in waits if match)


def run_metrics(result: AgentResult) -> dict[str, Any]:
    """Métricas do tracing: os totais das folhas, mais latência do run, retentativas e ferramentas.

    run_latency: duração do run (span "run"), com as esperas de retry; net_latency: a mesma sem as
    esperas por 429 (rate_limit_wait), o que sobra de rede, fila e geração;
    tool_errors: "ferramenta: tipo" de cada tool_call rejeitada (desconhecida, args inválidos,
    chamada malformada); tools_used: nomes na ordem do primeiro uso.
    """
    metrics: dict[str, Any] = totals(result.spans)
    run = next((s for s in result.spans if s.kind == "run"), None)
    tool_calls = [s for s in result.spans if s.kind == "tool_call"]
    retries = [note for s in result.spans if s.kind == "llm_call" for note in s.attrs.get("retries") or []]
    metrics["run_latency"] = run.latency if run is not None else None
    metrics["retries"] = len(retries)
    metrics["rate_limit_wait"] = rate_limit_wait(retries)
    metrics["net_latency"] = (
        max(0.0, metrics["run_latency"] - metrics["rate_limit_wait"]) if metrics["run_latency"] is not None else None
    )
    metrics["tool_errors"] = [f"{s.name}: {s.attrs['error']}" for s in tool_calls if "error" in s.attrs]
    metrics["tools_used"] = list(dict.fromkeys(s.name for s in tool_calls))
    return metrics


def normalize(text: str) -> str:
    """Minúsculas e sem acentos, para comparar termos ("Dinâmico" casa com "dinamico")."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def is_completed(result: AgentResult) -> bool:
    """O run terminou com resposta final não vazia? ("length" = cortada, não conta)"""
    return result.reason in COMPLETED_REASONS and bool((result.answer or "").strip())


def evaluate(case: EvalCase, result: AgentResult) -> list[CheckResult]:
    """Sempre: concluiu e ferramentas sem erro; os demais conforme os campos do caso."""
    metrics = run_metrics(result)
    answer = normalize(result.answer or "")
    checks = [
        CheckResult("concluiu", is_completed(result), result.reason + (f" ({result.error})" if result.error else "")),
        CheckResult("ferramentas sem erro", not metrics["tool_errors"], "; ".join(metrics["tool_errors"])),
    ]
    for item in case.must_include:
        options = [item] if isinstance(item, str) else list(item)
        checks.append(CheckResult(f"contém {_quoted(options)}", any(normalize(o) in answer for o in options)))
    for term in case.must_not_include:
        checks.append(CheckResult(f"não contém {_quoted([term])}", normalize(term) not in answer))
    if case.max_steps is not None:
        checks.append(CheckResult(f"passos <= {case.max_steps}", result.steps <= case.max_steps, str(result.steps)))
    used = metrics["tools_used"]
    detail = f"usadas: {', '.join(used) or '-'}"  # só quando o check falha
    checks.extend(
        CheckResult(f"chamou {name}", name in used, "" if name in used else detail) for name in case.expected_tools
    )
    checks.extend(
        CheckResult(f"não chamou {name}", name not in used, "" if name not in used else detail)
        for name in case.forbidden_tools
    )
    return checks


def _quoted(options: list[str]) -> str:
    return " ou ".join(f'"{option}"' for option in options)
