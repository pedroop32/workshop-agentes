"""Linha de comando: python -m workshop_agentes.evals [--dry-run] [--only ID ...] [--judge] [--rescore JSON].

Sem opções, roda todos os casos do golden set contra a API real (≥ 2 chamadas por caso: a 1ª é
sempre o plano), mostra o relatório e grava o .txt (com o tracing de cada run) e o .json (para
--rescore) em eval_results/. Código de saída: 0 se nenhum caso falhou ou ficou sem rodar.
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from workshop_agentes.agent.config import DEFAULT_MODEL, AgentConfig, get_secret
from workshop_agentes.agent.loop import Agent
from workshop_agentes.agent.memory import WorkingMemory
from workshop_agentes.evals.cases import EvalCase, load_golden_set
from workshop_agentes.evals.judge import Judge
from workshop_agentes.evals.report import format_case, format_summary
from workshop_agentes.evals.runner import load_results, missing_tools, rescore, run_suite, save_results
from workshop_agentes.llm import build_llm
from workshop_agentes.tools.core import update_plan
from workshop_agentes.tools.registry import ToolRegistry
from workshop_agentes.tracing.tracer import Tracer

RESULTS_DIR = "eval_results"
JUDGE_TEMPERATURE = 0.0  # o juiz deve variar o mínimo entre execuções


def build_registry() -> ToolRegistry:
    """As ferramentas do agente avaliado, as mesmas do notebook.

    Ferramenta nova: registre-a aqui (e no notebook); os casos que a têm em expected_tools
    deixam de ser pulados.
    """
    registry = ToolRegistry()
    registry.register(update_plan)
    return registry


def build_agent(model: str, api_key: str, verbose: bool = True) -> Agent:
    """O agente padrão do README e do notebook, com o modelo dado."""
    config = AgentConfig(verbose=verbose)
    llm = build_llm(model, api_key, config.temperature)
    return Agent(llm, build_registry(), WorkingMemory(config.system_prompt), Tracer(), config)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m workshop_agentes.evals", description="Avalia o agente com o golden set.")
    parser.add_argument("--dry-run", action="store_true", help="valida o golden set e mostra o que rodaria, sem API")
    parser.add_argument("--only", nargs="+", metavar="ID", help="roda só estes casos")
    parser.add_argument("--judge", action="store_true", help="avalia as rubricas com o juiz LLM (+1 chamada por caso)")
    parser.add_argument("--rescore", metavar="JSON", help="reavalia os runs salvos neste arquivo, sem rodar o agente")
    parser.add_argument("--model", help="modelo do agente (padrão: GROQ_MODEL ou DEFAULT_MODEL)")
    parser.add_argument("--judge-model", help="modelo do juiz (padrão: o do agente)")
    parser.add_argument("--golden", help="arquivo do golden set (padrão: workshop_agentes/evals/golden_set.json)")
    parser.add_argument("--out", default=RESULTS_DIR, help=f"pasta dos relatórios (padrão: {RESULTS_DIR})")
    parser.add_argument("--quiet", action="store_true", help="não mostra ao vivo as notas (→ external) e as retentativas")
    return parser.parse_args(argv)


def dry_run(cases: Sequence[EvalCase], judge: bool) -> int:
    """Mostra o que rodaria e a estimativa de chamadas, sem chave e sem API."""
    registered = [tool.name for tool in build_registry().tools]
    print(f"Golden set: {len(cases)} casos | ferramentas registradas: {', '.join(registered)}")
    runs = judge_calls = 0
    for case in cases:
        missing = missing_tools(case, registered)
        if missing:
            print(f"  pula  {case.id} (ferramenta não registrada: {', '.join(missing)})")
            continue
        runs += 1
        judged = judge and bool(case.rubric)
        judge_calls += judged
        print(f"  roda  {case.id}{' + juiz' if judged else ''}")
    extra = f" + {judge_calls} do juiz" if judge else ""
    print(f"Estimativa: ≥ {2 * runs} chamadas ao agente (a 1ª de cada run é o plano){extra}. Nenhuma foi feita.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_golden_set(args.golden)
    if args.only:
        unknown = sorted(set(args.only) - {case.id for case in cases})
        if unknown:
            print(f"Caso não encontrado no golden set: {', '.join(unknown)}")
            return 2
        cases = [case for case in cases if case.id in args.only]
    if args.dry_run:
        return dry_run(cases, args.judge)

    meta, saved = load_results(args.rescore) if args.rescore else ({}, {})
    model = meta.get("model") or args.model or get_secret("GROQ_MODEL", DEFAULT_MODEL)
    api_key = None
    if not args.rescore or args.judge:  # o rescore sem juiz não chama a API
        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            print("Defina GROQ_API_KEY (.env ou variável de ambiente). --dry-run e --rescore sem --judge não usam a API.")
            return 2
    judge_model = (args.judge_model or model) if args.judge else None
    judge = Judge(build_llm(judge_model, api_key, JUDGE_TEMPERATURE), verbose=not args.quiet) if judge_model else None
    if args.rescore:
        results = rescore(cases, saved, judge)
    else:
        results = run_suite(build_agent(model, api_key, verbose=not args.quiet), cases, judge)

    header = f"Modelo: {model}" + (f" | juiz: {judge_model}" if judge_model else "")
    if args.rescore:
        header += f" | reavaliado de {args.rescore}"
    summary = format_summary(results)
    print(f"\n{header}\n")
    print("\n\n".join(format_case(r) for r in results))
    print(f"\n{summary}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path, json_path = out / f"{stamp}.txt", out / f"{stamp}.json"
    report = "\n\n".join([header, *(format_case(r, trace=True) for r in results), summary])
    report_path.write_text(report + "\n", encoding="utf-8")
    save_results(results, json_path, model=model, judge_model=judge_model, created=stamp, rescored_from=args.rescore)
    print(f"\nRelatório: {report_path}\nRuns (para --rescore): {json_path}")
    return 0 if all(r.status in ("pass", "skip") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
