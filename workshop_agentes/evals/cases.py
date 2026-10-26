"""Casos do golden set: o que pedir ao agente e o que esperar da resposta."""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

GOLDEN_SET = Path(__file__).with_name("golden_set.json")
LIST_FIELDS = ("must_include", "must_not_include", "expected_tools", "forbidden_tools")


@dataclass(slots=True)
class EvalCase:
    """Um caso do golden set.

    must_include: cada item é um termo ou uma lista de alternativas (basta uma), comparados sem
      diferenciar maiúsculas e acentos;
    max_steps: check de eficiência (o limite do próprio agente continua em AgentConfig.max_steps);
    expected_tools / forbidden_tools: ferramentas que o run deve / não pode chamar. Um caso cujas
      expected_tools não estão registradas é pulado, sem chamar a API;
    rubric: critério em inglês para o juiz LLM (só avaliado com --judge).
    """

    id: str
    task: str
    must_include: list[str | list[str]] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    max_steps: int | None = None
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    rubric: str | None = None


def load_golden_set(path: str | Path | None = None) -> list[EvalCase]:
    """Lê e valida o golden set (padrão: golden_set.json ao lado deste módulo).

    Campo desconhecido levanta ValueError: um erro de digitação (ex.: "must_includes") desligaria
    um check em silêncio. Pelo mesmo motivo, os campos de lista precisam ser listas ("391" seria
    lido caractere por caractere). id e task são obrigatórios e os ids, únicos.
    """
    raw = json.loads(Path(path or GOLDEN_SET).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("o golden set deve ser uma lista de casos")
    known = {f.name for f in fields(EvalCase)}
    cases, seen = [], set()
    for number, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"caso #{number}: deve ser um objeto")
        label = item.get("id", f"#{number}")
        unknown = sorted(set(item) - known)
        if unknown:
            raise ValueError(f"caso {label}: campo desconhecido {', '.join(unknown)}")
        if not item.get("id") or not item.get("task"):
            raise ValueError(f"caso {label}: id e task são obrigatórios")
        not_lists = [name for name in LIST_FIELDS if not isinstance(item.get(name, []), list)]
        if not_lists:
            raise ValueError(f"caso {label}: {', '.join(not_lists)} deve ser uma lista")
        if item["id"] in seen:
            raise ValueError(f"caso {label}: id repetido")
        seen.add(item["id"])
        cases.append(EvalCase(**item))
    return cases
