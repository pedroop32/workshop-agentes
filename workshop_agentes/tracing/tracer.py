"""Tracing próprio e explícito (D3): spans numa lista plana, ligados por parent_id."""

import itertools
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class Span:
    """Um trecho medido. Hierarquia: run -> step -> llm_call | tool_call."""

    id: int
    parent_id: int | None
    kind: str
    name: str
    t_start: float
    t_end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def latency(self) -> float | None:
        """Segundos entre início e fim (None enquanto o span está aberto)."""
        return None if self.t_end is None else self.t_end - self.t_start


class Tracer:
    """Registra spans; uma pilha interna acha o pai do próximo span em O(1)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Cria listas NOVAS (nunca .clear(): um AgentResult anterior guarda as antigas)."""
        self.spans: list[Span] = []
        self._stack: list[Span] = []
        self._ids = itertools.count()

    @contextmanager
    def span(self, kind: str, name: str, **attrs: Any) -> Iterator[Span]:
        """Abre um span filho do span atual e o fecha mesmo se houver exceção.

        Em caso de exceção, registra attrs["error"] = tipo e re-propaga.
        attrs livres (ex.: section="demo") vão direto para span.attrs.
        """
        parent_id = self._stack[-1].id if self._stack else None
        current = Span(next(self._ids), parent_id, kind, name, perf_counter(), attrs=dict(attrs))
        self.spans.append(current)
        self._stack.append(current)
        try:
            yield current
        except Exception as exc:
            current.attrs["error"] = type(exc).__name__
            raise
        finally:
            current.t_end = perf_counter()
            self._stack.pop()
