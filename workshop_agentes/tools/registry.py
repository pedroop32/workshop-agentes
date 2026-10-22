"""Registry de ferramentas: nome -> ferramenta, mais o conjunto das terminais."""

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import ValidationError


class ToolRegistry:
    """Guarda as ferramentas do agente e as executa sem nunca levantar exceção."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._terminal: set[str] = set()

    def register(self, tool: BaseTool, terminal: bool = False) -> None:
        """Registra a ferramenta; terminal=True encerra o agente quando ela dá certo.

        Registrar de novo o mesmo nome substitui a ferramenta e a marca de terminal.
        """
        self._tools[tool.name] = tool
        if terminal:
            self._terminal.add(tool.name)
        else:
            self._terminal.discard(tool.name)

    @property
    def tools(self) -> list[BaseTool]:
        """Ferramentas na ordem de registro (para bind_tools)."""
        return list(self._tools.values())

    def is_terminal(self, name: str) -> bool:
        """A ferramenta encerra o agente? (busca O(1) num set)"""
        return name in self._terminal

    def run(self, name: str, args: Any) -> dict:
        """Executa a ferramenta e devolve sempre um dict; nunca levanta.

        Nome desconhecido, args inválidos ou falha na execução viram
        {"error": "Tipo: msg curta"}, que volta ao modelo como observação.
        Retorno que não é dict vira {"result": valor}.
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        if not isinstance(args, dict):
            return {"error": f"TypeError: arguments must be an object, got {type(args).__name__}"}
        try:
            out = tool.invoke(args)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {_short_message(exc)}"}
        return out if isinstance(out, dict) else {"result": out}


def _short_message(exc: Exception, limit: int = 200) -> str:
    """Mensagem de erro curta e de uma linha (a do pydantic tem várias e uma URL)."""
    if isinstance(exc, ValidationError):
        message = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'arguments'}: {err['msg']}"
            for err in exc.errors()
        )
    else:
        lines = str(exc).strip().splitlines()
        message = lines[0] if lines else ""
    return message[:limit]
