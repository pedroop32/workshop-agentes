"""Loop do agente: chama o modelo, executa as ferramentas pedidas e repete (D1).

O loop não conhece "finish": termina quando uma ferramenta terminal dá certo (D5).
"""

import json
from dataclasses import dataclass
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from workshop_agentes.agent.config import AgentConfig
from workshop_agentes.agent.memory import WorkingMemory
from workshop_agentes.llm import invoke_with_retry, text_of
from workshop_agentes.tools.registry import ToolRegistry
from workshop_agentes.tracing.tracer import Span, Tracer

MALFORMED_CALL_ERROR = {
    "error": "Malformed function call: it could not be parsed. Call the tools again with valid arguments."
}


@dataclass(slots=True)
class AgentResult:
    """Resultado de um run.

    reason: "finish" | "no_tool_call" | "max_steps" | "error".
    error: só o tipo da exceção (nunca a mensagem, que poderia vazar dados).
    """

    answer: str | None
    steps: int
    reason: str
    spans: list[Span]
    messages: list[BaseMessage]
    error: str | None = None


class Agent:
    """Agente reutilizável com dependências injetadas (llm, registry, memória, tracer, config)."""

    def __init__(
        self,
        llm: BaseChatModel,
        registry: ToolRegistry,
        memory: WorkingMemory,
        tracer: Tracer,
        config: AgentConfig,
    ) -> None:
        self.llm = llm.bind_tools(registry.tools)  # uma vez só (D8)
        self.registry = registry
        self.memory = memory
        self.tracer = tracer
        self.cfg = config

    def run(self, task: str, section: str | None = None) -> AgentResult:
        """Executa a tarefa até uma ferramenta terminal, resposta só em texto, erro ou max_steps.

        Reinicia memória e tracer a cada chamada; o AgentResult anterior fica intacto.
        section (opcional) vai para attrs do span "run".
        """
        self.memory.reset()
        self.tracer.reset()
        self.memory.add_user(task)
        answer, reason, error, steps = None, "max_steps", None, 0
        with self.tracer.span("run", "agent", section=section):
            for step in range(self.cfg.max_steps):
                steps = step + 1
                stop = False
                with self.tracer.span("step", f"step_{step}"):
                    try:
                        with self.tracer.span("llm_call", "model") as s:
                            ai, attempts = invoke_with_retry(self.llm, self.memory.messages)
                            s.attrs["attempts"] = attempts
                            s.attrs["usage"] = ai.usage_metadata or {}
                            s.attrs["finish_reason"] = ai.response_metadata.get("finish_reason")
                    except Exception as exc:  # o span já fechou com attrs["error"]
                        answer, reason, error, stop = None, "error", type(exc).__name__, True
                        if self.cfg.verbose:
                            print(f"[erro na chamada ao modelo: {error}]")
                    else:
                        self.memory.add_model(ai)
                        if ai.tool_calls:
                            terminal_out = self._run_tool_calls(ai)
                            if terminal_out is not None:
                                answer = terminal_out.get("answer", json.dumps(terminal_out, ensure_ascii=False))
                                reason, stop = "finish", True
                        elif _is_malformed(ai):
                            # Erro vira observação para o modelo, não exceção (6.1).
                            self.memory.add_user(json.dumps(MALFORMED_CALL_ERROR))
                        else:
                            answer, reason, stop = text_of(ai.content), "no_tool_call", True
                if stop:
                    break
        # Montado só depois de fechar todos os spans.
        return AgentResult(answer, steps, reason, self.tracer.spans, self.memory.messages, error)

    def _run_tool_calls(self, ai: AIMessage) -> dict | None:
        """Executa TODAS as chamadas da resposta e devolve a saída da terminal bem-sucedida, se houver.

        Cada chamada recebe seu ToolMessage antes de o loop poder encerrar.
        Erro de ferramenta, inclusive da terminal, volta ao modelo como observação.
        """
        terminal_out = None
        for call in ai.tool_calls:
            name = call["name"]
            call_id = call.get("id") or str(uuid4())
            args = call.get("args") or {}
            with self.tracer.span("tool_call", name) as s:
                s.attrs["args"] = args
                out = self.registry.run(name, args)
                content = json.dumps(out, ensure_ascii=False, default=str)
                s.attrs["result_chars"] = len(content)
            self.memory.add_tool_result(ToolMessage(content=content, tool_call_id=call_id, name=name))
            if self.cfg.verbose and name == "update_plan" and isinstance(args, dict) and args.get("external"):
                print(f"→ {args['external']}")
            if self.registry.is_terminal(name) and "error" not in out:
                terminal_out = out
        return terminal_out


def _is_malformed(ai: AIMessage) -> bool:
    """Resposta sem tool_calls válidas porque a chamada veio malformada."""
    return bool(ai.invalid_tool_calls) or ai.response_metadata.get("finish_reason") == "MALFORMED_FUNCTION_CALL"
