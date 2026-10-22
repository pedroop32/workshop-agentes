"""Loop do agente: chama o modelo, executa as ferramentas pedidas e repete (D1).

Termina quando o modelo responde só em texto (a resposta final, como no padrão da
documentação do Groq) ou quando uma ferramenta terminal dá certo (D5). O loop não
conhece nenhuma ferramenta pelo nome.
"""

import json
from dataclasses import dataclass
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from workshop_agentes.agent.config import AgentConfig
from workshop_agentes.agent.memory import WorkingMemory
from workshop_agentes.llm import error_label, invoke_with_retry, text_of
from workshop_agentes.tools.registry import ToolRegistry
from workshop_agentes.tracing.tracer import Span, Tracer

MALFORMED_CALL_ERROR = {
    "error": "Malformed function call: it could not be parsed. Call the tools again with valid arguments."
}


@dataclass(slots=True)
class AgentResult:
    """Resultado de um run.

    reason, por que o run parou:
      "no_tool_call"  o modelo respondeu só em texto (a resposta final);
      "length"        a resposta em texto foi cortada pelo limite de tokens (answer incompleta);
      "finish"        uma ferramenta terminal deu certo;
      "max_steps"     acabaram os passos sem resposta final;
      "error"         a chamada ao modelo falhou.
    error: só o tipo da exceção (nunca a mensagem, que poderia vazar dados).
    finish_reason: o motivo informado pelo provedor na última resposta recebida
      ("stop", "tool_calls", "length"...).
    """

    answer: str | None
    steps: int
    reason: str
    spans: list[Span]
    messages: list[BaseMessage]
    error: str | None = None
    finish_reason: str | None = None


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
        """Executa a tarefa até a resposta em texto, uma ferramenta terminal, erro ou max_steps.

        Reinicia memória e tracer a cada chamada; o AgentResult anterior fica intacto.
        section (opcional) vai para attrs do span "run", junto com o motivo de parada.
        """
        self.memory.reset()
        self.tracer.reset()
        self.memory.add_user(task)
        answer, reason, error, steps, finish_reason = None, "max_steps", None, 0, None
        with self.tracer.span("run", "agent", section=section) as run_span:
            for step in range(self.cfg.max_steps):
                steps = step + 1
                stop = False
                with self.tracer.span("step", f"step_{step}"):
                    try:
                        with self.tracer.span("llm_call", "model") as s:
                            # Cada tentativa que falhou e foi retentada: rótulo seguro + espera.
                            retries = s.attrs["retries"] = []
                            ai, attempts = invoke_with_retry(
                                self.llm,
                                self.memory.messages,
                                on_retry=lambda exc, delay: retries.append(f"{error_label(exc)} (espera {delay:.1f}s)"),
                            )
                            s.attrs["attempts"] = attempts
                            s.attrs["usage"] = ai.usage_metadata or {}
                            finish_reason = s.attrs["finish_reason"] = ai.response_metadata.get("finish_reason")
                    except Exception as exc:  # o span já fechou com attrs["error"]
                        answer, reason, error, stop = None, "error", type(exc).__name__, True
                        if self.cfg.verbose:
                            print(f"[erro na chamada ao modelo: {error}]")
                    else:
                        self.memory.add_model(ai)
                        if ai.tool_calls or ai.invalid_tool_calls:
                            terminal_out = self._run_tool_calls(ai)
                            if terminal_out is not None:
                                answer = terminal_out.get("answer", json.dumps(terminal_out, ensure_ascii=False))
                                reason, stop = "finish", True
                        else:
                            # Só texto: é a resposta final; "length" = o provedor a cortou no limite.
                            answer, stop = text_of(ai.content), True
                            reason = "length" if finish_reason == "length" else "no_tool_call"
                if stop:
                    break
            run_span.attrs.update(reason=reason, finish_reason=finish_reason)
        # Montado só depois de fechar todos os spans.
        return AgentResult(answer, steps, reason, self.tracer.spans, self.memory.messages, error, finish_reason)

    def _run_tool_calls(self, ai: AIMessage) -> dict | None:
        """Responde TODAS as chamadas da resposta e devolve a saída da terminal bem-sucedida, se houver.

        Cada chamada recebe seu ToolMessage antes de o loop poder encerrar, inclusive as
        malformadas (invalid_tool_calls): o ChatGroq as reenvia como tool_calls e a API
        espera uma resposta por id. Erro de ferramenta, inclusive da terminal, volta ao
        modelo como observação (6.1).
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
        for call in ai.invalid_tool_calls:  # argumentos ilegíveis: nada é executado
            name = call.get("name") or "?"
            with self.tracer.span("tool_call", name) as s:
                s.attrs.update(args=call.get("args"), error="MalformedArguments")
            self.memory.add_tool_result(
                ToolMessage(
                    content=json.dumps(MALFORMED_CALL_ERROR), tool_call_id=call.get("id") or str(uuid4()), name=name
                )
            )
        return terminal_out
