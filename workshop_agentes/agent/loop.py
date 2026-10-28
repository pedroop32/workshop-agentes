"""Loop do agente: chama o modelo, executa as ferramentas pedidas e repete (D1).

Termina quando o modelo responde só em texto não vazio (a resposta final, como no padrão
da documentação do Groq) ou quando uma ferramenta terminal dá certo (D5). O loop não
conhece nenhuma ferramenta pelo nome: a que a 1ª chamada é obrigada a usar vem de
AgentConfig.first_tool.
"""

import json
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from workshop_agentes.agent.config import AgentConfig
from workshop_agentes.agent.memory import WorkingMemory
from workshop_agentes.llm import error_label, invoke_with_retry, text_of
from workshop_agentes.tools.registry import ToolRegistry
from workshop_agentes.tracing.report import preview_args
from workshop_agentes.tracing.tracer import Span, Tracer

MALFORMED_CALL_ERROR = {
    "error": "Malformed function call: it could not be parsed. Call the tools again with valid arguments."
}
EMPTY_REPLY_NUDGE = "Your last reply was empty. Reply to the User with the final answer in plain text."


@dataclass(slots=True)
class AgentResult:
    """Resultado de um run.

    reason, por que o run parou:
      "no_tool_call"  o modelo respondeu só em texto (a resposta final);
      "length"        a resposta em texto foi cortada pelo limite de tokens (answer incompleta);
      "finish"        uma ferramenta terminal deu certo;
      "max_steps"     acabaram os passos sem resposta final;
      "error"         a chamada ao modelo falhou.
    error: rótulo seguro da falha, o mesmo do tracing: tipo + limite, código ou status
      (ex.: "RateLimitError OTPM: Limit 1000, Used 999"); nunca a mensagem, que poderia vazar dados.
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
    """Agente reutilizável com dependências injetadas (llm, registry, memória, tracer, config).

    Registre as ferramentas antes de criar o Agent: o bind_tools acontece no __init__, e uma
    ferramenta registrada depois não chega ao modelo.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        registry: ToolRegistry,
        memory: WorkingMemory,
        tracer: Tracer,
        config: AgentConfig,
    ) -> None:
        # bind_tools uma vez por modo (D8): livre e, na 1ª chamada, obrigado a usar first_tool.
        self.llm = llm.bind_tools(registry.tools)
        self.first_llm = self.llm
        if config.first_tool is not None:
            if config.first_tool not in {t.name for t in registry.tools}:
                raise ValueError(f"first_tool {config.first_tool!r} não está registrada no registry")
            self.first_llm = llm.bind_tools(registry.tools, tool_choice=config.first_tool)
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
                            if step == 0 and self.cfg.first_tool is not None:
                                s.attrs["forced_tool"] = self.cfg.first_tool
                            s.attrs["retries"] = []
                            ai, attempts = invoke_with_retry(
                                self.first_llm if step == 0 else self.llm,
                                self.memory.messages,
                                on_retry=self._retry_recorder(s.attrs["retries"]),
                            )
                            s.attrs["attempts"] = attempts
                            s.attrs["usage"] = ai.usage_metadata or {}
                            finish_reason = s.attrs["finish_reason"] = ai.response_metadata.get("finish_reason")
                    except Exception as exc:  # o span já fechou com attrs["error"] = tipo
                        # Rótulo seguro (tipo + limite e números do 429), nunca a mensagem.
                        answer, reason, stop = None, "error", True
                        error = s.attrs["error"] = error_label(exc)
                        if self.cfg.verbose:
                            print(f"[erro na chamada ao modelo: {error}]")
                    else:
                        self.memory.add_model(ai)
                        text = text_of(ai.content)
                        if ai.tool_calls or ai.invalid_tool_calls:
                            if text.strip():  # com chamadas, o texto não é a resposta: fica só registrado
                                s.attrs["text_chars"] = len(text)
                            terminal_out = self._run_tool_calls(ai)
                            if terminal_out is not None:
                                answer = terminal_out.get("answer", json.dumps(terminal_out, ensure_ascii=False))
                                reason, stop = "finish", True
                        elif not text.strip() and finish_reason != "length":
                            # Resposta vazia não é resposta final: pede de novo, dentro de max_steps.
                            s.attrs["empty_reply"] = True
                            self.memory.add_user(EMPTY_REPLY_NUDGE)
                        else:
                            # Só texto: é a resposta final; "length" = o provedor a cortou no limite.
                            answer, stop = text, True
                            reason = "length" if finish_reason == "length" else "no_tool_call"
                if stop:
                    break
            run_span.attrs.update(reason=reason, finish_reason=finish_reason)
        # Montado só depois de fechar todos os spans.
        return AgentResult(answer, steps, reason, self.tracer.spans, self.memory.messages, error, finish_reason)

    def _retry_recorder(self, retries: list[str]) -> Callable[[BaseException, float], None]:
        """on_retry do invoke_with_retry: guarda cada tentativa retentada (rótulo seguro + espera)
        no span e, no verbose, avisa ao vivo, ex.: limite por minuto atingido e quanto vai esperar."""

        def record(exc: BaseException, delay: float) -> None:
            retries.append(f"{error_label(exc)} (espera {delay:.1f}s)")
            if self.cfg.verbose:
                print(f"[retentando em {delay:.1f}s: {error_label(exc)}]")

        return record

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
                if "error" in out:  # o modelo recebe o erro como observação; no tracing, só o tipo
                    s.attrs["error"] = str(out["error"]).split(":")[0]
            self.memory.add_tool_result(ToolMessage(content=content, tool_call_id=call_id, name=name))
            # "external" é a nota para o User (D4), em qualquer ferramenta que a tenha. O raciocínio
            # aparece mesmo quando a chamada é rejeitada, com o aviso.
            rejected = f" [chamada rejeitada: {s.attrs['error']}]" if "error" in s.attrs else ""
            if self.cfg.verbose and isinstance(args, dict) and args.get("external"):
                print(f"→ {args['external']}{rejected}")
            elif self.cfg.verbose:  # as demais (ex.: uma busca): o que foi pedido e o tamanho do resultado, ao vivo
                print(f"  [{name} {preview_args(args)} → {s.attrs['result_chars']} caracteres]{rejected}")
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
