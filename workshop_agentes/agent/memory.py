"""Memória de trabalho: o histórico completo, reenviado a cada chamada (D2)."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


class WorkingMemory:
    """Lista de mensagens com só append e leitura sequencial.

    A API é stateless: tudo fica aqui e é reenviado. O AIMessage é guardado sem
    modificação, o que preserva as thought signatures. Sem truncamento na v1
    (se houver no futuro, nunca separar uma chamada do seu ToolMessage).
    """

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.reset()

    def reset(self) -> None:
        """Começa uma lista NOVA só com o system prompt.

        Nunca .clear(): um AgentResult anterior guarda a referência da lista antiga.
        """
        self._messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]

    @property
    def messages(self) -> list[BaseMessage]:
        """Histórico atual (somente leitura por convenção)."""
        return self._messages

    def add_user(self, text: str) -> None:
        """Adiciona uma mensagem do usuário (ou uma observação para o modelo)."""
        self._messages.append(HumanMessage(content=text))

    def add_model(self, msg: AIMessage) -> None:
        """Adiciona a resposta do modelo exatamente como veio."""
        self._messages.append(msg)

    def add_tool_result(self, msg: ToolMessage) -> None:
        """Adiciona o resultado de uma chamada de ferramenta."""
        self._messages.append(msg)

    def check_invariant(self) -> bool:
        """Cada AIMessage com tool_calls é seguida IMEDIATAMENTE por um ToolMessage por id.

        Verifica presença e posição: o bloco logo após a chamada tem exatamente um
        ToolMessage por tool_call, e nenhum ToolMessage aparece fora desse bloco.
        Ids None (o loop gera um uuid) casam por posição.
        """
        msgs = self._messages
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            i += 1
            if isinstance(msg, ToolMessage):
                return False  # resultado solto, fora do bloco da sua chamada
            if isinstance(msg, AIMessage) and msg.tool_calls:
                calls = msg.tool_calls
                block = msgs[i : i + len(calls)]
                if len(block) < len(calls) or not all(isinstance(m, ToolMessage) for m in block):
                    return False
                answered = {m.tool_call_id for m in block}
                if any(c["id"] is not None and c["id"] not in answered for c in calls):
                    return False
                i += len(calls)
        return True
