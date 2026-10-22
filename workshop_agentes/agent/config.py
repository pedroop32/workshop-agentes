"""Configuração do agente e leitura de segredos."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Verificado na API real (2026-10-21): termina a tarefa do workshop em 2 passos.
# É Preview no Groq (pode ser descontinuado com pouco aviso). Alternativa de
# produção: "openai/gpt-oss-120b", que falhou na mesma tarefa (400 tool_use_failed
# ao pôr a resposta longa em Markdown no finish).
# O secret/variável GROQ_MODEL tem prioridade sobre este valor.
DEFAULT_MODEL = "qwen/qwen3.8-27b"

# Vale para modelos com uma chamada por resposta (sem exigir update_plan "em toda resposta").
SYSTEM_PROMPT = (
    "Call `update_plan` with all required parameters (previous_step, plan, next_step, external) "
    "before any other tool call. When you have the final answer, call `finish`."
)


def get_secret(name: str, default: str | None = None) -> str | None:
    """Lê um segredo: Colab Secrets primeiro; depois variável de ambiente ou .env.

    Nunca imprime o valor. No Colab, o acesso do notebook ao secret precisa
    estar ativado no painel Secrets (ícone de chave).
    """
    try:
        from google.colab import userdata  # só existe no Colab
    except ImportError:
        pass
    else:
        try:
            value = userdata.get(name)
        except userdata.NotebookAccessError:
            raise RuntimeError(
                f"O notebook não tem acesso ao secret {name}: ative 'Notebook access' "
                "no painel Secrets (ícone de chave) do Colab."
            ) from None
        except userdata.SecretNotFoundError:
            value = None
        if value:
            return value
    load_dotenv()  # não sobrescreve variáveis já definidas
    return os.getenv(name) or default


@dataclass(slots=True)
class AgentConfig:
    """Parâmetros do agente.

    temperature=None usa o padrão do ChatGroq (0.7);
    verbose=True imprime ao vivo a nota "external" de cada update_plan.
    """

    max_steps: int = 8
    temperature: float | None = None
    system_prompt: str = SYSTEM_PROMPT
    verbose: bool = True
