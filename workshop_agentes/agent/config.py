"""Configuração do agente e leitura de segredos."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# PROVISÓRIO: estável e com plano gratuito nas páginas models/pricing (2026-10-20).
# Confirmar na integração com a API; alternativa: "gemini-3.5-flash-lite".
# O secret/variável GEMINI_MODEL tem prioridade sobre este valor.
DEFAULT_MODEL = "gemini-3.8-flash"

SYSTEM_PROMPT = (
    "Before calling any other tool, in every response you MUST first call `update_plan` "
    "with all required parameters (previous_step, plan, next_step, external). "
    "When you have the final answer, call `finish`."
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

    temperature=None usa o padrão do modelo (recomendado no Gemini 3);
    verbose=True imprime ao vivo a nota "external" de cada update_plan.
    """

    max_steps: int = 8
    temperature: float | None = None
    system_prompt: str = SYSTEM_PROMPT
    verbose: bool = True
