"""Juiz LLM opcional: avalia a resposta do agente contra a rubrica do caso (1 chamada por caso).

Só roda com --judge (ou JUIZ = True no notebook), em casos com rubrica e resposta não vazia.
A chamada passa pelo invoke_with_retry, o mesmo retry com limite do agente (D9). O juiz vê só a
tarefa, a rubrica e a resposta (não o plano do agente).
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from workshop_agentes.evals.cases import EvalCase
from workshop_agentes.evals.checks import CheckResult
from workshop_agentes.llm import error_label, invoke_with_retry

JUDGE_CHECK = "rubrica (juiz)"
JUDGE_PROMPT = (
    "You grade an AI assistant's answer against a rubric. Read the task, the rubric and the answer, "
    "then call `Verdict`. Set passed to true only if the answer meets every point of the rubric. "
    "Do not solve the task yourself."
)


class Verdict(BaseModel):
    """Your grade for the answer."""

    passed: bool = Field(description="True only if the answer meets every point of the rubric.")
    reason: str = Field(description="One short sentence explaining the grade.")


class Judge:
    """Juiz com o modelo injetado: bind_tools uma vez, obrigado a chamar Verdict (tool_choice com o nome).

    verbose=True avisa ao vivo cada retentativa, como o agente (ex.: limite por minuto e a espera).
    """

    def __init__(self, llm: BaseChatModel, verbose: bool = True) -> None:
        self.llm = llm.bind_tools([Verdict], tool_choice="Verdict")
        self.verbose = verbose

    def grade(self, case: EvalCase, answer: str) -> tuple[CheckResult, int]:
        """Veredito e tokens gastos; falha do juiz vira passed=None (não avaliado), nunca reprova o agente."""
        messages = [
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=f"Task:\n{case.task}\n\nRubric:\n{case.rubric}\n\nAnswer:\n{answer}"),
        ]
        try:
            ai, _ = invoke_with_retry(self.llm, messages, on_retry=self._announce_retry)
        except Exception as exc:  # rótulo seguro, nunca a mensagem
            return CheckResult(JUDGE_CHECK, None, f"juiz falhou: {error_label(exc)}"), 0
        tokens = int((ai.usage_metadata or {}).get("total_tokens") or 0)
        try:
            verdict = Verdict.model_validate(ai.tool_calls[0]["args"])
        except (IndexError, ValidationError):
            return CheckResult(JUDGE_CHECK, None, "juiz falhou: veredito ausente ou inválido"), tokens
        return CheckResult(JUDGE_CHECK, verdict.passed, verdict.reason), tokens

    def _announce_retry(self, exc: BaseException, delay: float) -> None:
        if self.verbose:
            print(f"[juiz retentando em {delay:.1f}s: {error_label(exc)}]")
