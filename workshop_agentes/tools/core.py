"""Ferramentas básicas: update_plan (raciocínio explícito, D4) e finish (terminal opcional, D5).

O agente padrão termina com a resposta em texto; finish continua disponível para quem
quiser uma ferramenta terminal: registry.register(finish, terminal=True).
Descrições em inglês: são lidas pelo modelo.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class UpdatePlanArgs(BaseModel):
    previous_step: str = Field(description="Key findings and outcomes since the previous step.")
    plan: str = Field(description="The current status of the plan.")
    next_step: str = Field(description="Brief explanation of the immediate next action according to the plan.")
    external: str = Field(
        description="A short, plain-language note shown to the User about what you are ABOUT TO DO next."
    )


@tool(args_schema=UpdatePlanArgs)
def update_plan(previous_step: str, plan: str, next_step: str, external: str) -> dict:
    """Record your plan: findings so far, plan status and the next action. It does not act or fetch information."""
    return {"status": "ok"}


class FinishArgs(BaseModel):
    answer: str = Field(description="The final answer for the User, complete and self-contained.")


@tool(args_schema=FinishArgs)
def finish(answer: str) -> dict:
    """Deliver the final answer to the User and end the task."""
    return {"answer": answer}
