"""Testes offline do loop do agente (9.1), com o modelo falso.

Escritos antes do loop para revelar cedo problemas de interface.
"""

import json
from functools import partial

import pytest
from fakes import PLAN_ARGS, ScriptedChatModel, ai, bad_request_error, call, tool_use_failed_error
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from workshop_agentes import llm
from workshop_agentes.agent import loop
from workshop_agentes.agent.config import AgentConfig
from workshop_agentes.agent.loop import Agent
from workshop_agentes.agent.memory import WorkingMemory
from workshop_agentes.tools.core import finish, update_plan
from workshop_agentes.tools.registry import ToolRegistry
from workshop_agentes.tracing.report import totals
from workshop_agentes.tracing.tracer import Tracer


def make_agent(script, **config):
    model = ScriptedChatModel(script=list(script))
    registry = ToolRegistry()
    registry.register(update_plan)
    registry.register(finish, terminal=True)
    cfg = AgentConfig(**{"verbose": False, **config})
    agent = Agent(model, registry, WorkingMemory(cfg.system_prompt), Tracer(), cfg)
    return agent, model


def tool_results(messages) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def tool_outputs(messages) -> list[dict]:
    return [json.loads(m.content) for m in tool_results(messages)]


# --- O próprio dublê (roda desde já) ----------------------------------------------


def test_fake_model_returns_scripted_message_and_records_input():
    msg = ai(call("update_plan", PLAN_ARGS))
    model = ScriptedChatModel(script=[msg])
    bound = model.bind_tools(["t"])
    assert bound.invoke([HumanMessage("oi")]) is msg
    assert model.bind_calls == 1
    assert [m.content for m in model.received[0]] == ["oi"]


def test_fake_model_raises_scripted_exception():
    error = bad_request_error()
    model = ScriptedChatModel(script=[error])
    with pytest.raises(type(error)):
        model.invoke([HumanMessage("oi")])


# --- Loop (9.1) ----------------------------------------------------------------


def test_update_plan_then_finish():
    agent, model = make_agent([
        ai(call("update_plan", PLAN_ARGS), usage=None),  # caso usage_metadata=None
        ai(call("update_plan", PLAN_ARGS), call("finish", {"answer": "Use array dinâmico."})),
    ])
    result = agent.run("Compare lista encadeada e array dinâmico para uma fila.")
    assert result.reason == "finish"
    assert result.answer == "Use array dinâmico."
    assert result.steps == 2
    assert result.error is None
    assert model.bind_calls == 1  # bind_tools uma vez só


def test_parallel_calls_one_tool_message_per_id():
    a, b = call("update_plan", PLAN_ARGS), call("update_plan", {**PLAN_ARGS, "next_step": "Conclude."})
    agent, _ = make_agent([ai(a, b), ai(call("finish", {"answer": "ok"}))])
    result = agent.run("tarefa")
    first_block = [m.tool_call_id for m in tool_results(result.messages)][:2]
    assert first_block == [a["id"], b["id"]]
    assert agent.memory.check_invariant()


def test_terminal_with_other_call_answers_all_before_ending():
    f, u = call("finish", {"answer": "pronto"}), call("update_plan", PLAN_ARGS)
    agent, _ = make_agent([ai(f, u)])  # terminal ANTES da outra chamada
    result = agent.run("tarefa")
    assert result.reason == "finish" and result.answer == "pronto"
    assert {m.tool_call_id for m in tool_results(result.messages)} == {f["id"], u["id"]}
    assert agent.memory.check_invariant()


@pytest.mark.parametrize("bad_args", [{}, {"answer": 123}])
def test_terminal_with_error_does_not_end(bad_args):
    agent, model = make_agent([
        ai(call("update_plan", PLAN_ARGS), call("finish", bad_args)),
        ai(call("update_plan", PLAN_ARGS), call("finish", {"answer": "agora sim"})),
    ])
    result = agent.run("tarefa")
    assert result.reason == "finish" and result.answer == "agora sim" and result.steps == 2
    seen_by_model = tool_outputs(model.received[1])
    assert "error" in seen_by_model[-1]  # o modelo recebeu o erro do finish


def test_unknown_tool_returns_error_and_agent_continues():
    agent, model = make_agent([
        ai(call("nao_existe", {"x": 1})),
        ai(call("finish", {"answer": "ok"})),
    ])
    result = agent.run("tarefa")
    assert result.reason == "finish"
    assert tool_outputs(model.received[1]) == [{"error": "Unknown tool: nao_existe"}]


def test_invalid_or_missing_args_do_not_raise():
    missing_external = {k: v for k, v in PLAN_ARGS.items() if k != "external"}
    agent, model = make_agent(
        [ai(call("update_plan", missing_external)), ai(call("finish", {"answer": "ok"}))],
        verbose=True,  # imprimir "external" ausente não pode dar KeyError
    )
    result = agent.run("tarefa")
    assert result.reason == "finish"
    assert "error" in tool_outputs(model.received[1])[0]


def test_max_steps_without_terminal():
    agent, _ = make_agent([ai(call("update_plan", PLAN_ARGS))], max_steps=1)
    result = agent.run("tarefa")
    assert result.reason == "max_steps"
    assert result.answer is None
    assert result.steps == 1


@pytest.mark.parametrize(
    "content, expected",
    [("Resposta direta.", "Resposta direta."), ([{"type": "text", "text": "Em blocos."}], "Em blocos.")],
)
def test_text_only_response_ends_with_no_tool_call(content, expected):
    agent, _ = make_agent([ai(content=content)])
    result = agent.run("tarefa")
    assert result.reason == "no_tool_call"
    assert result.answer == expected


def test_model_failure_returns_error_without_raising():
    agent, _ = make_agent([bad_request_error()])
    result = agent.run("tarefa")
    assert result.reason == "error"
    assert result.error == "BadRequestError"  # só o tipo, nunca a mensagem
    assert result.answer is None
    assert all(s.t_end is not None for s in result.spans)
    llm_span = next(s for s in result.spans if s.kind == "llm_call")
    assert llm_span.attrs["error"] == "BadRequestError"


def test_retried_errors_are_traced_in_the_llm_span(monkeypatch):
    monkeypatch.setattr(loop, "invoke_with_retry", partial(llm.invoke_with_retry, sleep=lambda seconds: None))
    agent, _ = make_agent([tool_use_failed_error(), ai(call("finish", {"answer": "ok"}))])
    result = agent.run("tarefa")
    assert result.reason == "finish"
    llm_span = next(s for s in result.spans if s.kind == "llm_call")
    assert llm_span.attrs["attempts"] == 2
    assert len(llm_span.attrs["retries"]) == 1
    assert llm_span.attrs["retries"][0].startswith("BadRequestError tool_use_failed (espera ")


def test_tool_call_with_none_id_gets_generated_id():
    agent, _ = make_agent([
        ai(call("update_plan", PLAN_ARGS, id=None)),
        ai(call("finish", {"answer": "ok"})),
    ])
    result = agent.run("tarefa")
    generated = tool_results(result.messages)[0].tool_call_id
    assert isinstance(generated, str) and generated
    assert agent.memory.check_invariant()


def test_spans_closed_and_totals_equal_sum_of_leaves():
    agent, _ = make_agent([
        ai(call("update_plan", PLAN_ARGS)),
        ai(call("update_plan", PLAN_ARGS), call("finish", {"answer": "ok"})),
    ])
    result = agent.run("tarefa")
    assert all(s.t_end is not None for s in result.spans)
    kinds = [s.kind for s in result.spans]
    assert kinds.count("run") == 1 and kinds.count("step") == 2
    assert kinds.count("llm_call") == 2 and kinds.count("tool_call") == 3
    total = totals(result.spans)
    llm_spans = [s for s in result.spans if s.kind == "llm_call"]
    assert total["total_tokens"] == sum(s.attrs["usage"]["total_tokens"] for s in llm_spans) == 240
    assert total["input_tokens"] == 200 and total["output_tokens"] == 40


def test_run_twice_does_not_accumulate_and_keeps_first_result():
    agent, _ = make_agent([
        ai(call("finish", {"answer": "primeira"})),
        ai(call("update_plan", PLAN_ARGS)),
        ai(call("finish", {"answer": "segunda"})),
    ])
    first = agent.run("tarefa 1")
    first_messages, first_spans = list(first.messages), list(first.spans)
    second = agent.run("tarefa 2")
    assert second.answer == "segunda" and second.steps == 2
    assert first.answer == "primeira"
    assert first.messages == first_messages and first.spans == first_spans  # intactos
    assert first.messages is not second.messages and first.spans is not second.spans
    assert sum(isinstance(m, HumanMessage) for m in second.messages) == 1


def test_verbose_prints_external(capsys):
    agent, _ = make_agent(
        [ai(call("update_plan", PLAN_ARGS), call("finish", {"answer": "ok"}))], verbose=True
    )
    agent.run("tarefa")
    assert PLAN_ARGS["external"] in capsys.readouterr().out


def test_malformed_call_becomes_error_observation():
    agent, model = make_agent([
        AIMessage(
            content="",
            invalid_tool_calls=[
                {"name": "finish", "args": '{"answer": ', "id": "call_bad", "error": "JSON inválido", "type": "invalid_tool_call"}
            ],
        ),
        ai(call("finish", {"answer": "ok"})),
    ])
    result = agent.run("tarefa")
    assert result.reason == "finish" and result.steps == 2
    observation = model.received[1][-1]
    assert isinstance(observation, HumanMessage) and "error" in json.loads(observation.content)
    assert agent.memory.check_invariant()
