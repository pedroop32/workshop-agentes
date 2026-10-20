"""Testes offline de agent/memory.py."""

from fakes import PLAN_ARGS, ai, call
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from workshop_agentes.agent.memory import WorkingMemory


def tool_msg(c: dict, content: str = '{"status": "ok"}') -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=c["id"], name=c["name"])


def test_reset_keeps_only_system_prompt():
    mem = WorkingMemory("sys")
    mem.add_user("tarefa")
    mem.reset()
    assert len(mem.messages) == 1
    assert isinstance(mem.messages[0], SystemMessage)
    assert mem.messages[0].content == "sys"


def test_append_and_read_preserve_ai_message():
    mem = WorkingMemory("sys")
    msg = ai(call("update_plan", PLAN_ARGS))
    msg.additional_kwargs["__gemini_function_call_thought_signatures__"] = {"x": "sig=="}
    mem.add_user("tarefa")
    mem.add_model(msg)
    assert mem.messages[2] is msg  # o mesmo objeto, sem cópia nem reconstrução
    assert mem.messages[2].additional_kwargs["__gemini_function_call_thought_signatures__"] == {"x": "sig=="}
    assert isinstance(mem.messages[1], HumanMessage)


def test_invariant_valid_with_parallel_calls():
    mem = WorkingMemory("sys")
    mem.add_user("tarefa")
    a, b = call("update_plan", PLAN_ARGS), call("finish", {"answer": "x"})
    mem.add_model(ai(a, b))
    mem.add_tool_result(tool_msg(b))  # a ordem dos ids dentro do bloco não importa
    mem.add_tool_result(tool_msg(a))
    mem.add_model(ai(content="fim"))
    assert mem.check_invariant()


def test_invariant_accepts_generated_id_for_none_id():
    mem = WorkingMemory("sys")
    c = call("update_plan", PLAN_ARGS, id=None)
    mem.add_model(ai(c))
    mem.add_tool_result(ToolMessage(content="{}", tool_call_id="uuid-gerado", name="update_plan"))
    assert mem.check_invariant()


def test_invariant_detects_missing_pair():
    mem = WorkingMemory("sys")
    mem.add_model(ai(call("update_plan", PLAN_ARGS)))
    assert not mem.check_invariant()


def test_invariant_detects_one_missing_parallel_result():
    mem = WorkingMemory("sys")
    a, b = call("update_plan", PLAN_ARGS), call("finish", {"answer": "x"})
    mem.add_model(ai(a, b))
    mem.add_tool_result(tool_msg(a))
    mem.add_user("outra coisa")
    assert not mem.check_invariant()


def test_invariant_detects_wrong_order():
    mem = WorkingMemory("sys")
    c = call("update_plan", PLAN_ARGS)
    mem.add_model(ai(c))
    mem.add_user("mensagem no meio")  # quebra o "imediatamente"
    mem.add_tool_result(tool_msg(c))
    assert not mem.check_invariant()


def test_invariant_detects_wrong_id():
    mem = WorkingMemory("sys")
    mem.add_model(ai(call("update_plan", PLAN_ARGS)))
    mem.add_tool_result(ToolMessage(content="{}", tool_call_id="outro-id", name="update_plan"))
    assert not mem.check_invariant()


def test_invariant_detects_orphan_tool_message():
    mem = WorkingMemory("sys")
    mem.add_tool_result(ToolMessage(content="{}", tool_call_id="x", name="update_plan"))
    assert not mem.check_invariant()


def test_list_obtained_before_reset_stays_intact():
    mem = WorkingMemory("sys")
    mem.add_user("primeira tarefa")
    old = mem.messages
    mem.reset()
    mem.add_user("segunda tarefa")
    assert [m.content for m in old] == ["sys", "primeira tarefa"]
    assert mem.messages is not old
