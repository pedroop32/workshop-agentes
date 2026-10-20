"""Testes offline de tools/registry.py e tools/core.py (sem API)."""

import pytest
from fakes import PLAN_ARGS
from langchain_core.tools import tool

from workshop_agentes.tools.core import finish, update_plan
from workshop_agentes.tools.registry import ToolRegistry


@tool
def boom(x: int) -> dict:
    """Always fails."""
    raise RuntimeError("deu ruim\nsegunda linha")


@tool
def double(x: int) -> int:
    """Returns 2 * x (not a dict)."""
    return 2 * x


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(update_plan)
    reg.register(finish, terminal=True)
    reg.register(boom)
    reg.register(double)
    return reg


def test_finish_runs_without_api(registry):
    assert registry.run("finish", {"answer": "Use array dinâmico."}) == {"answer": "Use array dinâmico."}


def test_update_plan_returns_ok(registry):
    assert registry.run("update_plan", PLAN_ARGS) == {"status": "ok"}


def test_unknown_tool(registry):
    assert registry.run("nao_existe", {}) == {"error": "Unknown tool: nao_existe"}


@pytest.mark.parametrize("args", [None, [1, 2], "texto", 3])
def test_non_dict_args_return_error(registry, args):
    out = registry.run("finish", args)
    assert out["error"].startswith("TypeError:")


def test_missing_required_arg_returns_short_error(registry):
    args = {k: v for k, v in PLAN_ARGS.items() if k != "external"}
    out = registry.run("update_plan", args)
    assert out == {"error": "ValidationError: external: Field required"}


@pytest.mark.parametrize("args", [{}, {"answer": 123}])
def test_invalid_finish_args_return_error(registry, args):
    out = registry.run("finish", args)
    assert out["error"].startswith("ValidationError: answer:")
    assert "\n" not in out["error"]


def test_tool_exception_returns_error(registry):
    assert registry.run("boom", {"x": 1}) == {"error": "RuntimeError: deu ruim"}


def test_non_dict_return_is_wrapped(registry):
    assert registry.run("double", {"x": 21}) == {"result": 42}


def test_is_terminal(registry):
    assert registry.is_terminal("finish")
    assert not registry.is_terminal("update_plan")
    assert not registry.is_terminal("nao_existe")


def test_tools_in_registration_order(registry):
    assert [t.name for t in registry.tools] == ["update_plan", "finish", "boom", "double"]


def test_update_plan_schema_descriptions():
    props = update_plan.args
    assert set(props) == {"previous_step", "plan", "next_step", "external"}
    assert props["previous_step"]["description"] == "Key findings and outcomes since the previous step."
    assert props["external"]["description"].startswith("A short, plain-language note shown to the User")
    assert finish.args["answer"]["type"] == "string"
