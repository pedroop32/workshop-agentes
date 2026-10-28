"""Testes offline das visualizações do notebook (workshop_agentes/visual.py): HTML gerado, sem IPython nem API."""

from html.parser import HTMLParser

import pytest
from fakes import PLAN_ARGS, FlakyRunnable, ScriptedChatModel, ai, call, not_found_error, rate_limit_error
from langchain_core.messages import AIMessage

from workshop_agentes import rag, visual
from workshop_agentes.agent.config import AgentConfig
from workshop_agentes.agent.loop import Agent
from workshop_agentes.agent.memory import WorkingMemory
from workshop_agentes.evals.cases import EvalCase
from workshop_agentes.evals.runner import run_suite
from workshop_agentes.tools.core import update_plan
from workshop_agentes.tools.documents import make_search_documents
from workshop_agentes.tools.registry import ToolRegistry
from workshop_agentes.tracing.tracer import Tracer

EVIL = "<script>alert(1)</script>"
TEXT = f"O prazo é de 30 dias. {EVIL}"


class FakeEngine:
    def __init__(self) -> None:
        self.files = ["manual.pdf"]
        self.chunks = [rag.Chunk("manual.pdf", 2, TEXT), rag.Chunk("manual.pdf", 3, "Outro trecho.")]
        self.warnings = ["manual.pdf, página 1: sem texto suficiente; pode precisar de OCR."]

    def refresh(self) -> None:
        pass

    def retrieve(self, question, *, pdfs=None, k=6):
        return [rag.Hit(chunk, 0.9) for chunk in self.chunks]


def make_agent(script):
    registry = ToolRegistry()
    registry.register(update_plan)
    registry.register(make_search_documents(FakeEngine()))
    cfg = AgentConfig(verbose=False)
    return Agent(ScriptedChatModel(script=list(script)), registry, WorkingMemory(cfg.system_prompt), Tracer(), cfg)


def rag_script(answer=f"30 dias (manual.pdf, p. 2). {EVIL}"):
    return [
        ai(call("update_plan", PLAN_ARGS), finish_reason="tool_calls"),
        ai(call("search_documents", {"query": "prazo"}), finish_reason="tool_calls",
           usage={"input_tokens": 300, "output_tokens": 30, "total_tokens": 330}),
        ai(content=answer, usage={"input_tokens": 900, "output_tokens": 40, "total_tokens": 940}),
    ]


@pytest.fixture
def rag_result():
    return make_agent(rag_script()).run("Qual é o prazo?")


@pytest.fixture
def shown(monkeypatch):
    """O que as funções show_* exibiriam no notebook."""
    out = []
    monkeypatch.setattr(visual, "_show", out.append)
    monkeypatch.setattr(visual, "_show_markdown", out.append)
    return out


class _TagBalance(HTMLParser):
    VOID = {"br", "hr", "img", "meta", "link", "input"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.scripts = 0

    def handle_starttag(self, tag, attrs):
        self.scripts += tag == "script"
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack and self.stack.pop() == tag, f"</{tag}> fora de ordem"


def assert_safe_html(text: str) -> None:
    """HTML bem formado e sem nenhuma tag <script> (o conteúdo de fora sai escapado)."""
    parser = _TagBalance()
    parser.feed(text)
    parser.close()
    assert not parser.stack, f"tags sem fechar: {parser.stack}"
    assert parser.scripts == 0


def test_every_view_escapes_outside_content(rag_result):
    for page in (
        visual.result_html(rag_result),
        visual.passages_html(rag_result),
        visual.trace_html(rag_result),
        visual.memory_html(rag_result),
        visual.index_html(FakeEngine(), 1.5),
        visual.tools_html(make_agent([])),
    ):
        assert_safe_html(page)
    assert "&lt;script&gt;" in visual.passages_html(rag_result)
    assert "&lt;script&gt;" in visual.memory_html(rag_result)


def test_answer_markdown_keeps_formatting_and_escapes_tags(rag_result):
    assert visual.answer_markdown(rag_result) == "30 dias (manual.pdf, p. 2). &lt;script&gt;alert(1)&lt;/script&gt;"


def test_result_shows_reason_steps_tools_and_tokens(rag_result):
    page = visual.result_html(rag_result)
    assert "motivo: no_tool_call" in page and "passos: 3" in page
    assert "ferramentas: update_plan, search_documents" in page


def test_tools_table_has_name_description_and_the_forced_first_tool():
    page = visual.tools_html(make_agent([]))
    assert "2 ferramentas que o modelo pode pedir" in page
    assert "<code>search_documents</code>" in page and "Search the User&#x27;s PDF documents" in page
    assert "1ª chamada obrigatória" in page and "<code>query</code>" in page


def test_index_shows_files_pages_chunks_and_warnings():
    page = visual.index_html(FakeEngine(), 54.5)
    assert "PDFs: 1" in page and "páginas com texto: 2" in page and "trechos: 2" in page and "tempo: 54.5s" in page
    assert "pode precisar de OCR" in page


def test_passages_show_each_search_with_file_and_page(rag_result):
    page = visual.passages_html(rag_result)
    assert "busca 1" in page and "Consulta: <b>prazo</b>" in page and "manual.pdf · p. 3" in page
    assert "Nenhuma busca neste run." in visual.passages_html(make_agent(rag_script()[2:]).run("t"))


def test_trace_has_one_row_per_step_with_context_growing(rag_result):
    page = visual.trace_html(rag_result)
    assert page.count("<tr><td><b>") == 3
    assert "tool_choice forçado: update_plan" in page and "próximo passo: Compare both structures." in page
    assert "<code>{&quot;query&quot;: &quot;prazo&quot;}</code>" in page
    assert "<td>900<div class='bar ' style='width:100%'>" in page  # o maior contexto é a referência das barras
    assert "resposta final em texto" in page and "Relatório completo (texto)" in page


def test_trace_shows_retries_and_errors():
    agent = make_agent([ai(call("update_plan", PLAN_ARGS)), ai(call("nao_existe")), ai(content="ok")])
    page = visual.trace_html(agent.run("t"))
    assert "ferramenta: nao_existe" in page and "rejeitada: Unknown tool" in page


def test_memory_is_grouped_by_step_with_folded_system_prompt_and_passages(rag_result):
    page = visual.memory_html(rag_result)
    assert page.count("<div class='who'>Passo ") == 3
    assert "<details><summary><span class='who'>System prompt</span>" in page
    assert "User · a tarefa" in page and "pede <code>search_documents</code>" in page
    assert "2 trechos (p. 2, p. 3)" in page and "Passo 3 · modelo: resposta final" in page
    assert "recebeu 900 tokens, gerou 40" in page
    assert f"{len(rag_result.messages)} mensagens" in page


def test_evals_table_shows_status_checks_and_summary():
    agent = make_agent(rag_script() + [ai(call("update_plan", PLAN_ARGS)), ai(content="Não sei.")])
    cases = [
        EvalCase("doc", "Qual é o prazo?", must_include=["30 dias"], expected_tools=["search_documents"]),
        EvalCase("nao-busca", "Oi", must_include=["391"], forbidden_tools=["search_documents"]),
        EvalCase("pulado", "t", expected_tools=["send_email"]),
    ]
    page = visual.evals_html(run_suite(agent, cases, log=None))
    assert_safe_html(page)
    assert "passaram: 1/2" in page and "pulados: 1" in page
    assert "PASSOU" in page and "FALHOU" in page and "PULADO" in page
    assert "✓ chamou search_documents" in page and "✗ contém &quot;391&quot;" in page
    assert "ferramenta não registrada: send_email" in page and "=== Resumo dos evals ===" in page


def test_check_connection_shows_the_reply_or_what_to_do(shown):
    reply = AIMessage(content="conexão funcionando!", usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                      response_metadata={"finish_reason": "stop"})
    assert visual.check_connection(FlakyRunnable([], reply)) is True
    assert "conectado" in shown[-1] and "conexão funcionando!" in shown[-1] and "entrada 12 · saída 4" in shown[-1]
    assert visual.check_connection(FlakyRunnable([not_found_error()])) is False
    assert "falhou" in shown[-1] and "NotFoundError" in shown[-1] and "GROQ_MODEL" in shown[-1]
    assert "does not exist" not in shown[-1]  # nunca a mensagem bruta do erro


def test_connection_error_for_rate_limit_explains_the_wait():
    page = visual.connection_error_html(rate_limit_error(window="output tokens per minute (OTPM)"))
    assert "RateLimitError OTPM" in page and "espere cerca de 1 minuto" in page and "org_fake" not in page


def test_show_result_displays_the_card_and_the_answer(shown, rag_result):
    visual.show_result(rag_result)
    assert len(shown) == 2 and "motivo: no_tool_call" in shown[0] and shown[1].startswith("30 dias")
