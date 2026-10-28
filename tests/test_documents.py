"""Testes offline da ferramenta search_documents: motor de busca falso e modelo falso, sem numpy nem API."""

import json
from types import SimpleNamespace

from fakes import PLAN_ARGS, ScriptedChatModel, ai, call

from workshop_agentes import rag
from workshop_agentes.agent.config import AgentConfig
from workshop_agentes.agent.loop import Agent
from workshop_agentes.agent.memory import WorkingMemory
from workshop_agentes.evals import __main__ as cli
from workshop_agentes.evals.cases import EvalCase, load_golden_set
from workshop_agentes.evals.runner import run_suite
from workshop_agentes.tools import documents
from workshop_agentes.tools.core import update_plan
from workshop_agentes.tools.documents import NO_PASSAGES_NOTE, make_search_documents
from workshop_agentes.tools.registry import ToolRegistry
from workshop_agentes.tracing.report import format_report
from workshop_agentes.tracing.tracer import Tracer

TEXT = "O prazo para solicitar reembolso é de 30 dias após a compra."


class FakeEngine:
    """O que a ferramenta usa do PdfRag (refresh, chunks, files e retrieve), sem PDFs nem embeddings."""

    def __init__(self, hits=None, files=("manual.pdf", "outro.pdf")) -> None:
        self.hits = [rag.Hit(rag.Chunk("manual.pdf", 2, TEXT), 0.9)] if hits is None else hits
        self.files = list(files)
        self.chunks = [hit.chunk for hit in self.hits] or ([rag.Chunk(self.files[0], 1, TEXT)] if self.files else [])
        self.refreshes = 0
        self.searches = []

    def refresh(self) -> None:
        self.refreshes += 1

    def retrieve(self, question, *, pdfs=None, k=6):
        self.searches.append((question, pdfs, k))
        return self.hits


def search(engine, **args) -> dict:
    """Roda a ferramenta como o agente roda: pelo ToolRegistry, que nunca levanta."""
    registry = ToolRegistry()
    registry.register(make_search_documents(engine))
    return registry.run("search_documents", args)


# --- A ferramenta -------------------------------------------------------------------


def test_returns_passages_with_file_page_and_text():
    engine = FakeEngine()
    out = search(engine, query="prazo de reembolso")
    assert out == {
        "passages": [{"id": 1, "file": "manual.pdf", "page": 2, "text": TEXT}],
        "files": ["manual.pdf", "outro.pdf"],
    }
    assert engine.refreshes == 1  # PDFs novos ou alterados entram antes da busca
    assert engine.searches == [("prazo de reembolso", None, 6)]


def test_chosen_files_are_passed_to_the_search():
    engine = FakeEngine()
    search(engine, query="compare os prazos", files=["manual.pdf", "outro.pdf"])
    assert engine.searches[0][1] == ["manual.pdf", "outro.pdf"]


def test_unknown_file_is_an_observation_listing_the_available_ones():
    engine = FakeEngine()
    out = search(engine, query="prazo", files=["manual.pdf", "contrato.pdf"])
    assert out == {"error": "UnknownFile: contrato.pdf; available: manual.pdf, outro.pdf"}
    assert engine.searches == []


def test_without_documents_the_model_gets_an_error():
    out = search(FakeEngine(hits=[], files=()), query="prazo")
    assert out["error"].startswith("NoDocuments:")


def test_search_without_passages_says_so():
    out = search(FakeEngine(hits=[]), query="cotação do dólar")
    assert out["passages"] == [] and out["note"] == NO_PASSAGES_NOTE


def test_invalid_arguments_are_rejected_by_the_schema():
    assert search(FakeEngine(), files=["manual.pdf"])["error"].startswith("ValidationError")


# --- No agente: tracing e evals -----------------------------------------------------


def make_agent(script, engine):
    model = ScriptedChatModel(script=list(script))
    registry = ToolRegistry()
    registry.register(update_plan)
    registry.register(make_search_documents(engine))
    cfg = AgentConfig(verbose=False)
    return Agent(model, registry, WorkingMemory(cfg.system_prompt), Tracer(), cfg), model


def plan_search_answer(query: str, answer: str) -> list:
    """O run típico com documentos: o plano forçado, a busca e a resposta em texto (3 chamadas)."""
    return [
        ai(call("update_plan", PLAN_ARGS), finish_reason="tool_calls"),
        ai(call("search_documents", {"query": query}), finish_reason="tool_calls"),
        ai(content=answer),
    ]


def test_agent_searches_then_answers_and_tracing_shows_the_query():
    agent, model = make_agent(
        plan_search_answer("prazo de reembolso", "O prazo é de 30 dias (manual.pdf, p. 2)."), FakeEngine()
    )
    result = agent.run("Qual é o prazo de reembolso?")
    assert result.reason == "no_tool_call" and result.steps == 3
    [span] = [s for s in result.spans if s.kind == "tool_call" and s.name == "search_documents"]
    assert span.attrs["args"] == {"query": "prazo de reembolso"} and "error" not in span.attrs
    observation = json.loads(model.received[2][-1].content)  # o modelo viu os trechos antes de responder
    assert observation["passages"][0]["page"] == 2
    report = format_report(result.spans)
    assert f'  search_documents: {{"query": "prazo de reembolso"}} → {span.attrs["result_chars"]} caracteres' in report
    assert "  plan: 1. Compare. 2. Conclude." in report  # o plano continua como antes


def test_eval_case_expecting_the_tool_passes():
    agent, _ = make_agent(plan_search_answer("prazo", "30 dias, segundo o manual.pdf."), FakeEngine())
    case = EvalCase("doc", "Qual é o prazo?", must_include=[".pdf"], expected_tools=["search_documents"])
    [outcome] = run_suite(agent, [case], log=None)
    assert outcome.status == "pass"


def test_golden_set_has_a_document_case_and_forbids_the_tool_where_it_does_not_help():
    cases = {c.id: c for c in load_golden_set()}
    assert cases["documentos-fonte"].expected_tools == ["search_documents"]
    assert cases["aritmetica"].forbidden_tools == ["search_documents"]


def test_evals_register_the_tool_only_when_documents_are_available(monkeypatch):
    monkeypatch.setattr(cli, "documents_available", lambda: False)
    assert [t.name for t in cli.build_registry().tools] == ["update_plan"]
    monkeypatch.setattr(cli, "documents_available", lambda: True)
    assert [t.name for t in cli.build_registry().tools] == ["update_plan", "search_documents"]


def test_document_case_is_skipped_without_documents(monkeypatch, capsys):
    monkeypatch.setattr(cli, "documents_available", lambda: False)
    assert cli.main(["--dry-run", "--only", "documentos-fonte"]) == 0
    assert "pula  documentos-fonte (ferramenta não registrada: search_documents)" in capsys.readouterr().out


def test_documents_available_needs_the_packages_and_a_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "DOCUMENTS", tmp_path)
    monkeypatch.setattr(documents, "find_spec", lambda name: SimpleNamespace())
    assert not documents.documents_available()  # pasta sem PDF
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.PDF").write_bytes(b"%PDF-1.4")
    assert documents.documents_available()
    monkeypatch.setattr(documents, "find_spec", lambda name: None if name == "fastembed" else SimpleNamespace())
    assert not documents.documents_available()  # falta uma dependência


def test_long_tool_arguments_are_shortened_in_the_report():
    tracer = Tracer()
    with tracer.span("step", "step_0"):
        with tracer.span("tool_call", "finish") as s:
            s.attrs.update(args={"answer": "x" * 500}, result_chars=520)
    line = next(line for line in format_report(tracer.spans).splitlines() if line.startswith("  finish:"))
    assert line.endswith("… → 520 caracteres") and len(line) < 200
