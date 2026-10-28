"""Regressões offline: evidências, recusa, atualização e métricas honestas."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from workshop_agentes import rag

np = pytest.importorskip("numpy")  # dependências do RAG: pip install -r requirements-rag.txt


TEXT = "O prazo para solicitar reembolso é de 30 dias após a compra."


def hit():
    return rag.Hit(rag.Chunk("manual.pdf", 2, TEXT), 0.9)


def payload():
    return {"claims": [{"text": "O prazo é de 30 dias após a compra.",
                        "evidence": [{"id": 1, "quote": TEXT}]}]}


@pytest.mark.parametrize("evidence", [
    {"id": 2, "quote": TEXT},
    {"id": True, "quote": TEXT},
    {"id": 1, "quote": "O prazo para solicitar reembolso é de 60 dias após a compra."},
    {"id": 1, "quote": "30 dias"},
])
def test_rejects_fabricated_or_unusable_evidence(evidence):
    data = payload()
    data["claims"][0]["evidence"] = [evidence]
    assert rag.validate_claims(data, [hit()]) == []


def test_answer_requires_both_real_quote_and_positive_review(monkeypatch):
    from workshop_agentes import llm

    responses = iter([payload(), {"supported": True}])
    monkeypatch.setattr(llm, "invoke_with_retry", lambda *a, **kw: (
        AIMessage(content=json.dumps(next(responses))), 1
    ))
    model = SimpleNamespace(bind=lambda **kw: None)
    result = rag.ask("Qual é o prazo?", [hit()], model)
    assert result["answered"]
    assert result["sources"][0]["page"] == 2
    assert result["answer"] == payload()["claims"][0]["text"]
    assert result["indicators"]["precision_percent"] is None
    assert result["indicators"]["model_review_approved"] is True
    assert rag.format_answer(result).endswith("(1.0)")
    responses = iter([payload(), {"supported": False}])
    assert not rag.ask("Qual é o prazo?", [hit()], model)["answered"]
    assert not rag.ask("Qual é o prazo?", [], model)["answered"]


def test_one_bad_quote_does_not_discard_other_supported_paragraphs():
    data = payload()
    data["claims"].append({"text": "Informação sem suporte.",
                           "evidence": [{"id": 99, "quote": TEXT}]})
    assert rag.validate_claims(data, [hit()]) == payload()["claims"]


def test_accepts_short_literal_evidence():
    data = payload()
    data["claims"][0]["evidence"][0]["quote"] = "após a compra."
    assert rag.validate_claims(data, [hit()]) == data["claims"]


def test_review_removes_only_unsupported_paragraph_and_recomputes_score(monkeypatch):
    from workshop_agentes import llm

    data = payload()
    data["claims"].insert(0, {"text": "O prazo é de 60 dias.",
                              "evidence": [{"id": 1, "quote": TEXT}]})
    responses = iter([data, {"supported_ids": [2]}])
    monkeypatch.setattr(llm, "invoke_with_retry", lambda *a, **kw: (
        AIMessage(content=json.dumps(next(responses))), 1
    ))
    result = rag.ask("Qual é o prazo?", [hit()], SimpleNamespace(bind=lambda **kw: None))
    assert result["answered"]
    assert "60" not in result["answer"]
    assert result["answer"] == payload()["claims"][0]["text"]
    assert result["indicators"]["textual_support_percent"] == 100


@pytest.mark.parametrize("approved", [[], [True], [2], "1"])
def test_invalid_or_empty_review_selection_never_approves_answer(monkeypatch, approved):
    from workshop_agentes import llm

    responses = iter([payload(), {"supported_ids": approved}])
    monkeypatch.setattr(llm, "invoke_with_retry", lambda *a, **kw: (
        AIMessage(content=json.dumps(next(responses))), 1
    ))
    result = rag.ask("Qual é o prazo?", [hit()], SimpleNamespace(bind=lambda **kw: None))
    assert not result["answered"]


def test_support_indicators_use_cited_sources_and_weakest_claim():
    data = payload()["claims"]
    data.append({"text": "O prazo é de 60 dias.", "evidence": [{"id": 1, "quote": TEXT}]})
    indicators = rag.response_indicators(data, [hit(), rag.Hit(rag.Chunk("outro.pdf", 1, TEXT), .99)])
    # prazo/dias têm suporte textual; o número 60 não tem. Fonte não citada é ignorada.
    assert indicators["textual_support_percent"] == 66.7
    assert indicators["semantic_similarity"] == .9
    assert indicators["precision_percent"] is None


def test_full_textual_overlap_is_not_reported_as_accuracy():
    indicators = rag.response_indicators(payload()["claims"], [hit()])
    assert indicators["textual_support_percent"] == 100
    assert indicators["precision_percent"] is None
    result = {"answer": "Resposta", "answered": True, "sources": [], "indicators": indicators}
    text = rag.format_answer(result)
    assert text == "Resposta (1.0)"


def test_refusal_has_no_invented_confidence_or_zero_accuracy():
    result = rag.ask("Pergunta sem documentos", [], None)
    assert all(value is None for value in result["indicators"].values())
    text = rag.format_answer(result)
    assert text == rag.REFUSAL


@pytest.mark.parametrize("support, expected", [(70, "0.7"), (75, "0.75"), (80, "0.8")])
def test_only_complete_answer_and_final_score_are_displayed(support, expected):
    result = {
        "answer": "Primeiro parágrafo completo.\n\nSegundo parágrafo com os detalhes.",
        "answered": True, "sources": [{"file": "manual.pdf", "page": 2, "quote": TEXT}],
        "indicators": {"textual_support_percent": support, "precision_percent": None},
    }
    assert rag.format_answer(result) == f"{result['answer']} ({expected})"


def test_changed_removed_and_cached_documents(tmp_path, monkeypatch):
    documents = tmp_path / "documentos"
    documents.mkdir()
    pdf = documents / "manual.PDF"
    pdf.write_bytes(b"version1")
    reads = []

    def extract(path, name):
        reads.append(path.read_bytes())
        return [rag.Chunk(name, 1, TEXT + path.read_text())], []

    class Encoder:
        def embed(self, texts):
            return iter(np.asarray([[1, 0, 0] for _ in texts], dtype=np.float32))

    monkeypatch.setattr(rag, "read_pdf", extract)
    monkeypatch.setattr(rag.PdfRag, "_encoder", lambda self: Encoder())
    cache = tmp_path / "cache"
    engine = rag.PdfRag(documents, cache)
    assert engine.retrieve("reembolso")[0].chunk.file == "manual.PDF"
    assert not engine.retrieve("reembolso", pdfs="inexistente.pdf")
    rag.PdfRag(documents, cache).refresh()
    assert len(reads) == 1  # índice persistido reutilizado
    pdf.write_bytes(b"version2")
    engine.refresh()
    assert len(reads) == 2
    assert "version2" in engine.chunks[0].text
    pdf.unlink()
    assert engine.retrieve("reembolso") == []
    assert engine.files == []


def test_metrics_do_not_claim_unreviewed_accuracy(tmp_path):
    path = tmp_path / "results.jsonl"
    records = [
        {"answered": True, "correct": True},
        {"answered": False, "correct": False},
        {"answered": True, "correct": None},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records))
    report = rag.evaluation_report(path)
    assert report["accuracy_reviewed"] == 0.5
    assert report["precision_answered_reviewed"] == 1
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["target_80_met"] is None


def test_split_preserves_text_and_limits_chunk_size():
    text = " ".join(f"palavra{i}" for i in range(600))
    chunks = rag.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert set(rag.words(text)) == set(rag.words(" ".join(chunks)))


def test_clean_pages_removes_repeated_header_and_joins_split_words():
    bodies = ["Abertura do tema", "Contexto legal", "Discussão central", "Conclusões finais"]
    pages = [f"Revista X, vol.01________{50 + n}\n{body}.\nTexto sobre forma-\nção ambiental.\n"
             f"Destacam -se as garantias; cabe realizá-\nlo logo.\nSeção {body.lower()}."
             for n, body in enumerate(bodies)]
    cleaned = rag.clean_pages(pages)
    assert all("Revista" not in text for text in cleaned)
    assert "formação" in cleaned[0]
    assert "Destacam-se" in cleaned[0]
    assert "realizálo" not in cleaned[0] and "Abertura do tema" in cleaned[0]
    # Com poucas páginas não há como saber o que é cabeçalho: nada é removido.
    assert "Revista" in rag.clean_pages(pages[:2])[0]


def make_two_pdf_engine(tmp_path, monkeypatch):
    documents = tmp_path / "documentos"
    documents.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (documents / name).write_bytes(name.encode())

    def extract(path, name):
        # a.pdf tem muito mais trechos sobre "reembolso"; b.pdf só um.
        if name == "a.pdf":
            return [rag.Chunk(name, n, f"Reembolso em {n} dias, regra {n} do manual A.") for n in range(1, 9)], []
        return [rag.Chunk(name, 1, "Reembolso no manual B depende de nota fiscal.")], []

    class Encoder:
        def embed(self, texts):
            return iter(np.asarray([[1, 0, 0] for _ in texts], dtype=np.float32))

    monkeypatch.setattr(rag, "read_pdf", extract)
    monkeypatch.setattr(rag.PdfRag, "_encoder", lambda self: Encoder())
    return rag.PdfRag(documents, tmp_path / "cache")


def test_two_pdfs_together_each_get_context_interleaved(tmp_path, monkeypatch):
    engine = make_two_pdf_engine(tmp_path, monkeypatch)
    hits = engine.retrieve("Qual é a regra de reembolso?", pdfs=["a.pdf", "b.pdf"])
    files = [h.chunk.file for h in hits]
    assert files[:2] == ["a.pdf", "b.pdf"]  # o segundo PDF não fica sem espaço
    assert files.count("a.pdf") == 4 and files.count("b.pdf") == 1
    assert {h.chunk.file for h in engine.retrieve("reembolso", pdfs=["b.pdf"])} == {"b.pdf"}
    assert [h.chunk.file for h in engine.retrieve("reembolso", pdfs=["b.pdf", "inexistente.pdf"])] == ["b.pdf"]


def test_ask_sends_file_names_so_answer_can_tell_documents_apart(monkeypatch):
    from workshop_agentes import llm

    sent = []

    def fake(model, messages, **kw):
        sent.append(messages)
        return AIMessage(content=json.dumps({"claims": []})), 1

    monkeypatch.setattr(llm, "invoke_with_retry", fake)
    hits = [hit(), rag.Hit(rag.Chunk("outro.pdf", 1, TEXT), .8)]
    rag.ask("Compare os prazos", hits, SimpleNamespace(bind=lambda **kw: None))
    passages = json.loads(sent[0][1].content)["passages"]
    assert [p["file"] for p in passages] == ["manual.pdf", "outro.pdf"]
    assert "mais de um" in sent[0][0].content


def test_evaluation_resumes_after_interruption(tmp_path, monkeypatch):
    records = [{"question": f"Pergunta {n}?", "expected_answer": "x", "pdfs": ["a.pdf"]} for n in range(3)]
    calls = []

    def fake_ask(question, hits, model):
        calls.append(question)
        if len(calls) == 2:
            raise RuntimeError("429")
        return {"answer": "ok", "answered": True, "sources": [], "indicators": {}}

    monkeypatch.setattr(rag, "ask", fake_ask)
    engine = SimpleNamespace(retrieve=lambda question, pdfs=None: [])
    output = tmp_path / "saida.jsonl"
    with pytest.raises(RuntimeError):
        rag.run_evaluation(records, output, engine, None)
    assert len(rag.read_jsonl(output)) == 1
    assert rag.run_evaluation(records, output, engine, None) == 2
    saved = rag.read_jsonl(output)
    assert [r["question"] for r in saved] == [r["question"] for r in records]
    assert saved[0]["pdfs"] == ["a.pdf"]
    with pytest.raises(ValueError):
        rag.run_evaluation([{"question": "Outra?", "expected_answer": "x"}], output, engine, None)


def test_retrieval_report_measures_search_without_api():
    pages = {"q1": [3, 1], "q2": [2]}
    engine = SimpleNamespace(retrieve=lambda question, pdfs=None: [
        rag.Hit(rag.Chunk("a.pdf", p, TEXT), .8) for p in pages[question]])
    report = rag.retrieval_report([{"question": "q1", "pages": [1]}, {"question": "q2", "pages": [9]}], engine)
    assert report["hit_rate"] == 0.5
    assert report["first_place_rate"] == 0
    assert report["mrr"] == 0.25


def test_query_stop_words_do_not_make_every_chunk_a_lexical_match():
    assert not {"o", "segundo", "documento", "qual"} & {
        w for w in rag.words("Qual é o prazo segundo o documento?") if w not in rag.QUERY_STOP_WORDS}


def test_quote_cited_with_neighbor_id_is_fixed_only_within_same_pdf():
    neighbor = rag.Hit(rag.Chunk("manual.pdf", 2, "reembolso é de 30 dias após a compra. Outra regra."), .8)
    data = payload()
    data["claims"][0]["evidence"][0]["id"] = 2  # a frase completa está no trecho 1
    claims = rag.validate_claims(data, [hit(), neighbor])
    assert claims[0]["evidence"][0]["id"] == 1
    assert data["claims"][0]["evidence"][0]["id"] == 2  # não altera o JSON original
    other_pdf = rag.Hit(rag.Chunk("outro.pdf", 1, "Texto sem a frase citada."), .8)
    data["claims"][0]["evidence"][0]["id"] = 2
    assert rag.validate_claims(data, [hit(), other_pdf]) == []
