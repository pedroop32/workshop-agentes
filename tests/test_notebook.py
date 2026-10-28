"""Estrutura do workshop.ipynb: o roteiro na ordem, células de código válidas e o ambiente recolhido."""

import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "workshop.ipynb"
SECTIONS = [
    "## 1. Preparar o ambiente",
    "## 2. Testar a conexão",
    "## 3. Montar o agente",
    "## 4. Executar o agente",
    "## 5. O agente tem a ferramenta certa",
    "## 6. RAG",
    "## 7. Bastidores",
    "## 8. Avaliação",
    "## 9. Limitações",
]


def cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def source(cell: dict) -> str:
    return "".join(cell["source"])


def test_sections_follow_the_workshop_order():
    headings = [line for c in cells() if c["cell_type"] == "markdown" for line in source(c).splitlines()
                if line.startswith("## ")]
    assert len(headings) == len(SECTIONS)
    assert all(h.startswith(s) for h, s in zip(headings, SECTIONS))


def test_removed_extras_are_gone():
    text = "\n".join(source(c) for c in cells())
    for gone in ("Métricas do Groq", "Limites da conta", "OUTRO_MODELO", "x-ratelimit", "### E"):
        assert gone not in text


def test_code_cells_compile_and_have_no_outputs():
    for c in cells():
        if c["cell_type"] == "code":
            ast.parse(source(c))
            assert c["outputs"] == [] and c["execution_count"] is None


def test_setup_cell_is_a_hidden_colab_form_that_installs_the_rag():
    setup = next(c for c in cells() if c["cell_type"] == "code")
    assert setup["metadata"].get("cellView") == "form"
    assert source(setup).startswith('# @title Preparar o ambiente { display-mode: "form" }')
    assert "requirements-rag.txt" in source(setup)


def test_the_example_pdf_used_by_the_notebook_is_in_the_repository():
    assert (NOTEBOOK.parent / "documentos" / "PPC2024BCC.pdf").is_file()
