import json
import sys
from types import SimpleNamespace

import pytest

from odabot import paper_agent


def make_notebook():
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Reproduce the paper",
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": "source_name = 'example'",
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_extract_paper_text_from_arxiv(monkeypatch):
    class Reader:
        def __init__(self, pdf):
            assert pdf.read() == b"%PDF-1.4"
            self.pages = [SimpleNamespace(extract_text=lambda: "Paper text")]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=Reader))
    monkeypatch.setattr(
        paper_agent,
        "requests",
        SimpleNamespace(
            get=lambda url, timeout: SimpleNamespace(
                content=b"%PDF-1.4", raise_for_status=lambda: None
            )
        ),
    )

    assert paper_agent.extract_paper_text("arxiv:2301.00001") == "Paper text"


def test_extract_paper_text_from_local_pdf(monkeypatch, tmp_path):
    class Reader:
        def __init__(self, pdf):
            assert pdf.read() == b"%PDF-1.4"
            self.pages = [SimpleNamespace(extract_text=lambda: "Local paper")]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=Reader))
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_bytes(b"%PDF-1.4")

    assert paper_agent.extract_paper_text(paper_path) == "Local paper"


def test_rejects_non_arxiv_urls():
    with pytest.raises(ValueError, match="arxiv.org"):
        paper_agent.extract_paper_text("https://example.com/paper.pdf")


def test_rejects_nonlocal_http_llm_endpoint():
    with pytest.raises(ValueError, match="HTTP is only allowed for a local"):
        paper_agent.PaperToWorkflowAgent("model", "http://llm.example/v1")


def test_generate_writes_unexecuted_notebook(monkeypatch, tmp_path):
    notebook = make_notebook()
    monkeypatch.setattr(paper_agent, "extract_paper_text", lambda source: "Paper text")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {"message": {"content": json.dumps(notebook)}}
                ]
            }

    def post(url, headers, json, timeout):
        assert url == "https://api.example/v1/chat/completions"
        assert headers["Authorization"] == "Bearer " + "test-key"
        assert "Paper text" in json["messages"][1]["content"]
        return Response()

    monkeypatch.setattr(paper_agent.requests, "post", post)
    output_path = tmp_path / "workflow.ipynb"
    agent = paper_agent.PaperToWorkflowAgent(
        "test-model", "https://api.example/v1", api_key="test-key"
    )

    assert agent.generate("paper.pdf", output_path) == output_path
    assert json.loads(output_path.read_text()) == notebook


def test_invalid_generated_notebook_is_not_written(monkeypatch, tmp_path):
    monkeypatch.setattr(paper_agent, "extract_paper_text", lambda source: "Paper text")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"nbformat": 3}'}}]}

    monkeypatch.setattr(
        paper_agent.requests, "post", lambda *args, **kwargs: Response()
    )
    output_path = tmp_path / "workflow.ipynb"
    agent = paper_agent.PaperToWorkflowAgent(
        "test-model", "https://api.example/v1", api_key="test-key"
    )

    with pytest.raises(ValueError, match="nbformat 4"):
        agent.generate("paper.pdf", output_path)
    assert not output_path.exists()


def test_rejects_notebook_with_executed_code():
    notebook = make_notebook()
    notebook["cells"][1]["outputs"] = [{"output_type": "stream"}]

    with pytest.raises(ValueError, match="must not contain executed output"):
        paper_agent.validate_notebook(notebook)
