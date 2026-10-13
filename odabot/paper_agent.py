import io
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import requests


ARXIV_ID = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
MAX_PAPER_CHARACTERS = 100_000


def extract_paper_text(source):
    source_path = Path(source).expanduser()
    if source_path.is_file():
        pdf_data = source_path.read_bytes()
    else:
        parsed = urlparse(source)
        arxiv_id = ARXIV_ID.match(source)
        if (parsed.scheme or parsed.netloc) and not arxiv_id:
            if parsed.scheme != "https" or parsed.hostname not in {
                "arxiv.org",
                "www.arxiv.org",
            }:
                raise ValueError("Paper URLs must be hosted on https://arxiv.org")
            match = re.fullmatch(
                r"/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?",
                parsed.path,
            )
            if not match:
                raise ValueError("Expected an arXiv paper URL")
            arxiv_id = match

        if arxiv_id:
            paper_id = arxiv_id.group(1)
            response = requests.get(
                f"https://arxiv.org/pdf/{paper_id}", timeout=30
            )
            response.raise_for_status()
            pdf_data = response.content
        else:
            raise FileNotFoundError(f"Paper PDF not found: {source}")

    if not pdf_data.startswith(b"%PDF"):
        raise ValueError("Paper source is not a PDF file")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires installing oda-bot with the [paper-agent] extra"
        ) from exc

    reader = PdfReader(io.BytesIO(pdf_data))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        raise ValueError("No extractable text found in the paper PDF")
    if len(text) > MAX_PAPER_CHARACTERS:
        return text[:MAX_PAPER_CHARACTERS] + "\n\n[Paper text truncated]"
    return text


def validate_notebook(notebook):
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4:
        raise ValueError("LLM response must be a Jupyter notebook in nbformat 4")
    if not isinstance(notebook.get("metadata"), dict) or not isinstance(
        notebook.get("nbformat_minor"), int
    ):
        raise ValueError("Notebook metadata and nbformat_minor are required")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("Generated notebook must contain at least one cell")

    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") not in {
            "markdown",
            "code",
        }:
            raise ValueError("Notebook cells must be markdown or code cells")
        if not isinstance(cell.get("metadata"), dict):
            raise ValueError("Notebook cell metadata must be an object")
        source = cell.get("source")
        if not isinstance(source, (str, list)):
            raise ValueError("Each notebook cell must have a source")
        if isinstance(source, list) and not all(
            isinstance(line, str) for line in source
        ):
            raise ValueError("Notebook cell source lines must be strings")
        if cell["cell_type"] == "code":
            if (
                not isinstance(cell.get("outputs"), list)
                or cell["outputs"]
                or cell.get("execution_count") is not None
            ):
                raise ValueError("Generated code cells must not contain executed output")


class PaperToWorkflowAgent:
    def __init__(self, model, api_url, api_key=None):
        self.model = model
        self.api_url = api_url.rstrip("/")
        endpoint = urlparse(self.api_url)
        if endpoint.scheme not in {"https", "http"} or not endpoint.hostname:
            raise ValueError("The LLM API URL must use HTTPS")
        if endpoint.scheme == "http" and endpoint.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("HTTP is only allowed for a local LLM API")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def generate(self, paper_source, output_path, force=False):
        output_path = Path(output_path)
        if output_path.exists() and not force:
            raise FileExistsError(
                f"{output_path} already exists; pass --force to overwrite it"
            )
        if not self.api_key:
            raise RuntimeError("Set OPENAI_API_KEY before generating a notebook")

        paper_text = extract_paper_text(paper_source)
        response = requests.post(
            f"{self.api_url}/chat/completions",
            headers={"Authorization": "Bearer " + self.api_key},
            json={
                "model": self.model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Create a reproducible Jupyter notebook that implements "
                            "the methods in the supplied research paper as an ODA "
                            "workflow. Return only a JSON nbformat 4 notebook object "
                            "with cells, metadata, nbformat, and nbformat_minor. "
                            "Use markdown cells to cite the paper and explain steps. "
                            "Put workflow parameters in clearly named top-level "
                            "Python assignments with comments. Use available ODA "
                            "services where appropriate, and do not invent data, "
                            "results, or APIs. State uncertainties in markdown. "
                            "Leave every code cell unexecuted with empty outputs."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Generate a notebook to reproduce this paper:\n\n"
                            + paper_text
                        ),
                    },
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        try:
            notebook = json.loads(
                response.json()["choices"][0]["message"]["content"]
            )
            validate_notebook(notebook)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("LLM returned an invalid notebook response") from exc

        with output_path.open("w" if force else "x", encoding="utf-8") as output:
            output.write(json.dumps(notebook, indent=2) + "\n")
        return output_path
