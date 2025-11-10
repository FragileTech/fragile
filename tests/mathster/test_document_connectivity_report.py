import importlib.util
import json
from pathlib import Path
import sys

MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "mathster" / "reports" / "document_connectivity_report.py"
spec = importlib.util.spec_from_file_location(
    "mathster.reports.document_connectivity_report",
    MODULE_PATH,
)
dcr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dcr
assert spec.loader is not None
spec.loader.exec_module(dcr)


def _write_preprocess_file(path: Path, items: list[dict]) -> None:
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _write_directives_file(path: Path, directive_type: str, items: list[dict]) -> None:
    payload = {
        "stage": "directives",
        "directive_type": directive_type,
        "items": items,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_connectivity_graph_with_preprocess(tmp_path) -> None:
    preprocess_dir = tmp_path / "preprocess"
    preprocess_dir.mkdir()

    _write_preprocess_file(
        preprocess_dir / "definition.json",
        [
            {
                "label": "def-a",
                "type": "definition",
                "title": "Definition A",
                "references": ["lem-b"],
                "document_id": "doc-a",
                "tags": ["core"],
            }
        ],
    )

    _write_preprocess_file(
        preprocess_dir / "lemma.json",
        [
            {
                "label": "lem-b",
                "type": "lemma",
                "title": "Lemma B",
                "document_id": "doc-a",
            }
        ],
    )

    graph = dcr.build_document_connectivity_graph("doc-a", preprocess_dir=preprocess_dir)
    assert graph.nodes["def-a"]["entity_type"] == "definition"
    assert graph.nodes["lem-b"]["entity_type"] == "lemma"
    assert graph.nodes["def-a"]["tags"] == ["core"]
    assert graph.has_edge("def-a", "lem-b")


def test_build_connectivity_graph_fallbacks_to_directives(tmp_path, monkeypatch) -> None:
    doc_id = "doc-fallback"
    directives_dir = tmp_path / "docs" / doc_id / "registry" / "directives"
    directives_dir.mkdir(parents=True)

    _write_directives_file(
        directives_dir / "definition.json",
        "definition",
        [
            {
                "label": "def-x",
                "directive_type": "definition",
                "title": "Definition X",
                "references": ["lem-y"],
                "_registry_context": {"document_id": doc_id},
            }
        ],
    )

    _write_directives_file(
        directives_dir / "lemma.json",
        "lemma",
        [
            {
                "label": "lem-y",
                "directive_type": "lemma",
                "title": "Lemma Y",
                "_registry_context": {"document_id": doc_id},
            }
        ],
    )

    def fake_discover(roots=None, subfolder=None, document=None):
        if subfolder == "directives":
            return [directives_dir]
        return []

    monkeypatch.setattr(dcr, "discover_registry_folders", fake_discover)
    monkeypatch.setattr(dcr, "UNIFIED_PREPROCESS_PATH", tmp_path / "missing_preprocess")
    monkeypatch.setattr(dcr, "UNIFIED_DIRECTIVES_PATH", directives_dir)

    graph = dcr.build_document_connectivity_graph(doc_id)
    assert graph.nodes["def-x"]["entity_type"] == "definition"
    assert graph.nodes["def-x"]["document_id"] == doc_id
    assert graph.nodes["lem-y"]["entity_type"] == "lemma"
    assert graph.has_edge("def-x", "lem-y")
