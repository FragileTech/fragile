import json
from pathlib import Path

import networkx as nx

from mathster.relationships.directives_graph import build_label_reference_graph


def _write_directive_file(path: Path, directive_type: str, items: list[dict]) -> None:
    payload = {
        "stage": "directives",
        "directive_type": directive_type,
        "items": items,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_label_reference_graph(tmp_path) -> None:
    directives_dir = tmp_path / "registry" / "directives"
    directives_dir.mkdir(parents=True)

    _write_directive_file(
        directives_dir / "definition.json",
        "definition",
        [
            {
                "label": "def-alpha",
                "directive_type": "definition",
                "title": "Alpha Definition",
                "section": "Section 1",
                "references": [
                    "lem-beta",
                    {"label": "thm-gamma"},
                    "cor-zeta",
                ],
                "start_line": 10,
                "end_line": 20,
                "_registry_context": {
                    "document_id": "doc-alpha",
                    "chapter_index": 2,
                    "chapter_file": "chapter_2.json",
                },
            }
        ],
    )

    _write_directive_file(
        directives_dir / "lemma.json",
        "lemma",
        [
            {
                "label": "lem-beta",
                "directive_type": "lemma",
                "title": "Lemma Beta",
                "references": [],
                "_registry_context": {"document_id": "doc-beta"},
            }
        ],
    )

    _write_directive_file(
        directives_dir / "theorem.json",
        "theorem",
        [
            {
                "label": "thm-gamma",
                "directive_type": "theorem",
                "title": "Theorem Gamma",
                "references": ["def-alpha"],
                "_registry_context": {"document_id": "doc-gamma"},
            }
        ],
    )

    graph = build_label_reference_graph(directives_dir)
    assert isinstance(graph, nx.DiGraph)

    # Node metadata stems from directive payload.
    assert graph.nodes["def-alpha"]["directive_type"] == "definition"
    assert graph.nodes["def-alpha"]["document_id"] == "doc-alpha"
    assert graph.nodes["def-alpha"]["chapter_index"] == 2

    assert graph.nodes["lem-beta"]["directive_type"] == "lemma"
    assert graph.nodes["lem-beta"]["document_id"] == "doc-beta"

    # Referenced labels missing from registry become placeholder nodes.
    assert graph.nodes["cor-zeta"]["directive_type"] == "unknown"
    assert graph.nodes["cor-zeta"]["document_id"] is None

    edge_alpha_beta = graph.get_edge_data("def-alpha", "lem-beta")
    assert edge_alpha_beta["weight"] == 1
    assert edge_alpha_beta["contexts"] == ["references"]

    edge_alpha_gamma = graph.get_edge_data("def-alpha", "thm-gamma")
    assert edge_alpha_gamma["contexts"] == ["references"]

    edge_alpha_cor = graph.get_edge_data("def-alpha", "cor-zeta")
    assert edge_alpha_cor["contexts"] == ["references"]

    edge_gamma_alpha = graph.get_edge_data("thm-gamma", "def-alpha")
    assert edge_gamma_alpha["weight"] == 1
