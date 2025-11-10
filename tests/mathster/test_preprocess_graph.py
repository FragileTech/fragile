import json
from pathlib import Path

import networkx as nx

from mathster.relationships.preprocess_graph import build_label_reference_graph


def _write_json(path: Path, payload: list[dict]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_label_reference_graph(tmp_path) -> None:
    preprocess_dir = tmp_path / "preprocess"
    preprocess_dir.mkdir()

    _write_json(
        preprocess_dir / "definition.json",
        [
            {
                "label": "def-alpha",
                "type": "definition",
                "title": "Alpha Definition",
                "tags": ["core"],
                "references": ["lem-beta", {"label": "thm-gamma"}],
                "related_refs": ["cor-eps"],
                "document_id": "doc-alpha",
                "section": "Section 1",
            },
            {
                "label": "def-delta",
                "type": "definition",
                "title": "Delta Definition",
                "tags": [],
                "references": [],
                "document_id": "doc-delta",
            },
        ],
    )

    _write_json(
        preprocess_dir / "lemma.json",
        [
            {
                "label": "lem-beta",
                "type": "lemma",
                "title": "Beta Lemma",
                "references": [],
                "registry_context": {"document_id": "doc-beta"},
            }
        ],
    )

    _write_json(
        preprocess_dir / "theorem.json",
        [
            {
                "label": "thm-gamma",
                "type": "theorem",
                "title": "Gamma Theorem",
                "tags": ["analysis"],
                "document": "doc-gamma",
                "proof": {
                    "references": ["lem-beta"],
                    "steps": [
                        {"references": ["def-delta"]},
                    ],
                },
            }
        ],
    )

    graph = build_label_reference_graph(preprocess_dir)
    assert isinstance(graph, nx.DiGraph)

    # Nodes exist for both known and inferred labels
    assert graph.nodes["def-alpha"]["entity_type"] == "definition"
    assert graph.nodes["def-alpha"]["document_id"] == "doc-alpha"
    assert graph.nodes["def-alpha"]["tags"] == ["core"]

    assert graph.nodes["lem-beta"]["entity_type"] == "lemma"
    assert graph.nodes["lem-beta"]["document_id"] == "doc-beta"

    assert graph.nodes["cor-eps"]["entity_type"] == "unknown"
    assert graph.nodes["cor-eps"]["tags"] == []

    # Edge metadata captures contexts and weights
    edge_alpha_beta = graph.get_edge_data("def-alpha", "lem-beta")
    assert edge_alpha_beta["weight"] == 1
    assert edge_alpha_beta["contexts"] == ["references"]
    assert edge_alpha_beta["context_counts"] == [("references", 1)]

    edge_alpha_gamma = graph.get_edge_data("def-alpha", "thm-gamma")
    assert edge_alpha_gamma["contexts"] == ["references"]

    edge_alpha_cor = graph.get_edge_data("def-alpha", "cor-eps")
    assert edge_alpha_cor["contexts"] == ["related_refs"]

    edge_gamma_beta = graph.get_edge_data("thm-gamma", "lem-beta")
    assert edge_gamma_beta["contexts"] == ["proof.references"]

    edge_gamma_delta = graph.get_edge_data("thm-gamma", "def-delta")
    assert edge_gamma_delta["contexts"] == ["proof.steps[0].references"]
