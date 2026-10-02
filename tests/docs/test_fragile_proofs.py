"""Render labelled proof references and the installed Sphinx proof index."""

from io import StringIO
from pathlib import Path

from sphinx.application import Sphinx


DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_labelled_proofs_render_references_and_domain_index(tmp_path, monkeypatch):
    """Exercise metadata consumers through an actual HTML build."""
    monkeypatch.syspath_prepend(str(DOCS))
    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text(
        "extensions = ['sphinx_proof', 'fragile_proofs']\n"
        "master_doc = 'index'\n"
        "html_theme = 'basic'\n"
        "html_domain_indices = True\n"
    )
    (source / "index.rst").write_text(
        "Proof compatibility\n"
        "===================\n\n"
        ".. prf:theorem:: Identity\n"
        "   :label: thm-smoke\n\n"
        "   The identity holds.\n\n"
        ".. prf:proof:: Detailed argument\n"
        "   :label: proof-smoke\n"
        "   :class: retained-class\n\n"
        "   This is the labelled proof body.\n\n"
        ".. prf:proof::\n"
        "   :name: proof-alias\n\n"
        "   This proof uses the existing name alias.\n\n"
        ".. prf:proof::\n\n"
        "   This proof remains unlabelled.\n\n"
        "Proof-domain links: :prf:ref:`proof-smoke`, :prf:ref:`proof-alias`.\n\n"
        "Standard link: :ref:`proof-smoke`.\n"
    )
    output = tmp_path / "html"
    warnings = StringIO()
    app = Sphinx(
        srcdir=source,
        confdir=source,
        outdir=output,
        doctreedir=tmp_path / "doctrees",
        buildername="html",
        status=StringIO(),
        warning=warnings,
        freshenv=True,
        warningiserror=True,
    )
    app.build(force_all=True)

    assert app.statuscode == 0, warnings.getvalue()
    assert not warnings.getvalue()
    for label in ("proof-smoke", "proof-alias"):
        entry = app.env.proof_list[label]
        assert entry["type"] == "proof"
        assert entry["nonumber"] is True
        assert entry["ids"] == [label]
    assert set(app.env.proof_list) == {"thm-smoke", "proof-smoke", "proof-alias"}

    html = (output / "index.html").read_text()
    assert 'id="proof-smoke"' in html
    assert 'href="#proof-smoke"' in html
    assert 'href="#proof-alias"' in html
    assert "retained-class" in html
    assert "Detailed argument" in html
    assert "This proof remains unlabelled." in html
    proof_index = (output / "prf-prf.html").read_text()
    assert "proof-smoke" in proof_index
    assert "proof-alias" in proof_index
