"""Behavioral checks for publication selection and migration compatibility."""

from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


pytest.importorskip("yaml")

DOCS = Path(__file__).resolve().parents[2] / "docs"
sys.path.insert(0, str(DOCS))

from book_manifest import published_documents
from collect_prf_directives import build_volume_output, extract_prf_blocks, volume_dirs
from fragile_redirects import redirect_html, write_redirects


def test_exports_follow_toc_and_exclude_unpublished_sources(tmp_path):
    volume = tmp_path / "source/2_fractal_gas"
    volume.mkdir(parents=True)
    block = ":::{{prf:lemma}} {}\n\nA statement.\n:::\n"
    for name in ("first", "second", "draft", "review"):
        (volume / f"{name}.md").write_text(block.format(name))
    (tmp_path / "intro.md").write_text("# Intro")
    (tmp_path / "_toc.yml").write_text(
        "root: intro\nchapters:\n"
        "  - file: source/2_fractal_gas/second\n"
        "  - file: source/2_fractal_gas/first\n"
    )
    output, blocks, files = build_volume_output(volume, True, True)
    assert output.index("second") < output.index("first")
    assert "draft" not in output and "review" not in output
    assert (blocks, files) == (2, 2)
    assert volume_dirs(tmp_path / "source", False) == [volume]


def test_nested_proofs_are_included_once_and_removed_from_short_export():
    source = (
        """::::{prf:theorem} Outer
:label: thm-example

The claim.

:::{prf:proof}
The derivation.
:::
::::

```python
"""
        + '""":::{prf:lemma} Fake\n:::"""'
        + "\n```\n"
    )
    full = extract_prf_blocks(source, True)
    brief = extract_prf_blocks(source, False)
    assert len(full) == len(brief) == 1
    assert full[0].count("The derivation.") == 1
    assert "The derivation." not in brief[0]
    assert "The claim." in brief[0]
    assert "Fake" not in "".join(full)


def test_missing_published_source_fails_instead_of_silently_omitting_it(tmp_path):
    (tmp_path / "_toc.yml").write_text("root: missing\n")
    with pytest.raises(ValueError, match="Missing TOC"):
        published_documents(tmp_path)


def test_backtick_directives_and_legacy_inline_proofs():
    source = "```{prf:theorem} Result\nA claim.\n\n**Proof.**\nAn argument.\n```\n"
    assert "An argument" in extract_prf_blocks(source, True)[0]
    brief = extract_prf_blocks(source, False)[0]
    assert "A claim" in brief and "An argument" not in brief
    assert brief.endswith("```")


def test_redirects_are_relative_and_preserve_request_state(tmp_path):
    src = tmp_path / "source"
    out = tmp_path / "html"
    src.mkdir()
    destination = out / "source/2_fractal_gas/chapter.html"
    destination.parent.mkdir(parents=True)
    destination.write_text("Current chapter")
    (src / "redirects.json").write_text(
        json.dumps({"source/3_fractal_gas/chapter": "source/2_fractal_gas/chapter"})
    )
    app = SimpleNamespace(
        srcdir=str(src),
        outdir=str(out),
        builder=SimpleNamespace(name="html"),
        env=SimpleNamespace(found_docs={"source/2_fractal_gas/chapter"}),
    )
    write_redirects(app, None)
    result = (out / "source/3_fractal_gas/chapter.html").read_text()
    assert "../2_fractal_gas/chapter.html" in result
    assert "window.location.search" in result and "window.location.hash" in result
    assert 'content="noindex"' in result
    mapped = redirect_html("chapter.html", {"old": "new"})
    assert '"old": "new"' in mapped
    assert "encodeURIComponent(anchor)" in mapped


def test_redirect_rejects_missing_destination(tmp_path):
    (tmp_path / "redirects.json").write_text('{"old": "missing"}')
    app = SimpleNamespace(
        srcdir=str(tmp_path),
        outdir=str(tmp_path),
        builder=SimpleNamespace(name="html"),
        env=SimpleNamespace(found_docs=set()),
    )
    with pytest.raises(ValueError, match="was not built"):
        write_redirects(app, None)


def test_built_audit_detects_formal_content_hidden_by_expert_mode(tmp_path):
    pytest.importorskip("bs4")
    from check_built_book import validate_html

    source = tmp_path / "source/2_fractal_gas/chapter.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Chapter\n")
    (tmp_path / "_toc.yml").write_text("root: source/2_fractal_gas/chapter\n")
    (tmp_path / "redirects.json").write_text("{}")
    output = tmp_path / "html"
    page = output / "source/2_fractal_gas/chapter.html"
    page.parent.mkdir(parents=True)
    (output / "searchindex.js").write_text("Search.setIndex({});")
    page.write_text(
        '<article><div class="feynman-prose"><div class="proof theorem" '
        'id="result">A theorem.</div></div></article>'
    )
    errors = validate_html(tmp_path, output)
    assert len(errors) == 1 and "hidden in Expert Mode" in errors[0]
    page.write_text('<article><div class="proof" id="result">A theorem.</div></article>')
    assert validate_html(tmp_path, output) == []


def test_labelled_proof_builds_with_cross_document_reference(tmp_path):
    pytest.importorskip("sphinx_proof")
    Sphinx = pytest.importorskip("sphinx.application").Sphinx

    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text(
        "extensions = ['sphinx_proof', 'fragile_proofs']\n"
        "master_doc = 'index'\nhtml_theme = 'basic'\n"
    )
    (source / "index.rst").write_text(
        "Book\n====\n\nSee :prf:ref:`stable-proof`.\n\n.. toctree::\n\n   argument\n"
    )
    (source / "argument.rst").write_text(
        "Argument\n========\n\n.. prf:proof:: A named argument\n   :label: stable-proof\n\n"
        "   The complete argument.\n\n.. prf:proof::\n\n   Another argument.\n"
    )
    warning = StringIO()
    output = tmp_path / "html"
    app = Sphinx(
        str(source),
        str(source),
        str(output),
        str(tmp_path / "doctrees"),
        "html",
        status=StringIO(),
        warning=warning,
        freshenv=True,
    )
    app.build(force_all=True)
    assert app.statuscode == 0, warning.getvalue()
    assert "not found" not in warning.getvalue()
    assert "argument.html#stable-proof" in (output / "index.html").read_text()
    rendered = (output / "argument.html").read_text()
    assert 'id="stable-proof"' in rendered and 'id="proof-0"' in rendered
    assert ":label:" not in rendered and "The complete argument." in rendered
