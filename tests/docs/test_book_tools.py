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

from assemble_docs import assemble_docs, broken_internal_links, stale_lecture_links
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


def test_theory_and_lab_have_independent_publication_trees():
    theory = {path.relative_to(DOCS).as_posix() for path in published_documents(DOCS)}
    assert not any(name.startswith("source/project/control_lab") for name in theory)

    lab_toc = json.loads(json.dumps(pytest.importorskip("yaml").safe_load(
        (DOCS / "_toc_lab.yml").read_text()
    )))
    names = [lab_toc["root"], *(chapter["file"] for chapter in lab_toc["chapters"])]
    assert names == [
        "source/project/control_laboratory",
        "source/project/control_lab_getting_started",
        "source/project/control_lab_controls",
        "source/project/control_lab_scenes",
        "source/project/control_lab_replay",
        "source/project/control_lab_experiments",
        "source/project/control_lab_architecture",
    ]


def test_documentation_assembly_scopes_sites_and_preserves_old_urls(tmp_path):
    theory = tmp_path / "theory"
    lab = tmp_path / "lab"
    portal = tmp_path / "docs/portal"
    for site in (theory, lab):
        site.mkdir(parents=True)
        (site / "index.html").write_text("redirect")
        (site / "searchindex.js").write_text("Search.setIndex({});")
    theory_page = theory / "source/1_agent/chapter.html"
    theory_page.parent.mkdir(parents=True)
    theory_page.write_text('<a href="next.html">Next</a>')
    (theory_page.parent / "next.html").write_text("Next chapter")
    lab_page = lab / "source/project/control_laboratory.html"
    lab_page.parent.mkdir(parents=True)
    lab_page.write_text("Lab guide")
    prompts = theory / "_static/prompts"
    prompts.mkdir(parents=True)
    (prompts / "volume.txt").write_text("Prompt")
    portal.mkdir(parents=True)
    (portal / "index.html").write_text("Documentation portal")
    (portal / "portal.css").write_text("body {}")
    (portal.parent / "logo.png").write_bytes(b"logo")
    (portal.parent / "favicon.png").write_bytes(b"favicon")

    output = tmp_path / "assembled"
    assemble_docs(theory, lab, portal, output)

    assert (output / "index.html").read_text() == "Documentation portal"
    assert (output / "theory/source/1_agent/chapter.html").is_file()
    assert (output / "lab/source/project/control_laboratory.html").is_file()
    assert (output / "_static/prompts/volume.txt").read_text() == "Prompt"
    theory_redirect = (output / "source/1_agent/chapter.html").read_text()
    lab_redirect = (output / "source/project/control_laboratory.html").read_text()
    assert "../../theory/source/1_agent/chapter.html" in theory_redirect
    assert "../../lab/source/project/control_laboratory.html" in lab_redirect
    assert "window.location.search" in theory_redirect
    assert "window.location.hash" in lab_redirect


def test_documentation_assembly_rejects_stale_root_links(tmp_path):
    site = tmp_path / "theory"
    site.mkdir()
    page = site / "chapter.html"
    page.write_text('<a href="/docs/source/1_agent/chapter.html">Old URL</a>')
    assert stale_lecture_links(site) == ["chapter.html"]
    page.write_text('<a href="/docs/theory/source/1_agent/chapter.html">Current URL</a>')
    assert stale_lecture_links(site) == []


def test_documentation_bundle_crawler_reports_missing_local_targets(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        '<a href="chapter.html#result">Chapter</a><a href="https://example.com/">External</a>'
    )
    assert broken_internal_links(site) == ["index.html: chapter.html#result"]
    (site / "chapter.html").write_text("Result")
    assert broken_internal_links(site) == []


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
