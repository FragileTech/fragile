"""Check lecture placement against the actual Volume II source and generated manifest."""

from collections import defaultdict
from html.parser import HTMLParser
import importlib.util
import json
from operator import itemgetter
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import urljoin, urlparse

from markdown_it import MarkdownIt
from mdit_py_plugins.colon_fence import colon_fence_plugin
import pytest
from sphinx.errors import ExtensionError


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SPEC = importlib.util.spec_from_file_location("fragile_gas_demos", DOCS / "fragile_gas_demos.py")
EXTENSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTENSION)
MANIFEST = DOCS / "_static_theory/gas-demos/manifest.json"
ENTRIES = json.loads(MANIFEST.read_text())
CHAPTERS = sorted({entry["chapter"] for entry in ENTRIES})


def app(confdir=DOCS, output_format="html"):
    """Small Sphinx surface with observable dependency tracking."""
    dependencies = []
    return SimpleNamespace(
        confdir=confdir,
        builder=SimpleNamespace(format=output_format),
        env=SimpleNamespace(note_dependency=dependencies.append),
        dependencies=dependencies,
    )


def parse(text):
    return MarkdownIt("commonmark").use(colon_fence_plugin).parse(text)


class FigureHTML(HTMLParser):
    """Collect HTML attributes and ensure every generated tag balances."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = defaultdict(list)
        self.stack = []

    def handle_starttag(self, tag, attributes):
        self.tags[tag].append(dict(attributes))
        if tag not in {"img", "br", "hr", "input", "meta", "link"}:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack.pop() == tag


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_every_demo_is_a_top_level_raw_block_with_working_published_paths(chapter):
    original = (DOCS / f"{chapter}.md").read_text()
    source = [original]
    host = app()
    EXTENSION.add_demos(host, chapter, source)
    expected = [entry for entry in ENTRIES if entry["chapter"] == chapter]
    tokens = parse(source[0])
    raw = [
        token
        for token in tokens
        if token.type == "fence"
        and token.info == "{raw} html"
        and 'class="gas-demo ' in token.content
    ]
    assert len(raw) == len(expected)
    assert all(token.level == 0 for token in raw)
    assert host.dependencies == [str(MANIFEST)]
    # Original theorem/proof directives, labels, code samples, and headings are untouched.
    strip_demo = re.sub(
        r'\n```\{raw\} html\n<figure class="gas-demo .*?</figure>\n```\n\n',
        "",
        source[0],
        flags=re.DOTALL,
    )
    assert strip_demo.rstrip() == original.rstrip()
    original_formal = [
        (token.type, token.info, token.content)
        for token in parse(original)
        if token.type in {"colon_fence", "fence"}
    ]
    actual_formal = [
        (token.type, token.info, token.content)
        for token in tokens
        if token.type in {"colon_fence", "fence"} and token not in raw
    ]
    assert actual_formal == original_formal
    labels = re.findall(r"^\([^)]+\)=$", original, re.MULTILINE)
    assert re.findall(r"^\([^)]+\)=$", source[0], re.MULTILINE) == labels
    base = f"https://fragile.tech/docs/theory/{chapter}.html"
    for entry in expected:
        block = next(token for token in raw if f'id="gas-demo-{entry["id"]}"' in token.content)
        html = FigureHTML()
        html.feed(block.content)
        assert html.stack == []
        assert html.tags["figure"][0]["id"] == f"gas-demo-{entry['id']}"
        link = html.tags["a"][0]["href"]
        assert html.tags["button"][0]["data-gas-url"] == link
        assert urljoin(base, link) == (
            f"https://fragile.tech/euclidean-gas/lecture.html?demo={entry['id']}"
        )
        image = html.tags["img"][0]
        assert urlparse(urljoin(base, image["src"])).path == (
            f"/docs/theory/_static/gas-demos/{entry['id']}.svg"
        )
        assert image["alt"] == entry["posterAlt"]
        assert (DOCS / f"_static_theory/gas-demos/{entry['id']}.svg").is_file()


def test_manifest_covers_all_42_unique_demos_in_21_chapters():
    assert len(ENTRIES) == len({entry["id"] for entry in ENTRIES}) == 42
    assert len(CHAPTERS) == 21
    assert {entry["part"] for entry in ENTRIES} == {"I", "II", "III", "IV"}


@pytest.mark.parametrize("entry", ENTRIES, ids=itemgetter("id"))
def test_each_reviewed_placement_is_outside_a_formal_directive_or_code_fence(entry):
    lines = (DOCS / f"{entry['chapter']}.md").read_text().splitlines()
    index = EXTENSION.section_end(lines, entry["target"])
    for token in parse("\n".join(lines)):
        if token.type in {"fence", "colon_fence", "code_block", "html_block"}:
            assert not token.map[0] < index < token.map[1], entry["id"]


def test_heading_like_code_and_nested_proof_headings_do_not_end_the_section():
    text = """# Chapter
(sec-target)=
## Target
```python
# Pseudocode heading
```
::::{prf:proof}
## Heading inside a proof
:::{note}
### Nested heading
:::
::::

(sec-next)=
(sec-next-alias)=
## Next
"""
    lines = text.splitlines()
    index = EXTENSION.section_end(lines, "sec-target")
    assert lines[index - 1] == "::::"
    assert lines[index:].index("(sec-next)=") < lines[index:].index("## Next")
    assert EXTENSION.section_end(lines, "Target") == index


def test_real_pseudocode_fixture_stays_intact():
    entry = next(entry for entry in ENTRIES if entry["id"] == "I-01")
    lines = (DOCS / f"{entry['chapter']}.md").read_text().splitlines()
    index = EXTENSION.section_end(lines, entry["target"])
    assert index > lines.index("# Pseudocode: one Fractal Gas step (high level)")
    assert lines[index - 1] == ":::"  # The overview figure also finishes first.


def test_missing_or_ambiguous_target_fails_loudly():
    with pytest.raises(ExtensionError, match="match once"):
        EXTENSION.section_end(["# One"], "missing")
    with pytest.raises(ExtensionError, match="match once"):
        EXTENSION.section_end(["# One", "# One"], "One")
    with pytest.raises(ExtensionError, match="no heading"):
        EXTENSION.section_end(["(sec-empty)="], "sec-empty")


def test_non_html_build_does_not_require_assets_or_modify_source(tmp_path):
    source = ["# Original"]
    EXTENSION.add_demos(app(tmp_path, "latex"), CHAPTERS[0], source)
    assert source == ["# Original"]


def test_html_build_requires_generated_assets(tmp_path):
    with pytest.raises(ExtensionError, match="Build lecture assets"):
        EXTENSION.add_demos(app(tmp_path), CHAPTERS[0], ["# Original"])


def test_unclosed_source_fence_rejects_insertion_instead_of_swallowing_demo(tmp_path):
    directory = tmp_path / "_static_theory/gas-demos"
    directory.mkdir(parents=True)
    entry = {**ENTRIES[0], "chapter": "example", "target": "Target"}
    (directory / "manifest.json").write_text(json.dumps([entry]))
    source = ["# Target\n```python\n# Unclosed code fence\n"]
    with pytest.raises(ExtensionError, match="inside another block"):
        EXTENSION.add_demos(app(tmp_path), "example", source)
    assert source[0].startswith("# Target\n```python")


def test_shared_section_keeps_manifest_order_and_escapes_html(tmp_path):
    directory = tmp_path / "_static_theory/gas-demos"
    directory.mkdir(parents=True)
    entries = [
        {
            **ENTRIES[0],
            "id": demo_id,
            "chapter": "example",
            "target": "Target",
            "title": '<tag> & "quoted"',
            "question": "What does x < y mean?",
            "prediction": "Compare A & B.",
        }
        for demo_id in ["II-02", "II-01"]
    ]
    (directory / "manifest.json").write_text(json.dumps(entries))
    source = ["# Target\nBody.\n\n(sec-next)=\n# Next\n"]
    EXTENSION.add_demos(app(tmp_path), "example", source)
    blocks = [token for token in parse(source[0]) if token.info == "{raw} html"]
    assert len(blocks) == 2
    assert 'id="gas-demo-II-02"' in blocks[0].content
    assert 'id="gas-demo-II-01"' in blocks[1].content
    for block in blocks:
        html = FigureHTML()
        html.feed(block.content)
        assert "h3" not in html.tags
        assert "tag" not in html.tags
        assert any(tag.get("class") == "gas-demo-title" for tag in html.tags["p"])
        assert "&lt;tag&gt; &amp; &quot;quoted&quot;" in block.content
        assert "<strong>What does x &lt; y mean?</strong> Compare A &amp; B." in block.content


@pytest.mark.parametrize("normalized", [False, True])
def test_setup_does_not_register_generated_config_assets_twice(normalized):
    scripts = [("gas-embeds.js", {})] if normalized else ["gas-embeds.js"]
    styles = [("gas-embeds.css", {})] if normalized else ["gas-embeds.css"]
    calls = []
    fixture = SimpleNamespace(
        config=SimpleNamespace(html_js_files=scripts, html_css_files=styles),
        connect=lambda *_args: None,
        add_js_file=lambda *args, **_kwargs: calls.append(args),
        add_css_file=lambda *args, **_kwargs: calls.append(args),
    )
    EXTENSION.setup(fixture)
    assert calls == []
