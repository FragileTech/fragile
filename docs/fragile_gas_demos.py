"""Insert the Volume II lecture experiments at their reviewed section placements."""

from html import escape
import json
from pathlib import Path
import posixpath
import re

from markdown_it import MarkdownIt
from mdit_py_plugins.colon_fence import colon_fence_plugin
from sphinx.errors import ExtensionError


def block_tokens(lines):
    """Parse structural blocks, keeping code and MyST directives opaque."""
    return MarkdownIt("commonmark").use(colon_fence_plugin).parse("\n".join(lines))


def section_end(lines, target):
    """Locate a unique source heading/label and the end of its section."""
    tokens = block_tokens(lines)
    headings = {
        token.map[0]: int(token.tag[1:])
        for token in tokens
        if token.type == "heading_open" and token.level == 0
    }
    opaque = {
        index
        for token in tokens
        if token.type in {"fence", "colon_fence", "code_block", "html_block"}
        for index in range(*token.map)
    }
    matches = [
        i
        for i, line in enumerate(lines)
        if i not in opaque
        and (
            line.strip() == f"({target})="
            if target.startswith("sec-")
            else i in headings and target in line
        )
    ]
    if len(matches) != 1:
        raise ExtensionError(f"Lecture placement must match once: {target}: {matches}")
    start = matches[0]
    if target.startswith("sec-"):
        start = next(
            (i for i in headings if i > start),
            len(lines),
        )
    if start == len(lines):
        raise ExtensionError(f"Lecture placement has no heading: {target}")
    level = headings[start]
    for i, heading_level in headings.items():
        if i > start and heading_level <= level:
            end = i
            while end > start + 1 and (
                not lines[end - 1].strip() or re.fullmatch(r"\([^)]+\)=", lines[end - 1].strip())
            ):
                end -= 1
            return end
    return len(lines)


def add_demos(app, docname, source):
    if app.builder.format != "html":
        return
    manifest_path = Path(app.confdir) / "_static_theory/gas-demos/manifest.json"
    if not manifest_path.exists():
        message = "Build lecture assets with npm run build:euclidean-lectures"
        raise ExtensionError(message)
    entries = [
        entry for entry in json.loads(manifest_path.read_text()) if entry["chapter"] == docname
    ]
    if not entries:
        return
    app.env.note_dependency(str(manifest_path))
    lines = source[0].splitlines()
    insertions = []
    for entry in entries:
        demo_id = entry["id"]
        poster = posixpath.relpath(f"_static/gas-demos/{demo_id}.svg", posixpath.dirname(docname))
        lab = (
            posixpath.relpath(
                "euclidean-gas/lecture.html",
                posixpath.dirname("docs/theory/" + docname),
            )
            + "?demo="
            + demo_id
        )
        html = (
            f'<figure class="gas-demo feynman-added" id="gas-demo-{demo_id}">'
            f'<div class="gas-demo-heading"><span>INTERACTIVE EXPERIMENT · {demo_id}</span>'
            f'<p class="gas-demo-title">{escape(entry["title"])}</p></div>'
            f'<div class="gas-demo-poster"><img src="{escape(poster)}" loading="lazy" '
            f'alt="{escape(entry["posterAlt"])}" width="960" height="510"></div>'
            f"<figcaption><strong>{escape(entry['question'])}</strong> "
            f"{escape(entry['prediction'])}</figcaption>"
            f'<p class="gas-demo-actions"><button type="button" data-gas-url="{escape(lab)}">'
            f'Load experiment</button> <a href="{escape(lab)}" target="_blank" rel="noopener">'
            f"Open full view ↗</a></p>"
            f'<p class="gas-demo-note">{escape(entry["kind"])} · Seed {entry["seed"]} · '
            f"Poster after {entry['posterTicks']} experiment steps. "
            f"Interactive view starts from the same seed.</p>"
            f"</figure>"
        )
        fence = chr(96) * 3
        block = "\n" + fence + "{raw} html\n" + html + "\n" + fence + "\n"
        insertions.append((section_end(lines, entry["target"]), block))
    ordered = sorted(enumerate(insertions), key=lambda item: (item[1][0], item[0]), reverse=True)
    for _, (index, block) in ordered:
        lines.insert(index, block)
    rendered = "\n".join(lines) + "\n"
    # An unclosed source directive/code fence must never swallow an experiment.
    raw_blocks = [
        token.content
        for token in block_tokens(rendered.splitlines())
        if token.type == "fence" and token.info.strip() == "{raw} html" and token.level == 0
    ]
    for entry in entries:
        marker = f'id="gas-demo-{entry["id"]}"'
        if sum(marker in block for block in raw_blocks) != 1:
            raise ExtensionError(f"Lecture placement is inside another block: {entry['id']}")
    source[0] = rendered


def setup(app):
    app.connect("source-read", add_demos)
    # Jupyter Book's generated conf.py may already contain extension assets.
    scripts = [
        item[0] if isinstance(item, (tuple, list)) else item for item in app.config.html_js_files
    ]
    styles = [
        item[0] if isinstance(item, (tuple, list)) else item for item in app.config.html_css_files
    ]
    if "gas-embeds.js" not in scripts:
        app.add_js_file("gas-embeds.js", defer="defer")
    if "gas-embeds.css" not in styles:
        app.add_css_file("gas-embeds.css")
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
