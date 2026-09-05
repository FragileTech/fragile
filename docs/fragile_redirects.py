"""Emit compatibility pages for the Volume 2 source migration."""

from __future__ import annotations

import html
import json
from pathlib import Path, PurePosixPath
import posixpath


def redirect_html(target: str, fragments: dict[str, str] | None = None) -> str:
    """Generate a relative redirect that preserves queries and mapped anchors."""
    href = html.escape(target, quote=True)
    script_target = json.dumps(target).replace("<", "\\u003c")
    script_fragments = json.dumps(fragments or {}).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Page moved</title><meta name="robots" content="noindex">
<link rel="canonical" href="{href}">
<script>
const target = {script_target};
const fragments = {script_fragments};
let anchor = window.location.hash.slice(1);
try {{ anchor = decodeURIComponent(anchor); }} catch (error) {{ /* retain raw anchor */ }}
if (Object.prototype.hasOwnProperty.call(fragments, anchor)) anchor = fragments[anchor];
window.location.replace(target + window.location.search + (anchor ? '#' + encodeURIComponent(anchor) : ''));
</script></head><body><p>This chapter has moved. <a href="{href}">Open the chapter</a>.</p></body></html>
"""


def write_redirects(app, exception) -> None:
    if exception is not None or app.builder.name != "html":
        return
    source = Path(app.srcdir)
    mappings = json.loads((source / "redirects.json").read_text(encoding="utf-8"))
    anchor_file = source / "redirect_fragments.json"
    anchors = json.loads(anchor_file.read_text()) if anchor_file.exists() else {}
    output = Path(app.outdir)
    for old, new in mappings.items():
        for name in (old, new):
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError(f"Invalid redirect path: {name}")
        destination = output / (new + ".html")
        if not destination.is_file():
            raise ValueError(f"Redirect target was not built: {new}")
        page = output / (old + ".html")
        if old in app.env.found_docs:
            raise ValueError(f"Redirect would overwrite a published chapter: {old}")
        relative = posixpath.relpath(new + ".html", posixpath.dirname(old))
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(redirect_html(relative, anchors.get(old)), encoding="utf-8")


def setup(app):
    app.connect("build-finished", write_redirects)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
