#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
TOC_FILE = DOCS_DIR / "_toc.yml"
OUTPUT_FILE = ROOT / "prompt.md"


def parse_toc(toc_path: Path) -> list[str]:
    if not toc_path.exists():
        raise FileNotFoundError(f"Missing Jupyter Book TOC: {toc_path}")

    entries: list[str] = []
    for line in toc_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("root:"):
            value = stripped.split(":", 1)[1].strip()
            if value:
                entries.append(value)
            continue
        if stripped.startswith("- file:") or stripped.startswith("file:"):
            value = stripped.split(":", 1)[1].strip()
            if value:
                entries.append(value)
    return entries


def resolve_doc_path(relative_path: str) -> Path:
    base = DOCS_DIR / relative_path
    if base.suffix and base.exists():
        return base
    for ext in (".md", ".ipynb", ".rst"):
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Document not found for TOC entry: {relative_path}")


def ensure_text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)


def render_outputs(outputs: list[dict]) -> str:
    blocks: list[str] = []
    for output in outputs:
        output_type = output.get("output_type")
        if output_type == "stream":
            text = ensure_text(output.get("text", ""))
            if text.strip():
                blocks.append(f"```text\n{text.rstrip()}\n```")
            continue
        if output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            if "text/markdown" in data:
                text = ensure_text(data.get("text/markdown", ""))
                if text.strip():
                    blocks.append(text.rstrip())
                continue
            if "text/plain" in data:
                text = ensure_text(data.get("text/plain", ""))
                if text.strip():
                    blocks.append(f"```text\n{text.rstrip()}\n```")
            continue
        if output_type == "error":
            traceback = ensure_text(output.get("traceback", []))
            if traceback.strip():
                blocks.append(f"```text\n{traceback.rstrip()}\n```")
    return "\n\n".join(blocks)


def render_notebook(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in data.get("cells", []):
        cell_type = cell.get("cell_type")
        source = ensure_text(cell.get("source", ""))
        if cell_type == "markdown":
            if source.strip():
                parts.append(source)
            continue
        if cell_type == "code":
            code = source.rstrip()
            if code:
                parts.append(f"```python\n{code}\n```")
            output_block = render_outputs(cell.get("outputs", []))
            if output_block:
                parts.append(output_block)
            continue
        if cell_type == "raw":
            if source.strip():
                parts.append(source)
    return "\n\n".join(parts)


def read_doc(path: Path) -> str:
    if path.suffix == ".ipynb":
        return render_notebook(path)
    return path.read_text(encoding="utf-8")


def build_prompt() -> None:
    entries = parse_toc(TOC_FILE)
    contents: list[str] = []
    for entry in entries:
        doc_path = resolve_doc_path(entry)
        content = read_doc(doc_path).strip("\n")
        if content:
            contents.append(content)
    OUTPUT_FILE.write_text("\n\n".join(contents) + "\n", encoding="utf-8")


def main() -> None:
    build_prompt()
    print(f"Wrote {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
