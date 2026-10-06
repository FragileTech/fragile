#!/usr/bin/env python3
"""
Collect MyST prf directives into per-volume markdown files.

By default, prf:proof blocks are excluded. Use --include-proofs to include them.
"""

from __future__ import annotations

import argparse
import operator
from pathlib import Path
import re

from book_manifest import markdown_source, published_documents


DIRECTIVE_OPEN_RE = re.compile(r"^\s*:{2,}\s*\{(?P<name>[^}]+)\}.*$")
DIRECTIVE_CLOSE_RE = re.compile(r"^\s*:{2,}\s*$")
VOLUME_DIR_RE = re.compile(r"^\d+_")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_source = script_dir / "source"
    default_out_dir = repo_root / "outputs" / "prf"

    parser = argparse.ArgumentParser(
        description="Extract prf directives from docs/source into per-volume markdown files.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help="Docs source directory (default: docs/source).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Output directory for condensed markdown files (default: outputs/prf).",
    )
    parser.add_argument(
        "--include-proofs",
        action="store_true",
        help="Include prf:proof directives (default: excluded).",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include published nonnumeric source groups as well as numbered volumes.",
    )
    parser.add_argument(
        "--include-file-headings",
        action="store_true",
        help="Add a '## <file>' heading before blocks from each file.",
    )
    return parser.parse_args()


def volume_dirs(source_dir: Path, include_all: bool) -> list[Path]:
    """Return only volumes containing published documents, in TOC order."""
    source_dir = source_dir.resolve()
    volumes = []
    for document in published_documents(source_dir.parent):
        if not document.is_relative_to(source_dir):
            continue
        volume = source_dir / document.relative_to(source_dir).parts[0]
        if volume.is_dir() and (include_all or VOLUME_DIR_RE.match(volume.name)):
            if volume not in volumes:
                volumes.append(volume)
    return volumes


def extract_prf_blocks(text: str, include_proofs: bool) -> list[str]:
    """Extract formal blocks once, preserving order and handling nested proofs.

    Code fences are opaque. Nested proofs are removed from their enclosing
    theorem when proofs are excluded, rather than leaking through its body.
    """
    lines = text.splitlines()
    opening = re.compile(r"^\s*(:{3,}|`{3,}|~{3,})\s*\{([^}]+)\}")
    fence = re.compile(r"^\s*(:{3,}|`{3,}|~{3,})\s*$")
    code_open = re.compile(r"^\s*(`{3,}|~{3,})")
    stack = []
    nodes = []
    for index, line in enumerate(lines):
        close = fence.match(line)
        if (
            stack
            and close
            and close[1][0] == stack[-1]["fence"][0]
            and len(close[1]) >= len(stack[-1]["fence"])
        ):
            node = stack.pop()
            node["end"] = index + 1
            nodes.append(node)
            continue
        if stack and stack[-1]["code"]:
            continue
        match = opening.match(line)
        if match:
            stack.append({
                "start": index,
                "fence": match[1],
                "name": match[2],
                "code": match[2] in {"code", "code-block", "code-cell", "math", "mermaid", "raw"},
            })
        elif code_open.match(line):
            stack.append({
                "start": index,
                "fence": code_open.match(line)[1],
                "name": "code",
                "code": True,
            })
    formal = sorted(
        (n for n in nodes if n["name"].startswith("prf:")), key=operator.itemgetter("start")
    )
    result = []
    for node in formal:
        if any(other["start"] < node["start"] < other["end"] for other in formal):
            continue
        if not include_proofs and node["name"] == "prf:proof":
            continue
        excluded = set()
        if not include_proofs:
            for child in formal:
                if child["name"] == "prf:proof" and node["start"] < child["start"] < node["end"]:
                    excluded.update(range(child["start"], child["end"]))
        if not include_proofs:
            for i in range(node["start"] + 1, node["end"] - 1):
                if re.match(
                    r"^\s*(?:\*{1,2})?Proof[.:]?(?:\*{1,2})?(?:\s|$)", lines[i], re.IGNORECASE
                ):
                    excluded.update(range(i, node["end"] - 1))
                    break
        result.append(
            "\n".join(lines[i] for i in range(node["start"], node["end"]) if i not in excluded)
        )
    return result


def build_volume_output(
    volume_dir: Path,
    include_proofs: bool,
    include_file_headings: bool,
) -> tuple[str, int, int]:
    volume_dir = volume_dir.resolve()
    parts = []
    block_count = file_count = 0
    for document in published_documents(volume_dir.parent.parent):
        if not document.is_relative_to(volume_dir):
            continue
        blocks = extract_prf_blocks(markdown_source(document), include_proofs)
        if not blocks:
            continue
        file_count += 1
        if include_file_headings:
            parts.append(f"## {document.relative_to(volume_dir)}")
        parts.extend(blocks)
        block_count += len(blocks)
    output = "\n\n".join(parts).rstrip()
    return output + ("\n" if output else ""), block_count, file_count


def main() -> None:
    args = parse_args()
    source_dir = args.source
    out_dir = args.out_dir

    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for volume_dir in volume_dirs(source_dir, include_all=args.include_all):
        output, block_count, file_count = build_volume_output(
            volume_dir,
            include_proofs=args.include_proofs,
            include_file_headings=args.include_file_headings,
        )
        out_path = out_dir / f"{volume_dir.name}.md"
        out_path.write_text(output, encoding="utf-8")
        print(f"{volume_dir.name}: {block_count} blocks from {file_count} files -> {out_path}")


if __name__ == "__main__":
    main()
