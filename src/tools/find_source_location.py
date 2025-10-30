#!/usr/bin/env python3
"""
Find Source Location CLI Tool.

User-friendly command-line tool for finding exact source locations of text
in markdown documents. Outputs SourceLocation JSON for easy copy-paste into
enrichment data.

Use Cases:
- Quickly find line range for a text snippet
- Locate directive by label
- Find equations by content
- Batch process multiple queries from file

Examples:
    # Find text snippet
    python src/tools/find_source_location.py find-text \\
        docs/source/1_euclidean_gas/03_cloning.md \\
        "The Keystone Principle states" \\
        --document-id 03_cloning

    # Find by directive label
    python src/tools/find_source_location.py find-directive \\
        docs/source/1_euclidean_gas/03_cloning.md \\
        thm-keystone \\
        --document-id 03_cloning

    # Find equation
    python src/tools/find_source_location.py find-equation \\
        docs/source/1_euclidean_gas/02_euclidean_gas.md \\
        "\\mathbb{E}[V] = 0" \\
        --document-id 02_euclidean_gas

    # Batch mode from CSV
    python src/tools/find_source_location.py batch \\
        queries.csv \\
        --output results.json

Maps to Lean:
    namespace FindSourceLocationCLI
      def find_text_cli : Args → IO Unit
      def find_directive_cli : Args → IO Unit
      def batch_cli : Args → IO Unit
    end FindSourceLocationCLI
"""

import argparse
import csv
import json
from pathlib import Path

# Add src to path for imports
import sys
from typing import Any, Dict, List, Optional


src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.tools.line_finder import (
    extract_lines,
    find_all_occurrences,
    find_directive_lines,
    find_equation_lines,
    find_section_lines,
    find_text_in_markdown,
)
from fragile.proofs.utils.source_helpers import SourceLocationBuilder


# =============================================================================
# UTILITIES
# =============================================================================


def read_markdown_file(file_path: Path) -> str:
    """Read markdown file content."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def infer_document_id(file_path: Path) -> str:
    """
    Infer document_id from file path.

    Examples:
        docs/source/1_euclidean_gas/03_cloning.md → "03_cloning"
        docs/source/2_geometric_gas/11_geometric_gas.md → "11_geometric_gas"
    """
    return file_path.stem


def create_source_location_json(
    document_id: str,
    file_path: str,
    line_range: tuple | None,
    directive_label: str | None = None,
    section: str | None = None,
    equation: str | None = None,
) -> dict[str, Any]:
    """Create SourceLocation dictionary for JSON output."""
    if line_range:
        loc = SourceLocationBuilder.from_markdown_location(
            document_id=document_id,
            file_path=file_path,
            start_line=line_range[0],
            end_line=line_range[1],
            section=section,
        )
        if directive_label:
            # Add directive label if available
            loc = SourceLocation(
                document_id=loc.document_id,
                file_path=loc.file_path,
                line_range=loc.line_range,
                directive_label=directive_label,
                section=loc.section,
                url_fragment=f"#{directive_label}",
            )
    elif directive_label:
        loc = SourceLocationBuilder.from_jupyter_directive(
            document_id=document_id,
            file_path=file_path,
            directive_label=directive_label,
            section=section,
            equation=equation,
        )
    elif section:
        loc = SourceLocationBuilder.from_section(
            document_id=document_id,
            file_path=file_path,
            section=section,
            equation=equation,
        )
    else:
        loc = SourceLocationBuilder.minimal(
            document_id=document_id,
            file_path=file_path,
        )

    return loc.model_dump(mode="json")


def print_result(
    result: dict[str, Any], markdown_content: str, show_text: bool = True, format: str = "pretty"
) -> None:
    """Print search result in requested format."""
    if format == "json":
        print(json.dumps(result, indent=2))
        return

    # Pretty format
    print("\n" + "=" * 60)
    print("SOURCE LOCATION FOUND")
    print("=" * 60)

    if result.get("line_range"):
        start, end = result["line_range"]
        print(f"Document:  {result['document_id']}")
        print(f"File:      {result['file_path']}")
        print(f"Lines:     {start}-{end}")

        if result.get("directive_label"):
            print(f"Directive: {result['directive_label']}")

        if result.get("section"):
            print(f"Section:   {result['section']}")

        if show_text:
            print(f"\n{'─' * 60}")
            print("TEXT AT LOCATION:")
            print(f"{'─' * 60}")
            extracted_text = extract_lines(markdown_content, (start, end))
            print(extracted_text)
            print(f"{'─' * 60}")

    else:
        print(f"Document:  {result['document_id']}")
        print(f"File:      {result['file_path']}")
        if result.get("directive_label"):
            print(f"Directive: {result['directive_label']}")
        if result.get("section"):
            print(f"Section:   {result['section']}")

    print("\nJSON OUTPUT (copy for enrichment):")
    print(json.dumps(result, indent=2))


def print_not_found(query: str, suggestion: str = "") -> None:
    """Print not found message with suggestion."""
    print(f"\n❌ Not found: {query}", file=sys.stderr)
    if suggestion:
        print(f"💡 Try: {suggestion}", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# COMMAND HANDLERS
# =============================================================================


def handle_find_text(args):
    """Find text snippet in markdown."""
    content = read_markdown_file(args.markdown_file)

    # Auto-infer document_id if not provided
    document_id = args.document_id or infer_document_id(args.markdown_file)

    line_range = find_text_in_markdown(
        content, args.text, context_lines=args.context, case_sensitive=args.case_sensitive
    )

    if not line_range:
        # Try finding all occurrences to give better feedback
        occurrences = find_all_occurrences(content, args.text, case_sensitive=args.case_sensitive)
        if occurrences:
            print(f"\n⚠ Found {len(occurrences)} similar matches:", file=sys.stderr)
            for i, (start, end) in enumerate(occurrences[:5], 1):
                print(f"  {i}. Lines {start}-{end}", file=sys.stderr)
            print("\n💡 Try: Provide more context or use --case-sensitive", file=sys.stderr)
            sys.exit(1)
        else:
            print_not_found(args.text, "Use shorter snippet or check spelling")

    result = create_source_location_json(
        document_id=document_id,
        file_path=str(args.markdown_file),
        line_range=line_range,
        section=args.section,
    )

    print_result(result, content, show_text=args.show_text, format=args.format)


def handle_find_directive(args):
    """Find Jupyter Book directive by label."""
    content = read_markdown_file(args.markdown_file)

    # Auto-infer document_id if not provided
    document_id = args.document_id or infer_document_id(args.markdown_file)

    line_range = find_directive_lines(content, args.label, directive_type=args.type)

    if not line_range:
        suggestion = f"Check directive label: '{args.label}'"
        if args.type:
            suggestion += f" with type '{args.type}'"
        print_not_found(args.label, suggestion)

    result = create_source_location_json(
        document_id=document_id,
        file_path=str(args.markdown_file),
        line_range=line_range,
        directive_label=args.label,
        section=args.section,
    )

    print_result(result, content, show_text=args.show_text, format=args.format)


def handle_find_equation(args):
    """Find LaTeX equation by content."""
    content = read_markdown_file(args.markdown_file)

    # Auto-infer document_id if not provided
    document_id = args.document_id or infer_document_id(args.markdown_file)

    line_range = find_equation_lines(content, args.latex, equation_label=args.equation_label)

    if not line_range:
        suggestion = "Check LaTeX content (without $$ delimiters)"
        print_not_found(args.latex, suggestion)

    result = create_source_location_json(
        document_id=document_id,
        file_path=str(args.markdown_file),
        line_range=line_range,
        section=args.section,
        equation=args.equation_label,
    )

    print_result(result, content, show_text=args.show_text, format=args.format)


def handle_find_section(args):
    """Find section by heading."""
    content = read_markdown_file(args.markdown_file)

    # Auto-infer document_id if not provided
    document_id = args.document_id or infer_document_id(args.markdown_file)

    line_range = find_section_lines(content, args.heading, exact_match=args.exact)

    if not line_range:
        suggestion = "Check section heading or try without --exact"
        print_not_found(args.heading, suggestion)

    result = create_source_location_json(
        document_id=document_id,
        file_path=str(args.markdown_file),
        line_range=line_range,
        section=args.heading,
    )

    print_result(result, content, show_text=args.show_text, format=args.format)


def handle_batch(args):
    """Process batch queries from CSV file."""
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    results = []
    failed = []

    with open(args.input_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            try:
                markdown_file = Path(row["markdown_file"])
                document_id = row.get("document_id") or infer_document_id(markdown_file)
                query_type = row["query_type"]
                query = row["query"]

                content = read_markdown_file(markdown_file)

                # Find location based on query type
                if query_type == "text":
                    line_range = find_text_in_markdown(content, query)
                    directive_label = None
                elif query_type == "directive":
                    line_range = find_directive_lines(content, query)
                    directive_label = query
                elif query_type == "equation":
                    line_range = find_equation_lines(content, query)
                    directive_label = None
                elif query_type == "section":
                    line_range = find_section_lines(content, query)
                    directive_label = None
                else:
                    print(
                        f"Warning: Unknown query type '{query_type}' on line {i}", file=sys.stderr
                    )
                    continue

                if line_range:
                    result = create_source_location_json(
                        document_id=document_id,
                        file_path=str(markdown_file),
                        line_range=line_range,
                        directive_label=directive_label,
                        section=row.get("section"),
                    )
                    results.append({
                        "query": query,
                        "query_type": query_type,
                        "found": True,
                        "source_location": result,
                    })
                else:
                    failed.append({
                        "line": i,
                        "query": query,
                        "query_type": query_type,
                        "file": str(markdown_file),
                    })

            except Exception as e:
                print(f"Error processing line {i}: {e}", file=sys.stderr)
                failed.append({"line": i, "query": row.get("query", ""), "error": str(e)})

    # Write results
    output_file = args.output or args.input_file.with_suffix(".results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "succeeded": len(results),
                "failed": len(failed),
                "results": results,
                "failed_queries": failed,
            },
            f,
            indent=2,
        )

    # Print summary
    print("\nBatch processing complete:")
    print(f"  ✓ Succeeded: {len(results)}")
    print(f"  ✗ Failed: {len(failed)}")
    print(f"\nResults written to: {output_file}")

    if failed:
        print("\nFailed queries:")
        for item in failed[:10]:  # Show first 10
            print(f"  Line {item.get('line', '?')}: {item['query']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    sys.exit(0 if len(failed) == 0 else 1)


# =============================================================================
# CLI SETUP
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Find source locations in markdown documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find text
  %(prog)s find-text docs/source/.../03_cloning.md "The Keystone Principle"

  # Find directive
  %(prog)s find-directive docs/source/.../03_cloning.md thm-keystone

  # Find equation
  %(prog)s find-equation docs/source/.../02_euclidean_gas.md "\\mathbb{E}[V] = 0"

  # Batch mode
  %(prog)s batch queries.csv --output results.json

CSV format for batch mode:
  markdown_file,document_id,query_type,query,section
  docs/.../03_cloning.md,03_cloning,text,"Keystone Principle",§3.2
  docs/.../02_euclidean_gas.md,02_euclidean_gas,directive,thm-main,
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments for all find commands
    def add_common_args(subparser):
        subparser.add_argument("markdown_file", type=Path, help="Path to markdown file")
        subparser.add_argument(
            "--document-id", "-d", help="Document ID (auto-inferred if not provided)"
        )
        subparser.add_argument("--section", "-s", help="Section reference (e.g., '§3.2')")
        subparser.add_argument(
            "--format", "-f", choices=["pretty", "json"], default="pretty", help="Output format"
        )
        subparser.add_argument(
            "--show-text", action="store_true", default=True, help="Show matched text"
        )
        subparser.add_argument(
            "--no-show-text",
            dest="show_text",
            action="store_false",
            help="Don't show matched text",
        )

    # find-text command
    text_parser = subparsers.add_parser("find-text", help="Find text snippet in markdown")
    add_common_args(text_parser)
    text_parser.add_argument("text", help="Text snippet to search for")
    text_parser.add_argument(
        "--context", "-c", type=int, default=0, help="Context lines before/after"
    )
    text_parser.add_argument("--case-sensitive", action="store_true", help="Case-sensitive search")

    # find-directive command
    directive_parser = subparsers.add_parser(
        "find-directive", help="Find Jupyter Book directive by label"
    )
    add_common_args(directive_parser)
    directive_parser.add_argument("label", help="Directive label (e.g., 'thm-keystone')")
    directive_parser.add_argument("--type", "-t", help="Directive type (e.g., 'theorem', 'lemma')")

    # find-equation command
    equation_parser = subparsers.add_parser("find-equation", help="Find LaTeX equation by content")
    add_common_args(equation_parser)
    equation_parser.add_argument("latex", help="LaTeX content (without $$ delimiters)")
    equation_parser.add_argument(
        "--equation-label", "-e", help="Equation label if known (e.g., '(2.3)')"
    )

    # find-section command
    section_parser = subparsers.add_parser("find-section", help="Find section by heading")
    add_common_args(section_parser)
    section_parser.add_argument("heading", help="Section heading text")
    section_parser.add_argument("--exact", action="store_true", help="Require exact match")

    # batch command
    batch_parser = subparsers.add_parser("batch", help="Process batch queries from CSV")
    batch_parser.add_argument("input_file", type=Path, help="Input CSV file")
    batch_parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON file (default: input.results.json)"
    )

    args = parser.parse_args()

    if args.command == "find-text":
        handle_find_text(args)
    elif args.command == "find-directive":
        handle_find_directive(args)
    elif args.command == "find-equation":
        handle_find_equation(args)
    elif args.command == "find-section":
        handle_find_section(args)
    elif args.command == "batch":
        handle_batch(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
