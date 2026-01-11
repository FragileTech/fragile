"""
MyST Markdown Directive Parser.

Extracts mathematical content from Jupyter Book MyST markdown documents.
Parses directives like {prf:definition}, {prf:theorem}, {prf:proof}, etc.

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


# MyST directive pattern: :::{prf:type} Title
# Captures opening (3 colons), title on same line, label on next line, content, closing (3 colons)
DIRECTIVE_PATTERN = re.compile(
    r"^:::\{prf:(\w+)\}\s+(?P<title>[^\n]+)\n"  # Opening with title on same line (3 colons)
    r":label:\s+(?P<label>[^\n]+)\n"  # Label on next line (required)
    r"\n?"  # Optional blank line
    r"(?P<content>.*?)"  # Content (non-greedy)
    r"\n^:::\s*$",  # Closing (3 colons)
    re.MULTILINE | re.DOTALL,
)

# Simpler pattern for directives without content
SIMPLE_DIRECTIVE_PATTERN = re.compile(
    r"^::::\{prf:(\w+)\}\s+([^\n]+)\n" r":label:\s+([^\n]+)\n" r"(.*?)" r"^::::\s*$",
    re.MULTILINE | re.DOTALL,
)

# Pattern for extracting math expressions
DISPLAY_MATH_PATTERN = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
INLINE_MATH_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)")

# Pattern for cross-references
REF_PATTERN = re.compile(r"\{prf:ref\}`([^`]+)`")


@dataclass
class MathDirective:
    """
    Represents a parsed MyST mathematical directive.

    Attributes:
        directive_type: Type of directive (definition, theorem, lemma, proof, etc.)
        label: Unique label for cross-referencing
        title: Human-readable title/name
        content: Full content of the directive
        math_expressions: List of LaTeX expressions found in content
        cross_refs: List of labels referenced in this directive
        line_start: Starting line number in source file
        line_end: Ending line number in source file
    """

    directive_type: str
    label: str
    title: str
    content: str
    math_expressions: list[str]
    cross_refs: list[str]
    line_start: int
    line_end: int

    def get_first_math(self) -> str | None:
        """Get first mathematical expression (useful for primary definition)."""
        return self.math_expressions[0] if self.math_expressions else None

    def has_cross_ref(self, label: str) -> bool:
        """Check if this directive references another label."""
        return label in self.cross_refs

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.directive_type,
            "label": self.label,
            "title": self.title,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "math_expression_count": len(self.math_expressions),
            "first_math": self.get_first_math(),
            "cross_refs": self.cross_refs,
            "line_range": [self.line_start, self.line_end],
        }


@dataclass
class DocumentInventory:
    """
    Complete inventory of all mathematical directives in a document.

    Attributes:
        source_file: Path to source document
        directives: List of all parsed directives
        directive_index: Map from label to directive
        type_index: Map from directive type to list of directives
    """

    source_file: Path
    directives: list[MathDirective]
    directive_index: dict[str, MathDirective]
    type_index: dict[str, list[MathDirective]]

    def get_by_label(self, label: str) -> MathDirective | None:
        """Get directive by its label."""
        return self.directive_index.get(label)

    def get_by_type(self, directive_type: str) -> list[MathDirective]:
        """Get all directives of a specific type."""
        return self.type_index.get(directive_type, [])

    def get_definitions(self) -> list[MathDirective]:
        """Get all definition directives."""
        return self.get_by_type("definition")

    def get_theorems(self) -> list[MathDirective]:
        """Get all theorem directives (theorem, lemma, proposition)."""
        return (
            self.get_by_type("theorem")
            + self.get_by_type("lemma")
            + self.get_by_type("proposition")
        )

    def get_proofs(self) -> list[MathDirective]:
        """Get all proof directives."""
        return self.get_by_type("proof")

    def get_axioms(self) -> list[MathDirective]:
        """Get all axiom directives."""
        return self.get_by_type("axiom")

    def count_by_type(self) -> dict[str, int]:
        """Get count of directives by type."""
        return {dtype: len(directives) for dtype, directives in self.type_index.items()}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_file": str(self.source_file),
            "total_directives": len(self.directives),
            "counts_by_type": self.count_by_type(),
            "directives": [d.to_dict() for d in self.directives],
        }


class MySTParser:
    """
    Parser for MyST markdown mathematical directives.

    Extracts all {prf:...} blocks from Jupyter Book documents and creates
    a structured inventory of mathematical content.
    """

    def __init__(self, source_file: Path):
        """
        Initialize parser with source document.

        Args:
            source_file: Path to MyST markdown file
        """
        self.source_file = Path(source_file)
        self.content = self.source_file.read_text()
        self.lines = self.content.split("\n")

    def parse(self) -> DocumentInventory:
        """
        Parse document and extract all directives.

        Returns:
            DocumentInventory with all parsed directives
        """
        directives = []

        # Find all directive matches
        for match in DIRECTIVE_PATTERN.finditer(self.content):
            directive = self._parse_directive_match(match)
            if directive:
                directives.append(directive)

        # Build indices
        directive_index = {d.label: d for d in directives if d.label}
        type_index: dict[str, list[MathDirective]] = {}
        for directive in directives:
            if directive.directive_type not in type_index:
                type_index[directive.directive_type] = []
            type_index[directive.directive_type].append(directive)

        return DocumentInventory(
            source_file=self.source_file,
            directives=directives,
            directive_index=directive_index,
            type_index=type_index,
        )

    def _parse_directive_match(self, match: re.Match) -> MathDirective | None:
        """
        Parse a single directive match into MathDirective.

        Args:
            match: Regex match object

        Returns:
            MathDirective or None if parsing fails
        """
        directive_type = match.group(1)
        title = match.group("title").strip()
        label = match.group("label")
        content = match.group("content").strip()

        # Extract line numbers
        start_pos = match.start()
        end_pos = match.end()
        line_start = self.content[:start_pos].count("\n") + 1
        line_end = self.content[:end_pos].count("\n") + 1

        # Extract mathematical expressions
        math_expressions = self._extract_math(content)

        # Extract cross-references
        cross_refs = self._extract_cross_refs(content)

        # Create label if not present
        if not label:
            label = self._generate_label(directive_type, title)

        return MathDirective(
            directive_type=directive_type,
            label=label,
            title=title,
            content=content,
            math_expressions=math_expressions,
            cross_refs=cross_refs,
            line_start=line_start,
            line_end=line_end,
        )

    def _extract_math(self, content: str) -> list[str]:
        """
        Extract all mathematical expressions from content.

        Args:
            content: Text content

        Returns:
            List of LaTeX expressions
        """
        expressions = []

        # Extract display math ($$...$$)
        for match in DISPLAY_MATH_PATTERN.finditer(content):
            expressions.append(match.group(1).strip())

        # Extract inline math ($...$)
        for match in INLINE_MATH_PATTERN.finditer(content):
            expressions.append(match.group(1).strip())

        return expressions

    def _extract_cross_refs(self, content: str) -> list[str]:
        """
        Extract all cross-references from content.

        Args:
            content: Text content

        Returns:
            List of referenced labels
        """
        refs = []
        for match in REF_PATTERN.finditer(content):
            refs.append(match.group(1))
        return refs

    def _generate_label(self, directive_type: str, title: str) -> str:
        """
        Generate label from directive type and title.

        Args:
            directive_type: Type of directive
            title: Title/name

        Returns:
            Generated label
        """
        # Convert title to kebab-case
        label_text = title.lower().replace(" ", "-")
        label_text = re.sub(r"[^a-z0-9-]", "", label_text)

        # Add prefix based on type
        prefix_map = {
            "definition": "def",
            "theorem": "thm",
            "lemma": "lem",
            "proposition": "prop",
            "axiom": "axiom",
            "proof": "proof",
            "algorithm": "alg",
            "remark": "remark",
        }
        prefix = prefix_map.get(directive_type, "item")

        return f"{prefix}-{label_text}"


def parse_document(source_file: Path) -> DocumentInventory:
    """
    Parse a MyST markdown document and extract all mathematical directives.

    Args:
        source_file: Path to MyST markdown file

    Returns:
        DocumentInventory with all parsed directives
    """
    parser = MySTParser(source_file)
    return parser.parse()
