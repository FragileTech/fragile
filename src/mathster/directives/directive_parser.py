"""
Directive Parser for Jupyter Book MyST Markdown.

This module implements a hybrid parsing approach for extracting mathematical
entities from Jupyter Book MyST markdown documents:

1. **Python Extraction**: Extract directive structure (type, label, line range)
2. **LLM Enrichment**: Provide hints to LLM for content validation and semantic processing

This approach combines the reliability of structural parsing with the flexibility
of LLM-based semantic understanding.

Supported Directives:
- {prf:definition}
- {prf:theorem}, {prf:lemma}, {prf:proposition}, {prf:corollary}
- {prf:proof}
- {prf:axiom}

Maps to Lean:
    namespace DirectiveParser
      structure DirectiveHint where
        directive_type : String
        label : String
        start_line : Nat
        end_line : Nat
        content : String
    end DirectiveParser
"""

from dataclasses import dataclass
import re


@dataclass
class DirectiveHint:
    """
    Structural hint extracted from Jupyter Book directive.

    Provides the LLM with reliable structural information about where
    mathematical entities are located in the document.
    """

    directive_type: str  # "definition", "theorem", "lemma", "proof", "axiom", etc.
    label: str  # The :label: value (e.g., "def-walker-state")
    title: str | None  # Title from first line or :name: field
    start_line: int  # Line number where directive starts (1-indexed)
    end_line: int  # Line number where directive ends (1-indexed)
    header_lines: list[int]  # Lines containing :field: metadata
    content_start: int  # First line of actual content (after metadata)
    content_end: int  # Last line of content (before closing :::)
    content: str  # Raw content between directive markers
    metadata: dict  # All :field: values (class, nonumber, name, etc.)
    section: str  # Section identifier where this appears
    references: list[str]  # Labels referenced inside the directive content

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "directive_type": self.directive_type,
            "label": self.label,
            "title": self.title,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "header_lines": self.header_lines,
            "content_start": self.content_start,
            "content_end": self.content_end,
            "content": self.content,
            "metadata": self.metadata,
            "section": self.section,
            "references": self.references,
        }


@dataclass
class DocumentSection:
    """
    A section of the document for parallel processing.

    Documents are split by headings to enable parallel extraction.
    """

    section_id: str  # Unique identifier (e.g., "�1-intro", "�2.1-main-results")
    title: str  # Section title
    level: int  # Heading level (1-6)
    start_line: int  # Line number where section starts (1-indexed)
    end_line: int  # Line number where section ends (1-indexed)
    content: str  # Full markdown content of the section
    directives: list[DirectiveHint]  # Directive hints found in this section


# =============================================================================
# REFERENCE EXTRACTION
# =============================================================================

REFERENCE_INLINE_PATTERN = re.compile(r"\{(?P<role>[\w:-]+)\}`(?P<body>[^`]+)`")
REFERENCE_ROLE_NAMES = {"ref", "prf:ref", "numref"}


def extract_label_references(text: str | None) -> list[str]:
    """
    Extract all label references from directive content.

    Supports common MyST inline roles such as ``{prf:ref}`label`` and
    ``{ref}`Title <label>``. Returns a list of unique labels preserving the
    order in which they appear.
    """
    if not text:
        return []

    references: list[str] = []
    seen: set[str] = set()

    for match in REFERENCE_INLINE_PATTERN.finditer(text):
        role = match.group("role").strip().lower()
        if role not in REFERENCE_ROLE_NAMES and not role.endswith(":ref"):
            continue

        candidate = _clean_reference_target(match.group("body"))
        if candidate and candidate not in seen:
            references.append(candidate)
            seen.add(candidate)

    return references


def _clean_reference_target(body: str) -> str | None:
    """Extract the label portion from a MyST inline role body."""
    body = body.strip()
    if not body:
        return None

    angle_match = re.search(r"<([^<>`]+)>", body)
    if angle_match:
        candidate = angle_match.group(1).strip()
        if candidate:
            return candidate

    return body or None


# =============================================================================
# DIRECTIVE EXTRACTION
# =============================================================================


def extract_jupyter_directives(
    markdown_text: str, section_id: str = "main"
) -> list[DirectiveHint]:
    """
    Extract Jupyter Book directive structure from markdown text.

    This function identifies {prf:*} directives and extracts their:
    - Type (definition, theorem, lemma, proof, axiom, etc.)
    - Label (from :label: field)
    - Line range (start and end)
    - Raw content

    The LLM then uses these hints to validate and enrich the content.

    Args:
        markdown_text: The markdown content to parse
        section_id: Identifier for the section this text comes from

    Returns:
        List of DirectiveHint objects with structural information

    Examples:
        >>> text = '''
        ... :::{prf:theorem} Main Convergence
        ... :label: thm-main-convergence
        ...
        ... Under the stated axioms, the Euclidean Gas converges.
        ... :::
        ... '''
        >>> hints = extract_jupyter_directives(text)
        >>> hints[0].directive_type
        'theorem'
        >>> hints[0].label
        'thm-main-convergence'
    """
    directives = []
    lines = markdown_text.split("\n")

    # Pattern for directive start: :::{prf:TYPE} TITLE
    directive_start_pattern = r"^:::\{prf:(\w+)\}(?:\s+(.+))?$"
    # Pattern for metadata field: :field: value
    metadata_pattern = r"^:(\w+):\s*(.*)$"
    # Pattern for directive end: :::
    directive_end_pattern = r"^:::$"

    i = 0
    while i < len(lines):
        line = lines[i]

        # Strip line number prefix if present (format: "NNN: content")
        line_stripped = re.sub(r"^\d+:\s*", "", line).strip()

        start_match = re.match(directive_start_pattern, line_stripped)

        if start_match:
            directive_type = start_match.group(1)
            title_from_first_line = start_match.group(2)  # Title on same line as opening
            start_line = i + 1  # 1-indexed

            # Extract all metadata fields and find content start
            label = None
            metadata = {}
            header_lines = []
            content_start_line = None
            j = i + 1

            while j < len(lines):
                # Strip line numbers
                line_stripped = re.sub(r"^\d+:\s*", "", lines[j]).strip()

                # Check for directive end first
                if re.match(directive_end_pattern, line_stripped):
                    break

                # Check for metadata field
                meta_match = re.match(metadata_pattern, line_stripped)
                if meta_match:
                    field_name = meta_match.group(1)
                    field_value = meta_match.group(2).strip()
                    metadata[field_name] = field_value
                    header_lines.append(j + 1)  # 1-indexed

                    # Special handling for label and name
                    if field_name == "label":
                        label = field_value
                    elif field_name == "name" and not title_from_first_line:
                        title_from_first_line = field_value

                elif line_stripped and not line_stripped.startswith(":"):
                    # First non-empty, non-field line is start of content
                    if content_start_line is None:
                        content_start_line = j + 1  # 1-indexed
                j += 1

            # If no content start found, use line after last metadata
            if content_start_line is None:
                content_start_line = j + 1

            # Find directive end
            end_line = None
            k = j
            while k < len(lines):
                # Strip line numbers before matching
                k_stripped = re.sub(r"^\d+:\s*", "", lines[k]).strip()
                if re.match(directive_end_pattern, k_stripped):
                    end_line = k + 1  # 1-indexed
                    break
                k += 1

            if end_line is not None:
                # Extract content (between content_start and end marker)
                content_end_line = k  # Line before closing :::
                if content_start_line <= k:
                    content_lines = lines[content_start_line - 1 : k]
                    content = "\n".join(content_lines).strip()
                else:
                    content = ""
                references = extract_label_references(content)

                directives.append(
                    DirectiveHint(
                        directive_type=directive_type,
                        label=label or f"unlabeled-{directive_type}-{start_line}",
                        title=title_from_first_line,
                        start_line=start_line,
                        end_line=end_line,
                        header_lines=header_lines,
                        content_start=content_start_line,
                        content_end=content_end_line,
                        content=content,
                        metadata=metadata,
                        section=section_id,
                        references=references,
                    )
                )

                # Move past this directive
                i = k + 1
            else:
                # Malformed directive, skip
                i += 1
        else:
            i += 1

    return directives


# =============================================================================
# SECTION SPLITTING
# =============================================================================


def split_into_sections(
    markdown_text: str, file_path: str = "unknown", heading_scope: str | None = "##"
) -> list[DocumentSection]:
    """
    Split markdown document into sections by headings.

    This enables parallel processing of sections during Stage 1 extraction.
    Each section is processed independently, then merged.

    Heading Levels:
        # = Level 1 (chapter)
        ## = Level 2 (major section)
        ### = Level 3 (subsection)
        etc.

    Section IDs:
        - Generated from heading text (lowercase, hyphenated)
        - Prefixed with  and numbering if present
        - Examples: "1-introduction", "2.1-main-results", "preliminaries"

    Args:
        markdown_text: The full markdown document
        file_path: Path to the document (for metadata)
        heading_scope: Optional heading string (e.g., "##", "###") that selects which
            heading depth starts a new section. Subheadings (with more '#') are kept
            inside the parent section. When ``None`` (default), every heading level
            starts a new section (legacy behavior).

    Returns:
        List of DocumentSection objects

    Examples:
        >>> text = '''
        ... # Chapter 1: Introduction
        ...
        ... Some intro text.
        ...
        ... ## Section 2.1: Main Results
        ...
        ... More content here.
        ... '''
        >>> sections = split_into_sections(text)
        >>> len(sections)
        2
        >>> sections[0].title
        'Chapter 1: Introduction'
    """
    sections = []
    lines = markdown_text.split("\n")

    # Pattern for markdown headings: # Title or ## Title or ### Title
    heading_pattern = r"^(#{1,6})\s+(.+)$"

    target_level: int | None = None
    if heading_scope is not None:
        if not heading_scope or any(ch != "#" for ch in heading_scope):
            msg = "heading_scope must be a non-empty string of '#' characters"
            raise ValueError(msg)
        target_level = len(heading_scope)
        if not 1 <= target_level <= 6:
            msg = "heading_scope must have between 1 and 6 '#' characters"
            raise ValueError(msg)

    # Track current section
    current_section = None
    current_start = 0
    current_lines = []

    for i, line in enumerate(lines):
        match = re.match(heading_pattern, line)

        if match:
            level = len(match.group(1))

            if current_section is not None and (target_level is None or level <= target_level):
                content = "\n".join(current_lines).strip()
                directives = extract_jupyter_directives(content, current_section["section_id"])
                sections.append(
                    DocumentSection(
                        section_id=current_section["section_id"],
                        title=current_section["title"],
                        level=current_section["level"],
                        start_line=current_start + 1,  # 1-indexed
                        end_line=i,  # 1-indexed (exclusive)
                        content=content,
                        directives=directives,
                    )
                )
                current_section = None
                current_lines = []

            if target_level is not None and level != target_level:
                if level > target_level and current_section is not None:
                    current_lines.append(line)
                continue

            title = match.group(2).strip()
            section_id = generate_section_id(title, i)

            current_section = {"section_id": section_id, "title": title, "level": level}
            current_start = i
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    # Save final section
    if current_section is not None:
        content = "\n".join(current_lines).strip()
        directives = extract_jupyter_directives(content, current_section["section_id"])
        sections.append(
            DocumentSection(
                section_id=current_section["section_id"],
                title=current_section["title"],
                level=current_section["level"],
                start_line=current_start + 1,
                end_line=len(lines),
                content=content,
                directives=directives,
            )
        )

    # If no sections found, treat whole document as one section
    if not sections:
        content = markdown_text.strip()
        directives = extract_jupyter_directives(content, "main")
        sections.append(
            DocumentSection(
                section_id="main",
                title="Main Document",
                level=1,
                start_line=1,
                end_line=len(lines),
                content=content,
                directives=directives,
            )
        )

    return sections


def generate_section_id(title: str, line_number: int) -> str:
    """
    Generate a section ID from the title.

    Examples:
        >>> generate_section_id("Chapter 1: Introduction", 0)
        '1-introduction'
        >>> generate_section_id("Main Results", 42)
        'main-results'
        >>> generate_section_id("Section 2.1: Preliminaries", 100)
        '2.1-preliminaries'
    """
    # Extract section number if present (e.g., "1", "2.1", "3.4.2")
    number_match = re.match(r"^(?:Chapter|Section)?\s*([0-9.]+)", title, re.IGNORECASE)

    if number_match:
        number = number_match.group(1)
        # Remove number prefix from title
        title_without_number = re.sub(
            r"^(?:Chapter|Section)?\s*[0-9.]+\s*:?\s*", "", title, flags=re.IGNORECASE
        )
        # Generate ID
        slug = re.sub(r"[^a-z0-9]+", "-", title_without_number.lower()).strip("-")
        return f"{number}-{slug}"
    # No number, just slugify the title
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_directive_summary(directives: list[DirectiveHint]) -> dict[str, int]:
    """
    Get a summary count of directives by type.

    Args:
        directives: List of directive hints

    Returns:
        Dictionary mapping directive type to count

    Examples:
        >>> hints = [
        ...     DirectiveHint("theorem", "thm-1", 1, 10, "...", "main"),
        ...     DirectiveHint("theorem", "thm-2", 11, 20, "...", "main"),
        ...     DirectiveHint("proof", "proof-1", 21, 30, "...", "main"),
        ... ]
        >>> get_directive_summary(hints)
        {'theorem': 2, 'proof': 1}
    """
    summary: dict[str, int] = {}
    for directive in directives:
        summary[directive.directive_type] = summary.get(directive.directive_type, 0) + 1
    return summary


def format_directive_hints_for_llm(directives: list[DirectiveHint]) -> str:
    """
    Format directive hints as a string for LLM prompts.

    This provides the LLM with structural hints about where mathematical
    entities are located in the document.

    Args:
        directives: List of directive hints

    Returns:
        Formatted string for LLM prompt

    Examples:
        >>> hints = [DirectiveHint("theorem", "thm-main", 10, 25, "...", "�2")]
        >>> print(format_directive_hints_for_llm(hints))
        Directive Hints:
        - [theorem] thm-main (lines 10-25)
    """
    if not directives:
        return "No directive hints available."

    lines = ["Directive Hints:"]
    for hint in directives:
        lines.append(
            f"- [{hint.directive_type}] {hint.label} (lines {hint.start_line}-{hint.end_line})"
        )
    return "\n".join(lines)
