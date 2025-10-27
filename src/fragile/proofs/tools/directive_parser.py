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

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DirectiveHint:
    """
    Structural hint extracted from Jupyter Book directive.

    Provides the LLM with reliable structural information about where
    mathematical entities are located in the document.
    """

    directive_type: str  # "definition", "theorem", "lemma", "proof", "axiom", etc.
    label: str  # The :label: value (e.g., "def-walker-state")
    start_line: int  # Line number where directive starts (1-indexed)
    end_line: int  # Line number where directive ends (1-indexed)
    content: str  # Raw content between directive markers
    section: str  # Section identifier where this appears


@dataclass
class DocumentSection:
    """
    A section of the document for parallel processing.

    Documents are split by headings to enable parallel extraction.
    """

    section_id: str  # Unique identifier (e.g., "§1-intro", "§2.1-main-results")
    title: str  # Section title
    level: int  # Heading level (1-6)
    start_line: int  # Line number where section starts (1-indexed)
    end_line: int  # Line number where section ends (1-indexed)
    content: str  # Full markdown content of the section
    directives: List[DirectiveHint]  # Directive hints found in this section


# =============================================================================
# DIRECTIVE EXTRACTION
# =============================================================================


def extract_jupyter_directives(markdown_text: str, section_id: str = "main") -> List[DirectiveHint]:
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
    lines = markdown_text.split('\n')

    # Pattern for directive start: :::{prf:TYPE} TITLE
    directive_start_pattern = r'^:::\{prf:(\w+)\}'
    # Pattern for label: :label: VALUE
    label_pattern = r'^:label:\s*(.+)$'
    # Pattern for directive end: :::
    directive_end_pattern = r'^:::$'

    i = 0
    while i < len(lines):
        line = lines[i]
        start_match = re.match(directive_start_pattern, line.strip())

        if start_match:
            directive_type = start_match.group(1)
            start_line = i + 1  # 1-indexed

            # Find label (should be in next few lines)
            label = None
            j = i + 1
            content_start = None
            while j < len(lines) and not lines[j].strip().startswith(':::'):
                label_match = re.match(label_pattern, lines[j].strip())
                if label_match:
                    label = label_match.group(1).strip()
                elif lines[j].strip() and not lines[j].strip().startswith(':'):
                    # First non-empty, non-field line is start of content
                    if content_start is None:
                        content_start = j
                j += 1

            # Find directive end
            end_line = None
            k = j
            while k < len(lines):
                if re.match(directive_end_pattern, lines[k].strip()):
                    end_line = k + 1  # 1-indexed
                    break
                k += 1

            if end_line is not None:
                # Extract content (between header and end marker)
                content_lines = lines[content_start:k] if content_start else []
                content = '\n'.join(content_lines).strip()

                directives.append(DirectiveHint(
                    directive_type=directive_type,
                    label=label or f"unlabeled-{directive_type}-{start_line}",
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    section=section_id
                ))

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


def split_into_sections(markdown_text: str, file_path: str = "unknown") -> List[DocumentSection]:
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
        - Prefixed with § and numbering if present
        - Examples: "§1-introduction", "§2.1-main-results", "preliminaries"

    Args:
        markdown_text: The full markdown document
        file_path: Path to the document (for metadata)

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
    lines = markdown_text.split('\n')

    # Pattern for markdown headings: # Title or ## Title or ### Title
    heading_pattern = r'^(#{1,6})\s+(.+)$'

    # Track current section
    current_section = None
    current_start = 0
    current_lines = []

    for i, line in enumerate(lines):
        match = re.match(heading_pattern, line)

        if match:
            # Save previous section if exists
            if current_section is not None:
                content = '\n'.join(current_lines).strip()
                directives = extract_jupyter_directives(content, current_section['section_id'])
                sections.append(DocumentSection(
                    section_id=current_section['section_id'],
                    title=current_section['title'],
                    level=current_section['level'],
                    start_line=current_start + 1,  # 1-indexed
                    end_line=i,  # 1-indexed (exclusive)
                    content=content,
                    directives=directives
                ))

            # Start new section
            level = len(match.group(1))
            title = match.group(2).strip()
            section_id = generate_section_id(title, i)

            current_section = {
                'section_id': section_id,
                'title': title,
                'level': level
            }
            current_start = i
            current_lines = []
        else:
            if current_section is not None:
                current_lines.append(line)

    # Save final section
    if current_section is not None:
        content = '\n'.join(current_lines).strip()
        directives = extract_jupyter_directives(content, current_section['section_id'])
        sections.append(DocumentSection(
            section_id=current_section['section_id'],
            title=current_section['title'],
            level=current_section['level'],
            start_line=current_start + 1,
            end_line=len(lines),
            content=content,
            directives=directives
        ))

    # If no sections found, treat whole document as one section
    if not sections:
        content = markdown_text.strip()
        directives = extract_jupyter_directives(content, "main")
        sections.append(DocumentSection(
            section_id="main",
            title="Main Document",
            level=1,
            start_line=1,
            end_line=len(lines),
            content=content,
            directives=directives
        ))

    return sections


def generate_section_id(title: str, line_number: int) -> str:
    """
    Generate a section ID from the title.

    Examples:
        >>> generate_section_id("Chapter 1: Introduction", 0)
        '§1-introduction'
        >>> generate_section_id("Main Results", 42)
        'main-results'
        >>> generate_section_id("Section 2.1: Preliminaries", 100)
        '§2.1-preliminaries'
    """
    # Extract section number if present (e.g., "1", "2.1", "3.4.2")
    number_match = re.match(r'^(?:Chapter|Section)?\s*([0-9.]+)', title, re.IGNORECASE)

    if number_match:
        number = number_match.group(1)
        # Remove number prefix from title
        title_without_number = re.sub(r'^(?:Chapter|Section)?\s*[0-9.]+\s*:?\s*', '', title, flags=re.IGNORECASE)
        # Generate ID
        slug = re.sub(r'[^a-z0-9]+', '-', title_without_number.lower()).strip('-')
        return f"§{number}-{slug}"
    else:
        # No number, just slugify the title
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        return slug


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_directive_summary(directives: List[DirectiveHint]) -> Dict[str, int]:
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
    summary: Dict[str, int] = {}
    for directive in directives:
        summary[directive.directive_type] = summary.get(directive.directive_type, 0) + 1
    return summary


def format_directive_hints_for_llm(directives: List[DirectiveHint]) -> str:
    """
    Format directive hints as a string for LLM prompts.

    This provides the LLM with structural hints about where mathematical
    entities are located in the document.

    Args:
        directives: List of directive hints

    Returns:
        Formatted string for LLM prompt

    Examples:
        >>> hints = [DirectiveHint("theorem", "thm-main", 10, 25, "...", "§2")]
        >>> print(format_directive_hints_for_llm(hints))
        Directive Hints:
        - [theorem] thm-main (lines 10-25)
    """
    if not directives:
        return "No directive hints available."

    lines = ["Directive Hints:"]
    for hint in directives:
        lines.append(f"- [{hint.directive_type}] {hint.label} (lines {hint.start_line}-{hint.end_line})")
    return '\n'.join(lines)
