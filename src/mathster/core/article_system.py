"""
Article System: Linking Mathematical Objects to Source Documents.

This module provides types for linking programmatic mathematical objects
(TheoremBox, ProofBox, etc.) to the markdown documents where they're defined.

The Article system enables:
- Bidirectional navigation between objects and documents
- Auto-generation of docs/glossary.md from structured data
- Document organization by tags and chapters
- Source location tracking for validation

Core types:
- TextLocation: Discontinuous line ranges for precise text location tracking
- SourceLocation: Links mathematical objects to their exact source location
- Article: Represents a mathematical document with metadata and organization
"""
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


LINE_BLOCK_SEPARATOR = "\n[...]\n"

class TextLocation(BaseModel):
    """
    Discontinuous line ranges in a document.

    A lightweight primitive for tracking text that spans multiple non-contiguous
    line ranges. Used by SourceLocation to represent precise text locations.

    Invariants:
        - All ranges are well-formed: start ≤ end, all line numbers ≥ 1
        - Ranges are sorted and non-overlapping
        - At least one range must be provided
    """

    model_config = ConfigDict(frozen=True)

    lines: list[tuple[int, int]] = Field(
        ...,
        description="List of (start_line, end_line) tuples, 1-indexed inclusive. "
        "Ranges must be well-formed (start ≤ end) and non-overlapping.",
    )

    @field_validator("lines")
    @classmethod
    def validate_ranges(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Validate that line ranges are well-formed and non-overlapping.

        Checks:
        - All ranges have start ≤ end
        - All line numbers ≥ 1
        - Ranges are sorted and non-overlapping

        Returns:
            Validated list of line ranges
        """
        if not v:
            raise ValueError("TextLocation must have at least one line range")

        # Check well-formedness
        for start, end in v:
            if start < 1:
                raise ValueError(f"Line numbers must be ≥ 1, got start={start}")
            if start > end:
                raise ValueError(f"Invalid range ({start}, {end}): start > end")

        # Check sorted and non-overlapping
        for i in range(len(v) - 1):
            curr_end = v[i][1]
            next_start = v[i + 1][0]
            if curr_end >= next_start:
                raise ValueError(
                    f"Overlapping ranges: {v[i]} and {v[i+1]} (ranges must be sorted and non-overlapping)"
                )

        return v

    @classmethod
    def from_single_range(cls, start: int, end: int) -> "TextLocation":
        """
        Create TextLocation from a single continuous range.

        Args:
            start: Starting line number (1-indexed)
            end: Ending line number (inclusive)

        Returns:
            TextLocation with single range

        Examples:
            >>> loc = TextLocation.from_single_range(10, 15)
            >>> loc.lines
            [(10, 15)]
        """
        return cls(lines=[(start, end)])

    def format_ranges(self) -> str:
        """
        Format line ranges as human-readable string.

        Returns:
            String like "10-15" or "10-15, 20-25, 30-32"

        Examples:
            >>> loc = TextLocation(lines=[(10, 15)])
            >>> loc.format_ranges()
            '10-15'

            >>> loc2 = TextLocation(lines=[(10, 15), (20, 25)])
            >>> loc2.format_ranges()
            '10-15, 20-25'
        """
        return ", ".join(f"{start}-{end}" for start, end in self.lines)

    def get_total_line_count(self) -> int:
        """
        Get total number of lines covered by all ranges.

        Returns:
            Sum of line counts across all ranges

        Examples:
            >>> loc = TextLocation(lines=[(10, 15), (20, 25)])
            >>> loc.get_total_line_count()
            12  # (15-10+1) + (25-20+1) = 6 + 6 = 12
        """
        return sum(end - start + 1 for start, end in self.lines)

    def contains_line(self, line_number: int) -> bool:
        """
        Check if a line number is contained in any range.

        Args:
            line_number: Line number to check (1-indexed)

        Returns:
            True if line is in any range, False otherwise

        Examples:
            >>> loc = TextLocation(lines=[(10, 15), (20, 25)])
            >>> loc.contains_line(12)
            True
            >>> loc.contains_line(18)
            False
        """
        return any(start <= line_number <= end for start, end in self.lines)


# Helper functions for extracting SourceLocation fields


def extract_volume_from_path(file_path: str) -> str | None:
    """
    Extract volume from file_path.

    Args:
        file_path: Path to markdown file

    Returns:
        Volume name (e.g., '1_euclidean_gas') or None if not found

    Examples:
        >>> extract_volume_from_path('docs/source/1_euclidean_gas/03_cloning.md')
        '1_euclidean_gas'
        >>> extract_volume_from_path('docs/source/2_geometric_gas/11_geometric.md')
        '2_geometric_gas'
        >>> extract_volume_from_path('other/path/file.md')
        None
    """
    pattern = r"(?:docs/source/)?(\d+_[a-z_]+)/"
    match = re.search(pattern, file_path)
    return match.group(1) if match else None


def extract_article_id_from_path(file_path: str) -> str:
    """
    Extract article_id from file_path.

    Args:
        file_path: Path to markdown file

    Returns:
        Article ID (e.g., '03_cloning')

    Raises:
        ValueError: If filename doesn't match pattern NN_name.md

    Examples:
        >>> extract_article_id_from_path('docs/source/1_euclidean_gas/03_cloning.md')
        '03_cloning'
        >>> extract_article_id_from_path('docs/source/2_geometric_gas/11_geometric_gas.md')
        '11_geometric_gas'
        >>> extract_article_id_from_path('invalid/path.md')
        Traceback (most recent call last):
            ...
        ValueError: Cannot extract article_id from path 'invalid/path.md'...
    """
    pattern = r"/(\d{2}_[a-z_]+)\.md$"
    match = re.search(pattern, file_path)
    if not match:
        raise ValueError(
            f"Cannot extract article_id from path '{file_path}'. "
            f"Expected pattern: '.../NN_name.md' where NN is 2 digits."
        )
    return match.group(1)


def extract_section_from_markdown(
    file_path: str, line_range: TextLocation
) -> tuple[str | None, str | None]:
    """
    Extract section number and name from markdown file.

    Finds the last header before the first line of line_range.

    Args:
        file_path: Path to markdown file
        line_range: TextLocation object with line ranges

    Returns:
        (section_number, section_name) tuple
        - section_number: e.g., "3.2.1" (or None if not found)
        - section_name: e.g., "The Keystone Principle" (or None if not found)

    Raises:
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> # Assuming line 625 is in section "#### 3.2.4 From Structural Error..."
        >>> extract_section_from_markdown(
        ...     "docs/source/1_euclidean_gas/03_cloning.md",
        ...     TextLocation.from_single_range(625, 635)
        ... )
        ('3.2.4', 'From Structural Error to Internal Swarm Variance')

        >>> # If no headers before the line
        >>> extract_section_from_markdown(
        ...     "docs/source/1_euclidean_gas/03_cloning.md",
        ...     TextLocation.from_single_range(1, 2)
        ... )
        (None, None)

    Algorithm:
        1. Read file up to first line of line_range
        2. Find all headers (lines starting with ##, ###, or ####)
        3. Take the last header before the line
        4. Parse with regex to extract section number and name
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    # Get the first line number from the line_range
    target_line = line_range.lines[0][0]

    # Read file and find headers before target line
    last_header = None
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if line_num >= target_line:
                break
            # Check if line is a header (##, ###, or ####)
            if line.startswith(("##", "###", "####")) and not line.startswith("#####"):
                last_header = line.strip()

    if not last_header:
        return (None, None)

    # Parse header: r'^(#{2,4})\s+(\d+(?:\.\d+)*)\.*\s+(.+)$'
    # Matches: ## 3. Section, ### 3.2 Name, #### 3.2.1. Name
    # Groups: (1) level, (2) number, (3) name
    pattern = r"^#{2,4}\s+(\d+(?:\.\d+)*)\.*\s+(.+)$"
    match = re.match(pattern, last_header)

    if match:
        section_number = match.group(1)
        section_name = match.group(2).strip()
        return (section_number, section_name)

    # Header without number (e.g., "## Introduction")
    # Extract just the text after the hashes
    text_match = re.match(r"^#{2,4}\s+(.+)$", last_header)
    if text_match:
        return (None, text_match.group(1).strip())

    return (None, None)


class SourceLocation(BaseModel):
    """
    Location of a mathematical object in markdown source documentation.

    Links programmatic objects (TheoremBox, ProofBox, DefinitionBox) to their exact
    location in markdown source files.

    All mathematical entities must provide:
    - file_path: Path to source markdown file
    - line_range: Precise line ranges where the entity is defined
    - label: Jupyter Book directive label for the entity or a new one if missing
    - article_id: Document identifier

    This enables bidirectional navigation and source traceability for all
    mathematical entities in the framework.
    """

    model_config = ConfigDict(frozen=True)

    # Required fields
    file_path: str = Field(
        ...,
        description="Path to source file relative to project root (e.g., 'docs/source/1_euclidean_gas/03_cloning.md')",
    )
    line_range: TextLocation = Field(
        ...,
        description="Text location in markdown file (potentially discontinuous line ranges). "
        "Essential for grounding mathematical objects to their exact location in source documents. "
        "Example: TextLocation(lines=[(142, 158)]) for continuous range, "
        "TextLocation(lines=[(10, 15), (20, 25)]) for discontinuous ranges.",
    )
    label: str = Field(
        ...,
        pattern=r"^[a-z][a-z0-9_-]*$",
        description="Jupyter Book directive label (e.g., 'thm-keystone', 'def-walker', 'section-introduction'). "
        "Required for all entities. Must be lowercase with letters, digits, underscores, and hyphens only. "
        "If a directive doesn't exist in the source document, "
        "create one following the pattern: {type}-{short-name} (e.g., 'thm-convergence', 'def-fitness', 'param_theta'). "
        "For section labels, use pattern: section-{normalized-title} (e.g., 'section-introduction', 'section-cloning').",
    )

    # Optional fields (can be None)
    volume: str | None = Field(
        None,
        description="High-level volume/folder name (e.g., '1_euclidean_gas', '2_geometric_gas').",
    )
    article_id: str = Field(
        ...,
        pattern=r"^[0-9]{2}_[a-z_]+$",
        description="Article ID (e.g., '03_cloning', '11_geometric_gas')",
    )
    section: str | None = Field(
        None,
        description="Section reference using the markdown header format of X.Y.Z (e.g., '3.2.1')",
    )
    section_name: str | None = Field(
        None, description="Human-readable section name (e.g., 'The Keystone Principle')"
    )

    @model_validator(mode="after")
    def auto_populate_optional_fields(self) -> "SourceLocation":
        """
        Always compute optional fields and validate user-provided values.

        This validator:
        1. Computes volume, section, and section_name from mandatory fields
        2. If user provided values that don't match computed values, emits warnings
        3. Keeps user-provided values (permissive policy) but alerts to inconsistencies
        4. Raises errors when computation fails (strict validation)

        Uses the same extraction logic as from_required_fields() to compute:
        - volume: from file_path pattern
        - section: from markdown headers
        - section_name: from markdown headers

        Raises:
            ValueError: If volume cannot be extracted from file_path
            FileNotFoundError: If markdown file doesn't exist for section extraction
            ValueError: If section cannot be extracted from markdown

        Warns:
            UserWarning: When user-provided values don't match computed values

        Returns:
            Self with optional fields populated (user values if provided, else computed)

        Examples:
            >>> # No warning - no user values provided
            >>> loc = SourceLocation(
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     line_range=TextLocation.from_single_range(100, 120),
            ...     label="thm-test",
            ...     article_id="03_cloning"
            ... )
            >>> # volume, section, section_name auto-populated from source

            >>> # Warning - user value doesn't match computed value
            >>> loc = SourceLocation(
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     line_range=TextLocation.from_single_range(100, 120),
            ...     label="thm-test",
            ...     article_id="03_cloning",
            ...     volume="2_geometric_gas"  # Wrong volume!
            ... )
            # UserWarning: SourceLocation field mismatch: User provided volume='2_geometric_gas'
            # but computed volume='1_euclidean_gas' from file_path. Using user value.
        """
        # Step 1: Always compute optional fields from mandatory fields
        computed_volume = extract_volume_from_path(self.file_path)
        if computed_volume is None:
            raise ValueError(
                f"Cannot extract volume from file_path: {self.file_path}. "
                "Path must match pattern 'N_volume_name' (e.g., '1_euclidean_gas')."
            )

        computed_section, computed_section_name = extract_section_from_markdown(
            self.file_path, self.line_range
        )

        # Allow None for section number (unnumbered sections like "Introduction")
        # But require section_name if there's any header before the target line
        if computed_section is None and computed_section_name is None:
            # Special case: no headers at all before target (e.g., preamble/chapter 0)
            # This is valid for document metadata
            pass  # Allow both to be None - will use object.__setattr__ below
        elif computed_section_name is None:
            # Header exists but couldn't parse name - this is an actual error
            raise ValueError(
                f"Cannot extract section_name from {self.file_path} at {self.line_range}. "
                "File contains unparseable section headers before the target line."
            )
        # Note: computed_section can be None for unnumbered sections - this is valid

        # Step 2: Compare user-provided values with computed values and warn on mismatches
        # Volume validation
        if self.volume is not None and self.volume != computed_volume:
            warnings.warn(
                f"SourceLocation field mismatch: User provided volume='{self.volume}' "
                f"but computed volume='{computed_volume}' from file_path. Using user value. "
                f"Location: {self.file_path}",
                UserWarning,
                stacklevel=2,
            )
        else:
            # No user value or matches computed - use computed value
            object.__setattr__(self, "volume", computed_volume)

        # Section validation
        if self.section is not None and self.section != computed_section:
            warnings.warn(
                f"SourceLocation field mismatch: User provided section='{self.section}' "
                f"but computed section='{computed_section}' from markdown headers at {self.line_range}. "
                f"Using user value. Location: {self.file_path}",
                UserWarning,
                stacklevel=2,
            )
        else:
            # No user value or matches computed - use computed value
            object.__setattr__(self, "section", computed_section)

        # Section name validation
        if self.section_name is not None and self.section_name != computed_section_name:
            warnings.warn(
                f"SourceLocation field mismatch: User provided section_name='{self.section_name}' "
                f"but computed section_name='{computed_section_name}' from markdown headers at {self.line_range}. "
                f"Using user value. Location: {self.file_path}",
                UserWarning,
                stacklevel=2,
            )
        else:
            # No user value or matches computed - use computed value
            object.__setattr__(self, "section_name", computed_section_name)

        return self

    def get_full_url(self, base_url: str = "https://docs.example.com") -> str:
        """
        Generate full URL for this location.

        Args:
            base_url: Base URL for the documentation site

        Returns:
            Full URL to the object's location in documentation, including the
            label as an anchor fragment (e.g., 'docs.com/03_cloning.html#thm-keystone').
        """
        doc_url = f"{base_url}/{self.article_id}.html"
        return f"{doc_url}#{self.label}"

    def get_display_location(self) -> str:
        """
        Get human-readable location string.

        Returns:
            String like "03_cloning 3.2.1" if section is provided, otherwise
            "03_cloning (lines 142-158)" showing the line ranges.

        Examples:
            >>> loc = SourceLocation(
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     line_range=TextLocation.from_single_range(142, 158),
            ...     label="thm-keystone",
            ...     article_id="03_cloning",
            ...     section="3.2.1"
            ... )
            >>> loc.get_display_location()
            '03_cloning 3.2.1'

            >>> loc2 = SourceLocation(
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     line_range=TextLocation.from_single_range(142, 158),
            ...     label="thm-keystone",
            ...     article_id="03_cloning"
            ... )
            >>> loc2.get_display_location()
            '03_cloning (lines 142-158)'
        """
        parts = [self.article_id]
        if self.section:
            parts.append(self.section)
        else:
            # line_range is required, so always available
            parts.append(f"(lines {self.line_range.format_ranges()})")
        return " ".join(parts)

    def get_full_text(self) -> str:
        """
        Load and return the full text referenced by this source location.

        Reads the file at file_path and extracts text from all line ranges
        defined in line_range. If there are multiple discontinuous ranges,
        the text blocks are joined using LINE_BLOCK_SEPARATOR.

        Returns:
            String containing the text from all line ranges, with multiple
            ranges separated by LINE_BLOCK_SEPARATOR.

        Raises:
            FileNotFoundError: If file_path doesn't exist
            ValueError: If line numbers exceed file length

        Examples:
            >>> loc = SourceLocation(
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     line_range=TextLocation.from_single_range(142, 158),
            ...     label="thm-keystone",
            ...     article_id="03_cloning",
            ... )
            >>> text = loc.get_full_text()
            >>> # Returns text from lines 142-158

            >>> # With discontinuous ranges
            >>> loc2 = SourceLocation(
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     line_range=TextLocation(lines=[(10, 15), (20, 25)]),
            ...     label="def-fitness",
            ...     article_id="03_cloning",
            ... )
            >>> text2 = loc2.get_full_text()
            >>> # Returns text from lines 10-15, then LINE_BLOCK_SEPARATOR, then lines 20-25
        """

        # Validate file exists
        file_path_obj = Path(self.file_path)
        if not file_path_obj.exists():
            msg = f"File not found: {self.file_path}"
            raise FileNotFoundError(msg)

        # Read all lines
        with file_path_obj.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        # Extract text chunks for each line range
        text_chunks = []
        for start, end in self.line_range.lines:
            # Validate line numbers
            if start < 1 or end > len(lines):
                msg = f"Line range ({start}, {end}) exceeds file length {len(lines)}"
                raise ValueError(msg)

            # Extract lines (1-indexed, inclusive)
            chunk_lines = lines[start - 1 : end]
            chunk_text = "".join(chunk_lines).rstrip()
            text_chunks.append(chunk_text)

        # Join chunks with separator
        return f"{LINE_BLOCK_SEPARATOR}".join(text_chunks)

    @classmethod
    def from_required_fields(
        cls,
        file_path: str,
        line_range: TextLocation,
        label: str,
    ) -> "SourceLocation":
        """
        Create SourceLocation with auto-extracted optional fields.

        Extracts optional fields from required fields:
        - volume: from file_path pattern
        - article_id: from filename
        - section: from markdown headers
        - section_name: from markdown headers

        Args:
            file_path: Path to markdown file
            line_range: TextLocation object
            label: Directive label string

        Returns:
            SourceLocation with all fields populated

        Raises:
            ValueError: If article_id cannot be extracted from file_path
            FileNotFoundError: If file doesn't exist (when extracting section)

        Examples:
            >>> loc = SourceLocation.from_required_fields(
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     line_range=TextLocation.from_single_range(621, 635),
            ...     label="lem-sx-implies-variance"
            ... )
            >>> loc.volume
            '1_euclidean_gas'
            >>> loc.article_id
            '03_cloning'
            >>> loc.section
            '3.2.4'
            >>> loc.section_name
            'From Structural Error to Internal Swarm Variance'

            >>> # Minimal example without optional fields extracted
            >>> loc2 = SourceLocation.from_required_fields(
            ...     file_path="other/path/03_test.md",
            ...     line_range=TextLocation.from_single_range(10, 20),
            ...     label="def-example"
            ... )
            >>> loc2.volume is None
            True
            >>> loc2.article_id
            '03_test'
        """
        # Extract volume (optional, None if not found)
        volume = extract_volume_from_path(file_path)

        # Extract article_id (required, raises ValueError if not found)
        article_id = extract_article_id_from_path(file_path)

        # Extract section and section_name (optional, None if not found)
        section, section_name = extract_section_from_markdown(file_path, line_range)

        return cls(
            file_path=file_path,
            line_range=line_range,
            label=label,
            volume=volume,
            article_id=article_id,
            section=section,
            section_name=section_name,
        )


class Article(BaseModel):
    """
    A mathematical document in the framework.

    Represents a source document (markdown file) containing mathematical
    definitions, theorems, mathster, etc. Used for organization, filtering,
    and auto-generating docs/glossary.md.

    Examples:
        >>> article = Article(
        ...     article_id="03_cloning",
        ...     title="The Keystone Principle and Cloning",
        ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
        ...     chapter=1,
        ...     tags=["cloning", "measurement", "fitness"],
        ...     contains_labels=["thm-keystone", "def-fitness"],
        ... )
        >>> article.has_tag("cloning")
        True
    """

    model_config = ConfigDict(frozen=True)

    # Identification
    article_id: str = Field(
        ...,
        pattern=r"^[0-9]{2}_[a-z_]+$",
        description="Unique document ID (e.g., '03_cloning', '11_geometric_gas')",
    )
    title: str = Field(..., description="Document title")
    file_path: str = Field(..., description="Path to markdown file relative to project root")

    # Organization
    chapter: int | None = Field(
        None,
        description="Chapter number in the framework (1 = Euclidean Gas, 2 = Geometric Gas)",
    )
    section_number: str | None = Field(
        None, description="Section number within chapter (e.g., '2.3', '1.3.2')"
    )

    # Categorization
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering/categorizing (e.g., ['cloning', 'convergence', 'mean-field'])",
    )

    # Versioning
    version: str = Field(default="0.1.0", description="Document version")
    last_modified: datetime | None = Field(
        None, description="Last modification timestamp (auto-populated)"
    )

    # Content metadata
    abstract: str | None = Field(None, description="Brief abstract or summary of document")
    contains_labels: list[str] = Field(
        default_factory=list,
        description="List of mathematical object labels defined in this document "
        "(e.g., ['thm-keystone', 'def-fitness'])",
    )

    def has_tag(self, tag: str) -> bool:
        """
        Check if document has a specific tag.

        Args:
            tag: Tag to check for

        Returns:
            True if document has the tag, False otherwise

        Examples:
            >>> article = Article(
            ...     article_id="03_cloning",
            ...     title="Cloning",
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     tags=["cloning", "measurement"],
            ... )
            >>> article.has_tag("cloning")
            True
            >>> article.has_tag("convergence")
            False
        """
        return tag in self.tags

    def has_label(self, label: str) -> bool:
        """
        Check if document contains a specific mathematical object label.

        Args:
            label: Label to check for (e.g., "thm-keystone")

        Returns:
            True if document contains the label, False otherwise

        Examples:
            >>> article = Article(
            ...     article_id="03_cloning",
            ...     title="Cloning",
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     contains_labels=["thm-keystone", "def-fitness"],
            ... )
            >>> article.has_label("thm-keystone")
            True
            >>> article.has_label("thm-unknown")
            False
        """
        return label in self.contains_labels
