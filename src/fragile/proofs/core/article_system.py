"""
Article System: Linking Mathematical Objects to Source Documents.

This module provides types for linking programmatic mathematical objects
(TheoremBox, ProofBox, etc.) to the markdown documents where they're defined.

The Article system enables:
- Bidirectional navigation between objects and documents
- Auto-generation of docs/glossary.md from structured data
- Document organization by tags and chapters
- Source location tracking for validation

Maps to Lean:
    structure SourceLocation where
      document_id : String
      file_path : String
      section : Option String
      directive_label : Option String
      ...

    structure Article where
      document_id : String
      title : String
      file_path : String
      chapter : Option Nat
      tags : List String
      contains_labels : List String
      ...
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from fragile.proofs.core.bibliography import Bibliography
    from fragile.proofs.sympy_integration.dual_representation import PaperContext


class SourceLocation(BaseModel):
    """
    Location of a mathematical object in markdown source documentation.

    Links programmatic objects (TheoremBox, ProofBox, DefinitionBox) to their exact
    location in markdown source files. Supports multiple levels of precision:
    - Document-level: document_id + file_path
    - Section-level: section reference
    - Directive-level: Jupyter Book directive label
    - Line-level: precise line_range in markdown file (most precise)

    This enables bidirectional navigation and source traceability for all
    mathematical entities in the framework.

    Examples:
        >>> # Directive-level location
        >>> loc = SourceLocation(
        ...     document_id="03_cloning",
        ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
        ...     directive_label="thm-keystone",
        ...     section="3.2",
        ... )
        >>> loc.get_full_url()
        'https://docs.example.com/03_cloning.html#thm-keystone'

        >>> # Line-level location (most precise)
        >>> loc2 = SourceLocation(
        ...     document_id="03_cloning",
        ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
        ...     line_range=(142, 158),
        ...     section="3.2.1",
        ... )
        >>> loc2.get_display_location()
        '03_cloning (line 142-158)'

    Maps to Lean:
        structure SourceLocation where
          document_id : String
          file_path : String
          section : Option String
          directive_label : Option String
          equation : Option String
          line_range : Option (Nat × Nat)
          url_fragment : Option String

          def get_full_url (loc : SourceLocation) (base_url : String) : String :=
            let doc_url := base_url ++ "/" ++ loc.document_id ++ ".html"
            match loc.url_fragment with
            | some frag => doc_url ++ frag
            | none => doc_url
    """

    model_config = ConfigDict(frozen=True)

    file_path: str = Field(
        ...,
        description="Path to source file relative to project root (e.g., 'docs/source/1_euclidean_gas/03_cloning.md')",
    )
    # Document identification
    chapter: str | None = Field(
        ...,
        description="High-level chapter/folder name (e.g., '1_euclidean_gas', '2_geometric_gas')",
    )
    document_id: str = Field(
        ...,
        pattern=r"^[0-9]{2}_[a-z_]+$",
        description="Document ID (e.g., '03_cloning', '11_geometric_gas')",
    )
    # Location within document
    section: str | None = Field(
        ...,
        description="Section reference using the markdown header format of X.Y.Z (e.g., '3.2.1')",
    )
    section_name: str | None = Field(
        ..., description="Human-readable section name (e.g., 'The Keystone Principle')"
    )
    directive_label: str | None = Field(
        ...,
        pattern=r"^[a-z][a-z0-9-]*$",
        description="Jupyter Book directive label (e.g., 'thm-keystone', 'def-walker'). "
        "If not present in the document, invent one based on object label.",
    )
    line_range: tuple[int, int] | None = Field(
        ...,
        description="Line range in markdown file (start_line, end_line, 1-indexed). "
        "Essential for grounding mathematical objects to their exact location in source documents. "
        "Example: (142, 158) means the object spans from line 142 to line 158 inclusive.",
    )
    equation: str | None = Field(
        None, description="Equation reference if applicable (e.g., 'Eq. (17)')"
    )

    # URL for documentation
    url_fragment: str | None = Field(
        None, description="URL fragment for Jupyter Book (e.g., '#thm-keystone')"
    )

    def get_full_url(self, base_url: str = "https://docs.example.com") -> str:
        """
        Generate full URL for this location.

        Args:
            base_url: Base URL for the documentation site

        Returns:
            Full URL to the object's location in documentation

        Maps to Lean:
            def get_full_url (loc : SourceLocation) (base_url : String) : String
        """
        doc_url = f"{base_url}/{self.document_id}.html"
        if self.url_fragment:
            return f"{doc_url}{self.url_fragment}"
        return doc_url

    def get_display_location(self) -> str:
        """
        Get human-readable location string.

        Returns:
            String like "03_cloning §3.2" or "03_cloning (line 142-158)"

        Maps to Lean:
            def get_display_location (loc : SourceLocation) : String
        """
        parts = [self.document_id]
        if self.section:
            parts.append(self.section)
        elif self.line_range:
            parts.append(f"(line {self.line_range[0]}-{self.line_range[1]})")
        elif self.directive_label:
            parts.append(f"({self.directive_label})")
        return " ".join(parts)


class Article(BaseModel):
    """
    A mathematical document in the framework.

    Represents a source document (markdown file) containing mathematical
    definitions, theorems, proofs, etc. Used for organization, filtering,
    and auto-generating docs/glossary.md.

    Examples:
        >>> article = Article(
        ...     document_id="03_cloning",
        ...     title="The Keystone Principle and Cloning",
        ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
        ...     chapter=1,
        ...     tags=["cloning", "measurement", "fitness"],
        ...     contains_labels=["thm-keystone", "def-fitness"],
        ... )
        >>> article.has_tag("cloning")
        True

    Maps to Lean:
        structure Article where
          document_id : String
          title : String
          file_path : String
          chapter : Option Nat
          section_number : Option String
          tags : List String
          version : String
          last_modified : Option DateTime
          abstract : Option String
          contains_labels : List String

          def has_tag (article : Article) (tag : String) : Bool :=
            tag ∈ article.tags
    """

    model_config = ConfigDict(frozen=True)

    # Identification
    document_id: str = Field(
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
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic version (e.g., '1.0.0', '2.1.0')",
    )
    last_modified: datetime | None = Field(None, description="Last modification timestamp")

    # Content
    abstract: str | None = Field(None, description="Brief abstract/summary of document content")
    contains_labels: list[str] = Field(
        default_factory=list,
        description="All mathematical object labels defined in this document",
    )

    # Enhanced Paper Processing (NEW - for PDF/paper extraction)
    bibliography: Bibliography | None = Field(
        None,
        description="The bibliography containing all references cited in this article. "
        "Essential for tracking theorem dependencies on external literature.",
    )

    global_context: PaperContext | None = Field(
        None,
        description="The global SymPy context for the paper, including symbols and assumptions "
        "defined for the whole document (e.g., 'h denotes a small parameter'). "
        "Built once by parsing introduction and notation sections.",
    )

    def has_tag(self, tag: str) -> bool:
        """
        Check if article has a specific tag.

        Args:
            tag: Tag to check

        Returns:
            True if article has the tag

        Maps to Lean:
            def has_tag (article : Article) (tag : String) : Bool :=
              tag ∈ article.tags
        """
        return tag in self.tags

    def has_label(self, label: str) -> bool:
        """
        Check if article contains a specific label.

        Args:
            label: Label to check

        Returns:
            True if article defines this label

        Maps to Lean:
            def has_label (article : Article) (label : String) : Bool :=
              label ∈ article.contains_labels
        """
        return label in self.contains_labels

    def get_summary(self) -> str:
        """
        Get a brief summary of the article.

        Returns:
            Summary string with title, chapter, and tag count

        Maps to Lean:
            def get_summary (article : Article) : String
        """
        parts = [self.title]
        if self.chapter is not None:
            parts.append(f"(Chapter {self.chapter})")
        if self.tags:
            parts.append(f"{len(self.tags)} tags")
        if self.contains_labels:
            parts.append(f"{len(self.contains_labels)} labels")
        return " - ".join(parts)
