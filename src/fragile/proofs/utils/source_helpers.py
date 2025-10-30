"""
Source Location Helpers: Utilities for creating SourceLocation objects.

Provides convenience functions for creating SourceLocation objects from
various inputs (Jupyter Book labels, markdown lines, etc.).

Maps to Lean:
    namespace SourceLocationBuilder
      def from_jupyter_directive : String → String → String → Option String → SourceLocation
      def from_markdown_location : String → String → Nat → Nat → Option String → SourceLocation
      ...
    end SourceLocationBuilder
"""

from fragile.proofs.core.article_system import SourceLocation


class SourceLocationBuilder:
    """
    Helper to create SourceLocation objects from various inputs.

    Provides static methods for common source location creation patterns:
    - From Jupyter Book directive labels
    - From markdown line ranges
    - From section references

    Examples:
        >>> loc = SourceLocationBuilder.from_jupyter_directive(
        ...     document_id="03_cloning",
        ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
        ...     directive_label="thm-keystone",
        ...     section="§3.2",
        ... )
        >>> loc.directive_label
        'thm-keystone'

    Maps to Lean:
        namespace SourceLocationBuilder
          def from_jupyter_directive : ... → SourceLocation
          def from_markdown_location : ... → SourceLocation
        end SourceLocationBuilder
    """

    @staticmethod
    def from_jupyter_directive(
        document_id: str,
        file_path: str,
        directive_label: str,
        section: str | None = None,
        equation: str | None = None,
    ) -> SourceLocation:
        """
        Create SourceLocation from Jupyter Book directive.

        This is the most common case: linking to a {prf:theorem}, {prf:lemma},
        or other Jupyter Book directive in the markdown docs.

        Args:
            document_id: Document ID (e.g., "03_cloning")
            file_path: Path to markdown file
            directive_label: Jupyter Book label (e.g., "thm-keystone")
            section: Optional section reference (e.g., "§3.2")
            equation: Optional equation reference

        Returns:
            SourceLocation with url_fragment set for Jupyter Book

        Examples:
            >>> loc = SourceLocationBuilder.from_jupyter_directive(
            ...     document_id="03_cloning",
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     directive_label="thm-keystone",
            ...     section="§3.2",
            ... )
            >>> loc.get_full_url()
            'https://docs.example.com/03_cloning.html#thm-keystone'

        Maps to Lean:
            def from_jupyter_directive
              (document_id : String)
              (file_path : String)
              (directive_label : String)
              (section : Option String)
              (equation : Option String)
              : SourceLocation
        """
        return SourceLocation(
            document_id=document_id,
            file_path=file_path,
            directive_label=directive_label,
            section=section,
            equation=equation,
            url_fragment=f"#{directive_label}",
        )

    @staticmethod
    def from_markdown_location(
        document_id: str,
        file_path: str,
        start_line: int,
        end_line: int,
        section: str | None = None,
    ) -> SourceLocation:
        """
        Create SourceLocation from markdown line range (RECOMMENDED for precise grounding).

        This is the most precise way to ground mathematical objects to their source.
        Line numbers enable exact traceability and validation against markdown documents.
        Use this when extracting objects from markdown files programmatically.

        Args:
            document_id: Document ID
            file_path: Path to markdown file
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (inclusive)
            section: Optional section reference

        Returns:
            SourceLocation with line_range set

        Examples:
            >>> loc = SourceLocationBuilder.from_markdown_location(
            ...     document_id="03_cloning",
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     start_line=142,
            ...     end_line=158,
            ...     section="§3.2.1",
            ... )
            >>> loc.get_display_location()
            '03_cloning (line 142-158)'

        Maps to Lean:
            def from_markdown_location
              (document_id : String)
              (file_path : String)
              (start_line : Nat)
              (end_line : Nat)
              (section : Option String)
              : SourceLocation
        """
        return SourceLocation(
            document_id=document_id,
            file_path=file_path,
            line_range=(start_line, end_line),
            section=section,
        )

    @staticmethod
    def from_section(
        document_id: str, file_path: str, section: str, equation: str | None = None
    ) -> SourceLocation:
        """
        Create SourceLocation from section reference only.

        Use when you only know the section but not the specific directive or line.
        Less specific than other methods, but still useful for linking.

        Args:
            document_id: Document ID
            file_path: Path to markdown file
            section: Section reference (e.g., "§3.2", "Section 3: Cloning")
            equation: Optional equation reference

        Returns:
            SourceLocation with section set

        Examples:
            >>> loc = SourceLocationBuilder.from_section(
            ...     document_id="03_cloning",
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ...     section="§3.2: The Keystone Principle",
            ... )
            >>> loc.get_display_location()
            '03_cloning §3.2: The Keystone Principle'

        Maps to Lean:
            def from_section
              (document_id : String)
              (file_path : String)
              (section : String)
              (equation : Option String)
              : SourceLocation
        """
        return SourceLocation(
            document_id=document_id, file_path=file_path, section=section, equation=equation
        )

    @staticmethod
    def minimal(document_id: str, file_path: str) -> SourceLocation:
        """
        Create minimal SourceLocation with just document_id and file_path.

        Use when you know which document defines an object, but not the
        specific location within that document. Better than no source at all.

        Args:
            document_id: Document ID
            file_path: Path to markdown file

        Returns:
            Minimal SourceLocation

        Examples:
            >>> loc = SourceLocationBuilder.minimal(
            ...     document_id="03_cloning",
            ...     file_path="docs/source/1_euclidean_gas/03_cloning.md",
            ... )
            >>> loc.get_display_location()
            '03_cloning'

        Maps to Lean:
            def minimal (document_id : String) (file_path : String) : SourceLocation
        """
        return SourceLocation(document_id=document_id, file_path=file_path)

    @staticmethod
    def from_raw_entity(
        entity_data: dict,
        document_id: str,
        file_path: str,
        line_range: tuple | None = None,
    ) -> SourceLocation:
        """
        Create SourceLocation from raw entity with automatic fallback strategy.

        This method implements a multi-level fallback strategy:
        1. Line range (if provided) - MOST PRECISE
        2. Directive label (extracted from entity) - PRECISE
        3. Section (from entity's source_section field) - LESS PRECISE
        4. Minimal (just document_id + file_path) - LEAST PRECISE

        Use this method when enriching raw entity JSON files with source locations.

        Args:
            entity_data: Dictionary containing raw entity data (RawTheorem, RawDefinition, etc.)
            document_id: Document ID (e.g., "03_cloning")
            file_path: Path to markdown file
            line_range: Optional tuple of (start_line, end_line) if already found

        Returns:
            SourceLocation with best available precision

        Examples:
            >>> # With line range (most precise)
            >>> entity = {"source_section": "§3.2"}
            >>> loc = SourceLocationBuilder.from_raw_entity(
            ...     entity, "03_cloning", "docs/.../03_cloning.md", line_range=(142, 158)
            ... )
            >>> loc.line_range
            (142, 158)

            >>> # With directive label (precise)
            >>> entity = {"label_text": "thm-keystone", "source_section": "§3.2"}
            >>> loc = SourceLocationBuilder.from_raw_entity(
            ...     entity, "03_cloning", "docs/.../03_cloning.md"
            ... )
            >>> loc.directive_label
            'thm-keystone'

            >>> # Section fallback
            >>> entity = {"source_section": "§3.2"}
            >>> loc = SourceLocationBuilder.from_raw_entity(
            ...     entity, "03_cloning", "docs/.../03_cloning.md"
            ... )
            >>> loc.section
            '§3.2'

        Maps to Lean:
            def from_raw_entity
              (entity : Json)
              (document_id : String)
              (file_path : String)
              (line_range : Option (Nat × Nat))
              : SourceLocation
        """
        # Extract section from entity (common field for all raw entities)
        section = entity_data.get("source_section")

        # STRATEGY 1: Line range (most precise)
        if line_range:
            return SourceLocationBuilder.from_markdown_location(
                document_id=document_id,
                file_path=file_path,
                start_line=line_range[0],
                end_line=line_range[1],
                section=section,
            )

        # STRATEGY 2: Directive label (precise)
        # Check for label_text field (used in refined/enriched data)
        directive_label = entity_data.get("label_text")
        if directive_label and isinstance(directive_label, str) and "-" in directive_label:
            return SourceLocationBuilder.from_jupyter_directive(
                document_id=document_id,
                file_path=file_path,
                directive_label=directive_label,
                section=section,
            )

        # STRATEGY 3: Section fallback
        if section:
            return SourceLocationBuilder.from_section(
                document_id=document_id,
                file_path=file_path,
                section=section,
            )

        # STRATEGY 4: Minimal fallback
        return SourceLocationBuilder.minimal(
            document_id=document_id,
            file_path=file_path,
        )

    @staticmethod
    def with_fallback(
        document_id: str,
        file_path: str,
        line_range: tuple | None = None,
        directive_label: str | None = None,
        section: str | None = None,
        equation: str | None = None,
    ) -> SourceLocation:
        """
        Create SourceLocation with automatic fallback to best available precision.

        Convenience method that selects the appropriate builder based on
        which fields are provided. Tries in order: line_range → directive → section → minimal.

        Args:
            document_id: Document ID
            file_path: Path to markdown file
            line_range: Optional (start_line, end_line)
            directive_label: Optional Jupyter Book label
            section: Optional section reference
            equation: Optional equation reference

        Returns:
            SourceLocation with best available precision

        Examples:
            >>> # All fields provided (uses line range)
            >>> loc = SourceLocationBuilder.with_fallback(
            ...     "03_cloning",
            ...     "docs/.../03_cloning.md",
            ...     line_range=(142, 158),
            ...     directive_label="thm-keystone",
            ...     section="§3.2",
            ... )
            >>> loc.line_range
            (142, 158)

            >>> # Only section provided
            >>> loc = SourceLocationBuilder.with_fallback(
            ...     "03_cloning", "docs/.../03_cloning.md", section="§3.2"
            ... )
            >>> loc.section
            '§3.2'

        Maps to Lean:
            def with_fallback
              (document_id : String)
              (file_path : String)
              (line_range : Option (Nat × Nat))
              (directive_label : Option String)
              (section : Option String)
              (equation : Option String)
              : SourceLocation
        """
        # Line range takes precedence (most precise)
        if line_range:
            return SourceLocationBuilder.from_markdown_location(
                document_id=document_id,
                file_path=file_path,
                start_line=line_range[0],
                end_line=line_range[1],
                section=section,
            )

        # Directive label (precise)
        if directive_label:
            return SourceLocationBuilder.from_jupyter_directive(
                document_id=document_id,
                file_path=file_path,
                directive_label=directive_label,
                section=section,
                equation=equation,
            )

        # Section fallback
        if section:
            return SourceLocationBuilder.from_section(
                document_id=document_id,
                file_path=file_path,
                section=section,
                equation=equation,
            )

        # Minimal fallback
        return SourceLocationBuilder.minimal(
            document_id=document_id,
            file_path=file_path,
        )
