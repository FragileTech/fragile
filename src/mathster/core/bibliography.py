"""
Bibliography System: Citations and Reference Management.

This module provides data structures for handling bibliographic references
in mathematical research papers, including BibTeX entries and in-text citations.

Maps to Lean:
    namespace Bibliography
      structure Citation where
        key : String
        authors : List String
        title : Option String
        journal : Option String
        year : Option Nat
        bibtex_entry : Option String

      structure Bibliography where
        entries : HashMap String Citation

      def get_citation (bib : Bibliography) (key : String) : Option Citation :=
        bib.entries.find? key
    end Bibliography
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """
    Represents a single bibliographic entry, typically from a BibTeX source.

    This class models an individual reference that can be cited in a paper.
    It stores both parsed metadata (authors, title, etc.) and the original
    BibTeX entry for perfect reconstruction.

    Examples:
        >>> citation = Citation(
        ...     key="han2016",
        ...     authors=["Han, R.", "Slepčev, D."],
        ...     title="Stochastic dynamics on hypergraphs",
        ...     journal="Journal of Functional Analysis",
        ...     year=2016,
        ... )
        >>> citation.key
        'han2016'

    Maps to Lean:
        structure Citation where
          key : String
          authors : List String
          title : Option String
          journal : Option String
          year : Option Nat
          bibtex_entry : Option String
    """

    model_config = ConfigDict(frozen=True)

    # The key used for in-text citations, e.g., 'han2016' or '[16]'
    key: str = Field(..., description="The citation key used to reference this entry in the text.")

    # Parsed fields for easier access
    authors: list[str] = Field(
        default_factory=list, description="List of author names as they appear in the citation."
    )

    title: str | None = Field(None, description="The title of the cited work.")

    journal: str | None = Field(
        None, description="The journal or venue where the work was published."
    )

    year: int | None = Field(
        None,
        description="The publication year.",
        ge=1900,  # Sanity check: no papers before 1900
        le=2100,  # Sanity check: no papers from far future
    )

    # Store the original entry for perfect reconstruction
    bibtex_entry: str | None = Field(
        None,
        description="The full, original BibTeX entry for the reference. "
        "Preserves all fields and formatting from the source.",
    )

    # Additional optional fields for richer metadata
    volume: str | None = Field(None, description="Journal volume number.")
    pages: str | None = Field(None, description="Page range, e.g., '123-145'.")
    doi: str | None = Field(None, description="Digital Object Identifier.")
    url: str | None = Field(None, description="URL to the paper.")
    arxiv_id: str | None = Field(None, description="arXiv identifier, e.g., '1234.5678'.")


class Bibliography(BaseModel):
    """
    A collection of all bibliographic citations for a given article, indexed by key.

    This class serves as a container and lookup system for all references
    cited in a mathematical paper. Citations are indexed by their key for
    fast retrieval during proof verification and cross-referencing.

    Examples:
        >>> bib = Bibliography(
        ...     entries={
        ...         "han2016": Citation(
        ...             key="han2016",
        ...             authors=["Han, R.", "Slepčev, D."],
        ...             title="Stochastic dynamics on hypergraphs",
        ...             year=2016,
        ...         )
        ...     }
        ... )
        >>> citation = bib.get_citation("han2016")
        >>> citation.year
        2016

    Maps to Lean:
        structure Bibliography where
          entries : HashMap String Citation

        def get_citation (bib : Bibliography) (key : String) : Option Citation :=
          bib.entries.find? key
    """

    model_config = ConfigDict(frozen=True)

    entries: dict[str, Citation] = Field(
        default_factory=dict,
        description="A mapping from citation key to Citation object. "
        "Keys should match the 'key' field in each Citation.",
    )

    # Pure function: Get citation by key
    def get_citation(self, key: str) -> Citation | None:
        """
        Total function: Retrieves a citation by its key.

        Args:
            key: The citation key to look up (e.g., "han2016")

        Returns:
            The Citation object if found, None otherwise

        Maps to Lean:
            def get_citation (bib : Bibliography) (key : String) : Option Citation :=
              bib.entries.find? key
        """
        return self.entries.get(key)

    # Pure function: Check if citation exists
    def has_citation(self, key: str) -> bool:
        """
        Pure function: Check if a citation key exists in the bibliography.

        Args:
            key: The citation key to check

        Returns:
            True if the citation exists, False otherwise

        Maps to Lean:
            def has_citation (bib : Bibliography) (key : String) : Bool :=
              bib.entries.contains key
        """
        return key in self.entries

    # Pure function: Add citation (returns new Bibliography)
    def add_citation(self, citation: Citation) -> Bibliography:
        """
        Pure function: Add a citation to the bibliography (immutable update).

        Args:
            citation: The Citation object to add

        Returns:
            A new Bibliography with the citation added

        Maps to Lean:
            def add_citation (bib : Bibliography) (cit : Citation) : Bibliography :=
              { bib with entries := bib.entries.insert cit.key cit }
        """
        new_entries = dict(self.entries)
        new_entries[citation.key] = citation
        return self.model_copy(update={"entries": new_entries})

    # Pure function: Get all citation keys
    def get_all_keys(self) -> list[str]:
        """
        Pure function: Get a list of all citation keys in the bibliography.

        Returns:
            List of citation keys sorted alphabetically

        Maps to Lean:
            def get_all_keys (bib : Bibliography) : List String :=
              bib.entries.keys.toList.sorted
        """
        return sorted(self.entries.keys())
