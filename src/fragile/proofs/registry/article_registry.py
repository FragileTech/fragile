"""
ArticleRegistry: Central registry for mathematical documents.

Provides singleton registry for managing Article objects, enabling:
- Registration and retrieval of articles by document_id
- Querying by tags, chapters, and labels
- Generation of docs/glossary.md from structured data
- Export/import for persistence

Maps to Lean:
    structure ArticleRegistry where
      articles : HashMap String Article
      label_to_document : HashMap String String
      tag_index : HashMap String (Set String)
      ...

      def register_article (reg : ArticleRegistry) (article : Article) : ArticleRegistry
      def get_article (reg : ArticleRegistry) (doc_id : String) : Option Article
      ...
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional

from fragile.proofs.core.article_system import Article


class ArticleRegistry:
    """
    Central registry for mathematical documents.

    Singleton pattern ensures single source of truth for article metadata.
    Provides querying, indexing, and glossary generation capabilities.

    Examples:
        >>> registry = get_article_registry()
        >>> registry.register_article(article)
        >>> found = registry.get_article_for_label("thm-keystone")
        >>> glossary = registry.generate_glossary_markdown()

    Maps to Lean:
        structure ArticleRegistry where
          articles : HashMap String Article
          label_to_document : HashMap String String
          tag_index : HashMap String (Set String)

          def register_article : ArticleRegistry → Article → ArticleRegistry
          def get_article : ArticleRegistry → String → Option Article
          def get_article_for_label : ArticleRegistry → String → Option Article
          def get_articles_by_tag : ArticleRegistry → String → List Article
          def generate_glossary_markdown : ArticleRegistry → String
    """

    _instance: Optional["ArticleRegistry"] = None

    def __new__(cls) -> "ArticleRegistry":
        """Singleton pattern: only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize registry data structures."""
        self._articles: dict[str, Article] = {}
        self._label_to_document: dict[str, str] = {}  # label → document_id
        self._tag_index: dict[str, set[str]] = {}  # tag → {document_ids}

    # ========================================================================
    # REGISTRATION
    # ========================================================================

    def register_article(self, article: Article) -> None:
        """
        Register an article and index its labels and tags.

        Args:
            article: Article to register

        Maps to Lean:
            def register_article (reg : ArticleRegistry) (article : Article) : ArticleRegistry
        """
        self._articles[article.document_id] = article

        # Index labels
        for label in article.contains_labels:
            self._label_to_document[label] = article.document_id

        # Index tags
        for tag in article.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(article.document_id)

    # ========================================================================
    # QUERIES
    # ========================================================================

    def get_article(self, document_id: str) -> Article | None:
        """
        Get article by document_id.

        Args:
            document_id: Document identifier (e.g., "03_cloning")

        Returns:
            Article if found, None otherwise

        Maps to Lean:
            def get_article (reg : ArticleRegistry) (doc_id : String) : Option Article
        """
        return self._articles.get(document_id)

    def get_article_for_label(self, label: str) -> Article | None:
        """
        Get article containing a given label.

        Args:
            label: Mathematical object label (e.g., "thm-keystone")

        Returns:
            Article defining this label, None if not found

        Maps to Lean:
            def get_article_for_label (reg : ArticleRegistry) (label : String) : Option Article
        """
        document_id = self._label_to_document.get(label)
        if document_id:
            return self._articles.get(document_id)
        return None

    def get_document_id_for_label(self, label: str) -> str | None:
        """
        Get document_id for a label (fast lookup without full Article).

        Args:
            label: Mathematical object label

        Returns:
            Document ID if found, None otherwise

        Maps to Lean:
            def get_document_id_for_label (reg : ArticleRegistry) (label : String) : Option String
        """
        return self._label_to_document.get(label)

    def get_articles_by_tag(self, tag: str) -> list[Article]:
        """
        Get all articles with a given tag.

        Args:
            tag: Tag to filter by (e.g., "cloning", "convergence")

        Returns:
            List of articles with this tag

        Maps to Lean:
            def get_articles_by_tag (reg : ArticleRegistry) (tag : String) : List Article
        """
        document_ids = self._tag_index.get(tag, set())
        return [self._articles[doc_id] for doc_id in document_ids if doc_id in self._articles]

    def get_articles_by_chapter(self, chapter: int) -> list[Article]:
        """
        Get all articles in a chapter.

        Args:
            chapter: Chapter number (1 = Euclidean Gas, 2 = Geometric Gas)

        Returns:
            List of articles in this chapter

        Maps to Lean:
            def get_articles_by_chapter (reg : ArticleRegistry) (chapter : Nat) : List Article
        """
        return [article for article in self._articles.values() if article.chapter == chapter]

    def get_all_articles(self) -> list[Article]:
        """
        Get all registered articles.

        Returns:
            List of all articles

        Maps to Lean:
            def get_all_articles (reg : ArticleRegistry) : List Article
        """
        return list(self._articles.values())

    def get_all_tags(self) -> list[str]:
        """
        Get all unique tags across all articles.

        Returns:
            Sorted list of all tags

        Maps to Lean:
            def get_all_tags (reg : ArticleRegistry) : List String
        """
        return sorted(self._tag_index.keys())

    def has_label(self, label: str) -> bool:
        """
        Check if a label is registered in any article.

        Args:
            label: Label to check

        Returns:
            True if label exists in registry

        Maps to Lean:
            def has_label (reg : ArticleRegistry) (label : String) : Bool
        """
        return label in self._label_to_document

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with counts and distributions

        Maps to Lean:
            def get_statistics (reg : ArticleRegistry) : Statistics
        """
        return {
            "total_articles": len(self._articles),
            "total_labels": len(self._label_to_document),
            "total_tags": len(self._tag_index),
            "by_chapter": self._get_chapter_stats(),
            "by_tag": {tag: len(docs) for tag, docs in self._tag_index.items()},
        }

    def _get_chapter_stats(self) -> dict[int, int]:
        """Get article count by chapter."""
        stats: dict[int, int] = {}
        for article in self._articles.values():
            if article.chapter is not None:
                stats[article.chapter] = stats.get(article.chapter, 0) + 1
        return stats

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def export_to_json(self, path: Path) -> None:
        """
        Export registry to JSON file.

        Args:
            path: Path to save JSON file

        Maps to Lean:
            def export_to_json (reg : ArticleRegistry) (path : FilePath) : IO Unit
        """
        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_articles": len(self._articles),
                "total_labels": len(self._label_to_document),
            },
            "articles": [article.model_dump() for article in self._articles.values()],
            "label_index": self._label_to_document,
        }
        path.write_text(json.dumps(data, indent=2))

    def load_from_json(self, path: Path) -> None:
        """
        Load registry from JSON file.

        Args:
            path: Path to JSON file

        Maps to Lean:
            def load_from_json (reg : ArticleRegistry) (path : FilePath) : IO ArticleRegistry
        """
        data = json.loads(path.read_text())

        # Clear existing data
        self._initialize()

        # Load articles
        for article_data in data.get("articles", []):
            article = Article(**article_data)
            self.register_article(article)

    # ========================================================================
    # GLOSSARY GENERATION
    # ========================================================================

    def generate_glossary_markdown(self) -> str:
        """
        Generate glossary.md content from registry.

        Organizes articles by chapter, then by document_id within each chapter.
        For each article, lists all contained labels.

        Returns:
            Markdown formatted glossary

        Maps to Lean:
            def generate_glossary_markdown (reg : ArticleRegistry) : String
        """
        lines = [
            "# Mathematical Glossary",
            "",
            "Auto-generated from Article Registry",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total: {len(self._articles)} documents, {len(self._label_to_document)} labels",
            "",
        ]

        # Group by chapter
        by_chapter: dict[int, list[Article]] = {}
        uncategorized: list[Article] = []

        for article in self._articles.values():
            if article.chapter is None:
                uncategorized.append(article)
            else:
                if article.chapter not in by_chapter:
                    by_chapter[article.chapter] = []
                by_chapter[article.chapter].append(article)

        # Write by chapter
        for chapter in sorted(by_chapter.keys()):
            chapter_name = (
                "Euclidean Gas"
                if chapter == 1
                else "Geometric Gas"
                if chapter == 2
                else f"Chapter {chapter}"
            )
            lines.extend((f"## Chapter {chapter}: {chapter_name}", ""))

            articles = sorted(by_chapter[chapter], key=lambda a: a.document_id)
            for article in articles:
                lines.extend(self._format_article_entry(article))

        # Write uncategorized
        if uncategorized:
            lines.extend(("## Uncategorized", ""))
            for article in sorted(uncategorized, key=lambda a: a.document_id):
                lines.extend(self._format_article_entry(article))

        return "\n".join(lines)

    def _format_article_entry(self, article: Article) -> list[str]:
        """Format a single article entry for the glossary."""
        lines = [f"### {article.title}"]
        lines.extend((
            f"**Document ID**: `{article.document_id}`",
            f"**File**: `{article.file_path}`",
        ))

        if article.tags:
            tags_str = ", ".join(f"`{tag}`" for tag in sorted(article.tags))
            lines.append(f"**Tags**: {tags_str}")

        if article.abstract:
            lines.append(f"**Abstract**: {article.abstract}")

        lines.append("")

        if article.contains_labels:
            lines.append("**Defined labels**:")
            for label in sorted(article.contains_labels):
                # Try to categorize by prefix
                prefix = label.split("-")[0]
                label_type = {
                    "thm": "Theorem",
                    "lem": "Lemma",
                    "prop": "Proposition",
                    "def": "Definition",
                    "axiom": "Axiom",
                    "cor": "Corollary",
                    "proof": "Proof",
                    "obj": "Object",
                }.get(prefix, "")

                if label_type:
                    lines.append(f"- `{label}` ({label_type})")
                else:
                    lines.append(f"- `{label}`")
            lines.append("")

        return lines


def get_article_registry() -> ArticleRegistry:
    """
    Get singleton ArticleRegistry instance.

    Returns:
        The singleton ArticleRegistry

    Maps to Lean:
        def get_article_registry : IO ArticleRegistry
    """
    return ArticleRegistry()
