"""Registry search utilities for retrieving mathematical entities by label.

This module provides efficient search functions for the preprocess and directives registries,
with caching to avoid repeated disk I/O.
"""

from pathlib import Path
from typing import Any

from mathster.relationships.directives_graph import load_directives_registry
from mathster.relationships.preprocess_graph import load_preprocess_registry


# Module-level cache for the preprocess registry
_PREPROCESS_REGISTRY_CACHE: dict[tuple[Path, ...], dict[str, dict[str, Any]]] = {}

# Module-level cache for the directives registry
_DIRECTIVES_REGISTRY_CACHE: dict[tuple[Path, ...], dict[str, dict[str, Any]]] = {}


def get_preprocess_label(
    label: str,
    preprocess_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Retrieve a preprocess entity by its label.

    This function searches the unified preprocess registry for an entity with the
    given label. The registry is loaded once and cached for efficiency.

    Args:
        label: Entity label to search for (e.g., "thm-kl-convergence", "def-baoab-update").
        preprocess_dir: Directory containing preprocess JSON files. If None, defaults
            to "unified_registry/preprocess/" relative to the project root.

    Returns:
        Dictionary containing the entity data if found, None otherwise. The returned
        dict includes fields like:
        - label: str - Unique identifier
        - title: str - Human-readable name
        - type: str - Entity type ("theorem", "axiom", "definition", etc.)
        - nl_summary or nl_statement: str - Natural language description
        - tags: list[str] - Classification tags
        - document_id: str - Source document
        - section: str - Section heading
        - span: dict - Line number positioning
        Plus entity-specific fields (equations, hypotheses, parameters, etc.)

    Examples:
        >>> entity = get_preprocess_label("thm-kl-convergence")
        >>> if entity:
        ...     print(entity["title"])
        ...     print(entity["type"])
        ...     print(entity["tags"])

        >>> # Use custom registry directory
        >>> entity = get_preprocess_label(
        ...     "def-baoab-update", preprocess_dir="path/to/custom/registry"
        ... )

    Note:
        - The registry is cached at module level to avoid repeated disk I/O
        - Entities are returned as raw dicts (not instantiated as Pydantic models)
        - Invalid or missing labels return None rather than raising exceptions
        - Use clear_preprocess_cache() to force reload if registry is updated
    """
    # Determine the preprocess directory
    if preprocess_dir is None:
        # Default to unified_registry/preprocess/ relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        preprocess_dir = project_root / "unified_registry" / "preprocess"
    else:
        preprocess_dir = Path(preprocess_dir)

    # Create cache key from the directory path
    cache_key = (preprocess_dir.resolve(),)

    # Load registry (from cache if available)
    if cache_key not in _PREPROCESS_REGISTRY_CACHE:
        _PREPROCESS_REGISTRY_CACHE[cache_key] = load_preprocess_registry(preprocess_dir)

    # Look up the label in the registry
    registry = _PREPROCESS_REGISTRY_CACHE[cache_key]
    return registry.get(label)


def get_directive_label(
    label: str,
    directives_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Retrieve a directive entity by its label.

    This function searches the unified directives registry for an entity with the
    given label. The registry is loaded once and cached for efficiency.

    Directive entities contain raw MyST directive blocks with line-level positioning,
    making them ideal for source navigation and document editing tasks. They differ
    from preprocess entities which contain structured mathematical components.

    Args:
        label: Entity label to search for (e.g., "thm-kl-convergence", "def-baoab-update").
        directives_dir: Directory containing directive JSON files. If None, defaults
            to "unified_registry/directives/" relative to the project root.

    Returns:
        Dictionary containing the entity data if found, None otherwise. The returned
        dict includes fields like:
        - label: str - Unique identifier
        - title: str - Human-readable name
        - directive_type: str - Entity type ("theorem", "axiom", "definition", etc.)
        - start_line: int - Start line in source document
        - end_line: int - End line in source document
        - content_start: int - First line of actual content (after header)
        - content_end: int - Last line of content
        - header_lines: list[int] - Lines containing directive header
        - content: list[dict] - Raw content with line numbers
        - metadata: dict - Directive metadata (e.g., ":label:", ":nonumber:")
        - section: str - Section heading
        - references: list[str] - Referenced labels in content
        - raw_directive: str - Full directive block as string
        - _registry_context: dict - Stage metadata (document_id, chapter_index, etc.)

    Examples:
        >>> entity = get_directive_label("thm-kl-convergence")
        >>> if entity:
        ...     print(entity["title"])
        ...     print(entity["directive_type"])
        ...     print(f"Lines {entity['start_line']}-{entity['end_line']}")

        >>> # Use custom registry directory
        >>> entity = get_directive_label(
        ...     "def-baoab-update", directives_dir="path/to/custom/registry"
        ... )

        >>> # Access registry context
        >>> if entity and "_registry_context" in entity:
        ...     context = entity["_registry_context"]
        ...     print(f"Document: {context.get('document_id')}")
        ...     print(f"Chapter: {context.get('chapter_index')}")

    Note:
        - The registry is cached at module level to avoid repeated disk I/O
        - Entities are returned as raw dicts (not instantiated as Pydantic models)
        - Invalid or missing labels return None rather than raising exceptions
        - Use clear_registry_cache() to force reload if registry is updated
        - Directive entities are optimized for source navigation (line numbers)
        - For structured mathematical analysis, use get_preprocess_label() instead
    """
    # Determine the directives directory
    if directives_dir is None:
        # Default to unified_registry/directives/ relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        directives_dir = project_root / "unified_registry" / "directives"
    else:
        directives_dir = Path(directives_dir)

    # Create cache key from the directory path
    cache_key = (directives_dir.resolve(),)

    # Load registry (from cache if available)
    if cache_key not in _DIRECTIVES_REGISTRY_CACHE:
        _DIRECTIVES_REGISTRY_CACHE[cache_key] = load_directives_registry(directives_dir)

    # Look up the label in the registry
    registry = _DIRECTIVES_REGISTRY_CACHE[cache_key]
    return registry.get(label)


def clear_registry_cache() -> None:
    """Clear all registry caches (preprocess and directives).

    Forces the next call to get_preprocess_label() or get_directive_label() to
    reload their respective registries from disk. Useful when registries have been
    updated externally.

    Example:
        >>> # After updating registry files
        >>> clear_registry_cache()
        >>> entity = get_preprocess_label("thm-new-result")  # Loads fresh from disk
        >>> directive = get_directive_label("def-new-concept")  # Also loads fresh
    """
    _PREPROCESS_REGISTRY_CACHE.clear()
    _DIRECTIVES_REGISTRY_CACHE.clear()


def clear_preprocess_cache() -> None:
    """Clear the preprocess registry cache.

    Forces the next call to get_preprocess_label() to reload the registry from disk.
    Useful when the registry has been updated externally.

    Note:
        This is a convenience function. Consider using clear_registry_cache() to
        clear all caches at once.

    Example:
        >>> # After updating preprocess registry files
        >>> clear_preprocess_cache()
        >>> entity = get_preprocess_label("thm-new-result")  # Loads fresh from disk
    """
    _PREPROCESS_REGISTRY_CACHE.clear()


def clear_directives_cache() -> None:
    """Clear the directives registry cache.

    Forces the next call to get_directive_label() to reload the registry from disk.
    Useful when the registry has been updated externally.

    Note:
        This is a convenience function. Consider using clear_registry_cache() to
        clear all caches at once.

    Example:
        >>> # After updating directives registry files
        >>> clear_directives_cache()
        >>> entity = get_directive_label("def-new-concept")  # Loads fresh from disk
    """
    _DIRECTIVES_REGISTRY_CACHE.clear()
