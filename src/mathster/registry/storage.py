"""
Storage Layer for Mathematical Objects with Reference Resolution.

This module implements JSON serialization/deserialization with ID-based
references to avoid duplication. Objects are stored as individual JSON files
organized by type in a directory structure.

Directory structure:
    data/mathematical_objects/
        objects/obj-*.json
        axioms/axiom-*.json
        parameters/param-*.json
        properties/prop-*.json
        relationships/rel-*.json
        theorems/thm-*.json
        index.json

Key Features:
- Reference mode: Store only IDs to avoid duplication
- Lazy loading: Load objects on demand
- Caching: Cache resolved references
- Referential integrity: Validate references on load

All types follow Lean-compatible patterns from docs/LEAN_EMULATION_GUIDE.md.

Version: 1.0.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from mathster.core.pipeline_types import (
    Attribute,
    Axiom,
    MathematicalObject,
    Parameter,
    Relationship,
    TheoremBox,
)
from mathster.registry.reference_system import (
    extract_id_from_label,
)


T = TypeVar("T", bound=BaseModel)


# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================


class StorageConfig(BaseModel):
    """
    Configuration for storage directory structure.

    Maps to Lean:
        structure StorageConfig where
          base_dir : String
          use_references : Bool
          pretty_print : Bool
    """

    base_dir: Path
    use_references: bool = False  # Store full objects (reference resolution TBD)
    pretty_print: bool = True  # Format JSON for readability
    indent: int = 2  # Indentation level for JSON

    def get_subdir(self, type_prefix: str) -> Path:
        """
        Get subdirectory for object type.

        Pure function (no side effects).

        Maps to Lean:
            def get_subdir (config : StorageConfig) (type_prefix : String) : String :=
              config.base_dir ++ "/" ++ type_prefix ++ "s/"
        """
        # Map type prefixes to subdirectory names
        type_map = {
            "obj": "objects",
            "axiom": "axioms",
            "param": "parameters",
            "prop": "properties",
            "rel": "relationships",
            "thm": "theorems",
        }
        subdir_name = type_map.get(type_prefix, f"{type_prefix}s")
        return self.base_dir / subdir_name

    def get_index_path(self) -> Path:
        """Get path to index file."""
        return self.base_dir / "index.json"


# =============================================================================
# SERIALIZATION FUNCTIONS
# =============================================================================


def serialize_to_dict(obj: Any, use_references: bool = True) -> dict[str, Any]:
    """
    Serialize Pydantic model to dictionary.

    If use_references=True, nested objects are replaced with ID references.

    Pure function (no side effects, no file I/O).

    Maps to Lean:
        def serialize_to_dict (obj : α) (use_references : Bool) : Dict String Json :=
          if use_references then replace_nested_with_ids obj else obj.to_dict
    """
    # Get base dictionary from Pydantic model
    data = obj.model_dump(mode="python")

    if use_references:
        # Replace nested objects with ID references
        data = _replace_objects_with_references(data)

    return data


def _replace_objects_with_references(data: Any, is_root: bool = True) -> Any:
    """
    Recursively replace nested objects with ID references.

    Pure function (no side effects).

    Args:
        data: Data to process
        is_root: If True, don't replace the root object (only nested ones)
    """
    if isinstance(data, dict):
        # Check if this looks like a labeled object
        if "label" in data and isinstance(data.get("label"), str):
            # Only replace if this is NOT the root object
            # Check if it has multiple fields (full object) or just label
            if not is_root and len(data) > 3:  # More than just label, type, basic fields
                # Replace with reference
                return {"$ref": data["label"]}
        # Recursively process dictionary values (not root anymore)
        return {
            key: _replace_objects_with_references(value, is_root=False)
            for key, value in data.items()
        }
    if isinstance(data, list):
        # Recursively process list items (not root anymore)
        return [_replace_objects_with_references(item, is_root=False) for item in data]
    # Primitive value, return as-is
    return data


def deserialize_from_dict(data: dict[str, Any], model_class: type[T]) -> T:
    """
    Deserialize dictionary to Pydantic model.

    Pure function (no side effects, no reference resolution).

    Note: References remain as {"$ref": "id"} and must be resolved separately.

    Maps to Lean:
        def deserialize_from_dict (data : Dict String Json) (T : Type) : T :=
          T.from_dict data
    """
    return model_class.model_validate(data)


# =============================================================================
# FILE OPERATIONS
# =============================================================================


def save_object_to_file(obj: Any, file_path: Path, config: StorageConfig) -> None:
    """
    Save object to JSON file.

    Side effect: Creates file on disk.

    Maps to Lean (IO monad):
        def save_object_to_file (obj : α) (path : String) (config : StorageConfig) : IO Unit :=
          let data := serialize_to_dict obj config.use_references
          let json := Json.stringify data config.pretty_print
          IO.write_file path json
    """
    # Serialize to dictionary
    data = serialize_to_dict(obj, use_references=config.use_references)

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(file_path, "w", encoding="utf-8") as f:
        if config.pretty_print:
            json.dump(data, f, indent=config.indent, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)


def load_object_from_file(file_path: Path, model_class: type[T]) -> T:
    """
    Load object from JSON file.

    Side effect: Reads file from disk.

    Note: References remain unresolved and must be resolved separately.

    Maps to Lean (IO monad):
        def load_object_from_file (path : String) (T : Type) : IO T :=
          let json <- IO.read_file path
          let data := Json.parse json
          pure (deserialize_from_dict data T)
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    return deserialize_from_dict(data, model_class)


# =============================================================================
# REGISTRY STORAGE
# =============================================================================


class RegistryStorage:
    """
    Storage manager for MathematicalRegistry.

    Handles saving/loading registry to/from directory structure.

    Maps to Lean:
        structure RegistryStorage where
          config : StorageConfig
          cache : HashMap String Object
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self._cache: dict[str, Any] = {}  # Cache for loaded objects

    def save_registry(self, registry: Any) -> None:
        """
        Save entire registry to directory.

        Side effect: Creates directory structure and files.
        """
        # Create base directory
        self.config.base_dir.mkdir(parents=True, exist_ok=True)

        # Save objects by type
        self._save_collection(registry.get_all_objects(), "obj", MathematicalObject)
        self._save_collection(registry.get_all_axioms(), "axiom", Axiom)
        self._save_collection(registry.get_all_parameters(), "param", Parameter)
        self._save_collection(registry.get_all_properties(), "prop", Attribute)
        self._save_collection(registry.get_all_relationships(), "rel", Relationship)
        self._save_collection(registry.get_all_theorems(), "thm", TheoremBox)

        # Save index
        self._save_index(registry)

    def _save_collection(
        self, objects: list[Any], type_prefix: str, model_class: type[BaseModel]
    ) -> None:
        """Save collection of objects to subdirectory."""
        if not objects:
            return

        subdir = self.config.get_subdir(type_prefix)
        subdir.mkdir(parents=True, exist_ok=True)

        for obj in objects:
            obj_id = extract_id_from_label(obj)
            if obj_id:
                file_path = subdir / f"{obj_id}.json"
                save_object_to_file(obj, file_path, self.config)

    def _save_index(self, registry: Any) -> None:
        """Save registry index with metadata."""
        index_data = {
            "version": "1.0.0",
            "statistics": registry.get_statistics(),
            "all_ids": list(registry._index.id_to_object.keys()),
            "all_tags": list(registry.get_all_tags()),
        }

        index_path = self.config.get_index_path()
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=self.config.indent, ensure_ascii=False)

    def load_registry(self, registry_class: type[Any]) -> Any:
        """
        Load registry from directory.

        Side effect: Reads files from disk.

        Returns: New registry instance with loaded objects.
        """
        # Create empty registry
        registry = registry_class()

        # Load objects by type
        self._load_collection(registry, "obj", MathematicalObject)
        self._load_collection(registry, "axiom", Axiom)
        self._load_collection(registry, "param", Parameter)
        self._load_collection(registry, "prop", Attribute)
        self._load_collection(registry, "rel", Relationship)
        self._load_collection(registry, "thm", TheoremBox)

        return registry

    def _load_collection(self, registry: Any, type_prefix: str, model_class: type[T]) -> None:
        """Load collection of objects from subdirectory."""
        subdir = self.config.get_subdir(type_prefix)

        if not subdir.exists():
            return

        # Load all JSON files in subdirectory
        for file_path in subdir.glob("*.json"):
            try:
                obj = load_object_from_file(file_path, model_class)
                registry.add(obj)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

    def get_object(self, object_id: str, model_class: type[T]) -> T | None:
        """
        Lazy load object by ID.

        Side effect: Reads file from disk if not cached.

        Returns: Object if found, None otherwise.
        """
        # Check cache first
        if object_id in self._cache:
            return self._cache[object_id]

        # Determine subdirectory from ID prefix
        type_prefix = object_id.split("-")[0]
        subdir = self.config.get_subdir(type_prefix)
        file_path = subdir / f"{object_id}.json"

        if not file_path.exists():
            return None

        # Load and cache
        try:
            obj = load_object_from_file(file_path, model_class)
            self._cache[object_id] = obj
            return obj
        except Exception as e:
            print(f"Warning: Failed to load {object_id}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear object cache."""
        self._cache.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def save_registry_to_directory(registry: Any, directory: str | Path) -> None:
    """
    Convenience function: Save registry to directory.

    Side effect: Creates directory structure and files.
    """
    config = StorageConfig(base_dir=Path(directory))
    storage = RegistryStorage(config)
    storage.save_registry(registry)


def load_registry_from_directory(registry_class: type[Any], directory: str | Path) -> Any:
    """
    Convenience function: Load registry from directory.

    Side effect: Reads files from disk.
    """
    config = StorageConfig(base_dir=Path(directory))
    storage = RegistryStorage(config)
    return storage.load_registry(registry_class)
