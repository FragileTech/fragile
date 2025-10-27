"""
Mathematical Registry - Central Index for All Mathematical Objects.

This module implements a registry that manages all mathematical objects,
relationships, theorems, etc. Supports:
- Adding/getting/removing objects
- Querying by tags
- Querying by relationships
- Reference resolution
- Validation (uniqueness, referential integrity)

All types follow Lean-compatible patterns from docs/LEAN_EMULATION_GUIDE.md.

Version: 2.0.0
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict

from fragile.proofs.core.pipeline_types import (
    Axiom,
    MathematicalObject,
    Parameter,
    Attribute,
    Relationship,
    TheoremBox,
)
from fragile.proofs.registry.reference_system import (
    CombinedTagQuery,
    QueryResult,
    Reference,
    ResolvedReference,
    ResolutionContext,
    TagQuery,
    UnresolvedReference,
    create_reference_resolved,
    extract_id_from_label,
    extract_tags_from_object,
)

# TypeVar for generic registry operations
T = TypeVar("T", bound=BaseModel)


# =============================================================================
# REGISTRY INDEX TYPES
# =============================================================================


class RegistryIndex(BaseModel):
    """
    Index structure for fast lookups.

    Maintains:
    - ID → Object mapping (primary index)
    - Tag → IDs mapping (for tag queries)
    - Type → IDs mapping (for type queries)

    Maps to Lean:
        structure RegistryIndex where
          id_to_object : HashMap String Any
          tag_to_ids : HashMap String (HashSet String)
          type_to_ids : HashMap String (HashSet String)
    """

    model_config = ConfigDict(frozen=False)  # Mutable for updates

    id_to_object: Dict[str, Any] = {}
    tag_to_ids: Dict[str, Set[str]] = {}
    type_to_ids: Dict[str, Set[str]] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self.id_to_object = {}
        self.tag_to_ids = {}
        self.type_to_ids = {}


# =============================================================================
# MATHEMATICAL REGISTRY
# =============================================================================


class MathematicalRegistry:
    """
    Central registry for all mathematical objects.

    Manages:
    - Mathematical objects
    - Axioms
    - Parameters
    - Properties
    - Relationships
    - Theorems

    Provides:
    - Add/get/remove operations
    - Tag-based queries
    - Relationship queries
    - Reference resolution
    - Validation (uniqueness, referential integrity)

    Maps to Lean:
        structure MathematicalRegistry where
          index : RegistryIndex
          objects : HashMap String MathematicalObject
          axioms : HashMap String Axiom
          parameters : HashMap String Parameter
          properties : HashMap String Attribute
          relationships : HashMap String Relationship
          theorems : HashMap String TheoremBox

          -- Operations
          def add (r : MathematicalRegistry) (obj : α) : MathematicalRegistry := ...
          def get (r : MathematicalRegistry) (id : String) : Option α := ...
          def query_by_tags (r : MathematicalRegistry) (query : TagQuery) : List α := ...
    """

    def __init__(self):
        """Initialize empty registry."""
        self._index = RegistryIndex()

        # Separate collections by type
        self._objects: Dict[str, MathematicalObject] = {}
        self._axioms: Dict[str, Axiom] = {}
        self._parameters: Dict[str, Parameter] = {}
        self._properties: Dict[str, Attribute] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._theorems: Dict[str, TheoremBox] = {}

    # =========================================================================
    # BASIC OPERATIONS
    # =========================================================================

    def add(self, obj: Any) -> None:
        """
        Add object to registry.

        Performs:
        1. Extract ID from object
        2. Check uniqueness
        3. Add to appropriate collection
        4. Update indices

        Raises:
            ValueError: If ID already exists or object has no label

        Maps to Lean:
            def add (r : MathematicalRegistry) (obj : α) : Result MathematicalRegistry String :=
              let id := extract_id obj
              if r.index.id_to_object.contains id then
                Err s!"Duplicate ID: {id}"
              else
                Ok { r with ... }
        """
        # Extract ID
        id = extract_id_from_label(obj)
        if id is None:
            raise ValueError(f"Object has no 'label' attribute: {type(obj).__name__}")

        # Check uniqueness
        if id in self._index.id_to_object:
            raise ValueError(f"Duplicate ID: {id}")

        # Add to appropriate collection
        if isinstance(obj, MathematicalObject):
            self._objects[id] = obj
        elif isinstance(obj, Axiom):
            self._axioms[id] = obj
        elif isinstance(obj, Parameter):
            self._parameters[id] = obj
        elif isinstance(obj, Attribute):
            self._properties[id] = obj
        elif isinstance(obj, Relationship):
            self._relationships[id] = obj
        elif isinstance(obj, TheoremBox):
            self._theorems[id] = obj
        else:
            raise ValueError(f"Unsupported object type: {type(obj).__name__}")

        # Update primary index
        self._index.id_to_object[id] = obj

        # Update type index
        type_name = type(obj).__name__
        if type_name not in self._index.type_to_ids:
            self._index.type_to_ids[type_name] = set()
        self._index.type_to_ids[type_name].add(id)

        # Update tag index
        tags = extract_tags_from_object(obj)
        for tag in tags:
            if tag not in self._index.tag_to_ids:
                self._index.tag_to_ids[tag] = set()
            self._index.tag_to_ids[tag].add(id)

    def add_all(self, objects: List[Any]) -> None:
        """
        Add multiple objects to registry.

        Maps to Lean:
            def add_all (r : MathematicalRegistry) (objs : List α) : Result MathematicalRegistry String :=
              objs.foldlM (fun r obj => r.add obj) r
        """
        for obj in objects:
            self.add(obj)

    def get(self, id: str) -> Optional[Any]:
        """
        Get object by ID.

        Total function (returns None if not found).

        Maps to Lean:
            def get (r : MathematicalRegistry) (id : String) : Option α :=
              r.index.id_to_object.find? id
        """
        return self._index.id_to_object.get(id)

    def get_object(self, id: str) -> Optional[MathematicalObject]:
        """Get mathematical object by ID."""
        return self._objects.get(id)

    def get_axiom(self, id: str) -> Optional[Axiom]:
        """Get axiom by ID."""
        return self._axioms.get(id)

    def get_parameter(self, id: str) -> Optional[Parameter]:
        """Get parameter by ID."""
        return self._parameters.get(id)

    def get_attribute(self, id: str) -> Optional[Attribute]:
        """Get property by ID."""
        return self._properties.get(id)

    def get_relationship(self, id: str) -> Optional[Relationship]:
        """Get relationship by ID."""
        return self._relationships.get(id)

    def get_theorem(self, id: str) -> Optional[TheoremBox]:
        """Get theorem by ID."""
        return self._theorems.get(id)

    def contains(self, id: str) -> bool:
        """Check if ID exists in registry."""
        return id in self._index.id_to_object

    def remove(self, id: str) -> bool:
        """
        Remove object from registry.

        Returns: True if removed, False if not found.

        Maps to Lean:
            def remove (r : MathematicalRegistry) (id : String) : MathematicalRegistry :=
              { r with
                index := r.index.remove id
                objects := r.objects.erase id
                ... }
        """
        if id not in self._index.id_to_object:
            return False

        obj = self._index.id_to_object[id]

        # Remove from appropriate collection
        if isinstance(obj, MathematicalObject):
            del self._objects[id]
        elif isinstance(obj, Axiom):
            del self._axioms[id]
        elif isinstance(obj, Parameter):
            del self._parameters[id]
        elif isinstance(obj, Attribute):
            del self._properties[id]
        elif isinstance(obj, Relationship):
            del self._relationships[id]
        elif isinstance(obj, TheoremBox):
            del self._theorems[id]

        # Remove from indices
        del self._index.id_to_object[id]

        type_name = type(obj).__name__
        if type_name in self._index.type_to_ids:
            self._index.type_to_ids[type_name].discard(id)

        tags = extract_tags_from_object(obj)
        for tag in tags:
            if tag in self._index.tag_to_ids:
                self._index.tag_to_ids[tag].discard(id)

        return True

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def query_by_tag(self, query: TagQuery) -> QueryResult[Any]:
        """
        Query objects by tag.

        Maps to Lean:
            def query_by_tag (r : MathematicalRegistry) (query : TagQuery) : List α :=
              r.index.id_to_object.values.filter (fun obj =>
                query.matches (extract_tags obj))
        """
        matching_ids: Set[str] = set()

        if query.mode == "any":
            # Union of all tag sets
            for tag in query.tags:
                if tag in self._index.tag_to_ids:
                    matching_ids.update(self._index.tag_to_ids[tag])

        elif query.mode == "all":
            # Intersection of all tag sets
            if not query.tags:
                matching_ids = set(self._index.id_to_object.keys())
            else:
                first_tag = query.tags[0]
                if first_tag in self._index.tag_to_ids:
                    matching_ids = self._index.tag_to_ids[first_tag].copy()

                for tag in query.tags[1:]:
                    if tag in self._index.tag_to_ids:
                        matching_ids &= self._index.tag_to_ids[tag]
                    else:
                        matching_ids = set()  # No objects have this tag
                        break

        elif query.mode == "none":
            # All objects minus those with any of the tags
            all_ids = set(self._index.id_to_object.keys())
            excluded_ids: Set[str] = set()
            for tag in query.tags:
                if tag in self._index.tag_to_ids:
                    excluded_ids.update(self._index.tag_to_ids[tag])
            matching_ids = all_ids - excluded_ids

        # Get objects for matching IDs
        matches = [self._index.id_to_object[id] for id in matching_ids]

        return QueryResult(matches=matches, total_count=len(matches))

    def query_by_tags(self, query: CombinedTagQuery) -> QueryResult[Any]:
        """
        Query objects by combined tag query.

        Maps to Lean:
            def query_by_tags (r : MathematicalRegistry) (query : CombinedTagQuery) : List α :=
              r.index.id_to_object.values.filter (fun obj =>
                query.matches (extract_tags obj))
        """
        matches = []
        for id, obj in self._index.id_to_object.items():
            tags = extract_tags_from_object(obj)
            if query.matches(tags):
                matches.append(obj)

        return QueryResult(matches=matches, total_count=len(matches))

    def query_by_type(self, type_name: str) -> QueryResult[Any]:
        """
        Query objects by type.

        Args:
            type_name: Type name (e.g., "MathematicalObject", "Theorem")

        Maps to Lean:
            def query_by_type (r : MathematicalRegistry) (type_name : String) : List α :=
              r.index.type_to_ids.find? type_name
                |>.map (fun ids => ids.map (fun id => r.get id))
                |>.getD []
        """
        if type_name not in self._index.type_to_ids:
            return QueryResult(matches=[], total_count=0)

        matching_ids = self._index.type_to_ids[type_name]
        matches = [self._index.id_to_object[id] for id in matching_ids]

        return QueryResult(matches=matches, total_count=len(matches))

    def get_all_objects(self) -> List[MathematicalObject]:
        """Get all mathematical objects."""
        return list(self._objects.values())

    def get_all_axioms(self) -> List[Axiom]:
        """Get all axioms."""
        return list(self._axioms.values())

    def get_all_parameters(self) -> List[Parameter]:
        """Get all parameters."""
        return list(self._parameters.values())

    def get_all_properties(self) -> List[Attribute]:
        """Get all properties."""
        return list(self._properties.values())

    def get_all_relationships(self) -> List[Relationship]:
        """Get all relationships."""
        return list(self._relationships.values())

    def get_all_theorems(self) -> List[TheoremBox]:
        """Get all theorems."""
        return list(self._theorems.values())

    # =========================================================================
    # RELATIONSHIP QUERIES
    # =========================================================================

    def get_related_objects(self, object_id: str) -> List[str]:
        """
        Get all objects related to the given object.

        Returns list of object IDs that are connected via relationships.

        Maps to Lean:
            def get_related_objects (r : MathematicalRegistry) (obj_id : String) : List String :=
              r.relationships.values
                .filter (fun rel => rel.source_object == obj_id || rel.target_object == obj_id)
                .map (fun rel => if rel.source_object == obj_id then rel.target_object else rel.source_object)
        """
        related = []
        for rel in self._relationships.values():
            if rel.source_object == object_id:
                related.append(rel.target_object)
            elif rel.target_object == object_id:
                if rel.is_symmetric():
                    # For symmetric relationships, include both directions
                    related.append(rel.source_object)

        return related

    def get_relationships_for_object(self, object_id: str) -> List[Relationship]:
        """
        Get all relationships involving the given object.

        Maps to Lean:
            def get_relationships_for_object (r : MathematicalRegistry) (obj_id : String) : List Relationship :=
              r.relationships.values
                .filter (fun rel => rel.source_object == obj_id || rel.target_object == obj_id)
        """
        return [
            rel for rel in self._relationships.values()
            if rel.source_object == object_id or rel.target_object == object_id
        ]

    def get_relationships_by_type(self, relationship_type: str) -> List[Relationship]:
        """
        Get all relationships of a specific type.

        Args:
            relationship_type: Type name (e.g., "equivalence", "embedding")
        """
        return [
            rel for rel in self._relationships.values()
            if rel.relationship_type.value == relationship_type
        ]

    # =========================================================================
    # REFERENCE RESOLUTION
    # =========================================================================

    def resolve_reference(self, ref: Reference[T]) -> Optional[T]:
        """
        Resolve a reference to its full object.

        Total function (returns None if cannot resolve).

        Maps to Lean:
            def resolve_reference (r : MathematicalRegistry) (ref : Reference α) : Option α :=
              match ref with
              | UnresolvedReference ur => r.get ur.id
              | ResolvedReference rr => some rr.value
        """
        if isinstance(ref, ResolvedReference):
            return ref.value

        if isinstance(ref, UnresolvedReference):
            return self.get(ref.id)

        return None

    def resolve_references(self, refs: List[Reference[T]]) -> List[Optional[T]]:
        """
        Resolve multiple references.

        Maps to Lean:
            def resolve_references (r : MathematicalRegistry) (refs : List (Reference α)) : List (Option α) :=
              refs.map (r.resolve_reference)
        """
        return [self.resolve_reference(ref) for ref in refs]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def count_total(self) -> int:
        """Get total number of objects in registry."""
        return len(self._index.id_to_object)

    def count_by_type(self) -> Dict[str, int]:
        """Get counts by type."""
        return {
            "MathematicalObject": len(self._objects),
            "Axiom": len(self._axioms),
            "Parameter": len(self._parameters),
            "Attribute": len(self._properties),
            "Relationship": len(self._relationships),
            "TheoremBox": len(self._theorems),
        }

    def get_all_tags(self) -> Set[str]:
        """Get set of all tags used in registry."""
        return set(self._index.tag_to_ids.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_objects": self.count_total(),
            "counts_by_type": self.count_by_type(),
            "total_tags": len(self.get_all_tags()),
            "all_tags": sorted(self.get_all_tags()),
        }

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_referential_integrity(self) -> List[str]:
        """
        Validate that all referenced IDs exist.

        Returns list of missing IDs (empty if all valid).

        Checks:
        - Theorem input_objects exist
        - Theorem input_axioms exist
        - Attribute object_label exists
        - Relationship source/target exist

        Maps to Lean:
            def validate_referential_integrity (r : MathematicalRegistry) : List String :=
              -- Check all references and return list of missing IDs
              ...
        """
        missing = []

        # Check theorems
        for thm in self._theorems.values():
            for obj_id in thm.input_objects:
                if not self.contains(obj_id):
                    missing.append(f"Theorem {thm.label} references missing object: {obj_id}")

            for axiom_id in thm.input_axioms:
                if not self.contains(axiom_id):
                    missing.append(f"Theorem {thm.label} references missing axiom: {axiom_id}")

        # Check properties
        for prop in self._properties.values():
            if not self.contains(prop.object_label):
                missing.append(f"Attribute {prop.label} references missing object: {prop.object_label}")

        # Check relationships
        for rel in self._relationships.values():
            if not self.contains(rel.source_object):
                missing.append(f"Relationship {rel.label} references missing source: {rel.source_object}")
            if not self.contains(rel.target_object):
                missing.append(f"Relationship {rel.label} references missing target: {rel.target_object}")

        return missing

    def is_valid(self) -> bool:
        """Check if registry has valid referential integrity."""
        return len(self.validate_referential_integrity()) == 0
