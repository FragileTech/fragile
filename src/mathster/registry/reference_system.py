"""
Reference System for ID-Based Storage.

This module implements a reference system that allows storing objects by ID
to avoid duplication in JSON files. References can be in two modes:
- ID mode: Just the string ID (for storage)
- Resolved mode: The full object (for computation)

All types follow Lean-compatible patterns from docs/LEAN_EMULATION_GUIDE.md.

Version: 2.0.0
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field


# TypeVars for generic types
T = TypeVar("T")
EntityType = TypeVar("EntityType", bound=BaseModel)


# =============================================================================
# REFERENCE TYPES
# =============================================================================


class UnresolvedReference(BaseModel, Generic[T]):
    """
    Reference containing only an ID (not yet resolved).

    Used for storage to avoid duplicating full objects in JSON.

    Maps to Lean:
        structure UnresolvedReference (T : Type) where
          id : String
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1, description="Object ID")
    type_hint: str | None = Field(
        None, description="Type hint for validation (e.g., 'MathematicalObject')"
    )

    def is_resolved(self) -> Literal[False]:
        """Always False for unresolved references."""
        return False


class ResolvedReference(BaseModel, Generic[T]):
    """
    Reference containing the full resolved object.

    Used for computation after loading from storage.

    Maps to Lean:
        structure ResolvedReference (T : Type) where
          id : String
          value : T
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1, description="Object ID")
    value: T = Field(..., description="The resolved object")

    def is_resolved(self) -> Literal[True]:
        """Always True for resolved references."""
        return True


# Union type for references (either unresolved or resolved)
Reference = Union[UnresolvedReference[T], ResolvedReference[T]]


# =============================================================================
# TAG QUERIES
# =============================================================================


class TagQuery(BaseModel):
    """
    Query specification for filtering objects by tags.

    Supports:
    - ANY: Match objects with ANY of the specified tags (OR)
    - ALL: Match objects with ALL of the specified tags (AND)
    - NONE: Match objects with NONE of the specified tags (NOT)

    Maps to Lean:
        inductive TagQueryMode where
          | any : TagQueryMode
          | all : TagQueryMode
          | none : TagQueryMode

        structure TagQuery where
          tags : List String
          mode : TagQueryMode
    """

    model_config = ConfigDict(frozen=True)

    tags: list[str] = Field(..., min_items=1, description="Tags to query")
    mode: Literal["any", "all", "none"] = Field("any", description="Query mode: any/all/none")

    def matches(self, object_tags: set[str]) -> bool:
        """
        Pure function: Check if object tags match this query.

        Maps to Lean:
            def matches (q : TagQuery) (object_tags : List String) : Bool :=
              match q.mode with
              | TagQueryMode.any => q.tags.any (fun t => t ∈ object_tags)
              | TagQueryMode.all => q.tags.all (fun t => t ∈ object_tags)
              | TagQueryMode.none => !q.tags.any (fun t => t ∈ object_tags)
        """
        tag_set = set(self.tags)
        if self.mode == "any":
            return bool(tag_set & object_tags)  # Intersection non-empty
        if self.mode == "all":
            return tag_set.issubset(object_tags)  # All tags present
        if self.mode == "none":
            return not bool(tag_set & object_tags)  # No intersection
        return False


class CombinedTagQuery(BaseModel):
    """
    Combined tag query with multiple conditions.

    Example:
        must_have=["euclidean-gas", "discrete"]  # AND
        any_of=["particle", "swarm"]             # OR
        must_not_have=["deprecated"]             # NOT

    Maps to Lean:
        structure CombinedTagQuery where
          must_have : List String
          any_of : List String
          must_not_have : List String
    """

    model_config = ConfigDict(frozen=True)

    must_have: list[str] = Field(
        default_factory=list, description="Tags that MUST be present (AND)"
    )
    any_of: list[str] = Field(
        default_factory=list, description="At least ONE must be present (OR)"
    )
    must_not_have: list[str] = Field(
        default_factory=list, description="Tags that MUST NOT be present (NOT)"
    )

    def matches(self, object_tags: set[str]) -> bool:
        """
        Pure function: Check if object tags match this combined query.

        Maps to Lean:
            def matches (q : CombinedTagQuery) (object_tags : List String) : Bool :=
              let must_have_ok := q.must_have.all (fun t => t ∈ object_tags)
              let any_of_ok := q.any_of.isEmpty || q.any_of.any (fun t => t ∈ object_tags)
              let must_not_have_ok := !q.must_not_have.any (fun t => t ∈ object_tags)
              must_have_ok && any_of_ok && must_not_have_ok
        """
        # Check must_have (AND)
        if self.must_have:
            if not set(self.must_have).issubset(object_tags):
                return False

        # Check any_of (OR)
        if self.any_of:
            if not bool(set(self.any_of) & object_tags):
                return False

        # Check must_not_have (NOT)
        if self.must_not_have:
            if bool(set(self.must_not_have) & object_tags):
                return False

        return True


# =============================================================================
# RESOLUTION FUNCTIONS
# =============================================================================


def create_reference_id(id: str, type_hint: str | None = None) -> UnresolvedReference[Any]:
    """
    Helper: Create unresolved reference from ID.

    Pure function (no side effects).

    Maps to Lean:
        def create_reference_id (id : String) (type_hint : Option String) : UnresolvedReference α :=
          { id := id, type_hint := type_hint }
    """
    return UnresolvedReference(id=id, type_hint=type_hint)


def create_reference_resolved(id: str, value: T) -> ResolvedReference[T]:
    """
    Helper: Create resolved reference from object.

    Pure function (no side effects).

    Maps to Lean:
        def create_reference_resolved (id : String) (value : α) : ResolvedReference α :=
          { id := id, value := value }
    """
    return ResolvedReference(id=id, value=value)


def is_reference_resolved(ref: Reference[T]) -> bool:
    """
    Pure function: Check if reference is resolved.

    Maps to Lean:
        def is_reference_resolved (ref : Reference α) : Bool :=
          match ref with
          | UnresolvedReference _ => false
          | ResolvedReference _ => true
    """
    return isinstance(ref, ResolvedReference)


def get_reference_id(ref: Reference[T]) -> str:
    """
    Pure function: Get ID from reference (works for both resolved and unresolved).

    Maps to Lean:
        def get_reference_id (ref : Reference α) : String :=
          ref.id
    """
    return ref.id


def get_reference_value(ref: Reference[T]) -> T | None:
    """
    Total function: Get value from reference (None if unresolved).

    Maps to Lean:
        def get_reference_value (ref : Reference α) : Option α :=
          match ref with
          | UnresolvedReference _ => none
          | ResolvedReference r => some r.value
    """
    if isinstance(ref, ResolvedReference):
        return ref.value
    return None


# =============================================================================
# QUERY RESULT TYPES
# =============================================================================


class QueryResult(BaseModel, Generic[T]):
    """
    Result of a query operation.

    Contains matched objects and metadata about the query.

    Maps to Lean:
        structure QueryResult (T : Type) where
          matches : List T
          total_count : Nat
          query_time_ms : Option Float
    """

    model_config = ConfigDict(frozen=True)

    matches: list[T] = Field(..., description="Objects matching the query")
    total_count: int = Field(..., ge=0, description="Total number of matches")
    query_time_ms: float | None = Field(
        None, ge=0, description="Query execution time in milliseconds"
    )

    def is_empty(self) -> bool:
        """Pure function: Check if query returned no results."""
        return self.total_count == 0

    def count(self) -> int:
        """Pure function: Get number of matches."""
        return self.total_count


# =============================================================================
# RESOLUTION CONTEXT
# =============================================================================


class ResolutionContext(BaseModel):
    """
    Context for resolving references.

    Tracks:
    - Which IDs have been resolved
    - Which IDs failed to resolve
    - Circular dependency detection

    Maps to Lean:
        structure ResolutionContext where
          resolved : HashSet String
          failed : HashSet String
          resolution_stack : List String
    """

    model_config = ConfigDict(frozen=False)  # Mutable for tracking

    resolved: set[str] = Field(default_factory=set, description="Successfully resolved IDs")
    failed: set[str] = Field(default_factory=set, description="Failed to resolve IDs")
    resolution_stack: list[str] = Field(
        default_factory=list, description="Current resolution stack (for cycle detection)"
    )

    def push_resolution(self, id: str) -> bool:
        """
        Check if ID is already in resolution stack (circular dependency).
        If not, push it.

        Returns: True if can proceed, False if circular dependency detected.
        """
        if id in self.resolution_stack:
            return False  # Circular dependency!
        self.resolution_stack.append(id)
        return True

    def pop_resolution(self, id: str) -> None:
        """Pop ID from resolution stack."""
        if self.resolution_stack and self.resolution_stack[-1] == id:
            self.resolution_stack.pop()

    def mark_resolved(self, id: str) -> None:
        """Mark ID as successfully resolved."""
        self.resolved.add(id)

    def mark_failed(self, id: str) -> None:
        """Mark ID as failed to resolve."""
        self.failed.add(id)

    def is_resolved(self, id: str) -> bool:
        """Check if ID has been resolved."""
        return id in self.resolved

    def is_failed(self, id: str) -> bool:
        """Check if ID failed to resolve."""
        return id in self.failed


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def extract_id_from_label(obj: Any) -> str | None:
    """
    Extract ID/label from an object that has a 'label' attribute.

    Total function (returns None if no label).

    Maps to Lean:
        def extract_id_from_label (obj : α) : Option String :=
          obj.label
    """
    if hasattr(obj, "label"):
        return obj.label
    return None


def extract_tags_from_object(obj: Any) -> set[str]:
    """
    Extract tags from an object that has a 'tags' attribute.

    Returns empty set if no tags.

    Maps to Lean:
        def extract_tags_from_object (obj : α) : List String :=
          match obj.tags with
          | some tags => tags
          | none => []
    """
    if hasattr(obj, "tags"):
        tags = getattr(obj, "tags")
        if isinstance(tags, list | tuple | set):
            return set(tags)
    return set()


def batch_create_references(
    ids: list[str], type_hint: str | None = None
) -> list[UnresolvedReference[Any]]:
    """
    Helper: Create multiple unresolved references from IDs.

    Pure function (no side effects).
    """
    return [create_reference_id(id, type_hint) for id in ids]
