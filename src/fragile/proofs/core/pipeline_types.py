"""
Pipeline Execution Infrastructure for Theorem-Proving-as-Data-Processing.

This module contains the pipeline execution machinery that processes mathematical
objects through theorems. Mathematical entities themselves (objects, theorems,
properties, etc.) are defined in math_types.py.

This module provides:
- Result types (Ok, Err) for total functions
- Graph structures (DataFlow, Dependency) for visualization
- Pipeline state tracking for execution monitoring

For mathematical document entities, import from math_types.py.
For backward compatibility, this module re-exports all math_types.

All types follow Lean-compatible patterns from docs/LEAN_EMULATION_GUIDE.md:
- frozen=True (immutability)
- Pure functions (no side effects)
- Total functions (Optional[T] instead of exceptions)

Version: 2.0.0
"""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Re-export all mathematical types for backward compatibility
from fragile.proofs.core.math_types import (
    Attribute,
    AttributeEvent,
    AttributeEventType,
    AttributeRefinement,
    Axiom,
    create_simple_object,
    create_simple_theorem,
    MathematicalObject,
    ObjectType,
    Parameter,
    ParameterType,
    RefinementType,
    Relationship,
    RelationshipAttribute,
    RelationType,
    TheoremBox,
    TheoremOutputType,
)


# TypeVar for generic Result type
T = TypeVar("T")

# Make re-exported types available
__all__ = [
    "Attribute",
    "AttributeEvent",
    "AttributeEventType",
    "AttributeRefinement",
    "Axiom",
    "DataFlowEdge",
    "DataFlowNode",
    "DependencyEdge",
    "DependencyNode",
    "Err",
    "MathematicalObject",
    "ObjectType",
    # Pipeline execution types (defined in this module)
    "Ok",
    "Parameter",
    "ParameterType",
    "PipelineState",
    "RefinementType",
    "RelationType",
    "Relationship",
    "RelationshipAttribute",
    "TheoremBox",
    # Mathematical types (re-exported from math_types)
    "TheoremOutputType",
    "create_simple_object",
    "create_simple_theorem",
]


# =============================================================================
# RESULT TYPE (For Total Functions)
# =============================================================================


# =============================================================================


class Ok(BaseModel, Generic[T]):
    """Success result containing a value."""

    model_config = ConfigDict(frozen=True)
    value: T


class Err(BaseModel):
    """Error result containing error message."""

    model_config = ConfigDict(frozen=True)
    error: str


# Result[T] = Ok[T] | Err
# Use Union[Ok[T], Err] for type hints


# =============================================================================
# GRAPH STRUCTURES
# =============================================================================


class DataFlowNode(BaseModel):
    """Node in data flow graph."""

    model_config = ConfigDict(frozen=True)

    id: str  # object or theorem label
    type: Literal["object", "theorem"]
    current_properties: list[str] = Field(default_factory=list)


class DataFlowEdge(BaseModel):
    """Edge in data flow graph."""

    model_config = ConfigDict(frozen=True)

    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    flow_type: Literal["input", "output", "property_addition"]
    properties_added: list[str] = Field(default_factory=list)


class DependencyNode(BaseModel):
    """Node in dependency graph."""

    model_config = ConfigDict(frozen=True)

    id: str  # theorem, lemma, proposition, or axiom label
    type: Literal["theorem", "lemma", "proposition", "axiom"]


class DependencyEdge(BaseModel):
    """Edge in dependency graph."""

    model_config = ConfigDict(frozen=True)

    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    dependency_type: Literal["uses", "requires", "extends"]


# =============================================================================
# PIPELINE STATE
# =============================================================================


class PipelineState(BaseModel):
    """
    State of the pipeline during or after execution.

    Tracks:
    - Current state of all objects
    - Execution order
    - Blocked theorems (missing properties)
    - Full execution trace

    Maps to Lean:
        structure PipelineState where
          objects : HashMap String MathematicalObject
          executed_theorems : List String
          blocked_theorems : List (String × List String)
          execution_trace : List ExecutionStep
    """

    model_config = ConfigDict(frozen=True)

    objects: dict[str, MathematicalObject] = Field(default_factory=dict)
    executed_theorems: list[str] = Field(default_factory=list)
    blocked_theorems: list[tuple[str, list[str]]] = Field(default_factory=list)
    execution_trace: list[dict[str, object]] = Field(default_factory=list)

    # Pure function: Get object state
    def get_object_state(self, label: str) -> MathematicalObject | None:
        """
        Total function: Get current state of an object.

        Maps to Lean:
            def get_object_state (state : PipelineState) (label : String) : Option MathematicalObject :=
              state.objects.find? label
        """
        return self.objects.get(label)

    # Pure function: Check if theorem executed
    def is_executed(self, theorem_label: str) -> bool:
        """
        Pure function: Check if theorem has been executed.

        Maps to Lean:
            def is_executed (state : PipelineState) (label : String) : Bool :=
              label ∈ state.executed_theorems
        """
        return theorem_label in self.executed_theorems

    # Pure function: Add executed theorem (returns new state)
    def add_executed_theorem(self, theorem_label: str) -> PipelineState:
        """
        Pure function: Mark theorem as executed (immutable update).

        Maps to Lean:
            def add_executed_theorem (state : PipelineState) (label : String) : PipelineState :=
              { state with executed_theorems := state.executed_theorems ++ [label] }
        """
        return self.model_copy(
            update={"executed_theorems": [*self.executed_theorems, theorem_label]}
        )

    # Pure function: Update object (returns new state)
    def update_object(self, label: str, obj: MathematicalObject) -> PipelineState:
        """
        Pure function: Update object state (immutable update).

        Maps to Lean:
            def update_object
              (state : PipelineState)
              (label : String)
              (obj : MathematicalObject)
              : PipelineState :=
              { state with objects := state.objects.insert label obj }
        """
        new_objects = dict(self.objects)
        new_objects[label] = obj
        return self.model_copy(update={"objects": new_objects})
