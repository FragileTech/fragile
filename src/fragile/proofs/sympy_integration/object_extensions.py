"""
SymPy Extensions for MathematicalObject and Attribute.

This module provides SymPy-enabled extensions of core pipeline types:

1. PropertyWithSymPy - Attribute extended with dual representation
2. MathematicalObjectWithSymPy - Object extended with symbol table
3. Helper functions for context creation and validation

Design Philosophy:
- Non-invasive extension: Existing types work without SymPy
- Opt-in enhancement: Add SymPy when it provides value
- Clean API: Hide complexity behind simple methods

All types follow Lean-compatible patterns.
"""

from pydantic import ConfigDict, Field

from fragile.proofs.core.pipeline_types import (
    Attribute,
    AttributeEvent,
    MathematicalObject,
    ObjectType,
)
from fragile.proofs.sympy_integration.dual_representation import (
    DualExpr,
    SymbolDeclaration,
    SymPyContext,
)
from fragile.proofs.sympy_integration.expressions import AssumptionSet


# =============================================================================
# EXTENDED PROPERTY
# =============================================================================


class PropertyWithSymPy(Attribute):
    """
    Attribute extended with SymPy dual representation.

    This is a non-invasive extension: if dual_expression is None,
    it behaves identically to Attribute.

    Maps to Lean:
        structure PropertyWithSymPy extends Attribute where
          dual_expression : Option DualExpr
          assumption_set : AssumptionSet
    """

    model_config = ConfigDict(frozen=True)

    dual_expression: DualExpr | None = Field(
        None,
        description="Dual representation (LaTeX + SymPy) of the property expression",
    )
    assumption_set: AssumptionSet = Field(
        default_factory=AssumptionSet,
        description="SymPy-compatible assumptions about this property",
    )

    def can_validate_with_sympy(self) -> bool:
        """Check if this property can be validated with SymPy."""
        return self.dual_expression is not None and self.dual_expression.can_validate()


# =============================================================================
# EXTENDED MATHEMATICAL OBJECT
# =============================================================================


class MathematicalObjectWithSymPy(MathematicalObject):
    """
    MathematicalObject extended with SymPy support.

    This is a non-invasive extension that adds:
    - Symbol table for SymPy validation
    - Global assumptions for this object
    - Helper methods for context creation

    If symbol_table is empty, behaves identically to MathematicalObject.

    Maps to Lean:
        structure MathematicalObjectWithSymPy extends MathematicalObject where
          symbol_table : HashMap String SymbolDeclaration
          global_assumptions : AssumptionSet
    """

    model_config = ConfigDict(frozen=True)

    symbol_table: dict[str, SymbolDeclaration] = Field(
        default_factory=dict,
        description="Symbol declarations for SymPy validation",
    )
    global_assumptions: AssumptionSet = Field(
        default_factory=AssumptionSet,
        description="Global assumptions that hold for this object",
    )

    def has_sympy_support(self) -> bool:
        """Check if this object has SymPy symbol table."""
        return bool(self.symbol_table)

    def get_symbol(self, name: str) -> SymbolDeclaration | None:
        """Get symbol declaration by name."""
        return self.symbol_table.get(name)

    def get_all_symbol_names(self) -> set[str]:
        """Get all declared symbol names."""
        return set(self.symbol_table.keys())

    def create_sympy_context(self) -> SymPyContext:
        """
        Create SymPy context from this object's symbol table.

        This is useful for validating proofs involving this object.
        """
        return SymPyContext(symbols=self.symbol_table)

    def add_symbol(
        self,
        name: str,
        declaration: SymbolDeclaration,
    ) -> "MathematicalObjectWithSymPy":
        """
        Add symbol to symbol table (returns new object, immutable update).

        Maps to Lean:
            def add_symbol (obj : MathObjectSymPy) (name : String) (decl : SymbolDecl) : MathObjectSymPy :=
              { obj with symbol_table := obj.symbol_table.insert name decl }
        """
        new_table = {**self.symbol_table, name: declaration}
        return self.model_copy(update={"symbol_table": new_table})

    def with_assumptions(self, assumptions: AssumptionSet) -> "MathematicalObjectWithSymPy":
        """
        Set global assumptions (returns new object).

        Maps to Lean:
            def with_assumptions (obj : MathObjectSymPy) (assump : AssumptionSet) : MathObjectSymPy :=
              { obj with global_assumptions := assump }
        """
        return self.model_copy(update={"global_assumptions": assumptions})

    # Override add_property to work with PropertyWithSymPy
    def add_property_with_sympy(
        self,
        prop: PropertyWithSymPy,
        timestamp: int,
    ) -> "MathematicalObjectWithSymPy":
        """
        Add SymPy-enhanced property to object (immutable update).

        This is like add_property but handles PropertyWithSymPy.
        """
        # Create property event
        from fragile.proofs.core.pipeline_types import AttributeEventType

        event = AttributeEvent(
            event_type=AttributeEventType.PROPERTY_ADDED,
            property_label=prop.label,
            theorem_label=prop.established_by,
            timestamp=timestamp,
        )

        # Update properties and history
        new_properties = [*self.current_attributes, prop]
        new_history = [*self.property_history, event]

        return self.model_copy(
            update={
                "current_attributes": new_properties,
                "property_history": new_history,
            }
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_object_with_symbols(
    label: str,
    name: str,
    mathematical_expression: str,
    object_type: ObjectType,
    symbols: list[tuple[str, str, AssumptionSet]],
    **kwargs,
) -> MathematicalObjectWithSymPy:
    r"""
    Create MathematicalObjectWithSymPy with symbol table.

    Args:
        label: Object ID (e.g., 'obj-discrete-system')
        name: Human-readable name
        mathematical_expression: LaTeX expression defining the object
        object_type: Type of mathematical object
        symbols: List of (name, latex_name, assumptions) tuples
        **kwargs: Additional fields for MathematicalObject

    Example:
        >>> obj = create_object_with_symbols(
        ...     label="obj-euclidean-space",
        ...     name="Euclidean Space",
        ...     mathematical_expression=r"\mathbb{R}^d",
        ...     object_type=ObjectType.SET,
        ...     symbols=[
        ...         ("x", "x", AssumptionSet(real=True)),
        ...         ("d", "d", AssumptionSet(integer=True, positive=True)),
        ...     ],
        ... )
    """
    # Create symbol table
    symbol_table = {}
    for sym_name, latex_name, assumptions in symbols:
        symbol_table[sym_name] = SymbolDeclaration(
            name=sym_name,
            latex_name=latex_name,
            assumptions=assumptions,
        )

    return MathematicalObjectWithSymPy(
        label=label,
        name=name,
        mathematical_expression=mathematical_expression,
        object_type=object_type,
        symbol_table=symbol_table,
        **kwargs,
    )


def create_property_with_dual_expr(
    label: str,
    expression: str,
    object_label: str,
    established_by: str,
    dual_expr: DualExpr | None = None,
    assumption_set: AssumptionSet | None = None,
    **kwargs,
) -> PropertyWithSymPy:
    r"""
    Create PropertyWithSymPy with dual representation.

    Args:
        label: Attribute ID (e.g., 'prop-lipschitz-continuity')
        expression: LaTeX expression (authoritative for display)
        object_label: Object this property belongs to
        established_by: Theorem that established this property
        dual_expr: Optional SymPy dual representation
        assumption_set: Optional assumptions for SymPy validation
        **kwargs: Additional fields for Attribute

    Example:
        >>> from fragile.proofs import create_dual_expr_from_latex
        >>> dual = create_dual_expr_from_latex(r"f'(x) \leq L")
        >>> prop = create_property_with_dual_expr(
        ...     label="prop-lipschitz",
        ...     expression=r"|f'(x)| \leq L",
        ...     object_label="obj-function-f",
        ...     established_by="thm-lipschitz",
        ...     dual_expr=dual,
        ... )
    """
    return PropertyWithSymPy(
        label=label,
        expression=expression,
        object_label=object_label,
        established_by=established_by,
        dual_expression=dual_expr,
        assumption_set=assumption_set or AssumptionSet(),
        **kwargs,
    )


def merge_symbol_tables(
    objects: list[MathematicalObjectWithSymPy],
) -> dict[str, SymbolDeclaration]:
    """
    Merge symbol tables from multiple objects.

    Useful for creating a unified SymPy context for proofs involving
    multiple objects.

    Raises:
        ValueError: If symbol names conflict with different declarations
    """
    merged: dict[str, SymbolDeclaration] = {}

    for obj in objects:
        for name, declaration in obj.symbol_table.items():
            if name in merged:
                # Check if declarations are compatible
                if merged[name] != declaration:
                    raise ValueError(
                        f"Symbol '{name}' has conflicting declarations:\n"
                        f"  From {obj.label}: {declaration}\n"
                        f"  Already merged: {merged[name]}"
                    )
            else:
                merged[name] = declaration

    return merged


def create_combined_sympy_context(
    objects: list[MathematicalObjectWithSymPy],
) -> SymPyContext:
    """
    Create unified SymPy context from multiple objects.

    This merges symbol tables and can be used for validating
    proofs that involve multiple objects.

    Example:
        >>> obj1 = create_object_with_symbols(...)
        >>> obj2 = create_object_with_symbols(...)
        >>> ctx = create_combined_sympy_context([obj1, obj2])
        >>> validator = SymPyValidator(context=ctx)
    """
    merged_symbols = merge_symbol_tables(objects)
    return SymPyContext(symbols=merged_symbols)


__all__ = [
    "MathematicalObjectWithSymPy",
    "PropertyWithSymPy",
    "create_combined_sympy_context",
    "create_object_with_symbols",
    "create_property_with_dual_expr",
    "merge_symbol_tables",
]
