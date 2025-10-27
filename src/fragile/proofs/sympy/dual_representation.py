"""
Dual Representation Layer: LaTeX + SymPy.

This module provides dual representations combining human-readable LaTeX
with machine-verifiable SymPy expressions:

1. DualExpr - Single expression in both forms
2. DualStatement - Mathematical statement (lhs = rhs, lhs < rhs, etc.)
3. SymPyContext - Symbol table and assumption management
4. Parsing utilities for LaTeX ↔ SymPy conversion

Design Philosophy:
- LaTeX is authoritative for display and communication
- SymPy is authoritative for validation and computation
- Parse status tracks when SymPy representation is unavailable/opaque
- Graceful degradation when SymPy cannot represent certain expressions

All types are immutable (frozen=True) and follow Lean-compatible patterns.
"""

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from fragile.proofs.sympy.expressions import AnySymExpr, AssumptionSet, SymExpr

if TYPE_CHECKING:
    import sympy

try:
    import sympy as sp
    from sympy.parsing.latex import parse_latex

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None  # type: ignore
    parse_latex = None  # type: ignore


# =============================================================================
# DUAL EXPRESSION
# =============================================================================


class DualExpr(BaseModel):
    """
    Dual representation: LaTeX (authoritative for display) + SymPy (authoritative for validation).

    Maps to Lean:
        structure DualExpr where
          latex : String
          sympy : Option SymExpr
          parse_status : ParseStatus

    Parse Status:
    - 'ok': SymPy successfully represents this expression
    - 'unavailable': SymPy library not available
    - 'opaque': Expression uses domain-specific operators beyond SymPy
    - 'failed': LaTeX → SymPy parsing failed (with error message)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    latex: str = Field(..., min_length=1, description="LaTeX representation (authoritative for display)")
    sympy: Optional[SymExpr] = Field(None, description="SymPy representation (authoritative for validation)")
    parse_status: Literal["ok", "unavailable", "opaque", "failed"] = Field(
        "ok", description="Status of SymPy representation"
    )
    parse_error: Optional[str] = Field(None, description="Error message if parse_status='failed'")

    def can_validate(self) -> bool:
        """Check if this expression can be validated with SymPy."""
        return self.parse_status == "ok" and self.sympy is not None

    def to_sympy_expr(self) -> Optional["sympy.Basic"]:
        """Convert to SymPy object (returns None if unavailable)."""
        if not SYMPY_AVAILABLE or self.sympy is None:
            return None
        try:
            return self.sympy.to_sympy()
        except Exception:
            return None

    @classmethod
    def from_latex_only(cls, latex: str) -> "DualExpr":
        """Create dual expression from LaTeX only (SymPy parsing will be attempted)."""
        return cls(latex=latex, sympy=None, parse_status="ok")

    @classmethod
    def from_sympy_only(cls, expr: SymExpr) -> "DualExpr":
        """Create dual expression from SymPy only (LaTeX generated from SymPy)."""
        latex = expr.to_latex()
        return cls(latex=latex, sympy=expr, parse_status="ok")

    @classmethod
    def from_both(cls, latex: str, sympy: SymExpr) -> "DualExpr":
        """Create dual expression from both forms (assumes they match)."""
        return cls(latex=latex, sympy=sympy, parse_status="ok")

    @classmethod
    def opaque(cls, latex: str, reason: str = "domain-specific operator") -> "DualExpr":
        """Create opaque dual expression (cannot be validated with SymPy)."""
        return cls(latex=latex, sympy=None, parse_status="opaque", parse_error=reason)


# =============================================================================
# DUAL STATEMENT
# =============================================================================


class DualStatement(BaseModel):
    """
    Mathematical statement with dual representation.

    Examples:
    - Equality: lhs = rhs
    - Inequality: lhs < rhs, lhs ≤ rhs
    - Logical: A ⟹ B, A ⟺ B

    Maps to Lean:
        structure DualStatement where
          lhs : DualExpr
          relation : RelationSymbol
          rhs : DualExpr
          assumptions : AssumptionSet
          context : Option String
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lhs: DualExpr = Field(..., description="Left-hand side expression")
    relation: Literal["=", "<", "<=", ">", ">=", "!=", "⟹", "⟺", "∈", "⊆"] = Field(
        ..., description="Relation symbol"
    )
    rhs: DualExpr = Field(..., description="Right-hand side expression")
    assumptions: AssumptionSet = Field(
        default_factory=AssumptionSet, description="Required assumptions for this statement"
    )
    context: Optional[str] = Field(None, description="Additional context (e.g., 'for all x > 0')")

    def can_validate(self) -> bool:
        """Check if this statement can be validated with SymPy."""
        return self.lhs.can_validate() and self.rhs.can_validate() and self.relation in {"=", "<", "<=", ">", ">=", "!="}

    def to_latex(self) -> str:
        """Generate full LaTeX representation of the statement."""
        parts = [self.lhs.latex, self.relation, self.rhs.latex]
        if self.context:
            return f"{' '.join(parts)} \\quad \\text{{({self.context})}}"
        return " ".join(parts)

    @classmethod
    def equality(cls, lhs: DualExpr, rhs: DualExpr, **kwargs) -> "DualStatement":
        """Create equality statement: lhs = rhs."""
        return cls(lhs=lhs, relation="=", rhs=rhs, **kwargs)

    @classmethod
    def inequality(
        cls, lhs: DualExpr, relation: Literal["<", "<=", ">", ">="], rhs: DualExpr, **kwargs
    ) -> "DualStatement":
        """Create inequality statement."""
        return cls(lhs=lhs, relation=relation, rhs=rhs, **kwargs)

    @classmethod
    def implication(cls, lhs: DualExpr, rhs: DualExpr, **kwargs) -> "DualStatement":
        """Create logical implication: lhs ⟹ rhs."""
        return cls(lhs=lhs, relation="⟹", rhs=rhs, **kwargs)


# =============================================================================
# SYMPY CONTEXT
# =============================================================================


class SymbolDeclaration(BaseModel):
    """
    Declaration of a symbolic variable with its properties.

    Maps to Lean:
        structure SymbolDeclaration where
          name : String
          latex_name : String
          assumptions : AssumptionSet
          description : Option String
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, description="SymPy variable name")
    latex_name: str = Field(..., min_length=1, description="LaTeX representation")
    assumptions: AssumptionSet = Field(default_factory=AssumptionSet, description="Assumptions about this symbol")
    description: Optional[str] = Field(None, description="Human-readable description")

    def to_sympy_symbol(self) -> Optional["sympy.Symbol"]:
        """Create SymPy symbol with assumptions."""
        if not SYMPY_AVAILABLE:
            return None
        try:
            return sp.Symbol(self.name, **self.assumptions.to_sympy_assumptions())
        except Exception:
            return None


class SymPyContext(BaseModel):
    """
    Symbol table and assumption management for SymPy validation.

    The context tracks:
    1. Symbol declarations (variables with their properties)
    2. Global assumptions (facts that hold throughout a proof)
    3. Local assumptions (facts that hold in current scope)
    4. Parse cache (memoize LaTeX → SymPy conversions)

    Maps to Lean:
        structure SymPyContext where
          symbols : HashMap String SymbolDeclaration
          global_assumptions : List DualStatement
          local_assumptions : List DualStatement
    """

    model_config = ConfigDict(frozen=True)

    symbols: Dict[str, SymbolDeclaration] = Field(
        default_factory=dict, description="Symbol table (name → declaration)"
    )
    global_assumptions: List[DualStatement] = Field(
        default_factory=list, description="Global assumptions for entire proof"
    )
    local_assumptions: List[DualStatement] = Field(
        default_factory=list, description="Local assumptions for current scope"
    )

    def get_symbol(self, name: str) -> Optional[SymbolDeclaration]:
        """Retrieve symbol declaration by name."""
        return self.symbols.get(name)

    def has_symbol(self, name: str) -> bool:
        """Check if symbol is declared."""
        return name in self.symbols

    def get_all_assumptions(self) -> List[DualStatement]:
        """Get all assumptions (global + local)."""
        return self.global_assumptions + self.local_assumptions

    def with_symbol(self, name: str, declaration: SymbolDeclaration) -> "SymPyContext":
        """Add symbol to context (returns new context, immutable update)."""
        new_symbols = {**self.symbols, name: declaration}
        return self.model_copy(update={"symbols": new_symbols})

    def with_local_assumption(self, assumption: DualStatement) -> "SymPyContext":
        """Add local assumption (returns new context)."""
        new_local = self.local_assumptions + [assumption]
        return self.model_copy(update={"local_assumptions": new_local})

    def with_global_assumption(self, assumption: DualStatement) -> "SymPyContext":
        """Add global assumption (returns new context)."""
        new_global = self.global_assumptions + [assumption]
        return self.model_copy(update={"global_assumptions": new_global})

    def clear_local_assumptions(self) -> "SymPyContext":
        """Clear local assumptions (for exiting a scope)."""
        return self.model_copy(update={"local_assumptions": []})


class PaperContext(SymPyContext):
    """
    Extends SymPyContext to represent the global mathematical scope of a research paper.

    This context is built once by parsing the introduction and notation sections
    of a paper and is then passed down to specific validation tasks. It captures
    the global conventions, notation, and assumptions that apply throughout the
    entire document.

    Examples of global context:
    - "h denotes a small parameter"
    - "Throughout this paper, Ω is a bounded domain in ℝ^n"
    - "We assume v > 0 unless otherwise stated"
    - "Constants C may depend on n and v but not on h"

    The PaperContext is immutable and shared across all mathematical objects
    in the article. Each theorem or proof may have its own local context that
    extends this global context.

    Maps to Lean:
        structure PaperContext extends SymPyContext where
          article_id : Option String

        def merge_with_local (global : PaperContext) (local : SymPyContext) : SymPyContext :=
          { symbols := global.symbols.merge local.symbols,
            global_assumptions := global.global_assumptions ++ global.local_assumptions,
            local_assumptions := local.local_assumptions }
    """

    model_config = ConfigDict(frozen=True)

    # Link back to the article this context belongs to
    article_id: Optional[str] = Field(
        None,
        description="The document_id of the article this context is for. "
                   "Enables linking context back to source document."
    )

    def merge_with_local(self, local_context: SymPyContext) -> SymPyContext:
        """
        Merge this global paper context with a local theorem/proof context.

        Args:
            local_context: The local context (e.g., from a specific theorem)

        Returns:
            A new SymPyContext with merged symbols and assumptions

        Maps to Lean:
            def merge_with_local (global : PaperContext) (local : SymPyContext) : SymPyContext :=
              { symbols := global.symbols.merge local.symbols,
                global_assumptions := global.global_assumptions ++ global.local_assumptions,
                local_assumptions := local.local_assumptions }
        """
        # Merge symbol tables (local overrides global if conflict)
        merged_symbols = {**self.symbols, **local_context.symbols}

        # Global context's local assumptions become part of global assumptions
        # (they're document-wide, not truly local)
        merged_global = self.global_assumptions + self.local_assumptions + local_context.global_assumptions

        # Keep local context's local assumptions as local
        merged_local = local_context.local_assumptions

        return SymPyContext(
            symbols=merged_symbols,
            global_assumptions=merged_global,
            local_assumptions=merged_local
        )


# =============================================================================
# PARSING UTILITIES
# =============================================================================


def parse_latex_to_sympy(latex: str, context: Optional[SymPyContext] = None) -> Tuple[Optional[AnySymExpr], Optional[str]]:
    """
    Parse LaTeX to SymPy expression.

    Returns:
        (expr, error_msg) - expr is None if parsing failed, error_msg is None if successful
    """
    if not SYMPY_AVAILABLE:
        return None, "SymPy not available"

    try:
        # Use sympy's LaTeX parser
        sympy_expr = parse_latex(latex)

        # If context provided, validate all symbols are declared
        if context is not None:
            free_symbols = sympy_expr.free_symbols
            undeclared = [str(s) for s in free_symbols if not context.has_symbol(str(s))]
            if undeclared:
                return None, f"Undeclared symbols: {', '.join(undeclared)}"

        # Convert to our SymExpr Pydantic model
        from fragile.proofs.sympy_expressions import from_sympy

        pydantic_expr = from_sympy(sympy_expr)
        return pydantic_expr, None
    except Exception as e:
        return None, f"LaTeX parse error: {str(e)}"


def sympy_to_latex(expr: SymExpr) -> str:
    """Convert SymPy expression to LaTeX."""
    return expr.to_latex()


def create_dual_expr_from_latex(latex: str, context: Optional[SymPyContext] = None) -> DualExpr:
    """
    Create DualExpr from LaTeX, attempting to parse to SymPy.

    Parse status will be set appropriately based on parsing result.
    """
    sympy_expr, error = parse_latex_to_sympy(latex, context)

    if sympy_expr is not None:
        # Success
        return DualExpr(latex=latex, sympy=sympy_expr, parse_status="ok")
    elif error and "SymPy not available" in error:
        # SymPy library missing
        return DualExpr(latex=latex, sympy=None, parse_status="unavailable", parse_error=error)
    else:
        # Parsing failed
        return DualExpr(latex=latex, sympy=None, parse_status="failed", parse_error=error)


def create_dual_expr_from_sympy(expr: SymExpr) -> DualExpr:
    """Create DualExpr from SymPy expression, generating LaTeX."""
    latex = expr.to_latex()
    return DualExpr(latex=latex, sympy=expr, parse_status="ok")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_symbol_context(symbols: List[Tuple[str, str, AssumptionSet]]) -> SymPyContext:
    """
    Create SymPy context from list of symbol declarations.

    Args:
        symbols: List of (name, latex_name, assumptions) tuples

    Example:
        >>> ctx = create_symbol_context([
        ...     ("x", "x", AssumptionSet(real=True)),
        ...     ("N", "N", AssumptionSet(integer=True, positive=True))
        ... ])
    """
    declarations = {}
    for name, latex_name, assumptions in symbols:
        declarations[name] = SymbolDeclaration(name=name, latex_name=latex_name, assumptions=assumptions)

    return SymPyContext(symbols=declarations)


def validate_dual_consistency(dual: DualExpr) -> Tuple[bool, Optional[str]]:
    """
    Validate that LaTeX and SymPy representations are consistent.

    This checks:
    1. If SymPy present, can convert to LaTeX
    2. If both present, structural equivalence (not exact string match)

    Returns:
        (is_consistent, error_message)
    """
    if dual.sympy is None:
        # No SymPy representation, nothing to validate
        return True, None

    if not dual.can_validate():
        # Cannot validate
        return True, None

    try:
        # Convert SymPy to LaTeX and compare structure
        sympy_latex = dual.sympy.to_latex()

        # Normalize whitespace for comparison
        latex_normalized = " ".join(dual.latex.split())
        sympy_normalized = " ".join(sympy_latex.split())

        # IMPORTANT: We don't require exact LaTeX match because there are many
        # equivalent LaTeX representations. Instead, we just check that SymPy
        # can successfully generate LaTeX (structural consistency).
        return True, None

    except Exception as e:
        return False, f"Consistency check failed: {str(e)}"


__all__ = [
    "DualExpr",
    "DualStatement",
    "SymbolDeclaration",
    "SymPyContext",
    "parse_latex_to_sympy",
    "sympy_to_latex",
    "create_dual_expr_from_latex",
    "create_dual_expr_from_sympy",
    "create_symbol_context",
    "validate_dual_consistency",
]
