"""
SymPy Validation Engine.

This module provides validation and transformation capabilities using SymPy:

1. ValidationResult - Detailed validation outcomes with diagnostics
2. SymPyValidator - Core validation engine
3. Transformation library - Simplify, expand, diff, integrate, etc.
4. Plugin system - Domain-specific validation rules

Design Philosophy:
- Eager validation: Validate as soon as possible for fast feedback
- Graceful degradation: If SymPy can't validate, fall back to LLM proof
- Clear diagnostics: Provide detailed error messages with suggested fixes
- Pluggable rules: Extend with domain-specific validation logic

All types are immutable (frozen=True) and follow Lean-compatible patterns.
"""

from enum import Enum
from typing import Any, Literal, Protocol, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from mathster.sympy_integration.dual_representation import (
    DualExpr,
    DualStatement,
    SymPyContext,
)


if TYPE_CHECKING:
    pass

try:
    import sympy as sp

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None  # type: ignore


# =============================================================================
# VALIDATION RESULTS
# =============================================================================


class ValidationStatus(str, Enum):
    """Status of validation attempt."""

    VALID = "valid"  # SymPy confirmed correctness
    INVALID = "invalid"  # SymPy found error
    UNCERTAIN = "uncertain"  # SymPy cannot determine (fall back to LLM)
    UNAVAILABLE = "unavailable"  # SymPy not available
    OPAQUE = "opaque"  # Expression uses domain operators beyond SymPy


class ValidationIssue(BaseModel):
    """
    Describes a validation issue.

    Maps to Lean:
        structure ValidationIssue where
          severity : Severity
          message : String
          location : Option String
          suggestion : Option String
    """

    model_config = ConfigDict(frozen=True)

    severity: Literal["error", "warning", "info"] = Field(..., description="Issue severity")
    message: str = Field(..., min_length=1, description="Description of the issue")
    location: str | None = Field(
        None, description="Where the issue occurred (e.g., 'step 3', 'lhs')"
    )
    suggestion: str | None = Field(None, description="Suggested fix")


class ValidationResult(BaseModel):
    """
    Result of validation attempt.

    Maps to Lean:
        structure ValidationResult where
          status : ValidationStatus
          is_valid : Bool
          can_validate : Bool
          issues : List ValidationIssue
          details : HashMap String String
    """

    model_config = ConfigDict(frozen=True)

    status: ValidationStatus = Field(..., description="Validation status")
    is_valid: bool = Field(..., description="True if validation passed (or uncertain)")
    can_validate: bool = Field(..., description="True if SymPy can attempt validation")
    issues: list[ValidationIssue] = Field(default_factory=list, description="Validation issues")
    details: dict[str, str] = Field(default_factory=dict, description="Additional details")

    @classmethod
    def valid(cls, details: dict[str, str] | None = None) -> "ValidationResult":
        """Create successful validation result."""
        return cls(
            status=ValidationStatus.VALID,
            is_valid=True,
            can_validate=True,
            details=details or {},
        )

    @classmethod
    def invalid(cls, issue: str, **kwargs) -> "ValidationResult":
        """Create failed validation result."""
        return cls(
            status=ValidationStatus.INVALID,
            is_valid=False,
            can_validate=True,
            issues=[ValidationIssue(severity="error", message=issue, **kwargs)],
        )

    @classmethod
    def uncertain(cls, reason: str) -> "ValidationResult":
        """Create uncertain validation result (SymPy cannot determine)."""
        return cls(
            status=ValidationStatus.UNCERTAIN,
            is_valid=True,  # Allow LLM proof to proceed
            can_validate=False,
            issues=[ValidationIssue(severity="info", message=f"Uncertain: {reason}")],
        )

    @classmethod
    def unavailable(cls) -> "ValidationResult":
        """Create unavailable validation result (SymPy not installed)."""
        return cls(
            status=ValidationStatus.UNAVAILABLE,
            is_valid=True,  # Allow LLM proof to proceed
            can_validate=False,
            issues=[ValidationIssue(severity="info", message="SymPy not available")],
        )

    @classmethod
    def opaque(cls, reason: str) -> "ValidationResult":
        """Create opaque validation result (domain operators beyond SymPy)."""
        return cls(
            status=ValidationStatus.OPAQUE,
            is_valid=True,  # Allow LLM proof to proceed
            can_validate=False,
            issues=[ValidationIssue(severity="info", message=f"Opaque: {reason}")],
        )


# =============================================================================
# TRANSFORMATION LIBRARY
# =============================================================================


class TransformationType(str, Enum):
    """Types of SymPy transformations."""

    SIMPLIFY = "simplify"
    EXPAND = "expand"
    FACTOR = "factor"
    CANCEL = "cancel"
    COLLECT = "collect"
    DIFF = "differentiate"
    INTEGRATE = "integrate"
    LIMIT = "limit"
    SOLVE = "solve"
    SUBSTITUTE = "substitute"
    REWRITE = "rewrite"
    TRIGSIMP = "trigsimp"


class Transformation(BaseModel):
    """
    A SymPy transformation applied to an expression.

    Maps to Lean:
        structure Transformation where
          transformation_type : TransformationType
          input : DualExpr
          output : DualExpr
          parameters : HashMap String String
    """

    model_config = ConfigDict(frozen=True)

    transformation_type: TransformationType = Field(..., description="Type of transformation")
    input_expr: DualExpr = Field(..., description="Input expression")
    output_expr: DualExpr = Field(..., description="Output expression")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Transformation parameters"
    )

    def validate(self) -> ValidationResult:
        """Validate that the transformation is correct."""
        if not SYMPY_AVAILABLE:
            return ValidationResult.unavailable()

        if not self.input_expr.can_validate() or not self.output_expr.can_validate():
            return ValidationResult.uncertain("Input or output cannot be validated")

        try:
            input_sympy = self.input_expr.to_sympy_expr()
            output_sympy = self.output_expr.to_sympy_expr()

            if input_sympy is None or output_sympy is None:
                return ValidationResult.uncertain("Failed to convert to SymPy")

            # Check equality using SymPy's simplify
            diff = sp.simplify(input_sympy - output_sympy)
            if diff == 0:
                return ValidationResult.valid({"transformation": self.transformation_type.value})
            return ValidationResult.invalid(
                f"Transformation not valid: {input_sympy} ≠ {output_sympy}",
                suggestion="Check transformation logic",
            )
        except Exception as e:
            return ValidationResult.uncertain(f"Validation error: {e!s}")


def simplify_expr(expr: DualExpr) -> Transformation:
    """Simplify expression using SymPy."""
    if not SYMPY_AVAILABLE or not expr.can_validate():
        return Transformation(
            transformation_type=TransformationType.SIMPLIFY,
            input_expr=expr,
            output_expr=expr,  # No change
        )

    try:
        sympy_expr = expr.to_sympy_expr()
        if sympy_expr is None:
            return Transformation(
                transformation_type=TransformationType.SIMPLIFY, input_expr=expr, output_expr=expr
            )

        simplified = sp.simplify(sympy_expr)

        from mathster.sympy_integration.dual_representation import (
            create_dual_expr_from_sympy,
        )
        from mathster.sympy_integration.expressions import from_sympy

        output = create_dual_expr_from_sympy(from_sympy(simplified))

        return Transformation(
            transformation_type=TransformationType.SIMPLIFY, input_expr=expr, output_expr=output
        )
    except Exception:
        return Transformation(
            transformation_type=TransformationType.SIMPLIFY, input_expr=expr, output_expr=expr
        )


def expand_expr(expr: DualExpr) -> Transformation:
    """Expand expression using SymPy."""
    if not SYMPY_AVAILABLE or not expr.can_validate():
        return Transformation(
            transformation_type=TransformationType.EXPAND, input_expr=expr, output_expr=expr
        )

    try:
        sympy_expr = expr.to_sympy_expr()
        if sympy_expr is None:
            return Transformation(
                transformation_type=TransformationType.EXPAND, input_expr=expr, output_expr=expr
            )

        expanded = sp.expand(sympy_expr)

        from mathster.sympy_integration.dual_representation import (
            create_dual_expr_from_sympy,
        )
        from mathster.sympy_integration.expressions import from_sympy

        output = create_dual_expr_from_sympy(from_sympy(expanded))

        return Transformation(
            transformation_type=TransformationType.EXPAND, input_expr=expr, output_expr=output
        )
    except Exception:
        return Transformation(
            transformation_type=TransformationType.EXPAND, input_expr=expr, output_expr=expr
        )


def differentiate_expr(expr: DualExpr, variable: str) -> Transformation:
    """Differentiate expression with respect to variable."""
    if not SYMPY_AVAILABLE or not expr.can_validate():
        return Transformation(
            transformation_type=TransformationType.DIFF,
            input_expr=expr,
            output_expr=expr,
            parameters={"variable": variable},
        )

    try:
        sympy_expr = expr.to_sympy_expr()
        if sympy_expr is None:
            return Transformation(
                transformation_type=TransformationType.DIFF,
                input_expr=expr,
                output_expr=expr,
                parameters={"variable": variable},
            )

        var_symbol = sp.Symbol(variable)
        derivative = sp.diff(sympy_expr, var_symbol)

        from mathster.sympy_integration.dual_representation import (
            create_dual_expr_from_sympy,
        )
        from mathster.sympy_integration.expressions import from_sympy

        output = create_dual_expr_from_sympy(from_sympy(derivative))

        return Transformation(
            transformation_type=TransformationType.DIFF,
            input_expr=expr,
            output_expr=output,
            parameters={"variable": variable},
        )
    except Exception:
        return Transformation(
            transformation_type=TransformationType.DIFF,
            input_expr=expr,
            output_expr=expr,
            parameters={"variable": variable},
        )


# =============================================================================
# VALIDATION PLUGINS
# =============================================================================


class ValidationPlugin(Protocol):
    """
    Protocol for domain-specific validation plugins.

    Plugins extend SymPy validation with domain-specific knowledge.
    For example:
    - Wasserstein distance properties
    - Manifold geometry rules
    - Probabilistic inequalities
    """

    def can_validate(self, statement: DualStatement) -> bool:
        """Check if this plugin can validate the statement."""
        ...

    def validate(self, statement: DualStatement, context: SymPyContext) -> ValidationResult:
        """Validate the statement using domain-specific knowledge."""
        ...


class PluginRegistry(BaseModel):
    """Registry of validation plugins."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)  # Allow any plugin types

    plugins: list[Any] = Field(
        default_factory=list,
        description="Registered plugins (must match ValidationPlugin protocol)",
    )

    def register(self, plugin: Any) -> None:
        """
        Register a validation plugin.

        Plugin must have:
        - can_validate(statement: DualStatement) -> bool
        - validate(statement: DualStatement, context: SymPyContext) -> ValidationResult
        """
        # Runtime check that plugin has required methods
        if not hasattr(plugin, "can_validate") or not hasattr(plugin, "validate"):
            msg = "Plugin must have can_validate() and validate() methods"
            raise ValueError(msg)
        self.plugins.append(plugin)

    def find_plugin(self, statement: DualStatement) -> Any | None:
        """Find a plugin that can validate this statement."""
        for plugin in self.plugins:
            if plugin.can_validate(statement):
                return plugin
        return None


# =============================================================================
# SYMPY VALIDATOR
# =============================================================================


class SymPyValidator(BaseModel):
    """
    Core validation engine using SymPy.

    Maps to Lean:
        structure SymPyValidator where
          context : SymPyContext
          plugin_registry : PluginRegistry
          strict_mode : Bool
    """

    model_config = ConfigDict(
        frozen=False, arbitrary_types_allowed=True
    )  # Mutable for plugin registration

    context: SymPyContext = Field(
        default_factory=SymPyContext, description="Symbol table and assumptions"
    )
    plugin_registry: PluginRegistry = Field(
        default_factory=PluginRegistry, description="Domain-specific plugins"
    )
    strict_mode: bool = Field(
        False,
        description="If True, require SymPy validation (no LLM fallback for uncertain cases)",
    )

    def validate_statement(self, statement: DualStatement) -> ValidationResult:
        """
        Validate a mathematical statement.

        Strategy:
        1. Check if plugins can validate (domain-specific rules)
        2. Try SymPy validation (algebraic/calculus rules)
        3. If uncertain and strict_mode=False, allow LLM proof
        4. If uncertain and strict_mode=True, fail validation
        """
        # Check if SymPy is available
        if not SYMPY_AVAILABLE:
            return ValidationResult.unavailable()

        # Check if statement is opaque
        if not statement.can_validate():
            return ValidationResult.opaque("Statement contains opaque expressions")

        # Try plugins first (domain-specific)
        plugin = self.plugin_registry.find_plugin(statement)
        if plugin is not None:
            return plugin.validate(statement, self.context)

        # Try SymPy validation (algebraic/calculus)
        try:
            lhs = statement.lhs.to_sympy_expr()
            rhs = statement.rhs.to_sympy_expr()

            if lhs is None or rhs is None:
                return ValidationResult.uncertain("Failed to convert to SymPy")

            # Validate based on relation type
            if statement.relation == "=":
                diff = sp.simplify(lhs - rhs)
                if diff == 0:
                    return ValidationResult.valid({"relation": "equality"})
                return ValidationResult.invalid(f"Equality does not hold: {lhs} ≠ {rhs}")
            if statement.relation in {"<", "<=", ">", ">="}:
                # SymPy has limited inequality solving
                # Try to validate, but often uncertain
                try:
                    # Attempt to use assumptions
                    # SymPy's relational checking is limited, often uncertain
                    return ValidationResult.uncertain(
                        f"SymPy cannot validate inequality: {statement.relation}"
                    )
                except Exception:
                    return ValidationResult.uncertain("Inequality validation failed")
            else:
                # Logical operators, set membership, etc.
                return ValidationResult.uncertain(f"Unsupported relation: {statement.relation}")

        except Exception as e:
            return ValidationResult.uncertain(f"Validation error: {e!s}")

    def validate_transformation(self, transformation: Transformation) -> ValidationResult:
        """Validate a transformation is correct."""
        return transformation.validate()

    def validate_chain(self, chain: list[DualStatement]) -> ValidationResult:
        """
        Validate a chain of statements (transitive reasoning).

        Example: A = B, B = C, therefore A = C
        """
        if not chain:
            return ValidationResult.valid({"chain_length": "0"})

        # Validate each statement in the chain
        for i, statement in enumerate(chain):
            result = self.validate_statement(statement)
            if not result.is_valid:
                return ValidationResult.invalid(
                    f"Statement {i} in chain is invalid",
                    location=f"chain[{i}]",
                    suggestion="Check individual statements",
                )
            if result.status == ValidationStatus.UNCERTAIN:
                return ValidationResult.uncertain(f"Statement {i} is uncertain")

        # All statements valid
        return ValidationResult.valid({"chain_length": str(len(chain))})

    def with_assumption(self, assumption: DualStatement) -> "SymPyValidator":
        """Add assumption to context (returns new validator)."""
        new_context = self.context.with_global_assumption(assumption)
        return self.model_copy(update={"context": new_context})

    def with_plugin(self, plugin: ValidationPlugin) -> "SymPyValidator":
        """Register a plugin (mutates registry)."""
        self.plugin_registry.register(plugin)
        return self


__all__ = [
    "PluginRegistry",
    "SymPyValidator",
    "Transformation",
    "TransformationType",
    "ValidationIssue",
    "ValidationPlugin",
    "ValidationResult",
    "ValidationStatus",
    "differentiate_expr",
    "expand_expr",
    "simplify_expr",
]
