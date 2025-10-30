"""
SymPy Expression Models - Pydantic representations of SymPy AST.

This module provides type-safe, serializable Pydantic models that mirror SymPy's
expression tree structure. Each model can convert to/from live SymPy objects while
maintaining JSON serializability for storage and transmission.

Design Philosophy:
- Dual representation: Pydantic models (serializable) ↔ SymPy objects (computational)
- Type safety: Discriminated union with literal type fields
- Extensibility: Easy to add new expression types
- Lean-compatible: Pure data structures, conversion methods are pure functions

All types follow Lean-compatible patterns:
- frozen=True (immutability)
- Pure functions (to_sympy, from_sympy)
- Explicit types (no Any except in OpExpr.attrs)

Version: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field
import sympy


# =============================================================================
# ENUMS
# =============================================================================


class RelationOp(str, Enum):
    """Relational operators."""

    EQ = "Eq"  # ==
    NE = "Ne"  # !=
    LT = "Lt"  # <
    LE = "Le"  # <=
    GT = "Gt"  # >
    GE = "Ge"  # >=


class LogicOp(str, Enum):
    """Logical operators."""

    AND = "And"
    OR = "Or"
    NOT = "Not"
    IMPLIES = "Implies"
    EQUIVALENT = "Equivalent"


class DomainOp(str, Enum):
    """Domain-specific operators (opaque to SymPy, handled by plugins)."""

    # Differential geometry
    GRAD = "Grad"
    DIV = "Div"
    LAPLACE_BELTRAMI = "LaplaceBeltrami"
    PUSHFORWARD = "Pushforward"
    PULLBACK = "Pullback"

    # Probability & measure theory
    EXPECTATION = "Expectation"
    VARIANCE = "Variance"
    PROBABILITY = "Probability"
    LAW = "Law"

    # Information theory
    KL_DIVERGENCE = "KL"
    ENTROPY = "Entropy"
    MUTUAL_INFORMATION = "MutualInformation"

    # Stochastic calculus
    ITO_INTEGRAL = "ItoIntegral"
    STRATONOVICH_INTEGRAL = "StratonovichIntegral"
    QUADRATIC_VARIATION = "QuadraticVariation"

    # Optimal transport
    WASSERSTEIN = "Wasserstein"
    WASSERSTEIN_2 = "Wasserstein2"
    OPTIMAL_TRANSPORT_MAP = "OptimalTransportMap"


# =============================================================================
# ASSUMPTION SYSTEM
# =============================================================================


class AssumptionSet(BaseModel):
    """
    Mathematical assumptions about symbols/expressions.

    Maps to SymPy's Q.* assumptions where possible.
    Also includes framework-specific properties that SymPy doesn't understand.

    Maps to Lean:
        structure AssumptionSet where
          real : Bool
          positive : Bool
          ...
    """

    model_config = ConfigDict(frozen=True)

    # SymPy-compatible assumptions (map to Q.*)
    real: bool | None = None
    positive: bool | None = None
    negative: bool | None = None
    zero: bool | None = None
    nonzero: bool | None = None
    integer: bool | None = None
    rational: bool | None = None
    complex: bool | None = None
    finite: bool | None = None
    infinite: bool | None = None
    even: bool | None = None
    odd: bool | None = None
    prime: bool | None = None

    # Matrix assumptions
    symmetric: bool | None = None
    hermitian: bool | None = None
    orthogonal: bool | None = None
    unitary: bool | None = None
    spd: bool | None = None  # Symmetric positive definite
    invertible: bool | None = None

    # Analysis assumptions (framework-specific)
    bounded: bool | None = None
    continuous: bool | None = None
    differentiable: bool | None = None
    lipschitz: bool | None = None
    measurable: bool | None = None
    integrable: bool | None = None

    # Custom properties (for non-SymPy reasoning)
    custom: dict[str, Any] = Field(default_factory=dict)

    def to_sympy_assumptions(self) -> dict[str, bool]:
        """
        Convert to SymPy-compatible assumptions dictionary.

        Only includes assumptions SymPy understands.
        """
        sympy_compatible = [
            "real",
            "positive",
            "negative",
            "zero",
            "nonzero",
            "integer",
            "rational",
            "complex",
            "finite",
            "infinite",
            "even",
            "odd",
            "prime",
            "symmetric",
            "hermitian",
            "orthogonal",
        ]

        assumptions = {}
        for key in sympy_compatible:
            value = getattr(self, key)
            if value is not None:
                assumptions[key] = value

        return assumptions


# =============================================================================
# BASE EXPRESSION CLASS
# =============================================================================


class SymExpr(BaseModel):
    """
    Base class for all SymPy expressions.

    Uses discriminated union pattern with 'type' field for JSON serialization
    and type safety.

    All subclasses must:
    1. Set type: Literal['SpecificType'] as class attribute
    2. Implement to_sympy() -> sympy.Basic
    3. Be frozen (immutable)

    Maps to Lean:
        inductive SymExpr where
          | Symbol : String → AssumptionSet → SymExpr
          | Integer : Int → SymExpr
          | Add : List SymExpr → SymExpr
          ...
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str = Field(..., description="Discriminant for union type")

    def to_sympy(self) -> sympy.Basic:
        """
        Convert Pydantic model to live SymPy object.

        This is a pure function with no side effects.
        Subclasses must implement this.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_sympy()")

    def to_latex(self) -> str:
        """
        Generate LaTeX representation.

        Default implementation converts to SymPy then uses SymPy's latex().
        Subclasses can override for custom LaTeX.
        """
        try:
            return sympy.latex(self.to_sympy())
        except Exception:
            return f"\\text{{{self.__class__.__name__}}}"


# =============================================================================
# ATOMIC EXPRESSIONS (Leaves)
# =============================================================================


class SymbolExpr(SymExpr):
    """
    Symbolic variable.

    Maps to Lean:
        structure SymbolExpr where
          name : String
          assumptions : AssumptionSet
    """

    type: Literal["Symbol"] = "Symbol"
    name: str = Field(..., min_length=1, description="Symbol name (e.g., 'x', 'theta')")
    assumptions: AssumptionSet = Field(
        default_factory=AssumptionSet, description="Mathematical assumptions"
    )

    def to_sympy(self) -> sympy.Symbol:
        """Convert to SymPy Symbol with assumptions."""
        return sympy.Symbol(self.name, **self.assumptions.to_sympy_assumptions())


class IntegerExpr(SymExpr):
    """Integer constant."""

    type: Literal["Integer"] = "Integer"
    value: int = Field(..., description="Integer value")

    def to_sympy(self) -> sympy.Integer:
        return sympy.Integer(self.value)


class RationalExpr(SymExpr):
    """Rational number p/q."""

    type: Literal["Rational"] = "Rational"
    p: int = Field(..., description="Numerator")
    q: int = Field(..., ge=1, description="Denominator (positive)")

    def to_sympy(self) -> sympy.Rational:
        return sympy.Rational(self.p, self.q)


class FloatExpr(SymExpr):
    """Floating point number."""

    type: Literal["Float"] = "Float"
    value: float = Field(..., description="Float value")
    precision: int | None = Field(None, ge=1, description="Precision in bits")

    def to_sympy(self) -> sympy.Float:
        if self.precision is not None:
            return sympy.Float(self.value, self.precision)
        return sympy.Float(self.value)


class MatrixSymbolExpr(SymExpr):
    """Matrix symbol with shape."""

    type: Literal["MatrixSymbol"] = "MatrixSymbol"
    name: str = Field(..., min_length=1)
    rows: int = Field(..., ge=1)
    cols: int = Field(..., ge=1)
    assumptions: AssumptionSet = Field(default_factory=AssumptionSet)

    def to_sympy(self) -> sympy.MatrixSymbol:
        # SymPy MatrixSymbol doesn't support assumptions in constructor
        # We'll track them separately
        return sympy.MatrixSymbol(self.name, self.rows, self.cols)


# =============================================================================
# N-ARY OPERATIONS
# =============================================================================


class AddExpr(SymExpr):
    """Addition: a + b + c + ..."""

    type: Literal["Add"] = "Add"
    args: list[AnySymExpr] = Field(..., min_length=2, description="Summands")

    def to_sympy(self) -> sympy.Add:
        sympy_args = [arg.to_sympy() for arg in self.args]
        return sympy.Add(*sympy_args)


class MulExpr(SymExpr):
    """Multiplication: a * b * c * ..."""

    type: Literal["Mul"] = "Mul"
    args: list[AnySymExpr] = Field(..., min_length=2, description="Factors")

    def to_sympy(self) -> sympy.Mul:
        sympy_args = [arg.to_sympy() for arg in self.args]
        return sympy.Mul(*sympy_args)


class PowExpr(SymExpr):
    """Power: base^exp."""

    type: Literal["Pow"] = "Pow"
    base: AnySymExpr = Field(..., description="Base")
    exp: AnySymExpr = Field(..., description="Exponent")

    def to_sympy(self) -> sympy.Pow:
        return sympy.Pow(self.base.to_sympy(), self.exp.to_sympy())


# =============================================================================
# FUNCTIONS
# =============================================================================


class FuncExpr(SymExpr):
    """
    Function application: f(x, y, ...).

    Handles both standard functions (sin, exp) and custom functions.
    """

    type: Literal["Function"] = "Function"
    name: str = Field(..., min_length=1, description="Function name (e.g., 'sin', 'exp', 'f')")
    args: list[AnySymExpr] = Field(default_factory=list, description="Function arguments")

    def to_sympy(self) -> sympy.Function:
        # Get standard function if it exists, otherwise create generic Function
        func_class = getattr(sympy, self.name, None)

        if func_class and callable(func_class):
            # Standard function (sin, exp, etc.)
            sympy_args = [arg.to_sympy() for arg in self.args]
            return func_class(*sympy_args)
        # Custom/undefined function
        func = sympy.Function(self.name)
        sympy_args = [arg.to_sympy() for arg in self.args]
        return func(*sympy_args)


# =============================================================================
# CALCULUS
# =============================================================================


class DerivativeExpr(SymExpr):
    """
    Derivative: d^n expr / d var^n.

    Can handle partial derivatives with multiple variables.
    """

    type: Literal["Derivative"] = "Derivative"
    expr: AnySymExpr = Field(..., description="Expression to differentiate")
    # List of (variable, order) pairs
    vars: list[tuple[AnySymExpr, int]] = Field(
        ..., min_length=1, description="Variables and their orders"
    )

    def to_sympy(self) -> sympy.Derivative:
        sympy_expr = self.expr.to_sympy()
        # Convert vars to SymPy format
        sympy_vars = []
        for var, order in self.vars:
            sympy_var = var.to_sympy()
            if order == 1:
                sympy_vars.append(sympy_var)
            else:
                sympy_vars.append((sympy_var, order))
        return sympy.Derivative(sympy_expr, *sympy_vars)


class IntegralExpr(SymExpr):
    """
    Integral: ∫ integrand d vars with limits.

    Supports definite and indefinite integrals, multi-dimensional integrals.
    """

    type: Literal["Integral"] = "Integral"
    integrand: AnySymExpr = Field(..., description="Expression to integrate")
    # List of (var, lower_bound, upper_bound) tuples
    # If bounds are None, indefinite integral
    limits: list[tuple[AnySymExpr, AnySymExpr | None, AnySymExpr | None]] = Field(
        ..., min_length=1, description="Integration limits"
    )
    measure: str | None = Field(
        None, description="Measure (e.g., 'dx', 'dμ'). Informational only."
    )

    def to_sympy(self) -> sympy.Integral:
        sympy_integrand = self.integrand.to_sympy()
        sympy_limits = []

        for var, lower, upper in self.limits:
            sympy_var = var.to_sympy()
            if lower is None and upper is None:
                # Indefinite integral
                sympy_limits.append(sympy_var)
            else:
                # Definite integral
                sympy_lower = lower.to_sympy() if lower is not None else None
                sympy_upper = upper.to_sympy() if upper is not None else None
                sympy_limits.append((sympy_var, sympy_lower, sympy_upper))

        return sympy.Integral(sympy_integrand, *sympy_limits)


class LimitExpr(SymExpr):
    """
    Limit: lim_{var → point} expr.

    Supports one-sided and two-sided limits.
    """

    type: Literal["Limit"] = "Limit"
    expr: AnySymExpr = Field(..., description="Expression")
    var: AnySymExpr = Field(..., description="Variable approaching")
    point: AnySymExpr = Field(..., description="Limit point")
    dir: Literal["+", "-", "both"] = Field("both", description="Direction (+, -, or both)")

    def to_sympy(self) -> sympy.Limit:
        sympy_expr = self.expr.to_sympy()
        sympy_var = self.var.to_sympy()
        sympy_point = self.point.to_sympy()

        # Map direction to SymPy format
        sympy_dir = {"+": "+", "-": "-", "both": "+-"}[self.dir]

        return sympy.Limit(sympy_expr, sympy_var, sympy_point, sympy_dir)


class SumExpr(SymExpr):
    """
    Discrete sum: Σ_{index=lower}^{upper} expr.
    """

    type: Literal["Sum"] = "Sum"
    expr: AnySymExpr = Field(..., description="Summand")
    index: AnySymExpr = Field(..., description="Index variable")
    lower: AnySymExpr = Field(..., description="Lower bound")
    upper: AnySymExpr = Field(..., description="Upper bound")

    def to_sympy(self) -> sympy.Sum:
        sympy_expr = self.expr.to_sympy()
        sympy_index = self.index.to_sympy()
        sympy_lower = self.lower.to_sympy()
        sympy_upper = self.upper.to_sympy()
        return sympy.Sum(sympy_expr, (sympy_index, sympy_lower, sympy_upper))


class ProductExpr(SymExpr):
    """
    Discrete product: Π_{index=lower}^{upper} expr.
    """

    type: Literal["Product"] = "Product"
    expr: AnySymExpr = Field(..., description="Factor")
    index: AnySymExpr = Field(..., description="Index variable")
    lower: AnySymExpr = Field(..., description="Lower bound")
    upper: AnySymExpr = Field(..., description="Upper bound")

    def to_sympy(self) -> sympy.Product:
        sympy_expr = self.expr.to_sympy()
        sympy_index = self.index.to_sympy()
        sympy_lower = self.lower.to_sympy()
        sympy_upper = self.upper.to_sympy()
        return sympy.Product(sympy_expr, (sympy_index, sympy_lower, sympy_upper))


# =============================================================================
# RELATIONS & LOGIC
# =============================================================================


class EqExpr(SymExpr):
    """Equality: lhs == rhs."""

    type: Literal["Eq"] = "Eq"
    lhs: AnySymExpr = Field(..., description="Left-hand side")
    rhs: AnySymExpr = Field(..., description="Right-hand side")

    def to_sympy(self) -> sympy.Eq:
        return sympy.Eq(self.lhs.to_sympy(), self.rhs.to_sympy())


class IneqExpr(SymExpr):
    """Inequality: lhs <op> rhs where op ∈ {<, ≤, >, ≥, ≠}."""

    type: Literal["Ineq"] = "Ineq"
    lhs: AnySymExpr = Field(..., description="Left-hand side")
    rhs: AnySymExpr = Field(..., description="Right-hand side")
    relation: RelationOp = Field(..., description="Relational operator")

    def to_sympy(self) -> sympy.Rel:
        sympy_lhs = self.lhs.to_sympy()
        sympy_rhs = self.rhs.to_sympy()

        # Map to SymPy relation classes
        relation_map = {
            RelationOp.LT: sympy.Lt,
            RelationOp.LE: sympy.Le,
            RelationOp.GT: sympy.Gt,
            RelationOp.GE: sympy.Ge,
            RelationOp.NE: sympy.Ne,
        }

        rel_class = relation_map[self.relation]
        return rel_class(sympy_lhs, sympy_rhs)


class BooleanExpr(SymExpr):
    """Boolean logic: And, Or, Not, Implies, Equivalent."""

    type: Literal["Boolean"] = "Boolean"
    op: LogicOp = Field(..., description="Logical operator")
    args: list[AnySymExpr] = Field(..., min_length=1, description="Operands")

    def to_sympy(self) -> sympy.Boolean:
        sympy_args = [arg.to_sympy() for arg in self.args]

        # Map to SymPy logic classes
        logic_map = {
            LogicOp.AND: sympy.And,
            LogicOp.OR: sympy.Or,
            LogicOp.NOT: sympy.Not,
            LogicOp.IMPLIES: sympy.Implies,
            LogicOp.EQUIVALENT: sympy.Equivalent,
        }

        logic_class = logic_map[self.op]
        return logic_class(*sympy_args)


# =============================================================================
# DOMAIN-SPECIFIC OPERATORS (Opaque)
# =============================================================================


class OpExpr(SymExpr):
    """
    Domain-specific opaque operator.

    Used for operators that SymPy doesn't understand (Wasserstein, manifold ops, etc.).
    These are treated as symbolic functions by SymPy but have semantic meaning
    in our framework.

    Validation of these expressions uses plugin rules, not SymPy's built-in logic.
    """

    type: Literal["Op"] = "Op"
    name: DomainOp = Field(..., description="Operator name")
    args: list[AnySymExpr] = Field(default_factory=list, description="Arguments")
    attrs: dict[str, Any] = Field(
        default_factory=dict, description="Additional attributes (e.g., metric, measure)"
    )

    def to_sympy(self) -> sympy.Function:
        """
        Convert to SymPy generic function.

        SymPy treats this as an undefined function, which is correct since
        SymPy doesn't understand the semantics.
        """
        func = sympy.Function(self.name.value)
        sympy_args = [arg.to_sympy() for arg in self.args]
        return func(*sympy_args)

    def to_latex(self) -> str:
        """Custom LaTeX for domain operators."""
        # Map operators to LaTeX representations
        latex_map = {
            DomainOp.GRAD: "\\nabla",
            DomainOp.DIV: "\\nabla \\cdot",
            DomainOp.LAPLACE_BELTRAMI: "\\Delta_g",
            DomainOp.EXPECTATION: "\\mathbb{E}",
            DomainOp.PROBABILITY: "\\mathbb{P}",
            DomainOp.WASSERSTEIN_2: "W_2",
            DomainOp.KL_DIVERGENCE: "D_{\\text{KL}}",
            DomainOp.ENTROPY: "H",
        }

        op_latex = latex_map.get(self.name, f"\\text{{{self.name.value}}}")

        if not self.args:
            return op_latex

        args_latex = ", ".join(arg.to_latex() for arg in self.args)
        return f"{op_latex}({args_latex})"


# =============================================================================
# PIECEWISE & CONDITIONAL
# =============================================================================


class PiecewiseExpr(SymExpr):
    """
    Piecewise function:
    { expr1 if cond1
    { expr2 if cond2
    { ...
    { otherwise

    Maps to Lean:
        structure PiecewiseExpr where
          clauses : List (SymExpr × SymExpr)
          otherwise : Option SymExpr
    """

    type: Literal["Piecewise"] = "Piecewise"
    # List of (expression, condition) pairs
    clauses: list[tuple[AnySymExpr, AnySymExpr]] = Field(
        ..., min_length=1, description="(expr, condition) pairs"
    )
    otherwise: AnySymExpr | None = Field(None, description="Default case")

    def to_sympy(self) -> sympy.Piecewise:
        sympy_clauses = [(expr.to_sympy(), cond.to_sympy()) for expr, cond in self.clauses]

        if self.otherwise is not None:
            sympy_clauses.append((self.otherwise.to_sympy(), True))

        return sympy.Piecewise(*sympy_clauses)


# =============================================================================
# UNION TYPE (All Expression Types)
# =============================================================================

# This is the main type that external code should use
AnySymExpr = Union[
    SymbolExpr,
    IntegerExpr,
    RationalExpr,
    FloatExpr,
    MatrixSymbolExpr,
    AddExpr,
    MulExpr,
    PowExpr,
    FuncExpr,
    DerivativeExpr,
    IntegralExpr,
    LimitExpr,
    SumExpr,
    ProductExpr,
    EqExpr,
    IneqExpr,
    BooleanExpr,
    OpExpr,
    PiecewiseExpr,
]

# Update forward references for recursive types
AddExpr.model_rebuild()
MulExpr.model_rebuild()
PowExpr.model_rebuild()
FuncExpr.model_rebuild()
DerivativeExpr.model_rebuild()
IntegralExpr.model_rebuild()
LimitExpr.model_rebuild()
SumExpr.model_rebuild()
ProductExpr.model_rebuild()
EqExpr.model_rebuild()
IneqExpr.model_rebuild()
BooleanExpr.model_rebuild()
OpExpr.model_rebuild()
PiecewiseExpr.model_rebuild()


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================


def from_sympy(expr: sympy.Basic) -> AnySymExpr:
    """
    Convert SymPy expression to Pydantic model.

    This is a best-effort conversion. Some SymPy expressions may not have
    a direct Pydantic equivalent and will be represented as FuncExpr.

    Pure function with no side effects.

    Args:
        expr: SymPy expression

    Returns:
        Pydantic SymExpr model

    Raises:
        ValueError: If expression cannot be converted
    """
    # Symbol
    if isinstance(expr, sympy.Symbol):
        # Extract assumptions from SymPy symbol
        assumptions_dict = {
            key: value for key, value in expr.assumptions0.items() if value is True
        }
        return SymbolExpr(name=expr.name, assumptions=AssumptionSet(**assumptions_dict))

    # Numbers
    if isinstance(expr, sympy.Integer):
        return IntegerExpr(value=int(expr))
    if isinstance(expr, sympy.Rational):
        return RationalExpr(p=expr.p, q=expr.q)
    if isinstance(expr, sympy.Float):
        return FloatExpr(value=float(expr))

    # Matrix symbol
    if isinstance(expr, sympy.MatrixSymbol):
        return MatrixSymbolExpr(name=str(expr.name), rows=expr.rows, cols=expr.cols)

    # Operations
    if isinstance(expr, sympy.Add):
        return AddExpr(args=[from_sympy(arg) for arg in expr.args])
    if isinstance(expr, sympy.Mul):
        return MulExpr(args=[from_sympy(arg) for arg in expr.args])
    if isinstance(expr, sympy.Pow):
        return PowExpr(base=from_sympy(expr.base), exp=from_sympy(expr.exp))

    # Calculus
    if isinstance(expr, sympy.Derivative):
        # Extract variable info
        vars_list = []
        for var_info in expr.variable_count:
            if isinstance(var_info, tuple):
                var, count = var_info
                vars_list.append((from_sympy(var), count))
            else:
                vars_list.append((from_sympy(var_info), 1))
        return DerivativeExpr(expr=from_sympy(expr.expr), vars=vars_list)

    if isinstance(expr, sympy.Integral):
        # Extract limits
        limits_list = []
        for limit in expr.limits:
            if len(limit) == 1:
                # Indefinite
                limits_list.append((from_sympy(limit[0]), None, None))
            elif len(limit) == 3:
                # Definite
                limits_list.append((
                    from_sympy(limit[0]),
                    from_sympy(limit[1]),
                    from_sympy(limit[2]),
                ))
        return IntegralExpr(integrand=from_sympy(expr.function), limits=limits_list)

    # Relations
    if isinstance(expr, sympy.Eq):
        return EqExpr(lhs=from_sympy(expr.lhs), rhs=from_sympy(expr.rhs))
    if isinstance(expr, sympy.Lt | sympy.Le | sympy.Gt | sympy.Ge | sympy.Ne):
        relation_map = {
            sympy.Lt: RelationOp.LT,
            sympy.Le: RelationOp.LE,
            sympy.Gt: RelationOp.GT,
            sympy.Ge: RelationOp.GE,
            sympy.Ne: RelationOp.NE,
        }
        return IneqExpr(
            lhs=from_sympy(expr.lhs),
            rhs=from_sympy(expr.rhs),
            relation=relation_map[type(expr)],
        )

    # Functions
    if isinstance(expr, sympy.Function):
        func_name = expr.func.__name__
        args = [from_sympy(arg) for arg in expr.args]
        return FuncExpr(name=func_name, args=args)

    # Fallback: treat as generic function
    if hasattr(expr, "func") and hasattr(expr, "args"):
        func_name = str(expr.func)
        args = [from_sympy(arg) for arg in expr.args]
        return FuncExpr(name=func_name, args=args)

    # Unknown: raise error
    raise ValueError(f"Cannot convert SymPy expression to Pydantic: {type(expr).__name__}")
