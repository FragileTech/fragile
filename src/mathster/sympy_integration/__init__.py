"""
SymPy Integration.

This module provides SymPy integration for the proof system:
- SymPy expression models (complete AST as Pydantic)
- Dual representation (LaTeX + SymPy)
- Validation engine
- Proof integration with SymPy validation
- Object extensions with symbol tables
"""

from mathster.sympy_integration.dual_representation import (
    create_dual_expr_from_latex,
    create_dual_expr_from_sympy,
    create_symbol_context,
    DualExpr,
    DualStatement,
    parse_latex_to_sympy,
    SymbolDeclaration,
    sympy_to_latex,
    SymPyContext,
    validate_dual_consistency,
)
from mathster.sympy_integration.expressions import (
    AddExpr,
    AnySymExpr,
    AssumptionSet,
    BooleanExpr,
    DerivativeExpr,
    DomainOp,
    EqExpr,
    FloatExpr,
    from_sympy,
    FuncExpr,
    IneqExpr,
    IntegerExpr,
    IntegralExpr,
    LimitExpr,
    LogicOp,
    MatrixSymbolExpr,
    MulExpr,
    OpExpr,
    PiecewiseExpr,
    PowExpr,
    ProductExpr,
    RationalExpr,
    RelationOp,
    SumExpr,
    SymbolExpr,
    SymExpr,
)
from mathster.sympy_integration.object_extensions import (
    create_combined_sympy_context,
    create_object_with_symbols,
    create_property_with_dual_expr,
    MathematicalObjectWithSymPy,
    merge_symbol_tables,
    PropertyWithSymPy,
)
from mathster.sympy_integration.proof_integration import (
    add_sympy_validation_to_step,
    create_sympy_proof_step,
    DirectDerivationWithSymPy,
    PropertyReferenceWithSymPy,
    SymPyProofStep,
    validate_proof_box_with_sympy,
    validate_proof_chain_with_sympy,
    validate_proof_step_with_sympy,
)
from mathster.sympy_integration.validation import (
    differentiate_expr,
    expand_expr,
    PluginRegistry,
    simplify_expr,
    SymPyValidator,
    Transformation,
    TransformationType,
    ValidationIssue,
    ValidationPlugin,
    ValidationResult,
    ValidationStatus,
)


__all__ = [
    "AddExpr",
    "AnySymExpr",
    "AssumptionSet",
    "BooleanExpr",
    "DerivativeExpr",
    "DirectDerivationWithSymPy",
    "DomainOp",
    # Dual representation
    "DualExpr",
    "DualStatement",
    "EqExpr",
    "FloatExpr",
    "FuncExpr",
    "IneqExpr",
    "IntegerExpr",
    "IntegralExpr",
    "LimitExpr",
    "LogicOp",
    "MathematicalObjectWithSymPy",
    "MatrixSymbolExpr",
    "MulExpr",
    "OpExpr",
    "PiecewiseExpr",
    "PluginRegistry",
    "PowExpr",
    "ProductExpr",
    # Proof integration
    "PropertyReferenceWithSymPy",
    # Object extensions
    "PropertyWithSymPy",
    "RationalExpr",
    "RelationOp",
    "SumExpr",
    # Expressions
    "SymExpr",
    "SymPyContext",
    "SymPyProofStep",
    "SymPyValidator",
    "SymbolDeclaration",
    "SymbolExpr",
    "Transformation",
    "TransformationType",
    "ValidationIssue",
    "ValidationPlugin",
    "ValidationResult",
    # Validation
    "ValidationStatus",
    "add_sympy_validation_to_step",
    "create_combined_sympy_context",
    "create_dual_expr_from_latex",
    "create_dual_expr_from_sympy",
    "create_object_with_symbols",
    "create_property_with_dual_expr",
    "create_symbol_context",
    "create_sympy_proof_step",
    "differentiate_expr",
    "expand_expr",
    "from_sympy",
    "merge_symbol_tables",
    "parse_latex_to_sympy",
    "simplify_expr",
    "sympy_to_latex",
    "validate_dual_consistency",
    "validate_proof_box_with_sympy",
    "validate_proof_chain_with_sympy",
    "validate_proof_step_with_sympy",
]

# Note: model_rebuild() is handled in fragile.mathster.core.__init__ to avoid circular imports
# PropertyWithSymPy.model_rebuild()
# MathematicalObjectWithSymPy.model_rebuild()
