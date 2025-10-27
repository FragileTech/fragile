"""
SymPy Integration.

This module provides SymPy integration for the proof system:
- SymPy expression models (complete AST as Pydantic)
- Dual representation (LaTeX + SymPy)
- Validation engine
- Proof integration with SymPy validation
- Object extensions with symbol tables
"""

from fragile.proofs.sympy.expressions import (
    AddExpr,
    AnySymExpr,
    AssumptionSet,
    BooleanExpr,
    DerivativeExpr,
    DomainOp,
    EqExpr,
    FloatExpr,
    FuncExpr,
    IneqExpr,
    IntegralExpr,
    IntegerExpr,
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
    from_sympy,
)
from fragile.proofs.sympy.dual_representation import (
    DualExpr,
    DualStatement,
    SymbolDeclaration,
    SymPyContext,
    create_dual_expr_from_latex,
    create_dual_expr_from_sympy,
    create_symbol_context,
    parse_latex_to_sympy,
    sympy_to_latex,
    validate_dual_consistency,
)
from fragile.proofs.sympy.validation import (
    PluginRegistry,
    SymPyValidator,
    Transformation,
    TransformationType,
    ValidationIssue,
    ValidationPlugin,
    ValidationResult,
    ValidationStatus,
    differentiate_expr,
    expand_expr,
    simplify_expr,
)
from fragile.proofs.sympy.proof_integration import (
    DirectDerivationWithSymPy,
    PropertyReferenceWithSymPy,
    SymPyProofStep,
    add_sympy_validation_to_step,
    create_sympy_proof_step,
    validate_proof_box_with_sympy,
    validate_proof_chain_with_sympy,
    validate_proof_step_with_sympy,
)
from fragile.proofs.sympy.object_extensions import (
    MathematicalObjectWithSymPy,
    PropertyWithSymPy,
    create_combined_sympy_context,
    create_object_with_symbols,
    create_property_with_dual_expr,
    merge_symbol_tables,
)

__all__ = [
    # Expressions
    "SymExpr",
    "AnySymExpr",
    "SymbolExpr",
    "IntegerExpr",
    "RationalExpr",
    "FloatExpr",
    "MatrixSymbolExpr",
    "AddExpr",
    "MulExpr",
    "PowExpr",
    "FuncExpr",
    "DerivativeExpr",
    "IntegralExpr",
    "LimitExpr",
    "SumExpr",
    "ProductExpr",
    "EqExpr",
    "IneqExpr",
    "BooleanExpr",
    "OpExpr",
    "PiecewiseExpr",
    "AssumptionSet",
    "RelationOp",
    "LogicOp",
    "DomainOp",
    "from_sympy",
    # Dual representation
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
    # Validation
    "ValidationStatus",
    "ValidationIssue",
    "ValidationResult",
    "TransformationType",
    "Transformation",
    "ValidationPlugin",
    "PluginRegistry",
    "SymPyValidator",
    "simplify_expr",
    "expand_expr",
    "differentiate_expr",
    # Proof integration
    "PropertyReferenceWithSymPy",
    "SymPyProofStep",
    "DirectDerivationWithSymPy",
    "validate_proof_step_with_sympy",
    "validate_proof_chain_with_sympy",
    "validate_proof_box_with_sympy",
    "create_sympy_proof_step",
    "add_sympy_validation_to_step",
    # Object extensions
    "PropertyWithSymPy",
    "MathematicalObjectWithSymPy",
    "create_object_with_symbols",
    "create_property_with_dual_expr",
    "merge_symbol_tables",
    "create_combined_sympy_context",
]

# Note: model_rebuild() is handled in fragile.proofs.core.__init__ to avoid circular imports
# PropertyWithSymPy.model_rebuild()
# MathematicalObjectWithSymPy.model_rebuild()
