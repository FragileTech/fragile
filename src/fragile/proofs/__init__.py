"""
Fragile Proof System - Comprehensive Mathematical Framework.

This module consolidates the entire mathematical proof framework including:
- Core types (theorems, objects, properties, proofs)
- Relationship system (relationships, graphs)
- Registry and storage
- SymPy integration
- Schema generation for LLM consumption

All organized into submodules for better maintainability.
"""

# Re-export everything from submodules
from fragile.proofs.core import *  # noqa: F403
from fragile.proofs.relationships import *  # noqa: F403
from fragile.proofs.registry import *  # noqa: F403
from fragile.proofs.sympy import *  # noqa: F403
from fragile.proofs.utils import *  # noqa: F403

# Import LLM pipeline modules
from fragile.proofs.llm import *  # noqa: F403
from fragile.proofs.tools import *  # noqa: F403
from fragile.proofs.prompts import *  # noqa: F403

# Import staging types
from fragile.proofs.staging_types import (
    RawAxiom,
    RawCitation,
    RawDefinition,
    RawEquation,
    RawParameter,
    RawProof,
    RawRemark,
    RawTheorem,
    StagingDocument,
)

# Import schema generators
from fragile.proofs.schema_generator import (
    generate_complete_schema,
    generate_proof_schema,
    generate_sketch_schema,
)

# Define __all__ by combining all submodule exports
from fragile.proofs.core import __all__ as core_all
from fragile.proofs.relationships import __all__ as relationships_all
from fragile.proofs.registry import __all__ as registry_all
from fragile.proofs.sympy import __all__ as sympy_all
from fragile.proofs.utils import __all__ as utils_all
from fragile.proofs.llm import __all__ as llm_all
from fragile.proofs.tools import __all__ as tools_all
from fragile.proofs.prompts import __all__ as prompts_all

__all__ = [
    *core_all,
    *relationships_all,
    *registry_all,
    *sympy_all,
    *utils_all,
    *llm_all,
    *tools_all,
    *prompts_all,
    # Staging types (raw extraction models)
    "RawDefinition",
    "RawTheorem",
    "RawProof",
    "RawAxiom",
    "RawCitation",
    "RawEquation",
    "RawParameter",
    "RawRemark",
    "StagingDocument",
    # Schema generation
    "generate_complete_schema",
    "generate_proof_schema",
    "generate_sketch_schema",
]
