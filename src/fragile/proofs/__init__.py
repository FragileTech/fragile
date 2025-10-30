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

# Define __all__ by combining all submodule exports
from fragile.proofs.core import __all__ as core_all

# Import LLM pipeline modules
from fragile.proofs.llm import *  # noqa: F403
from fragile.proofs.llm import __all__ as llm_all
from fragile.proofs.prompts import *  # noqa: F403
from fragile.proofs.prompts import __all__ as prompts_all
from fragile.proofs.registry import *  # noqa: F403
from fragile.proofs.registry import __all__ as registry_all
from fragile.proofs.relationships import *  # noqa: F403
from fragile.proofs.relationships import __all__ as relationships_all

# Import schema generators
from fragile.proofs.schema_generator import (
    generate_complete_schema,
    generate_proof_schema,
    generate_sketch_schema,
)

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
from fragile.proofs.sympy_integration import *  # noqa: F403
from fragile.proofs.sympy_integration import __all__ as sympy_all
from fragile.proofs.tools import *  # noqa: F403
from fragile.proofs.tools import __all__ as tools_all
from fragile.proofs.utils import *  # noqa: F403
from fragile.proofs.utils import __all__ as utils_all


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

# =============================================================================
# MODEL REBUILD (Fix forward references after all imports complete)
# =============================================================================

# Rebuild models that reference SourceLocation and other forward-referenced types
# This must happen AFTER all imports are complete to avoid NameError

# 1. Rebuild staging types (use SourceLocation via TYPE_CHECKING)
RawDefinition.model_rebuild()
RawTheorem.model_rebuild()
RawProof.model_rebuild()
RawAxiom.model_rebuild()
RawCitation.model_rebuild()
RawEquation.model_rebuild()
RawParameter.model_rebuild()
RawRemark.model_rebuild()
StagingDocument.model_rebuild()

# 2. Rebuild LLM container models
from fragile.proofs.llm.document_container import EnrichedEntities, MathematicalDocument


EnrichedEntities.model_rebuild()
MathematicalDocument.model_rebuild()
