"""
Fragile Proof System - Comprehensive Mathematical Framework.

This module consolidates the entire mathematical proof framework including:
- Core types (theorems, objects, properties, mathster)
- Relationship system (relationships, graphs)
- Registry and storage
- SymPy integration
- Schema generation for LLM consumption

All organized into submodules for better maintainability.
"""

# Re-export everything from submodules
from mathster.core import *  # noqa: F403

# Define __all__ by combining all submodule exports
try:
    from mathster.core import __all__ as core_all
except ImportError:
    core_all = []  # Handle missing __all__

# Import LLM pipeline modules
from mathster.llm import *  # noqa: F403
try:
    from mathster.llm import __all__ as llm_all
except ImportError:
    llm_all = []

from mathster.prompts import *  # noqa: F403
try:
    from mathster.prompts import __all__ as prompts_all
except ImportError:
    prompts_all = []

from mathster.registry import *  # noqa: F403
try:
    from mathster.registry import __all__ as registry_all
except ImportError:
    registry_all = []

from mathster.relationships import *  # noqa: F403
try:
    from mathster.relationships import __all__ as relationships_all
except ImportError:
    relationships_all = []

# Import schema generators
from mathster.schema_generator import (
    generate_complete_schema,
    generate_proof_schema,
    generate_sketch_schema,
)

# Import staging types (now in core.raw_data)
from mathster.core.raw_data import (
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
from mathster.sympy_integration import *  # noqa: F403
try:
    from mathster.sympy_integration import __all__ as sympy_all
except ImportError:
    sympy_all = []

from mathster.tools import *  # noqa: F403
try:
    from mathster.tools import __all__ as tools_all
except ImportError:
    tools_all = []

try:
    from mathster.utils import *  # noqa: F403
    from mathster.utils import __all__ as utils_all
except ImportError:
    utils_all = []


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
from mathster.llm.document_container import EnrichedEntities, MathematicalDocument


EnrichedEntities.model_rebuild()
MathematicalDocument.model_rebuild()
