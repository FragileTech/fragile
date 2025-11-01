"""
Backward compatibility module for pipeline_types imports.

This module re-exports all types from mathster.proof_pipeline.pipeline_types
to maintain backward compatibility with code that imports from mathster.core.pipeline_types.

New code should import directly from mathster.proof_pipeline.pipeline_types.
"""

# Re-export everything from proof_pipeline.pipeline_types
from mathster.proof_pipeline.pipeline_types import *  # noqa: F403, F401
