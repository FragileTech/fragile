"""
Backward compatibility module for review_system imports.

This module re-exports all types from mathster.core.reviews
to maintain backward compatibility with code that imports from mathster.core.review_system.

New code should import directly from mathster.core.reviews.
"""

# Re-export everything from reviews
from mathster.core.reviews import *  # noqa: F403
