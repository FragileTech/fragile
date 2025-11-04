"""
Change tracking models for improvement workflow.

Defines models for tracking changes made during the improvement process,
including the type of change, affected entity, and metadata.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ChangeOperation(str, Enum):
    """Type of change operation."""

    ADD = "ADD"
    MODIFY = "MODIFY"
    DELETE = "DELETE"
    NO_CHANGE = "NO_CHANGE"


class EntityChange(BaseModel):
    """Record of a change to a single entity."""

    entity_type: Literal[
        "definition", "theorem", "proof", "axiom", "parameter", "remark", "citation"
    ]
    label: str
    operation: ChangeOperation
    reason: str = Field(..., description="Why this change was made")
    old_data: dict | None = Field(None, description="Original entity data (for MODIFY/DELETE)")
    new_data: dict | None = Field(None, description="New entity data (for ADD/MODIFY)")
