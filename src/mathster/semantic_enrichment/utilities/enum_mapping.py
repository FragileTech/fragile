"""
Automated enum mapping utilities.

Provides functions to convert string values to Pydantic enum types used in enriched entities.
"""

from mathster.core.enriched_data import ParameterScope, RemarkType


def map_parameter_scope(scope_str: str) -> ParameterScope:
    """
    Map scope string to ParameterScope enum.

    Args:
        scope_str: Scope as string ("global", "local", "universal")

    Returns:
        ParameterScope enum value

    Raises:
        ValueError: If scope_str is not recognized
    """
    mapping = {
        "global": ParameterScope.GLOBAL,
        "local": ParameterScope.LOCAL,
        "universal": ParameterScope.UNIVERSAL,
    }

    scope_lower = scope_str.lower().strip()

    if scope_lower not in mapping:
        raise ValueError(f"Unknown parameter scope: {scope_str}. Valid: global, local, universal")

    return mapping[scope_lower]


def map_remark_type(remark_type_str: str) -> RemarkType:
    """
    Map remark type string to RemarkType enum.

    Args:
        remark_type_str: Remark type as string

    Returns:
        RemarkType enum value

    Raises:
        ValueError: If type not recognized
    """
    mapping = {
        "note": RemarkType.NOTE,
        "remark": RemarkType.REMARK,
        "observation": RemarkType.OBSERVATION,
        "comment": RemarkType.COMMENT,
        "example": RemarkType.EXAMPLE,
        "intuition": RemarkType.INTUITION,
        "warning": RemarkType.WARNING,
    }

    type_lower = remark_type_str.lower().strip()

    if type_lower not in mapping:
        # Default to REMARK if not recognized
        return RemarkType.REMARK

    return mapping[type_lower]
