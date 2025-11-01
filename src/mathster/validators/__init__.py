"""
Validators for Mathematical Objects.

This module provides automated validators for:
- Dataflow validation (ProofBox property flow)
- SymPy validation (mathematical expressions)
- Framework consistency checking (against docs/glossary.md)

Version: 1.0.0
"""

from mathster.validators.framework_checker import FrameworkChecker


__all__ = ["FrameworkChecker"]
