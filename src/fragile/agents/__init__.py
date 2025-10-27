"""
Autonomous agents for mathematical document processing.

This module contains agents that can autonomously process mathematical documents,
extract structured data, and interact with the fragile.proofs framework.

Agents:
- MathDocumentParser: Extracts mathematical content from MyST markdown documents
"""

from fragile.agents.math_document_parser import MathDocumentParser

__all__ = ["MathDocumentParser"]
