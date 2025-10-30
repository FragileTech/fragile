"""
Document Container for Mathematical Entities.

This module provides the MathematicalDocument class, which serves as the main
container for all mathematical entities extracted and enriched from a markdown
document.

Design Principles:
- Single source of truth for document-level entities
- Combines raw staging data with enriched final models
- Provides statistics and validation methods
- Supports incremental construction during pipeline execution

Maps to Lean:
    namespace MathematicalDocument
      structure MathematicalDocument where
        document_id : String
        chapter : Option String
        staging : StagingDocument
        enriched : EnrichedEntities
    end MathematicalDocument
"""

from pydantic import BaseModel, ConfigDict, Field

from fragile.proofs.core import (
    Axiom,
    DefinitionBox,
    MathematicalObject,
    Parameter,
    ProofBox,
    TheoremBox,
)
from fragile.proofs.core.enriched_types import (
    EquationBox,
    ParameterBox,
    RemarkBox,
)
from fragile.proofs.staging_types import (
    StagingDocument,
)


# =============================================================================
# ENRICHED ENTITIES CONTAINER
# =============================================================================


class EnrichedEntities(BaseModel):
    """
    Container for all enriched entities from Stage 2 processing.

    These are the final validated models after semantic enrichment,
    resolution, and validation.
    """

    model_config = ConfigDict(frozen=True)

    definitions: dict[str, DefinitionBox] = Field(
        default_factory=dict, description="Enriched definitions keyed by label"
    )

    theorems: dict[str, TheoremBox] = Field(
        default_factory=dict, description="Enriched theorems keyed by label"
    )

    axioms: dict[str, Axiom] = Field(
        default_factory=dict, description="Enriched axioms keyed by label"
    )

    proofs: dict[str, ProofBox] = Field(
        default_factory=dict, description="Enriched proofs keyed by proof_id"
    )

    objects: dict[str, MathematicalObject] = Field(
        default_factory=dict, description="Mathematical objects keyed by label"
    )

    parameters: dict[str, Parameter] = Field(
        default_factory=dict, description="Global parameters keyed by label"
    )

    equations: dict[str, EquationBox] = Field(
        default_factory=dict, description="Equations keyed by equation_id"
    )

    parameter_boxes: dict[str, ParameterBox] = Field(
        default_factory=dict, description="Parameter boxes keyed by parameter_id"
    )

    remarks: dict[str, RemarkBox] = Field(
        default_factory=dict, description="Remarks keyed by remark_id"
    )

    @property
    def total_entities(self) -> int:
        """Total number of enriched entities."""
        return (
            len(self.definitions)
            + len(self.theorems)
            + len(self.axioms)
            + len(self.proofs)
            + len(self.objects)
            + len(self.parameters)
            + len(self.equations)
            + len(self.parameter_boxes)
            + len(self.remarks)
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of enriched entities."""
        return (
            f"Enriched Entities: {self.total_entities} total\n"
            f"  - Definitions: {len(self.definitions)}\n"
            f"  - Theorems: {len(self.theorems)}\n"
            f"  - Axioms: {len(self.axioms)}\n"
            f"  - Proofs: {len(self.proofs)}\n"
            f"  - Objects: {len(self.objects)}\n"
            f"  - Parameters: {len(self.parameters)}\n"
            f"  - Equations: {len(self.equations)}\n"
            f"  - Parameter Boxes: {len(self.parameter_boxes)}\n"
            f"  - Remarks: {len(self.remarks)}"
        )


# =============================================================================
# MATHEMATICAL DOCUMENT CONTAINER
# =============================================================================


class MathematicalDocument(BaseModel):
    """
    Complete container for all mathematical entities from a document.

    This is the main data structure produced by the Extract-then-Enrich pipeline.
    It contains:
    1. Raw staging data from Stage 1 extraction
    2. Enriched final models from Stage 2 semantic processing
    3. Document-level metadata
    4. Statistics and validation information

    Usage:
        # Create empty document
        doc = MathematicalDocument(
            document_id="01_fragile_gas_framework",
            chapter="1_euclidean_gas"
        )

        # Add staging data from extraction
        doc = doc.add_staging_document(staging_doc)

        # Add enriched entities after processing
        doc = doc.add_enriched_definition(definition)
        doc = doc.add_enriched_theorem(theorem)

        # Get statistics
        print(doc.get_summary())

    Maps to Lean:
        structure MathematicalDocument where
          document_id : String
          chapter : Option String
          file_path : Option String
          staging_documents : List StagingDocument
          enriched : EnrichedEntities
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(
        ..., description="Unique identifier for the document (e.g., '01_fragile_gas_framework')"
    )

    chapter: str | None = Field(
        None,
        description="High-level chapter/folder name (e.g., '1_euclidean_gas', '2_geometric_gas')",
    )

    file_path: str | None = Field(None, description="Path to the source markdown file")

    staging_documents: list[StagingDocument] = Field(
        default_factory=list,
        description="Raw staging documents from Stage 1 extraction (one per section)",
    )

    enriched: EnrichedEntities = Field(
        default_factory=EnrichedEntities,
        description="All enriched entities from Stage 2 processing",
    )

    @property
    def total_raw_entities(self) -> int:
        """Total number of raw entities across all staging documents."""
        return sum(doc.total_entities for doc in self.staging_documents)

    @property
    def total_enriched_entities(self) -> int:
        """Total number of enriched entities."""
        return self.enriched.total_entities

    @property
    def enrichment_rate(self) -> float:
        """Percentage of raw entities successfully enriched."""
        if self.total_raw_entities == 0:
            return 0.0
        return (self.total_enriched_entities / self.total_raw_entities) * 100

    def get_summary(self) -> str:
        """Get a comprehensive summary of the document."""
        return (
            f"Mathematical Document: {self.document_id}\n"
            f"Chapter: {self.chapter or 'N/A'}\n"
            f"File: {self.file_path or 'N/A'}\n"
            f"\n"
            f"Raw Entities (Stage 1): {self.total_raw_entities}\n"
            f"  - Sections processed: {len(self.staging_documents)}\n"
            f"\n"
            f"Enriched Entities (Stage 2): {self.total_enriched_entities}\n"
            f"  - Enrichment rate: {self.enrichment_rate:.1f}%\n"
            f"\n"
            f"{self.enriched.get_summary()}"
        )

    # =============================================================================
    # INCREMENTAL CONSTRUCTION METHODS
    # =============================================================================

    def add_staging_document(self, staging_doc: StagingDocument) -> "MathematicalDocument":
        """
        Add a staging document from section processing.

        Since models are immutable, this returns a new MathematicalDocument instance.
        """
        return self.model_copy(
            update={"staging_documents": [*self.staging_documents, staging_doc]}
        )

    def add_enriched_definition(self, definition: DefinitionBox) -> "MathematicalDocument":
        """Add an enriched definition."""
        new_definitions = {**self.enriched.definitions, definition.label: definition}
        new_enriched = self.enriched.model_copy(update={"definitions": new_definitions})
        return self.model_copy(update={"enriched": new_enriched})

    def add_enriched_theorem(self, theorem: TheoremBox) -> "MathematicalDocument":
        """Add an enriched theorem."""
        new_theorems = {**self.enriched.theorems, theorem.label: theorem}
        new_enriched = self.enriched.model_copy(update={"theorems": new_theorems})
        return self.model_copy(update={"enriched": new_enriched})

    def add_enriched_axiom(self, axiom: Axiom) -> "MathematicalDocument":
        """Add an enriched axiom."""
        new_axioms = {**self.enriched.axioms, axiom.label: axiom}
        new_enriched = self.enriched.model_copy(update={"axioms": new_axioms})
        return self.model_copy(update={"enriched": new_enriched})

    def add_enriched_proof(self, proof: ProofBox) -> "MathematicalDocument":
        """Add an enriched proof."""
        new_proofs = {**self.enriched.proofs, proof.proof_id: proof}
        new_enriched = self.enriched.model_copy(update={"proofs": new_proofs})
        return self.model_copy(update={"enriched": new_enriched})

    def add_enriched_object(self, obj: MathematicalObject) -> "MathematicalDocument":
        """Add an enriched mathematical object."""
        new_objects = {**self.enriched.objects, obj.label: obj}
        new_enriched = self.enriched.model_copy(update={"objects": new_objects})
        return self.model_copy(update={"enriched": new_enriched})

    def add_enriched_parameter(self, param: Parameter) -> "MathematicalDocument":
        """Add an enriched parameter."""
        new_parameters = {**self.enriched.parameters, param.label: param}
        new_enriched = self.enriched.model_copy(update={"parameters": new_parameters})
        return self.model_copy(update={"enriched": new_enriched})

    # =============================================================================
    # LOOKUP METHODS
    # =============================================================================

    def get_definition(self, label: str) -> DefinitionBox | None:
        """Get a definition by label."""
        return self.enriched.definitions.get(label)

    def get_theorem(self, label: str) -> TheoremBox | None:
        """Get a theorem by label."""
        return self.enriched.theorems.get(label)

    def get_axiom(self, label: str) -> Axiom | None:
        """Get an axiom by label."""
        return self.enriched.axioms.get(label)

    def get_object(self, label: str) -> MathematicalObject | None:
        """Get a mathematical object by label."""
        return self.enriched.objects.get(label)

    def get_parameter(self, label: str) -> Parameter | None:
        """Get a parameter by label."""
        return self.enriched.parameters.get(label)
