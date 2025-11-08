#!/usr/bin/env python3
"""
Proof Pipeline Interactive Dashboard.

Visualizes the mathematical proof pipeline using fragile.mathster data structures:
- Mathematical objects and their property evolution
- Relationships between objects (equivalence, embedding, approximation, etc.)
- Theorem dependency structures (internal lemma DAGs)
- Proof dataflow (property flow through proof steps)
- Relationship-type-specific networks

Features:
- Multi-source data loading (saved registry, runtime examples, user path)
- 5 visualization modes (property evolution, relationships, theorem DAG, proof dataflow, type-specific)
- Type-specific filtering (object type, relationship type, theorem output type, proof status)
- Manual reload capability
- Interactive node selection with detailed views

Usage:
    # Run with panel serve
    panel serve src/fragile/mathster/proof_pipeline_dashboard.py --show

    # Or run directly
    python src/fragile/mathster/proof_pipeline_dashboard.py
"""

import logging
from pathlib import Path

import holoviews as hv
import networkx as nx
import panel as pn

from fragile.shaolin.graph import (
    create_graphviz_layout,
    edges_as_df,
    find_closest_point,
    InteractiveGraph,
    nodes_as_df,
)

# Import directly from submodules to avoid circular import with sympy
# DO NOT import from fragile.mathster (top-level __init__.py triggers sympy import)
from mathster.core.math_types import (
    Axiom,
    create_simple_object,
    MathematicalObject,
    ObjectType,
    Relationship,
    RelationType,
    TheoremBox,
    TheoremOutputType,
)
from mathster.core.proof_system import (
    ProofStepStatus,
)
from mathster.registry.registry import MathematicalRegistry
from mathster.registry.storage import load_registry_from_directory
from mathster.reports import render_enriched_to_markdown


hv.extension("bokeh")
# Enable MathJax so $...$ and $$...$$ in Markdown render as math
pn.extension("mathjax")

# Local logger for console diagnostics
logger = logging.getLogger("fragile.mathster.pipeline_dashboard")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.INFO)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# =============================================================================
# COLOR SCHEMES
# =============================================================================

# =============================================================================
# COLOR PALETTES - Semantically meaningful, perceptually distinct
# All colors are unique across palettes (100% efficiency)
# =============================================================================

# Object types (8 types from ObjectType enum)
# Semantic mapping by mathematical nature
OBJECT_TYPE_COLORS = {
    "SET": "#0066cc",  # Deep blue - foundational structures
    "SPACE": "#009999",  # Teal - geometric spaces
    "FUNCTION": "#228833",  # Forest green - transformations
    "OPERATOR": "#dd4488",  # Hot pink - dynamic operators
    "MEASURE": "#ff8800",  # Orange - quantitative measures
    "DISTRIBUTION": "#8844cc",  # Purple - probabilistic distributions
    "PROCESS": "#996633",  # Brown - temporal processes
    "OTHER": "#888888",  # Gray - uncategorized
}

# Relationship types (8 types from RelationType enum)
# Directional and structural semantics
RELATIONSHIP_TYPE_COLORS = {
    "EQUIVALENCE": "#77cc00",  # Lime - symmetric equality
    "EMBEDDING": "#0055aa",  # Navy - directed inclusion
    "APPROXIMATION": "#ff9944",  # Warm orange - convergence
    "REDUCTION": "#cc2244",  # Crimson - simplification
    "EXTENSION": "#7744bb",  # Violet - expansion
    "GENERALIZATION": "#00bbdd",  # Bright cyan - abstraction
    "SPECIALIZATION": "#445566",  # Slate - concretization
    "OTHER": "#aaaaaa",  # Light gray - uncategorized
}

# Theorem output types (13 types from TheoremOutputType enum)
# Grouped by theorem nature and result type
THEOREM_OUTPUT_TYPE_COLORS = {
    "Property": "#4466ee",  # Royal blue - descriptive properties
    "Relation": "#22aa77",  # Sea green - relational results
    "Existence": "#44cc44",  # Kelly green - constructive existence
    "Construction": "#00bb77",  # Emerald - explicit construction
    "Uniqueness": "#aa66dd",  # Orchid - characterization
    "Classification": "#8855cc",  # Amethist - categorization
    "Impossibility": "#ee3333",  # Red - negative results
    "Embedding": "#007788",  # Deep teal - structural embedding
    "Extension": "#6633bb",  # Indigo - structural expansion
    "Approximation": "#ff7722",  # Tangerine - convergence/approximation
    "Equivalence": "#99dd00",  # Chartreuse - symmetric equivalence
    "Decomposition": "#cc4499",  # Magenta - structural breakdown
    "Reduction": "#994433",  # Maroon - simplification
}

# Proof step status (3 status levels)
# Traffic light pattern - excellent semantic mapping
PROOF_STEP_STATUS_COLORS = {
    "SKETCHED": "#f1c40f",  # Gold - in progress
    "EXPANDED": "#3498db",  # Dodger blue - complete but unverified
    "VERIFIED": "#2ecc71",  # Success green - reviewed and verified
}

# Node type colors (for mixed graphs)
# Distinct primary colors for different entity types
NODE_TYPE_COLORS = {
    "object": "#0088ee",  # Sky blue - mathematical objects
    "theorem": "#ee4444",  # Fire red - theorems (standout)
    "axiom": "#7733cc",  # Royal purple - foundational axioms
    "relationship": "#33bb55",  # Fresh green - relationships/edges
    "proof_step": "#ffaa00",  # Amber - proof workflow steps
}

# Chapter colors (high-level organizational structure)
# Primary spectrum for major divisions
CHAPTER_COLORS = {
    "1_euclidean_gas": "#0077dd",  # Azure - Euclidean Gas chapter
    "2_geometric_gas": "#00aa66",  # Jade - Geometric Gas chapter
    "3_adaptive_gas": "#ff7744",  # Coral - Adaptive Gas chapter (future)
    "root": "#bbbbbb",  # Silver - root level
    "unknown": "#666666",  # Dim gray - unknown
}

# Document colors (fine-grained organizational structure)
# Full spectrum with perceptual spacing for 16 documents
DOCUMENT_COLORS = {
    "01_fragile_gas_framework": "#1166cc",  # Blue
    "02_euclidean_gas": "#ff6600",  # Orange
    "03_cloning": "#00aa44",  # Green
    "04_wasserstein_contraction": "#dd2255",  # Red
    "05_kinetic_contraction": "#9944dd",  # Purple
    "06_convergence": "#aa5522",  # Brown
    "07_mean_field": "#ee55aa",  # Pink
    "08_propagation_chaos": "#777777",  # Gray
    "09_kl_convergence": "#ccaa00",  # Gold
    "10_qsd_exchangeability_theory": "#00cccc",  # Cyan
    "11_hk_convergence": "#5588ff",  # Light blue
    "11_hk_convergence_bounded_density_rigorous_proof": "#88bbff",  # Lighter blue
    "12_quantitative_error_bounds": "#ffaa66",  # Peach
    "13_geometric_gas_c3_regularity": "#66dd88",  # Light green
    "root": "#cccccc",  # Light silver
    "unknown": "#555555",  # Darker gray
}


# =============================================================================
# DASHBOARD CLASS
# =============================================================================


class ProofPipelineDashboard:
    """Interactive dashboard for proof pipeline visualization."""

    def __init__(self, default_registry_path: str | None = None):
        """Initialize dashboard.

        Args:
            default_registry_path: Optional path to registry directory
        """
        self.current_registry: MathematicalRegistry | None = None
        self.current_graph_mode = "property_evolution"

        # Graph cache (keyed by mode)
        self.graph_cache: dict[str, nx.DiGraph] = {}

        # Layout cache (keyed by mode + layout_name)
        self.layout_cache: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}

        # Default data source - use per-document pipeline registry
        if default_registry_path:
            self.default_registry_path = default_registry_path
        else:
            # Dashboard is at src/fragile/mathster/, project root is 3 levels up
            project_root = Path(__file__).parent.parent.parent.parent
            # Default to pipeline registry for 01_fragile_gas_framework
            # NOTE: This registry is synced from docs/source to registries/per_document
            # using scripts/sync_registries.py
            self.default_registry_path = str(
                project_root
                / "docs"
                / "source"
                / "1_euclidean_gas"
                / "01_fragile_gas_framework"
                / "pipeline_registry"
            )

        # Create UI components
        self._create_data_source_widgets()
        self._create_graph_mode_widgets()
        self._create_filter_widgets()
        self._create_reactive_components()

        # In-app console initialization
        self.console_lines: list[str] = []
        self.console_pane = pn.pane.Markdown("```\nConsole initialized.\n```", width=480)

        # Load initial data
        self._load_initial_data()

    def _discover_available_registries(self) -> dict[str, str]:
        """Discover available registries dynamically.

        Returns:
            Dict mapping display name to registry identifier
        """
        project_root = Path(__file__).parent.parent.parent.parent
        per_document_root = project_root / "registries" / "per_document"
        combined_root = project_root / "registries" / "combined"
        docs_source_root = project_root / "docs" / "source"

        options = {}

        # Discover per-document registries
        if per_document_root.exists():
            for doc_dir in sorted(per_document_root.iterdir()):
                if not doc_dir.is_dir():
                    continue

                doc_name = doc_dir.name

                # Check for pipeline registry
                if (doc_dir / "pipeline").exists() and (
                    doc_dir / "pipeline" / "index.json"
                ).exists():
                    display_name = f"{doc_name} (Pipeline)"
                    identifier = f"per_doc_{doc_name}_pipeline"
                    options[display_name] = identifier

                # Check for refined registry
                if (doc_dir / "refined").exists() and (
                    doc_dir / "refined" / "index.json"
                ).exists():
                    display_name = f"{doc_name} (Refined)"
                    identifier = f"per_doc_{doc_name}_refined"
                    options[display_name] = identifier

        # Discover combined registries
        if combined_root.exists():
            if (combined_root / "pipeline").exists() and (
                combined_root / "pipeline" / "index.json"
            ).exists():
                options["Combined Pipeline Registry"] = "combined_pipeline"

            if (combined_root / "refined").exists() and (
                combined_root / "refined" / "index.json"
            ).exists():
                options["Combined Refined Registry"] = "combined_refined"

        # Always add custom path option
        options["Custom Path"] = "saved_custom"

        # Discover refined docs directly from docs/source (Refined Files)
        # e.g., docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data
        if docs_source_root.exists():
            for chapter_dir in sorted(docs_source_root.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                for doc_dir in sorted(chapter_dir.iterdir()):
                    if not doc_dir.is_dir():
                        continue
                    if (doc_dir / "refined_data").exists():
                        display_name = f"Docs: {chapter_dir.name}/{doc_dir.name} (Refined Files)"
                        # Use '|' as a safe separator to avoid ambiguity with underscores
                        identifier = f"docs_refined|{chapter_dir.name}|{doc_dir.name}"
                        options[display_name] = identifier

        return options

    def _create_data_source_widgets(self):
        """Create data source selection widgets."""
        # Discover available registries dynamically
        available_registries = self._discover_available_registries()

        # Determine default value (prefer Docs Refined Files if available)
        default_value = None
        for identifier in available_registries.values():
            if identifier.startswith("docs_refined|"):
                default_value = identifier
                break
        if default_value is None:
            for identifier in available_registries.values():
                if "pipeline" in identifier.lower() and "per_doc" in identifier:
                    default_value = identifier
                    break
        if default_value is None and available_registries:
            default_value = next(iter(available_registries.values()))

        self.data_source_selector = pn.widgets.Select(
            name="Data Source",
            options=available_registries,
            value=default_value or "saved_custom",
            width=280,
            description="Select registry to visualize",
        )

        self.custom_path_input = pn.widgets.TextInput(
            name="Registry Path",
            value=self.default_registry_path,
            width=280,
            visible=False,  # Only show when custom path selected
        )

        self.reload_button = pn.widgets.Button(
            name="Reload Data",
            button_type="primary",
            width=280,
        )
        self.reload_button.on_click(self._on_reload_data)

        # Show/hide custom path based on selection
        def update_path_visibility(event):
            self.custom_path_input.visible = event.new == "saved_custom"

        self.data_source_selector.param.watch(update_path_visibility, "value")

    def _create_graph_mode_widgets(self):
        """Create graph mode selection widgets."""
        self.graph_mode_selector = pn.widgets.RadioButtonGroup(
            name="Graph Visualization Mode",
            options={
                "Property Evolution": "property_evolution",
                "Relationship Network": "relationship_network",
                "Theorem DAG": "theorem_dag",
                "Proof Dataflow": "proof_dataflow",
                "By Relationship Type": "relationship_type_specific",
            },
            value="property_evolution",
            button_type="primary",
        )

        # Relationship type selector (only shown in relationship_type_specific mode)
        self.relationship_type_selector = pn.widgets.Select(
            name="Relationship Type",
            options=[rt.value for rt in RelationType],
            value=RelationType.EQUIVALENCE.value,
            width=280,
            visible=False,
        )

        # Show/hide relationship type selector based on graph mode
        def update_reltype_visibility(event):
            self.relationship_type_selector.visible = event.new == "relationship_type_specific"

        self.graph_mode_selector.param.watch(update_reltype_visibility, "value")

    def _create_filter_widgets(self):
        """Create filter control widgets."""
        # Object type filter
        self.object_type_filter = pn.widgets.MultiChoice(
            name="Object Type",
            options=[ot.value for ot in ObjectType],
            value=[ot.value for ot in ObjectType],
            width=280,
            description="Filter objects by type",
        )

        # Relationship type filter
        self.relationship_type_filter = pn.widgets.MultiChoice(
            name="Relationship Type",
            options=[rt.value for rt in RelationType],
            value=[rt.value for rt in RelationType],
            width=280,
            description="Filter relationships by type",
        )

        # Theorem output type filter
        self.theorem_output_type_filter = pn.widgets.MultiChoice(
            name="Theorem Output Type",
            options=[tot.value for tot in TheoremOutputType],
            value=[tot.value for tot in TheoremOutputType],
            width=280,
            description="Filter theorems by output type",
        )

        # Proof status filter
        self.proof_status_filter = pn.widgets.MultiChoice(
            name="Proof Status",
            options=[ps.value for ps in ProofStepStatus],
            value=[ps.value for ps in ProofStepStatus],
            width=280,
            description="Filter mathster by status",
        )

        # Axiom framework filter (dynamically populated)
        self.axiom_framework_filter = pn.widgets.MultiChoice(
            name="Axiom Framework",
            options=[],
            value=[],
            width=280,
            description="Filter axioms by foundational framework",
        )

        # Document filter (dynamically populated)
        self.document_filter = pn.widgets.MultiChoice(
            name="Document",
            options=[],
            value=[],
            width=280,
            description="Filter by source document",
        )

        # Layout selector
        self.layout_selector = pn.widgets.Select(
            name="Graph Layout Algorithm",
            options={
                "Spring Model (neato)": "neato",
                "Hierarchical (dot)": "dot",
                "Force-Directed (sfdp)": "sfdp",
                "Spectral (NetworkX)": "spectral",
                "Circular (NetworkX)": "circular",
            },
            value="neato",
            width=280,
            description="Algorithm for positioning nodes",
        )

        # Hover tooltip fields selector
        self.hover_columns_selector = pn.widgets.MultiChoice(
            name="Hover Tooltip Fields",
            options=[
                "label",
                "node_type",
                "object_type",
                "theorem_output_type",
                "document",
                "chapter",
            ],
            value=["label", "node_type", "object_type"],
            width=280,
            description="Choose which fields show in hover tooltip",
        )

        # Reset button
        self.reset_filter_button = pn.widgets.Button(
            name="Reset All Filters",
            button_type="warning",
            width=280,
        )
        self.reset_filter_button.on_click(self._on_filter_reset)

    def _create_reactive_components(self):
        """Create reactive graph, controls, and statistics views."""
        # Reactive graph view (plot only)
        self.graph_view = pn.bind(
            self._create_graph_plot,
            graph_mode=self.graph_mode_selector,
            relationship_type=self.relationship_type_selector,
            object_types=self.object_type_filter,
            relationship_types=self.relationship_type_filter,
            theorem_output_types=self.theorem_output_type_filter,
            proof_statuses=self.proof_status_filter,
            axiom_frameworks=self.axiom_framework_filter,
            documents=self.document_filter,
            layout=self.layout_selector,
            hover_columns=self.hover_columns_selector,
        )

        # Reactive statistics view
        self.stats_view = pn.bind(
            self._format_statistics,
            graph_mode=self.graph_mode_selector,
        )

        # Reactive visual controls view
        # IMPORTANT: Must be bound to ALL the same parameters as graph_view
        # to ensure controls stay synchronized with the displayed graph
        self.visual_controls_view = pn.bind(
            self._get_visual_controls_panel,
            graph_mode=self.graph_mode_selector,
            relationship_type=self.relationship_type_selector,
            object_types=self.object_type_filter,
            relationship_types=self.relationship_type_filter,
            theorem_output_types=self.theorem_output_type_filter,
            proof_statuses=self.proof_status_filter,
            axiom_frameworks=self.axiom_framework_filter,
            documents=self.document_filter,
            layout=self.layout_selector,
        )

        # Node details container (updated after graph creation)
        self.node_details_container = pn.Column(
            pn.pane.Markdown(
                "**Click a node to see details**\n\n*Tip: Click on any node in the graph above*",
                width=480,
            )
        )

    def _load_initial_data(self):
        """Load initial data based on data source setting."""
        self._load_registry_data()

    def _on_reload_data(self, event):
        """Handle reload button click."""
        print("Reloading data...")
        self.graph_cache.clear()
        self.layout_cache.clear()
        self._load_registry_data()

    def _on_filter_reset(self, event):
        """Reset all filters to defaults."""
        self.object_type_filter.value = [ot.value for ot in ObjectType]
        self.relationship_type_filter.value = [rt.value for rt in RelationType]
        self.theorem_output_type_filter.value = [tot.value for tot in TheoremOutputType]
        self.proof_status_filter.value = [ps.value for ps in ProofStepStatus]
        self.axiom_framework_filter.value = self.axiom_framework_filter.options  # Select all
        self.document_filter.value = self.document_filter.options  # Select all
        self.layout_selector.value = "neato"

    def _load_registry_data(self):
        """Load registry based on data source selection (document-agnostic)."""
        source = self.data_source_selector.value
        project_root = Path(__file__).parent.parent.parent.parent
        self.current_refined_doc_path = None

        try:
            # Parse source identifier dynamically
            if source == "saved_custom":
                registry_path = Path(self.custom_path_input.value)
                print(f"Loading custom registry from: {registry_path}")
            elif source.startswith("per_doc_"):
                # Parse: per_doc_{document_name}_{type}
                # Example: per_doc_01_fragile_gas_framework_pipeline
                parts = source.split("_", 2)  # Split into ['per', 'doc', '{name}_{type}']
                if len(parts) >= 3:
                    # Extract document name and type from remainder
                    remainder = parts[2]  # e.g., "01_fragile_gas_framework_pipeline"
                    # Type is last component (pipeline or refined)
                    if remainder.endswith("_pipeline"):
                        doc_name = remainder[:-9]  # Remove "_pipeline"
                        registry_type = "pipeline"
                    elif remainder.endswith("_refined"):
                        doc_name = remainder[:-8]  # Remove "_refined"
                        registry_type = "refined"
                    else:
                        raise ValueError(f"Cannot parse registry type from: {source}")

                    registry_path = (
                        project_root / "registries" / "per_document" / doc_name / registry_type
                    )
                    print(f"Loading {doc_name} {registry_type} registry...")
                else:
                    raise ValueError(f"Invalid per_doc identifier: {source}")
            elif source == "combined_pipeline":
                registry_path = project_root / "registries" / "combined" / "pipeline"
                print("Loading combined pipeline registry...")
            elif source == "combined_refined":
                registry_path = project_root / "registries" / "combined" / "refined"
                print("Loading combined refined registry...")
            elif source.startswith("docs_refined|"):
                # Parse: docs_refined|{chapter}|{document}
                parts = source.split("|")
                if len(parts) == 3:
                    chapter, doc = parts[1], parts[2]
                    self.current_refined_doc_path = (
                        project_root / "docs" / "source" / chapter / doc / "refined_data"
                    )
                    print(f"Using docs refined files at: {self.current_refined_doc_path}")
                    self.current_registry = None
                else:
                    raise ValueError(f"Invalid docs_refined identifier: {source}")
            else:
                raise ValueError(f"Unknown data source: {source}")

            # Check if path exists
            if source.startswith("docs_refined|"):
                # Populate document filter for this doc only
                if self.current_refined_doc_path and self.current_refined_doc_path.exists():
                    # Set a single option matching this document
                    doc_name = self.current_refined_doc_path.parent.name
                    self.document_filter.options = [doc_name]
                    self.document_filter.value = [doc_name]
                else:
                    print(
                        f"Error: Refined doc path does not exist: {self.current_refined_doc_path}"
                    )
                # No registry to load; return after setting filters
                self._update_axiom_framework_filter()
                return
            if not registry_path.exists():
                print(f"Error: Registry path does not exist: {registry_path}")
                print("Creating minimal example registry for demonstration...")
                self.current_registry = self._create_minimal_registry()
            else:
                # Load registry
                self.current_registry = load_registry_from_directory(
                    MathematicalRegistry, registry_path
                )
                print(f"✓ Loaded registry from: {registry_path}")

            # Print statistics
            print(f"  - {len(self.current_registry.get_all_objects())} objects")
            print(f"  - {len(self.current_registry.get_all_theorems())} theorems")
            print(f"  - {len(self.current_registry.get_all_relationships())} relationships")
            print(f"  - {len(self.current_registry.get_all_axioms())} axioms")

            # Populate filter options
            self._update_axiom_framework_filter()
            self._update_document_filter()

        except Exception as e:
            print(f"Error loading registry: {e}")
            import traceback

            traceback.print_exc()
            print("Creating minimal example registry...")
            self.current_registry = self._create_minimal_registry()
            self._update_axiom_framework_filter()
            self._update_document_filter()

    def _update_axiom_framework_filter(self):
        """Update axiom framework filter options based on current registry."""
        if self.current_registry is None:
            self.axiom_framework_filter.options = []
            self.axiom_framework_filter.value = []
            return

        # Collect all unique axiom frameworks
        frameworks = set()
        frameworks.update(
            axiom.foundational_framework for axiom in self.current_registry.get_all_axioms()
        )

        # Sort and update filter
        frameworks_sorted = sorted(frameworks)
        self.axiom_framework_filter.options = frameworks_sorted
        self.axiom_framework_filter.value = frameworks_sorted  # Select all by default

    def _update_document_filter(self):
        """Update document filter options based on current registry."""
        if self.current_registry is None:
            self.document_filter.options = []
            self.document_filter.value = []
            return

        # Collect all unique documents from all entity types
        documents = set()

        for obj in self.current_registry.get_all_objects():
            doc = obj.document or "unknown"
            documents.add(doc)

        for thm in self.current_registry.get_all_theorems():
            doc = thm.document or "unknown"
            documents.add(doc)

        for axiom in self.current_registry.get_all_axioms():
            doc = axiom.document or "unknown"
            documents.add(doc)

        # Sort and update filter
        documents_sorted = sorted(documents)
        self.document_filter.options = documents_sorted
        self.document_filter.value = documents_sorted  # Select all by default

    def _create_example_registry(self) -> MathematicalRegistry:
        """Create example registry with sample data for demonstration."""
        from mathster.core.math_types import (
            create_simple_theorem,
            Relationship,
            RelationshipAttribute,
            RelationType,
        )

        registry = MathematicalRegistry()

        # Create some example objects
        obj1 = create_simple_object(
            label="obj-euclidean-gas-discrete",
            name="Euclidean Gas (Discrete)",
            expr="S_N = {(x_i, v_i, r_i)}_{i=1}^N",
            obj_type=ObjectType.SET,
            tags=["euclidean-gas", "discrete", "particle-system"],
        )

        obj2 = create_simple_object(
            label="obj-euclidean-gas-continuous",
            name="Euclidean Gas (Continuous)",
            expr="∂_t μ = L_kin μ + L_clone μ",
            obj_type=ObjectType.FUNCTION,
            tags=["euclidean-gas", "continuous", "pde"],
        )

        obj3 = create_simple_object(
            label="obj-adaptive-gas",
            name="Adaptive Gas",
            expr="S_adaptive = {(x_i, v_i, F_i)}",
            obj_type=ObjectType.SET,
            tags=["adaptive-gas", "discrete"],
        )

        # Create relationships
        rel1 = Relationship(
            label="rel-discrete-continuous-equivalence",
            relationship_type=RelationType.EQUIVALENCE,
            source_object="obj-euclidean-gas-discrete",
            target_object="obj-euclidean-gas-continuous",
            bidirectional=True,
            established_by="thm-mean-field-equivalence",
            expression="S_N ≡ μ_t + O(N^{-1/d})",
            attributes=[
                RelationshipAttribute(
                    label="approx-error-N",
                    expression="O(N^{-1/d})",
                    description="Approximation error scales as N^{-1/d}",
                )
            ],
            tags=["mean-field"],
        )

        rel2 = Relationship(
            label="rel-euclidean-adaptive-extension",
            relationship_type=RelationType.EXTENSION,
            source_object="obj-euclidean-gas-discrete",
            target_object="obj-adaptive-gas",
            bidirectional=False,
            established_by="thm-adaptive-extension",
            expression="Euclidean Gas ⊂ Adaptive Gas",
            tags=["framework-extension"],
        )

        # Create theorems
        thm1 = create_simple_theorem(
            label="thm-mean-field-equivalence",
            name="Mean Field Equivalence",
            output_type=TheoremOutputType.EQUIVALENCE,
            input_objects=["obj-euclidean-gas-discrete", "obj-euclidean-gas-continuous"],
        )
        thm1 = thm1.model_copy(update={"relations_established": [rel1]})

        thm2 = create_simple_theorem(
            label="thm-adaptive-extension",
            name="Adaptive Gas Extension",
            output_type=TheoremOutputType.EXTENSION,
            input_objects=["obj-euclidean-gas-discrete", "obj-adaptive-gas"],
        )
        thm2 = thm2.model_copy(update={"relations_established": [rel2]})

        # Add all to registry
        registry.add_all([obj1, obj2, obj3, rel1, rel2, thm1, thm2])

        return registry

    def _create_minimal_registry(self) -> MathematicalRegistry:
        """Create minimal registry (fallback for errors)."""
        registry = MathematicalRegistry()

        obj = create_simple_object(
            label="obj-test",
            name="Test Object",
            expr="x ∈ ℝ^d",
            obj_type=ObjectType.SET,
            tags=["test"],
        )

        registry.add(obj)
        return registry

    # ==========================================================================
    # GRAPH BUILDERS (to be implemented in next steps)
    # ==========================================================================

    def _build_property_evolution_graph(self) -> nx.DiGraph:
        """Build graph showing property evolution through theorems.

        Nodes: Objects + Axioms + Theorems
        Edges: Axiom → Theorem, Object → Theorem → Enriched Object
        """
        G = nx.DiGraph()

        if self.current_registry is None:
            return G

        # Add object nodes
        for obj in self.current_registry.get_all_objects():
            G.add_node(
                obj.label,
                node_type="object",
                object_type=obj.object_type.value,
                name=obj.name,
                expr=obj.mathematical_expression,
                tags=obj.tags or [],
                type=obj.object_type.value,  # For coloring
                chapter=obj.chapter or "unknown",
                document=obj.document or "unknown",
            )

        # Add axiom nodes
        for axiom in self.current_registry.get_all_axioms():
            G.add_node(
                axiom.label,
                node_type="axiom",
                name=axiom.statement,
                expr=axiom.mathematical_expression,
                framework=axiom.foundational_framework,
                type="axiom",  # For coloring
                chapter=axiom.chapter or "unknown",
                document=axiom.document or "unknown",
            )

        # Add theorem nodes
        for thm in self.current_registry.get_all_theorems():
            G.add_node(
                thm.label,
                node_type="theorem",
                theorem_output_type=thm.output_type.value,
                name=thm.name,
                type="theorem",  # For coloring
                chapter=thm.chapter or "unknown",
                document=thm.document or "unknown",
            )

            # Add edges: input axioms → theorem
            for axiom_label in thm.input_axioms:
                if G.has_node(axiom_label):
                    G.add_edge(axiom_label, thm.label, edge_type="axiom_input")

            # Add edges: input objects → theorem
            for obj_label in thm.input_objects:
                if G.has_node(obj_label):
                    G.add_edge(obj_label, thm.label, edge_type="input")

            # Add edges: theorem → output objects (objects that gain attributes)
            for attr in thm.attributes_added:
                if G.has_node(attr.object_label):
                    G.add_edge(
                        thm.label, attr.object_label, edge_type="output", attribute=attr.label
                    )

        return G

    def _build_relationship_network(self) -> nx.DiGraph:
        """Build graph showing relationships between objects.

        Nodes: Objects only
        Edges: Relationships
        """
        G = nx.DiGraph()

        if self.current_registry is None:
            return G

        # Add object nodes
        for obj in self.current_registry.get_all_objects():
            G.add_node(
                obj.label,
                node_type="object",
                object_type=obj.object_type.value,
                name=obj.name,
                expr=obj.mathematical_expression,
                tags=obj.tags or [],
                type=obj.object_type.value,  # For coloring
                chapter=obj.chapter or "unknown",
                document=obj.document or "unknown",
            )

        # Add relationship edges
        for rel in self.current_registry.get_all_relationships():
            # Add edge from source to target
            if G.has_node(rel.source_object) and G.has_node(rel.target_object):
                G.add_edge(
                    rel.source_object,
                    rel.target_object,
                    edge_type="relationship",
                    relationship_type=rel.relationship_type.value,
                    relationship_label=rel.label,
                    expression=rel.expression,
                    bidirectional=rel.bidirectional,
                    established_by=rel.established_by,
                    type=rel.relationship_type.value,  # For coloring
                )

                # If bidirectional, add reverse edge
                if rel.bidirectional:
                    G.add_edge(
                        rel.target_object,
                        rel.source_object,
                        edge_type="relationship",
                        relationship_type=rel.relationship_type.value,
                        relationship_label=rel.label,
                        expression=rel.expression,
                        bidirectional=True,
                        established_by=rel.established_by,
                        type=rel.relationship_type.value,
                    )

        return G

    def _build_theorem_dag(self) -> nx.DiGraph:
        """Build graph showing theorem dependency structure.

        Nodes: Axioms + Theorems + Internal Lemmas/Propositions
        Edges: Axiom → Theorem, Lemma DAG edges
        """
        G = nx.DiGraph()

        if self.current_registry is None:
            return G

        # Add axiom nodes (foundational dependencies)
        for axiom in self.current_registry.get_all_axioms():
            G.add_node(
                axiom.label,
                node_type="axiom",
                name=axiom.statement,
                expr=axiom.mathematical_expression,
                framework=axiom.foundational_framework,
                type="axiom",
                chapter=axiom.chapter or "unknown",
                document=axiom.document or "unknown",
            )

        # Add theorem nodes
        for thm in self.current_registry.get_all_theorems():
            G.add_node(
                thm.label,
                node_type="theorem",
                theorem_output_type=thm.output_type.value,
                name=thm.name,
                type="theorem",
                chapter=thm.chapter or "unknown",
                document=thm.document or "unknown",
            )

            # Add edges: input axioms → theorem (foundational dependencies)
            for axiom_label in thm.input_axioms:
                if G.has_node(axiom_label):
                    G.add_edge(axiom_label, thm.label, edge_type="axiom_dependency")

            # Add internal lemma/proposition nodes
            for lemma_label in thm.internal_lemmas:
                if not G.has_node(lemma_label):
                    G.add_node(
                        lemma_label,
                        node_type="lemma",
                        parent_theorem=thm.label,
                        type="lemma",
                    )

            for prop_label in thm.internal_propositions:
                if not G.has_node(prop_label):
                    G.add_node(
                        prop_label,
                        node_type="proposition",
                        parent_theorem=thm.label,
                        type="proposition",
                    )

            # Add DAG edges (lemma dependencies)
            for source, target in thm.lemma_dag_edges:
                if G.has_node(source) and G.has_node(target):
                    G.add_edge(source, target, edge_type="dependency")

        return G

    def _build_proof_dataflow_graph(self, theorem_label: str | None = None) -> nx.DiGraph:
        """Build graph showing proof dataflow.

        Nodes: Proof steps
        Edges: Property flow
        """
        G = nx.DiGraph()

        if self.current_registry is None:
            return G

        # Get theorems with attached mathster
        theorems_with_proofs = [
            thm for thm in self.current_registry.get_all_theorems() if thm.proof is not None
        ]

        if not theorems_with_proofs:
            return G

        # If specific theorem requested, filter to that
        if theorem_label:
            theorems_with_proofs = [
                thm for thm in theorems_with_proofs if thm.label == theorem_label
            ]

        # Build proof dataflow for each theorem
        for thm in theorems_with_proofs:
            proof = thm.proof

            # Add proof step nodes (inherit chapter/document from parent theorem)
            for step in proof.steps:
                step_id = f"{proof.proof_id}_{step.step_id}"
                G.add_node(
                    step_id,
                    node_type="proof_step",
                    step_type=step.step_type.value,
                    status=step.status.value,
                    description=(
                        step.natural_language_description
                        if getattr(step, "natural_language_description", None)
                        else ""
                    ),
                    type=step.status.value,  # For coloring by status
                    chapter=thm.chapter or "unknown",
                    document=thm.document or "unknown",
                )

            # Add dataflow edges (property flow between steps)
            # This requires analyzing proof inputs/outputs
            # For now, just add sequential edges
            for i in range(len(proof.steps) - 1):
                source_id = f"{proof.proof_id}_{proof.steps[i].step_id}"
                target_id = f"{proof.proof_id}_{proof.steps[i + 1].step_id}"
                G.add_edge(source_id, target_id, edge_type="sequential")

        return G

    def _build_relationship_type_graph(self, rel_type: str) -> nx.DiGraph:
        """Build graph filtered to specific relationship type.

        Nodes: Objects
        Edges: Relationships of given type
        """
        G = nx.DiGraph()

        if self.current_registry is None:
            return G

        # Add object nodes
        for obj in self.current_registry.get_all_objects():
            G.add_node(
                obj.label,
                node_type="object",
                object_type=obj.object_type.value,
                name=obj.name,
                expr=obj.mathematical_expression,
                tags=obj.tags or [],
                type=obj.object_type.value,
                chapter=obj.chapter or "unknown",
                document=obj.document or "unknown",
            )

        # Add only relationships of specified type
        for rel in self.current_registry.get_all_relationships():
            if rel.relationship_type.value == rel_type:
                if G.has_node(rel.source_object) and G.has_node(rel.target_object):
                    G.add_edge(
                        rel.source_object,
                        rel.target_object,
                        edge_type="relationship",
                        relationship_type=rel.relationship_type.value,
                        relationship_label=rel.label,
                        expression=rel.expression,
                        bidirectional=rel.bidirectional,
                        established_by=rel.established_by,
                        type=rel.relationship_type.value,
                    )

                    if rel.bidirectional:
                        G.add_edge(
                            rel.target_object,
                            rel.source_object,
                            edge_type="relationship",
                            relationship_type=rel.relationship_type.value,
                            relationship_label=rel.label,
                            expression=rel.expression,
                            bidirectional=True,
                            established_by=rel.established_by,
                            type=rel.relationship_type.value,
                        )

        return G

    # ==========================================================================
    # GRAPH VISUALIZATION
    # ==========================================================================

    def _get_or_build_graph(self, mode: str, **kwargs) -> nx.DiGraph:
        """Get graph from cache or build it."""
        # Create cache key
        cache_key = mode
        if getattr(self, "current_refined_doc_path", None):
            cache_key = f"docsrefined_{mode}_{self.current_refined_doc_path}"
        if mode == "relationship_type_specific":
            cache_key = f"{mode}_{kwargs.get('relationship_type', 'EQUIVALENCE')}"
        elif mode == "proof_dataflow":
            cache_key = f"{mode}_{kwargs.get('theorem_label', 'all')}"

        # Check cache
        if cache_key in self.graph_cache:
            return self.graph_cache[cache_key]

        # Build graph
        print(f"Building {mode} graph...")
        # If using docs refined files, build from refined_data files
        if getattr(self, "current_refined_doc_path", None):
            G = self._build_refined_files_graph(self.current_refined_doc_path)
        elif mode == "property_evolution":
            G = self._build_property_evolution_graph()
        elif mode == "relationship_network":
            G = self._build_relationship_network()
        elif mode == "theorem_dag":
            G = self._build_theorem_dag()
        elif mode == "proof_dataflow":
            G = self._build_proof_dataflow_graph(kwargs.get("theorem_label"))
        elif mode == "relationship_type_specific":
            G = self._build_relationship_type_graph(kwargs.get("relationship_type", "EQUIVALENCE"))
        else:
            raise ValueError(f"Unknown graph mode: {mode}")

        # Cache and return
        self._last_graph = G
        self.graph_cache[cache_key] = G
        print(f"✓ Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _build_refined_files_graph(self, refined_dir: Path) -> nx.DiGraph:
        """Build a graph from docs refined_data files for the selected document.

        Nodes: objects, axioms, definitions, theorems, lemmas, propositions, corollaries
        Edges:
          - definition -> object (via definition_label)
          - definition relations: uses/generalizes/required_by
          - theorem inputs: object/axiom/definition -> theorem
          - theorem lemma DAG edges
        """
        G = nx.DiGraph()
        try:
            doc = refined_dir.parent.name
            chapter = refined_dir.parent.parent.name

            import json

            def load_dir(sub: str) -> list[dict]:
                folder = refined_dir / sub
                items: list[dict] = []
                if folder.exists():
                    for p in folder.glob("*.json"):
                        try:
                            items.append(json.loads(p.read_text()))
                        except Exception:
                            pass
                return items

            objects = load_dir("objects")
            axioms = load_dir("axioms")
            definitions = load_dir("definitions")
            theorems = []
            theorems += load_dir("theorems")
            theorems += load_dir("lemmas")
            theorems += load_dir("propositions")
            theorems += load_dir("corollaries")

            # Add nodes
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                label = obj.get("label")
                if not label:
                    continue
                G.add_node(
                    label,
                    node_type="object",
                    object_type=obj.get("object_type"),
                    name=obj.get("name", label),
                    type="object",
                    chapter=chapter,
                    document=doc,
                )
            for ax in axioms:
                if not isinstance(ax, dict):
                    continue
                label = ax.get("label")
                if not label:
                    continue
                G.add_node(
                    label,
                    node_type="axiom",
                    name=ax.get("name", label),
                    type="axiom",
                    chapter=chapter,
                    document=doc,
                )
            for d in definitions:
                if not isinstance(d, dict):
                    continue
                label = d.get("label")
                if not label:
                    continue
                G.add_node(
                    label,
                    node_type="definition",
                    name=d.get("name", label),
                    type="definition",
                    chapter=chapter,
                    document=doc,
                )
            for th in theorems:
                if not isinstance(th, dict):
                    continue
                label = th.get("label")
                if not label:
                    continue
                out_type = th.get("output_type") or th.get("statement_type") or "Property"
                G.add_node(
                    label,
                    node_type="theorem",
                    theorem_output_type=str(out_type),
                    name=th.get("name", label),
                    type="theorem",
                    chapter=chapter,
                    document=doc,
                )

            # Add edges
            # Definitions relations and definition -> object
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                label = obj.get("label")
                def_label = obj.get("definition_label")
                if label and def_label and G.has_node(def_label):
                    G.add_edge(def_label, label, edge_type="defined_by")

            for d in definitions:
                if not isinstance(d, dict):
                    continue
                src = d.get("label")
                rels = d.get("relations") or {}
                if isinstance(rels, dict):
                    # uses
                    for tgt in rels.get("uses", []) or []:
                        if src and G.has_node(tgt):
                            G.add_edge(src, tgt, edge_type="uses")
                    # generalizes
                    for tgt in rels.get("generalizes", []) or []:
                        if src and G.has_node(tgt):
                            G.add_edge(src, tgt, edge_type="generalizes")
                    # required_by
                    for tgt in rels.get("required_by", []) or []:
                        if src and G.has_node(tgt):
                            G.add_edge(src, tgt, edge_type="required_by")

            for th in theorems:
                if not isinstance(th, dict):
                    continue
                tgt = th.get("label")
                # Inputs
                for src in th.get("input_objects", []) or []:
                    if G.has_node(src) and tgt:
                        G.add_edge(src, tgt, edge_type="input_object")
                for src in th.get("input_axioms", []) or []:
                    if G.has_node(src) and tgt:
                        G.add_edge(src, tgt, edge_type="input_axiom")
                for src in th.get("uses_definitions", []) or []:
                    if G.has_node(src) and tgt:
                        G.add_edge(src, tgt, edge_type="uses_definition")
                # Lemma DAG
                edges = th.get("lemma_dag_edges") or []
                for e in edges:
                    if isinstance(e, dict):
                        s = e.get("source")
                        t = e.get("target")
                    else:
                        try:
                            s, t = e
                        except Exception:
                            s = t = None
                    if s and t and G.has_node(s) and G.has_node(t):
                        G.add_edge(s, t, edge_type="lemma_dependency")

        except Exception as e:
            print(f"Error building refined files graph: {e}")
        return G

    def _get_or_compute_layout(
        self, graph: nx.DiGraph, layout_name: str, mode: str
    ) -> dict[str, tuple[float, float]]:
        """Get layout from cache or compute it."""
        cache_key = (mode, layout_name)

        if cache_key in self.layout_cache:
            return self.layout_cache[cache_key]

        print(f"Computing {layout_name} layout for {mode} (may take a few seconds)...")

        if layout_name in {"dot", "neato", "fdp", "sfdp", "circo", "twopi"}:
            layout = create_graphviz_layout(graph, top_to_bottom=False, prog=layout_name)
        elif layout_name == "spectral":
            layout = nx.spectral_layout(graph)
        elif layout_name == "circular":
            layout = nx.circular_layout(graph)
        elif layout_name == "spring":
            layout = nx.spring_layout(graph, k=0.5, iterations=50)
        else:
            print(f"Unknown layout '{layout_name}', using neato")
            layout = create_graphviz_layout(graph, top_to_bottom=False, prog="neato")

        self.layout_cache[cache_key] = layout
        print("✓ Layout computed and cached")
        return layout

    def _create_graph_plot(
        self,
        graph_mode: str,
        relationship_type: str,
        object_types: list[str],
        relationship_types: list[str],
        theorem_output_types: list[str],
        proof_statuses: list[str],
        axiom_frameworks: list[str],
        documents: list[str],
        layout: str,
        hover_columns: list[str],
    ):
        """Create interactive graph plot based on current settings.

        Args:
            graph_mode: Visualization mode (property_evolution, relationship_network, etc.)
            relationship_type: Type of relationship for type-specific view
            object_types: List of object types to include
            relationship_types: List of relationship types to include
            theorem_output_types: List of theorem output types to include
            proof_statuses: List of proof statuses to include
            axiom_frameworks: List of axiom frameworks to include
            documents: List of source documents to include
            layout: Graph layout algorithm (neato, dot, sfdp, etc.)
        """
        if self.current_registry is None and not getattr(self, "current_refined_doc_path", None):
            return pn.pane.Markdown(
                "**No data loaded**\n\nSelect a data source (Docs Refined Files recommended) and click Reload Data.",
                sizing_mode="stretch_width",
            )

        # Get or build graph for current mode
        kwargs = {}
        if graph_mode == "relationship_type_specific":
            kwargs["relationship_type"] = relationship_type

        try:
            graph = self._get_or_build_graph(graph_mode, **kwargs)
        except Exception as e:
            return pn.pane.Markdown(
                f"**Error building graph**\n\n```\n{e}\n```",
                sizing_mode="stretch_width",
            )

        if graph.number_of_nodes() == 0:
            return pn.pane.Markdown(
                f"**Empty graph for mode: {graph_mode}**\n\n"
                "No nodes to display. Try selecting a different data source or graph mode.",
                sizing_mode="stretch_width",
            )

        # Get layout
        layout_positions = self._get_or_compute_layout(graph, layout, graph_mode)

        # Create dataframes for InteractiveGraph
        import pandas as pd

        df_nodes = nodes_as_df(graph, layout_positions)
        # Expose node label as a column for hover tooltips
        try:
            df_nodes["label"] = df_nodes.index
        except Exception:
            pass

        if graph.number_of_edges() > 0:
            df_edges = edges_as_df(graph, data=True)
        else:
            df_edges = pd.DataFrame(columns=["source", "target"])

        # Convert categorical string columns to pandas categorical dtype
        # This allows them to be used for color/size mapping in InteractiveGraph
        categorical_cols = [
            "node_type",
            "object_type",
            "theorem_output_type",
            "type",
            "chapter",
            "document",
        ]
        for col in categorical_cols:
            if col in df_nodes.columns:
                df_nodes[col] = pd.Categorical(df_nodes[col])

        # Apply filters based on node types
        nodes_to_keep = []

        for idx in df_nodes.index:
            node_data = df_nodes.loc[idx]
            keep_node = True

            # Filter by node type
            node_type = node_data.get("node_type", None)

            # Filter objects by object_type
            if node_type == "object":
                if "object_type" in node_data:
                    if node_data["object_type"] not in object_types:
                        keep_node = False

            # Filter theorems by output_type
            elif node_type == "theorem":
                if "theorem_output_type" in node_data:
                    if node_data["theorem_output_type"] not in theorem_output_types:
                        keep_node = False

            # Filter axioms by framework
            elif node_type == "axiom":
                if "framework" in node_data:
                    if node_data["framework"] not in axiom_frameworks:
                        keep_node = False

            # Filter relationships by type (if applicable)
            elif node_type == "relationship":
                # Relationships may have relationship_type attribute
                if "relationship_type" in node_data:
                    if node_data["relationship_type"] not in relationship_types:
                        keep_node = False

            # Filter by document (applies to all node types)
            if keep_node and "document" in node_data:
                if node_data["document"] not in documents:
                    keep_node = False

            if keep_node:
                nodes_to_keep.append(idx)

        # Filter dataframes
        df_nodes = df_nodes.loc[nodes_to_keep]

        # Filter edges to only include edges between remaining nodes
        # Note: edges_as_df creates columns named "from" and "to", not "source" and "target"
        if not df_edges.empty and "from" in df_edges.columns and "to" in df_edges.columns:
            df_edges = df_edges[
                df_edges["from"].isin(nodes_to_keep) & df_edges["to"].isin(nodes_to_keep)
            ]

        # Only ignore columns that actually exist in the dataframe
        # Do not ignore 'label' so it appears in hover tooltips
        potential_ignore_cols = ["statement", "source", "expr", "tags"]
        ignore_node_cols = tuple(col for col in potential_ignore_cols if col in df_nodes.columns)

        # Create InteractiveGraph
        ig = InteractiveGraph(
            df_nodes=df_nodes,
            df_edges=df_edges,
            ignore_node_cols=ignore_node_cols,
            n_cols=3,
        )

        # Store reference for node details and visual controls
        self._current_ig = ig
        self._current_df_nodes = df_nodes

        # Restore reliable wiring via InteractiveGraph helper and update container
        # Configure hover tooltips based on user selection
        try:
            selected_fields = list(hover_columns) if isinstance(hover_columns, list) else []
            if not selected_fields:
                selected_fields = ["label"]
            # Only include fields present in df_nodes
            selected_fields = [f for f in selected_fields if f in df_nodes.columns]
            # Always include 'label' if present
            if "label" in df_nodes.columns and "label" not in selected_fields:
                selected_fields.insert(0, "label")
            hover_tooltips = [(f, f"@{f}") for f in selected_fields]
            ig.dmap = ig.dmap.opts(
                tools=["tap", "hover"],
                hover_tooltips=hover_tooltips,
            )
        except Exception:
            pass
        # Bind directly to xy-based selection so we can log coordinates and debug
        bound = ig.bind_to_stream(self._on_node_select_xy)
        self.node_details_container.objects = [bound]

        # Return only the graph (visual controls now in separate Card)
        return pn.pane.HoloViews(ig.dmap, height=700, sizing_mode="stretch_width")

    def _on_node_select_indexed(self, ix: int, df):
        """Handle node selection once nearest index is computed by the decorator."""
        try:
            if df is None or df.empty:
                return pn.pane.Markdown(
                    "**Click a node to see details**\n\n*No nodes available.*",
                    width=480,
                )
            node_label = df.index[ix]
            logger.info("[NodeClick] ix=%s, label=%s", ix, node_label)
            self._append_console(f"[NodeClick] ix={ix}, label={node_label}")
            return self._render_node_details(str(node_label))
        except Exception as e:
            logger.exception("[NodeClick][ERROR] %s", e)
            self._append_console(f"[NodeClick][ERROR] {e}")
            return pn.pane.Markdown(
                f"**Error displaying node details**\n\n```\n{e}\n```",
                width=480,
            )

    def _on_node_select_xy(self, x: float, y: float):
        """Handle node selection with xy coordinates; show debug info and details."""
        try:
            df = getattr(self, "_current_df_nodes", None)
            if df is None or df.empty:
                logger.info("[NodeClick] x=%.3f, y=%.3f (no nodes df)", x, y)
                self._append_console(f"[NodeClick] x={x:.3f}, y={y:.3f} (no nodes df)")
                return pn.pane.Markdown(
                    "**Click a node to see details**\n\n*No nodes available.*",
                    width=480,
                )

            # Prepare positions and filter invalid rows
            pos_df = df[["x", "y"]].copy()
            pos_df["label"] = df.index
            valid = pos_df[["x", "y"]].notna().all(axis=1)
            pos_df_valid = pos_df[valid]
            if pos_df_valid.empty:
                logger.warning("[NodeClick] all node positions are NaN")
                self._append_console("[NodeClick] all node positions are NaN")
                return pn.pane.Markdown("**No valid node positions**", width=480)

            # Compute nearest by Euclidean distance
            pts = pos_df_valid[["x", "y"]].values
            ix_sub = int(find_closest_point(pts, y, x))
            node_label = pos_df_valid.iloc[ix_sub]["label"]

            # Distances for debugging (top 50 nearest)
            dx = pts[:, 0] - y
            dy = pts[:, 1] - x
            dists = (dx * dx + dy * dy) ** 0.5
            debug_df = pos_df_valid.copy()
            debug_df["dist"] = dists
            debug_df = (
                debug_df.sort_values("dist")
                .reset_index(drop=True)
                .loc[:, ["label", "x", "y", "dist"]]
            )
            pn.pane.DataFrame(debug_df.head(50), width=480, height=220)

            # Header with click coords and selected label
            header_md = f"""**Last Click**
Click: x=`{x:.3f}`, y=`{y:.3f}`
Selected label: `{node_label}`

---
"""
            pn.pane.Markdown(header_md, width=480)

            details = self._render_node_details(str(node_label))
            # Log and console
            logger.info("[NodeClick] x=%.3f, y=%.3f → label=%s", x, y, node_label)
            self._append_console(f"[NodeClick] x={x:.3f}, y={y:.3f} → label={node_label}")

            return pn.Column(details)
        except Exception as e:
            logger.exception("[NodeClick][ERROR] %s", e)
            self._append_console(f"[NodeClick][ERROR] {e}")
            return pn.pane.Markdown(
                f"**Error displaying node details**\n\n```\n{e}\n```",
                width=480,
            )

    def _on_node_tap(self, x: float, y: float):
        """Handle raw tap coordinates → log and render node details.

        Logs tap coords and resolved node (index + label) to console.
        """
        try:
            df = getattr(self, "_current_df_nodes", None)
            if df is None or df.empty:
                logger.info("[NodeTap] x=%.3f, y=%.3f (no nodes df)", x, y)
                return pn.pane.Markdown(
                    "**Click a node to see details**\n\n*No nodes available.*",
                    width=480,
                )

            # Compute closest node index
            points = df[["x", "y"]].values
            ix = int(find_closest_point(points, x, y))
            node_label = df.index[ix]
            logger.info("[NodeTap] x=%.3f, y=%.3f → ix=%s, label=%s", x, y, ix, node_label)
            self._append_console(f"[NodeTap] x={x:.3f}, y={y:.3f} → ix={ix}, label={node_label}")
            details = self._render_node_details(str(node_label))
            header_md = f"""**Last Click**
label: `{node_label}`
ix: `{ix}`
x: `{x:.3f}`, y: `{y:.3f}`

---
"""
            pn.pane.Markdown(header_md, width=480)
            return pn.Column(details)
        except Exception as e:
            logger.exception("[NodeTap][ERROR] %s", e)
            return pn.pane.Markdown(
                f"**Error displaying node details**\n\n```\n{e}\n```",
                width=480,
            )

    def _render_node_details(self, node_label: str):
        """Render node details based on node type.

        Preferred order: enriched/pipeline/raw file rendering; otherwise registry
        entity panel; otherwise graph node metadata (refined-files mode).
        """
        # Attempt to load and render corresponding JSON file as Markdown report
        file_panel = self._try_render_file_report(node_label)
        if file_panel is not None:
            return file_panel

        # If no registry (refined-files mode), show node data from df
        if self.current_registry is None:
            try:
                df = getattr(self, "_current_df_nodes", None)
                if df is not None and node_label in df.index:
                    row = df.loc[node_label]
                    lines = [
                        f"## Node\n\n**Label:** `{node_label}`\n",
                        f"**Type:** {row.get('node_type', 'unknown')}\n",
                    ]
                    if row.get("name"):
                        lines.append(f"**Name:** {row['name']}\n")
                    if row.get("object_type"):
                        lines.append(f"**Object Type:** {row['object_type']}\n")
                    if row.get("theorem_output_type"):
                        lines.append(f"**Output Type:** {row['theorem_output_type']}\n")
                    if row.get("chapter"):
                        lines.append(f"**Chapter:** {row['chapter']}\n")
                    if row.get("document"):
                        lines.append(f"**Document:** {row['document']}\n")
                    return pn.pane.Markdown("\n".join(lines), width=480)
            except Exception:
                pass

        # Try to find the object in registry
        obj = None
        obj_type = None

        # Check if it's a mathematical object
        try:
            obj = self.current_registry.get(node_label)
            if isinstance(obj, Axiom):
                obj_type = "axiom"
            elif isinstance(obj, MathematicalObject):
                obj_type = "object"
            elif isinstance(obj, TheoremBox):
                obj_type = "theorem"
            elif isinstance(obj, Relationship):
                obj_type = "relationship"
        except:
            pass

        if obj is None:
            return pn.pane.Markdown(
                f"**Node:** `{node_label}`\n\n*Not found in registry*",
                width=480,
            )

        # Render based on type
        if obj_type == "axiom":
            return self._render_axiom_details(obj)
        if obj_type == "object":
            return self._render_object_details(obj)
        if obj_type == "theorem":
            return self._render_theorem_details(obj)
        if obj_type == "relationship":
            return self._render_relationship_details(obj)
        return pn.pane.Markdown(
            f"**Node:** `{node_label}`\n\nUnknown type",
            width=480,
        )

    # ------------------------------------------------------------------
    # Enriched/Pipeline JSON → Markdown report integration
    # ------------------------------------------------------------------

    def _try_render_file_report(self, node_label: str):
        """Try to find a corresponding JSON file for the node and render it.

        Uses enriched report renderers for eq-/param-/remark- labels; otherwise
        falls back to pretty-printing JSON as Markdown.
        """
        try:
            chapter, document = self._lookup_node_source(node_label)
            files = self._find_all_entity_files(node_label, chapter, document)

            import json

            enriched_data = pipeline_data = raw_data = None
            enriched_path = files.get("enriched")
            pipeline_path = files.get("pipeline")
            raw_path = files.get("raw")
            if enriched_path and Path(enriched_path).exists():
                enriched_data = json.loads(Path(enriched_path).read_text(encoding="utf-8"))
            if pipeline_path and Path(pipeline_path).exists():
                pipeline_data = json.loads(Path(pipeline_path).read_text(encoding="utf-8"))
            if raw_path and Path(raw_path).exists():
                raw_data = json.loads(Path(raw_path).read_text(encoding="utf-8"))

            # Rendered tab: prefer enriched; fallback to pipeline; else JSON block
            rendered_panel = None
            if enriched_data is not None:
                try:
                    md_body = render_enriched_to_markdown(enriched_data)
                    header = f"**Source file:** `{enriched_path}`\n\n---\n\n"
                    rendered_panel = pn.pane.Markdown(header + md_body, width=480)
                except Exception as e:
                    print(f"Rendered (enriched) failed for {node_label}: {e}")
            if rendered_panel is None and pipeline_data is not None:
                try:
                    md_body = render_enriched_to_markdown(pipeline_data)
                    header = f"**Source file:** `{pipeline_path}`\n\n---\n\n"
                    rendered_panel = pn.pane.Markdown(header + md_body, width=480)
                except Exception as e:
                    print(f"Rendered (pipeline) failed for {node_label}: {e}")
            if rendered_panel is None:
                none_found = not any([enriched_path, pipeline_path, raw_path])
                if none_found:
                    # Explicit missing file indicator
                    msg = [
                        f"### No source files found for `{node_label}`",
                        "",
                        f"Looked in `{chapter}/{document}` under:",
                        "- refined_data/*/{label}.json",
                        "- pipeline_data/*/{label}.json",
                        "- raw_data/*/{label}.json",
                    ]
                    rendered_panel = pn.pane.Markdown(
                        "\n".join(msg).replace("{label}", node_label), width=480
                    )
                else:
                    any_path = enriched_path or pipeline_path or raw_path
                    any_data = enriched_data or pipeline_data or raw_data or {}
                    pretty = json.dumps(any_data, indent=2, ensure_ascii=False)
                    header = f"**Source file:** `{any_path}`\n\n---\n\n" if any_path else ""
                    rendered_panel = pn.pane.Markdown(
                        header + "```json\n" + pretty + "\n```\n", width=480
                    )

            # Build JSON tabs for enriched/pipeline/raw
            def json_tab(title: str, data: dict | None, path):
                if data is None:
                    return (title, pn.pane.Markdown("_No file found_", width=480))
                header = pn.pane.Markdown(f"**Source file:** `{path}`\n\n---\n", width=480)
                view = pn.pane.JSON(data, depth=2, sizing_mode="stretch_width")
                return (title, pn.Column(header, view))

            return pn.Tabs(
                ("Rendered", rendered_panel),
                json_tab("Enriched JSON", enriched_data, enriched_path),
                json_tab("Pipeline JSON", pipeline_data, pipeline_path),
                json_tab("Raw JSON", raw_data, raw_path),
            )
        except Exception as e:
            print(f"Failed rendering file report for {node_label}: {e}")
            return None

    def _lookup_node_source(self, node_label: str) -> tuple[str, str]:
        """Get chapter/document for the node from the current nodes dataframe.

        Returns (chapter, document); defaults to ("1_euclidean_gas", "01_fragile_gas_framework")
        if not available.
        """
        default = ("1_euclidean_gas", "01_fragile_gas_framework")
        try:
            df = getattr(self, "_current_df_nodes", None)
            if df is not None and node_label in df.index:
                row = df.loc[node_label]
                chapter = str(row.get("chapter", "")) if "chapter" in row else ""
                document = str(row.get("document", "")) if "document" in row else ""
                chapter = chapter if chapter and chapter != "unknown" else default[0]
                document = document if document and document != "unknown" else default[1]
                return chapter, document
        except Exception:
            pass
        return default

    def _find_entity_file(self, node_label: str, chapter: str, document: str) -> Path | None:
        """Find a JSON file on disk matching the node label within the doc tree.

        Search order (within the document folder):
        - refined_data/*/{label}.json
        - pipeline_data/*/{label}.json
        - raw_data/*/{label}.json
        If not found, recursively search the document folder for {label}.json.
        As a final fallback, search the entire docs/source tree.
        """
        project_root = Path(__file__).parent.parent.parent.parent
        doc_root = project_root / "docs" / "source" / chapter / document

        candidates: list[Path] = []
        try_subdirs = [
            doc_root / "refined_data" / "objects",
            doc_root / "refined_data" / "axioms",
            doc_root / "refined_data" / "definitions",
            doc_root / "refined_data" / "theorems",
            doc_root / "refined_data" / "lemmas",
            doc_root / "refined_data" / "propositions",
            doc_root / "refined_data" / "corollaries",
            doc_root / "pipeline_data" / "parameters",
            doc_root / "pipeline_data" / "equations",
            doc_root / "pipeline_data" / "remarks",
            doc_root / "raw_data" / "parameters",
            doc_root / "raw_data" / "equations",
            doc_root / "raw_data" / "remarks",
        ]

        for sub in try_subdirs:
            path = sub / f"{node_label}.json"
            if path.exists():
                candidates.append(path)

        if candidates:
            # Prefer refined over pipeline/raw
            def score(p: Path) -> int:
                s = 0
                if "refined_data" in p.parts:
                    s += 2
                if any(seg in p.parts for seg in ("objects", "axioms", "definitions")):
                    s += 1
                return -s  # sort ascending by negative score → highest score first

            return min(candidates, key=score)

        # Recursive search within document folder
        try:
            for p in doc_root.rglob(f"{node_label}.json"):
                return p
        except Exception:
            pass

        # Global search as last resort
        docs_root = project_root / "docs" / "source"
        try:
            for p in docs_root.rglob(f"{node_label}.json"):
                return p
        except Exception:
            pass

        return None

    def _find_all_entity_files(self, node_label: str, chapter: str, document: str) -> dict:
        """Find all relevant JSON files (enriched, pipeline, raw) for a node.

        Returns a dict with keys: 'enriched', 'pipeline', 'raw', each mapping to a Path or None.
        """
        project_root = Path(__file__).parent.parent.parent.parent
        doc_root = project_root / "docs" / "source" / chapter / document

        result = {"enriched": None, "pipeline": None, "raw": None}

        # Direct subdir checks
        enriched_dirs = [
            doc_root / "refined_data" / "objects",
            doc_root / "refined_data" / "axioms",
            doc_root / "refined_data" / "definitions",
            doc_root / "refined_data" / "theorems",
            doc_root / "refined_data" / "lemmas",
            doc_root / "refined_data" / "propositions",
            doc_root / "refined_data" / "corollaries",
        ]
        pipeline_dirs = [
            doc_root / "pipeline_data" / "parameters",
            doc_root / "pipeline_data" / "equations",
            doc_root / "pipeline_data" / "remarks",
        ]
        raw_dirs = [
            doc_root / "raw_data" / "parameters",
            doc_root / "raw_data" / "equations",
            doc_root / "raw_data" / "remarks",
        ]

        for d in enriched_dirs:
            p = d / f"{node_label}.json"
            if p.exists():
                result["enriched"] = p
                break

        for d in pipeline_dirs:
            p = d / f"{node_label}.json"
            if p.exists():
                result["pipeline"] = p
                break

        for d in raw_dirs:
            p = d / f"{node_label}.json"
            if p.exists():
                result["raw"] = p
                break

        # Fallback: recursive search within refined_data/pipeline_data/raw_data
        try:
            if result["enriched"] is None and (doc_root / "refined_data").exists():
                for p in (doc_root / "refined_data").rglob(f"{node_label}.json"):
                    result["enriched"] = p
                    break
        except Exception:
            pass
        try:
            if result["pipeline"] is None and (doc_root / "pipeline_data").exists():
                for p in (doc_root / "pipeline_data").rglob(f"{node_label}.json"):
                    result["pipeline"] = p
                    break
        except Exception:
            pass
        try:
            if result["raw"] is None and (doc_root / "raw_data").exists():
                for p in (doc_root / "raw_data").rglob(f"{node_label}.json"):
                    result["raw"] = p
                    break
        except Exception:
            pass

        return result

    def _render_object_details(self, obj: MathematicalObject):
        """Render mathematical object details."""
        md = f"""
## Mathematical Object

**Label:** `{obj.label}`

**Name:** {obj.name}

**Type:** {obj.object_type.value}

**Expression:**
$$
{obj.mathematical_expression}
$$

"""

        if obj.tags:
            md += f"**Tags:** {', '.join(obj.tags)}\n\n"

        if obj.current_properties:
            md += f"**Properties:** {len(obj.current_properties)} established\n\n"

        return pn.pane.Markdown(md, width=480)

    def _render_theorem_details(self, thm: TheoremBox):
        """Render theorem details."""
        md = f"""
## Theorem

**Label:** `{thm.label}`

**Name:** {thm.name}

**Output Type:** {thm.output_type.value}

"""

        if thm.input_objects:
            md += f"**Input Objects:** {len(thm.input_objects)}\n"
            for obj_label in thm.input_objects[:5]:  # Show first 5
                md += f"- `{obj_label}`\n"
            if len(thm.input_objects) > 5:
                md += f"- *...and {len(thm.input_objects) - 5} more*\n"
            md += "\n"

        if thm.attributes_added:
            md += f"**Attributes Added:** {len(thm.attributes_added)}\n\n"

        if thm.relations_established:
            md += f"**Relationships Established:** {len(thm.relations_established)}\n\n"

        if thm.internal_lemmas:
            md += f"**Internal Lemmas:** {len(thm.internal_lemmas)}\n\n"

        if thm.proof is not None:
            md += f"**Proof Status:** {thm.proof_status}\n\n"

        return pn.pane.Markdown(md, width=480)

    def _render_relationship_details(self, rel: Relationship):
        """Render relationship details."""
        md = f"""
## Relationship

**Label:** `{rel.label}`

**Type:** {rel.relationship_type.value}

**Direction:** {"Bidirectional" if rel.bidirectional else "Directed"}

**Source:** `{rel.source_object}`

**Target:** `{rel.target_object}`

**Expression:**
$$
{rel.expression}
$$

**Established By:** `{rel.established_by}`

"""

        if rel.properties:
            md += "**Properties:**\n"
            for prop in rel.properties:
                md += f"- **{prop.label}**: ${prop.expression}$\n"
            md += "\n"

        if rel.tags:
            md += f"**Tags:** {', '.join(rel.tags)}\n\n"

        return pn.pane.Markdown(md, width=480)

    def _render_axiom_details(self, axiom):
        """Render axiom details."""
        md = f"""
## Axiom

**Label:** `{axiom.label}`

**Statement:** {axiom.statement}

**Framework:** {axiom.foundational_framework}

**Expression:**
$$
{axiom.mathematical_expression}
$$
"""

        if axiom.chapter:
            md += f"\n**Chapter:** {axiom.chapter}\n"
        if axiom.document:
            md += f"\n**Document:** {axiom.document}\n"

        return pn.pane.Markdown(md, width=480)

    def _format_statistics(self, graph_mode: str) -> pn.pane.Markdown:
        """Format statistics for current registry and graph mode."""
        # Refined-files mode: compute stats from current graph if no registry
        if self.current_registry is None:
            try:
                G = getattr(self, "_last_graph", None)
                if G is None or G.number_of_nodes() == 0:
                    return pn.pane.Markdown("**No data loaded**")

                # Basic counts
                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()

                # Counts by node_type
                from collections import Counter

                node_types = Counter(nx.get_node_attributes(G, "node_type").values())
                theorems = [n for n, d in G.nodes(data=True) if d.get("node_type") == "theorem"]
                axiom_frameworks = [
                    d.get("framework", "unknown")
                    for n, d in G.nodes(data=True)
                    if d.get("node_type") == "axiom"
                ]
                frameworks = Counter(axiom_frameworks)
                theorem_types = Counter([
                    G.nodes[n].get("theorem_output_type", "Property") for n in theorems
                ])
                chapters = Counter([d.get("chapter", "unknown") for n, d in G.nodes(data=True)])
                documents = Counter([d.get("document", "unknown") for n, d in G.nodes(data=True)])

                md = f"""
## Refined Data Statistics

**Mode**: {graph_mode.replace("_", " ").title()}

---

### Graph Summary
**Nodes**: {n_nodes}

**Edges**: {n_edges}

---

### Counts by Type
"""
                if node_types:
                    md += "\n**Nodes by Type:**\n"
                    for nt, count in sorted(node_types.items()):
                        md += f"- **{nt}**: {count}\n"
                if theorem_types:
                    md += "\n**Theorems by Output Type:**\n"
                    for t, count in sorted(theorem_types.items()):
                        md += f"- **{t}**: {count}\n"
                if frameworks:
                    md += "\n**Axioms by Framework:**\n"
                    for fw, count in sorted(frameworks.items()):
                        md += f"- **{fw}**: {count}\n"
                if chapters:
                    md += "\n**Items by Chapter:**\n"
                    for ch, count in sorted(chapters.items()):
                        md += f"- **{ch}**: {count}\n"
                if documents:
                    md += "\n**Items by Document:**\n"
                    for doc, count in sorted(documents.items()):
                        md += f"- **{doc}**: {count}\n"

                return pn.pane.Markdown(md, width=480)
            except Exception:
                return pn.pane.Markdown("**No data loaded**")

        # Get counts
        n_objects = len(self.current_registry.get_all_objects())
        n_theorems = len(self.current_registry.get_all_theorems())
        n_relationships = len(self.current_registry.get_all_relationships())
        n_axioms = len(self.current_registry.get_all_axioms())
        n_parameters = len(self.current_registry.get_all_parameters())

        md = f"""
## Registry Statistics

### Current Graph Mode
**Mode**: {graph_mode.replace("_", " ").title()}

---

### Registry Contents
**Objects**: {n_objects}

**Theorems**: {n_theorems}

**Relationships**: {n_relationships}

**Axioms**: {n_axioms}

**Parameters**: {n_parameters}

---

### Counts by Type
"""

        # Count objects by type
        obj_type_counts = {}
        for obj in self.current_registry.get_all_objects():
            ot = obj.object_type.value
            obj_type_counts[ot] = obj_type_counts.get(ot, 0) + 1

        if obj_type_counts:
            md += "\n**Objects by Type:**\n"
            for ot, count in sorted(obj_type_counts.items()):
                color = OBJECT_TYPE_COLORS.get(ot, "#95a5a6")
                md += f'- <span style="color:{color}">**{ot}**</span>: {count}\n'

        # Count relationships by type
        rel_type_counts = {}
        for rel in self.current_registry.get_all_relationships():
            rt = rel.relationship_type.value
            rel_type_counts[rt] = rel_type_counts.get(rt, 0) + 1

        if rel_type_counts:
            md += "\n**Relationships by Type:**\n"
            for rt, count in sorted(rel_type_counts.items()):
                color = RELATIONSHIP_TYPE_COLORS.get(rt, "#95a5a6")
                md += f'- <span style="color:{color}">**{rt}**</span>: {count}\n'

        # Count theorems by output type
        thm_type_counts = {}
        for thm in self.current_registry.get_all_theorems():
            tot = thm.output_type.value
            thm_type_counts[tot] = thm_type_counts.get(tot, 0) + 1

        if thm_type_counts:
            md += "\n**Theorems by Output Type:**\n"
            for tot, count in sorted(thm_type_counts.items()):
                color = THEOREM_OUTPUT_TYPE_COLORS.get(tot, "#95a5a6")
                md += f'- <span style="color:{color}">**{tot}**</span>: {count}\n'

        # Count axioms by foundational framework
        axiom_framework_counts = {}
        for axiom in self.current_registry.get_all_axioms():
            fw = axiom.foundational_framework
            axiom_framework_counts[fw] = axiom_framework_counts.get(fw, 0) + 1

        if axiom_framework_counts:
            md += "\n**Axioms by Framework:**\n"
            for fw, count in sorted(axiom_framework_counts.items()):
                md += f'- <span style="color:#9b59b6">**{fw}**</span>: {count}\n'

        # Count by chapter (objects + theorems)
        chapter_counts = {}
        for obj in self.current_registry.get_all_objects():
            ch = obj.chapter or "unknown"
            chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
        for thm in self.current_registry.get_all_theorems():
            ch = thm.chapter or "unknown"
            chapter_counts[ch] = chapter_counts.get(ch, 0) + 1

        if chapter_counts:
            md += "\n**Items by Chapter:**\n"
            for ch, count in sorted(chapter_counts.items()):
                color = CHAPTER_COLORS.get(ch, "#95a5a6")
                md += f'- <span style="color:{color}">**{ch}**</span>: {count}\n'

        # Count by document (objects + theorems)
        document_counts = {}
        for obj in self.current_registry.get_all_objects():
            doc = obj.document or "unknown"
            document_counts[doc] = document_counts.get(doc, 0) + 1
        for thm in self.current_registry.get_all_theorems():
            doc = thm.document or "unknown"
            document_counts[doc] = document_counts.get(doc, 0) + 1

        if document_counts:
            md += "\n**Items by Document:**\n"
            for doc, count in sorted(document_counts.items()):
                color = DOCUMENT_COLORS.get(doc, "#95a5a6")
                md += f'- <span style="color:{color}">**{doc}**</span>: {count}\n'

        return pn.pane.Markdown(md, width=480)

    def _get_node_details_panel(self):
        """Get the current node details panel."""
        if self.node_details_view is not None:
            return self.node_details_view
        return pn.pane.Markdown(
            "**Click a node to see details**\n\n" "*Tip: Click on any node in the graph above*",
            width=480,
        )

    def _get_visual_controls_panel(
        self,
        graph_mode=None,
        relationship_type=None,
        object_types=None,
        relationship_types=None,
        theorem_output_types=None,
        proof_statuses=None,
        axiom_frameworks=None,
        documents=None,
        layout=None,
    ):
        """Get the visual customization controls from current InteractiveGraph.

        Args:
            graph_mode: Current graph mode (used for reactive updates)
            relationship_type: Relationship type for type-specific view
            object_types: Object type filter values
            relationship_types: Relationship type filter values
            theorem_output_types: Theorem output type filter values
            proof_statuses: Proof status filter values
            axiom_frameworks: Axiom framework filter values
            documents: Document filter values
            layout: Graph layout algorithm

        Note:
            All parameters are needed for reactive binding to work correctly.
            When any parameter changes, this method is called and returns controls
            for the newly created InteractiveGraph instance.
        """
        if hasattr(self, "_current_ig") and self._current_ig is not None:
            return self._current_ig.layout()
        return pn.pane.Markdown(
            "**Visual controls will appear here**\n\n"
            "*Controls load after graph is created. Adjust graph mode, filters, or reload data.*",
            sizing_mode="stretch_width",
        )

    def create_dashboard(self) -> pn.Template:
        """Create the complete dashboard layout."""
        # Sidebar: Data source + Graph mode + Filters
        sidebar_content = [
            # Data source section
            pn.pane.Markdown(
                "## Data Source\n*Select where to load registry from*",
                styles={"font-size": "0.95em"},
                margin=(0, 0, 10, 0),
            ),
            self.data_source_selector,
            self.custom_path_input,
            self.reload_button,
            pn.layout.Divider(),
            # Graph mode section
            pn.pane.Markdown(
                "## Graph Mode\n*Choose visualization type*",
                styles={"font-size": "0.95em"},
                margin=(10, 0, 10, 0),
            ),
            self.graph_mode_selector,
            self.relationship_type_selector,
            pn.layout.Divider(),
            # Filters section
            pn.pane.Markdown(
                "## Filters\n*Updates apply automatically*",
                styles={"font-size": "0.95em"},
                margin=(10, 0, 10, 0),
            ),
            self.object_type_filter,
            self.relationship_type_filter,
            self.theorem_output_type_filter,
            self.proof_status_filter,
            self.axiom_framework_filter,
            self.document_filter,
            pn.layout.Divider(),
            # Layout section
            pn.pane.Markdown(
                "## Layout Algorithm\n*May take 2-5s to compute*",
                styles={"font-size": "0.95em"},
                margin=(10, 0, 10, 0),
            ),
            self.layout_selector,
            pn.layout.Divider(),
            # Reset button
            self.reset_filter_button,
        ]

        # Main content
        main_content = [
            # Graph
            pn.Card(
                pn.panel(self.graph_view),
                title="Proof Pipeline Graph",
                collapsed=False,
                sizing_mode="stretch_width",
                styles={"background": "#f8f9fa"},
            ),
            # Info row (reordered: Node Details → Console → Statistics)
            pn.Row(
                pn.Card(
                    self.node_details_container,
                    title="Node Details",
                    width=1000,
                    height=750,
                    scroll=True,
                    collapsed=False,
                ),
                # pn.Card(
                #     self.console_pane,
                #     title="Console",
                #     width=500,
                #     height=450,
                #     scroll=True,
                #     collapsed=False,
                # ),
                pn.Card(
                    pn.panel(self.stats_view),
                    title="Statistics",
                    width=500,
                    height=750,
                    scroll=True,
                    collapsed=False,
                ),
                sizing_mode="stretch_width",
            ),
            # Visual Customization Controls moved below the info row
            pn.Card(
                pn.panel(self.visual_controls_view),
                title="🎨 Visual Customization",
                collapsed=True,  # Collapsed by default for cleaner view
                sizing_mode="stretch_width",
                styles={"background": "#f8f9fa"},
            ),
        ]

        # Create template
        return pn.template.FastListTemplate(
            title="Proof Pipeline Dashboard",
            sidebar=sidebar_content,
            main=main_content,
            accent_base_color="#3498db",
            header_background="#2c3e50",
        )

    # ------------------------------------------------------------------
    # In-app Console helpers
    # ------------------------------------------------------------------
    def _append_console(self, msg: str) -> None:
        """Append a message to the in-app console and update the pane."""
        try:
            self.console_lines.append(msg)
            # Keep only the most recent 300 lines to bound memory
            if len(self.console_lines) > 300:
                self.console_lines = self.console_lines[-300:]
            text = "```\n" + "\n".join(self.console_lines) + "\n```"
            self.console_pane.object = text
        except Exception:
            # Avoid breaking the UI due to logging errors
            pass


def main():
    """Main entry point - supports both direct execution and panel serve."""
    dashboard = ProofPipelineDashboard()
    return dashboard.create_dashboard()


# Support both execution modes
if __name__ == "__main__":
    # Direct Python execution: open in browser
    template = main()
    pn.serve(template.servable(), port=5006, show=False)
else:
    # Panel serve: register as servable
    template = main()
    template.servable()
