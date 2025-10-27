#!/usr/bin/env python3
"""
Proof Pipeline Interactive Dashboard.

Visualizes the mathematical proof pipeline using fragile.proofs data structures:
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
    panel serve src/fragile/proofs/proof_pipeline_dashboard.py --show

    # Or run directly
    python src/fragile/proofs/proof_pipeline_dashboard.py
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import holoviews as hv
import networkx as nx
import numpy as np
import panel as pn

# Import directly from submodules to avoid circular import with sympy
# DO NOT import from fragile.proofs (top-level __init__.py triggers sympy import)
from fragile.proofs.core.pipeline_types import (
    Axiom,
    MathematicalObject,
    ObjectType,
    Relationship,
    RelationType,
    TheoremBox,
    TheoremOutputType,
    create_simple_object,
)
from fragile.proofs.core.proof_system import (
    ProofBox,
    ProofStepStatus,
)
from fragile.proofs.registry.registry import MathematicalRegistry
from fragile.proofs.registry.storage import load_registry_from_directory
from fragile.proofs.relationships.graphs import build_relationship_graph_from_registry
from fragile.shaolin.graph import (
    InteractiveGraph,
    create_graphviz_layout,
    edges_as_df,
    nodes_as_df,
    select_index,
)

hv.extension("bokeh")
pn.extension()


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
    "SET": "#0066cc",         # Deep blue - foundational structures
    "SPACE": "#009999",       # Teal - geometric spaces
    "FUNCTION": "#228833",    # Forest green - transformations
    "OPERATOR": "#dd4488",    # Hot pink - dynamic operators
    "MEASURE": "#ff8800",     # Orange - quantitative measures
    "DISTRIBUTION": "#8844cc", # Purple - probabilistic distributions
    "PROCESS": "#996633",     # Brown - temporal processes
    "OTHER": "#888888",       # Gray - uncategorized
}

# Relationship types (8 types from RelationType enum)
# Directional and structural semantics
RELATIONSHIP_TYPE_COLORS = {
    "EQUIVALENCE": "#77cc00",      # Lime - symmetric equality
    "EMBEDDING": "#0055aa",        # Navy - directed inclusion
    "APPROXIMATION": "#ff9944",    # Warm orange - convergence
    "REDUCTION": "#cc2244",        # Crimson - simplification
    "EXTENSION": "#7744bb",        # Violet - expansion
    "GENERALIZATION": "#00bbdd",   # Bright cyan - abstraction
    "SPECIALIZATION": "#445566",   # Slate - concretization
    "OTHER": "#aaaaaa",            # Light gray - uncategorized
}

# Theorem output types (13 types from TheoremOutputType enum)
# Grouped by theorem nature and result type
THEOREM_OUTPUT_TYPE_COLORS = {
    "Property": "#4466ee",         # Royal blue - descriptive properties
    "Relation": "#22aa77",         # Sea green - relational results
    "Existence": "#44cc44",        # Kelly green - constructive existence
    "Construction": "#00bb77",     # Emerald - explicit construction
    "Uniqueness": "#aa66dd",       # Orchid - characterization
    "Classification": "#8855cc",   # Amethist - categorization
    "Impossibility": "#ee3333",    # Red - negative results
    "Embedding": "#007788",        # Deep teal - structural embedding
    "Extension": "#6633bb",        # Indigo - structural expansion
    "Approximation": "#ff7722",    # Tangerine - convergence/approximation
    "Equivalence": "#99dd00",      # Chartreuse - symmetric equivalence
    "Decomposition": "#cc4499",    # Magenta - structural breakdown
    "Reduction": "#994433",        # Maroon - simplification
}

# Proof step status (3 status levels)
# Traffic light pattern - excellent semantic mapping
PROOF_STEP_STATUS_COLORS = {
    "SKETCHED": "#f1c40f",    # Gold - in progress
    "EXPANDED": "#3498db",    # Dodger blue - complete but unverified
    "VERIFIED": "#2ecc71",    # Success green - reviewed and verified
}

# Node type colors (for mixed graphs)
# Distinct primary colors for different entity types
NODE_TYPE_COLORS = {
    "object": "#0088ee",        # Sky blue - mathematical objects
    "theorem": "#ee4444",       # Fire red - theorems (standout)
    "axiom": "#7733cc",         # Royal purple - foundational axioms
    "relationship": "#33bb55",  # Fresh green - relationships/edges
    "proof_step": "#ffaa00",    # Amber - proof workflow steps
}

# Chapter colors (high-level organizational structure)
# Primary spectrum for major divisions
CHAPTER_COLORS = {
    "1_euclidean_gas": "#0077dd",     # Azure - Euclidean Gas chapter
    "2_geometric_gas": "#00aa66",     # Jade - Geometric Gas chapter
    "3_adaptive_gas": "#ff7744",      # Coral - Adaptive Gas chapter (future)
    "root": "#bbbbbb",                # Silver - root level
    "unknown": "#666666",             # Dim gray - unknown
}

# Document colors (fine-grained organizational structure)
# Full spectrum with perceptual spacing for 16 documents
DOCUMENT_COLORS = {
    "01_fragile_gas_framework": "#1166cc",                                  # Blue
    "02_euclidean_gas": "#ff6600",                                         # Orange
    "03_cloning": "#00aa44",                                               # Green
    "04_wasserstein_contraction": "#dd2255",                               # Red
    "05_kinetic_contraction": "#9944dd",                                   # Purple
    "06_convergence": "#aa5522",                                           # Brown
    "07_mean_field": "#ee55aa",                                            # Pink
    "08_propagation_chaos": "#777777",                                     # Gray
    "09_kl_convergence": "#ccaa00",                                        # Gold
    "10_qsd_exchangeability_theory": "#00cccc",                            # Cyan
    "11_hk_convergence": "#5588ff",                                        # Light blue
    "11_hk_convergence_bounded_density_rigorous_proof": "#88bbff",         # Lighter blue
    "12_quantitative_error_bounds": "#ffaa66",                             # Peach
    "13_geometric_gas_c3_regularity": "#66dd88",                           # Light green
    "root": "#cccccc",                                                      # Light silver
    "unknown": "#555555",                                                   # Darker gray
}


# =============================================================================
# DASHBOARD CLASS
# =============================================================================


class ProofPipelineDashboard:
    """Interactive dashboard for proof pipeline visualization."""

    def __init__(self, default_registry_path: Optional[str] = None):
        """Initialize dashboard.

        Args:
            default_registry_path: Optional path to registry directory
        """
        self.current_registry: Optional[MathematicalRegistry] = None
        self.current_graph_mode = "property_evolution"

        # Graph cache (keyed by mode)
        self.graph_cache: Dict[str, nx.DiGraph] = {}

        # Layout cache (keyed by mode + layout_name)
        self.layout_cache: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}

        # Default data source - use combined_registry if it exists, otherwise example
        if default_registry_path:
            self.default_registry_path = default_registry_path
        else:
            # Dashboard is at project root, so paths are relative to project root
            combined_path = Path(__file__).parent / "combined_registry"
            example_path = Path(__file__).parent / "examples" / "complete_workflow_storage"

            if combined_path.exists():
                self.default_registry_path = str(combined_path)
            else:
                self.default_registry_path = str(example_path)

        # Create UI components
        self._create_data_source_widgets()
        self._create_graph_mode_widgets()
        self._create_filter_widgets()
        self._create_reactive_components()

        # Load initial data
        self._load_initial_data()

    def _create_data_source_widgets(self):
        """Create data source selection widgets."""
        self.data_source_selector = pn.widgets.Select(
            name="Data Source",
            options={
                "Example Registry (Runtime)": "runtime",
                "Saved Registry (Default)": "saved_default",
                "Saved Registry (Custom Path)": "saved_custom",
            },
            value="saved_default",
            width=280,
            description="Where to load mathematical objects from",
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
            description="Filter proofs by status",
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

        # Node details view (will be populated after graph creation)
        self.node_details_view = None

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
        """Load registry based on data source selection."""
        source = self.data_source_selector.value

        try:
            if source == "runtime":
                print("Creating example registry at runtime...")
                self.current_registry = self._create_example_registry()
            elif source == "saved_default":
                print(f"Loading saved registry from: {self.default_registry_path}")
                registry_path = Path(self.default_registry_path)
                if not registry_path.exists():
                    print(f"Warning: Default path does not exist: {registry_path}")
                    print("Falling back to runtime example...")
                    self.current_registry = self._create_example_registry()
                else:
                    self.current_registry = load_registry_from_directory(MathematicalRegistry, registry_path)
            elif source == "saved_custom":
                custom_path = Path(self.custom_path_input.value)
                print(f"Loading saved registry from: {custom_path}")
                if not custom_path.exists():
                    print(f"Error: Custom path does not exist: {custom_path}")
                    print("Falling back to runtime example...")
                    self.current_registry = self._create_example_registry()
                else:
                    self.current_registry = load_registry_from_directory(MathematicalRegistry, custom_path)
            else:
                raise ValueError(f"Unknown data source: {source}")

            print(f"✓ Loaded registry:")
            print(f"  - {len(self.current_registry.get_all_objects())} objects")
            print(f"  - {len(self.current_registry.get_all_theorems())} theorems")
            print(f"  - {len(self.current_registry.get_all_relationships())} relationships")
            print(f"  - {len(self.current_registry.get_all_axioms())} axioms")

            # Populate axiom framework filter options
            self._update_axiom_framework_filter()

            # Populate document filter options
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
        for axiom in self.current_registry.get_all_axioms():
            frameworks.add(axiom.foundational_framework)

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
            doc = obj.document if obj.document else "unknown"
            documents.add(doc)

        for thm in self.current_registry.get_all_theorems():
            doc = thm.document if thm.document else "unknown"
            documents.add(doc)

        for axiom in self.current_registry.get_all_axioms():
            doc = axiom.document if axiom.document else "unknown"
            documents.add(doc)

        # Sort and update filter
        documents_sorted = sorted(documents)
        self.document_filter.options = documents_sorted
        self.document_filter.value = documents_sorted  # Select all by default

    def _create_example_registry(self) -> MathematicalRegistry:
        """Create example registry with sample data for demonstration."""
        from fragile.proofs.core.pipeline_types import (
            RelationshipProperty,
            create_simple_theorem,
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
            properties=[
                RelationshipProperty(
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
                tags=obj.tags if obj.tags else [],
                type=obj.object_type.value,  # For coloring
                chapter=obj.chapter if obj.chapter else "unknown",
                document=obj.document if obj.document else "unknown",
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
                chapter=axiom.chapter if axiom.chapter else "unknown",
                document=axiom.document if axiom.document else "unknown",
            )

        # Add theorem nodes
        for thm in self.current_registry.get_all_theorems():
            G.add_node(
                thm.label,
                node_type="theorem",
                theorem_output_type=thm.output_type.value,
                name=thm.name,
                type="theorem",  # For coloring
                chapter=thm.chapter if thm.chapter else "unknown",
                document=thm.document if thm.document else "unknown",
            )

            # Add edges: input axioms → theorem
            for axiom_label in thm.input_axioms:
                if G.has_node(axiom_label):
                    G.add_edge(axiom_label, thm.label, edge_type="axiom_input")

            # Add edges: input objects → theorem
            for obj_label in thm.input_objects:
                if G.has_node(obj_label):
                    G.add_edge(obj_label, thm.label, edge_type="input")

            # Add edges: theorem → output objects (objects that gain properties)
            for prop in thm.properties_added:
                if G.has_node(prop.object_label):
                    G.add_edge(thm.label, prop.object_label, edge_type="output", property=prop.label)

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
                tags=obj.tags if obj.tags else [],
                type=obj.object_type.value,  # For coloring
                chapter=obj.chapter if obj.chapter else "unknown",
                document=obj.document if obj.document else "unknown",
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
                chapter=axiom.chapter if axiom.chapter else "unknown",
                document=axiom.document if axiom.document else "unknown",
            )

        # Add theorem nodes
        for thm in self.current_registry.get_all_theorems():
            G.add_node(
                thm.label,
                node_type="theorem",
                theorem_output_type=thm.output_type.value,
                name=thm.name,
                type="theorem",
                chapter=thm.chapter if thm.chapter else "unknown",
                document=thm.document if thm.document else "unknown",
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

    def _build_proof_dataflow_graph(self, theorem_label: Optional[str] = None) -> nx.DiGraph:
        """Build graph showing proof dataflow.

        Nodes: Proof steps
        Edges: Property flow
        """
        G = nx.DiGraph()

        if self.current_registry is None:
            return G

        # Get theorems with attached proofs
        theorems_with_proofs = [
            thm for thm in self.current_registry.get_all_theorems()
            if thm.proof is not None
        ]

        if not theorems_with_proofs:
            return G

        # If specific theorem requested, filter to that
        if theorem_label:
            theorems_with_proofs = [
                thm for thm in theorems_with_proofs
                if thm.label == theorem_label
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
                    description=step.description if step.description else "",
                    type=step.status.value,  # For coloring by status
                    chapter=thm.chapter if thm.chapter else "unknown",
                    document=thm.document if thm.document else "unknown",
                )

            # Add dataflow edges (property flow between steps)
            # This requires analyzing proof inputs/outputs
            # For now, just add sequential edges
            for i in range(len(proof.steps) - 1):
                source_id = f"{proof.proof_id}_{proof.steps[i].step_id}"
                target_id = f"{proof.proof_id}_{proof.steps[i+1].step_id}"
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
                tags=obj.tags if obj.tags else [],
                type=obj.object_type.value,
                chapter=obj.chapter if obj.chapter else "unknown",
                document=obj.document if obj.document else "unknown",
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
        if mode == "relationship_type_specific":
            cache_key = f"{mode}_{kwargs.get('relationship_type', 'EQUIVALENCE')}"
        elif mode == "proof_dataflow":
            cache_key = f"{mode}_{kwargs.get('theorem_label', 'all')}"

        # Check cache
        if cache_key in self.graph_cache:
            return self.graph_cache[cache_key]

        # Build graph
        print(f"Building {mode} graph...")
        if mode == "property_evolution":
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
        self.graph_cache[cache_key] = G
        print(f"✓ Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _get_or_compute_layout(
        self, graph: nx.DiGraph, layout_name: str, mode: str
    ) -> Dict[str, Tuple[float, float]]:
        """Get layout from cache or compute it."""
        cache_key = (mode, layout_name)

        if cache_key in self.layout_cache:
            return self.layout_cache[cache_key]

        print(f"Computing {layout_name} layout for {mode} (may take a few seconds)...")

        if layout_name in ["dot", "neato", "fdp", "sfdp", "circo", "twopi"]:
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
        print(f"✓ Layout computed and cached")
        return layout

    def _create_graph_plot(
        self,
        graph_mode: str,
        relationship_type: str,
        object_types: List[str],
        relationship_types: List[str],
        theorem_output_types: List[str],
        proof_statuses: List[str],
        axiom_frameworks: List[str],
        documents: List[str],
        layout: str,
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
        if self.current_registry is None:
            return pn.pane.Markdown(
                "**No registry loaded**\n\nPlease select a data source and click Reload Data.",
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

        if graph.number_of_edges() > 0:
            df_edges = edges_as_df(graph, data=True)
        else:
            df_edges = pd.DataFrame(columns=["source", "target"])

        # Convert categorical string columns to pandas categorical dtype
        # This allows them to be used for color/size mapping in InteractiveGraph
        categorical_cols = ["node_type", "object_type", "theorem_output_type", "type", "chapter", "document"]
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
                df_edges["from"].isin(nodes_to_keep) &
                df_edges["to"].isin(nodes_to_keep)
            ]

        # Only ignore columns that actually exist in the dataframe
        potential_ignore_cols = ["label", "statement", "source", "expr", "tags"]
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

        # Set up node details view
        self.node_details_view = ig.bind_to_stream(self._on_node_select)

        # Return only the graph (visual controls now in separate Card)
        return pn.pane.HoloViews(ig.dmap, height=700, sizing_mode="stretch_width")

    @select_index()
    def _on_node_select(self, ix, df):
        """Handle node selection."""
        if ix is None or df is None or df.empty:
            return pn.pane.Markdown(
                "**Click a node to see details**\n\n"
                "*Tip: Click on any node in the graph above*",
                width=480,
            )

        try:
            node_label = df.index[ix]
            return self._render_node_details(node_label)
        except Exception as e:
            print(f"ERROR in _on_node_select: {e}")
            import traceback
            traceback.print_exc()
            return pn.pane.Markdown(
                f"**Error displaying node details**\n\n```\n{e}\n```",
                width=480,
            )

    def _render_node_details(self, node_label: str):
        """Render node details based on node type."""
        if self.current_registry is None:
            return pn.pane.Markdown("**No registry loaded**", width=480)

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
        elif obj_type == "object":
            return self._render_object_details(obj)
        elif obj_type == "theorem":
            return self._render_theorem_details(obj)
        elif obj_type == "relationship":
            return self._render_relationship_details(obj)
        else:
            return pn.pane.Markdown(
                f"**Node:** `{node_label}`\n\nUnknown type",
                width=480,
            )

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

        if thm.properties_added:
            md += f"**Properties Added:** {len(thm.properties_added)}\n\n"

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

**Direction:** {'Bidirectional' if rel.bidirectional else 'Directed'}

**Source:** `{rel.source_object}`

**Target:** `{rel.target_object}`

**Expression:**
$$
{rel.expression}
$$

**Established By:** `{rel.established_by}`

"""

        if rel.properties:
            md += f"**Properties:**\n"
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
        if self.current_registry is None:
            return pn.pane.Markdown("**No registry loaded**")

        # Get counts
        n_objects = len(self.current_registry.get_all_objects())
        n_theorems = len(self.current_registry.get_all_theorems())
        n_relationships = len(self.current_registry.get_all_relationships())
        n_axioms = len(self.current_registry.get_all_axioms())
        n_parameters = len(self.current_registry.get_all_parameters())

        md = f"""
## Registry Statistics

### Current Graph Mode
**Mode**: {graph_mode.replace('_', ' ').title()}

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
            ch = obj.chapter if obj.chapter else "unknown"
            chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
        for thm in self.current_registry.get_all_theorems():
            ch = thm.chapter if thm.chapter else "unknown"
            chapter_counts[ch] = chapter_counts.get(ch, 0) + 1

        if chapter_counts:
            md += "\n**Items by Chapter:**\n"
            for ch, count in sorted(chapter_counts.items()):
                color = CHAPTER_COLORS.get(ch, "#95a5a6")
                md += f'- <span style="color:{color}">**{ch}**</span>: {count}\n'

        # Count by document (objects + theorems)
        document_counts = {}
        for obj in self.current_registry.get_all_objects():
            doc = obj.document if obj.document else "unknown"
            document_counts[doc] = document_counts.get(doc, 0) + 1
        for thm in self.current_registry.get_all_theorems():
            doc = thm.document if thm.document else "unknown"
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
        else:
            return pn.pane.Markdown(
                "**Click a node to see details**\n\n"
                "*Tip: Click on any node in the graph above*",
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
        if hasattr(self, '_current_ig') and self._current_ig is not None:
            return self._current_ig.layout()
        else:
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

            # Visual Customization Controls (separate Card)
            pn.Card(
                pn.panel(self.visual_controls_view),
                title="🎨 Visual Customization",
                collapsed=True,  # Collapsed by default for cleaner view
                sizing_mode="stretch_width",
                styles={"background": "#f8f9fa"},
            ),

            # Info row
            pn.Row(
                pn.Card(
                    pn.panel(self.stats_view),
                    title="Statistics",
                    width=500,
                    height=450,
                    scroll=True,
                    collapsed=False,
                ),
                pn.Card(
                    pn.panel(self._get_node_details_panel),
                    title="Node Details",
                    width=500,
                    height=450,
                    scroll=True,
                    collapsed=False,
                ),
                sizing_mode="stretch_width",
            ),
        ]

        # Create template
        template = pn.template.FastListTemplate(
            title="Proof Pipeline Dashboard",
            sidebar=sidebar_content,
            main=main_content,
            accent_base_color="#3498db",
            header_background="#2c3e50",
        )

        return template


def main():
    """Main entry point - supports both direct execution and panel serve."""
    dashboard = ProofPipelineDashboard()
    template = dashboard.create_dashboard()
    return template


# Support both execution modes
if __name__ == "__main__":
    # Direct Python execution: open in browser
    template = main()
    pn.serve(template.servable(), port=5006, show=False)
else:
    # Panel serve: register as servable
    template = main()
    template.servable()
