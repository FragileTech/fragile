"""
Relationship Graph Analysis for Mathematical Objects.

This module implements graph structures and algorithms for analyzing
relationships between mathematical objects. Supports:

1. RelationshipGraph: Network of object relationships
2. ObjectLineage: Trace relationship chains
3. EquivalenceClasses: Group equivalent objects
4. FrameworkFlow: Theorem-driven connections

All types follow Lean-compatible patterns from docs/LEAN_EMULATION_GUIDE.md.

Version: 1.0.0
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fragile.proofs.core.pipeline_types import Relationship, RelationType


# =============================================================================
# GRAPH NODE AND EDGE TYPES
# =============================================================================


class GraphNode(BaseModel):
    """
    Node in relationship graph.

    Maps to Lean:
        structure GraphNode where
          id : String
          node_type : String
          tags : List String
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1, description="Object ID")
    node_type: str = Field(..., description="Object type (e.g., 'MathematicalObject')")
    tags: List[str] = Field(default_factory=list, description="Object tags")


class GraphEdge(BaseModel):
    """
    Edge in relationship graph.

    Maps to Lean:
        structure GraphEdge where
          source : String
          target : String
          relationship_id : String
          relationship_type : RelationType
          bidirectional : Bool
    """

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., description="Source object ID")
    target: str = Field(..., description="Target object ID")
    relationship_id: str = Field(..., description="Relationship ID")
    relationship_type: RelationType = Field(..., description="Type of relationship")
    bidirectional: bool = Field(..., description="Whether edge is bidirectional")

    def reverse(self) -> GraphEdge:
        """Create reverse edge (for bidirectional relationships)."""
        return GraphEdge(
            source=self.target,
            target=self.source,
            relationship_id=self.relationship_id,
            relationship_type=self.relationship_type,
            bidirectional=self.bidirectional,
        )


# =============================================================================
# RELATIONSHIP GRAPH
# =============================================================================


class RelationshipGraph:
    """
    Graph structure representing relationships between mathematical objects.

    Uses adjacency list representation for efficient traversal.

    Maps to Lean:
        structure RelationshipGraph where
          nodes : HashMap String GraphNode
          edges : List GraphEdge
          adjacency : HashMap String (List String)
    """

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)

    def add_node(self, node: GraphNode) -> None:
        """Add node to graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        """
        Add edge to graph.

        If bidirectional, also adds reverse edge to adjacency list.
        """
        self.edges.append(edge)
        self._adjacency[edge.source].append(edge)
        self._reverse_adjacency[edge.target].append(edge)

        # Add reverse adjacency for bidirectional edges
        if edge.bidirectional:
            reverse_edge = edge.reverse()
            self._adjacency[edge.target].append(reverse_edge)
            self._reverse_adjacency[edge.source].append(reverse_edge)

    def get_neighbors(self, node_id: str) -> List[str]:
        """
        Get all neighbor node IDs (outgoing edges).

        Pure function (no side effects).
        """
        return [edge.target for edge in self._adjacency.get(node_id, [])]

    def get_incoming_neighbors(self, node_id: str) -> List[str]:
        """
        Get all nodes with incoming edges to this node.

        Pure function (no side effects).
        """
        return [edge.source for edge in self._reverse_adjacency.get(node_id, [])]

    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """Get all outgoing edges from node."""
        return self._adjacency.get(node_id, [])

    def get_edges_to(self, node_id: str) -> List[GraphEdge]:
        """Get all incoming edges to node."""
        return self._reverse_adjacency.get(node_id, [])

    def has_node(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        return node_id in self.nodes

    def has_edge(self, source: str, target: str) -> bool:
        """Check if edge exists between nodes."""
        return any(edge.target == target for edge in self._adjacency.get(source, []))

    def node_count(self) -> int:
        """Get number of nodes."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Get number of edges (not counting reverse edges)."""
        return len(self.edges)

    def get_connected_component(self, start_node: str) -> Set[str]:
        """
        Get all nodes in the connected component containing start_node.

        Uses BFS to find reachable nodes.

        Maps to Lean:
            def get_connected_component (graph : RelationshipGraph) (start : String) : HashSet String :=
              bfs graph [start] {}
        """
        if not self.has_node(start_node):
            return set()

        visited = set()
        queue = deque([start_node])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)

            # Add neighbors (both outgoing and incoming due to bidirectional edges)
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)

        return visited

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Find shortest path from source to target using BFS.

        Returns: List of node IDs representing path, or None if no path exists.

        Maps to Lean:
            def find_path (graph : RelationshipGraph) (source target : String) : Option (List String) :=
              bfs_path graph source target
        """
        if not self.has_node(source) or not self.has_node(target):
            return None

        if source == target:
            return [source]

        visited = {source}
        queue = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()

            for neighbor in self.get_neighbors(current):
                if neighbor == target:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def get_subgraph(self, node_ids: Set[str]) -> RelationshipGraph:
        """
        Extract subgraph containing only specified nodes.

        Pure function (creates new graph).
        """
        subgraph = RelationshipGraph()

        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])

        # Add edges (only if both endpoints are in subgraph)
        for edge in self.edges:
            if edge.source in node_ids and edge.target in node_ids:
                subgraph.add_edge(edge)

        return subgraph


# =============================================================================
# OBJECT LINEAGE
# =============================================================================


class LineagePath(BaseModel):
    """
    Path through relationship graph showing object lineage.

    Maps to Lean:
        structure LineagePath where
          nodes : List String
          edges : List GraphEdge
          length : Nat
    """

    model_config = ConfigDict(frozen=True)

    nodes: List[str] = Field(..., min_items=1, description="Node IDs in path")
    edges: List[GraphEdge] = Field(default_factory=list, description="Edges in path")

    def length(self) -> int:
        """Get path length (number of edges)."""
        return len(self.edges)

    def contains_node(self, node_id: str) -> bool:
        """Check if node is in path."""
        return node_id in self.nodes


class ObjectLineage:
    """
    Trace lineage and derivation chains for mathematical objects.

    Answers questions like:
    - How is object A derived from object B?
    - What are all objects derived from A?
    - What is the derivation depth of A?

    Maps to Lean:
        structure ObjectLineage where
          graph : RelationshipGraph
    """

    def __init__(self, graph: RelationshipGraph):
        self.graph = graph

    def trace_lineage(self, node_id: str, max_depth: Optional[int] = None) -> List[LineagePath]:
        """
        Trace all lineage paths from node.

        Returns all paths from node to other nodes (up to max_depth).

        Maps to Lean:
            def trace_lineage (lineage : ObjectLineage) (node : String) (max_depth : Option Nat) : List LineagePath :=
              dfs_all_paths lineage.graph node max_depth
        """
        if not self.graph.has_node(node_id):
            return []

        paths = []
        self._dfs_lineage(node_id, [node_id], [], paths, max_depth or float("inf"), set())

        return paths

    def _dfs_lineage(
        self,
        current: str,
        path_nodes: List[str],
        path_edges: List[GraphEdge],
        results: List[LineagePath],
        max_depth: int,
        visited: Set[str],
    ) -> None:
        """DFS helper for lineage tracing."""
        if len(path_edges) >= max_depth:
            return

        visited.add(current)

        # Get outgoing edges
        edges = self.graph.get_edges_from(current)

        if not edges:
            # Leaf node - add path
            if len(path_nodes) > 1:  # Don't add single-node paths
                results.append(LineagePath(nodes=path_nodes.copy(), edges=path_edges.copy()))
        else:
            for edge in edges:
                if edge.target not in visited:
                    # Recurse
                    self._dfs_lineage(
                        edge.target,
                        path_nodes + [edge.target],
                        path_edges + [edge],
                        results,
                        max_depth,
                        visited.copy(),
                    )

        # Also add current path if we have edges
        if len(path_nodes) > 1:
            results.append(LineagePath(nodes=path_nodes.copy(), edges=path_edges.copy()))

    def get_ancestors(self, node_id: str, max_depth: Optional[int] = None) -> Set[str]:
        """
        Get all ancestor nodes (nodes that derive current node).

        Uses reverse traversal (incoming edges).
        """
        if not self.graph.has_node(node_id):
            return set()

        ancestors = set()
        queue = deque([(node_id, 0)])
        visited = {node_id}

        while queue:
            current, depth = queue.popleft()

            if max_depth is not None and depth >= max_depth:
                continue

            for neighbor in self.graph.get_incoming_neighbors(current):
                if neighbor not in visited:
                    ancestors.add(neighbor)
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return ancestors

    def get_descendants(self, node_id: str, max_depth: Optional[int] = None) -> Set[str]:
        """
        Get all descendant nodes (nodes derived from current node).

        Uses forward traversal (outgoing edges).
        """
        if not self.graph.has_node(node_id):
            return set()

        descendants = set()
        queue = deque([(node_id, 0)])
        visited = {node_id}

        while queue:
            current, depth = queue.popleft()

            if max_depth is not None and depth >= max_depth:
                continue

            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    descendants.add(neighbor)
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return descendants


# =============================================================================
# EQUIVALENCE CLASSES
# =============================================================================


class EquivalenceClass(BaseModel):
    """
    Set of mutually equivalent objects.

    Maps to Lean:
        structure EquivalenceClass where
          members : List String
          representative : String
    """

    model_config = ConfigDict(frozen=True)

    members: List[str] = Field(..., min_items=1, description="Object IDs in equivalence class")
    representative: str = Field(..., description="Representative element")

    def size(self) -> int:
        """Get size of equivalence class."""
        return len(self.members)

    def contains(self, node_id: str) -> bool:
        """Check if node is in equivalence class."""
        return node_id in self.members


class EquivalenceClassifier:
    """
    Compute equivalence classes from equivalence relationships.

    Uses Union-Find algorithm for efficient equivalence class computation.

    Maps to Lean:
        structure EquivalenceClassifier where
          graph : RelationshipGraph
    """

    def __init__(self, graph: RelationshipGraph):
        self.graph = graph

    def compute_equivalence_classes(self) -> List[EquivalenceClass]:
        """
        Compute all equivalence classes.

        Uses Union-Find algorithm on equivalence relationships.

        Maps to Lean:
            def compute_equivalence_classes (classifier : EquivalenceClassifier) : List EquivalenceClass :=
              union_find classifier.graph equivalence_edges
        """
        # Initialize Union-Find structure
        parent: Dict[str, str] = {}
        rank: Dict[str, int] = {}

        # Initialize each node as its own parent
        for node_id in self.graph.nodes:
            parent[node_id] = node_id
            rank[node_id] = 0

        def find(x: str) -> str:
            """Find root with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            """Union by rank."""
            root_x = find(x)
            root_y = find(y)

            if root_x == root_y:
                return

            # Union by rank
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1

        # Process equivalence edges
        for edge in self.graph.edges:
            if edge.relationship_type == RelationType.EQUIVALENCE:
                union(edge.source, edge.target)

        # Group nodes by root
        classes: Dict[str, List[str]] = defaultdict(list)
        for node_id in self.graph.nodes:
            root = find(node_id)
            classes[root].append(node_id)

        # Create EquivalenceClass objects
        result = []
        for root, members in classes.items():
            if len(members) > 0:
                result.append(EquivalenceClass(members=sorted(members), representative=root))

        return result

    def get_equivalence_class(self, node_id: str) -> Optional[EquivalenceClass]:
        """Get equivalence class containing node."""
        classes = self.compute_equivalence_classes()
        for eq_class in classes:
            if eq_class.contains(node_id):
                return eq_class
        return None

    def are_equivalent(self, node_a: str, node_b: str) -> bool:
        """
        Check if two nodes are equivalent.

        Pure function (no side effects).
        """
        eq_class = self.get_equivalence_class(node_a)
        if eq_class is None:
            return False
        return eq_class.contains(node_b)


# =============================================================================
# FRAMEWORK FLOW
# =============================================================================


class TheoremNode(BaseModel):
    """
    Theorem node in framework flow graph.

    Maps to Lean:
        structure TheoremNode where
          theorem_id : String
          input_objects : List String
          output_objects : List String
          relations_established : List String
    """

    model_config = ConfigDict(frozen=True)

    theorem_id: str = Field(..., description="Theorem ID")
    input_objects: List[str] = Field(default_factory=list, description="Input object IDs")
    output_objects: List[str] = Field(
        default_factory=list, description="Output/derived object IDs"
    )
    relations_established: List[str] = Field(
        default_factory=list, description="Relationship IDs established"
    )


class FrameworkFlow:
    """
    Track how theorems establish relationships and derive new objects.

    Shows the "flow" of the mathematical framework:
    - Which theorems establish which relationships
    - How objects are derived through theorems
    - Dependency chains in the framework

    Maps to Lean:
        structure FrameworkFlow where
          relationship_graph : RelationshipGraph
          theorem_nodes : HashMap String TheoremNode
    """

    def __init__(self, relationship_graph: RelationshipGraph):
        self.relationship_graph = relationship_graph
        self.theorem_nodes: Dict[str, TheoremNode] = {}

    def add_theorem(self, theorem_node: TheoremNode) -> None:
        """Add theorem to framework flow."""
        self.theorem_nodes[theorem_node.theorem_id] = theorem_node

    def get_establishing_theorems(self, relationship_id: str) -> List[str]:
        """
        Get all theorems that establish a given relationship.

        Pure function (no side effects).
        """
        results = []
        for thm_id, thm_node in self.theorem_nodes.items():
            if relationship_id in thm_node.relations_established:
                results.append(thm_id)
        return results

    def get_theorems_for_object(self, object_id: str) -> List[str]:
        """
        Get all theorems that involve an object (input or output).

        Pure function (no side effects).
        """
        results = []
        for thm_id, thm_node in self.theorem_nodes.items():
            if object_id in thm_node.input_objects or object_id in thm_node.output_objects:
                results.append(thm_id)
        return results

    def get_theorem_dependencies(self, theorem_id: str) -> Set[str]:
        """
        Get all theorems that this theorem depends on (transitively).

        A theorem depends on another if it uses objects derived by that theorem.
        """
        if theorem_id not in self.theorem_nodes:
            return set()

        theorem = self.theorem_nodes[theorem_id]
        dependencies = set()

        # For each input object, find theorems that output it
        for input_obj in theorem.input_objects:
            for other_thm_id, other_thm in self.theorem_nodes.items():
                if other_thm_id != theorem_id and input_obj in other_thm.output_objects:
                    dependencies.add(other_thm_id)
                    # Recursively add dependencies
                    dependencies.update(self.get_theorem_dependencies(other_thm_id))

        return dependencies

    def get_framework_layers(self) -> List[Set[str]]:
        """
        Compute layered structure of theorems.

        Layer 0: Theorems with no dependencies (axioms, base definitions)
        Layer 1: Theorems depending only on layer 0
        Layer 2: Theorems depending on layers 0-1
        etc.

        Returns: List of sets, where each set contains theorem IDs at that layer.
        """
        # Compute dependencies for all theorems
        dependencies = {thm_id: self.get_theorem_dependencies(thm_id) for thm_id in self.theorem_nodes}

        # Assign layers
        layers: List[Set[str]] = []
        assigned: Set[str] = set()

        while len(assigned) < len(self.theorem_nodes):
            # Find theorems whose dependencies are all assigned
            current_layer = set()
            for thm_id, deps in dependencies.items():
                if thm_id not in assigned and deps.issubset(assigned):
                    current_layer.add(thm_id)

            if not current_layer:
                # Circular dependencies or error
                break

            layers.append(current_layer)
            assigned.update(current_layer)

        return layers


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def build_relationship_graph_from_registry(registry: any) -> RelationshipGraph:
    """
    Build RelationshipGraph from MathematicalRegistry.

    Pure function (no side effects on registry).
    """
    graph = RelationshipGraph()

    # Add nodes from all objects
    for obj in registry.get_all_objects():
        node = GraphNode(id=obj.label, node_type="MathematicalObject", tags=obj.tags)
        graph.add_node(node)

    for axiom in registry.get_all_axioms():
        node = GraphNode(id=axiom.label, node_type="Axiom", tags=axiom.tags)
        graph.add_node(node)

    for param in registry.get_all_parameters():
        node = GraphNode(id=param.label, node_type="Parameter", tags=[])
        graph.add_node(node)

    for prop in registry.get_all_properties():
        node = GraphNode(id=prop.label, node_type="Attribute", tags=prop.tags)
        graph.add_node(node)

    # Add edges from relationships
    for rel in registry.get_all_relationships():
        edge = GraphEdge(
            source=rel.source_object,
            target=rel.target_object,
            relationship_id=rel.label,
            relationship_type=rel.relationship_type,
            bidirectional=rel.bidirectional,
        )
        graph.add_edge(edge)

    return graph


def build_framework_flow_from_registry(
    registry: any, relationship_graph: RelationshipGraph
) -> FrameworkFlow:
    """
    Build FrameworkFlow from MathematicalRegistry.

    Pure function (no side effects on registry).
    """
    flow = FrameworkFlow(relationship_graph)

    # Add theorem nodes
    for thm in registry.get_all_theorems():
        relation_ids = [rel.label for rel in thm.relations_established]

        # For output objects, extract from relations_established targets
        output_objects = []
        for rel in thm.relations_established:
            output_objects.append(rel.target_object)

        theorem_node = TheoremNode(
            theorem_id=thm.label,
            input_objects=thm.input_objects,
            output_objects=output_objects,
            relations_established=relation_ids,
        )
        flow.add_theorem(theorem_node)

    return flow
