#!/usr/bin/env python3
"""
Visualize the dependency graph for geometric gas theorems.
Creates a summary showing dependency layers.
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_dependency_layers(dependency_graph, execution_order):
    """
    Group theorems into layers based on dependency depth.
    Layer 0: No dependencies
    Layer n: Depends on theorems in layers < n
    """
    # Create lookup
    graph_dict = {item['label']: item for item in dependency_graph}

    # Calculate depth for each theorem
    depths = {}

    def calc_depth(label, visited=None):
        if visited is None:
            visited = set()

        if label in depths:
            return depths[label]

        if label in visited:
            # Circular dependency - shouldn't happen after topological sort
            return 0

        visited.add(label)

        item = graph_dict.get(label)
        if not item:
            return 0

        # Find dependencies that will be proven
        deps = [d['label'] for d in item['dependencies']
                if d['status'] == 'in_current_pipeline' and d.get('will_be_proven', False)]

        if not deps:
            depths[label] = 0
            return 0

        max_dep_depth = max(calc_depth(dep, visited.copy()) for dep in deps)
        depths[label] = max_dep_depth + 1
        return depths[label]

    # Calculate all depths
    for item in dependency_graph:
        calc_depth(item['label'])

    # Group by layer
    layers = defaultdict(list)
    for label, depth in depths.items():
        layers[depth].append(label)

    return layers, depths

def main():
    json_path = Path('docs/source/2_geometric_gas/theorem_dependencies.json')
    output_path = Path('docs/source/2_geometric_gas/DEPENDENCY_LAYERS.md')

    with open(json_path, 'r') as f:
        data = json.load(f)

    layers, depths = analyze_dependency_layers(
        data['dependency_graph'],
        data['execution_order']
    )

    # Write layer analysis
    with open(output_path, 'w') as f:
        f.write("# Dependency Layers for Geometric Gas Theorems\n\n")
        f.write("**Generated**: 2025-10-25\n\n")
        f.write("Theorems grouped by dependency depth. Layer 0 has no dependencies, "
                "layer n depends on theorems from layers 0 to n-1.\n\n")

        f.write(f"## Summary\n\n")
        f.write(f"- Total layers: {max(layers.keys()) + 1}\n")
        f.write(f"- Total theorems: {sum(len(layer) for layer in layers.values())}\n\n")

        for layer_num in sorted(layers.keys()):
            theorems = layers[layer_num]
            f.write(f"## Layer {layer_num} ({len(theorems)} theorems)\n\n")

            if layer_num == 0:
                f.write("**Foundation layer** - No internal dependencies. "
                       "These theorems can be proven in any order.\n\n")
            else:
                f.write(f"**Depends on**: Layers 0-{layer_num - 1}\n\n")

            # Group by document
            by_doc = defaultdict(list)
            graph_dict = {item['label']: item for item in data['dependency_graph']}

            for label in theorems:
                item = graph_dict[label]
                by_doc[item['document']].append((label, item))

            for doc in sorted(by_doc.keys()):
                items = by_doc[doc]
                f.write(f"### {doc}\n\n")

                for label, item in sorted(items, key=lambda x: x[1]['line']):
                    # Find dependencies for this theorem
                    deps = [d['label'] for d in item['dependencies']
                           if d['status'] == 'in_current_pipeline' and d.get('will_be_proven', False)]

                    f.write(f"- `{label}` (line {item['line']})")
                    if deps:
                        f.write(f" - Depends on: {', '.join(f'`{d}`' for d in deps[:3])}")
                        if len(deps) > 3:
                            f.write(f" and {len(deps) - 3} more")
                    f.write("\n")

                f.write("\n")

        # Add critical path analysis
        f.write("## Critical Path Analysis\n\n")
        f.write("The critical path is the longest dependency chain. "
               "The depth of the graph determines the minimum number of sequential proof steps.\n\n")

        max_depth = max(depths.values())
        critical_theorems = [label for label, depth in depths.items() if depth == max_depth]

        f.write(f"- **Maximum depth**: {max_depth}\n")
        f.write(f"- **Theorems at maximum depth** ({len(critical_theorems)}): ")
        f.write(", ".join(f"`{t}`" for t in critical_theorems[:5]))
        if len(critical_theorems) > 5:
            f.write(f" and {len(critical_theorems) - 5} more")
        f.write("\n\n")

        # Show one example critical path
        if critical_theorems:
            example = critical_theorems[0]
            f.write(f"### Example Critical Path (to `{example}`)\n\n")

            path = [example]
            current = example
            graph_dict = {item['label']: item for item in data['dependency_graph']}

            while depths.get(current, 0) > 0:
                item = graph_dict[current]
                deps = [d['label'] for d in item['dependencies']
                       if d['status'] == 'in_current_pipeline' and d.get('will_be_proven', False)]

                if not deps:
                    break

                # Find dependency with maximum depth
                max_dep = max(deps, key=lambda d: depths.get(d, 0))
                path.insert(0, max_dep)
                current = max_dep

            for i, label in enumerate(path):
                item = graph_dict[label]
                f.write(f"{i + 1}. `{label}` ({item['document']}:{item['line']})\n")

            f.write("\n")

    print(f"✓ Layer analysis written to {output_path}")
    print(f"\nLayer distribution:")
    for layer_num in sorted(layers.keys()):
        print(f"  Layer {layer_num}: {len(layers[layer_num])} theorems")

    max_depth = max(depths.values())
    print(f"\nCritical path depth: {max_depth}")

if __name__ == '__main__':
    main()
