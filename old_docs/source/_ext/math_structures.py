"""
Custom Sphinx extension for mathematical structure directives in FractalAI documentation.

This extension provides custom directives for definitions, theorems, algorithms, etc.
with proper styling and cross-referencing support.
"""

import json
import re

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx.domains import Domain
from sphinx.locale import _
from sphinx.roles import XRefRole
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_id


class MathStructureNode(nodes.Admonition, nodes.Element):
    """Base node class for all mathematical structures."""


def visit_math_structure_node(self, node):
    """Visit method for MathStructureNode."""
    self.visit_admonition(node)


def depart_math_structure_node(self, node):
    """Depart method for MathStructureNode."""
    self.depart_admonition(node)


class MathStructureDirective(SphinxDirective):
    """Base directive for mathematical structures."""

    has_content = True
    required_arguments = 0
    optional_arguments = 1  # For the title
    final_argument_whitespace = True
    option_spec = {
        "label": directives.unchanged,
        "name": directives.unchanged,
        "number": directives.unchanged,
        "class": directives.unchanged,
    }

    # Override these in subclasses
    node_class = MathStructureNode
    struct_type = "structure"
    icon = "📘"
    color_class = "blue"

    def run(self):
        """Process the directive."""
        env = self.env

        # Get label and title
        label = self.options.get("label", "")
        if self.arguments:
            title = self.arguments[0]
        else:
            title = self.options.get("name", self.struct_type.title())

        # Auto-number if requested
        number = self.options.get("number", "")
        if number == "auto":
            # Get domain and increment counter
            domain = env.get_domain("math_structures")
            number = domain.get_next_number(self.struct_type)

        # Create the node
        node = self.node_class("\n".join(self.content))
        node["classes"] = ["admonition", "math-structure", f"math-{self.struct_type}"]
        if "class" in self.options:
            node["classes"].extend(self.options["class"].split())

        # Add label for cross-referencing
        if label:
            node["ids"].append(make_id(env, self.state.document, "", label))
            # Store in domain for cross-referencing
            domain = env.get_domain("math_structures")
            domain.add_structure(self.struct_type, label, env.docname, title, number)

        # Create header
        header = nodes.container()
        header["classes"] = ["math-structure-header"]

        # Add title with number
        if number:
            title_text = f"{self.struct_type.title()} {number}: {title}"
        else:
            title_text = (
                f"{self.struct_type.title()}: {title}" if title else self.struct_type.title()
            )

        title_node = nodes.strong(title_text, title_text)
        header += title_node

        # Add label in header if present
        if label:
            label_node = nodes.inline("", f"{label}")
            label_node["classes"] = ["math-structure-label"]
            header += label_node

        # Create content container
        content_container = nodes.container()
        content_container["classes"] = ["math-structure-content"]

        # Parse content
        self.state.nested_parse(self.content, self.content_offset, content_container)

        # Build final structure
        node += header
        node += content_container

        return [node]


# Define specific directive classes
class DefinitionDirective(MathStructureDirective):
    struct_type = "definition"
    icon = "📘"
    color_class = "blue"


class TheoremDirective(MathStructureDirective):
    struct_type = "theorem"
    icon = "🎯"
    color_class = "pink"


class AlgorithmDirective(MathStructureDirective):
    struct_type = "algorithm"
    icon = "⚙️"
    color_class = "orange"


class LemmaDirective(MathStructureDirective):
    struct_type = "lemma"
    icon = "💠"
    color_class = "purple"


class PropertyDirective(MathStructureDirective):
    struct_type = "property"
    icon = "🔷"
    color_class = "cyan"


class ProofDirective(MathStructureDirective):
    struct_type = "proof"
    icon = "✓"
    color_class = "green"

    def run(self):
        """Override to add QED symbol at the end."""
        nodes_list = super().run()

        # Add QED symbol
        if nodes_list and len(nodes_list[0]) > 1:
            content_container = nodes_list[0][1]
            qed = nodes.container()
            qed["classes"] = ["math-proof-qed"]
            qed += nodes.Text("∎")
            content_container += qed

        return nodes_list


# Science directive for colored scientific method boxes
class ScienceDirective(SphinxDirective):
    """Directive for scientific method documentation with different colors."""

    has_content = True
    required_arguments = 1  # Color: blue, green, orange, purple
    optional_arguments = 1  # Title
    final_argument_whitespace = True
    option_spec = {
        "title": directives.unchanged,
        "class": directives.unchanged,
        "equation-ref": directives.unchanged,  # Reference to equation in table
        "theorem-ref": directives.unchanged,  # Reference to theorem in table
        "algorithm-ref": directives.unchanged,  # Reference to algorithm in table
        "glossary-ref": directives.unchanged,  # Reference to glossary term
        "name": directives.unchanged,  # Custom ID for direct linking
    }

    color_icons = {
        "blue": "💡",
        "green": "🌟",
        "orange": "⚡",
        "purple": "🔮",
    }

    def run(self):
        """Process the science directive."""
        color = self.arguments[0].lower()
        if color not in self.color_icons:
            color = "blue"  # Default

        title = self.options.get("title", "Insight")

        # Create container with unique ID
        container = nodes.container()
        container["classes"] = ["science", color]
        if "class" in self.options:
            container["classes"].extend(self.options["class"].split())

        # Create a unique ID for this science box
        from sphinx.util.nodes import make_id

        # Use custom ID if provided, otherwise generate one from title
        custom_id = self.options.get("name", "")
        if custom_id:
            science_id = make_id(self.env, self.state.document, "", custom_id)
        else:
            science_id = make_id(
                self.env,
                self.state.document,
                "",
                f"science-{title.lower().replace(' ', '-').replace(':', '')}",
            )
        container["ids"] = [science_id]

        # Add title with anchor link
        title_container = nodes.container()
        title_container["classes"] = ["science-title"]
        title_node = nodes.strong(title, title)
        title_container += title_node

        # Add anchor link symbol (simpler approach)
        anchor_text = nodes.Text(" #")
        anchor_span = nodes.inline()
        anchor_span["classes"] = ["headerlink"]
        anchor_span += anchor_text
        title_container += anchor_span

        container += title_container

        # Add content
        content_container = nodes.container()
        content_container["classes"] = ["science-content"]

        # Process content with special handling for display math
        self._process_content_with_math(content_container)

        # Add reference links if provided
        self._add_reference_links(content_container)

        container += content_container

        # Create a proper target node for better cross-referencing
        target = nodes.target("", "", ids=[science_id])
        self.state.document.note_explicit_target(target)

        return [target, container]

    def _add_reference_links(self, parent):
        """Add reference links to various tables if provided."""
        ref_container = nodes.container()
        ref_container["classes"] = ["table-references"]
        has_refs = False

        # Map of reference types to their table names and files
        ref_types = {
            "equation-ref": ("equation", "equations.html", "equations table"),
            "theorem-ref": ("theorem", "theorems.html", "theorems table"),
            "algorithm-ref": ("algorithm", "algorithms.html", "algorithms table"),
            "glossary-ref": ("term", "glossary.html", "glossary"),
        }

        # Create paragraph for all references
        ref_para = nodes.paragraph()
        ref_para["classes"] = ["reference-links"]

        # Add each reference type if provided
        for ref_option, (ref_type, ref_file, table_name) in ref_types.items():
            ref_value = self.options.get(ref_option, "")
            if ref_value:
                if has_refs:
                    ref_para += nodes.Text(" • ")

                ref_para += nodes.Text(f"See {ref_type} ")
                ref_para += nodes.strong(ref_value, ref_value)
                ref_para += nodes.Text(" in the ")

                # Create link to the appropriate table
                ref_link = nodes.reference()
                ref_link["internal"] = True
                ref_link["refuri"] = ref_file
                ref_link += nodes.Text(table_name)
                ref_para += ref_link

                has_refs = True

        if has_refs:
            ref_container += ref_para
            parent += ref_container

    def _process_content_with_math(self, parent):
        """Process content, converting $$ math $$ to proper math nodes."""
        import re

        from docutils import nodes
        from docutils.statemachine import StringList

        content_text = "\n".join(self.content)

        # Pattern to match display math: $$...$$
        display_math_pattern = r"\$\$(.*?)\$\$"

        # Split content by display math blocks
        parts = re.split(display_math_pattern, content_text, flags=re.DOTALL)

        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Regular text - parse normally
                if part.strip():
                    text_lines = part.strip().split("\n")
                    self.state.nested_parse(StringList(text_lines), self.content_offset, parent)
            else:
                # Math content - create math node
                math_node = nodes.math_block()
                math_node["classes"].append("math")
                math_node["nowrap"] = False
                math_node["number"] = None

                # Clean up the math content
                math_content = part.strip()
                math_node.append(nodes.Text(math_content))

                # Wrap in a container div for centering
                math_container = nodes.container()
                math_container["classes"] = ["math-display", "displaymath"]
                math_container += math_node

                parent += math_container


# Domain for cross-referencing
class MathStructuresDomain(Domain):
    """Domain for mathematical structures."""

    name = "math_structures"
    label = "Mathematical Structures"

    roles = {
        "ref": XRefRole(),
        "def": XRefRole(),
        "thm": XRefRole(),
        "alg": XRefRole(),
        "lem": XRefRole(),
        "prop": XRefRole(),
        "proof": XRefRole(),
    }

    initial_data = {
        "structures": {},  # label -> (docname, type, title, number)
        "counters": {},  # type -> current count
    }

    def add_structure(self, struct_type, label, docname, title, number):
        """Add a mathematical structure to the domain."""
        self.data["structures"][label] = (docname, struct_type, title, number)

    def get_next_number(self, struct_type):
        """Get the next auto-number for a structure type."""
        count = self.data["counters"].get(struct_type, 0) + 1
        self.data["counters"][struct_type] = count
        return str(count)

    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        """Resolve cross-references."""
        if target in self.data["structures"]:
            docname, struct_type, title, number = self.data["structures"][target]
            if number:
                text = f"{struct_type.title()} {number}"
            else:
                text = title

            return nodes.reference(
                "",
                text,
                internal=True,
                refuri=builder.get_relative_uri(fromdocname, docname) + "#" + target,
            )

        return None


def setup(app):
    """Setup the extension."""
    # Add CSS for styling
    app.add_css_file("custom.css")

    # Register the node and its visitor methods
    app.add_node(
        MathStructureNode,
        html=(visit_math_structure_node, depart_math_structure_node),
        latex=(visit_math_structure_node, depart_math_structure_node),
        text=(visit_math_structure_node, depart_math_structure_node),
    )

    # Add directives
    app.add_directive("math-definition", DefinitionDirective)
    app.add_directive("math-theorem", TheoremDirective)
    app.add_directive("math-algorithm", AlgorithmDirective)
    app.add_directive("math-lemma", LemmaDirective)
    app.add_directive("math-property", PropertyDirective)
    app.add_directive("math-proof", ProofDirective)
    app.add_directive("science", ScienceDirective)

    # Add domain
    app.add_domain(MathStructuresDomain)

    # Add custom role for science cross-references
    from sphinx.roles import XRefRole

    app.add_role("science", XRefRole())

    # Configuration handler to ensure MyST processes nested directives properly
    def config_inited(app, config):
        """Ensure MyST is configured to handle nested directives."""
        # Enable colon_fence if not already enabled (needed for {math} blocks)
        if "colon_fence" not in config.myst_enable_extensions:
            config.myst_enable_extensions.append("colon_fence")

        # Ensure dollarmath is enabled for math rendering
        if "dollarmath" not in config.myst_enable_extensions:
            config.myst_enable_extensions.append("dollarmath")

    # Connect the configuration handler
    app.connect("config-inited", config_inited)

    return {
        "version": "1.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
