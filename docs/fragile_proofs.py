"""Make labelled proof bodies addressable in the published mathematics."""

from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx_proof.nodes import proof_node


LOGGER = logging.getLogger(__name__)


class LabelledProofDirective(SphinxDirective):
    """Render proof bodies with stable IDs and proof-domain references."""

    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        "class": directives.class_option,
        "label": directives.unchanged_required,
        "name": directives.unchanged_required,
    }

    def run(self):
        label = self.options.get("label", self.options.get("name"))
        anchor = label or f"proof-{self.env.new_serialno('fragile-proof')}"
        section = nodes.admonition(classes=["proof", *self.options.get("class", [])], ids=[anchor])
        section += nodes.title(text="Proof")
        if self.arguments:
            source = self.env.doc2path(self.env.docname)
            self.content.insert(0, "", source=source)
            self.content.insert(0, self.arguments[0], source=source)
        self.state.nested_parse(self.content, self.content_offset, section)
        self.state.document.note_explicit_target(section)
        if label:
            if not hasattr(self.env, "proof_list"):
                self.env.proof_list = {}
            if label in self.env.proof_list:
                LOGGER.warning("Duplicate proof label %s", label, location=section)
            self.env.proof_list[label] = {
                "docname": self.env.docname,
                "countertype": "theorem",
                "realtype": "proof",
                "ids": [anchor],
                "label": label,
                "prio": 0,
                "nonumber": True,
            }
            standard = self.env.get_domain("std")
            standard.labels[label] = (self.env.docname, anchor, "Proof")
            standard.anonlabels[label] = (self.env.docname, anchor)
        wrapper = proof_node()
        wrapper += section
        return [wrapper]


def setup(app):
    app.setup_extension("sphinx_proof")
    app.add_directive_to_domain("prf", "proof", LabelledProofDirective, override=True)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
