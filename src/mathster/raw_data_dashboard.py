#!/usr/bin/env python3
"""
Raw Data Visualization Dashboard.

Interactive dashboard for exploring raw mathematical entities from docs/source/*/raw_data/
directories, displaying JSON data side-by-side with rendered markdown source (with LaTeX).

Features:
- Auto-discovery of raw_data directories and entities
- Filtering by entity type, document, chapter, and text search
- Split view: Raw JSON (left) + Rendered Markdown with LaTeX (right)
- Entity list with clickable items
- Statistics and coverage tracking

Usage:
    # Run with panel serve
    panel serve src/fragile/mathster/raw_data_dashboard.py --show

    # Or run directly
    python src/fragile/mathster/raw_data_dashboard.py
"""

import json
import logging
from pathlib import Path
from typing import Any

import panel as pn


hv_extension = "bokeh"  # Using bokeh for consistency

# Enable MathJax for LaTeX rendering in Markdown panes
pn.extension("mathjax")

# Local logger
logger = logging.getLogger("fragile.mathster.raw_data_dashboard")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.INFO)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# =============================================================================
# COLOR SCHEMES
# =============================================================================

# Entity type colors (semantic mapping)
ENTITY_TYPE_COLORS = {
    "axioms": "#7733cc",  # Purple - foundational
    "definitions": "#0088ee",  # Blue - structural
    "theorems": "#ee4444",  # Red - key results
    "lemmas": "#33bb55",  # Green - supporting results
    "propositions": "#22aa77",  # Teal - intermediate results
    "corollaries": "#ff7722",  # Orange - derived results
    "parameters": "#ffaa00",  # Amber - configuration
    "remarks": "#888888",  # Gray - commentary
    "mathster": "#9944dd",  # Violet - logical derivations
    "objects": "#0066cc",  # Deep blue - mathematical objects
}


# =============================================================================
# DASHBOARD CLASS
# =============================================================================


class RawDataDashboard:
    """Interactive dashboard for raw data visualization."""

    def __init__(self):
        """Initialize dashboard."""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.docs_root = self.project_root / "docs" / "source"

        # State
        self.current_document_path: Path | None = None
        self.all_entities: list[dict[str, Any]] = []
        self.filtered_entities: list[dict[str, Any]] = []
        self.selected_entity: dict[str, Any] | None = None

        # Create UI components
        self._create_filter_widgets()
        self._create_reactive_components()

        # Load initial data
        self._load_initial_data()

    def _discover_available_documents(self) -> dict[str, str]:
        """Discover documents with raw_data directories.

        Returns:
            Dict mapping display name to identifier
        """
        options = {}

        if not self.docs_root.exists():
            logger.warning(f"Docs root does not exist: {self.docs_root}")
            return options

        for chapter_dir in sorted(self.docs_root.iterdir()):
            if not chapter_dir.is_dir() or chapter_dir.name.startswith("."):
                continue

            for doc_dir in sorted(chapter_dir.iterdir()):
                if not doc_dir.is_dir() or doc_dir.name.startswith("."):
                    continue

                raw_data_dir = doc_dir / "raw_data"
                if raw_data_dir.exists() and raw_data_dir.is_dir():
                    # Count entities
                    entity_count = sum(
                        1
                        for subdir in raw_data_dir.iterdir()
                        if subdir.is_dir()
                        for f in subdir.glob("*.json")
                        if f.name != "refinement_report.json"
                    )

                    display_name = f"{chapter_dir.name} / {doc_dir.name} ({entity_count} entities)"
                    identifier = f"{chapter_dir.name}|{doc_dir.name}"
                    options[display_name] = identifier

        return options

    def _create_filter_widgets(self):
        """Create filter control widgets."""
        # Document selector
        available_docs = self._discover_available_documents()
        default_value = None
        if available_docs:
            # Prefer 01_fragile_gas_framework
            for identifier in available_docs.values():
                if "01_fragile_gas_framework" in identifier:
                    default_value = identifier
                    break
            if default_value is None:
                default_value = next(iter(available_docs.values()))

        self.document_selector = pn.widgets.Select(
            name="Document",
            options=available_docs,
            value=default_value,
            width=380,
            description="Select document to explore",
        )
        self.document_selector.param.watch(self._on_document_change, "value")

        # Entity type filter
        entity_types = [
            "axioms",
            "definitions",
            "theorems",
            "lemmas",
            "propositions",
            "corollaries",
            "parameters",
            "remarks",
            "mathster",
            "objects",
        ]
        self.entity_type_filter = pn.widgets.MultiChoice(
            name="Entity Types",
            options=entity_types,
            value=entity_types,
            width=380,
            description="Filter by entity type",
        )
        self.entity_type_filter.param.watch(self._on_filter_change, "value")

        # Search input
        self.search_input = pn.widgets.TextInput(
            name="Search (label or content)",
            placeholder="e.g., boundary, axiom-",
            width=380,
        )
        self.search_input.param.watch(self._on_filter_change, "value")

        # Sort selector
        self.sort_selector = pn.widgets.Select(
            name="Sort By",
            options={
                "Label (A-Z)": "label",
                "Line Number": "line_number",
                "Entity Type": "entity_type",
            },
            value="line_number",
            width=380,
        )
        self.sort_selector.param.watch(self._on_filter_change, "value")

        # Show entities with missing line ranges
        self.show_missing_ranges = pn.widgets.Checkbox(
            name="Show entities missing line ranges",
            value=True,
            width=380,
        )
        self.show_missing_ranges.param.watch(self._on_filter_change, "value")

        # Reset button
        self.reset_button = pn.widgets.Button(
            name="Reset Filters",
            button_type="warning",
            width=380,
        )
        self.reset_button.on_click(self._on_reset_filters)

    def _create_reactive_components(self):
        """Create reactive components."""
        # Entity list view (reactive)
        self.entity_list_view = pn.bind(
            self._render_entity_list,
            entity_types=self.entity_type_filter,
            search_text=self.search_input,
            sort_by=self.sort_selector,
            show_missing=self.show_missing_ranges,
        )

        # Statistics view (reactive)
        self.stats_view = pn.bind(
            self._render_statistics,
            entity_types=self.entity_type_filter,
            search_text=self.search_input,
        )

        # JSON panel (always visible, updated when entity selected)
        self.json_panel = pn.Column(
            pn.pane.Markdown(
                "**No entity selected**\n\n*Click an entity from the list to view its JSON data*",
                sizing_mode="stretch_width",
                styles={"text-align": "center", "color": "#666"},
            ),
            sizing_mode="stretch_width",
        )

        # Full markdown document viewer with line numbers
        self.markdown_document_panel = pn.pane.HTML(
            "<p><i>Select a document to view source</i></p>",
            sizing_mode="stretch_both",
            styles={"overflow": "auto", "padding": "10px", "background": "#f8f9fa"},
        )

        # Track selected line range for highlighting
        self.selected_line_range: list[int] | None = None

    def _load_initial_data(self):
        """Load initial data based on document selector."""
        self._load_document_data()
        self._update_markdown_document_panel()

    def _on_document_change(self, event):
        """Handle document selection change."""
        self._load_document_data()
        self._apply_filters()
        # Update markdown document panel (clear any previous highlighting)
        self.selected_line_range = None
        self._update_markdown_document_panel()

    def _on_filter_change(self, event):
        """Handle filter change."""
        self._apply_filters()

    def _on_reset_filters(self, event):
        """Reset all filters to defaults."""
        entity_types = list(self.entity_type_filter.options)
        self.entity_type_filter.value = entity_types
        self.search_input.value = ""
        self.sort_selector.value = "line_number"
        self.show_missing_ranges.value = True

    def _load_document_data(self):
        """Load entities from selected document."""
        identifier = self.document_selector.value
        if not identifier:
            logger.info("No document selected")
            self.all_entities = []
            self.current_document_path = None
            return

        # Parse identifier: chapter|document
        parts = identifier.split("|")
        if len(parts) != 2:
            logger.error(f"Invalid identifier: {identifier}")
            self.all_entities = []
            self.current_document_path = None
            return

        chapter, document = parts
        doc_path = self.docs_root / chapter / document
        raw_data_path = doc_path / "raw_data"

        if not raw_data_path.exists():
            logger.error(f"Raw data path does not exist: {raw_data_path}")
            self.all_entities = []
            self.current_document_path = None
            return

        # Find the markdown file for this document
        # The markdown file is at: docs/source/{chapter}/{document}.md
        markdown_file = self.docs_root / chapter / f"{document}.md"

        if not markdown_file.exists():
            logger.warning(f"Markdown file not found: {markdown_file}")
            self.current_document_path = None
        else:
            self.current_document_path = markdown_file

        logger.info(f"Loading entities from: {raw_data_path}")
        logger.info(f"Markdown file: {self.current_document_path}")

        # Load all entities
        entities = []
        for entity_type_dir in raw_data_path.iterdir():
            if not entity_type_dir.is_dir() or entity_type_dir.name.startswith("."):
                continue

            entity_type = entity_type_dir.name

            for json_file in entity_type_dir.glob("*.json"):
                # Skip metadata files
                if json_file.name in {
                    "refinement_report.json",
                    "object_refinement_report.json",
                    "object_fix_report.json",
                }:
                    continue

                try:
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)

                    # Add metadata
                    data["_entity_type"] = entity_type
                    data["_file_path"] = str(json_file)

                    # Extract line number for sorting
                    line_range = data.get("source", {}).get("line_range")
                    if line_range and line_range.get("lines"):
                        data["_line_number"] = line_range["lines"][0][0]
                    else:
                        data["_line_number"] = float("inf")

                    entities.append(data)

                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")

        self.all_entities = entities
        logger.info(f"Loaded {len(entities)} entities")

        # Apply filters
        self._apply_filters()

    def _apply_filters(self):
        """Apply current filters to all entities."""
        if not self.all_entities:
            self.filtered_entities = []
            return

        filtered = self.all_entities

        # Filter by entity type
        entity_types = set(self.entity_type_filter.value)
        filtered = [e for e in filtered if e.get("_entity_type") in entity_types]

        # Filter by search text
        search_text = self.search_input.value.strip().lower()
        if search_text:
            filtered = [
                e
                for e in filtered
                if search_text in e.get("label", "").lower()
                or search_text in str(e.get("full_statement_text", "")).lower()
                or search_text in str(e.get("statement", "")).lower()
                or search_text in str(e.get("name", "")).lower()
            ]

        # Filter by missing line ranges
        if not self.show_missing_ranges.value:
            filtered = [
                e
                for e in filtered
                if e.get("source", {}).get("line_range")
                and e.get("source", {}).get("line_range", {}).get("lines")
            ]

        # Sort
        sort_by = self.sort_selector.value
        if sort_by == "label":
            filtered = sorted(filtered, key=lambda e: e.get("label", ""))
        elif sort_by == "line_number":
            filtered = sorted(filtered, key=lambda e: e.get("_line_number", float("inf")))
        elif sort_by == "entity_type":
            filtered = sorted(
                filtered,
                key=lambda e: (e.get("_entity_type", ""), e.get("_line_number", float("inf"))),
            )

        self.filtered_entities = filtered

    def _render_entity_list(
        self, entity_types: list[str], search_text: str, sort_by: str, show_missing: bool
    ):
        """Render the entity list based on current filters."""
        if not self.filtered_entities:
            return pn.pane.Markdown(
                "**No entities found**\n\nTry adjusting filters or selecting a different document.",
                sizing_mode="stretch_width",
            )

        # Create entity cards
        entity_items = []
        for entity in self.filtered_entities[:500]:  # Limit to 500 for performance
            label = entity.get("label", "unknown")
            entity_type = entity.get("_entity_type", "unknown")
            line_range = entity.get("source", {}).get("line_range", {}).get("lines")

            # Get content preview
            content = (
                entity.get("full_statement_text")
                or entity.get("statement")
                or entity.get("name")
                or ""
            )
            preview = str(content)[:100].replace("\n", " ").strip()
            if len(str(content)) > 100:
                preview += "..."

            # Line range display
            if line_range:
                start, end = line_range[0]
                line_info = f"Lines {start}-{end}"
            else:
                line_info = "No line range"

            # Color badge
            ENTITY_TYPE_COLORS.get(entity_type, "#888888")

            # Create clickable button with custom styling
            button = pn.widgets.Button(
                name=f"{entity_type.upper()}: {label}",
                button_type="light",
                sizing_mode="stretch_width",
                styles={
                    "text-align": "left",
                    "padding": "12px",
                    "margin-bottom": "8px",
                },
            )
            button.on_click(lambda event, e=entity: self._on_entity_select(e))

            # Create info markdown
            info_md = f"""
<div style="font-size: 0.85em; color: #666; margin-bottom: 4px; margin-left: 12px;">{line_info}</div>
<div style="font-size: 0.9em; color: #555; margin-left: 12px; margin-bottom: 8px;">{preview}</div>
"""
            info_pane = pn.pane.HTML(info_md, sizing_mode="stretch_width")

            # Combine button and info
            item = pn.Column(button, info_pane, sizing_mode="stretch_width")

            entity_items.append(item)

        return pn.Column(
            *entity_items,
            scroll=True,
            height=800,
            sizing_mode="stretch_width",
        )

    def _on_entity_select(self, entity: dict):
        """Handle entity selection."""
        self.selected_entity = entity
        logger.info(f"Selected entity: {entity.get('label')}")

        # Update JSON panel
        self._update_json_panel()

        # Update markdown document panel with highlighting
        source = entity.get("source", {})
        line_range = source.get("line_range", {}).get("lines")
        if line_range and len(line_range) > 0:
            self.selected_line_range = line_range[0]  # [start, end]
            self._update_markdown_document_panel()
        else:
            self.selected_line_range = None
            self._update_markdown_document_panel()

    def _update_json_panel(self):
        """Update JSON panel with selected entity."""
        if not self.selected_entity:
            self.json_panel.objects = [
                pn.pane.Markdown(
                    "**No entity selected**\n\n*Click an entity from the list to view its JSON data*",
                    sizing_mode="stretch_width",
                    styles={"text-align": "center", "color": "#666"},
                )
            ]
            return

        label = self.selected_entity.get("label", "unknown")
        entity_type = self.selected_entity.get("_entity_type", "unknown")
        self.selected_entity.get("_file_path", "")

        # Header (compact for narrow column)
        header = pn.pane.Markdown(
            f"**{label}**\n\n*Type:* {entity_type}",
            sizing_mode="stretch_width",
            styles={"font-size": "0.9em"},
        )

        # JSON viewer
        # Remove internal metadata fields for display
        display_data = {k: v for k, v in self.selected_entity.items() if not k.startswith("_")}
        json_pane = pn.pane.JSON(display_data, depth=2, sizing_mode="stretch_width")

        self.json_panel.objects = [header, json_pane]

    def _update_markdown_document_panel(self):
        """Update the full markdown document panel with optional highlighting."""
        html_content = self._render_full_markdown_with_lines(
            file_path=self.current_document_path, highlight_range=self.selected_line_range
        )
        self.markdown_document_panel.object = html_content

    def _extract_markdown(self, file_path: str, line_range: list) -> str:
        """Extract markdown section from file using line range.

        Args:
            file_path: Path to markdown file
            line_range: [[start, end]] array (1-indexed)

        Returns:
            Extracted markdown content
        """
        # Resolve relative path
        if not Path(file_path).is_absolute():
            full_path = self.project_root / file_path
        else:
            full_path = Path(file_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {full_path}")

        start, end = line_range[0]

        with open(full_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Validate bounds
        if start < 1 or end > len(lines):
            raise ValueError(
                f"Line range [{start}, {end}] out of bounds (file has {len(lines)} lines)"
            )

        # Extract lines (convert from 1-indexed to 0-indexed)
        return "".join(lines[start - 1 : end])

    def _render_full_markdown_with_lines(
        self, file_path: Path | None = None, highlight_range: list[int] | None = None
    ) -> str:
        """Render full markdown document with line numbers and optional highlighting.

        Args:
            file_path: Path to markdown file (defaults to current document)
            highlight_range: [start, end] line range to highlight (1-indexed)

        Returns:
            HTML string with line-numbered markdown
        """
        if file_path is None and self.current_document_path is None:
            return "<p><i>No document selected</i></p>"

        target_path = file_path or self.current_document_path

        if not target_path.exists():
            return f"<p><i>File not found: {target_path}</i></p>"

        try:
            with open(target_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Build HTML with line numbers
            html_parts = [
                """
                <style>
                .line-numbered-markdown {
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 0.9em;
                    line-height: 1.5;
                    background: #ffffff;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .markdown-line {
                    display: flex;
                    padding: 2px 0;
                }
                .line-number {
                    min-width: 50px;
                    padding-right: 15px;
                    text-align: right;
                    color: #999;
                    user-select: none;
                    border-right: 2px solid #e0e0e0;
                    margin-right: 15px;
                    font-weight: bold;
                }
                .line-content {
                    flex: 1;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    color: #333;
                }
                .highlighted-line {
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding-left: 8px;
                    scroll-margin-top: 100px;
                    scroll-margin-bottom: 100px;
                }
                .highlighted-line .line-number {
                    color: #856404;
                    font-weight: bold;
                }
                </style>
                <div class="line-numbered-markdown">
                """
            ]

            # Determine highlight range
            highlight_start = None
            highlight_end = None
            if highlight_range and len(highlight_range) == 2:
                highlight_start, highlight_end = highlight_range

            # Render each line
            for i, line in enumerate(lines, start=1):
                is_highlighted = (
                    highlight_start is not None
                    and highlight_end is not None
                    and highlight_start <= i <= highlight_end
                )

                # Escape HTML in content
                escaped_line = (
                    line.rstrip("\n")
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                )

                line_class = (
                    "markdown-line highlighted-line" if is_highlighted else "markdown-line"
                )

                # Add anchor ID to first highlighted line for scroll targeting
                anchor_id = f' id="line-{i}"' if (is_highlighted and i == highlight_start) else ""

                html_parts.append(
                    f'<div{anchor_id} class="{line_class}">'
                    f'<div class="line-number">{i}</div>'
                    f'<div class="line-content">{escaped_line}</div>'
                    f"</div>"
                )

            html_parts.append("</div>")

            # Add scroll-to script if highlighted
            if highlight_start:
                html_parts.append(
                    f"""
                    <script>
                    // Function to scroll to highlighted line with multiple strategies
                    function scrollToHighlighted() {{
                        // Try anchor-based scroll first
                        var anchor = document.getElementById('line-{highlight_start}');
                        if (anchor) {{
                            anchor.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                            return true;
                        }}

                        // Fallback to class-based selection
                        var highlighted = document.querySelector('.highlighted-line');
                        if (highlighted) {{
                            highlighted.scrollIntoView({{behavior: 'smooth', block: 'center'}});

                            // Also try direct scroll manipulation for nested containers
                            var parent = highlighted.parentElement;
                            if (parent) {{
                                var topPos = highlighted.offsetTop - (parent.clientHeight / 2);
                                parent.scrollTop = topPos;
                            }}
                            return true;
                        }}
                        return false;
                    }}

                    // Retry scroll multiple times with increasing delays to ensure it works
                    setTimeout(scrollToHighlighted, 300);
                    setTimeout(scrollToHighlighted, 600);
                    setTimeout(scrollToHighlighted, 1000);
                    </script>
                    """
                )

            return "".join(html_parts)

        except Exception as e:
            logger.exception(f"Error rendering markdown with line numbers: {e}")
            return f"<p><i>Error rendering document: {e}</i></p>"

    def _render_statistics(self, entity_types: list[str], search_text: str):
        """Render statistics panel."""
        if not self.all_entities:
            return pn.pane.Markdown("**No data loaded**", width=400)

        total = len(self.all_entities)
        filtered = len(self.filtered_entities)

        # Count by entity type
        type_counts = {}
        for entity in self.all_entities:
            entity_type = entity.get("_entity_type", "unknown")
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

        # Count entities with/without line ranges
        with_ranges = sum(
            1
            for e in self.all_entities
            if e.get("source", {}).get("line_range")
            and e.get("source", {}).get("line_range", {}).get("lines")
        )
        without_ranges = total - with_ranges
        coverage = (with_ranges / total * 100) if total > 0 else 0

        # Build markdown
        md = f"""
## Statistics

**Total Entities:** {total}
**Filtered (visible):** {filtered}

### Coverage
- **With line ranges:** {with_ranges} ({coverage:.1f}%)
- **Missing line ranges:** {without_ranges}

### By Entity Type
"""

        for entity_type in sorted(type_counts.keys()):
            count = type_counts[entity_type]
            color = ENTITY_TYPE_COLORS.get(entity_type, "#888888")
            md += f'<span style="color: {color}; font-weight: bold;">■</span> **{entity_type}:** {count}\n\n'

        return pn.pane.Markdown(md, width=400, sizing_mode="stretch_width")

    def create_dashboard(self):
        """Create and return the dashboard template."""
        # Sidebar
        sidebar_content = [
            pn.pane.Markdown(
                "# Raw Data Explorer\n*Browse raw mathematical entities*",
                styles={"font-size": "1.1em"},
            ),
            pn.layout.Divider(),
            # Data source
            pn.pane.Markdown(
                "## Data Source\n*Select document to explore*",
                styles={"font-size": "0.95em"},
                margin=(10, 0, 10, 0),
            ),
            self.document_selector,
            pn.layout.Divider(),
            # Filters
            pn.pane.Markdown(
                "## Filters\n*Updates apply automatically*",
                styles={"font-size": "0.95em"},
                margin=(10, 0, 10, 0),
            ),
            self.entity_type_filter,
            self.search_input,
            self.sort_selector,
            self.show_missing_ranges,
            pn.layout.Divider(),
            self.reset_button,
        ]

        # Main content: Three-column layout (1:2:3 ratio)
        # Column 1: Entity list (16.67% ~ 1/6)
        entity_list_column = pn.Card(
            pn.panel(self.entity_list_view),
            title="Entity List (click to select)",
            collapsed=False,
            sizing_mode="stretch_height",
            scroll=True,
            styles={"background": "#f8f9fa"},
            width=280,
        )

        # Column 2: JSON representation (33.33% ~ 1/3) - DOUBLED WIDTH
        json_column = pn.Card(
            self.json_panel,
            title="Selected Entity JSON",
            collapsed=False,
            sizing_mode="stretch_height",
            scroll=True,
            styles={"background": "#ffffff"},
            width=560,
        )

        # Column 3: Markdown document with line numbers (50% ~ 1/2) - REDUCED
        markdown_column = pn.Card(
            self.markdown_document_panel,
            title="Markdown Source (with line numbers)",
            collapsed=False,
            sizing_mode="stretch_height",
            styles={"background": "#ffffff"},
        )

        # Main content
        main_content = [
            # Three-column row (1:2:3 ratio - entity list : JSON : markdown)
            pn.Row(
                entity_list_column,
                json_column,
                markdown_column,
                sizing_mode="stretch_both",
                height=900,
            ),
            # Statistics at bottom
            pn.Card(
                pn.panel(self.stats_view),
                title="Statistics",
                collapsed=True,
                sizing_mode="stretch_width",
                styles={"background": "#f8f9fa"},
            ),
        ]

        # Create template
        return pn.template.FastListTemplate(
            title="Raw Data Visualization Dashboard",
            sidebar=sidebar_content,
            main=main_content,
            accent_base_color="#3498db",
            header_background="#2c3e50",
        )


def main():
    """Main entry point - supports both direct execution and panel serve."""
    dashboard = RawDataDashboard()
    return dashboard.create_dashboard()


# Support both execution modes
if __name__ == "__main__":
    # Direct Python execution
    template = main()
    pn.serve(template.servable(), port=5007, show=False)
else:
    # Panel serve: register as servable
    template = main()
    template.servable()
