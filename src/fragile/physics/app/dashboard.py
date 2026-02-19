"""QFT dashboard with simulation and analysis tabs."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import holoviews as hv
import panel as pn

from fragile.physics.app.algorithm import build_algorithm_diagnostics_tab
from fragile.physics.app.companion_correlators import build_companion_correlator_tab
from fragile.physics.app.diagnostics import build_coupling_diagnostics_tab
from fragile.physics.app.electroweak_correlators import build_electroweak_correlator_tab
from fragile.physics.app.electroweak_mass_tab import build_electroweak_mass_tab
from fragile.physics.app.gravity import build_holographic_principle_tab
from fragile.physics.app.mass_extraction_tab import build_mass_extraction_tab
from fragile.physics.app.simulation import SimulationTab
from fragile.physics.fractal_gas.history import RunHistory


# Prevent Plotly from probing the system browser during import.
os.environ.setdefault("PLOTLY_RENDERER", "json")
hv.extension("bokeh")

__all__ = ["create_app"]


def _run_tab_computation(state, status_pane, label, compute_fn):
    """Run a correlator tab computation with shared guard/try/except pattern."""
    history = state.get("history")
    if history is None:
        status_pane.object = "**Error:** Load a RunHistory first."
        return
    status_pane.object = f"**Computing {label}...**"
    try:
        compute_fn(history)
    except Exception as e:
        status_pane.object = f"**Error:** {e}"
        import traceback

        traceback.print_exc()


def create_app() -> pn.template.FastListTemplate:
    """Create the QFT convergence + analysis dashboard."""
    pn.extension("plotly", "tabulator")

    debug = os.environ.get("QFT_DASH_DEBUG", "").lower() in {"1", "true", "yes"}
    skip_sidebar = os.environ.get("QFT_DASH_SKIP_SIDEBAR", "").lower() in {"1", "true", "yes"}
    skip_visual = os.environ.get("QFT_DASH_SKIP_VIS", "").lower() in {"1", "true", "yes"}

    def _debug(msg: str):
        if debug:
            print(f"[qft-dashboard] {msg}", flush=True)

    sidebar = pn.Column(
        pn.pane.Markdown("## QFT Dashboard"),
        pn.pane.Markdown("Starting dashboard..."),
        sizing_mode="stretch_width",
    )
    main = pn.Column(
        pn.pane.Markdown("Loading visualization..."),
        sizing_mode="stretch_both",
    )

    # Tab selector lives in the template header bar (always visible).
    tab_selector = pn.widgets.RadioButtonGroup(
        options=["Loading\u2026"],
        value="Loading\u2026",
        disabled=True,
        button_type="default",
        button_style="outline",
        stylesheets=[":host .bk-btn { color: #000 !important; }"],
    )

    template = pn.template.FastListTemplate(
        title="QFT Swarm Convergence Dashboard",
        sidebar=[sidebar],
        main=[main],
        header=[tab_selector],
        sidebar_width=435,
        main_max_width="100%",
        raw_css=[
            # Hide the built-in Bokeh tab bar (navigation lives in the app header)
            ".bk-Tabs > .bk-header { display: none !important; }"
        ],
    )

    def _build_ui():
        start_total = time.time()
        _debug("initializing extensions")

        sim_tab = SimulationTab(bounds_extent=2.0, debug_fn=_debug)

        state: dict[str, Any] = {
            "history": None,
            "history_path": None,
            "fractal_set_points": None,
            "fractal_set_regressions": None,
            "fractal_set_frame_summary": None,
            "_multiscale_geodesic_distance_by_frame": None,
            "_multiscale_geodesic_distribution": None,
            "coupling_diagnostics_output": None,
            "companion_correlator_output": None,
            "electroweak_correlator_output": None,
            "companion_mass_output": None,
            "electroweak_mass_output": None,
        }

        algorithm_section = build_algorithm_diagnostics_tab(state)

        holographic_section = build_holographic_principle_tab(
            state=state,
            run_tab_computation=_run_tab_computation,
        )

        coupling_section = build_coupling_diagnostics_tab(
            state=state,
            run_tab_computation=_run_tab_computation,
        )

        companion_section = build_companion_correlator_tab(
            state=state,
            run_tab_computation=_run_tab_computation,
        )

        ew_correlator_section = build_electroweak_correlator_tab(
            state=state,
            run_tab_computation=_run_tab_computation,
        )

        companion_mass_section = build_mass_extraction_tab(
            state=state,
            run_tab_computation=_run_tab_computation,
            correlator_state_key="companion_correlator_output",
            output_state_key="companion_mass_output",
            tab_label="Mass Extraction",
            button_label="Extract Masses",
            source_label="Companion Correlators",
            computation_label="mass extraction",
        )

        ew_mass_section = build_electroweak_mass_tab(
            state=state,
            run_tab_computation=_run_tab_computation,
        )

        # Wire companion correlator completion to enable mass button.
        _orig_companion_on_run = companion_section.on_run

        def _on_companion_run(event):
            _orig_companion_on_run(event)
            if state["companion_correlator_output"] is not None:
                companion_mass_section.on_correlators_ready()

        companion_section.run_button.on_click(_on_companion_run)

        # Wire electroweak correlator completion to enable ew mass button.
        _orig_ew_on_run = ew_correlator_section.on_run

        def _on_ew_run(event):
            _orig_ew_on_run(event)
            if state["electroweak_correlator_output"] is not None:
                ew_mass_section.on_correlators_ready()

        ew_correlator_section.run_button.on_click(_on_ew_run)

        # Wire SimulationTab history changes to all analysis sections.
        def _on_history_changed(
            history: RunHistory,
            history_path: Any,
            defer: bool,
        ) -> None:
            state["history"] = history
            state["history_path"] = history_path
            state["_multiscale_geodesic_distance_by_frame"] = None
            state["_multiscale_geodesic_distribution"] = None
            algorithm_section.on_history_ready()
            if not defer:
                algorithm_section.reset_plots()
            holographic_section.on_history_changed(defer)
            coupling_section.on_history_changed(defer)
            companion_section.on_history_changed(defer)
            ew_correlator_section.on_history_changed(defer)
            companion_mass_section.on_history_changed(defer)
            ew_mass_section.on_history_changed(defer)

        sim_tab.on_history_changed(_on_history_changed)
        holographic_section.fractal_set_run_button.on_click(holographic_section.on_run_fractal_set)

        # -- Sidebar assembly --
        if skip_sidebar:
            sidebar.objects = [
                pn.pane.Markdown(
                    "## QFT Dashboard\nSidebar disabled via QFT_DASH_SKIP_SIDEBAR=1."
                ),
                pn.pane.Markdown("### Load QFT RunHistory"),
                sim_tab.build_run_history_panel(),
            ]
        else:
            sidebar.objects = [
                pn.pane.Markdown(
                    "## QFT Dashboard\n"
                    "Run a simulation or load a RunHistory, then analyze results."
                ),
                pn.Accordion(
                    ("RunHistory", sim_tab.build_run_history_panel()),
                    ("Simulation", sim_tab.build_simulation_sidebar_panel()),
                    ("Visualization", sim_tab.build_visualization_controls()),
                    sizing_mode="stretch_width",
                ),
            ]

        # -- Main area assembly --
        if skip_visual:
            main.objects = [
                pn.pane.Markdown(
                    "Visualization disabled via QFT_DASH_SKIP_VIS=1.",
                    sizing_mode="stretch_both",
                )
            ]
        else:
            tab_names = [
                "Simulation",
                "Algorithm",
                "Holographic",
                "Coupling",
                "Strong Force",
                "EW Correlators",
                "Mass Extraction",
                "EW Mass",
            ]
            tabs_widget = pn.Tabs(
                ("Simulation", sim_tab.build_tab()),
                ("Algorithm", algorithm_section.tab),
                ("Holographic Principle", holographic_section.fractal_set_tab),
                ("Coupling Diagnostics", coupling_section.coupling_diagnostics_tab),
                ("Companion Correlators", companion_section.tab),
                ("Electroweak Correlators", ew_correlator_section.tab),
                ("Mass Extraction", companion_mass_section.tab),
                ("Electroweak Mass", ew_mass_section.tab),
                dynamic=True,
            )

            # Wire header tab selector <-> Tabs widget (bidirectional)
            tab_selector.options = tab_names
            tab_selector.value = tab_names[0]
            tab_selector.disabled = False

            def _on_tab_selector(event):
                idx = tab_names.index(event.new)
                if tabs_widget.active != idx:
                    tabs_widget.active = idx

            def _on_tabs_active(event):
                if tab_selector.value != tab_names[event.new]:
                    tab_selector.value = tab_names[event.new]

            tab_selector.param.watch(_on_tab_selector, "value")
            tabs_widget.param.watch(_on_tabs_active, "active")

            main.objects = [tabs_widget]

        _debug(f"ui ready ({time.time() - start_total:.2f}s)")

    pn.state.onload(_build_ui)
    return template


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5007)
    parser.add_argument("--open", action="store_true", help="Open browser on launch")
    parser.add_argument("--address", type=str, default="0.0.0.0", help="Bind address")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("Starting QFT Swarm Convergence Dashboard...", flush=True)
    print(
        f"QFT Swarm Convergence Dashboard running at http://{args.address}:{args.port} "
        f"(use --open to launch a browser)",
        flush=True,
    )
    pn.serve(
        create_app,
        port=args.port,
        address=args.address,
        show=args.open,
        title="QFT Swarm Convergence Dashboard",
        websocket_origin="*",
    )
