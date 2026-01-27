"""Integration tests that actually launch the dashboard server.

These tests verify the dashboard can be served over HTTP without errors.
"""

import multiprocessing
import time

import pytest
import requests


def run_dashboard_server(mode="standard", port=5008):
    """Run dashboard in subprocess."""
    import sys

    sys.argv = ["test"]

    if mode == "qft":
        sys.argv.append("--qft")

    import holoviews as hv
    import panel as pn

    from fragile.fractalai.experiments.gas_visualization_dashboard import (
        create_app,
        create_qft_app,
    )

    hv.extension("bokeh")
    pn.extension()

    if mode == "qft":
        app = create_qft_app()
    else:
        app = create_app()

    # Show without opening browser
    app.show(port=port, open=False, threaded=False)


class TestDashboardServer:
    """Integration tests for dashboard server."""

    @pytest.mark.timeout(30)
    def test_standard_dashboard_server_starts(self):
        """Test standard dashboard server can start and serve content."""
        port = 5008
        process = multiprocessing.Process(target=run_dashboard_server, args=("standard", port))

        try:
            process.start()
            time.sleep(5)  # Wait for server to start

            # Try to connect
            response = requests.get(f"http://localhost:{port}", timeout=5)
            assert response.status_code == 200

        finally:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

    @pytest.mark.timeout(30)
    def test_qft_dashboard_server_starts(self):
        """Test QFT dashboard server can start and serve content."""
        port = 5009
        process = multiprocessing.Process(target=run_dashboard_server, args=("qft", port))

        try:
            process.start()
            time.sleep(5)

            response = requests.get(f"http://localhost:{port}", timeout=5)
            assert response.status_code == 200

        finally:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
