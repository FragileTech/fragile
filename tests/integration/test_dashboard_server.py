"""Launch each dashboard in a clean process and verify it serves HTTP."""

import multiprocessing
import socket
import time

import pytest
import requests


def run_dashboard_server(mode, port):
    """Initialize Panel independently of the parent process's thread pools."""
    import holoviews as hv
    import panel as pn

    from fragile.fractalai.experiments.gas_visualization_dashboard import (
        create_app,
        create_qft_app,
    )

    hv.extension("bokeh")
    pn.extension()
    app = create_qft_app() if mode == "qft" else create_app()
    app.show(port=port, open=False, threaded=False)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mode", ["standard", "qft"])
def test_dashboard_server_starts(mode):
    """Wait for readiness instead of assuming dashboard initialization takes five seconds."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = multiprocessing.get_context("spawn").Process(
        target=run_dashboard_server, args=(mode, port)
    )
    process.start()
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            assert process.is_alive(), f"Dashboard exited with code {process.exitcode}"
            try:
                response = requests.get(f"http://127.0.0.1:{port}", timeout=(2, 45))
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(0.2)
                continue
            assert response.status_code == 200
            return
        pytest.fail(f"{mode} dashboard did not become ready within 90 seconds")
    finally:
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
