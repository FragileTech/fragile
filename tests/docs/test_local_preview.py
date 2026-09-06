"""Exercise the local server's real HTTP routes and browser loading headers."""

from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import importlib.util
from pathlib import Path
from threading import Thread

import pytest


TOOLS = Path(__file__).resolve().parents[2] / "fractal-gas-web/tools"
SPEC = importlib.util.spec_from_file_location("serve_control", TOOLS / "serve-control.py")
preview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview)


@pytest.fixture
def local_site(tmp_path):
    web = tmp_path / "web"
    docs = tmp_path / "docs"
    for root, name, contents in [
        (web, "lab/index.html", "Simulator"),
        (web, "lab/engine/control.wasm", "wasm"),
        (docs, "index.html", "Portal"),
        (docs, "theory/index.html", "Lectures"),
        (docs, "lab/index.html", "Guide"),
        (docs, "theory/_static/book.css", "body {}"),
        (docs, "theory/space name.html", "Encoded name"),
    ]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    (tmp_path / "private.txt").write_text("Outside the mounts")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(preview.Handler, directory=web, docs_directory=docs, docs_home=True),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, method="GET"):
        connection = HTTPConnection(*server.server_address, timeout=5)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            return (
                response.status,
                {key.lower(): value for key, value in response.getheaders()},
                response.read(),
            )
        finally:
            connection.close()

    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.mark.parametrize("prefix", ["", "/fragile"])
def test_portal_guide_and_simulator_use_distinct_routes(local_site, prefix):
    for path, content in [
        ("/docs/", b"Portal"),
        ("/docs/theory/", b"Lectures"),
        ("/docs/lab/", b"Guide"),
        ("/lab/", b"Simulator"),
        ("/docs/theory/space%20name.html?query=1", b"Encoded name"),
    ]:
        status, _, body = local_site(prefix + path)
        assert status == 200
        assert body == content
    status, headers, _ = local_site(prefix + "/")
    assert status == 302
    assert headers["location"] == "docs/"
    status, headers, _ = local_site(prefix + "/docs")
    assert status == 301
    assert headers["location"] == prefix + "/docs/"


def test_docs_allow_external_renderers_and_lab_retains_shared_memory(local_site):
    status, headers, body = local_site("/docs/theory/_static/book.css", "HEAD")
    assert status == 200
    assert body == b""
    assert headers["content-type"] == "text/css"
    assert "cross-origin-embedder-policy" not in headers
    for path in ["/lab/", "/fragile/lab/engine/control.wasm"]:
        status, headers, _ = local_site(path)
        assert status == 200
        assert headers["cross-origin-embedder-policy"] == "require-corp"
        assert headers["cross-origin-opener-policy"] == "same-origin"
    assert headers["content-type"] == "application/wasm"


@pytest.mark.parametrize(
    "path", ["/docs/../private.txt", "/docs/%2e%2e/private.txt", "/lab/../../private.txt"]
)
def test_requests_cannot_escape_the_selected_mount(local_site, path):
    assert local_site(path)[0] == 404
