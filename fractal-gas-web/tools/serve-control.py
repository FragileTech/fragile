"""Preview the lab and documentation using the published site's URL layout."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


def site_path(path):
    """Resolve the optional Pages project prefix before selecting a mount."""
    path = unquote(urlsplit(path).path)
    if path == "/fragile" or path.startswith("/fragile/"):
        path = path[len("/fragile") :] or "/"
    return path


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, docs_directory=None, docs_home=False, **kwargs):
        self.docs_directory = docs_directory
        self.docs_home = docs_home
        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        # Support both a local origin and the GitHub Pages project prefix.
        path = site_path(path)
        directory = self.directory
        if self.docs_directory and (path == "/docs" or path.startswith("/docs/")):
            self.directory = str(self.docs_directory)
            path = path[len("/docs") :] or "/"
        try:
            # Let the standard handler normalize traversal segments within the
            # selected mount. Re-quote so it decodes the URL exactly once.
            return super().translate_path(quote(path, safe="/"))
        finally:
            self.directory = directory

    def send_head(self):
        if self.docs_home and urlsplit(self.path).path in {"/", "/fragile/"}:
            self.send_response(302)
            self.send_header("Location", "docs/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        return super().send_head()

    def end_headers(self):
        path = site_path(self.path)
        # The lab needs shared memory. Docs retain ordinary browser loading
        # rules for external MathJax, Mermaid, fonts, and reference previews.
        if not (path == "/docs" or path.startswith("/docs/")):
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--docs", action="store_true", help="Open the documentation portal at /")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    directory = repo / "fractal-gas-web/web"
    docs_directory = repo / "docs/_build/html"
    if args.docs and not (docs_directory / "index.html").is_file():
        parser.error("Documentation has not been built. Run 'make docs' or 'make serve' first.")
    server = ThreadingHTTPServer(
        (args.bind, args.port),
        partial(
            Handler,
            directory=directory,
            docs_directory=docs_directory,
            docs_home=args.docs,
        ),
    )
    print(f"Optimization laboratory: http://{args.bind}:{args.port}/optimization/", flush=True)
    print(f"LLM laboratory: http://{args.bind}:{args.port}/llm/", flush=True)
    print(f"Control laboratory: http://{args.bind}:{args.port}/lab/", flush=True)
    if (docs_directory / "index.html").is_file():
        print(f"Documentation: http://{args.bind}:{args.port}/docs/", flush=True)
    else:
        print("Run 'make docs' to enable local documentation at /docs/.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
