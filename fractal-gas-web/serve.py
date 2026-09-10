#!/usr/bin/env python3
"""Static server for the fractal gas web demo.

Adds the COOP/COEP headers required for SharedArrayBuffer (wasm pthreads)
and the correct MIME type for .wasm/.mjs.

Usage:  python serve.py [port]   then open http://localhost:8000/web/arcade.html
"""

import http.server
import socketserver
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".mjs": "text/javascript",
        ".js": "text/javascript",
    }

    def do_POST(self):
        # Debug upload endpoint for headless visual tests: saves the body
        # under /tmp/fg-uploads/<name>.
        import os
        if not self.path.startswith("/debug-upload/"):
            self.send_error(404)
            return
        name = os.path.basename(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        os.makedirs("/tmp/fg-uploads", exist_ok=True)
        with open(os.path.join("/tmp/fg-uploads", name), "wb") as f:
            f.write(self.rfile.read(length))
        self.send_response(200)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    with socketserver.ThreadingTCPServer(("", port), Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"Serving on http://localhost:{port}/web/arcade.html (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
