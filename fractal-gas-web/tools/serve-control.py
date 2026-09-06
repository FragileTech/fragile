"""Serve the browser laboratory with headers required for parallel WebAssembly."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "web"
    server = ThreadingHTTPServer((args.bind, args.port), partial(Handler, directory=directory))
    print(f"Control laboratory: http://{args.bind}:{args.port}/lab/", flush=True)
    server.serve_forever()
