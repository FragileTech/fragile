"""QFT-focused dashboard for 3D swarm convergence visualization.

Run:
    python -m fragile.fractalai.experiments.qft_convergence_dashboard
"""

import argparse

import panel as pn

from fragile.fractalai.qft.new_dashboard import create_app


__all__ = ["create_app"]


def _parse_args():
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
