"""QFT-focused dashboard for 3D swarm convergence visualization.

Run:
    python -m fragile.fractalai.experiments.qft_convergence_dashboard
"""

from fragile.fractalai.qft.dashboard import create_app


__all__ = ["create_app"]


def _parse_args():
    from fragile.fractalai.qft.dashboard import _parse_args as _dashboard_parse_args

    return _dashboard_parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("Starting QFT Swarm Convergence Dashboard...", flush=True)
    app = create_app()
    print(
        f"QFT Swarm Convergence Dashboard running at http://localhost:{args.port} "
        f"(use --open to launch a browser)",
        flush=True,
    )
    app.show(port=args.port, open=args.open)
