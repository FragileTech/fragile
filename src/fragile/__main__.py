"""CLI entry point for fragile: ``uv run fragile <command>``."""

import click


@click.group()
def run():
    """Fragile – FractalAI toolkit."""


@run.command()
@click.option("--port", default=5006, type=int, help="Port to run server on.")
@click.option("--open", "open_browser", is_flag=True, help="Open browser on launch.")
@click.option("--threaded", is_flag=True, help="Use multi-threaded Tornado.")
@click.option("--headless", is_flag=True, help="Force headless env vars (auto-detected on WSL).")
def videogames(port, open_browser, threaded, headless):
    """Launch the Atari Fractal Gas dashboard."""
    import panel as pn

    from fragile.fractalai.videogames.dashboard import (
        _configure_headless_wsl,
        create_app,
    )

    _configure_headless_wsl(force=headless)
    click.echo("Starting Atari Fractal Gas Dashboard...")
    app = create_app()
    if not threaded:
        click.echo("Running in single-threaded mode (use --threaded for multi-threaded)")
    click.echo(f"Open http://localhost:{port} in your browser (Ctrl+C to stop)")
    pn.serve({"/": app}, port=port, show=open_browser, threaded=threaded)


@run.command()
@click.option("--port", default=5007, type=int, help="Port to run server on.")
@click.option("--open", "open_browser", is_flag=True, help="Open browser on launch.")
@click.option("--threaded", is_flag=True, help="Use multi-threaded Tornado.")
def robots(port, open_browser, threaded):
    """Launch the DM Control Fractal Gas dashboard."""
    import panel as pn

    from fragile.fractalai.robots.dashboard import (
        _configure_mujoco_offscreen,
        create_app,
    )

    _configure_mujoco_offscreen()
    click.echo("Starting DM Control Fractal Gas Dashboard...")
    app = create_app()
    if not threaded:
        click.echo("Running in single-threaded mode (use --threaded for multi-threaded)")
    click.echo(f"Open http://localhost:{port} in your browser (Ctrl+C to stop)")
    pn.serve({"/": app}, port=port, show=open_browser, threaded=threaded)


@run.command()
@click.option("--port", default=5007, type=int, help="Port to run server on.")
@click.option("--open", "open_browser", is_flag=True, help="Open browser on launch.")
@click.option("--address", default="0.0.0.0", help="Bind address.")
def physics(port, open_browser, address):
    """Launch the QFT Swarm Convergence dashboard."""
    import panel as pn

    from fragile.physics.app.dashboard import create_app

    click.echo("Starting QFT Swarm Convergence Dashboard...")
    click.echo(
        f"QFT Swarm Convergence Dashboard running at http://{address}:{port} "
        f"(use --open to launch a browser)"
    )
    pn.serve(
        create_app,
        port=port,
        address=address,
        show=open_browser,
        title="QFT Swarm Convergence Dashboard",
        websocket_origin="*",
    )


@run.command()
@click.option("--port", default=5008, type=int, help="Port to run server on.")
@click.option("--open", "open_browser", is_flag=True, help="Open browser on launch.")
@click.option("--outputs", default="outputs", help="Directory containing topoencoder runs.")
def dl(port, open_browser, outputs):
    """Launch the TopoEncoder learning dashboard."""
    import panel as pn

    from fragile.learning.dashboard import create_app

    click.echo("Starting TopoEncoder Dashboard...")
    click.echo(f"Open http://localhost:{port} in your browser (Ctrl+C to stop)")
    pn.serve(
        lambda: create_app(outputs_dir=outputs),
        port=port,
        show=open_browser,
        title="TopoEncoder Dashboard",
        websocket_origin="*",
    )


@run.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def train(args):
    """Train the TopoEncoder (Attentive Atlas vs Standard VQ-VAE).

    All arguments after 'train' are forwarded to the training script.
    Run `uv run fragile train -- --help` for all training options.
    """
    import sys

    from fragile.learning.topoencoder_mnist import main

    original_argv = sys.argv
    sys.argv = ["fragile-train", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@run.command("vla-dashboard")
@click.option("--port", default=5009, type=int, help="Port to run server on.")
@click.option("--open", "open_browser", is_flag=True, help="Open browser on launch.")
@click.option("--outputs", default="outputs/vla", help="VLA outputs directory.")
def vla_dashboard(port, open_browser, outputs):
    """Launch the VLA World Model dashboard."""
    import panel as pn

    from fragile.learning.vla.dashboard import create_app

    click.echo("Starting VLA World Model Dashboard...")
    click.echo(f"Open http://localhost:{port} in your browser (Ctrl+C to stop)")
    pn.serve(
        lambda: create_app(outputs_dir=outputs),
        port=port,
        show=open_browser,
        title="VLA World Model Dashboard",
        websocket_origin="*",
    )


@run.command(
    "vla-train",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vla_train(args):
    """Train the TopoEncoder VLA experiment (3-phase pipeline).

    All arguments after 'vla-train' are forwarded to the training script.
    Run `uv run fragile vla-train -- --help` for all config options.
    """
    import sys

    from fragile.learning.vla.train import main

    original_argv = sys.argv
    sys.argv = ["fragile-vla-train", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@run.command(
    "vla-unsup",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vla_unsup(args):
    """Unsupervised TopoEncoder training on VLA features.

    All arguments after 'vla-unsup' are forwarded to the training script.
    Run `uv run fragile vla-unsup -- --help` for all training options.
    """
    import sys

    from fragile.learning.vla.train_unsupervised import main

    original_argv = sys.argv
    sys.argv = ["fragile-vla-unsup", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@run.command(
    "dataset",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def dataset(args):
    """Extract SmolVLA latents and cache them for world model training.

    Runs the frozen SmolVLA backbone over every frame in a LeRobot dataset
    and saves per-episode features, actions, and states as .pt files.

    Run `uv run fragile dataset -- --help` for all options.
    """
    import sys

    from fragile.learning.vla.create_latent_dataset import main

    original_argv = sys.argv
    sys.argv = ["fragile-dataset", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@run.command(
    "vla-joint",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vla_joint(args):
    """Joint encoder + world model training (3-phase pipeline).

    All arguments after 'vla-joint' are forwarded to the training script.
    Run `uv run fragile vla-joint -- --help` for all training options.
    """
    import sys

    from fragile.learning.vla.train_joint import main

    original_argv = sys.argv
    sys.argv = ["fragile-vla-joint", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    run()
