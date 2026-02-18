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


if __name__ == "__main__":
    run()
