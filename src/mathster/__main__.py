"""Entry point for ``python -m mathster``."""

from mathster.cli import cli


def main() -> None:
    """Invoke the Click CLI."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
