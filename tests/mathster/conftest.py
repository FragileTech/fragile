"""Shared pytest fixtures for mathster tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def article_fragile_framework_path() -> Path:
    """Path to the fragile framework article: 'docs/source/1_euclidean_gas/10_fragile_gas_framwork.md'."""

    return (
        Path(__file__).parent.parent.parent
        / "docs"
        / "source"
        / "1_euclidean_gas"
        / "10_fragile_gas_framework.md"
    )
