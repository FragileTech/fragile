"""Pytest fixtures for mathster tests."""
import pytest

@pytest.fixture
def sample_extraction():
    """Sample chapter extraction."""
    return {
        "definitions": [{"parameters_mentioned": ["alpha", "beta"]}],
        "theorems": [{"parameters_mentioned": ["tau"]}],
    }
