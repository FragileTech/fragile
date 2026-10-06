"""Isolate Docutils registrations between Sphinx integration builds."""

import pytest
from sphinx.util.docutils import docutils_namespace


@pytest.fixture(autouse=True)
def isolated_docutils_namespace():
    with docutils_namespace():
        yield
