import numpy
import pytest

import fragile


# plangym is optional - only import if available
try:
    import plangym
except ImportError:
    plangym = None


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["fragile"] = fragile
    if plangym is not None:
        doctest_namespace["plangym"] = plangym
