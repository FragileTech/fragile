from importlib.machinery import SourceFileLoader
from pathlib import Path

from setuptools import find_packages, setup


version = SourceFileLoader(
    "fragile.version", str(Path(__file__).parent / "fragile" / "version.py"),
).load_module()

# Module-specific dependencies.
extras = {
    "atari": ["atari-py==0.1.1", "Pillow", "opencv-python", "gym"],
    "dataviz": [
        "matplotlib",
        "bokeh<2.0.0",
        "pandas",
        "panel",
        "holoviews",
        "hvplot",
        "plotly",
        "streamz",
        "param",
    ],
    "test": ["pytest>=5.3.5", "hypothesis==5.6.0"],
    # TODO: Update when ray support is ready "ray": ["ray", "setproctitle"],
}

# Meta dependency groups.
extras["all"] = [item for group in extras.values() for item in group]

setup(
    name="fragile",
    description="Framework for developing FractalAI based algorithms.",
    packages=find_packages(),
    version=version.__version__,
    license="AGPLv3.0",
    author="Guillem Duran Ballester",
    author_email="guillem.db@gmail.com",
    url="https://github.com/Guillemdb/fragile",
    download_url="https://github.com/Guillemdb/fragile",
    keywords=["reinforcement learning", "artificial intelligence", "monte carlo", "planning"],
    tests_require=["pytest", "hypothesis"],
    extras_require=extras,
    install_requires=[
        "attrs",
        "numpy",
        "scipy",
        "networkx",
        "pillow-simd",
        "psycopg2-binary",
        "pyarrow",
        "scikit-learn",
        "numba",
    ],
    package_data={"": ["README.md"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
    ],
)
