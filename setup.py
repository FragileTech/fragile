from setuptools import setup


setup(
    name="fragile",
    description="Fractal AI utilities and algorithms.",
    version="0.0.1a",
    license="Propietary",
    author="Guillem Duran Ballester",
    author_email="guillem.db@gmail.com",
    url="https://github.com/Guillemdb/fragile",
    download_url="https://github.com/Guillemdb/fragile",
    keywords=["reinforcement learning", "artificial intelligence", "monte carlo", "planning"],
    install_requires=["torch", "torchvision", "numpy", "Pillow-simd", "plangym", "hypothesis"],
    package_data={"": ["README.md"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT license",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
    ],
)