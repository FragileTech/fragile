# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

```{contents}
```

## Bug reports

When [reporting a bug](https://github.com/FragileTech/fragile/issues) please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

## Documentation improvements

Fragile could always use more documentation, whether as part of the official Fragile docs, in docstrings, or even on the web in blog posts, articles, and such.

## Feature requests and feedback

The best way to send feedback is to file an issue at [https://github.com/FragileTech/fragile/issues](https://github.com/FragileTech/fragile/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that code contributions are welcome :)

## Development

To set up `fragile` for local development:

1. Fork [fragile](https://github.com/FragileTech/fragile) (look for the "Fork" button).
2. Clone your fork locally:

    ```bash
    git clone git@github.com:YOURGITHUBNAME/fragile.git
    ```

3. Create a branch for local development:

    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```

   Now you can make your changes locally.

4. Run the Python checks and documentation build:

    ```bash
    uv run ruff check .
    uv run ruff format --check .
    make test
    make docs
    ```

5. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

6. Submit a pull request through the GitHub website.

## Reproducing Python CI

CI uses Python 3.10 and the versions in `uv.lock`, with CPU PyTorch. In a fresh
checkout, install and run that environment with:

```bash
uv export --locked --extra test --no-hashes --no-emit-project --output-file /tmp/fragile-constraints.txt
uv venv --python 3.10
uv pip install --torch-backend=cpu --constraint /tmp/fragile-constraints.txt -e '.[test]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync pytest -q -n 2 --dist loadfile tests
```

Linux rendering tests require EGL and Mesa (`libegl1` and `libgl1-mesa-dri` on
Ubuntu). The test configuration defaults MuJoCo to headless EGL. Dashboard tests
start temporary local HTTP servers. Training integration tests use synthetic data.

For Lab changes, run `make control-test`, `make control-web`, and
`npm --prefix fractal-gas-web run test:lab`. Browser suites run through
`bash fractal-gas-web/tools/test-control-browser.sh node fractal-gas-web/tests/control-browser.mjs`;
the control workflow lists the additional browser suites. Set
`CONTROL_CAPTURE_SCREENSHOTS=1` to request diagnostic captures in the main,
racing, and mining suites. CI checks their layout and interactions directly.

## Pull Request Guidelines

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (run `make test`).
2. Update documentation when there's new API, functionality etc.
3. Add a note to `CHANGELOG.md` about the changes.
4. Add yourself to `AUTHORS.md`.
