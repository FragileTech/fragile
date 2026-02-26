# Fragile

Fragile is a research codebase for the **Fractal Gas** optimization algorithm and the
**Fractal Set** data structure, with a Jupyter Book that captures the theory, derivations,
and experiments.

## Install

```bash
uv sync --all-extras          # recommended
uv pip install -e .           # alternative
uv tool install hatch         # one-time, for hatch commands
```

## Dashboards

All dashboards launch via `uv run fragile <command>` and serve a Panel app in the browser.

### Videogames — Atari Fractal Gas

Explore Fractal Gas swarm dynamics on Atari environments (Breakout, Pong, etc.).

```bash
uv run fragile videogames                # http://localhost:5006
uv run fragile videogames --port 8080    # custom port
```

### Robots — DM Control Fractal Gas

Visualize Fractal Gas on continuous-control MuJoCo tasks (cartpole, humanoid, etc.).

```bash
uv run fragile robots                    # http://localhost:5007
```

### Physics — QFT Swarm Convergence

Analyse lattice QFT simulations: correlator fits, mass extraction, and convergence plots.

```bash
uv run fragile physics                   # http://localhost:5007
uv run fragile physics --open            # auto-open browser
```

### DL — TopoEncoder Learning

Monitor TopoEncoder training runs: loss curves, topology metrics, atlas visualisations.

```bash
uv run fragile dl                        # http://localhost:5008
uv run fragile dl --outputs runs/        # custom outputs directory
```

## Training

Train the **TopoEncoder** (Attentive Atlas vs Standard VQ-VAE) directly from the CLI.
All arguments after `--` are forwarded to the training script's argparse.

```bash
uv run fragile train -- --help                        # show all training options
uv run fragile train -- --epochs 500 --dataset mnist  # full run
uv run fragile train -- --epochs 1 --dataset mnist    # quick smoke test
```

Key arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `mnist` | Dataset (`mnist`, `fashionmnist`, `cifar10`) |
| `--epochs` | `200` | Training epochs |
| `--batch-size` | `128` | Batch size |
| `--latent-dim` | `64` | Latent dimension |
| `--num-embeddings` | `512` | Codebook size |
| `--use-atlas` | off | Enable Attentive Atlas |
| `--num-charts` | `8` | Number of atlas charts |
| `--output-dir` | `outputs/` | Output directory |

## Documentation (Jupyter Book)

```bash
uv run hatch run docs:build   # build only
uv run hatch run docs:docs    # build and serve
```

## Development

```bash
uv run hatch run test:test     # tests
uv run hatch run test:doctest  # doctests
uv run hatch run lint:all      # lint / format
```

## Project Structure

```
src/fragile/
  fractalai/          # Fractal Gas algorithm
    core/             # companion selection, fitness, cloning, kinetics
    videogames/       # Atari dashboard & environments
    robots/           # DM Control dashboard & environments
  physics/            # lattice QFT simulations & analysis dashboard
  learning/           # TopoEncoder, training script & dashboard
docs/                 # Jupyter Book (source in docs/source/)
tests/                # pytest suites
```

## License

MIT
