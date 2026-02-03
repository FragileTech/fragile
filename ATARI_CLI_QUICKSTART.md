# Atari Fractal Gas - Command Line Quick Start

## Overview

If the dashboard has XCB/GUI issues on WSL, you can run simulations directly from the command line without any GUI.

## Installation

```bash
# Install Atari ROMs and dependencies
pip install gymnasium[atari] gymnasium[accept-rom-license]

# OR use plangym
pip install plangym

# On WSL, install xvfb for headless rendering
sudo apt-get install xvfb mesa-utils libgl1-mesa-glx libgl1-mesa-dri
```

## Quick Start

### On WSL (Recommended)

Use the WSL wrapper script that handles all xvfb configuration:

```bash
# Run Pong with 10 walkers for 100 iterations
bash scripts/run_atari_gas_wsl.sh --game Pong --N 10 --iterations 100

# Run Breakout with more walkers
bash scripts/run_atari_gas_wsl.sh --game Breakout --N 20 --iterations 500

# Save results to disk
bash scripts/run_atari_gas_wsl.sh --game Pong --N 10 --iterations 100 \
    --output-dir results/pong

# Record frames for later visualization
bash scripts/run_atari_gas_wsl.sh --game Pong --N 10 --iterations 100 \
    --record-frames --output-dir results/pong
```

### On Linux/Mac (Direct)

If you have a display or proper X11 setup:

```bash
# Run directly
python scripts/run_atari_gas_cli.py --game Pong --N 10 --iterations 100

# Or with xvfb if headless
xvfb-run -a python scripts/run_atari_gas_cli.py --game Pong --N 10 --iterations 100
```

## Command Line Options

### Game Configuration

- `--game NAME` - Atari game name (default: Pong)
  - Examples: Pong, Breakout, SpaceInvaders, Asteroids, etc.
- `--obs-type TYPE` - Observation type: `rgb` or `ram` (default: rgb)

### Algorithm Parameters

- `--N NUM` - Number of walkers (default: 10)
- `--iterations NUM` - Maximum iterations (default: 100)
- `--dist-coef FLOAT` - Distance coefficient (default: 1.0)
- `--reward-coef FLOAT` - Reward coefficient (default: 1.0)
- `--no-cumulative-reward` - Disable cumulative reward
- `--dt-min FLOAT` - Minimum time step (default: 0.5)
- `--dt-max FLOAT` - Maximum time step (default: 5.0)

### System Configuration

- `--device DEVICE` - PyTorch device: `cpu` or `cuda` (default: cpu)
- `--seed NUM` - Random seed for reproducibility

### Output Configuration

- `--record-frames` - Record frames during simulation
- `--output-dir PATH` - Directory to save results

## Example Usage

### Basic Simulation

```bash
bash scripts/run_atari_gas_wsl.sh --game Pong --N 10 --iterations 100
```

Output:
```
======================================================================
ATARI FRACTAL GAS - COMMAND LINE
======================================================================

Game: Pong
Walkers (N): 10
Max iterations: 100
...

Running simulation...
----------------------------------------------------------------------
Iteration    1/100 | Episodes:   0 | Best reward:    -inf | Avg reward:     0.0 | Speed: 2.1 it/s
Iteration   10/100 | Episodes:   1 | Best reward:    -21.0 | Avg reward:   -21.0 | Speed: 3.5 it/s
...
Iteration  100/100 | Episodes:   5 | Best reward:    -15.0 | Avg reward:   -18.2 | Speed: 4.2 it/s
----------------------------------------------------------------------

Simulation complete!
Time elapsed: 23.8s
Iterations: 100/100
Episodes completed: 5
Best reward: -15.0
Average reward: -18.2
Std reward: 2.4
======================================================================
```

### Save Results to Disk

```bash
bash scripts/run_atari_gas_wsl.sh --game Pong --N 10 --iterations 100 \
    --output-dir results/pong
```

Creates:
- `results/pong/Pong_20260203_153045_results.json` - Simulation metrics
- `results/pong/Pong_20260203_153045_frames.npy` (if `--record-frames` enabled)

### Custom Parameters

```bash
# More walkers, longer simulation
bash scripts/run_atari_gas_wsl.sh --game Breakout --N 20 --iterations 500 \
    --dist-coef 1.5 --reward-coef 2.0 --seed 42

# Use RAM observations (faster, no frames)
bash scripts/run_atari_gas_wsl.sh --game Pong --obs-type ram --N 10 --iterations 100

# Record frames for visualization
bash scripts/run_atari_gas_wsl.sh --game Pong --N 10 --iterations 100 \
    --record-frames --output-dir results/pong
```

## Analyzing Results

Results are saved as JSON files with the following structure:

```json
{
  "game": "Pong",
  "N": 10,
  "max_iterations": 100,
  "iterations_completed": 100,
  "best_reward": -15.0,
  "total_reward": -91.0,
  "episode_rewards": [-21.0, -19.0, -18.0, -18.0, -15.0],
  "episode_lengths": [23, 45, 67, 89, 100],
  "timestamps": [0.1, 0.2, 0.3, ...]
}
```

### Load and Analyze in Python

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open("results/pong/Pong_20260203_153045_results.json") as f:
    results = json.load(f)

# Plot episode rewards
plt.figure(figsize=(10, 6))
plt.plot(results["episode_rewards"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(f"{results['game']} - Episode Rewards (N={results['N']})")
plt.grid(True)
plt.savefig("episode_rewards.png")

# Load frames if recorded
if Path("results/pong/Pong_20260203_153045_frames.npy").exists():
    frames = np.load("results/pong/Pong_20260203_153045_frames.npy")
    print(f"Loaded {len(frames)} frames, shape: {frames[0].shape}")
```

## Advantages over Dashboard

✅ **No GUI/XCB issues** - Runs without Panel/Bokeh/X11
✅ **Lightweight** - Lower memory footprint
✅ **Scriptable** - Easy to automate and batch process
✅ **Remote-friendly** - Works over SSH without X11 forwarding
✅ **Batch processing** - Run multiple simulations in parallel

## Available Games

Common Atari games (see [Gymnasium Atari docs](https://gymnasium.farama.org/environments/atari/) for full list):

- **Pong** - Classic paddle game
- **Breakout** - Brick breaking
- **SpaceInvaders** - Shoot aliens
- **Asteroids** - Avoid asteroids
- **MsPacman** - Navigate maze
- **Qbert** - Jump on cubes
- **Seaquest** - Underwater rescue
- **BeamRider** - Shoot enemies
- **Enduro** - Racing game
- And many more...

## Troubleshooting

### "Environment not found"

```bash
# Install Atari ROMs
pip install gymnasium[atari] gymnasium[accept-rom-license]

# Or use plangym
pip install plangym
```

### "xvfb-run not found"

```bash
sudo apt-get install xvfb
```

### "OpenGL/Display not available"

```bash
# Make sure you're using the WSL wrapper script
bash scripts/run_atari_gas_wsl.sh --game Pong --N 10

# Or manually with xvfb
xvfb-run -a python scripts/run_atari_gas_cli.py --game Pong --N 10
```

### Still getting XCB errors?

Try RAM observations (doesn't use OpenGL):

```bash
bash scripts/run_atari_gas_wsl.sh --game Pong --obs-type ram --N 10 --iterations 100
```

### Slow performance?

- Reduce number of walkers: `--N 5`
- Use RAM observations: `--obs-type ram`
- Disable frame recording (it's off by default)
- Use GPU if available: `--device cuda`

## Batch Processing Example

Run multiple games in sequence:

```bash
#!/bin/bash
# run_all_games.sh

GAMES=("Pong" "Breakout" "SpaceInvaders" "Asteroids")
OUTPUT_BASE="results/batch_$(date +%Y%m%d_%H%M%S)"

for game in "${GAMES[@]}"; do
    echo "Running $game..."
    bash scripts/run_atari_gas_wsl.sh \
        --game "$game" \
        --N 10 \
        --iterations 200 \
        --output-dir "$OUTPUT_BASE/$game" \
        --seed 42
    echo "---"
done

echo "All simulations complete!"
echo "Results saved to: $OUTPUT_BASE"
```

## Performance Tips

1. **Start small**: Begin with `--N 10 --iterations 100` to test
2. **RAM observations**: Use `--obs-type ram` for faster simulations
3. **GPU acceleration**: Use `--device cuda` if you have NVIDIA GPU
4. **Batch mode**: Run multiple simulations in parallel using `&`
5. **Progress tracking**: Output is designed for logging, redirect to file:
   ```bash
   bash scripts/run_atari_gas_wsl.sh --game Pong --N 10 --iterations 100 2>&1 | tee pong.log
   ```

## Comparison: CLI vs Dashboard

| Feature | CLI | Dashboard |
|---------|-----|-----------|
| GUI | ❌ No | ✅ Yes |
| XCB issues | ✅ Avoided | ⚠️ Possible on WSL |
| Memory usage | ✅ Lower | ⚠️ Higher |
| Real-time viz | ❌ No | ✅ Yes |
| Batch processing | ✅ Easy | ⚠️ Harder |
| Remote SSH | ✅ Works well | ⚠️ Needs X11 forwarding |
| Progress updates | ✅ Terminal | ✅ Web UI |
| Result saving | ✅ JSON/NPY | ⚠️ Manual |

## Next Steps

1. **Run your first simulation:**
   ```bash
   bash scripts/run_atari_gas_wsl.sh --game Pong --N 10 --iterations 100
   ```

2. **Save results for analysis:**
   ```bash
   bash scripts/run_atari_gas_wsl.sh --game Pong --N 10 --iterations 100 \
       --output-dir results/pong --record-frames
   ```

3. **Experiment with parameters:**
   ```bash
   bash scripts/run_atari_gas_wsl.sh --game Breakout --N 20 --iterations 500 \
       --dist-coef 1.5 --reward-coef 2.0
   ```

4. **Analyze results in Python/Jupyter**

## Help

For all options:
```bash
python scripts/run_atari_gas_cli.py --help
```

For issues, see:
- `README_WSL.md` - WSL setup guide
- `XCB_THREADING_FIX.md` - Threading details
- GitHub issues: https://github.com/anthropics/claude-code/issues
