# Atari Fractal Gas Dashboard - Quick Start Guide

## Installation

```bash
# Install dashboard dependencies
pip install panel holoviews bokeh pillow

# Install Atari environment support
pip install plangym
```

### WSL / Headless Systems

The dashboard UI will load on WSL, but **running actual Atari simulations requires a display** (OpenGL/pyglet). On WSL, you have two options:

**Option 1: Use pre-recorded data (Recommended for WSL)**
- Load existing simulation results saved from a system with display
- The dashboard can visualize any saved `AtariHistory` without needing Atari environments

**Option 2: Set up virtual display**
```bash
# Install xvfb
sudo apt-get install xvfb

# Run with virtual display
xvfb-run python src/fragile/fractalai/videogames/dashboard.py

# Or set DISPLAY environment variable
export DISPLAY=:0
python src/fragile/fractalai/videogames/dashboard.py
```

**Option 3: Run simulations on native Linux/macOS**
- Run the algorithm on a machine with display
- Save the history to a file
- Load it in the dashboard on WSL for visualization

## 1. Basic Algorithm Use (No Dashboard)

```python
from plangym import AtariEnvironment
from fragile.fractalai.videogames import AtariFractalGas, AtariHistory

# Create environment
env = AtariEnvironment(name="PongNoFrameskip-v4", obs_type="ram")

# Create algorithm with new features
gas = AtariFractalGas(
    env=env,
    N=30,
    use_cumulative_reward=True,  # NEW: Use cumulative rewards for fitness
    record_frames=True,           # NEW: Record best walker frames
    device="cpu",
    seed=42,
)

# Run simulation
state = gas.reset()
infos = []

for i in range(100):
    state, info = gas.step(state)
    infos.append(info)

    if i % 10 == 0:
        print(f"Iteration {i}: Max reward = {info['max_reward']:.1f}")

# Build history
history = AtariHistory.from_run(infos, state, N=30, game_name="Pong")

print(f"\nMax reward achieved: {max(history.rewards_max):.1f}")
print(f"Frames recorded: {history.has_frames}")

env.close()
```

## 2. Launch Interactive Dashboard

### Method A: Python Script

```python
from fragile.fractalai.videogames import get_dashboard_components

_, _, create_app = get_dashboard_components()

app = create_app()
app.show(port=5006)

# Navigate to http://localhost:5006
```

### Method B: Command Line

```bash
python examples/atari_dashboard_example.py
```

### Method C: Direct Module

```bash
python -m fragile.fractalai.videogames.dashboard
```

## 3. Dashboard Workflow

1. **Select Game**: Choose from Pong, Breakout, MsPacman, or SpaceInvaders
2. **Configure Parameters**:
   - N: Number of walkers (5-200)
   - dist_coef: Distance coefficient (0.0-5.0)
   - reward_coef: Reward coefficient (0.0-5.0)
   - use_cumulative_reward: Toggle cumulative vs step rewards
   - max_iterations: How many iterations to run (10-1000)
   - record_frames: Enable frame recording (required for visualization)
3. **Run Simulation**: Click "Run Simulation" button
4. **Monitor Progress**: Watch progress bar and status updates
5. **Visualize Results**:
   - Use time slider to navigate through iterations
   - View best walker's game frame
   - Analyze reward progression curves
   - Examine metric histograms

## 4. Understanding the New Features

### Cumulative vs Step Rewards

**Step Rewards** (default, `use_cumulative_reward=False`):
- Fitness based on immediate reward from last action
- Encourages exploration and diversity
- Original algorithm behavior

**Cumulative Rewards** (`use_cumulative_reward=True`):
- Fitness based on total accumulated reward
- Stronger selection pressure toward high-reward states
- May reduce population diversity
- Better for convergence to high-reward regions

**When to use each:**
- Exploration-heavy tasks → Step rewards
- Convergence-focused tasks → Cumulative rewards
- Experiment with both to see what works best!

### Frame Recording

When `record_frames=True`:
- Stores best walker's RGB frame each iteration
- ~100KB per frame (210×160×3 uint8 array)
- Available in `info['best_frame']` after each step
- Required for dashboard visualization

**Memory consideration:**
- 100 iterations ≈ 10MB
- 1000 iterations ≈ 100MB
- Only best walker frame stored (not all N walkers)

## 5. Accessing Recorded Data

```python
# After running simulation
history = AtariHistory.from_run(infos, final_state, N=30, game_name="Pong")

# Access metrics
print(f"Rewards over time: {history.rewards_max}")
print(f"Alive counts: {history.alive_counts}")
print(f"Cloning events: {history.num_cloned}")

# Access frames (if recorded)
if history.has_frames:
    first_frame = history.best_frames[0]  # RGB array [210, 160, 3]
    last_frame = history.best_frames[-1]
```

## 6. Programmatic Visualization

```python
from fragile.fractalai.videogames import get_dashboard_components

_, AtariGasVisualizer, _ = get_dashboard_components()

# Create visualizer with existing history
visualizer = AtariGasVisualizer(history=history)

# Display panel (in Jupyter or standalone)
visualizer.panel()
```

## 7. Common Use Cases

### Compare Cumulative vs Step Rewards

```python
# Run with step rewards
gas_step = AtariFractalGas(env, N=30, use_cumulative_reward=False)
state, infos_step = gas_step.run(max_iterations=100)

# Run with cumulative rewards
gas_cumulative = AtariFractalGas(env, N=30, use_cumulative_reward=True)
state, infos_cumulative = gas_cumulative.run(max_iterations=100)

# Compare results
max_reward_step = max(info['max_reward'] for info in infos_step)
max_reward_cumulative = max(info['max_reward'] for info in infos_cumulative)

print(f"Step rewards: {max_reward_step:.1f}")
print(f"Cumulative rewards: {max_reward_cumulative:.1f}")
```

### Record and Save Frames

```python
import numpy as np

gas = AtariFractalGas(env, N=30, record_frames=True)
state = gas.reset()
infos = []

for _ in range(50):
    state, info = gas.step(state)
    infos.append(info)

# Save frames as images
from PIL import Image

for i, info in enumerate(infos):
    if 'best_frame' in info:
        img = Image.fromarray(info['best_frame'])
        img.save(f"frame_{i:03d}.png")
```

### Live Monitoring (Simple)

```python
import matplotlib.pyplot as plt

gas = AtariFractalGas(env, N=30, use_cumulative_reward=True)
state = gas.reset()

rewards = []

for i in range(100):
    state, info = gas.step(state)
    rewards.append(info['max_reward'])

    if i % 10 == 0:
        plt.clf()
        plt.plot(rewards)
        plt.xlabel('Iteration')
        plt.ylabel('Max Reward')
        plt.title('Cumulative Reward Progress')
        plt.pause(0.01)

plt.show()
```

## 8. Troubleshooting

### Dashboard won't start

```bash
# Install missing dependencies
pip install panel holoviews bokeh pillow
```

### Frames show as black

- Set `obs_type="rgb"` (not "ram" or "grayscale")
- Ensure `record_frames=True`
- Some environments may not support rendering

### Import hangs

- Dashboard uses lazy imports via `get_dashboard_components()`
- Don't import dashboard modules directly at top level
- Use the helper function to load on demand

### High memory usage

- Use `record_frames=False` when visualization not needed
- Reduce `max_iterations` when recording
- Only best frame stored (~100KB each), but adds up

## 9. Next Steps

- Read full documentation: `ATARI_DASHBOARD_README.md`
- Explore example: `examples/atari_dashboard_example.py`
- Check implementation details: `ATARI_DASHBOARD_IMPLEMENTATION.md`
- Run tests: `pytest tests/fractalai/videogames/test_dashboard.py`

## 10. Quick Reference

| Feature | Parameter | Default | Location |
|---------|-----------|---------|----------|
| Cumulative rewards | `use_cumulative_reward` | `False` | `AtariFractalGas`, `FractalCloningOperator` |
| Frame recording | `record_frames` | `False` | `AtariFractalGas` |
| Dashboard | `get_dashboard_components()` | - | `fragile.fractalai.videogames` |

**Key classes:**
- `AtariFractalGas` - Main algorithm
- `AtariHistory` - History container
- `AtariGasConfigPanel` - Dashboard controls
- `AtariGasVisualizer` - Dashboard visualization

**Key methods:**
- `gas.step(state)` - Single iteration (returns state, info)
- `gas.run(max_iterations)` - Run multiple iterations
- `AtariHistory.from_run(infos, state, N, game_name)` - Build history
- `create_app().show(port)` - Launch dashboard
