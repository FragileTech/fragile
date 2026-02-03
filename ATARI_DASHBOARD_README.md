# Atari Fractal Gas Dashboard

Interactive real-time dashboard for visualizing and analyzing the Atari Fractal Gas algorithm.

## Features

### 1. Cumulative Reward Option

The algorithm now supports using cumulative rewards (instead of step rewards) for fitness calculation:

```python
from fragile.fractalai.videogames import AtariFractalGas

gas = AtariFractalGas(
    env=env,
    N=30,
    use_cumulative_reward=True,  # Use cumulative rewards for fitness
    record_frames=True,           # Record best walker frames
)
```

**Default behavior**: `use_cumulative_reward=False` (uses step rewards, preserves original behavior)

### 2. Frame Recording

Record the best walker's game frame at each iteration for visualization:

```python
gas = AtariFractalGas(env=env, N=30, record_frames=True)
state = gas.reset()

for _ in range(100):
    state, info = gas.step(state)
    if 'best_frame' in info:
        frame = info['best_frame']  # RGB array [210, 160, 3]
        best_idx = info['best_walker_idx']
```

**Memory efficient**: Only stores best walker frame per iteration (~100KB/iteration)

### 3. Interactive Dashboard

Launch an interactive dashboard with:
- Real-time best walker frame display
- Cumulative reward progression curves
- Live metric histograms (alive count, cloning events, virtual rewards)
- Frame-by-frame playback with time slider
- Parameter controls and simulation runner

```python
from fragile.fractalai.videogames import get_dashboard_components

AtariGasConfigPanel, AtariGasVisualizer, create_app = get_dashboard_components()

# Launch dashboard
app = create_app()
app.show(port=5006)
```

Or directly from command line:

```bash
python -m fragile.fractalai.videogames.dashboard
```

## Installation

The dashboard requires additional dependencies:

```bash
pip install panel holoviews bokeh pillow
```

For Atari environments:

```bash
pip install plangym
```

## Usage Examples

### Basic Usage with New Features

```python
from plangym import AtariEnvironment
from fragile.fractalai.videogames import AtariFractalGas, AtariHistory

# Create environment
env = AtariEnvironment(name="PongNoFrameskip-v4", obs_type="ram")

# Create algorithm with new features
gas = AtariFractalGas(
    env=env,
    N=30,
    dist_coef=1.0,
    reward_coef=1.0,
    use_cumulative_reward=True,  # NEW: Use cumulative rewards
    record_frames=True,           # NEW: Record frames
    device="cpu",
    seed=42,
)

# Run simulation
state = gas.reset()
infos = []

for i in range(100):
    state, info = gas.step(state)
    infos.append(info)
    print(f"Iteration {i}: Max reward = {info['max_reward']:.1f}")

# Build history for analysis
history = AtariHistory.from_run(infos, state, N=30, game_name="Pong")

print(f"Max reward achieved: {max(history.rewards_max):.1f}")
print(f"Frames recorded: {history.has_frames}")
```

### Dashboard Usage

```python
# Import dashboard components on demand (avoids heavy imports)
from fragile.fractalai.videogames import get_dashboard_components

AtariGasConfigPanel, AtariGasVisualizer, create_app = get_dashboard_components()

# Launch dashboard application
app = create_app()
app.show(port=5006)  # Navigate to http://localhost:5006
```

**Dashboard workflow:**
1. Select Atari game (Pong, Breakout, MsPacman, SpaceInvaders)
2. Configure algorithm parameters
3. Toggle "Use cumulative reward" option
4. Set max iterations and enable frame recording
5. Click "Run Simulation"
6. Watch progress bar and real-time metrics
7. Use time slider to review recorded frames
8. Analyze reward progression curves and histograms

### Programmatic Visualization

```python
from fragile.fractalai.videogames import get_dashboard_components

_, AtariGasVisualizer, _ = get_dashboard_components()

# Create visualizer with existing history
visualizer = AtariGasVisualizer(history=history)

# Display in Jupyter notebook
visualizer.panel()
```

## API Reference

### AtariFractalGas

**New parameters:**
- `use_cumulative_reward: bool = False` - Use cumulative rewards for fitness calculation (default: step rewards)
- `record_frames: bool = False` - Record best walker frames at each iteration

**New info dict entries** (when `record_frames=True`):
- `best_frame: np.ndarray` - RGB frame [210, 160, 3] of best walker
- `best_walker_idx: int` - Index of best performing walker

### FractalCloningOperator

**New parameters:**
- `use_cumulative_reward: bool = False` - Use cumulative rewards for fitness

**Updated signature:**
```python
def calculate_fitness(
    self,
    observations: Tensor,
    cumulative_rewards: Tensor,  # Cumulative rewards
    step_rewards: Tensor,         # Step rewards
    alive: Tensor,
) -> tuple[Tensor, Tensor]:
```

### AtariHistory

Container for algorithm execution history:

```python
@dataclass
class AtariHistory:
    iterations: list[int]
    rewards_mean: list[float]
    rewards_max: list[float]
    rewards_min: list[float]
    alive_counts: list[int]
    num_cloned: list[int]
    virtual_rewards_mean: list[float]
    virtual_rewards_max: list[float]
    best_frames: list[np.ndarray | None]  # Recorded frames
    best_rewards: list[float]
    best_indices: list[int]
    N: int
    max_iterations: int
    game_name: str

    @property
    def has_frames(self) -> bool:
        """Check if frames were recorded."""
```

**Construction:**
```python
history = AtariHistory.from_run(infos, final_state, N=30, game_name="Pong")
```

## Dashboard Components

### AtariGasConfigPanel

Parameter control panel with:
- Environment selection (game, observation type)
- Algorithm parameters (N, dist_coef, reward_coef, use_cumulative_reward)
- Simulation controls (max_iterations, record_frames, device, seed)
- Run/Stop buttons and progress tracking

### AtariGasVisualizer

Visualization panel with:
- Best walker frame display (with time player)
- Cumulative reward progression curves (max and mean)
- Metric histograms (alive walkers, cloning events, virtual rewards)
- Iteration info display

## Performance Considerations

### Memory Usage

**Frame recording overhead:**
- Each frame: ~100KB (210×160×3 uint8 array)
- 100 iterations: ~10MB
- 1000 iterations: ~100MB

**Recommendation**: Enable `record_frames=True` only when visualization is needed.

### Cumulative vs Step Rewards

**Step rewards** (default):
- Fitness based on immediate reward signal
- Encourages exploration of diverse trajectories
- Default behavior, backward compatible

**Cumulative rewards**:
- Fitness based on total accumulated reward
- Stronger selection pressure toward high-reward states
- May reduce diversity in walker population

**When to use each:**
- Use step rewards for exploration-heavy tasks
- Use cumulative rewards when you want faster convergence to high-reward regions
- Experiment with both to see which works better for your game

## Testing

Run tests with:

```bash
pytest tests/fractalai/videogames/test_dashboard.py -v
```

Key tests:
- `test_cumulative_reward_option`: Verify cumulative vs step reward behavior
- `test_history_construction`: Test AtariHistory creation
- `test_frame_recording`: Verify frame recording functionality
- `test_dashboard_creation`: Test dashboard initialization

## Files

### Modified
- `src/fragile/fractalai/videogames/cloning.py` - Added `use_cumulative_reward` parameter
- `src/fragile/fractalai/videogames/atari_gas.py` - Added frame recording and parameter passing
- `src/fragile/fractalai/videogames/__init__.py` - Added `get_dashboard_components()` function

### New
- `src/fragile/fractalai/videogames/atari_history.py` - History container dataclass
- `src/fragile/fractalai/videogames/dashboard.py` - Dashboard implementation
- `tests/fractalai/videogames/test_dashboard.py` - Unit tests
- `examples/atari_dashboard_example.py` - Usage examples

## Running on WSL

The dashboard works on WSL but requires special configuration for OpenGL/display support.

### Quick Start (WSL)

Use the provided launcher script:

```bash
bash scripts/run_dashboard_wsl.sh
```

This script automatically configures:
- Software OpenGL rendering (Mesa llvmpipe)
- Virtual framebuffer (xvfb)
- Environment variables for headless operation

### Manual WSL Setup

See [README_WSL.md](README_WSL.md) for detailed WSL setup instructions, including:
- Installing xvfb and Mesa dependencies
- Manual environment configuration
- Docker alternative
- Troubleshooting XCB threading errors

## Troubleshooting

### Dashboard won't start

Ensure dependencies are installed:
```bash
pip install panel holoviews bokeh pillow
```

### XCB Threading Error (WSL)

```
[xcb] Extra reply data still left in queue
```

**Solution**: Use the WSL launcher script:
```bash
bash scripts/run_dashboard_wsl.sh
```

See [README_WSL.md](README_WSL.md) for details.

### Frames show as black

- Check that `obs_type="rgb"` (not "ram" or "grayscale")
- Verify `record_frames=True` is set
- Some environments may not support rendering

### High memory usage

- Reduce `max_iterations` when recording frames
- Use `record_frames=False` when visualization is not needed
- Only ~100KB per iteration, but can add up for long runs

## Future Enhancements

Potential improvements:
- [ ] Video export functionality (save frames as MP4)
- [ ] Comparison mode (run multiple configurations side-by-side)
- [ ] Live streaming to dashboard (update during simulation)
- [ ] Support for non-Atari environments
- [ ] Custom reward visualization overlays

## License

Same as fragile project license.
