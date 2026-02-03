# Fractal Gas Quick Start Guide

## Installation

```bash
# Install plangym for Atari environments
pip install plangym[atari]
```

## Basic Usage

```python
from plangym import AtariEnvironment
from fragile.fractalai.videogames import AtariFractalGas

# Create environment (RAM observations recommended for speed)
env = AtariEnvironment(name="Pong-v5", obs_type="ram")

# Initialize fractal gas
gas = AtariFractalGas(
    env=env,
    N=30,                # Number of walkers
    dist_coef=1.0,       # Distance weight in fitness
    reward_coef=1.0,     # Reward weight in fitness
    dt_range=(1, 4),     # Frame skip range
    device="cpu",        # or "cuda"
    seed=42
)

# Run the algorithm
final_state, history = gas.run(max_iterations=200)

# Get results
best_idx, best_reward = gas.get_best_walker(final_state)
print(f"Best reward: {best_reward:.2f}")
print(f"Total steps: {gas.total_steps}")

env.close()
```

## Key Parameters

- **N**: Number of walkers (more = better exploration, slower)
- **dist_coef**: Distance coefficient in fitness (higher = favor diverse states)
- **reward_coef**: Reward coefficient in fitness (higher = favor high rewards)
- **dt_range**: Frame skip range (higher = faster but less control)
- **device**: "cpu" or "cuda" for GPU acceleration

## Understanding the Output

### WalkerState
The state of all walkers at any point:
- `states`: Environment states [N]
- `observations`: RAM observations [N, 128]
- `rewards`: Cumulative rewards [N]
- `alive`: Boolean mask of alive walkers [N]
- `virtual_rewards`: Fitness values [N]

### History
List of info dicts from each iteration:
- `iteration`: Iteration number
- `num_cloned`: Number of cloning events
- `alive_count`: Number of alive walkers
- `mean_reward`: Average cumulative reward
- `max_reward`: Best cumulative reward
- `mean_virtual_reward`: Average fitness

## Tips

1. **Use RAM observations**: 221x faster than pixels
   ```python
   env = AtariEnvironment(name="YourGame-v5", obs_type="ram")
   ```

2. **Start with fewer walkers**: Test with N=10-20 before scaling up

3. **Tune coefficients**: Balance exploration (dist_coef) and exploitation (reward_coef)
   - High dist_coef → more exploration
   - High reward_coef → more exploitation

4. **Monitor alive count**: If walkers die too fast, adjust dt_range or add more walkers

5. **Use GPU for large N**: Significant speedup for N > 50
   ```python
   device="cuda" if torch.cuda.is_available() else "cpu"
   ```

## Examples

See `examples/fractal_gas_example.py` for complete working examples.

## Running Tests

```bash
# Run all unit tests
pytest tests/fractalai/videogames/ -v --ignore=tests/fractalai/videogames/test_pong_integration.py

# Run integration tests (requires plangym)
pytest tests/fractalai/videogames/test_pong_integration.py -v -s
```

## Troubleshooting

**Import Error**: Make sure fragile is installed
```bash
pip install -e .
```

**plangym hanging**: Some Atari games take time to initialize first time

**Out of memory**: Reduce N or use smaller dt_range

**Slow performance**: Use RAM observations and GPU if available
