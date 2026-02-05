# AtariEnv Test Suite - Quick Reference

## Quick Test Commands

```bash
# Run all 44 tests (~8 seconds)
pytest tests/fractalai/videogames/test_atari_env.py -v

# Run specific test categories
pytest tests/fractalai/videogames/test_atari_env.py -k "batch"        # 9 tests
pytest tests/fractalai/videogames/test_atari_env.py -k "determinism"  # 2 tests
pytest tests/fractalai/videogames/test_atari_env.py -k "headless"     # 4 tests
pytest tests/fractalai/videogames/test_atari_env.py -k "fractal_gas"  # 6 tests

# Run manual verification (~20 seconds)
python verify_atari_integration.py
```

## Test Results Summary

✓ **44/44 tests pass** (100% success rate)
✓ All tests use **real Atari environments** (no mocks)
✓ **Deterministic replay verified** (critical for Fractal Gas)
✓ **WSL headless mode** fully supported
✓ **AtariFractalGas integration** validated (50 iterations, 30 walkers)

## Test Categories Breakdown

| Category | Tests | Focus |
|----------|-------|-------|
| A. Environment Setup | 6 | Initialization, obs_types, action_space |
| B. State Management | 8 | Cloning, restoration, determinism |
| C. Single Step | 8 | step(), dt parameter, rewards |
| D. Batch Operations | 8 | step_batch(), N=1 to N=50 walkers |
| E. AtariFractalGas | 6 | Full integration with fractal gas |
| F. WSL Headless | 4 | PYGLET_HEADLESS, render(), no DISPLAY |
| G. Edge Cases | 4 | Multiple games, RGB/grayscale obs |

## Critical Tests for Production

1. **Deterministic Replay** (`test_clone_restore_roundtrip_determinism`)
   - Ensures clone/restore produces identical results
   - Critical for Fractal Gas exploration algorithm

2. **Batch Operations** (`test_step_batch_*` - 8 tests)
   - Validates 6-tuple output format
   - Tests N=1 to N=50 walker scaling

3. **AtariFractalGas Full Run** (`test_atari_fractal_gas_full_run`)
   - 50 iterations with 30 walkers
   - End-to-end validation

4. **WSL Headless** (`test_all_obs_types_work_in_headless_wsl`)
   - All observation types in headless mode
   - No X11/DISPLAY dependencies

## Files Overview

```
tests/fractalai/videogames/
├── test_atari_env.py          # 44 integration tests (NEW)

verify_atari_integration.py     # Manual verification script (NEW)
TEST_SUITE_SUMMARY.md           # Detailed test documentation (NEW)
ATARI_TEST_QUICK_REFERENCE.md   # This file (NEW)

src/fragile/fractalai/videogames/
└── atari.py                    # AtariEnv implementation (TESTED)
```

## Common Test Patterns

### Running Specific Test
```bash
pytest tests/fractalai/videogames/test_atari_env.py::test_clone_restore_roundtrip_determinism -v
```

### Running with Verbose Output
```bash
pytest tests/fractalai/videogames/test_atari_env.py -v -s
```

### Quick Smoke Test (fastest tests)
```bash
pytest tests/fractalai/videogames/test_atari_env.py -k "creation or reset or step_with_random"
```

## Manual Verification Sections

Run `python verify_atari_integration.py` to execute:

1. ✓ Basic AtariEnv Verification
2. ✓ Batch Operations Verification
3. ✓ Observation Types Verification
4. ✓ WSL Headless Rendering Verification
5. ✓ AtariFractalGas Integration Verification
6. ✓ Multiple Games Verification

Expected output: "✓ ALL VERIFICATIONS PASSED"

## Environment Setup

Already configured in `conftest.py`:
```python
os.environ.setdefault("PYGLET_HEADLESS", "1")
```

Dependencies already installed:
- gymnasium[atari] >= 1.2.3
- ale-py >= 0.11.2

## Key Implementation Details

### AtariState Structure
```python
@dataclass
class AtariState:
    ale_state: object           # Immutable ALE state
    observation: np.ndarray     # Deep copied
    rgb_frame: np.ndarray | None  # Deep copied

    def copy(self) -> "AtariState":
        # Creates independent copy
```

### step_batch Output Format
```python
(
    new_states,   # np.ndarray[object], N AtariState instances
    observations, # np.ndarray[float32], shape (N, obs_dim)
    rewards,      # np.ndarray[float32], shape (N,)
    dones,        # np.ndarray[bool], shape (N,)
    truncated,    # np.ndarray[bool], shape (N,)
    infos,        # list[dict], length N
)
```

### Observation Shapes
- **RAM**: (128,) uint8
- **RGB**: (210, 160, 3) uint8
- **Grayscale**: (210, 160) uint8

## Troubleshooting

### Tests fail with "cannot read termcap database"
- **This is a warning, not an error** - tests still run correctly
- Cosmetic issue with terminal settings in WSL

### Tests fail with X11/XCB errors
- Check `PYGLET_HEADLESS=1` is set (should be in conftest.py)
- Verify: `echo $PYGLET_HEADLESS` should output "1"

### Tests fail with import errors
- Reinstall: `pip install "gymnasium[atari]" "gymnasium[accept-rom-license]"`
- Verify: `python -c "import gymnasium; import ale_py"`

### Determinism tests fail
- Very rare - may indicate ALE version regression
- Check ALE-py version: `pip show ale-py`
- Expected version: 0.11.2+

## Performance Benchmarks

- **Single test**: ~0.2 seconds
- **All 44 tests**: ~8 seconds
- **Manual verification**: ~20 seconds
- **50-iteration Fractal Gas run**: ~10 seconds (30 walkers)

## Test Fixtures

Available fixtures (defined in test file):
- `device` - Returns "cpu"
- `pong_env` - RAM observations
- `pong_env_rgb` - RGB observations
- `pong_env_grayscale` - Grayscale observations
- `pong_env_no_rgb` - RAM without RGB frames
- `breakout_env` - Breakout game

All fixtures auto-cleanup (call env.close())

## Next Steps

1. ✓ Tests implemented and passing
2. ✓ Manual verification successful
3. ✓ Documentation complete
4. Ready for production use

## Contact

For issues or questions about the test suite, refer to:
- Implementation: `src/fragile/fractalai/videogames/atari.py`
- Tests: `tests/fractalai/videogames/test_atari_env.py`
- Summary: `TEST_SUITE_SUMMARY.md`
