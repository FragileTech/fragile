# Running Atari Dashboard on WSL

## Quick Start

Use the provided launcher script (handles all configuration):

```bash
bash scripts/run_dashboard_wsl.sh
```

Then open http://localhost:5006 in your Windows browser.

## Manual Setup

If you prefer to configure manually:

### 1. Install Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    xvfb \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-vulkan-drivers
```

### 2. Configure Environment

```bash
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
```

### 3. Run with xvfb

```bash
xvfb-run -a \
    --server-args="-screen 0 1024x768x24 +extension GLX" \
    python src/fragile/fractalai/videogames/dashboard.py
```

## Troubleshooting

### XCB Threading Error

```
[xcb] Extra reply data still left in queue
[xcb] This is most likely caused by a broken X extension library
python: xcb_io.c:581: int _XReply(Display *, xReply *, int, int):
  Assertion `!xcb_xlib_extra_reply_data_left' failed.
Aborted (core dumped)
```

**Root Cause**: X11/XCB is not thread-safe. This error occurs when OpenGL/X11 is initialized from a background thread.

**Solution**: The dashboard now creates Atari environments in the main thread before spawning worker threads. This is handled automatically - just use the launcher script:

```bash
bash scripts/run_dashboard_wsl.sh
```

**Technical Details**: The dashboard previously created gymnasium/plangym environments inside the simulation worker thread. When `gym.make()` or `AtariEnvironment()` was called from a background thread, it triggered X11/OpenGL initialization, causing the XCB threading error. The fix:

1. Environment creation now happens in `_on_run_clicked()` (main thread)
2. Pre-created environment is passed to `_run_simulation_worker()`
3. Worker thread only calls `env.reset()` and `env.step()` (thread-safe operations)

This pattern is confirmed by `tests/test_xcb_threading.py` which demonstrates:
- Creating environments in background threads fails with XCB errors (Test 3)
- Pre-creating environments in main thread and using them in workers succeeds (Test 5)

### Mesa Not Found

```bash
sudo apt-get install mesa-utils
# Verify installation
glxinfo | grep "OpenGL renderer"
# Should show: "llvmpipe" (software rendering)
```

### Display Issues

The dashboard starts but simulations fail:
- Make sure xvfb-run is wrapping the Python command
- Check that DISPLAY is set: `echo $DISPLAY`
- Verify Mesa: `glxinfo | head`

## Docker Alternative

If xvfb still doesn't work, use Docker:

```bash
docker build -f docker/Dockerfile.atari-dashboard -t atari-dashboard .
docker run -p 5006:5006 atari-dashboard
```

## WSL2 + WSLg

If you're on Windows 11 with WSL2, WSLg might work directly:

```bash
# Check if WSLg is available
echo $DISPLAY  # Should show something like :0

# If yes, just run normally
python src/fragile/fractalai/videogames/dashboard.py
```

## Advanced Options

The launcher script supports several options:

```bash
# Use a different port
bash scripts/run_dashboard_wsl.sh --port 8080

# Enable multi-threaded Tornado (not recommended on WSL)
bash scripts/run_dashboard_wsl.sh --threaded

# Open browser automatically (requires Windows browser integration)
bash scripts/run_dashboard_wsl.sh --open

# See all options
bash scripts/run_dashboard_wsl.sh --help
```

## Why plangym?

This dashboard uses plangym (not pure gymnasium) because it provides:
- State get/set capabilities needed for the Fractal Gas algorithm
- Time travel and state cloning operations
- Better integration with the existing codebase

However, the dashboard will automatically fall back to gymnasium if plangym is not available or has display issues.

## Known Issues

### Issue: "pyglet.canvas.xlib.NoSuchDisplayException"
**Solution**: Make sure you're using xvfb-run or have DISPLAY set

### Issue: "ModuleNotFoundError: No module named 'plangym'"
**Solution**: Install plangym: `pip install plangym`

### Issue: Multi-threaded Tornado causes XCB errors
**Solution**: Use single-threaded mode (default) by not passing `--threaded`

## Performance Notes

- Software rendering (llvmpipe) is slower than GPU rendering
- For production use, consider running on a system with proper GPU support
- The launcher script uses single-threaded mode by default for stability
