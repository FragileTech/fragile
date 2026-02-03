# Atari Dashboard on WSL - Quick Start

## TL;DR

```bash
# 1. Check your setup
python test_wsl_setup.py

# 2. Run the dashboard
bash scripts/run_dashboard_wsl.sh

# 3. Open http://localhost:5006 in your browser
```

## First Time Setup

If you're on a fresh WSL installation:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y xvfb mesa-utils libgl1-mesa-glx libgl1-mesa-dri

# Install Python dependencies
pip install gymnasium[atari] gymnasium[accept-rom-license]
pip install panel holoviews bokeh pillow

# Verify setup
python test_wsl_setup.py
```

## Common Issues

### "xvfb-run not found"
```bash
sudo apt-get install xvfb
```

### "No module named 'gymnasium'"
```bash
pip install gymnasium[atari] gymnasium[accept-rom-license]
```

### "No module named 'panel'"
```bash
pip install panel holoviews bokeh pillow
```

### XCB threading errors
Always use the launcher script - it handles all the environment configuration:
```bash
bash scripts/run_dashboard_wsl.sh
```

## What the Launcher Script Does

The `run_dashboard_wsl.sh` script automatically:
1. ✅ Checks dependencies are installed
2. ✅ Sets up OpenGL software rendering
3. ✅ Configures environment variables
4. ✅ Starts xvfb virtual framebuffer
5. ✅ Launches dashboard in single-threaded mode

## Using the Dashboard

1. **Configure simulation** in the left sidebar:
   - Select game (Pong, Breakout, etc.)
   - Set number of walkers (N)
   - Choose max iterations
   - Enable frame recording

2. **Run simulation**:
   - Click "Run Simulation"
   - Watch progress bar
   - Wait for completion

3. **View results**:
   - Use time slider to review frames
   - Check reward progression curves
   - Analyze metric histograms

## Advanced Options

```bash
# Use different port
bash scripts/run_dashboard_wsl.sh --port 8080

# See all options
bash scripts/run_dashboard_wsl.sh --help
```

## Need More Help?

- **Detailed WSL guide**: See [README_WSL.md](README_WSL.md)
- **Dashboard features**: See [ATARI_DASHBOARD_README.md](ATARI_DASHBOARD_README.md)
- **Implementation details**: See [WSL_IMPLEMENTATION_SUMMARY.md](WSL_IMPLEMENTATION_SUMMARY.md)

## Docker Alternative

If the launcher script doesn't work:

```bash
docker build -f docker/Dockerfile.atari-dashboard -t atari-dashboard .
docker run -p 5006:5006 atari-dashboard
```

Then open http://localhost:5006

---

**Questions?** Check the documentation files listed above or open an issue.
