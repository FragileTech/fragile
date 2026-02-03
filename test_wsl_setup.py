#!/usr/bin/env python3
"""Quick test script to verify WSL setup for Atari Dashboard."""

import os
import sys


def check_display():
    """Check if DISPLAY is set."""
    display = os.environ.get('DISPLAY')
    if display:
        print(f"✓ DISPLAY is set: {display}")
        return True
    else:
        print("✗ DISPLAY not set")
        print("  Run with: xvfb-run -a python test_wsl_setup.py")
        return False


def check_opengl():
    """Check if OpenGL/Mesa is available."""
    # Check for mesa libraries
    try:
        import subprocess
        result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=2)
        if 'libGL.so' in result.stdout:
            print("✓ OpenGL libraries found")
            # Check if software rendering is configured
            if os.environ.get('LIBGL_ALWAYS_SOFTWARE') == '1':
                print("  (Software rendering enabled)")
            return True
        else:
            print("✗ OpenGL libraries not found")
            print("  Install: sudo apt-get install mesa-utils libgl1-mesa-glx libgl1-mesa-dri")
            return False
    except Exception as e:
        print(f"⚠ Could not check OpenGL libraries: {e}")
        return False


def check_pyglet():
    """Check if pyglet can be imported."""
    try:
        # Only check if pyglet is importable, don't try to use it
        # (using it may trigger XCB errors without proper setup)
        import importlib.util
        spec = importlib.util.find_spec("pyglet")
        if spec is not None:
            print("✓ pyglet is installed")
            return True
        else:
            print("✗ pyglet not installed (install: pip install pyglet)")
            return False
    except ImportError:
        print("✗ pyglet not installed (install: pip install pyglet)")
        return False
    except Exception as e:
        print(f"⚠ Error checking pyglet: {e}")
        return False


def check_atari_env():
    """Check if Atari environments are available."""
    # Only check if packages are importable, don't try to create environments
    # (creating environments may trigger display/OpenGL issues)

    # Try gymnasium first
    try:
        import importlib.util
        spec = importlib.util.find_spec("gymnasium")
        if spec is not None:
            print("✓ gymnasium is installed")
            # Check for atari-specific package
            atari_spec = importlib.util.find_spec("ale_py")
            if atari_spec is not None:
                print("✓ Atari environments available (gymnasium)")
                return True
            else:
                print("⚠ gymnasium installed but ale_py not found")
                print("  Install: pip install gymnasium[atari] gymnasium[accept-rom-license]")
    except Exception:
        print("⚠ gymnasium not installed")

    # Try plangym
    try:
        import importlib.util
        spec = importlib.util.find_spec("plangym")
        if spec is not None:
            print("✓ plangym is installed")
            return True
    except Exception:
        pass

    print("✗ Neither gymnasium nor plangym available")
    print("  Install: pip install gymnasium[atari] gymnasium[accept-rom-license]")
    print("  Or: pip install plangym")
    return False


def check_dashboard_deps():
    """Check dashboard dependencies."""
    deps = ['panel', 'holoviews', 'bokeh', 'PIL']
    all_ok = True

    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep} is installed")
        except ImportError:
            print(f"✗ {dep} not installed")
            all_ok = False

    if not all_ok:
        print("  Install: pip install panel holoviews bokeh pillow")

    return all_ok


def main():
    """Run all checks."""
    print("=" * 60)
    print("WSL Setup Check for Atari Dashboard")
    print("=" * 60)
    print()

    results = {
        "DISPLAY": check_display(),
        "OpenGL": check_opengl(),
        "pyglet": check_pyglet(),
        "Atari Environments": check_atari_env(),
        "Dashboard Dependencies": check_dashboard_deps(),
    }

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}")

    print()

    if all(results.values()):
        print("🎉 All checks passed! You can run the dashboard:")
        print("   python src/fragile/fractalai/videogames/dashboard.py")
    elif results["DISPLAY"] and results["Atari Environments"] and results["Dashboard Dependencies"]:
        print("⚠ Basic requirements met. Dashboard will start but simulations may fail.")
        print("  Use the launcher script for full support:")
        print("   bash scripts/run_dashboard_wsl.sh")
    else:
        print("❌ Some requirements are missing. See errors above.")
        print()
        print("For WSL setup, see: README_WSL.md")
        print("Or use the launcher script: bash scripts/run_dashboard_wsl.sh")
        sys.exit(1)


if __name__ == "__main__":
    main()
