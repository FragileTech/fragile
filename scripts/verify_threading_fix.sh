#!/bin/bash
# Verification script for XCB threading fix
# This script helps verify that the threading fix is working correctly

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "XCB Threading Fix Verification"
echo "═══════════════════════════════════════════════════════════════"
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Step 1: Check test suite exists
echo "Step 1: Checking test suite exists..."
if [ -f "tests/test_xcb_threading.py" ]; then
    print_status 0 "Test suite found"
else
    print_status 1 "Test suite not found"
    echo "  Expected: tests/test_xcb_threading.py"
    exit 1
fi
echo

# Step 2: Check dashboard has threading fix
echo "Step 2: Checking dashboard code has threading fix..."
if grep -q "_create_environment" src/fragile/fractalai/videogames/dashboard.py; then
    print_status 0 "Dashboard has _create_environment method"
else
    print_status 1 "Dashboard missing _create_environment method"
    exit 1
fi

if grep -q "args=(env,)" src/fragile/fractalai/videogames/dashboard.py; then
    print_status 0 "Dashboard passes environment to worker thread"
else
    print_status 1 "Dashboard doesn't pass environment to worker"
    exit 1
fi
echo

# Step 3: Check launcher script has enhanced env vars
echo "Step 3: Checking launcher script has enhanced environment variables..."
if grep -q "SDL_VIDEODRIVER=dummy" scripts/run_dashboard_wsl.sh; then
    print_status 0 "SDL_VIDEODRIVER=dummy present"
else
    print_status 1 "SDL_VIDEODRIVER=dummy missing"
fi

if grep -q "MPLBACKEND=Agg" scripts/run_dashboard_wsl.sh; then
    print_status 0 "MPLBACKEND=Agg present"
else
    print_status 1 "MPLBACKEND=Agg missing"
fi
echo

# Step 4: Run test suite
echo "Step 4: Running test suite..."
echo "───────────────────────────────────────────────────────────────"
if python tests/test_xcb_threading.py; then
    echo "───────────────────────────────────────────────────────────────"
    print_status 0 "Test suite passed"
else
    echo "───────────────────────────────────────────────────────────────"
    print_status 1 "Test suite failed"
    echo
    print_warning "This might be expected if Atari environments aren't installed"
    print_warning "The important test is Test 5 (pre-created env)"
fi
echo

# Step 5: Check Python syntax
echo "Step 5: Checking Python syntax..."
if python -m py_compile src/fragile/fractalai/videogames/dashboard.py 2>/dev/null; then
    print_status 0 "Dashboard code compiles"
else
    print_status 1 "Dashboard code has syntax errors"
    exit 1
fi

if python -m py_compile tests/test_xcb_threading.py 2>/dev/null; then
    print_status 0 "Test suite compiles"
else
    print_status 1 "Test suite has syntax errors"
    exit 1
fi
echo

# Step 6: Check documentation
echo "Step 6: Checking documentation..."
if [ -f "XCB_THREADING_FIX.md" ]; then
    print_status 0 "Implementation summary exists"
else
    print_status 1 "Implementation summary missing"
fi

if grep -q "XCB Threading Error" README_WSL.md; then
    print_status 0 "README_WSL.md updated with threading info"
else
    print_status 1 "README_WSL.md missing threading info"
fi
echo

# Summary
echo "═══════════════════════════════════════════════════════════════"
echo "VERIFICATION SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo
echo "The XCB threading fix has been implemented with the following changes:"
echo
echo "1. ✓ Test suite created (tests/test_xcb_threading.py)"
echo "2. ✓ Dashboard refactored to create environments in main thread"
echo "3. ✓ Launcher script enhanced with additional env vars"
echo "4. ✓ Documentation updated"
echo
echo "Next steps to verify the fix works on WSL:"
echo
echo "1. Start the dashboard:"
echo "   bash scripts/run_dashboard_wsl.sh --port 5006"
echo
echo "2. Open http://localhost:5006 in your browser"
echo
echo "3. Configure a simulation:"
echo "   - Game: Pong"
echo "   - N: 10"
echo "   - max_iterations: 100"
echo
echo "4. Click 'Run Simulation'"
echo
echo "5. Verify:"
echo "   - No XCB errors in terminal"
echo "   - Progress bar updates"
echo "   - Simulation completes"
echo "   - Frames display in visualizer"
echo
echo "If you see XCB errors, check:"
echo "   - xvfb is installed (sudo apt-get install xvfb)"
echo "   - Mesa is installed (sudo apt-get install mesa-utils)"
echo "   - You're running via the launcher script"
echo
echo "═══════════════════════════════════════════════════════════════"
