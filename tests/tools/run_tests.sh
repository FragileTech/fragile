#!/usr/bin/env bash

# Test runner for format_math_blocks.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_SCRIPT="$SCRIPT_DIR/../../src/tools/format_math_blocks.py"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running format_math_blocks tests..."
echo "===================================="
echo

# Counter for tracking
total=0
passed=0

# Run tests
for test_file in "$SCRIPT_DIR"/test_*.md; do
    filename=$(basename "$test_file")
    echo "Testing: $filename"

    if python3 "$TOOL_SCRIPT" "$test_file" -o "$OUTPUT_DIR/$filename"; then
        ((total++))
        ((passed++))
    else
        ((total++))
        echo "  ❌ FAILED"
    fi
done

echo
echo "===================================="
echo "Tests complete: $passed/$total passed"
echo
echo "Output files are in: $OUTPUT_DIR"
echo
echo "To compare with expected results:"
echo "  diff expected_01_basic.md output/test_01_basic.md"
