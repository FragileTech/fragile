# Format Math Blocks Testing Guide

## Overview

This directory contains comprehensive tests for the `format_math_blocks.py` script, which formats LaTeX `$$..$$` blocks in markdown files according to specific formatting rules.

## Test Files

| Test File | Description | Key Features Tested |
|-----------|-------------|---------------------|
| `test_01_basic.md` | Basic inline math block | Simple case with text before and after |
| `test_02_multiline.md` | Multiline equation | Extra whitespace handling |
| `test_03_consecutive.md` | Two consecutive blocks | Proper spacing between blocks |
| `test_04_inline_code.md` | Inline code protection | Ensures `` `$$..$$` `` is not formatted |
| `test_05_fenced_code.md` | Fenced code protection | Ensures ` ```...$$...``` ` is not formatted |
| `test_06_empty_lines.md` | Empty lines inside math | Removes empty lines within equations |
| `test_07_start_of_file.md` | Math at file start | Handles math block at beginning |
| `test_08_end_of_file.md` | Math at file end | Handles math block at end |
| `test_09_complex.md` | Complex document | Multiple scenarios combined |
| `test_10_aligned.md` | Aligned equations | Multi-line aligned and cases environments |

## Expected Output Files

Files prefixed with `expected_` contain the correct formatted output for comparison:
- `expected_01_basic.md`
- `expected_02_multiline.md`
- `expected_03_consecutive.md`
- `expected_06_empty_lines.md`
- `expected_07_start_of_file.md`
- `expected_08_end_of_file.md`

## Running Tests

### Quick Run
```bash
cd tests/tools
./run_tests.sh
```

### Manual Run
```bash
cd tests/tools
python3 ../../src/tools/format_math_blocks.py test_01_basic.md -o output/test_01_basic.md
```

### Verify Output
```bash
diff expected_01_basic.md output/test_01_basic.md
```

## Formatting Rules Enforced

1. **One blank line before opening `$$`**
2. **Opening `$$` on its own line**
3. **Closing `$$` on its own line immediately after content**
4. **No blank line between opening `$$` and content**
5. **No blank line between content and closing `$$`**
6. **Empty lines within equation blocks are removed**
7. **Math in code blocks is protected** (both inline and fenced)

## Bug Fixes Applied

### Issue 1: Code Block Protection
- **Problem**: Only inline code was protected, not fenced code blocks
- **Fix**: Added regex to protect ` ```...$$...``` ` blocks

### Issue 2: Extra Blank Lines Inside Math
- **Problem**: Spacing rules were adding blank lines inside math blocks
- **Fix**: Replaced math blocks with placeholders before applying spacing rules

### Issue 3: No Spacing After Closing `$$`
- **Problem**: Text immediately after `$$` wasn't separated
- **Fix**: Added regex to handle both newline and no-newline cases after placeholders

## Test Results

All 10 tests pass successfully. The script correctly:
- ✓ Formats basic math blocks
- ✓ Handles multiline equations
- ✓ Spaces consecutive blocks properly
- ✓ Protects inline code
- ✓ Protects fenced code blocks
- ✓ Removes empty lines in math
- ✓ Handles math at file start
- ✓ Handles math at file end
- ✓ Processes complex documents
- ✓ Preserves aligned equations structure
