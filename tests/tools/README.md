# Format Math Blocks Tests

This directory contains test cases for the `format_math_blocks.py` script.

## Test Files

- **test_01_basic.md**: Basic inline math block that needs formatting
- **test_02_multiline.md**: Multiline equation with extra whitespace
- **test_03_consecutive.md**: Two math blocks right after each other
- **test_04_inline_code.md**: Math symbols inside inline code (should NOT be formatted)
- **test_05_fenced_code.md**: Math symbols inside fenced code blocks (should NOT be formatted)
- **test_06_empty_lines.md**: Math block with empty lines inside (should be removed)
- **test_07_start_of_file.md**: Math block at the very start of file
- **test_08_end_of_file.md**: Math block at the very end of file
- **test_09_complex.md**: Complex document with multiple scenarios
- **test_10_aligned.md**: Aligned equations and cases environments

## Expected Files

Files prefixed with `expected_` show the correct output format for comparison.

## Running Tests

```bash
cd tests/tools
chmod +x run_tests.sh
./run_tests.sh
```

This will create an `output/` directory with the formatted versions of all test files.

## Manual Verification

Compare the output with expected results:

```bash
diff expected_01_basic.md output/test_01_basic.md
diff expected_02_multiline.md output/test_02_multiline.md
# etc.
```

## Formatting Rules

The script enforces these rules:

1. One blank line before the opening `$$`
2. The opening `$$` is on its own line
3. The closing `$$` is on its own line immediately after content
4. No blank line between opening `$$` and content
5. No blank line between content and closing `$$`
6. Empty lines within equation blocks are removed
7. Math in code blocks (inline `` ` `` or fenced ` ``` `) is protected
