#!/usr/bin/env python3
import re
import argparse
import os


def format_latex_blocks(file_path: str, output_path: str = None) -> None:
    """
    Reads a file, formats the $$...$$ LaTeX blocks, and writes to output file.

    The formatting rules are:
    - One blank line before the opening $$.
    - The opening $$ is on its own line.
    - The closing $$ is on its own line immediately after the content.
    - No blank line between opening $$ and content.
    - No blank line between content and closing $$.
    - Empty lines within equation blocks are removed.

    Args:
        file_path: The path to the file to format.
        output_path: The path to write the formatted content. If None, overwrites the input file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at `{file_path}`")
        return

    # If no output path specified, overwrite the input file
    if output_path is None:
        output_path = file_path

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file `{file_path}`: {e}")
        return

    # Step 1: Protect ALL code blocks (not just those with $$)
    code_blocks = []

    def protect_code(match):
        code_blocks.append(match.group(0))
        return f"__CODEBLOCK_{len(code_blocks)-1}__"

    # Protect ALL fenced code blocks (```...```)
    # This must come before protecting inline code to handle nested cases
    content = re.sub(r'```[\s\S]*?```', protect_code, content)

    # Protect inline code (backticks)
    content = re.sub(r'`[^`]+`', protect_code, content)

    # Step 2: Fix all math blocks
    def fix_math_block(match):
        # Get the entire match including $$
        block = match.group(0)
        # Extract content between $$ markers
        inner = match.group(1)

        # Clean up the content:
        # 1. Strip leading/trailing whitespace
        # 2. Remove empty lines
        # 3. Keep actual content lines
        lines = inner.strip().split('\n')
        content_lines = [line.rstrip() for line in lines if line.strip()]

        if content_lines:
            cleaned = '\n'.join(content_lines)
            return f'$$\n{cleaned}\n$$'
        else:
            return '$$\n$$'

    # Match $$ ... $$ blocks (including multiline)
    # First pass: fix internal structure
    content = re.sub(r'\$\$(.*?)\$\$', fix_math_block, content, flags=re.DOTALL)

    # Step 3: Protect the formatted blocks and store them
    formatted_blocks = []

    def protect_formatted(match):
        formatted_blocks.append(match.group(0))
        return f"__MATHBLOCK_{len(formatted_blocks)-1}__"

    content = re.sub(r'\$\$\n.*?\n\$\$', protect_formatted, content, flags=re.DOTALL)

    # Step 4: Add proper spacing around the placeholders
    # Add blank line before placeholder if there isn't one
    content = re.sub(r'([^\n])\n(__MATHBLOCK_\d+__)', r'\1\n\n\2', content)
    # Add newline before placeholder if there isn't even that
    content = re.sub(r'([^\n])(__MATHBLOCK_\d+__)', r'\1\n\n\2', content)
    # Add blank line after placeholder if there isn't one (when followed by newline)
    content = re.sub(r'(__MATHBLOCK_\d+__)\n(?!\n)([^\n])', r'\1\n\n\2', content)
    # Add blank line after placeholder if there isn't even a newline
    content = re.sub(r'(__MATHBLOCK_\d+__)([^\n])', r'\1\n\n\2', content)

    # Step 5: Restore the formatted blocks
    for i, block in enumerate(formatted_blocks):
        content = content.replace(f"__MATHBLOCK_{i}__", block)

    # Step 5: Clean up excessive blank lines (4+ becomes 3)
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    # Step 6: Restore code blocks
    for i, block in enumerate(code_blocks):
        content = content.replace(f"__CODEBLOCK_{i}__", block)

    # Ensure file ends with newline
    if not content.endswith('\n'):
        content += '\n'

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        if output_path == file_path:
            print(f"Successfully formatted `{file_path}`.")
        else:
            print(f"Successfully formatted `{file_path}` and saved to `{output_path}`.")
    except Exception as e:
        print(f"Error writing to file `{output_path}`: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixes the formatting of $$...$$ LaTeX blocks in a file."
    )
    parser.add_argument("file_path", type=str, help="The path to the file to be formatted.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="The output file path. If not specified, overwrites the input file.",
        default=None
    )
    args = parser.parse_args()
    format_latex_blocks(args.file_path, args.output)