#!/usr/bin/python3.13
"""Fix $$ block formatting in the algorithmic.md file."""
import re
import sys
import traceback

outpath = "/tmp/fix_dd_output.txt"
out = open(outpath, "w")

try:
    filepath = "/home/guillem/fragile/docs/source/2_hypostructure/09_mathematical/05_algorithmic.md"

    with open(filepath, "r") as f:
        lines = f.readlines()

    out.write("Total lines in file: %d\n" % len(lines))

    lines_orig = list(lines)  # save original for reporting

    # Find all $$ lines (standalone, possibly indented)
    dd_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "$$" or (stripped.startswith("$$") and stripped.endswith(")") and "(" in stripped[2:]):
            dd_indices.append(i)

    out.write("Total standalone $$ lines found: %d\n" % len(dd_indices))

    # Pair them as opening/closing
    violations_r1 = []  # opening $$ without blank line before
    violations_r2 = []  # closing $$ with blank line before

    for idx in range(0, len(dd_indices), 2):
        opening = dd_indices[idx]
        if idx + 1 < len(dd_indices):
            closing = dd_indices[idx + 1]
        else:
            out.write("WARNING: Unpaired $$ at line %d\n" % (opening + 1))
            continue

        # Rule 1: Opening $$ must have blank line before
        if opening > 0 and lines[opening - 1].strip() != "":
            violations_r1.append(opening)

        # Rule 2: Closing $$ must NOT have blank line before
        if closing > 0 and lines[closing - 1].strip() == "":
            violations_r2.append(closing)

    out.write("Rule 1 violations (opening $$ needs blank line before): %d\n" % len(violations_r1))
    for v in violations_r1:
        out.write("  Line %d: prev='%s'\n" % (v + 1, lines_orig[v - 1].rstrip()[:80]))

    out.write("Rule 2 violations (closing $$ has blank line before): %d\n" % len(violations_r2))
    for v in violations_r2:
        out.write("  Line %d\n" % (v + 1))

    # Fix violations
    # Process in reverse order so line numbers don't shift
    # Rule 2: remove blank line before closing $$
    for v in reversed(violations_r2):
        del lines[v - 1]

    # Re-find $$ lines after Rule 2 fixes
    dd_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "$$" or (stripped.startswith("$$") and stripped.endswith(")") and "(" in stripped[2:]):
            dd_indices.append(i)

    # Re-pair and fix Rule 1
    insertions = []
    for idx in range(0, len(dd_indices), 2):
        opening = dd_indices[idx]
        if opening > 0 and lines[opening - 1].strip() != "":
            insertions.append(opening)

    out.write("Rule 1 insertions needed after Rule 2 fixes: %d\n" % len(insertions))

    # Insert blank lines in reverse order
    for v in reversed(insertions):
        lines.insert(v, "\n")

    with open(filepath, "w") as f:
        f.writelines(lines)

    out.write("Done. File written successfully.\n")

except Exception as e:
    out.write("ERROR: %s\n" % str(e))
    out.write(traceback.format_exc())

out.close()
