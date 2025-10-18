#!/usr/bin/env python3
"""
Add concise descriptions to glossary.md entries.

This script reads glossary.md and source documents to add one-sentence
descriptions (< 15 words) for each mathematical entry.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_glossary_entries(glossary_path: Path) -> List[Dict[str, str]]:
    """Extract all entries from glossary.md."""
    entries = []
    current_entry = None

    with open(glossary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # New entry starts with ### (but not #### which is subsection)
        if line.startswith('### ') and not line.startswith('#### '):
            if current_entry:
                entries.append(current_entry)

            current_entry = {
                'title': line.strip('# \n'),
                'line_number': i + 1,
                'type': None,
                'label': None,
                'tags': None,
                'source': None,
                'description': None,
            }
        elif current_entry and line.startswith('- **Type:**'):
            current_entry['type'] = line.split(':', 1)[1].strip()
        elif current_entry and line.startswith('- **Label:**'):
            current_entry['label'] = line.split(':', 1)[1].strip().strip('`')
        elif current_entry and line.startswith('- **Tags:**'):
            current_entry['tags'] = line.split(':', 1)[1].strip()
        elif current_entry and line.startswith('- **Source:**'):
            current_entry['source'] = line.split(':', 1)[1].strip()
        elif current_entry and line.startswith('- **Description:**'):
            current_entry['description'] = line.split(':', 1)[1].strip()

    if current_entry:
        entries.append(current_entry)

    return entries


def extract_math_statement_from_source(source_file: Path, label: str) -> str:
    """Extract the mathematical statement for a given label from source."""
    if not source_file.exists():
        return ""

    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Look for the label
    pattern = rf':label:\s*{re.escape(label)}'
    match = re.search(pattern, content)

    if not match:
        return ""

    # Extract a few lines after the label to get the statement
    start = match.end()
    # Find the next 200 characters after label
    snippet = content[start:start+300]

    # Clean up the snippet
    snippet = re.sub(r'\$\$[^\$]+\$\$', '', snippet)  # Remove display math
    snippet = re.sub(r'\$[^\$]+\$', '', snippet)  # Remove inline math
    snippet = re.sub(r'\s+', ' ', snippet)  # Normalize whitespace
    snippet = snippet.strip()

    # Return first sentence
    if '.' in snippet:
        return snippet.split('.')[0] + '.'

    return snippet[:80] if len(snippet) > 80 else snippet


def generate_description(entry: Dict[str, str], docs_root: Path) -> str:
    """Generate a concise description for an entry."""
    title = entry['title']
    entry_type = entry['type']

    # Try to extract from source if label exists
    if entry['label'] and entry['label'] != 'unlabeled':
        # Parse source to get file path
        source = entry['source']
        if source:
            # Extract file name from markdown link
            match = re.search(r'\[([^\]]+\.md)\]', source)
            if match:
                filename = match.group(1)
                # Determine which chapter
                if '1_euclidean_gas' in source or 'euclidean' in source.lower():
                    source_file = docs_root / 'source' / '1_euclidean_gas' / filename
                elif '2_geometric_gas' in source or 'geometric' in source.lower():
                    source_file = docs_root / 'source' / '2_geometric_gas' / filename
                else:
                    source_file = docs_root / 'source' / filename

                if source_file.exists():
                    desc = extract_math_statement_from_source(source_file, entry['label'])
                    if desc and len(desc.split()) < 15:
                        return desc

    # Fallback: generate from title and type
    type_templates = {
        'Definition': f"Defines {title.lower()}",
        'Theorem': f"Proves {title.lower()}",
        'Lemma': f"Establishes {title.lower()}",
        'Proposition': f"Shows {title.lower()}",
        'Corollary': f"Derives {title.lower()}",
        'Axiom': f"Assumes {title.lower()}",
        'Assumption': f"Assumes {title.lower()}",
        'Remark': f"Notes on {title.lower()}",
        'Algorithm': f"Describes {title.lower()} procedure",
    }

    return type_templates.get(entry_type, f"{title}")


def main():
    """Main function to add descriptions to glossary."""
    project_root = Path(__file__).parent.parent.parent
    glossary_path = project_root / 'docs' / 'glossary.md'
    docs_root = project_root / 'docs'

    print(f"Reading glossary from: {glossary_path}")
    entries = extract_glossary_entries(glossary_path)
    print(f"Found {len(entries)} entries")

    # Generate descriptions
    for i, entry in enumerate(entries):
        if not entry.get('description'):
            desc = generate_description(entry, docs_root)
            entry['description'] = desc
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(entries)} entries...")

    print(f"Generated descriptions for all entries")

    # Write updated glossary
    # We'll output to a new file for safety
    output_path = project_root / 'docs' / 'glossary_with_descriptions.md'

    with open(glossary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Insert descriptions after Source line
    new_lines = []
    entry_idx = 0

    for i, line in enumerate(lines):
        new_lines.append(line)

        # Check if this is a Source line and we need to add description
        if line.startswith('- **Source:**') and entry_idx < len(entries):
            entry = entries[entry_idx]
            # Verify this is the right entry
            if entry['line_number'] <= i + 1:
                if entry.get('description') and '- **Description:**' not in ''.join(lines[max(0, i-5):i+2]):
                    desc_line = f"- **Description:** {entry['description']}\n"
                    new_lines.append(desc_line)
                entry_idx += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"Updated glossary written to: {output_path}")
    print("Please review and then replace glossary.md with this file.")


if __name__ == '__main__':
    main()
