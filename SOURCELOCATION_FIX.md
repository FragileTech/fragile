# SourceLocation Validation Fix - Unnumbered Sections Support

## Problem

The DSPy extraction pipeline was failing with this error:

```
ValueError: Cannot extract section from {file_path} at {line_range}.
File must contain numbered section headers before the target line.
```

This occurred when extracting:
- **Chapter 0/Preambles** (no headers before line 1)
- **Unnumbered sections** like "## Introduction"
- **Any content** appearing before the first numbered section

## Root Cause

The `SourceLocation` validator in `article_system.py` was **too strict**:

1. It called `extract_section_from_markdown()` which **correctly handles** unnumbered sections by returning `(None, "Introduction")`
2. But then it **rejected** `computed_section is None` at line 413
3. This created an inconsistency: the extraction function supported unnumbered sections, but the validator rejected them

### The Problematic Code (Before)

```python
# Lines 410-421 in article_system.py
computed_section, computed_section_name = extract_section_from_markdown(
    self.file_path, self.line_range
)
if computed_section is None:
    raise ValueError(
        f"Cannot extract section from {self.file_path} at {self.line_range}. "
        "File must contain numbered section headers before the target line."
    )
if computed_section_name is None:
    raise ValueError(
        f"Cannot extract section_name from {self.file_path} at {self.line_range}. "
        "File must contain section headers with names before the target line."
    )
```

This logic was **wrong** because:
- It required **numbered** sections (like "1.2.3") for ALL entities
- But `extract_section_from_markdown()` already handles unnumbered sections correctly
- Preambles and unnumbered "Introduction" sections are **valid** in markdown

## Solution

Modified the validator to be **permissive** for unnumbered sections while maintaining validation for malformed headers.

### The Fixed Code (After)

```python
# Lines 410-426 in article_system.py
computed_section, computed_section_name = extract_section_from_markdown(
    self.file_path, self.line_range
)

# Allow None for section number (unnumbered sections like "Introduction")
# But require section_name if there's any header before the target line
if computed_section is None and computed_section_name is None:
    # Special case: no headers at all before target (e.g., preamble/chapter 0)
    # This is valid for document metadata
    pass  # Allow both to be None - will use object.__setattr__ below
elif computed_section_name is None:
    # Header exists but couldn't parse name - this is an actual error
    raise ValueError(
        f"Cannot extract section_name from {self.file_path} at {self.line_range}. "
        "File contains unparseable section headers before the target line."
    )
# Note: computed_section can be None for unnumbered sections - this is valid
```

### The Logic

The new validation allows three valid cases:

1. **Both None** (preamble/chapter 0):
   - `computed_section = None`
   - `computed_section_name = None`
   - Example: Content at lines 1-3 with no headers before it

2. **section=None, section_name=valid** (unnumbered section):
   - `computed_section = None`
   - `computed_section_name = "Introduction"`
   - Example: Content after "## Introduction" header

3. **Both valid** (numbered section):
   - `computed_section = "1.2.3"`
   - `computed_section_name = "Mathematical Framework"`
   - Example: Content after "## 1.2.3 Mathematical Framework"

The validator **rejects** only truly malformed cases:
- Headers exist but couldn't be parsed (section_name=None when a header exists)

## What This Fixes

### ✅ Extraction Now Works For

1. **Chapter 0/Preambles**
   ```markdown
   # Document Title


   Some preamble content here.
   ```
   - Lines 1-3 have no headers before them
   - Now creates: `section=None, section_name=None`

2. **Unnumbered Sections**
   ```markdown
   ## Introduction

   This is the introduction section.
   ```
   - Lines after "## Introduction"
   - Now creates: `section=None, section_name="Introduction"`

3. **Numbered Sections** (already worked, still works)
   ```markdown
   ## 1. First Chapter

   ### 1.1. Subsection

   Content here.
   ```
   - Lines after headers with numbers
   - Creates: `section="1.1", section_name="Subsection"`

### ✅ DSPy Pipeline Impact

The DSPy pipeline (`extract_workflow.py`) creates section labels like:
- `section-preamble` for chapter 0
- `section-introduction` for unnumbered sections
- `section-1-mathematical-framework` for numbered sections

All of these now **validate correctly** with SourceLocation.

## Testing

### Test Results

```
Test 1: Chapter 0/Preamble (no headers before target)
✓ SUCCESS: Preamble SourceLocation created!
  - section: None (None is now allowed)
  - section_name: None (None is now allowed)

Test 2: Unnumbered '## Introduction' section
✓ SUCCESS: Unnumbered section SourceLocation created!
  - section: None (None for unnumbered)
  - section_name: 'Introduction'

Test 3: Numbered section (existing behavior maintained)
✓ SUCCESS: Numbered section SourceLocation created!
  - section: '1.1'
  - section_name: 'Subsection'
```

### Verification Commands

```bash
# Verify syntax
python3 -m py_compile src/mathster/core/article_system.py

# Test imports
python3 -c "from mathster.core.article_system import SourceLocation; print('✓ OK')"

# Run DSPy pipeline (should now work for all documents)
python -m mathster.parsing.dspy_pipeline docs/source/.../document.md
```

## Impact

### Documents Now Supported

- ✅ `07_mean_field.md` chapter 0 (was failing)
- ✅ All documents with preambles
- ✅ All documents with unnumbered "Introduction" sections
- ✅ All existing numbered section documents (unchanged)

### Breaking Changes

**None** - This change only makes validation **more permissive**:
- Existing valid SourceLocations still work
- Previously invalid (but correct) cases now work
- No API changes

### File Modified

- `src/mathster/core/article_system.py` (lines 410-426)

## Summary

The fix makes `SourceLocation` validation consistent with markdown document structure:
- **Preambles are valid** (no headers before content)
- **Unnumbered sections are valid** (e.g., "## Introduction")
- **Numbered sections still work** (e.g., "## 1.2 Framework")

This allows the DSPy extraction pipeline to process **all** chapters in mathematical documents, not just those with numbered sections.

**Result**: The error "File must contain numbered section headers before the target line" is now resolved.
