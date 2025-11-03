# Improvement Mode Data Preservation - Implementation Summary

## Problem Statement

The original improvement mode in `dspy_pipeline.py` had the following issues:

1. **Lost original data on failure**: When improvement failed, the original extraction was replaced with an error report
2. **No fallback mechanism**: Critical errors resulted in no data being saved
3. **No tracking of failed attempts**: Users couldn't see history of improvement attempts

## Solution

Implemented a robust data preservation strategy that **ALWAYS** keeps the original data as a fallback.

## Changes Made

### 1. **Initialize `existing_data` at Loop Start** (Line 221)

```python
output_file = output_dir / f"chapter_{i}.json"
existing_data = None  # Initialize for later checks
```

**Why**: Ensures variable is defined for conditional checks later in the code.

### 2. **Safe Data Loading with Fallback** (Lines 228-236)

```python
# Load existing extraction (we'll preserve this as fallback)
try:
    with open(output_file, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
except Exception as e:
    print(f"  ✗ Failed to load existing data: {e}")
    print(f"  → Switching to EXTRACT mode")
    # Treat as new file if we can't load existing data
    existing_data = None
```

**Why**: If we can't load existing data (corrupted file, wrong format), gracefully fall back to extract mode.

### 3. **Data Preservation on Conversion Failure** (Lines 272-287)

```python
else:
    # Improvement conversion failed - PRESERVE ORIGINAL DATA
    save_data = existing_data.copy()

    # Add metadata about failed improvement attempt
    if "_improvement_attempts" not in save_data:
        save_data["_improvement_attempts"] = []

    save_data["_improvement_attempts"].append({
        "status": "failed",
        "errors": errors,
        "summary": improvement_result.get_summary() if improvement_result else "No result"
    })

    if verbose:
        print(f"  ⚠ Improvement failed - preserving original data")
```

**Why**: When the improvement workflow returns `None` (conversion failed), we keep the original data and add metadata about the failed attempt.

### 4. **Data Preservation on Critical Error** (Lines 312-335)

```python
except Exception as e:
    # Critical failure in improvement workflow - PRESERVE ORIGINAL DATA
    print(f"  ✗ Critical error in improvement workflow: {e}")
    if verbose:
        import traceback
        traceback.print_exc()

    # Save original data with error metadata
    save_data = existing_data.copy()

    if "_improvement_attempts" not in save_data:
        save_data["_improvement_attempts"] = []

    save_data["_improvement_attempts"].append({
        "status": "critical_error",
        "error": str(e),
        "traceback": traceback.format_exc() if verbose else None
    })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Original data preserved despite error")
    print()
```

**Why**: Even when the improvement workflow crashes completely, we save the original data with error details.

### 5. **Fallback to Extract Mode** (Lines 341-342)

```python
# EXTRACT WORKFLOW (for new files OR when existing data couldn't be loaded)
if not output_file.exists() or (output_file.exists() and existing_data is None):
```

**Why**: If we couldn't load existing data, treat it as a new file and run extraction.

## Data Preservation Guarantees

The improved implementation provides the following guarantees:

### ✅ **Original Data Never Lost**

- On improvement failure → Original data preserved with `_improvement_attempts` metadata
- On critical error → Original data preserved with error details
- On load failure → Falls back to fresh extraction

### ✅ **Full Change Tracking**

**On Success:**
```json
{
  ...improved data...,
  "_improvement_metadata": {
    "changes": [
      {
        "entity_type": "definition",
        "label": "def-new",
        "operation": "ADD",
        "reason": "New entity found",
        "new_data": {...}
      }
    ],
    "summary": {
      "entities_added": 1,
      "entities_modified": 2,
      "entities_deleted": 0,
      "entities_unchanged": 10
    }
  }
}
```

**On Failure:**
```json
{
  ...original data preserved...,
  "_improvement_attempts": [
    {
      "status": "failed",
      "errors": ["Error 1", "Error 2"],
      "summary": "Improvement Summary: Added: 0, Modified: 0..."
    }
  ]
}
```

**On Critical Error:**
```json
{
  ...original data preserved...,
  "_improvement_attempts": [
    {
      "status": "critical_error",
      "error": "Exception message",
      "traceback": "Full traceback..."
    }
  ]
}
```

### ✅ **Incremental Improvement History**

The `_improvement_attempts` array accumulates across multiple runs:
- Each failed improvement adds a new entry
- Users can see the full history of improvement attempts
- Successful improvements replace this with `_improvement_metadata`

## Workflow Logic

```
┌─────────────────────────────────────────────────────────────┐
│ Process Chapter                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Does chapter_{N}.json exist?                              │
│                                                             │
│  YES ──> Try to load existing data                          │
│          │                                                  │
│          ├─ Load SUCCESS ──> Run improve_chapter()          │
│          │                    │                             │
│          │                    ├─ Improvement SUCCESS        │
│          │                    │  └─> Save improved data     │
│          │                    │      + _improvement_metadata│
│          │                    │                             │
│          │                    ├─ Conversion FAILED          │
│          │                    │  └─> Save ORIGINAL data     │
│          │                    │      + _improvement_attempts│
│          │                    │                             │
│          │                    └─ Critical ERROR             │
│          │                       └─> Save ORIGINAL data     │
│          │                           + _improvement_attempts│
│          │                                                  │
│          └─ Load FAILED ──> Fall through to EXTRACT         │
│                                                             │
│  NO ───> Run extract_chapter()                              │
│          └─> Save fresh extraction                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Benefits

1. **Data Safety**: Original extractions are never lost
2. **Transparency**: Full history of improvement attempts tracked
3. **Robustness**: Graceful degradation at every failure point
4. **Debuggability**: Detailed error information preserved
5. **Incremental Progress**: Can retry improvements without losing work

## Testing

Run the test script to verify behavior:

```bash
python3 test_improvement_preservation.py
```

## Usage Example

```bash
# First run: Fresh extraction
python -m mathster.parsing.dspy_pipeline document.md

# Second run: Improvement (preserves data on failure)
python -m mathster.parsing.dspy_pipeline document.md

# Third run: Another improvement attempt (history tracked)
python -m mathster.parsing.dspy_pipeline document.md
```

Each run either:
- ✅ Succeeds → Saves improved data with change metadata
- ⚠️ Fails → Preserves original data with attempt metadata
- ❌ Crashes → Preserves original data with error details

**Original data is NEVER lost.**
