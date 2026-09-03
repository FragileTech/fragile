# QFT Analysis Skill - Quick Reference

## Overview

The `/qft-analysis` skill automates QFT experiment analysis, calibration validation, and multi-run comparison with support for different mass scale anchors.

## Quick Start

### Basic Usage

```bash
# Analyze latest run with default Z boson anchor
/qft-analysis

# Analyze specific run
/qft-analysis --history-path outputs/sweep_nu1.00_vls1.00_history.pt

# Compare all anchors for one run
/qft-analysis --history-path outputs/my_run_history.pt --anchors all
```

### Multi-Run Comparison

```bash
# Compare parameter sweep (IMPORTANT: Quote glob patterns!)
/qft-analysis --history-path "outputs/sweep_nu*.pt" --anchors z

# Compare across multiple anchors
/qft-analysis --history-path "outputs/sweep_*.pt" --anchors electron,z,tau
```

### Custom Anchor

```bash
# Use Higgs mass as calibration anchor
/qft-analysis --anchors custom --anchor-mass 125.1 --anchor-label Higgs
```

### Advanced Options

```bash
# Skip particle computation (faster, field correlations only)
/qft-analysis --no-particles

# High-resolution particle analysis
/qft-analysis --particle-max-lag 200 --particle-fit-stop 50
```

## Command-Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--history-path` | Path or glob pattern to history files | Auto-detect latest | `outputs/run.pt` |
| `--anchors` | Comma-separated anchor list | `z` | `electron,z,tau` |
| `--anchor-mass` | Custom anchor mass (GeV) | - | `125.1` |
| `--anchor-label` | Custom anchor label | `Custom` | `Higgs` |
| `--no-particles` | Skip particle mass computation | false | - |
| `--particle-max-lag` | Override max lag points | Script default | `200` |
| `--particle-fit-stop` | Override fit range | Script default | `50` |

## Mass Scale Anchors

| Name | Mass (GeV) | Code | Use Case |
|------|------------|------|----------|
| Electron | 0.000511 | `electron` | Light fermion scale |
| Z Boson | 91.1876 | `z` | Electroweak scale (default) |
| Tau | 1.77686 | `tau` | Heavy lepton scale |
| Custom | User-specified | `custom` | Experimental (requires `--anchor-mass`) |

## Output

### Terminal Output

The skill displays 5 comprehensive tables:

1. **Run Summary** - Stability, alive %, QSD samples
2. **Gauge Coupling Validation** - α_em, sin²θ_W, α_s matching
3. **Algorithmic Parameters** - ε_c, ε_d, ρ, τ, ν, ε_F by anchor
4. **Particle Mass Predictions** - Meson, baryon, glueball masses with R²
5. **Correlation & Field Diagnostics** - ξ, R², Lyapunov ratio

Plus a comprehensive quality assessment with actionable recommendations.

### Generated Files

**Calibration outputs:**
```
outputs/qft_calibration/<run_id>_calibration.json
```

**Analysis outputs:**
```
outputs/fractal_gas_potential_well_analysis/<analysis_id>_metrics.json
```

**Comprehensive report:**
```
outputs/qft_analysis_reports/<timestamp>_qft_analysis_report.md
```

## Standard Model Reference Values

Used for validation:
- **α_em** = 1/137.036 ≈ 0.007297 (fine structure constant)
- **sin²θ_W** = 0.23121 (weak mixing angle at M_Z)
- **α_s(M_Z)** = 0.1179 (strong coupling at M_Z)

## Quality Thresholds

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| R² (fit) | > 0.8 | > 0.5 | > 0.3 | ≤ 0.3 |
| Lyapunov ratio | < 0.005 | < 0.01 | < 0.05 | ≥ 0.05 |
| Correlation ξ | > 0.1 | > 0.05 | > 0.01 | ≤ 0.01 |
| Alive % | 100% | > 95% | > 90% | ≤ 90% |

## Workflow

1. **Discovery** - Find and validate history files
2. **Calibration** - Run calibration for each (history, anchor) pair
3. **Analysis** - Compute particle masses and correlations
4. **Comparison** - Generate tables and quality metrics
5. **Reporting** - Create comprehensive markdown report

## Example Workflows

### Single Run, All Anchors

Useful for understanding how different mass scales affect calibrated parameters:

```bash
/qft-analysis --history-path outputs/my_run_history.pt --anchors all
```

**Output:** Compares electron, Z boson, and tau anchors side-by-side.

### Parameter Sweep Comparison

Useful for analyzing how algorithmic parameters affect results:

```bash
/qft-analysis --history-path "outputs/sweep_nu*.pt" --anchors z
```

**Output:** Multi-run comparison with Z boson anchor.

### Custom Calibration Point

Useful for experimental mass scales:

```bash
/qft-analysis --anchors custom --anchor-mass 125.1 --anchor-label Higgs
```

**Output:** Calibration using Higgs boson mass as reference.

### Quick Field Check

When you only need correlation diagnostics (faster):

```bash
/qft-analysis --no-particles
```

**Output:** Field correlations and QSD metrics only, no particle masses.

## Troubleshooting

### "History file not found"
- Check path is correct
- Use quotes around glob patterns: `"outputs/sweep_*.pt"`
- Omit `--history-path` to auto-detect latest

### "Calibration failed"
- Verify history file contains QSD data
- Try reducing `--qsd-iter` if convergence issues
- Check history file is valid RunHistory object

### "Analysis failed"
- Use `--no-particles` for faster field-only analysis
- Ensure simulation has sufficient data (N > 10, steps > 100)
- Add `--build-fractal-set` is implicit for glueball operator

### "Custom anchor requires --anchor-mass"
- Provide mass: `/qft-analysis --anchors custom --anchor-mass 125.1`
- Optionally add label: `--anchor-label Higgs`

## Tips

- **Quote glob patterns** in your shell: `"outputs/sweep_*.pt"`
- **Start with Z anchor** (default) - it's the standard electroweak scale
- **Use `all` for comparison** to see parameter scaling across anchors
- **Check reports directory** for comprehensive markdown output
- **Review recommendations** in the quality assessment for next steps

## Script Integration

The skill orchestrates these existing scripts:

- `src/experiments/calibrate_fractal_gas_qft.py` - Calibration
- `src/experiments/analyze_fractal_gas_qft.py` - Analysis
- `src/experiments/constants_check.py` - Reference constants

You can run these scripts directly if you need more fine-grained control, but the skill provides a streamlined workflow for common analysis tasks.
