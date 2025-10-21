"""Example demonstrating metric correction in scutoid curvature computation.

This script shows how to use the three metric correction modes:
1. 'none': Pure flat-space deficit angles
2. 'diagonal': Diagonal metric correction (cheap, O(N))
3. 'full': Full metric tensor correction (accurate, O(N·k))

The corrections bridge the gap between intrinsic curvature (from walker
configuration) and extrinsic curvature (from fitness landscape geometry).
"""

import numpy as np
from src.fragile.core.scutoids import create_scutoid_history


def demonstrate_metric_correction(history):
    """Demonstrate the three metric correction modes on a RunHistory.
    
    Args:
        history: RunHistory instance with recorded trajectory
    """
    print("=" * 70)
    print("Scutoid Metric Correction Demonstration")
    print("=" * 70)
    print()
    
    # Mode 1: No correction (flat-space)
    print("1. No Correction (flat-space deficit angles)")
    print("-" * 70)
    scutoid_flat = create_scutoid_history(history, metric_correction='none')
    scutoid_flat.build_tessellation()
    scutoid_flat.compute_ricci_scalars()
    
    ricci_flat = scutoid_flat.get_ricci_scalars()
    if ricci_flat is not None:
        valid_flat = ricci_flat[~np.isnan(ricci_flat)]
        if len(valid_flat) > 0:
            print(f"  Mean Ricci scalar: {np.mean(valid_flat):.6f}")
            print(f"  Std Ricci scalar:  {np.std(valid_flat):.6f}")
            print(f"  Min/Max:           {np.min(valid_flat):.6f} / {np.max(valid_flat):.6f}")
        else:
            print("  No valid Ricci scalars computed")
    print()
    
    # Mode 2: Diagonal correction
    print("2. Diagonal Correction (O(N) approximation)")
    print("-" * 70)
    scutoid_diag = create_scutoid_history(history, metric_correction='diagonal')
    scutoid_diag.build_tessellation()
    scutoid_diag.compute_ricci_scalars()
    
    ricci_diag = scutoid_diag.get_ricci_scalars()
    if ricci_diag is not None:
        valid_diag = ricci_diag[~np.isnan(ricci_diag)]
        if len(valid_diag) > 0:
            print(f"  Mean Ricci scalar: {np.mean(valid_diag):.6f}")
            print(f"  Std Ricci scalar:  {np.std(valid_diag):.6f}")
            print(f"  Min/Max:           {np.min(valid_diag):.6f} / {np.max(valid_diag):.6f}")
            
            # Compare to flat
            if ricci_flat is not None and len(valid_flat) > 0:
                correction = np.mean(valid_diag) - np.mean(valid_flat)
                print(f"  Correction:        {correction:+.6f} (mean)")
        else:
            print("  No valid Ricci scalars computed")
    print()
    
    # Mode 3: Full correction
    print("3. Full Correction (O(N·k) with neighbor gradients)")
    print("-" * 70)
    scutoid_full = create_scutoid_history(history, metric_correction='full')
    scutoid_full.build_tessellation()
    scutoid_full.compute_ricci_scalars()
    
    ricci_full = scutoid_full.get_ricci_scalars()
    if ricci_full is not None:
        valid_full = ricci_full[~np.isnan(ricci_full)]
        if len(valid_full) > 0:
            print(f"  Mean Ricci scalar: {np.mean(valid_full):.6f}")
            print(f"  Std Ricci scalar:  {np.std(valid_full):.6f}")
            print(f"  Min/Max:           {np.min(valid_full):.6f} / {np.max(valid_full):.6f}")
            
            # Compare to flat
            if ricci_flat is not None and len(valid_flat) > 0:
                correction = np.mean(valid_full) - np.mean(valid_flat)
                print(f"  Correction:        {correction:+.6f} (mean)")
        else:
            print("  No valid Ricci scalars computed")
    print()
    
    print("=" * 70)
    print("Physical Interpretation:")
    print("=" * 70)
    print("• Flat-space: Intrinsic curvature from walker configuration")
    print("• Diagonal:   Adds local scale corrections (cheap approximation)")
    print("• Full:       Adds full metric tensor effects (more accurate)")
    print()
    print("At equilibrium, corrected values should approximate the true Ricci")
    print("scalar of the emergent geometry g = H + ε_Σ I (Theorem 5.4.1)")
    print()


if __name__ == "__main__":
    # Example usage
    print("\nThis script demonstrates metric correction usage.")
    print("To use with your own RunHistory:")
    print()
    print("  from fragile.core.history import RunHistory")
    print("  history = RunHistory.load('your_experiment.pt')")
    print("  demonstrate_metric_correction(history)")
    print()
