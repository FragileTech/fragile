#!/usr/bin/env python
"""
SMoC (Standard Model of Cognition) Analysis of Fractal Gas Simulation.

This script:
1. Runs a Fractal Gas simulation in a potential well
2. Applies the SMoC pipeline to extract particle-like masses
3. Computes effective mass plateaus for validation
4. Generates diagnostic plots

Usage:
    python src/experiments/smoc_fractal_gas_analysis.py
    python src/experiments/smoc_fractal_gas_analysis.py --n-walkers 500 --n-steps 500
    python src/experiments/smoc_fractal_gas_analysis.py --output-dir outputs/smoc_test
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from fragile.fractalai.qft.simulation import (
    OperatorConfig,
    PotentialWellConfig,
    RunConfig,
    run_simulation,
)
from fragile.fractalai.qft.smoc_pipeline import (
    AggregatedCorrelator,
    ChannelProjector,
    CorrelatorComputer,
    CorrelatorConfig,
    MassExtractionConfig,
    MassExtractor,
    ProjectorConfig,
    aggregate_correlators,
)


@dataclass
class SMoCAnalysisConfig:
    """Configuration for SMoC analysis."""
    channels: tuple[str, ...] = ("scalar", "pion", "rho", "sigma")
    warmup_fraction: float = 0.1
    use_connected: bool = True
    normalize_correlators: bool = True
    min_window_length: int = 5
    min_t_start: int = 3
    max_t_end_fraction: float = 0.5  # Use first half of data for fitting


def _json_safe(value: Any) -> Any:
    """Make value JSON serializable."""
    if isinstance(value, np.ndarray | torch.Tensor):
        if value.ndim == 0:
            return float(value)
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        if value != value:  # NaN check
            return "nan"
    return value


def extract_phase_space_history(history: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract position, velocity, and alive tensors from RunHistory.
    
    Returns:
        positions: (n_steps, n_walkers, dims)
        velocities: (n_steps, n_walkers, dims)
        alive: (n_steps, n_walkers)
    """
    positions = history.x_final  # (n_recorded, n_walkers, dims)
    velocities = history.v_final  # (n_recorded, n_walkers, dims)
    alive = history.alive_mask  # (n_steps, n_walkers)
    
    # Align dimensions - alive_mask may be one step shorter
    min_steps = min(positions.shape[0], alive.shape[0])
    positions = positions[:min_steps]
    velocities = velocities[:min_steps]
    alive = alive[:min_steps]
    
    return positions, velocities, alive


def build_internal_state(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Build internal state representation from phase space.
    
    Combines position and velocity into a single internal state vector
    suitable for SMoC channel projection.
    
    Args:
        positions: (n_steps, n_walkers, dims)
        velocities: (n_steps, n_walkers, dims)
        normalize: Whether to normalize internal states
        
    Returns:
        internal: (n_steps, n_walkers, 2*dims)
    """
    # Concatenate position and velocity
    internal = torch.cat([positions, velocities], dim=-1)
    
    if normalize:
        # Normalize each walker's internal state
        norms = torch.linalg.vector_norm(internal, dim=-1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        internal = internal / norms
    
    return internal


def compute_effective_mass_series(
    correlator: torch.Tensor,
    delta_t: float = 1.0,
) -> torch.Tensor:
    """
    Compute effective mass: m_eff(t) = ln(C(t)/C(t+1)) / Δt
    
    This should plateau at the true mass.
    """
    # Ensure positive correlator
    corr_safe = torch.clamp(correlator, min=1e-12)
    
    # Compute ratio C(t)/C(t+1)
    ratio = corr_safe[:-1] / corr_safe[1:]
    
    # Take log and divide by delta_t
    m_eff = torch.log(ratio) / delta_t
    
    return m_eff


def run_smoc_analysis(
    history: Any,
    config: SMoCAnalysisConfig,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run SMoC analysis on Fractal Gas history.
    
    Args:
        history: RunHistory from Fractal Gas simulation
        config: SMoC analysis configuration
        verbose: Print progress messages
        
    Returns:
        Dictionary with all analysis results
    """
    results: dict[str, Any] = {
        "config": asdict(config),
        "channels": {},
    }
    
    # Extract phase space history
    if verbose:
        print("Extracting phase space history...")
    positions, velocities, alive = extract_phase_space_history(history)
    
    n_steps, n_walkers, dims = positions.shape
    if verbose:
        print(f"  Shape: {n_steps} steps × {n_walkers} walkers × {dims} dims")
    
    # Apply warmup (discard initial transient)
    warmup_steps = int(n_steps * config.warmup_fraction)
    positions = positions[warmup_steps:]
    velocities = velocities[warmup_steps:]
    alive = alive[warmup_steps:]
    n_steps = positions.shape[0]
    
    if verbose:
        print(f"  After warmup: {n_steps} steps")
    
    # Build internal state representation
    if verbose:
        print("Building internal state representation...")
    internal = build_internal_state(positions, velocities, normalize=True)
    internal_dim = internal.shape[-1]
    
    if verbose:
        print(f"  Internal dimension: {internal_dim}")
    
    # Add batch dimension (treat each walker trajectory as independent)
    # Reshape: (n_steps, n_walkers, dim) -> (n_walkers, n_steps, 1, dim)
    # The "grid" dimension is 1 since each walker is its own "universe"
    history_tensor = internal.permute(1, 0, 2).unsqueeze(2)  # (n_walkers, n_steps, 1, dim)
    
    if verbose:
        print(f"  History tensor shape: {history_tensor.shape}")
    
    # Build projector
    proj_config = ProjectorConfig(
        internal_dim=internal_dim,
        device=str(positions.device),
        dtype=positions.dtype,
    )
    projector = ChannelProjector(proj_config)
    
    # Build correlator computer
    corr_config = CorrelatorConfig(
        use_connected=config.use_connected,
        normalize=config.normalize_correlators,
    )
    correlator_computer = CorrelatorComputer(corr_config)
    
    # Build mass extractor
    max_t_end = int(n_steps * config.max_t_end_fraction)
    mass_config = MassExtractionConfig(
        min_window_length=config.min_window_length,
        min_t_start=config.min_t_start,
        max_t_end=max_t_end,
    )
    extractor = MassExtractor(mass_config)
    
    # Process each channel
    for channel in config.channels:
        if verbose:
            print(f"\nProcessing channel: {channel}")
        
        try:
            # Project onto channel
            field = projector.project(history_tensor, channel)  # (n_walkers, n_steps, 1)
            field = field.squeeze(-1)  # (n_walkers, n_steps)
            
            if verbose:
                print(f"  Field shape: {field.shape}")
            
            # Mask by alive status
            alive_transposed = alive.permute(1, 0)  # (n_walkers, n_steps)
            field = field * alive_transposed.float()
            
            # Compute correlators for each walker (as independent samples)
            # Use FFT method
            correlators = correlator_computer.compute_autocorrelation_fft(field)
            
            if verbose:
                print(f"  Correlators shape: {correlators.shape}")
            
            # Aggregate across walkers
            agg = aggregate_correlators(correlators, keep_raw=False)
            
            # Compute effective mass series
            m_eff = compute_effective_mass_series(agg.mean)
            m_eff_err = compute_effective_mass_series(agg.mean + agg.std_err) - m_eff
            
            # Extract mass using AIC
            mass_result = extractor.extract_mass(agg)
            
            if verbose:
                m = mass_result["mass"]
                err = mass_result["mass_error"]
                n_win = mass_result["n_valid_windows"]
                print(f"  Mass: {m:.4f} ± {err:.4f} ({n_win} valid windows)")
            
            # Store results
            results["channels"][channel] = {
                "correlator_mean": agg.mean.cpu().numpy(),
                "correlator_std": agg.std.cpu().numpy(),
                "correlator_err": agg.std_err.cpu().numpy(),
                "effective_mass": m_eff.cpu().numpy(),
                "effective_mass_err": m_eff_err.abs().cpu().numpy(),
                "mass": mass_result["mass"],
                "mass_error": mass_result["mass_error"],
                "n_valid_windows": mass_result["n_valid_windows"],
                "best_window": mass_result.get("best_window"),
            }
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            results["channels"][channel] = {"error": str(e)}
    
    # Summary statistics
    results["summary"] = {
        "n_steps_analyzed": n_steps,
        "n_walkers": n_walkers,
        "dims": dims,
        "internal_dim": internal_dim,
        "warmup_fraction": config.warmup_fraction,
    }
    
    # Check mass ordering (physics validation)
    masses = {ch: results["channels"][ch].get("mass", float("inf")) 
              for ch in config.channels if "error" not in results["channels"].get(ch, {})}
    
    if "pion" in masses and "rho" in masses:
        results["validation"] = {
            "pion_lighter_than_rho": masses["pion"] < masses["rho"],
            "mass_ratio_rho_pion": masses["rho"] / masses["pion"] if masses["pion"] > 0 else float("inf"),
        }
    
    return results


def plot_smoc_results(
    results: dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> None:
    """Generate diagnostic plots for SMoC analysis."""
    
    channels = results["channels"]
    n_channels = len([ch for ch in channels if "error" not in channels[ch]])
    
    if n_channels == 0:
        print("No valid channels to plot.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 4 * n_channels))
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    for channel, data in channels.items():
        if "error" in data:
            continue
        
        ax_corr = axes[plot_idx, 0]
        ax_meff = axes[plot_idx, 1]
        
        # Plot correlator
        corr = data["correlator_mean"]
        corr_err = data["correlator_err"]
        t = np.arange(len(corr))
        
        ax_corr.errorbar(t, corr, yerr=corr_err, fmt='o-', markersize=3, 
                        capsize=2, label=f'{channel}')
        ax_corr.set_yscale('log')
        ax_corr.set_xlabel('Time lag τ')
        ax_corr.set_ylabel('C(τ)')
        ax_corr.set_title(f'{channel.capitalize()} Correlator')
        ax_corr.legend()
        ax_corr.grid(True, alpha=0.3)
        
        # Plot effective mass
        m_eff = data["effective_mass"]
        m_eff_err = data["effective_mass_err"]
        t_eff = np.arange(len(m_eff))
        
        # Filter out invalid values
        valid = np.isfinite(m_eff) & (m_eff > 0) & (m_eff < 10)
        
        if valid.sum() > 0:
            ax_meff.errorbar(t_eff[valid], m_eff[valid], yerr=m_eff_err[valid],
                           fmt='o', markersize=4, capsize=2, label='m_eff(t)')
        
        # Add horizontal line for extracted mass
        mass = data["mass"]
        mass_err = data["mass_error"]
        ax_meff.axhline(mass, color='red', linestyle='--', linewidth=2,
                       label=f'M = {mass:.4f} ± {mass_err:.4f}')
        ax_meff.axhspan(mass - mass_err, mass + mass_err, alpha=0.2, color='red')
        
        # Highlight best window if available
        best = data.get("best_window")
        if best:
            ax_meff.axvspan(best["t_start"], best["t_end"], alpha=0.1, color='green',
                          label=f'Best window [{best["t_start"]}, {best["t_end"]}]')
        
        ax_meff.set_xlabel('Time lag τ')
        ax_meff.set_ylabel('m_eff(τ)')
        ax_meff.set_title(f'{channel.capitalize()} Effective Mass Plateau')
        ax_meff.legend(loc='upper right')
        ax_meff.grid(True, alpha=0.3)
        
        # Set reasonable y-limits
        if valid.sum() > 0:
            m_eff_valid = m_eff[valid]
            ymin = max(0, np.percentile(m_eff_valid, 5) - 0.1)
            ymax = np.percentile(m_eff_valid, 95) + 0.1
            ax_meff.set_ylim(ymin, ymax)
        
        plot_idx += 1
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / f"{run_id}_smoc_analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {plot_path}")
    
    # Also create a mass spectrum plot
    fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 6))
    
    channel_names = []
    masses = []
    errors = []
    
    for channel, data in channels.items():
        if "error" not in data and data["mass"] > 0:
            channel_names.append(channel.capitalize())
            masses.append(data["mass"])
            errors.append(data["mass_error"])
    
    if len(masses) > 0:
        x = np.arange(len(channel_names))
        ax_spectrum.bar(x, masses, yerr=errors, capsize=5, color='steelblue', alpha=0.7)
        ax_spectrum.set_xticks(x)
        ax_spectrum.set_xticklabels(channel_names)
        ax_spectrum.set_ylabel('Extracted Mass')
        ax_spectrum.set_title('SMoC Mass Spectrum from Fractal Gas')
        ax_spectrum.grid(True, alpha=0.3, axis='y')
        
        # Add validation info
        validation = results.get("validation", {})
        if validation:
            text = []
            if "pion_lighter_than_rho" in validation:
                status = "✓" if validation["pion_lighter_than_rho"] else "✗"
                text.append(f"{status} M_π < M_ρ")
            if "mass_ratio_rho_pion" in validation:
                ratio = validation["mass_ratio_rho_pion"]
                if ratio < float("inf"):
                    text.append(f"M_ρ/M_π = {ratio:.2f}")
            if text:
                ax_spectrum.text(0.02, 0.98, "\n".join(text), transform=ax_spectrum.transAxes,
                               verticalalignment='top', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    spectrum_path = output_dir / f"{run_id}_mass_spectrum.png"
    fig_spectrum.savefig(spectrum_path, dpi=150, bbox_inches='tight')
    plt.close(fig_spectrum)
    print(f"Saved spectrum: {spectrum_path}")


def main():
    parser = argparse.ArgumentParser(description="SMoC Analysis of Fractal Gas")
    parser.add_argument("--n-walkers", type=int, default=200, help="Number of walkers")
    parser.add_argument("--n-steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--dims", type=int, default=3, help="Spatial dimensions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs/smoc_analysis",
                       help="Output directory")
    parser.add_argument("--warmup-fraction", type=float, default=0.1,
                       help="Fraction of steps to discard as warmup")
    parser.add_argument("--channels", type=str, nargs="+", 
                       default=["scalar", "pion", "rho", "sigma"],
                       help="Channels to analyze")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("SMoC Analysis of Fractal Gas Simulation")
    print("=" * 60)
    print(f"N walkers: {args.n_walkers}")
    print(f"N steps: {args.n_steps}")
    print(f"Dimensions: {args.dims}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print()
    
    # Configure simulation
    potential_cfg = PotentialWellConfig(
        dims=args.dims,
        alpha=0.1,
        bounds_extent=10.0,
    )
    
    operator_cfg = OperatorConfig()  # Use defaults
    
    run_cfg = RunConfig(
        N=args.n_walkers,
        n_steps=args.n_steps,
        record_every=1,
        seed=args.seed,
        device="cpu",
    )
    
    # Run simulation
    print("Phase 1: Running Fractal Gas simulation...")
    history, potential = run_simulation(potential_cfg, operator_cfg, run_cfg, show_progress=True)
    print(f"  Recorded {history.n_recorded} steps")
    print()
    
    # Run SMoC analysis
    print("Phase 2-6: Running SMoC analysis pipeline...")
    smoc_config = SMoCAnalysisConfig(
        channels=tuple(args.channels),
        warmup_fraction=args.warmup_fraction,
        use_connected=True,
        normalize_correlators=True,
    )
    
    results = run_smoc_analysis(history, smoc_config, verbose=True)
    print()
    
    # Generate plots
    print("Generating diagnostic plots...")
    plot_smoc_results(results, output_dir, run_id)
    print()
    
    # Save results
    results_path = output_dir / f"{run_id}_results.json"
    with open(results_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"Saved results: {results_path}")
    
    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Channel':<12} {'Mass':>10} {'Error':>10} {'Windows':>10}")
    print("-" * 44)
    for channel, data in results["channels"].items():
        if "error" in data:
            print(f"{channel:<12} {'ERROR':>10} {'-':>10} {'-':>10}")
        else:
            print(f"{channel:<12} {data['mass']:>10.4f} {data['mass_error']:>10.4f} {data['n_valid_windows']:>10}")
    
    validation = results.get("validation", {})
    if validation:
        print()
        print("Physics Validation:")
        if "pion_lighter_than_rho" in validation:
            status = "PASS ✓" if validation["pion_lighter_than_rho"] else "FAIL ✗"
            print(f"  M_π < M_ρ: {status}")
        if "mass_ratio_rho_pion" in validation:
            ratio = validation["mass_ratio_rho_pion"]
            if ratio < float("inf"):
                print(f"  M_ρ / M_π = {ratio:.3f}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
