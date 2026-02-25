"""Chirality autocorrelation for electroweak fermion mass extraction.

Classifies walkers into four types at each cloning step based on their
interaction with the cloning operator:

  - Delta (D):           will_clone[i] = True. Gets teleported, phases destroyed.
  - Strong resister (SR): Not cloning, but is the target of at least one delta.
                          Local neighborhood disrupted, phase coherence broken.
  - Weak resister (WR):  Not cloning, not targeted, but has a fitter peer.
                          Phases untouched.
  - Persister (P):       Not cloning, not targeted, no fitter peer.
                          Phases untouched.

The chirality label groups these into left-handed (L = D + SR, phases altered
by cloning) and right-handed (R = WR + P, invisible to cloning). This mirrors
the V-A structure of the weak interaction: the cloning operator (W boson)
couples exclusively to the left-handed sector.

The fermion mass is extracted from the chirality autocorrelation:

    C_χ(τ) = (1/N) Σ_i ⟨χ_i(n) χ_i(n+τ)⟩_n  ~  exp(-m_f τ)

where χ_i = +1 for L, -1 for R. Fast role-switching (rapid decay) = heavy
fermion; slow switching = light fermion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from fragile.physics.fractal_gas.history import RunHistory


# ---------------------------------------------------------------------------
# Walker classification
# ---------------------------------------------------------------------------


@dataclass
class WalkerClassification:
    """Per-frame walker classification into four types.

    All masks are [T, N] boolean tensors. Each walker at each frame
    belongs to exactly one type.
    """

    delta: Tensor          # will clone (teleported, phases destroyed)
    strong_resister: Tensor  # target of at least one delta
    weak_resister: Tensor    # not targeted, has fitter peer
    persister: Tensor        # not targeted, no fitter peer

    # Derived chirality label: +1 for L (D+SR), -1 for R (WR+P)
    chi: Tensor              # [T, N] float

    # Per-frame counts for diagnostics
    n_delta: Tensor          # [T]
    n_strong_resister: Tensor  # [T]
    n_weak_resister: Tensor    # [T]
    n_persister: Tensor        # [T]

    @property
    def left_handed(self) -> Tensor:
        """Mask for left-handed walkers (D + SR)."""
        return self.delta | self.strong_resister

    @property
    def right_handed(self) -> Tensor:
        """Mask for right-handed walkers (WR + P)."""
        return self.weak_resister | self.persister

    @property
    def n_frames(self) -> int:
        return self.chi.shape[0]

    @property
    def n_walkers(self) -> int:
        return self.chi.shape[1]


def classify_walkers(
    will_clone: Tensor,
    companions_clone: Tensor,
    fitness: Tensor,
    alive: Tensor | None = None,
) -> WalkerClassification:
    """Classify walkers into delta/strong_resister/weak_resister/persister.

    Args:
        will_clone: Boolean cloning mask [T, N].
        companions_clone: Companion indices for cloning [T, N] (long).
        fitness: Fitness values [T, N].
        alive: Optional alive mask [T, N]. If None, all walkers assumed alive.

    Returns:
        WalkerClassification with all masks and chirality labels.
    """
    T, N = will_clone.shape
    device = will_clone.device

    if alive is None:
        alive = torch.ones(T, N, device=device, dtype=torch.bool)
    alive = alive.bool()
    will_clone = will_clone.bool()

    # Delta: will clone
    delta = will_clone & alive

    # Strong resister: at least one delta targets this walker
    # Build target mask: for each frame, scatter True to companion indices of deltas
    is_target = torch.zeros(T, N, device=device, dtype=torch.bool)
    for t in range(T):
        delta_mask_t = delta[t]
        if delta_mask_t.any():
            targets = companions_clone[t, delta_mask_t]
            # Clamp to valid range
            targets = targets.clamp(0, N - 1)
            is_target[t].scatter_(0, targets, True)

    strong_resister = ~will_clone & alive & is_target

    # Weak resister vs persister: does the walker have a fitter peer?
    # A walker's companion is its peer; if companion has higher fitness, walker is a resister
    companion_fitness = torch.gather(fitness, 1, companions_clone.clamp(0, N - 1))
    has_fitter_peer = alive & (companion_fitness > fitness)

    weak_resister = ~will_clone & alive & ~is_target & has_fitter_peer
    persister = ~will_clone & alive & ~is_target & ~has_fitter_peer

    # Chirality: +1 for L (delta + strong resister), -1 for R (weak resister + persister)
    chi = torch.where(delta | strong_resister, 1.0, -1.0)
    # Dead walkers get 0
    chi = torch.where(alive, chi, torch.zeros_like(chi))

    return WalkerClassification(
        delta=delta,
        strong_resister=strong_resister,
        weak_resister=weak_resister,
        persister=persister,
        chi=chi,
        n_delta=delta.sum(dim=1),
        n_strong_resister=strong_resister.sum(dim=1),
        n_weak_resister=weak_resister.sum(dim=1),
        n_persister=persister.sum(dim=1),
    )


def classify_walkers_vectorized(
    will_clone: Tensor,
    companions_clone: Tensor,
    fitness: Tensor,
    alive: Tensor | None = None,
) -> WalkerClassification:
    """Classify walkers — fully vectorized version (no Python loop).

    Same semantics as :func:`classify_walkers` but avoids the per-frame loop
    for the target scatter by using a flattened scatter.

    Args:
        will_clone: Boolean cloning mask [T, N].
        companions_clone: Companion indices for cloning [T, N] (long).
        fitness: Fitness values [T, N].
        alive: Optional alive mask [T, N]. If None, all walkers assumed alive.

    Returns:
        WalkerClassification with all masks and chirality labels.
    """
    T, N = will_clone.shape
    device = will_clone.device

    if alive is None:
        alive = torch.ones(T, N, device=device, dtype=torch.bool)
    alive = alive.bool()
    will_clone_b = will_clone.bool()

    delta = will_clone_b & alive

    # Vectorized target detection: flatten [T, N] → [T*N], scatter, reshape
    frame_offsets = torch.arange(T, device=device).unsqueeze(1) * N  # [T, 1]
    flat_targets = (companions_clone.clamp(0, N - 1) + frame_offsets).view(-1)  # [T*N]
    flat_delta = delta.view(-1)  # [T*N]

    is_target_flat = torch.zeros(T * N, device=device, dtype=torch.bool)
    # Only scatter from deltas
    delta_targets = flat_targets[flat_delta]
    if delta_targets.numel() > 0:
        is_target_flat.scatter_(0, delta_targets, True)
    is_target = is_target_flat.view(T, N)

    strong_resister = ~will_clone_b & alive & is_target

    companion_fitness = torch.gather(fitness, 1, companions_clone.clamp(0, N - 1))
    has_fitter_peer = alive & (companion_fitness > fitness)

    weak_resister = ~will_clone_b & alive & ~is_target & has_fitter_peer
    persister = ~will_clone_b & alive & ~is_target & ~has_fitter_peer

    chi = torch.where(delta | strong_resister, 1.0, -1.0)
    chi = torch.where(alive, chi, torch.zeros_like(chi))

    return WalkerClassification(
        delta=delta,
        strong_resister=strong_resister,
        weak_resister=weak_resister,
        persister=persister,
        chi=chi,
        n_delta=delta.sum(dim=1),
        n_strong_resister=strong_resister.sum(dim=1),
        n_weak_resister=weak_resister.sum(dim=1),
        n_persister=persister.sum(dim=1),
    )


# ---------------------------------------------------------------------------
# Chirality autocorrelation
# ---------------------------------------------------------------------------


def _fft_autocorrelation(
    series: Tensor,
    max_lag: int,
    normalize: bool = True,
) -> Tensor:
    """FFT-based autocorrelation of a 1D series.

    Args:
        series: Input series [T].
        max_lag: Maximum lag to compute.
        normalize: If True, normalize so C(0) = 1.

    Returns:
        Autocorrelation [max_lag + 1].
    """
    T = series.shape[0]
    effective_lag = min(max_lag, T - 1)

    # Zero-pad to next power of 2
    n_fft = T + effective_lag + 1
    n_fft = 1 << (n_fft - 1).bit_length()

    s_pad = F.pad(series.unsqueeze(0), (0, n_fft - T)).squeeze(0)
    fft_s = torch.fft.fft(s_pad)
    power = (fft_s * fft_s.conj()).real
    corr = torch.fft.ifft(power).real

    # Normalize by number of overlapping pairs
    counts = torch.arange(T, T - effective_lag - 1, -1, device=series.device, dtype=torch.float32)
    result = corr[:effective_lag + 1] / counts

    if normalize and result[0].abs() > 1e-30:
        result = result / result[0]

    # Pad if effective_lag < max_lag
    if effective_lag < max_lag:
        result = F.pad(result, (0, max_lag - effective_lag), value=0.0)

    return result


@dataclass
class ChiralityCorrelatorOutput:
    """Output of chirality autocorrelation analysis.

    Attributes:
        c_chi: Mean chirality autocorrelation [max_lag + 1].
        c_chi_per_walker: Per-walker autocorrelation [N, max_lag + 1] if requested.
        c_chi_connected: Connected (mean-subtracted) autocorrelation [max_lag + 1].
        fermion_mass: Estimated fermion mass (decay rate of C_χ).
        fermion_mass_err: Uncertainty on fermion mass from fit.
        classification: Full WalkerClassification used.
        left_fraction: Mean fraction of walkers that are left-handed [T].
        role_transition_rate: Mean rate of L↔R transitions per walker per step.
        chi_mean: Mean chirality per frame [T].
        n_cloning_frames: Number of frames where cloning actually occurred.
        frame_mask: Boolean mask of frames with cloning events [T].
    """

    c_chi: Tensor
    c_chi_per_walker: Tensor | None
    c_chi_connected: Tensor
    fermion_mass: float
    fermion_mass_err: float
    classification: WalkerClassification
    left_fraction: Tensor
    role_transition_rate: float
    chi_mean: Tensor
    n_cloning_frames: int
    frame_mask: Tensor


def compute_chirality_autocorrelation(
    history: RunHistory,
    *,
    max_lag: int = 80,
    warmup_fraction: float = 0.1,
    end_fraction: float = 1.0,
    fit_start: int = 1,
    fit_stop: int | None = None,
    per_walker: bool = False,
    cloning_frames_only: bool = True,
) -> ChiralityCorrelatorOutput:
    """Compute chirality autocorrelation from run history.

    The chirality label χ_i(n) = +1 for left-handed (delta + strong resister)
    and -1 for right-handed (weak resister + persister). The autocorrelation
    C_χ(τ) = ⟨χ_i(n) χ_i(n+τ)⟩ decays as exp(-m_f τ), giving the fermion mass.

    Args:
        history: RunHistory with will_clone, companions_clone, fitness, alive_mask.
        max_lag: Maximum lag for autocorrelation.
        warmup_fraction: Fraction of frames to discard as warmup.
        end_fraction: Fraction of frames to use (from end).
        fit_start: First lag to include in exponential fit.
        fit_stop: Last lag to include (None = max_lag).
        per_walker: If True, also compute per-walker autocorrelation.
        cloning_frames_only: If True, only use frames where cloning occurred.

    Returns:
        ChiralityCorrelatorOutput with correlator and mass estimate.
    """
    device = history.fitness.device
    T_total = history.will_clone.shape[0]

    # Frame selection
    start_idx = int(T_total * warmup_fraction)
    end_idx = int(T_total * end_fraction)
    if end_idx <= start_idx:
        end_idx = T_total

    will_clone = history.will_clone[start_idx:end_idx]
    companions_clone = history.companions_clone[start_idx:end_idx]
    fitness = history.fitness[start_idx:end_idx]

    # alive_mask may be offset by 1 from other arrays
    alive_end = min(end_idx, history.alive_mask.shape[0])
    alive_start = min(start_idx, alive_end)
    alive = history.alive_mask[alive_start:alive_end]
    # Pad if needed
    T_sel = will_clone.shape[0]
    if alive.shape[0] < T_sel:
        pad_size = T_sel - alive.shape[0]
        alive = torch.cat([alive, alive[-1:].expand(pad_size, -1)], dim=0)

    # Classify walkers
    classification = classify_walkers_vectorized(
        will_clone=will_clone,
        companions_clone=companions_clone,
        fitness=fitness,
        alive=alive,
    )

    chi = classification.chi  # [T, N]

    # Optionally filter to only cloning frames
    if cloning_frames_only:
        frame_has_cloning = will_clone.any(dim=1)  # [T]
        frame_mask = frame_has_cloning
    else:
        frame_mask = torch.ones(T_sel, device=device, dtype=torch.bool)

    n_cloning_frames = int(frame_mask.sum().item())

    if n_cloning_frames < 2:
        empty = torch.zeros(max_lag + 1, device=device)
        return ChiralityCorrelatorOutput(
            c_chi=empty,
            c_chi_per_walker=None,
            c_chi_connected=empty.clone(),
            fermion_mass=0.0,
            fermion_mass_err=float("inf"),
            classification=classification,
            left_fraction=torch.zeros(T_sel, device=device),
            role_transition_rate=0.0,
            chi_mean=torch.zeros(T_sel, device=device),
            n_cloning_frames=n_cloning_frames,
            frame_mask=frame_mask,
        )

    # Extract cloning frames
    chi_filtered = chi[frame_mask]  # [T_eff, N]
    T_eff, N = chi_filtered.shape

    # Per-walker autocorrelation then average
    alive_filtered = alive[frame_mask] if alive.shape[0] >= T_sel else None

    # Compute per-walker chirality autocorrelation
    # C_χ(τ) = (1/N) Σ_i ⟨χ_i(n) χ_i(n+τ)⟩
    all_walker_corrs = torch.zeros(N, max_lag + 1, device=device)
    valid_walkers = 0

    for i in range(N):
        chi_i = chi_filtered[:, i]
        # Skip walkers that are always dead or constant
        if chi_i.abs().sum() < 1e-12:
            continue
        if chi_i.std() < 1e-12:
            continue
        all_walker_corrs[i] = _fft_autocorrelation(chi_i, max_lag, normalize=False)
        valid_walkers += 1

    if valid_walkers == 0:
        empty = torch.zeros(max_lag + 1, device=device)
        return ChiralityCorrelatorOutput(
            c_chi=empty,
            c_chi_per_walker=None,
            c_chi_connected=empty.clone(),
            fermion_mass=0.0,
            fermion_mass_err=float("inf"),
            classification=classification,
            left_fraction=classification.left_handed.float().mean(dim=1),
            role_transition_rate=0.0,
            chi_mean=chi.mean(dim=1),
            n_cloning_frames=n_cloning_frames,
            frame_mask=frame_mask,
        )

    # Average over walkers
    c_chi = all_walker_corrs.sum(dim=0) / max(valid_walkers, 1)

    # Normalize
    if c_chi[0].abs() > 1e-30:
        c_chi = c_chi / c_chi[0]

    # Connected correlator (subtract mean chirality squared)
    chi_mean_global = chi_filtered.mean()
    c_chi_connected = c_chi - chi_mean_global ** 2 / max(c_chi[0].item(), 1e-30)

    # Role transition rate
    chi_diff = (chi_filtered[1:] != chi_filtered[:-1]).float()
    role_transition_rate = float(chi_diff.mean().item()) if T_eff > 1 else 0.0

    # Fit exponential decay to extract mass
    fermion_mass, fermion_mass_err = _fit_exponential_decay(
        c_chi_connected, fit_start, fit_stop or max_lag
    )

    # Diagnostics
    left_fraction = classification.left_handed.float().mean(dim=1)
    chi_mean_series = chi.mean(dim=1)

    return ChiralityCorrelatorOutput(
        c_chi=c_chi,
        c_chi_per_walker=all_walker_corrs if per_walker else None,
        c_chi_connected=c_chi_connected,
        fermion_mass=fermion_mass,
        fermion_mass_err=fermion_mass_err,
        classification=classification,
        left_fraction=left_fraction,
        role_transition_rate=role_transition_rate,
        chi_mean=chi_mean_series,
        n_cloning_frames=n_cloning_frames,
        frame_mask=frame_mask,
    )


# ---------------------------------------------------------------------------
# Mass extraction
# ---------------------------------------------------------------------------


def _fit_exponential_decay(
    correlator: Tensor,
    fit_start: int,
    fit_stop: int,
    min_points: int = 3,
) -> tuple[float, float]:
    """Fit C(τ) ~ A exp(-m τ) via linear regression on log|C|.

    Args:
        correlator: Autocorrelation [max_lag + 1].
        fit_start: First lag to include.
        fit_stop: Last lag to include.
        min_points: Minimum number of valid points for fit.

    Returns:
        (mass, mass_uncertainty) tuple.
    """
    fit_stop = min(fit_stop, correlator.shape[0] - 1)
    if fit_stop <= fit_start:
        return 0.0, float("inf")

    tau = torch.arange(fit_start, fit_stop + 1, device=correlator.device, dtype=torch.float32)
    c_vals = correlator[fit_start:fit_stop + 1]

    # Only fit positive values (log requires positive)
    pos_mask = c_vals > 1e-30
    if pos_mask.sum() < min_points:
        return 0.0, float("inf")

    tau_fit = tau[pos_mask]
    log_c = torch.log(c_vals[pos_mask])

    # Linear regression: log C = log A - m τ
    n = tau_fit.shape[0]
    tau_mean = tau_fit.mean()
    logc_mean = log_c.mean()
    ss_tau = ((tau_fit - tau_mean) ** 2).sum()

    if ss_tau < 1e-30:
        return 0.0, float("inf")

    slope = ((tau_fit - tau_mean) * (log_c - logc_mean)).sum() / ss_tau
    mass = float(-slope.item())  # C ~ exp(-m τ), so slope = -m

    # Uncertainty from residuals
    predicted = logc_mean + slope * (tau_fit - tau_mean)
    residuals = log_c - predicted
    mse = (residuals ** 2).sum() / max(n - 2, 1)
    slope_err = torch.sqrt(mse / ss_tau)
    mass_err = float(slope_err.item())

    # Mass should be positive (correlator should decay)
    if mass < 0:
        mass = 0.0

    return mass, mass_err


# ---------------------------------------------------------------------------
# Convenience: from history directly
# ---------------------------------------------------------------------------


def compute_chirality_from_history(
    history: RunHistory,
    **kwargs: Any,
) -> ChiralityCorrelatorOutput:
    """Convenience wrapper for compute_chirality_autocorrelation.

    Accepts any keyword arguments that compute_chirality_autocorrelation takes.
    """
    return compute_chirality_autocorrelation(history, **kwargs)


# ---------------------------------------------------------------------------
# L-R coupling strength (Dirac mass proxy)
# ---------------------------------------------------------------------------


def compute_lr_coupling(
    history: RunHistory,
    warmup_fraction: float = 0.1,
    end_fraction: float = 1.0,
    h_eff: float = 1.0,
) -> dict[str, Tensor]:
    """Compute left-right coupling strength at each cloning step.

    The L-R coupling measures phase information transfer between the
    left-handed (D + SR) and right-handed (WR + P) populations during
    cloning. This is the Dirac mass term: stronger coupling = heavier fermion.

    Specifically, for each delta i that clones to a right-handed companion c:
        M(n) = (1/N_delta) Σ_{i∈D, comp(i)∈R} exp(i(S_c - S_i)/ℏ_eff)

    Args:
        history: RunHistory.
        warmup_fraction: Warmup fraction to skip.
        end_fraction: End fraction to use.
        h_eff: Effective Planck constant for phase.

    Returns:
        Dictionary with:
            - "lr_coupling_magnitude": |M(n)| per frame [T_eff]
            - "lr_coupling_phase": arg(M(n)) per frame [T_eff]
            - "lr_coupling_complex": M(n) per frame [T_eff] (complex)
            - "lr_fraction": fraction of deltas with R-handed companions [T_eff]
    """
    device = history.fitness.device
    T_total = history.will_clone.shape[0]
    N = history.will_clone.shape[1]

    start_idx = int(T_total * warmup_fraction)
    end_idx = int(T_total * end_fraction)

    will_clone = history.will_clone[start_idx:end_idx]
    companions_clone = history.companions_clone[start_idx:end_idx]
    fitness = history.fitness[start_idx:end_idx]
    alive_end = min(end_idx, history.alive_mask.shape[0])
    alive_start = min(start_idx, alive_end)
    alive = history.alive_mask[alive_start:alive_end]
    T_sel = will_clone.shape[0]
    if alive.shape[0] < T_sel:
        pad_size = T_sel - alive.shape[0]
        alive = torch.cat([alive, alive[-1:].expand(pad_size, -1)], dim=0)

    classification = classify_walkers_vectorized(
        will_clone=will_clone,
        companions_clone=companions_clone,
        fitness=fitness,
        alive=alive,
    )

    # For each delta, check if its companion is right-handed
    right_mask = classification.right_handed  # [T, N]
    delta_mask = classification.delta  # [T, N]

    # Companion of each walker
    comp_idx = companions_clone.clamp(0, N - 1)
    # Gather right-handedness of companions
    comp_is_right = torch.gather(right_mask, 1, comp_idx)  # [T, N]

    # Cross-chirality cloning: delta with R-handed companion
    cross_mask = delta_mask & comp_is_right  # [T, N]

    # Fitness phase difference
    comp_fitness = torch.gather(fitness, 1, comp_idx)
    phase = (comp_fitness - fitness) / max(h_eff, 1e-12)
    phase_exp = torch.exp(1j * phase.to(torch.complex64))

    # Average over cross-chirality deltas per frame
    cross_count = cross_mask.float().sum(dim=1).clamp(min=1)  # [T]
    delta_count = delta_mask.float().sum(dim=1).clamp(min=1)  # [T]

    lr_complex = (phase_exp * cross_mask.to(torch.complex64)).sum(dim=1) / cross_count

    return {
        "lr_coupling_magnitude": lr_complex.abs(),
        "lr_coupling_phase": lr_complex.angle(),
        "lr_coupling_complex": lr_complex,
        "lr_fraction": cross_mask.float().sum(dim=1) / delta_count,
    }
