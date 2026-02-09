"""Dirac spectrum analysis via antisymmetric kernel SVD.

Computes the fermion mass spectrum from the SVD of the antisymmetric kernel
(discrete Dirac operator). Generations emerge as eigenvalue clusters of K_tilde
projected onto gauge sectors defined by isospin (will_clone) and color
(viscous force magnitude).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fragile.fractalai.core.history import RunHistory


def _to_numpy(t):
    """Convert a tensor or array-like to a numpy array."""
    if hasattr(t, "cpu"):
        return t.cpu().numpy()
    return np.asarray(t)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiracSpectrumConfig:
    """Configuration for Dirac spectrum analysis."""

    mc_time_index: int | None = None
    epsilon_clone: float = 0.01
    n_generations: int | None = None
    color_threshold: str | float = 1.0
    min_sector_size: int = 10
    svd_top_k: int | None = None
    time_average: bool = False
    time_range: tuple[int, int] | None = None
    warmup_fraction: float = 0.1
    max_avg_frames: int = 80


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SectorSpectrum:
    """SVD result for one gauge sector."""

    sector_name: str
    singular_values: np.ndarray
    n_walkers: int
    generation_boundaries: list[int]
    generation_masses: list[float]
    generation_counts: list[int]


@dataclass
class DiracSpectrumResult:
    """Full Dirac spectrum analysis result."""

    # Full spectrum (unprojected)
    full_singular_values: np.ndarray
    full_generation_boundaries: list[int]
    full_generation_masses: list[float]

    # Per-sector projected spectra
    sectors: dict[str, SectorSpectrum]

    # Walker classification arrays (for plotting)
    color_magnitude: np.ndarray
    isospin_label: np.ndarray
    sector_assignment: np.ndarray
    walker_fitness: np.ndarray
    alive_indices: np.ndarray

    # Banks-Casher chiral condensate
    chiral_condensate: float
    near_zero_count: int

    # Metadata
    mc_time_index: int
    n_alive: int
    n_dimensions: int
    color_threshold_value: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def dedup_skew_sv(sigma: np.ndarray) -> np.ndarray:
    """Extract unique singular values from a skew-symmetric matrix SVD.

    Skew-symmetric matrices have exactly paired singular values
    (σ₁=σ₂, σ₃=σ₄, …) plus one zero if the dimension is odd.
    Taking every other entry recovers one representative per pair.
    """
    if len(sigma) == 0:
        return sigma
    return sigma[0::2]


SECTOR_NAMES = {
    0: "up_quark",
    1: "down_quark",
    2: "neutrino",
    3: "charged_lepton",
}

# PDG fermion masses (GeV) organized by sector and generation (heaviest first).
# Generation ordering: Gen1 = heaviest cluster, Gen2 = middle, Gen3 = lightest
# (matching SVD descending singular value convention).
FERMION_REFS: dict[str, list[tuple[str, float]]] = {
    "up_quark": [
        ("top", 172.69),
        ("charm", 1.27),
        ("up", 0.00216),
    ],
    "down_quark": [
        ("bottom", 4.18),
        ("strange", 0.0934),
        ("down", 0.00467),
    ],
    "neutrino": [
        ("nu_tau", 1e-10),
        ("nu_mu", 1e-10),
        ("nu_e", 1e-10),
    ],
    "charged_lepton": [
        ("tau", 1.77686),
        ("muon", 0.10566),
        ("electron", 0.000511),
    ],
}


def build_antisymmetric_kernel(
    fitness: np.ndarray,
    alive_mask: np.ndarray,
    epsilon_clone: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the antisymmetric kernel K_tilde from fitness values.

    K_tilde[i,j] = (V_j - V_i) * (V_i + V_j + 2*eps) / ((V_i + eps) * (V_j + eps))

    Args:
        fitness: Fitness values [N].
        alive_mask: Boolean alive mask [N].
        epsilon_clone: Regularization parameter.

    Returns:
        K_tilde: Antisymmetric matrix [N_alive, N_alive].
        alive_indices: Indices of alive walkers in the original array.
    """
    alive = alive_mask.astype(bool)
    alive_indices = np.where(alive)[0]
    V_alive = fitness[alive].astype(np.float64)

    Vi = V_alive[:, None]  # [N_alive, 1]
    Vj = V_alive[None, :]  # [1, N_alive]

    eps = epsilon_clone
    numerator = (Vj - Vi) * (Vi + Vj + 2.0 * eps)
    denominator = (Vi + eps) * (Vj + eps)

    # Avoid division by zero
    denominator = np.where(np.abs(denominator) < 1e-30, 1e-30, denominator)
    K_tilde = numerator / denominator

    return K_tilde, alive_indices


def classify_walkers(
    force_viscous: np.ndarray,
    will_clone: np.ndarray,
    alive_mask: np.ndarray,
    color_threshold: str | float = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Classify alive walkers into 4 gauge sectors.

    Color axis: ||F_visc|| vs threshold -> quark (above) / lepton (below).
    Isospin axis: will_clone=True -> up-type (cloner), False -> down-type.

    Sector indices:
        0 = up_quark      (cloner + strong viscous)
        1 = down_quark    (persister + strong viscous)
        2 = neutrino      (cloner + weak viscous)
        3 = charged_lepton (persister + weak viscous)

    Args:
        force_viscous: Viscous force [N, d].
        will_clone: Cloning mask [N] bool.
        alive_mask: Alive mask [N] bool.
        color_threshold: "median" for adaptive threshold, or a float for manual.

    Returns:
        sector_assignment: Sector index per alive walker [N_alive].
        color_magnitude: ||F_visc|| per alive walker [N_alive].
        isospin_label: +1 (up-type) or -1 (down-type) per alive walker [N_alive].
        threshold_value: The color threshold used.
    """
    alive = alive_mask.astype(bool)
    fv_alive = force_viscous[alive]
    wc_alive = will_clone[alive].astype(bool)

    color_mag = np.linalg.norm(fv_alive, axis=1)

    if isinstance(color_threshold, (int, float)):
        threshold_value = float(color_threshold)
    else:
        threshold_value = float(np.median(color_mag))

    is_quark = color_mag > threshold_value  # strong viscous = quark
    is_up = wc_alive  # cloner = up-type

    sector = np.zeros(len(color_mag), dtype=np.int32)
    sector[is_up & is_quark] = 0       # up_quark
    sector[~is_up & is_quark] = 1      # down_quark
    sector[is_up & ~is_quark] = 2      # neutrino
    sector[~is_up & ~is_quark] = 3     # charged_lepton

    isospin_label = np.where(is_up, 1, -1)

    return sector, color_mag, isospin_label, threshold_value


def find_generation_gaps(
    singular_values: np.ndarray,
    n_generations: int | None = None,
    d: int | None = None,
) -> list[int]:
    """Find cluster boundaries in the log-spectrum via largest gaps.

    Args:
        singular_values: Sorted descending singular values.
        n_generations: Number of generation clusters. If None, use d.
        d: Spatial dimension (used as fallback: d-1 largest gaps -> d clusters).

    Returns:
        Boundary indices splitting sigma into clusters.
    """
    if len(singular_values) < 2:
        return []

    # Work in log space for better gap detection
    sv = singular_values[singular_values > 0]
    if len(sv) < 2:
        return []

    log_sv = np.log(sv)
    gaps = np.abs(np.diff(log_sv))

    # Number of boundaries = n_clusters - 1
    if n_generations is not None:
        n_boundaries = max(n_generations - 1, 0)
    elif d is not None:
        n_boundaries = max(d - 1, 0)
    else:
        n_boundaries = 2  # default: 3 generations

    n_boundaries = min(n_boundaries, len(gaps))
    if n_boundaries == 0:
        return []

    # Find the n_boundaries largest gaps
    boundary_indices = np.argsort(gaps)[-n_boundaries:]
    # Convert to sorted boundary positions (after each gap index)
    boundaries = sorted((int(idx) + 1) for idx in boundary_indices)
    return boundaries


def _cluster_masses(
    singular_values: np.ndarray,
    boundaries: list[int],
) -> tuple[list[float], list[int]]:
    """Compute median mass and count per cluster given boundary indices."""
    sv = singular_values
    splits = [0] + boundaries + [len(sv)]
    masses = []
    counts = []
    for i in range(len(splits) - 1):
        cluster = sv[splits[i]:splits[i + 1]]
        if len(cluster) > 0:
            masses.append(float(np.median(cluster)))
            counts.append(len(cluster))
        else:
            masses.append(0.0)
            counts.append(0)
    return masses, counts


def compute_banks_casher(
    singular_values: np.ndarray,
    threshold_fraction: float = 0.1,
) -> tuple[float, int]:
    """Estimate chiral condensate from near-zero singular value density.

    <psi_bar psi> = -pi * rho(sigma ~ 0)

    Args:
        singular_values: All singular values.
        threshold_fraction: Fraction of max sigma defining "near zero".

    Returns:
        chiral_condensate: pi * rho(sigma ~ 0).
        near_zero_count: Number of near-zero singular values.
    """
    if len(singular_values) == 0:
        return 0.0, 0

    sv_max = singular_values.max()
    if sv_max <= 0:
        return 0.0, 0

    threshold = sv_max * threshold_fraction
    near_zero = singular_values[singular_values < threshold]
    near_zero_count = len(near_zero)

    # Density estimate: count / bin_width
    if threshold > 0:
        rho_zero = near_zero_count / threshold
    else:
        rho_zero = 0.0

    chiral_condensate = float(np.pi * rho_zero)
    return chiral_condensate, int(near_zero_count)


def compute_dirac_spectrum(
    history: RunHistory,
    config: DiracSpectrumConfig | None = None,
) -> DiracSpectrumResult:
    """Compute the Dirac spectrum from a RunHistory.

    Steps:
        1. Select MC frame from config.
        2. Extract fitness, force_viscous, will_clone, alive_mask.
        3. Build K_tilde for alive walkers.
        4. Classify walkers into 4 sectors.
        5. SVD of full K_tilde.
        6. For each sector: project K_tilde, SVD, find generation gaps.
        7. Banks-Casher estimate.

    Args:
        history: RunHistory instance.
        config: Analysis configuration.

    Returns:
        DiracSpectrumResult with full and per-sector spectra.
    """
    if config is None:
        config = DiracSpectrumConfig()

    # 1. Select MC frame
    max_t = history.n_recorded - 2
    if config.mc_time_index is not None:
        t = min(max(config.mc_time_index, 0), max_t)
    else:
        t = max(max_t, 0)

    n_dims = int(history.d)

    if config.time_average:
        return _compute_time_averaged(history, config, t, max_t, n_dims)

    # --- Single-frame path (original) ---

    # 2. Extract fields
    alive_mask = _to_numpy(history.alive_mask[t]).astype(bool)
    fitness = _to_numpy(history.fitness[t]).astype(np.float64)
    will_clone = _to_numpy(history.will_clone[t]).astype(bool)

    if history.force_viscous is not None:
        force_viscous = _to_numpy(history.force_viscous[t])
    else:
        # Fallback: zeros if force_viscous not recorded
        force_viscous = np.zeros((len(alive_mask), history.d))

    n_alive = int(alive_mask.sum())

    # 3. Build K_tilde
    K_tilde, alive_indices = build_antisymmetric_kernel(
        fitness, alive_mask, config.epsilon_clone,
    )

    # 4. Classify walkers
    sector_assignment, color_magnitude, isospin_label, threshold_value = classify_walkers(
        force_viscous, will_clone, alive_mask, config.color_threshold,
    )
    walker_fitness = fitness[alive_mask]

    # 5. SVD of full K_tilde (dedup skew-symmetric pairs)
    _, full_sigma_raw, _ = np.linalg.svd(K_tilde)
    # Banks-Casher uses the raw spectrum (pair density matters)
    chiral_condensate, near_zero_count = compute_banks_casher(full_sigma_raw)
    full_sigma = dedup_skew_sv(full_sigma_raw)
    if config.svd_top_k is not None:
        full_sigma = full_sigma[:config.svd_top_k]

    full_boundaries = find_generation_gaps(full_sigma, config.n_generations, n_dims)
    full_masses, _ = _cluster_masses(full_sigma, full_boundaries)

    # 6. Per-sector SVD
    sectors: dict[str, SectorSpectrum] = {}
    for sector_idx, sector_name in SECTOR_NAMES.items():
        idx = np.where(sector_assignment == sector_idx)[0]
        if len(idx) < config.min_sector_size:
            # Too few walkers for a meaningful SVD
            sectors[sector_name] = SectorSpectrum(
                sector_name=sector_name,
                singular_values=np.array([]),
                n_walkers=len(idx),
                generation_boundaries=[],
                generation_masses=[],
                generation_counts=[],
            )
            continue

        K_sector = K_tilde[np.ix_(idx, idx)]
        _, sigma_sector, _ = np.linalg.svd(K_sector)
        sigma_sector = dedup_skew_sv(sigma_sector)
        if config.svd_top_k is not None:
            sigma_sector = sigma_sector[:config.svd_top_k]

        boundaries = find_generation_gaps(sigma_sector, config.n_generations, n_dims)
        masses, counts = _cluster_masses(sigma_sector, boundaries)

        sectors[sector_name] = SectorSpectrum(
            sector_name=sector_name,
            singular_values=sigma_sector,
            n_walkers=len(idx),
            generation_boundaries=boundaries,
            generation_masses=masses,
            generation_counts=counts,
        )

    return DiracSpectrumResult(
        full_singular_values=full_sigma,
        full_generation_boundaries=full_boundaries,
        full_generation_masses=full_masses,
        sectors=sectors,
        color_magnitude=color_magnitude,
        isospin_label=isospin_label,
        sector_assignment=sector_assignment,
        walker_fitness=walker_fitness,
        alive_indices=alive_indices,
        chiral_condensate=chiral_condensate,
        near_zero_count=near_zero_count,
        mc_time_index=t,
        n_alive=n_alive,
        n_dimensions=n_dims,
        color_threshold_value=threshold_value,
    )


def _average_spectra(all_spectra: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Average variable-length spectra by truncating to the shortest length."""
    if not all_spectra:
        return np.array([]), np.array([])
    min_len = min(len(s) for s in all_spectra)
    if min_len == 0:
        return np.array([]), np.array([])
    stacked = np.stack([s[:min_len] for s in all_spectra])
    return stacked.mean(axis=0), stacked.std(axis=0)


def _compute_time_averaged(
    history: RunHistory,
    config: DiracSpectrumConfig,
    t_last: int,
    max_t: int,
    n_dims: int,
) -> DiracSpectrumResult:
    """Time-averaged Dirac spectrum over post-warmup frames."""
    # Determine frame range
    warmup_start = max(1, int(history.n_recorded * config.warmup_fraction))
    if config.time_range is not None:
        t_start, t_end = config.time_range
        t_start = max(t_start, warmup_start)
        t_end = min(t_end, max_t)
    else:
        t_start = warmup_start
        t_end = max_t

    # Subsample frames if the range exceeds max_avg_frames
    total_frames = t_end - t_start + 1
    if total_frames > config.max_avg_frames:
        frame_indices = np.linspace(t_start, t_end, config.max_avg_frames, dtype=int)
        frame_indices = np.unique(frame_indices)  # deduplicate after rounding
    else:
        frame_indices = np.arange(t_start, t_end + 1)

    # Phase 1: Collect per-frame SVD spectra
    all_full_sigmas: list[np.ndarray] = []
    all_sector_sigmas: dict[str, list[np.ndarray]] = {
        name: [] for name in SECTOR_NAMES.values()
    }
    all_chiral: list[float] = []
    all_near_zero: list[int] = []

    for frame_t in frame_indices:
        alive_t = _to_numpy(history.alive_mask[frame_t]).astype(bool)
        fitness_t = _to_numpy(history.fitness[frame_t]).astype(np.float64)
        will_clone_t = _to_numpy(history.will_clone[frame_t]).astype(bool)

        if history.force_viscous is not None:
            fv_t = _to_numpy(history.force_viscous[frame_t])
        else:
            fv_t = np.zeros((len(alive_t), history.d))

        if alive_t.sum() < 2:
            continue

        K_t, _ = build_antisymmetric_kernel(fitness_t, alive_t, config.epsilon_clone)
        _, sigma_t_raw, _ = np.linalg.svd(K_t)
        # Banks-Casher on raw spectrum (needs near-zero values + pair density)
        cc, nz = compute_banks_casher(sigma_t_raw)
        all_chiral.append(cc)
        all_near_zero.append(nz)
        # Dedup for gap detection / mass extraction
        sigma_t = dedup_skew_sv(sigma_t_raw)
        if config.svd_top_k is not None:
            sigma_t = sigma_t[:config.svd_top_k]
        all_full_sigmas.append(sigma_t)

        # Per-frame sector classification
        sector_t, _, _, _ = classify_walkers(
            fv_t, will_clone_t, alive_t, config.color_threshold,
        )
        for sector_idx, sector_name in SECTOR_NAMES.items():
            idx = np.where(sector_t == sector_idx)[0]
            if len(idx) >= config.min_sector_size:
                K_sec = K_t[np.ix_(idx, idx)]
                _, sig_sec, _ = np.linalg.svd(K_sec)
                sig_sec = dedup_skew_sv(sig_sec)
                if config.svd_top_k is not None:
                    sig_sec = sig_sec[:config.svd_top_k]
                all_sector_sigmas[sector_name].append(sig_sec)

    # Phase 2: Average spectra (truncate to common min length)
    if not all_full_sigmas:
        # Fallback: no valid frames, return empty result
        full_sigma = np.array([])
    else:
        full_sigma, _full_sigma_std = _average_spectra(all_full_sigmas)

    # Gap detection + mass extraction on averaged spectrum
    full_boundaries = find_generation_gaps(full_sigma, config.n_generations, n_dims)
    full_masses, _ = _cluster_masses(full_sigma, full_boundaries)

    # Per-sector averaged spectra
    sectors: dict[str, SectorSpectrum] = {}
    for sector_idx, sector_name in SECTOR_NAMES.items():
        sector_spectra = all_sector_sigmas[sector_name]
        if not sector_spectra:
            sectors[sector_name] = SectorSpectrum(
                sector_name=sector_name,
                singular_values=np.array([]),
                n_walkers=0,
                generation_boundaries=[],
                generation_masses=[],
                generation_counts=[],
            )
            continue

        avg_sigma, _sec_std = _average_spectra(sector_spectra)
        boundaries = find_generation_gaps(avg_sigma, config.n_generations, n_dims)
        masses, counts = _cluster_masses(avg_sigma, boundaries)
        # Use average walker count across frames for this sector
        avg_n = int(np.mean([len(s) for s in sector_spectra]))

        sectors[sector_name] = SectorSpectrum(
            sector_name=sector_name,
            singular_values=avg_sigma,
            n_walkers=avg_n,
            generation_boundaries=boundaries,
            generation_masses=masses,
            generation_counts=counts,
        )

    # Phase 3: Use LAST frame for walker classification plot data
    t_plot = min(t_last, max_t)
    alive_plot = _to_numpy(history.alive_mask[t_plot]).astype(bool)
    fitness_plot = _to_numpy(history.fitness[t_plot]).astype(np.float64)
    will_clone_plot = _to_numpy(history.will_clone[t_plot]).astype(bool)
    if history.force_viscous is not None:
        fv_plot = _to_numpy(history.force_viscous[t_plot])
    else:
        fv_plot = np.zeros((len(alive_plot), history.d))

    _, alive_indices = build_antisymmetric_kernel(
        fitness_plot, alive_plot, config.epsilon_clone,
    )
    sector_assignment, color_magnitude, isospin_label, threshold_value = classify_walkers(
        fv_plot, will_clone_plot, alive_plot, config.color_threshold,
    )
    walker_fitness = fitness_plot[alive_plot]
    n_alive = int(alive_plot.sum())

    # Banks-Casher: average per-frame estimates (computed on raw spectra)
    if all_chiral:
        chiral_condensate = float(np.mean(all_chiral))
        near_zero_count = int(np.mean(all_near_zero))
    else:
        chiral_condensate, near_zero_count = 0.0, 0

    return DiracSpectrumResult(
        full_singular_values=full_sigma,
        full_generation_boundaries=full_boundaries,
        full_generation_masses=full_masses,
        sectors=sectors,
        color_magnitude=color_magnitude,
        isospin_label=isospin_label,
        sector_assignment=sector_assignment,
        walker_fitness=walker_fitness,
        alive_indices=alive_indices,
        chiral_condensate=chiral_condensate,
        near_zero_count=near_zero_count,
        mc_time_index=t_plot,
        n_alive=n_alive,
        n_dimensions=n_dims,
        color_threshold_value=threshold_value,
    )


# ---------------------------------------------------------------------------
# Fermion mass comparison (extracted vs PDG)
# ---------------------------------------------------------------------------

_SKIP_SECTORS_FOR_SCALE = {"neutrino"}  # masses too uncertain for fitting


def _best_fit_scale_dirac(
    alg_masses: list[float],
    obs_masses: list[float],
) -> float | None:
    """Log-space least-squares scale: s = geometric_mean(m_i / σ_i).

    Minimizes Σ(log(s·σ_i) - log(m_i))² so that each particle contributes
    equally regardless of its absolute mass scale.
    """
    log_ratios = []
    for a, m in zip(alg_masses, obs_masses):
        if a > 0 and m > 0:
            log_ratios.append(np.log(m / a))
    if not log_ratios:
        return None
    return float(np.exp(np.mean(log_ratios)))


def build_fermion_comparison(
    result: DiracSpectrumResult,
    refs: dict[str, list[tuple[str, float]]] | None = None,
) -> tuple[list[dict], float | None]:
    """Compare extracted generation masses against PDG fermion masses.

    Returns:
        rows: list of dicts with columns sector, generation, particle,
              alg_mass, obs_mass_GeV, pred_mass_GeV, error_pct.
        scale: the best-fit scale (GeV per σ-unit), or None.
    """
    if refs is None:
        refs = FERMION_REFS

    # Collect (alg_mass, obs_mass) pairs excluding neutrinos for scale fit
    fit_alg: list[float] = []
    fit_obs: list[float] = []
    for sector_name, spec in result.sectors.items():
        if sector_name in _SKIP_SECTORS_FOR_SCALE:
            continue
        ref_list = refs.get(sector_name, [])
        masses = spec.generation_masses
        n = min(len(masses), len(ref_list))
        for i in range(n):
            if masses[i] > 0 and ref_list[i][1] > 0:
                fit_alg.append(masses[i])
                fit_obs.append(ref_list[i][1])

    scale = _best_fit_scale_dirac(fit_alg, fit_obs)

    # Build comparison rows for all sectors
    rows: list[dict] = []
    for sector_name, spec in result.sectors.items():
        ref_list = refs.get(sector_name, [])
        masses = spec.generation_masses
        n = min(len(masses), len(ref_list))
        for i in range(n):
            particle, obs_mass = ref_list[i]
            alg_mass = masses[i]
            pred_mass = alg_mass * scale if scale is not None else None
            if obs_mass > 0 and pred_mass is not None:
                error_pct = abs(pred_mass - obs_mass) / obs_mass * 100.0
            else:
                error_pct = None
            rows.append({
                "sector": sector_name,
                "generation": i + 1,
                "particle": particle,
                "alg_mass": round(alg_mass, 6),
                "obs_mass_GeV": obs_mass,
                "pred_mass_GeV": round(pred_mass, 6) if pred_mass is not None else None,
                "error_pct": round(error_pct, 2) if error_pct is not None else None,
            })

    return rows, scale


def build_fermion_ratio_comparison(
    result: DiracSpectrumResult,
    refs: dict[str, list[tuple[str, float]]] | None = None,
) -> list[dict]:
    """Compare inter-generation mass ratios against PDG ratios.

    For each sector with ≥2 generations, compare Gen_i/Gen_{i+1} ratio
    against the corresponding PDG mass ratio.

    Returns:
        rows: list of dicts with columns sector, ratio_label, measured,
              observed, error_pct.
    """
    if refs is None:
        refs = FERMION_REFS

    rows: list[dict] = []
    for sector_name, spec in result.sectors.items():
        ref_list = refs.get(sector_name, [])
        masses = spec.generation_masses
        n = min(len(masses), len(ref_list))
        if n < 2:
            continue
        for i in range(n - 1):
            alg_hi, alg_lo = masses[i], masses[i + 1]
            _, obs_hi = ref_list[i]
            _, obs_lo = ref_list[i + 1]
            measured = alg_hi / alg_lo if alg_lo > 0 else None
            observed = obs_hi / obs_lo if obs_lo > 0 else None
            if measured is not None and observed is not None and observed > 0:
                error_pct = abs(measured - observed) / observed * 100.0
            else:
                error_pct = None
            rows.append({
                "sector": sector_name,
                "ratio_label": f"Gen{i + 1}/Gen{i + 2}",
                "measured": round(measured, 4) if measured is not None else None,
                "observed": round(observed, 4) if observed is not None else None,
                "error_pct": round(error_pct, 2) if error_pct is not None else None,
            })

    return rows
