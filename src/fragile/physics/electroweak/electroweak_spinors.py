"""Electroweak spinor operators: chiral projectors, gauge links, and Standard Model currents.

Combines three independent structures to construct electroweak current operators:

1. **Dirac chirality** (P_L, P_R from gamma5): projects spinor bilinears into
   left-/right-handed components.

2. **Walker-role chirality** (L/R from cloning classification): walkers that
   participate in cloning (delta + strong resister) are "left-handed" in the
   dynamical sense, while persisters and weak resisters are "right-handed".

3. **Gauge links** (fitness Wilson lines): U(1) and SU(2) phase factors that
   dress the bilinear with electroweak gauge information.

Operator map to Standard Model physics:

    j_vector_L_su2  ->  W boson propagator (left current x SU(2) link)
    j_vector_u1     ->  Photon propagator (vector current x U(1) link)
    j_vector_L_u1   ->  Z boson (approximate: left current x U(1) link)
    o_yukawa_LR     ->  Fermion mass (Yukawa: cross-chirality scalar)
    parity_violation_dirac  ->  (|J_L|^2 - |J_R|^2) / (|J_L|^2 + |J_R|^2)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from fragile.physics.new_channels.dirac_spinors import (
    build_dirac_gamma_matrices,
    color_to_dirac_spinor,
)


# ===========================================================================
# Chiral projectors
# ===========================================================================


def build_chiral_projectors(
    gamma5: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build chiral projection matrices P_L = (1 - gamma5)/2, P_R = (1 + gamma5)/2.

    Args:
        gamma5: The gamma5 matrix [4, 4].

    Returns:
        Tuple (P_L, P_R), each [4, 4].
    """
    I4 = torch.eye(4, device=gamma5.device, dtype=gamma5.dtype)
    P_L = 0.5 * (I4 - gamma5)
    P_R = 0.5 * (I4 + gamma5)
    return P_L, P_R


# ===========================================================================
# Gauge links
# ===========================================================================


def compute_u1_gauge_link(
    fitness_i: Tensor,
    fitness_j: Tensor,
    h_eff: float,
) -> Tensor:
    """Compute U(1) hypercharge gauge link between walker pairs.

    U_ij = exp(i (S_j - S_i) / h_eff)

    where S = fitness is the action. This is the discrete analog of the
    parallel transporter exp(i integral A_mu dx^mu) for the U(1) gauge field.

    Args:
        fitness_i: Fitness at site i [...].
        fitness_j: Fitness at site j [...].
        h_eff: Effective Planck constant.

    Returns:
        Complex phase [...] with |U| = 1.
    """
    phase = (fitness_j - fitness_i) / max(h_eff, 1e-12)
    return torch.exp(1j * phase.to(torch.float64)).to(torch.complex128)


def compute_su2_gauge_link(
    fitness_i: Tensor,
    fitness_j: Tensor,
    h_eff: float,
    epsilon_clone: float = 1e-8,
) -> Tensor:
    """Compute SU(2) isospin gauge link between walker pairs.

    U_ij = exp(i |Delta S| / (|Delta S| + epsilon) * pi/2 / h_eff)

    The SU(2) link uses the magnitude of the fitness difference
    normalized by (|Delta S| + epsilon), giving a phase between 0 and pi/2.
    This captures the isospin rotation strength.

    Args:
        fitness_i: Fitness at site i [...].
        fitness_j: Fitness at site j [...].
        h_eff: Effective Planck constant.
        epsilon_clone: Regularizer for fitness normalization.

    Returns:
        Complex phase [...] with |U| = 1.
    """
    delta_s = (fitness_j - fitness_i).abs()
    normalized = delta_s / (delta_s + epsilon_clone)
    phase = normalized * (torch.pi / 2.0) / max(h_eff, 1e-12)
    return torch.exp(1j * phase.to(torch.float64)).to(torch.complex128)


# ===========================================================================
# Chiral bilinear
# ===========================================================================


def _compute_chiral_bilinear(
    psi_i: Tensor,
    psi_j: Tensor,
    gamma0: Tensor,
    Gamma: Tensor,
    P_chirality: Tensor | None = None,
    gauge_link: Tensor | None = None,
) -> Tensor:
    """Compute chiral bilinear psi_bar_i Gamma P psi_j, optionally gauge-dressed.

    Computes: Re(U_ij * psi_i^dag gamma0 Gamma P psi_j)

    where P is an optional chiral projector and U is an optional gauge link.

    Args:
        psi_i: Dirac spinors at site i [..., 4].
        psi_j: Dirac spinors at site j [..., 4].
        gamma0: gamma0 matrix [4, 4].
        Gamma: Dirac structure [4, 4].
        P_chirality: Optional chiral projector [4, 4]. If None, uses identity.
        gauge_link: Optional complex gauge link [...]. If None, uses 1.

    Returns:
        Real bilinear values [...].
    """
    # Build M = gamma0 @ Gamma @ P
    M = gamma0 @ Gamma
    if P_chirality is not None:
        M = M @ P_chirality

    # psi_bar_i M psi_j = psi_i^dag M psi_j
    bilinear = torch.einsum("...a,ab,...b->...", psi_i.conj(), M, psi_j)

    if gauge_link is not None:
        bilinear = bilinear * gauge_link

    return bilinear.real.float()


# ===========================================================================
# Output dataclass
# ===========================================================================


@dataclass
class ElectroweakSpinorOutput:
    """All electroweak spinor operator time series.

    Each field is a [T] tensor — the spatially averaged operator value at each
    MC timestep.

    Operator families:
        - Chiral currents: j_vector_L, j_vector_R, j_vector_V, o_scalar_L, o_scalar_R
        - Walker-role restricted: j_vector_walkerL, j_vector_walkerR,
          j_vector_L_walkerL, j_vector_R_walkerR
        - Cross-chirality (Yukawa): o_yukawa_LR, o_yukawa_RL
        - Gauge-dressed: j_vector_u1, j_vector_L_u1, j_vector_L_su2, j_vector_R_su2
        - Diagnostics: n_valid_pairs, n_valid_pairs_LL, n_valid_pairs_RR,
          n_valid_pairs_LR, parity_violation_dirac, parity_violation_walker
    """

    # -- Chiral currents (no gauge dressing) --
    j_vector_L: Tensor       # [T] left-handed vector current
    j_vector_R: Tensor       # [T] right-handed vector current
    j_vector_V: Tensor       # [T] full vector current (V = L + R)
    o_scalar_L: Tensor       # [T] left-handed scalar
    o_scalar_R: Tensor       # [T] right-handed scalar

    # -- Walker-role restricted --
    j_vector_walkerL: Tensor   # [T] vector current, walker-L pairs
    j_vector_walkerR: Tensor   # [T] vector current, walker-R pairs
    j_vector_L_walkerL: Tensor  # [T] Dirac-L on walker-L pairs
    j_vector_R_walkerR: Tensor  # [T] Dirac-R on walker-R pairs

    # -- Cross-chirality (Yukawa) --
    o_yukawa_LR: Tensor      # [T] scalar P_L on L->R walker pairs
    o_yukawa_RL: Tensor      # [T] scalar P_R on R->L walker pairs

    # -- Gauge-dressed --
    j_vector_u1: Tensor       # [T] vector current x U(1) link
    j_vector_L_u1: Tensor     # [T] left current x U(1) link
    j_vector_L_su2: Tensor    # [T] left current x SU(2) link
    j_vector_R_su2: Tensor    # [T] right current x SU(2) link

    # -- Diagnostics --
    n_valid_pairs: Tensor           # [T] total valid pairs
    n_valid_pairs_LL: Tensor        # [T] both-L pairs
    n_valid_pairs_RR: Tensor        # [T] both-R pairs
    n_valid_pairs_LR: Tensor        # [T] cross-LR pairs
    parity_violation_dirac: Tensor  # [T] (|j_L|^2 - |j_R|^2)/(|j_L|^2 + |j_R|^2)
    parity_violation_walker: Tensor  # [T] (|j_wL|^2 - |j_wR|^2)/(|j_wL|^2 + |j_wR|^2)


# ===========================================================================
# Main computation
# ===========================================================================


def compute_electroweak_spinor_operators(
    color: Tensor,
    color_valid: Tensor,
    sample_indices: Tensor,
    neighbor_indices: Tensor,
    alive: Tensor,
    fitness: Tensor,
    walker_chi: Tensor,
    *,
    h_eff: float = 1.0,
    epsilon_clone: float = 1e-8,
    sample_edge_weights: Tensor | None = None,
) -> ElectroweakSpinorOutput:
    """Compute all electroweak spinor operators.

    Args:
        color: Complex color states [T, N, 3].
        color_valid: Color validity [T, N].
        sample_indices: Sampled walker indices [T, S].
        neighbor_indices: Neighbor indices [T, S, k].
        alive: Alive mask [T, N].
        fitness: Fitness values [T, N].
        walker_chi: Walker chirality labels [T, N] (+1=L, -1=R, 0=dead).
        h_eff: Effective Planck constant.
        epsilon_clone: SU(2) fitness normalization regularizer.
        sample_edge_weights: Optional Riemannian weights [T, S].

    Returns:
        ElectroweakSpinorOutput with all operator time series.
    """
    T, N, d = color.shape
    S = sample_indices.shape[1]
    device = color.device

    if d != 3:
        raise ValueError(f"Dirac spinor requires d=3, got d={d}")

    # --- Build infrastructure ---
    gamma = build_dirac_gamma_matrices(device=device)
    gamma0 = gamma["gamma0"]
    gamma_k = gamma["gamma_k"]  # [3, 4, 4]
    gamma5 = gamma["gamma5"]
    I4 = torch.eye(4, device=device, dtype=gamma0.dtype)

    P_L, P_R = build_chiral_projectors(gamma5)

    # Convert color -> Dirac spinor
    spinor, spinor_valid = color_to_dirac_spinor(color)  # [T, N, 4], [T, N]
    spinor_valid = spinor_valid & color_valid

    # --- Gather pairs ---
    t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
    psi_i = spinor[t_idx, sample_indices]        # [T, S, 4]
    first_nb = neighbor_indices[:, :, 0]          # [T, S]
    psi_j = spinor[t_idx, first_nb]              # [T, S, 4]

    # Validity
    v_i = (
        spinor_valid[t_idx, sample_indices]
        & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
    )
    v_j = (
        spinor_valid[t_idx, first_nb]
        & alive[t_idx.clamp(max=alive.shape[0] - 1), first_nb]
    )
    valid = v_i & v_j & (first_nb != sample_indices)

    # Walker chirality for pairs
    chi_i = walker_chi[t_idx, sample_indices]    # [T, S]  +1=L, -1=R
    chi_j = walker_chi[t_idx, first_nb]          # [T, S]

    walker_L_i = chi_i > 0    # [T, S]
    walker_L_j = chi_j > 0
    walker_R_i = chi_i < 0
    walker_R_j = chi_j < 0

    # Population masks for pair types
    both_L = valid & walker_L_i & walker_L_j
    both_R = valid & walker_R_i & walker_R_j
    cross_LR = valid & walker_L_i & walker_R_j
    cross_RL = valid & walker_R_i & walker_L_j

    # Fitness for gauge links
    fit_i = fitness[t_idx, sample_indices]        # [T, S]
    fit_j = fitness[t_idx, first_nb]              # [T, S]

    # Gauge links
    u1_link = compute_u1_gauge_link(fit_i, fit_j, h_eff)        # [T, S] complex
    su2_link = compute_su2_gauge_link(fit_i, fit_j, h_eff, epsilon_clone)  # [T, S]

    # --- Weights ---
    if sample_edge_weights is not None:
        w = sample_edge_weights.to(device=device, dtype=torch.float32)
    else:
        w = torch.ones(T, S, device=device)

    # --- Helper: averaged operator ---
    zero = torch.zeros(T, S, device=device)

    def _avg(op_vals: Tensor, mask: Tensor) -> Tensor:
        """Weighted average of op_vals over pairs satisfying mask -> [T]."""
        op_masked = torch.where(mask, op_vals, zero)
        w_masked = torch.where(mask, w, zero)
        w_sum = w_masked.sum(dim=1).clamp(min=1e-12)
        return (op_masked * w_masked).sum(dim=1) / w_sum

    def _count(mask: Tensor) -> Tensor:
        return mask.sum(dim=1)

    # --- Helper: vector current averaged over spatial directions ---
    def _vector_current(
        P_chiral: Tensor | None,
        mask: Tensor,
        gauge: Tensor | None = None,
    ) -> Tensor:
        """(1/3) sum_k psi_bar gamma_k P psi averaged over pairs in mask."""
        accum = zero.clone()
        for k in range(3):
            vals = _compute_chiral_bilinear(
                psi_i, psi_j, gamma0, gamma_k[k],
                P_chirality=P_chiral, gauge_link=gauge,
            )
            accum = accum + vals
        return _avg(accum / 3.0, mask)

    # --- Helper: scalar bilinear ---
    def _scalar_op(
        P_chiral: Tensor | None,
        mask: Tensor,
        gauge: Tensor | None = None,
    ) -> Tensor:
        """psi_bar P psi averaged over pairs in mask."""
        vals = _compute_chiral_bilinear(
            psi_i, psi_j, gamma0, I4,
            P_chirality=P_chiral, gauge_link=gauge,
        )
        return _avg(vals, mask)

    # === Compute all operators ===

    # --- Chiral currents (no gauge dressing, all valid pairs) ---
    j_vector_L = _vector_current(P_L, valid)
    j_vector_R = _vector_current(P_R, valid)
    j_vector_V = _vector_current(None, valid)

    o_scalar_L = _scalar_op(P_L, valid)
    o_scalar_R = _scalar_op(P_R, valid)

    # --- Walker-role restricted ---
    j_vector_walkerL = _vector_current(None, both_L | cross_LR)  # i in L
    j_vector_walkerR = _vector_current(None, both_R | cross_RL)  # i in R

    # Double chiral: Dirac projection AND walker restriction
    j_vector_L_walkerL = _vector_current(P_L, both_L)
    j_vector_R_walkerR = _vector_current(P_R, both_R)

    # --- Cross-chirality (Yukawa) ---
    o_yukawa_LR = _scalar_op(P_L, cross_LR)
    o_yukawa_RL = _scalar_op(P_R, cross_RL)

    # --- Gauge-dressed operators ---
    j_vector_u1 = _vector_current(None, valid, gauge=u1_link)
    j_vector_L_u1 = _vector_current(P_L, valid, gauge=u1_link)
    j_vector_L_su2 = _vector_current(P_L, valid, gauge=su2_link)
    j_vector_R_su2 = _vector_current(P_R, valid, gauge=su2_link)

    # --- Diagnostics ---
    eps_pv = 1e-30

    jL2 = j_vector_L ** 2
    jR2 = j_vector_R ** 2
    pv_dirac = (jL2 - jR2) / (jL2 + jR2 + eps_pv)

    jwL2 = j_vector_walkerL ** 2
    jwR2 = j_vector_walkerR ** 2
    pv_walker = (jwL2 - jwR2) / (jwL2 + jwR2 + eps_pv)

    return ElectroweakSpinorOutput(
        j_vector_L=j_vector_L,
        j_vector_R=j_vector_R,
        j_vector_V=j_vector_V,
        o_scalar_L=o_scalar_L,
        o_scalar_R=o_scalar_R,
        j_vector_walkerL=j_vector_walkerL,
        j_vector_walkerR=j_vector_walkerR,
        j_vector_L_walkerL=j_vector_L_walkerL,
        j_vector_R_walkerR=j_vector_R_walkerR,
        o_yukawa_LR=o_yukawa_LR,
        o_yukawa_RL=o_yukawa_RL,
        j_vector_u1=j_vector_u1,
        j_vector_L_u1=j_vector_L_u1,
        j_vector_L_su2=j_vector_L_su2,
        j_vector_R_su2=j_vector_R_su2,
        n_valid_pairs=_count(valid),
        n_valid_pairs_LL=_count(both_L),
        n_valid_pairs_RR=_count(both_R),
        n_valid_pairs_LR=_count(cross_LR),
        parity_violation_dirac=pv_dirac,
        parity_violation_walker=pv_walker,
    )


# ===========================================================================
# Convenience: from RunHistory
# ===========================================================================


def compute_electroweak_spinor_from_history(
    history: "RunHistory",
    *,
    warmup_fraction: float = 0.1,
    end_fraction: float = 1.0,
    h_eff: float = 1.0,
    epsilon_clone: float = 1e-8,
    cloning_frames_only: bool = True,
) -> ElectroweakSpinorOutput:
    """Compute electroweak spinor operators directly from RunHistory.

    Handles frame selection, walker classification, color state extraction,
    and operator computation in one call.

    Args:
        history: RunHistory object.
        warmup_fraction: Fraction of frames to skip.
        end_fraction: Fraction of frames to use.
        h_eff: Effective Planck constant.
        epsilon_clone: SU(2) regularizer.
        cloning_frames_only: If True, restrict to frames with cloning events.

    Returns:
        ElectroweakSpinorOutput.
    """
    from fragile.physics.electroweak.chirality import classify_walkers_vectorized
    from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0_auto

    device = history.fitness.device
    T_total = history.will_clone.shape[0]

    # Frame selection
    start_idx = int(T_total * warmup_fraction)
    end_idx = int(T_total * end_fraction)
    if end_idx <= start_idx:
        end_idx = T_total

    # Extract data slices
    will_clone = history.will_clone[start_idx:end_idx]
    companions_clone = history.companions_clone[start_idx:end_idx]
    fitness = history.fitness[start_idx:end_idx]

    alive_end = min(end_idx, history.alive_mask.shape[0])
    alive_start = min(start_idx, alive_end)
    alive = history.alive_mask[alive_start:alive_end]
    T_sel = will_clone.shape[0]
    if alive.shape[0] < T_sel:
        pad = T_sel - alive.shape[0]
        alive = torch.cat([alive, alive[-1:].expand(pad, -1)], dim=0)

    # Optional: filter to cloning frames
    if cloning_frames_only:
        frame_mask = will_clone.any(dim=1)
        if frame_mask.sum() < 2:
            frame_mask = torch.ones(T_sel, device=device, dtype=torch.bool)
    else:
        frame_mask = torch.ones(T_sel, device=device, dtype=torch.bool)

    will_clone = will_clone[frame_mask]
    companions_clone = companions_clone[frame_mask]
    fitness_sel = fitness[frame_mask]
    alive_sel = alive[frame_mask]

    # Classify walkers
    classification = classify_walkers_vectorized(
        will_clone=will_clone,
        companions_clone=companions_clone,
        fitness=fitness_sel,
        alive=alive_sel,
    )

    # Compute color states for selected frames
    frame_indices = torch.where(frame_mask)[0] + start_idx
    T_eff = frame_indices.shape[0]

    # Estimate ell0
    ell0 = estimate_ell0_auto(history)

    # Build color states frame by frame
    color_list = []
    valid_list = []
    for idx in frame_indices:
        idx_int = int(idx.item())
        c, v = compute_color_states_batch(
            history,
            start_idx=idx_int,
            h_eff=h_eff,
            mass=1.0,
            ell0=ell0,
            end_idx=idx_int + 1,
        )
        color_list.append(c)
        valid_list.append(v)

    color = torch.cat(color_list, dim=0)        # [T_eff, N, d]
    color_valid = torch.cat(valid_list, dim=0)  # [T_eff, N]

    # Build simple neighbor indices from companions
    N = companions_clone.shape[1]
    sample_indices = torch.arange(N, device=device).unsqueeze(0).expand(T_eff, -1)
    neighbor_indices = companions_clone.unsqueeze(-1)  # [T_eff, N, 1]

    return compute_electroweak_spinor_operators(
        color=color,
        color_valid=color_valid,
        sample_indices=sample_indices,
        neighbor_indices=neighbor_indices,
        alive=alive_sel,
        fitness=fitness_sel,
        walker_chi=classification.chi,
        h_eff=h_eff,
        epsilon_clone=epsilon_clone,
    )


# ===========================================================================
# Mass spectrum extraction
# ===========================================================================


@dataclass
class ElectroweakMassSpectrum:
    """Mass spectrum extracted from electroweak spinor correlators.

    Each mass is the exponential decay rate of the corresponding
    operator autocorrelation.
    """

    # Gauge boson masses
    m_W: float          # from j_vector_L_su2 (left current x SU(2) link)
    m_Z: float          # from mixed L/R neutral current
    m_photon: float     # from j_vector_u1 (should be ~0)

    # Fermion masses
    m_fermion_L: float  # from o_scalar_L autocorrelation
    m_fermion_R: float  # from o_scalar_R autocorrelation
    m_yukawa: float     # from o_yukawa_LR (cross-chirality = Dirac mass)

    # Parity violation
    mean_pv_dirac: float    # mean parity violation from Dirac projectors
    mean_pv_walker: float   # mean parity violation from walker roles

    # Uncertainties
    m_W_err: float
    m_Z_err: float
    m_photon_err: float
    m_fermion_L_err: float
    m_fermion_R_err: float
    m_yukawa_err: float


def extract_ew_masses(
    ops: ElectroweakSpinorOutput,
    dt: float = 1.0,
    max_lag: int = 40,
    fit_start: int = 1,
    fit_stop: int | None = None,
) -> ElectroweakMassSpectrum:
    """Extract electroweak mass spectrum from operator time series.

    Computes autocorrelation of each operator channel and fits
    exponential decay to extract masses.

    Args:
        ops: ElectroweakSpinorOutput from compute_electroweak_spinor_operators.
        dt: Time step between frames (in simulation units).
        max_lag: Maximum lag for autocorrelation.
        fit_start: First lag to include in fit.
        fit_stop: Last lag (None = max_lag).

    Returns:
        ElectroweakMassSpectrum.
    """
    from fragile.physics.electroweak.chirality import _fit_exponential_decay

    def _autocorr(series: Tensor) -> Tensor:
        T = series.shape[0]
        s = series.float() - series.float().mean()
        eff = min(max_lag, T - 1)
        n_fft = 1 << (T + eff).bit_length()
        s_pad = F.pad(s.unsqueeze(0), (0, n_fft - T)).squeeze(0)
        power = torch.fft.fft(s_pad).abs() ** 2
        corr = torch.fft.ifft(power).real
        counts = torch.arange(T, T - eff - 1, -1, device=s.device, dtype=torch.float32)
        result = corr[:eff + 1] / counts
        if result[0].abs() > 1e-30:
            result = result / result[0]
        if eff < max_lag:
            result = F.pad(result, (0, max_lag - eff))
        return result

    def _extract(series: Tensor) -> tuple[float, float]:
        c = _autocorr(series)
        m, err = _fit_exponential_decay(c, fit_start, fit_stop or max_lag)
        return m / dt, err / dt

    m_W, m_W_e = _extract(ops.j_vector_L_su2)
    m_photon, m_photon_e = _extract(ops.j_vector_u1)
    m_Z, m_Z_e = _extract(ops.j_vector_L_u1)

    m_fL, m_fL_e = _extract(ops.o_scalar_L)
    m_fR, m_fR_e = _extract(ops.o_scalar_R)
    m_Y, m_Y_e = _extract(ops.o_yukawa_LR)

    return ElectroweakMassSpectrum(
        m_W=m_W, m_Z=m_Z, m_photon=m_photon,
        m_fermion_L=m_fL, m_fermion_R=m_fR, m_yukawa=m_Y,
        mean_pv_dirac=float(ops.parity_violation_dirac.mean()),
        mean_pv_walker=float(ops.parity_violation_walker.mean()),
        m_W_err=m_W_e, m_Z_err=m_Z_e, m_photon_err=m_photon_e,
        m_fermion_L_err=m_fL_e, m_fermion_R_err=m_fR_e, m_yukawa_err=m_Y_e,
    )
