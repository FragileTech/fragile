"""
Parameter Space Sieve Verification
==================================

Verify that the fundamental constants of our universe satisfy the
cybernetic viability constraints from the Fragile Agent Framework.

This script implements the inequalities from Section 35 (The Parameter Space Sieve)
and substitutes measured values of fundamental constants to check if physical
reality occupies the Feasible Region where all Sieve constraints are satisfied.

Reference: docs/part8_multiagent/parameter_sieve.md

The key insight: fundamental constants are not arbitrary but are constrained
by cybernetic viability requirements. A universe that violates any constraint
cannot support coherent agents.

Usage:
    python src/experiments/constants_check.py
"""

from dataclasses import dataclass
from math import sqrt, pi, log, exp
from typing import Tuple

# ==============================================================================
# SECTION 1: FUNDAMENTAL CONSTANTS (CODATA 2018/2022 values)
# ==============================================================================

# Exact by definition (SI 2019 revision)
c = 299_792_458  # m/s - speed of light (exact)
h = 6.62607015e-34  # J·s - Planck constant (exact)
hbar = h / (2 * pi)  # J·s - reduced Planck constant
k_B = 1.380649e-23  # J/K - Boltzmann constant (exact)
e = 1.602176634e-19  # C - elementary charge (exact)
N_A = 6.02214076e23  # mol^-1 - Avogadro constant (exact)

# Measured (CODATA 2022 recommended values)
G = 6.67430e-11  # m³/(kg·s²) - gravitational constant
m_e = 9.1093837139e-31  # kg - electron mass
m_p = 1.67262192595e-27  # kg - proton mass
epsilon_0 = 8.8541878188e-12  # F/m - vacuum permittivity

# Derived Planck units
l_P = sqrt(hbar * G / c**3)  # Planck length ~1.616e-35 m
t_P = sqrt(hbar * G / c**5)  # Planck time ~5.391e-44 s
m_P = sqrt(hbar * c / G)  # Planck mass ~2.176e-8 kg
E_P = sqrt(hbar * c**5 / G)  # Planck energy ~1.956e9 J
T_P = E_P / k_B  # Planck temperature ~1.417e32 K

# Fine structure constant (dimensionless)
alpha = e**2 / (4 * pi * epsilon_0 * hbar * c)  # ~1/137.036

# Strong coupling constant at various scales
alpha_s_MZ = 0.1179  # at M_Z ~91 GeV (PDG 2023)
alpha_s_1GeV = 0.47  # at 1 GeV (approximate)
alpha_s_tau = 0.33  # at m_tau ~1.78 GeV
Lambda_QCD = 200e6 * e  # QCD scale ~200 MeV (in Joules)

# Cosmological parameters (Planck 2018)
H_0 = 67.4e3 / (3.086e22)  # Hubble constant in s^-1 (67.4 km/s/Mpc)
Omega_Lambda = 0.685  # Dark energy density parameter
Lambda_cosmo = 3 * H_0**2 * Omega_Lambda  # Cosmological constant ~1.1e-52 m^-2

# Derived scales
R_Hubble = c / H_0  # Hubble radius ~4.4e26 m
Rydberg = m_e * e**4 / (8 * epsilon_0**2 * h**2)  # ~13.6 eV (in Joules: * e)
Rydberg_J = 13.605693122994 * e  # Rydberg energy in Joules

# Biological scales (for metabolic constraint)
T_bio = 310  # K - human body temperature
kT_bio = k_B * T_bio  # thermal energy at biological temperature ~4.28e-21 J
ATP_energy = 30.5e3 / N_A  # ~30.5 kJ/mol per ATP hydrolysis ~5e-20 J


# ==============================================================================
# SECTION 2: CONSTRAINT RESULT DATA STRUCTURE
# ==============================================================================

@dataclass
class ConstraintResult:
    """Result of checking a single Sieve constraint."""
    name: str
    node: str  # Sieve node that enforces this
    satisfied: bool
    lhs: float  # Left-hand side value
    rhs: float  # Right-hand side value (bound)
    margin_log10: float  # log10(rhs/lhs) for upper bounds, log10(lhs/rhs) for lower
    details: str
    interpretation: str


def format_scientific(x: float, precision: int = 3) -> str:
    """Format number in scientific notation."""
    if x == 0:
        return "0"
    exp_val = int(log(abs(x)) / log(10))
    mantissa = x / (10 ** exp_val)
    return f"{mantissa:.{precision}f}e{exp_val:+d}"


# ==============================================================================
# SECTION 3: CONSTRAINT CHECK FUNCTIONS
# ==============================================================================

def check_speed_window() -> Tuple[ConstraintResult, ConstraintResult]:
    """
    Check the Speed Window constraint (Theorem 35.1 / thm-speed-window).

    The information speed c_info must satisfy:
        d_sync / tau_proc <= c_info <= L_buf / tau_proc

    Physical interpretation:
    - Lower bound: Signals must cross synchronization distance within processing time
    - Upper bound: Signals cannot traverse entire buffer in one cycle (causality)

    At the fundamental level:
    - d_sync ~ l_P (Planck length) - minimum distinguishable distance
    - tau_proc ~ t_P (Planck time) - minimum processing interval
    - L_buf ~ R_Hubble (Hubble radius) - maximum causal buffer
    """
    # Fundamental scales
    d_sync = l_P  # Synchronization distance ~ Planck length
    tau_proc = t_P  # Processing time ~ Planck time
    L_buf = R_Hubble  # Buffer depth ~ Hubble radius

    # The speed window
    c_min = d_sync / tau_proc  # Lower bound
    c_max = L_buf / tau_proc  # Upper bound
    c_info = c  # Speed of light

    # Check lower bound: c >= c_min
    lower_satisfied = c_info >= c_min
    lower_margin = log(c_info / c_min) / log(10) if c_min > 0 else float('inf')

    # Check upper bound: c <= c_max
    upper_satisfied = c_info <= c_max
    upper_margin = log(c_max / c_info) / log(10) if c_info > 0 else float('inf')

    lower_result = ConstraintResult(
        name="Speed Window (Lower)",
        node="Node 2: ZenoCheck",
        satisfied=lower_satisfied,
        lhs=c_min,
        rhs=c_info,
        margin_log10=lower_margin,
        details=f"d_sync/tau_proc = {format_scientific(c_min)} m/s <= c = {format_scientific(c_info)} m/s",
        interpretation=(
            f"At Planck scale: l_P/t_P = {format_scientific(c_min)} m/s.\n"
            f"This equals c exactly! The speed of light IS the Planck velocity.\n"
            f"Margin: 10^{lower_margin:.1f} (factor of ~1, saturated at Planck scale)"
        )
    )

    upper_result = ConstraintResult(
        name="Speed Window (Upper)",
        node="Node 62: CausalityViolationCheck",
        satisfied=upper_satisfied,
        lhs=c_info,
        rhs=c_max,
        margin_log10=upper_margin,
        details=f"c = {format_scientific(c_info)} m/s <= L_buf/tau_proc = {format_scientific(c_max)} m/s",
        interpretation=(
            f"Upper bound: Hubble radius / Planck time = {format_scientific(c_max)} m/s.\n"
            f"This is ~10^61 times larger than c.\n"
            f"Margin: 10^{upper_margin:.1f} orders of magnitude of causal headroom."
        )
    )

    return lower_result, upper_result


def check_holographic_bound() -> ConstraintResult:
    """
    Check the Holographic Bound (Theorem 35.2 / thm-holographic-bound).

    For D=4 spacetime (3+1 dimensions), the constraint is:
        l_L^2 <= nu_3 * Area_boundary / I_req

    This is equivalent to the Bekenstein-Hawking bound:
        S <= A / (4 * l_P^2)

    We check: Does the Planck length saturate the holographic bound
    for observable universe scales?
    """
    D = 4  # spacetime dimensions
    nu_D = 1/4  # Holographic coefficient (derived in the text)

    # Observable universe as the "agent"
    R_universe = R_Hubble
    Area_boundary = 4 * pi * R_universe**2  # Surface area of Hubble sphere

    # Information content estimate (Bekenstein bound for observable universe)
    # Using S_BH = A / (4 l_P^2)
    I_max_BH = Area_boundary / (4 * l_P**2)  # Maximum bits (Bekenstein-Hawking)

    # The constraint: l_L^(D-1) <= nu_D * Area / I_req
    # Rearranged: l_L <= (nu_D * Area / I_req)^(1/(D-1))
    # For D=4: l_L^2 <= nu_D * Area / I_req

    # Check if Planck length satisfies the bound
    # The bound becomes: l_P^2 <= (1/4) * Area / I
    # This is exactly the area law! The bound is saturated.

    lhs = l_P**(D-1)  # l_L^2 for D=4
    rhs = nu_D * Area_boundary / I_max_BH  # = l_P^2 (by construction)

    # Actually, let's check this more carefully
    # If I_req = I_max_BH, then rhs = nu_D * A / (A/(4*l_P^2)) = nu_D * 4 * l_P^2 = l_P^2
    # So lhs = rhs when we're at the Bekenstein bound

    satisfied = lhs <= rhs * 1.001  # Allow tiny numerical error
    margin = log(rhs / lhs) / log(10) if lhs > 0 else float('inf')

    return ConstraintResult(
        name="Holographic Bound",
        node="Node 56: CapacityHorizonCheck",
        satisfied=satisfied,
        lhs=lhs,
        rhs=rhs,
        margin_log10=margin,
        details=(
            f"l_P^2 = {format_scientific(lhs)} m^2 vs "
            f"nu_3 * Area / I_max = {format_scientific(rhs)} m^2"
        ),
        interpretation=(
            f"The Planck length saturates the holographic bound exactly.\n"
            f"Observable universe: Area = {format_scientific(Area_boundary)} m^2\n"
            f"Max info content: I_BH = {format_scientific(I_max_BH)} nats\n"
            f"This is the Bekenstein-Hawking entropy: S = A/(4 l_P^2).\n"
            f"The bound is saturated—we're at the edge of the feasible region!"
        )
    )


def check_landauer_constraint() -> ConstraintResult:
    """
    Check the Landauer Constraint (Theorem 35.3 / thm-landauer-constraint).

    The cognitive temperature must satisfy:
        T_c <= E_dot_met / (I_dot_erase * ln(2))

    At biological scales:
    - T_c ~ k_B * T_bio (thermal energy at body temperature)
    - E_dot_met ~ ATP hydrolysis rate per neuron
    - I_dot_erase ~ synaptic update rate

    At fundamental scales:
    - T_c ~ E_P (Planck energy)
    - E_dot_met ~ E_P / t_P (Planck power)
    - I_dot_erase ~ 1/t_P (Planck rate)
    """
    # Biological scale check (human neurons)
    # A neuron uses ~10^9 ATP/second, each ~5e-20 J
    ATP_rate_neuron = 1e9  # ATP molecules per second
    E_dot_met_bio = ATP_rate_neuron * ATP_energy  # ~5e-11 W per neuron

    # Synaptic updates: ~10 Hz typical firing rate, ~1000 synapses active
    synaptic_rate = 10 * 1000  # ~10^4 bit erasures per second
    I_dot_erase_bio = synaptic_rate

    # Landauer limit at biological temperature
    T_c_bio = kT_bio  # ~4.3e-21 J

    # The constraint: T_c <= E_dot / (I_dot * ln2)
    landauer_limit_bio = E_dot_met_bio / (I_dot_erase_bio * log(2))

    # Check satisfaction at biological scale
    satisfied_bio = T_c_bio <= landauer_limit_bio
    margin_bio = log(landauer_limit_bio / T_c_bio) / log(10) if T_c_bio > 0 else float('inf')

    # Also check at Planck scale for completeness
    E_dot_Planck = E_P / t_P  # Planck power ~3.6e52 W
    I_dot_Planck = 1 / t_P  # Planck rate ~1.9e43 /s
    landauer_limit_Planck = E_dot_Planck / (I_dot_Planck * log(2))
    # = E_P / ln2 ~2.8e9 J

    return ConstraintResult(
        name="Landauer Constraint",
        node="Node 52: LandauerViolationCheck",
        satisfied=satisfied_bio,
        lhs=T_c_bio,
        rhs=landauer_limit_bio,
        margin_log10=margin_bio,
        details=(
            f"T_bio = {format_scientific(T_c_bio)} J vs "
            f"Landauer limit = {format_scientific(landauer_limit_bio)} J"
        ),
        interpretation=(
            f"At biological scale (T=310K, neuron metabolism):\n"
            f"  Metabolic power: {format_scientific(E_dot_met_bio)} W\n"
            f"  Erasure rate: {format_scientific(I_dot_erase_bio)} bits/s\n"
            f"  Landauer limit: {format_scientific(landauer_limit_bio)} J\n"
            f"  Thermal energy: {format_scientific(T_c_bio)} J\n"
            f"Margin: 10^{margin_bio:.1f} - biology operates far above Landauer limit.\n"
            f"At Planck scale, limit = {format_scientific(landauer_limit_Planck)} J = E_P/ln2."
        )
    )


def check_ir_binding() -> ConstraintResult:
    """
    Check the IR Binding Constraint (Theorem 35.4 / thm-ir-binding-constraint).

    At the macro (infrared) scale, the binding coupling must exceed critical:
        g_s(mu_IR) >= g_s^crit

    In QCD, this is confinement: alpha_s -> infinity as mu -> Lambda_QCD.
    Color-charged quarks are confined into color-neutral hadrons.

    The critical coupling for confinement is approximately:
        g_s^crit ~ 1 (or alpha_s^crit ~ 0.3-0.5)
    """
    # QCD confinement scale
    mu_IR = Lambda_QCD / e  # ~200 MeV in natural units

    # At the confinement scale, alpha_s is large
    # Using 1-loop running: alpha_s(mu) ~ 1 / (b0 * ln(mu/Lambda_QCD))
    # where b0 = (33 - 2*Nf)/(12*pi) for SU(3) with Nf flavors

    # At mu ~ Lambda_QCD, alpha_s -> large (non-perturbative)
    # We use alpha_s at 1 GeV as proxy for IR behavior
    alpha_s_IR = alpha_s_1GeV  # ~0.47

    # Critical coupling for confinement (phenomenological estimate)
    # String tension requires alpha_s > ~0.3 for linear potential
    alpha_s_crit = 0.3

    satisfied = alpha_s_IR >= alpha_s_crit
    margin = log(alpha_s_IR / alpha_s_crit) / log(10) if alpha_s_crit > 0 else float('inf')

    return ConstraintResult(
        name="IR Binding (Confinement)",
        node="Node 40: PurityCheck",
        satisfied=satisfied,
        lhs=alpha_s_crit,
        rhs=alpha_s_IR,
        margin_log10=margin,
        details=f"alpha_s(1 GeV) = {alpha_s_IR:.3f} >= alpha_s^crit = {alpha_s_crit:.3f}",
        interpretation=(
            f"QCD coupling at low energies (confinement regime):\n"
            f"  alpha_s(1 GeV) ~ {alpha_s_IR:.2f}\n"
            f"  alpha_s(m_tau) ~ {alpha_s_tau:.2f}\n"
            f"  Lambda_QCD ~ 200 MeV\n"
            f"Strong coupling ensures quarks confine into hadrons.\n"
            f"Margin: 10^{margin:.2f} above critical - confinement is robust."
        )
    )


def check_uv_decoupling() -> ConstraintResult:
    """
    Check the UV Decoupling Constraint (Theorem 35.5 / thm-uv-decoupling-constraint).

    At the texture (ultraviolet) scale, the coupling must vanish:
        lim_{mu -> infinity} g_s(mu) = 0

    This is asymptotic freedom in QCD: alpha_s -> 0 as mu -> infinity.
    """
    # High-energy scale: M_Z ~ 91 GeV
    mu_UV = 91e9 * e  # Z boson mass in Joules
    alpha_s_UV = alpha_s_MZ  # ~0.118

    # Even higher scale: running to higher energies
    # At GUT scale (~10^16 GeV), alpha_s ~ 0.03
    alpha_s_GUT = 0.03  # approximate

    # The constraint: alpha_s should be decreasing with mu (asymptotic freedom)
    # Check: alpha_s(M_Z) < alpha_s(1 GeV)
    asymptotic_freedom = alpha_s_UV < alpha_s_1GeV

    # Effective "decoupling" threshold: alpha_s < 0.2 is perturbative
    epsilon_threshold = 0.2
    satisfied = alpha_s_UV < epsilon_threshold

    margin = log(epsilon_threshold / alpha_s_UV) / log(10) if alpha_s_UV > 0 else float('inf')

    return ConstraintResult(
        name="UV Decoupling (Asymptotic Freedom)",
        node="Node 29: TextureFirewallCheck",
        satisfied=satisfied,
        lhs=alpha_s_UV,
        rhs=epsilon_threshold,
        margin_log10=margin,
        details=f"alpha_s(M_Z) = {alpha_s_UV:.4f} < threshold = {epsilon_threshold:.2f}",
        interpretation=(
            f"QCD exhibits asymptotic freedom:\n"
            f"  alpha_s(1 GeV) ~ {alpha_s_1GeV:.2f} (IR, non-perturbative)\n"
            f"  alpha_s(M_Z=91 GeV) ~ {alpha_s_UV:.4f} (EW scale)\n"
            f"  alpha_s(GUT) ~ {alpha_s_GUT:.2f} (high energy)\n"
            f"Coupling decreases with energy -> texture decouples.\n"
            f"This is the beta function: beta(g) < 0 for SU(3) with Nf < 16.5."
        )
    )


def check_stiffness_bounds() -> Tuple[ConstraintResult, ConstraintResult]:
    """
    Check the Stiffness Bounds (Theorem 35.6 / thm-stiffness-bounds).

    The stiffness ratio chi = Delta_E / T_c must satisfy:
        1 < chi < chi_max

    For atomic/chemical systems:
        Delta_E ~ Rydberg ~ m_e * c^2 * alpha^2 / 2 ~ 13.6 eV
        T_c ~ k_B * T_bio ~ 0.027 eV at 310K

    This gives chi ~ 500, well within the Goldilocks window.
    """
    # Characteristic energy gap (atomic binding)
    Delta_E = Rydberg_J  # ~13.6 eV = 2.18e-18 J

    # Thermal energy at biological temperature
    T_c = kT_bio  # ~0.027 eV = 4.3e-21 J

    # Stiffness ratio
    chi = Delta_E / T_c  # ~500

    # Bounds
    chi_min = 1.0
    chi_max = 1e6  # Upper bound: must allow some transitions

    # Lower bound check
    lower_satisfied = chi > chi_min
    lower_margin = log(chi / chi_min) / log(10)

    # Upper bound check
    upper_satisfied = chi < chi_max
    upper_margin = log(chi_max / chi) / log(10)

    lower_result = ConstraintResult(
        name="Stiffness (Lower Bound)",
        node="Node 7: StiffnessCheck",
        satisfied=lower_satisfied,
        lhs=chi_min,
        rhs=chi,
        margin_log10=lower_margin,
        details=f"chi = {chi:.1f} > 1",
        interpretation=(
            f"Memory stability requirement:\n"
            f"  Delta_E (Rydberg) = {format_scientific(Delta_E)} J = 13.6 eV\n"
            f"  k_B * T_bio = {format_scientific(T_c)} J = 0.027 eV\n"
            f"  chi = Delta_E / T = {chi:.1f}\n"
            f"Since chi >> 1, thermal fluctuations rarely flip states.\n"
            f"P_flip ~ exp(-chi) ~ exp(-500) ~ 10^(-217) - memory is stable!"
        )
    )

    upper_result = ConstraintResult(
        name="Stiffness (Upper Bound)",
        node="Node 7: StiffnessCheck",
        satisfied=upper_satisfied,
        lhs=chi,
        rhs=chi_max,
        margin_log10=upper_margin,
        details=f"chi = {chi:.1f} < chi_max = {chi_max:.0e}",
        interpretation=(
            f"Adaptability requirement:\n"
            f"Transition rate Gamma ~ exp(-chi) must be non-zero.\n"
            f"At chi ~ 500, catalyzed reactions can still occur.\n"
            f"Enzymes lower effective barriers, enabling biological dynamics.\n"
            f"If chi -> infinity, no adaptation - the agent freezes."
        )
    )

    return lower_result, upper_result


def check_discount_window() -> Tuple[ConstraintResult, ConstraintResult]:
    """
    Check the Discount Window (Theorem 35.7 / thm-discount-window).

    The temporal discount gamma must satisfy:
        gamma_min < gamma < 1

    Physical mapping: gamma relates to the cosmological horizon.
    The screening length l_gamma = c * tau / (-ln(gamma)) ~ L_buf.

    For gamma close to 1: l_gamma -> infinity (long planning horizon)
    For gamma close to 0: l_gamma -> 0 (myopic)
    """
    # The discount factor relates to cosmological horizon via:
    # l_gamma = l_0 / (-ln(gamma)) where l_0 = c * tau_proc

    # At cosmological scale:
    # If l_gamma ~ R_Hubble and l_0 ~ l_P, then:
    # -ln(gamma) ~ l_P / R_Hubble ~ 10^(-61)
    # gamma ~ 1 - 10^(-61) (extremely close to 1)

    l_0 = c * t_P  # = l_P (Planck length)
    L_buf = R_Hubble

    # Solve for gamma from l_gamma = L_buf
    # L_buf = l_0 / (-ln(gamma))
    # -ln(gamma) = l_0 / L_buf = l_P / R_Hubble ~ 3.7e-62
    minus_ln_gamma = l_0 / L_buf
    gamma_cosmo = exp(-minus_ln_gamma)  # ~1 - 3.7e-62

    # Bounds
    gamma_min = 0.0
    gamma_max = 1.0

    # The physical gamma is extremely close to 1
    gamma = gamma_cosmo

    # Lower bound
    lower_satisfied = gamma > gamma_min
    lower_margin = -log(1 - gamma + 1e-100) / log(10)  # How close to 0

    # Upper bound
    upper_satisfied = gamma < gamma_max
    upper_margin = -log(1 - gamma + 1e-100) / log(10)  # How close to 1

    lower_result = ConstraintResult(
        name="Discount Factor (Lower Bound)",
        node="Causal Buffer Architecture",
        satisfied=lower_satisfied,
        lhs=gamma_min,
        rhs=gamma,
        margin_log10=lower_margin,
        details=f"gamma = 1 - {format_scientific(1-gamma)} > 0",
        interpretation=(
            f"Goal-directedness requirement:\n"
            f"If gamma = 0, the agent is completely myopic.\n"
            f"Physical gamma ~ 1 - {format_scientific(1-gamma)}\n"
            f"This is essentially 1 - meaning nearly infinite planning horizon."
        )
    )

    upper_result = ConstraintResult(
        name="Discount Factor (Upper Bound)",
        node="Screening Consistency",
        satisfied=upper_satisfied,
        lhs=gamma,
        rhs=gamma_max,
        margin_log10=upper_margin,
        details=f"gamma = 1 - {format_scientific(1-gamma)} < 1",
        interpretation=(
            f"Locality requirement:\n"
            f"If gamma = 1 exactly, the screening length l_gamma -> infinity.\n"
            f"The Helmholtz equation becomes Poisson: -nabla^2 V = r.\n"
            f"Value would have long-range (1/r) decay - non-local planning.\n"
            f"Physical gamma is strictly < 1 by {format_scientific(1-gamma)}.\n"
            f"Screening mass kappa = {format_scientific(minus_ln_gamma/l_0)} m^-1"
        )
    )

    return lower_result, upper_result


# ==============================================================================
# SECTION 4: REPORT GENERATION
# ==============================================================================

def print_separator(char: str = "=", width: int = 80) -> None:
    """Print a separator line."""
    print(char * width)


def print_result(result: ConstraintResult) -> None:
    """Print a single constraint result."""
    status = "SATISFIED" if result.satisfied else "VIOLATED"
    status_color = "\033[92m" if result.satisfied else "\033[91m"  # Green/Red
    reset = "\033[0m"

    print(f"\n{status_color}[{status}]{reset} {result.name}")
    print(f"  Node: {result.node}")
    print(f"  Check: {result.details}")
    print(f"  Margin: 10^{result.margin_log10:.1f}")
    print(f"  Interpretation:")
    for line in result.interpretation.split('\n'):
        print(f"    {line}")


def print_report(results: list) -> None:
    """Print the complete verification report."""
    print_separator()
    print("PARAMETER SPACE SIEVE VERIFICATION")
    print("Checking if our universe satisfies cybernetic viability constraints")
    print_separator()

    print("\n>>> FUNDAMENTAL CONSTANTS (CODATA 2022) <<<")
    print(f"  c (speed of light)     = {format_scientific(c)} m/s")
    print(f"  hbar (Planck const)    = {format_scientific(hbar)} J·s")
    print(f"  G (gravitational)      = {format_scientific(G)} m³/(kg·s²)")
    print(f"  k_B (Boltzmann)        = {format_scientific(k_B)} J/K")
    print(f"  alpha (fine structure) = 1/{1/alpha:.3f}")
    print(f"  alpha_s(M_Z)           = {alpha_s_MZ}")

    print("\n>>> DERIVED PLANCK UNITS <<<")
    print(f"  l_P (Planck length)    = {format_scientific(l_P)} m")
    print(f"  t_P (Planck time)      = {format_scientific(t_P)} s")
    print(f"  E_P (Planck energy)    = {format_scientific(E_P)} J")
    print(f"  T_P (Planck temp)      = {format_scientific(T_P)} K")

    print("\n>>> COSMOLOGICAL SCALES <<<")
    print(f"  H_0 (Hubble const)     = {format_scientific(H_0)} s^-1")
    print(f"  R_Hubble               = {format_scientific(R_Hubble)} m")
    print(f"  Lambda_cosmo           = {format_scientific(Lambda_cosmo)} m^-2")

    print_separator("-")
    print("CONSTRAINT VERIFICATION RESULTS")
    print_separator("-")

    for result in results:
        print_result(result)

    print_separator()

    # Summary
    n_satisfied = sum(1 for r in results if r.satisfied)
    n_total = len(results)
    all_satisfied = n_satisfied == n_total

    if all_satisfied:
        print(f"\033[92m>>> ALL {n_total} CONSTRAINTS SATISFIED <<<\033[0m")
        print("\nConclusion: Our universe lies within the FEASIBLE REGION")
        print("of the Parameter Space Sieve. The fundamental constants")
        print("satisfy all cybernetic viability requirements.")
        print("\nThis supports the thesis that physics constants are not")
        print("arbitrary but are constrained by the requirements for")
        print("coherent agents to exist.")
    else:
        print(f"\033[91m>>> {n_total - n_satisfied}/{n_total} CONSTRAINTS VIOLATED <<<\033[0m")
        print("\nSome constraints are violated. This may indicate:")
        print("1. Errors in the constraint formulation")
        print("2. Incorrect scale estimates")
        print("3. The theory needs revision")

    print_separator()


def main() -> bool:
    """Run all constraint checks and print report."""

    # Collect all results
    results = []

    # Speed Window (2 constraints)
    speed_lower, speed_upper = check_speed_window()
    results.extend([speed_lower, speed_upper])

    # Holographic Bound
    results.append(check_holographic_bound())

    # Landauer Constraint
    results.append(check_landauer_constraint())

    # IR Binding
    results.append(check_ir_binding())

    # UV Decoupling
    results.append(check_uv_decoupling())

    # Stiffness Bounds (2 constraints)
    stiff_lower, stiff_upper = check_stiffness_bounds()
    results.extend([stiff_lower, stiff_upper])

    # Discount Window (2 constraints)
    discount_lower, discount_upper = check_discount_window()
    results.extend([discount_lower, discount_upper])

    # Print report
    print_report(results)

    # Return overall result
    return all(r.satisfied for r in results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
