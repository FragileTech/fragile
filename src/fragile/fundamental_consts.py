import numpy as np
import matplotlib.pyplot as plt

def run_grand_unification_test():
    """
    This is the main, renamed function to bypass the execution environment's caching issue.
    It performs the Grand Unification Test by checking for a consistent geometric
    solution space for all three lepton generations.
    """

    # --- 1. Define All Constants and Functions Locally ---

    print("--- Running FINAL, ENCAPSULATED script for Grand Unification Test ---")

    # A) Measured Physical Constants
    ALPHA_FS = 1 / 137.036
    SIGMA = 440**2
    DELTA_YM = 1600
    HBAR_EFF = 1.0

    # B) The Three Lepton Generations
    particles = {
        'Electron': {'mass': 0.511, 'color': 'firebrick', 'linestyle': '--'},
        'Muon':     {'mass': 105.66, 'color': 'forestgreen', 'linestyle': '-.'},
        'Tau':      {'mass': 1776.86, 'color': 'darkorange', 'linestyle': ':'}
    }

    # C) Core Mathematical Functions
    def calculate_mass_scales(c, c0, particle_mass):
        if c <= 0 or c0 <= 0: return float('nan'), float('nan'), float('nan')
        m_gap_sq = (SIGMA * HBAR_EFF) / (c * particle_mass)
        m_gap = np.sqrt(m_gap_sq) if m_gap_sq > 0 else float('nan')
        m_mf_sq = 1 / (ALPHA_FS * HBAR_EFF)
        m_mf = np.sqrt(m_mf_sq)
        m_clone_sq = (c0 * SIGMA * HBAR_EFF) / (c * DELTA_YM**2)
        m_clone = np.sqrt(m_clone_sq) if m_clone_sq > 0 else float('nan')
        return m_gap, m_mf, m_clone

    def check_hierarchy(m_gap, m_mf, m_clone):
        if any(np.isnan([m_gap, m_mf, m_clone])): return False
        return m_gap < m_mf and m_mf < m_clone

    def get_c_lower_bound(mass):
        return (SIGMA * ALPHA_FS * HBAR_EFF**2) / mass

    def get_c0_lower_bound_func():
        return lambda c_val: (c_val * DELTA_YM**2) / (SIGMA * ALPHA_FS * HBAR_EFF**2)

    # --- 2. Main Analysis and Visualization Logic ---

    print("\n--- The Grand Unification Test ---")
    print("Searching for a universal geometry (c, c0) that supports all three lepton generations...")

    plt.figure(figsize=(14, 10))

    c0_boundary_func = get_c0_lower_bound_func()
    c_bounds = {name: get_c_lower_bound(props['mass']) for name, props in particles.items()}
    c_intersect_min = max(c_bounds.values())
    c_intersect_particle = max(c_bounds, key=c_bounds.get)

    print(f"\nMost Restrictive Constraint on c comes from the {c_intersect_particle}: c > {c_intersect_min:,.2f}")

    c_max_plot = c_intersect_min * 2.5
    c_vals = np.linspace(c_intersect_min * 0.5, c_max_plot, 500)
    c0_boundary_vals = c0_boundary_func(c_vals)

    plt.plot(c_vals, c0_boundary_vals,
             label=r'$c_0 > c \cdot \frac{\Delta_{YM}^2}{\sigma \alpha_{FS} \hbar_{eff}^2}$ (Universal Boundary from $m_{MF} < m_{clone}$)',
             color='royalblue', linewidth=3, zorder=10)

    for name, props in particles.items():
        plt.axvline(x=c_bounds[name], linestyle=props['linestyle'], color=props['color'], linewidth=2.5,
                    label=f'$c >$ Bound for {name} (from $m_{gap} < m_{MF}$)')

    valid_c_vals = c_vals[c_vals > c_intersect_min]
    plt.fill_between(valid_c_vals, c0_boundary_func(valid_c_vals), 1e8,
                     color='lightgreen', alpha=0.5, label='Unified Valid Solution Space')

    c_solution = c_intersect_min * 1.5
    c0_solution = c0_boundary_func(c_solution) * 1.5

    plt.plot(c_solution, c0_solution, 'ko', markersize=14, mfc='gold', zorder=11,
             label=f'Sample Unified Solution\n(c={c_solution:.0f}, c0={c0_solution:,.0f})')

    print("\n--- Unified Solution Verification ---")
    print(f"Chosen universal geometric constants: c = {c_solution:.2f}, c0 = {c0_solution:,.2f}\n")
    all_valid = True
    for name, props in particles.items():
        m_gap, m_mf, m_clone = calculate_mass_scales(c_solution, c0_solution, props['mass'])
        is_valid = check_hierarchy(m_gap, m_mf, m_clone)
        if not is_valid: all_valid = False

        print(f"--- Verification for {name} (m={props['mass']:.2f} MeV) ---")
        print(f"  m_gap   = {m_gap:,.2f} MeV (Particle-specific)")
        print(f"  m_MF    = {m_mf:,.2f} MeV (Universal)")
        print(f"  m_clone = {m_clone:,.2f} MeV (Universal)")
        print(f"  Hierarchy Check: {m_gap:,.2f} < {m_mf:,.2f} < {m_clone:,.2f}  ->  {'PASSED' if is_valid else 'FAILED'}")

    print(f"\n{'='*50}")
    print(f"OVERALL TEST RESULT: {'SUCCESS! A consistent universal solution exists.' if all_valid else 'FAILURE! Inconsistency found.'}")
    print(f"{'='*50}")

    plt.xlabel('Universal Geometric Constant c (Dimensionless)', fontsize=14)
    plt.ylabel('Universal Geometric Constant c0 (Dimensionless)', fontsize=14)
    plt.title('Grand Unification Test: Solution Space for a 3D, Self-Organizing Universe', fontsize=16)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.yscale('log')
    plt.xlim(left=c_intersect_min * 0.75, right=c_max_plot)
    plt.ylim(bottom=1e6, top=1e8)

    plt.show()

if __name__ == "__main__":
    run_grand_unification_test()