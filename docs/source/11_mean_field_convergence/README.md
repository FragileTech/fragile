# Mean-Field Convergence Analysis (Continuous-Time)

This folder contains the mean-field (N→∞) convergence proof using continuous-time PDE analysis.

## Document Sequence

**Stage 0: Revival Operator Analysis**
- **[11_stage0_revival_kl.md](11_stage0_revival_kl.md)** - Proves revival operator is KL-expansive
  - Critical GO/NO-GO investigation
  - Result: Revival increases entropy (verified by Gemini)
  - Decision: Proceed with kinetic dominance strategy

**Stage 0.5: QSD Regularity**
- **[11_stage05_qsd_regularity.md](11_stage05_qsd_regularity.md)** - Proves QSD existence, uniqueness, and regularity
  - Proves all six regularity properties (R1-R6)
  - Establishes foundation for NESS hypocoercivity
  - Status: ✅ Complete (all proofs verified)

**Stage 1: Full Generator Analysis**
- **[11_stage1_entropy_production.md](11_stage1_entropy_production.md)** - Corrected entropy production analysis
  - Analyzes full generator $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{jump}}$
  - Uses NESS hypocoercivity framework (Dolbeault et al. 2015)
  - Proves kinetic dissipation dominates jump expansion
  - Status: ✅ Framework complete

**Stage 2: Explicit Constants**
- **[11_stage2_explicit_constants.md](11_stage2_explicit_constants.md)** - Explicit hypocoercivity constants
  - Derives fully explicit formulas for all constants
  - LSI constant: $\lambda_{\text{LSI}} \ge \alpha_{\exp}/(1 + C_{\Delta v}/\alpha_{\exp})$
  - Coercivity gap: $\delta = \lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}$
  - Convergence rate: $\alpha_{\text{net}} = \delta/2$
  - Status: ✅ Complete with numerical verification strategy

**Stage 3: Parameter Analysis**
- **[11_stage3_parameter_analysis.md](11_stage3_parameter_analysis.md)** - Parameter dependence and tuning guide
  - Explicit formula: $\alpha_{\text{net}}(\tau, \gamma, \sigma, \lambda_{\text{revive}}, \kappa_{\max}, L_U)$
  - Parameter sensitivities: $S_{\sigma}, S_{\gamma}, S_{\tau}, S_{\kappa}, S_{\lambda}$
  - Critical diffusion: $\sigma_{\text{crit}} \gtrsim (2L_U^3/\gamma)^{1/4}$
  - Optimal scaling: $\gamma^* \sim L_U^{3/7}, \sigma^* \sim L_U^{9/14}$
  - Finite-N corrections: $\alpha_N = \alpha_{\text{net}}(1 - C_N/N)(1 - \tau\alpha_{\text{net}}/(2\gamma))$
  - Worked examples with diagnostic procedures
  - Status: ✅ Complete with implementation guide

## Key Insight

The mean-field proof mirrors the finite-N structure from [../kl_convergence/10_kl_convergence.md](../kl_convergence/10_kl_convergence.md):

| Finite-N | Mean-Field |
|:---------|:-----------|
| Discrete-time operators | Continuous-time generator |
| Hypocoercive Lyapunov | Entropy + Fisher information |
| Cloning preserves LSI | Revival expansive but bounded |
| Kinetic dissipation > cloning expansion | Kinetic dissipation > jump expansion |

## Parent Document

See [../11_convergence_mean_field.md](../11_convergence_mean_field.md) for the original mean-field convergence framework (this folder contains the detailed technical development).

## Status

- ✅ Stage 0 complete: Revival operator analysis verified
- ✅ Stage 0.5 complete: QSD regularity proven (R1-R6)
- ✅ Stage 1 complete: Full generator entropy production established
- ✅ Stage 2 complete: Explicit hypocoercivity constants derived
- ✅ Stage 3 complete: Parameter analysis and simulation guide
- 🎯 **Ready for implementation**: All theoretical formulas complete and actionable

## Practical Usage

**For practitioners running simulations**:
1. Start with [11_stage3_parameter_analysis.md](11_stage3_parameter_analysis.md) for parameter tuning
2. Use Section 6 diagnostic procedures to validate convergence
3. See Section 7 for worked examples with real parameter values
4. Refer to [../../../src/fragile/gas_parameters.py](../../../src/fragile/gas_parameters.py) for code implementation

**For theorists**:
1. [11_stage05_qsd_regularity.md](11_stage05_qsd_regularity.md) - Foundation (QSD properties)
2. [11_stage1_entropy_production.md](11_stage1_entropy_production.md) - Framework (NESS hypocoercivity)
3. [11_stage2_explicit_constants.md](11_stage2_explicit_constants.md) - Technical details (all constants)
4. [11_stage3_parameter_analysis.md](11_stage3_parameter_analysis.md) - Applications (parameter tuning)
