# KMS and Equilibrium Reversibility Audit (Volume 3 / Fractal Set)

## Scope
Searched Volume 3 sources for statements about KMS, detailed balance, reversibility, time reversal, QSD equilibrium, and reflection positivity, with focus on Fractal Set chapters (not Euclidean lattice assumptions).

## Where KMS/detailed-balance claims are asserted
- `docs/source/3_fractal_gas/2_fractal_set/03_lattice_qft.md:25` claims detailed balance of the reversible BAOAB diffusion kernel at QSD equilibrium, yielding KMS and Wick rotation.
- `docs/source/3_fractal_gas/2_fractal_set/03_lattice_qft.md:568` repeats: preserving QSD/Gibbs measure implies reversibility (detailed balance) and hence KMS.
- `docs/source/3_fractal_gas/2_fractal_set/04_standard_model.md:503` claims reversible diffusion kernel satisfies detailed balance at equilibrium and thus KMS/Wick rotation.
- `docs/source/3_fractal_gas/2_fractal_set/05_yang_mills_noether.md:3200` claims detailed balance for BAOAB diffusion kernel at QSD equilibrium; uses Kossakowski-Frigerio-Gorini-Verri to infer KMS.
- `docs/source/3_fractal_gas/3_fitness_manifold/06_cosmology.md:163` attributes vanishing source terms at QSD to detailed balance of the reversible BAOAB diffusion kernel.

## Internal constraints / conflicts already in Volume 3
- Selection breaks detailed balance: `docs/source/3_fractal_gas/intro_fractal_gas.md:323`.
- QSD can exist without detailed balance; detailed balance is only needed to support KMS/Wick: `docs/source/3_fractal_gas/appendices/14_faq.md:199`.
- Kinetic generator is not self-adjoint (non-reversible): `docs/source/3_fractal_gas/appendices/15_kl_convergence.md:583`.
- BAOAB is symmetric as an integrator, but symmetry alone does not imply detailed balance: `docs/source/3_fractal_gas/appendices/05_kinetic_contraction.md:486`.
- Time reversal on CST is structurally forbidden: `docs/source/3_fractal_gas/2_fractal_set/04_standard_model.md:1119-1125`. This blocks any generalized detailed-balance claim that requires a global time-reversal involution on CST edges.
- The Adaptive Gas SDE includes viscous and adaptive forces (not pure gradient flow), so reversibility is not automatic: `docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md:592-602`.

## Supporting facts available in Volume 3 (Fractal Set context)
- QSD is the equilibrium object (conditioned on survival); equilibrium statistics are defined with respect to QSD: `docs/source/3_fractal_gas/2_fractal_set/05_yang_mills_noether.md:2477`, `docs/source/3_fractal_gas/intro_fractal_gas.md:460-462`.
- QSD equilibrium includes flux-balance language (net kinetic flux cancels net cloning source) but is not stated as pointwise current zero: `docs/source/3_fractal_gas/appendices/07_discrete_qsd.md:337`.
- Velocity marginal thermalizes to Maxwell-Boltzmann at QSD in mean-field limit: `docs/source/3_fractal_gas/appendices/07_discrete_qsd.md:283-298`.
- Generator decomposes into a symmetric OU part and anti-symmetric transport part in L^2(rho_infty): `docs/source/3_fractal_gas/appendices/10_kl_hypocoercive.md:200` and `docs/source/3_fractal_gas/appendices/10_kl_hypocoercive.md:258`.
- Reflecting boundary conditions for mean-field equations are defined in appendices (velocity/position reflection): `docs/source/3_fractal_gas/appendices/08_mean_field.md:633-634`.

## Gaps for an unconditional KMS proof (Fractal Set only)
1. There is no internal proof that the full equilibrium Fractal Gas kernel (kinetic + cloning) is reversible or satisfies detailed balance with respect to the QSD. In fact, non-reversibility of the kinetic generator is stated explicitly (`appendices/15_kl_convergence.md:583`).
2. The KMS inference in the Fractal Set chapters relies on “preserves Gibbs/QSD measure ⇒ detailed balance,” which is not valid without reversibility; this is not proven internally.
3. A generalized detailed-balance statement would normally require a time-reversal involution; but time reversal is explicitly forbidden on CST (`04_standard_model.md:1119-1125`), so it cannot be invoked globally on the Fractal Set.
4. The OS2 proof in `05_yang_mills_noether.md` uses KMS from detailed balance; but the document already asserts reflection positivity from hypercontractivity/transfer matrix. The KMS step is not independently justified within Volume 3.

## Implications for a minimal, unconditional fix (no algorithm change)
- If KMS must remain in the chain, Volume 3 needs a new proof that the *equilibrium* diffusion subkernel on the Fractal Set is reversible in the sense used (either standard or generalized detailed balance) **without** invoking time reversal on CST. No such proof is currently present.
- If KMS is not strictly required for OS2 in your framework (since you already have a reflection-positivity argument), then the cleanest Fractal-Set-consistent fix is to remove KMS as a dependency in OS2 and keep KMS as an optional equilibrium corollary only when a reversible subkernel is explicitly identified.
- Periodic boundary conditions are currently disabled by assumption in the Latent Gas framework (`docs/source/3_fractal_gas/1_the_algorithm/02_fractal_gas_latent.md:1020`), so introducing PBCs would be a change to existing Volume 3 assumptions.

## Specific lines that would need revision later (after approval)
- `docs/source/3_fractal_gas/2_fractal_set/05_yang_mills_noether.md:3200-3214` (detailed balance ⇒ KMS).
- `docs/source/3_fractal_gas/2_fractal_set/03_lattice_qft.md:25,505,568` (detailed balance from QSD preservation).
- `docs/source/3_fractal_gas/2_fractal_set/04_standard_model.md:503-517` (same detailed-balance/KMS chain).
- `docs/source/3_fractal_gas/3_fitness_manifold/06_cosmology.md:163` (equilibrium source vanishing attributed to detailed balance).

## Summary
Volume 3 asserts KMS via detailed balance at QSD equilibrium, but it also contains explicit statements of non-reversibility and structural time-irreversibility on the CST. As written, the KMS step is not unconditionally supported inside the Fractal Set framework. An unconditional fix requires either (a) a new Fractal-Set-native proof of equilibrium reversibility that does not rely on CST time reversal, or (b) decoupling OS2 from KMS and treating KMS as a corollary when reversible subdynamics are isolated.
