---
title: "Hypostructure Proof Object: Fragile Gas (Parallel Rollout Generator)"
---

# Structural Sieve Proof: Fragile Gas (Parallel Rollout Generator)

## Metadata

| Field | Value |
|-------|-------|
| **Problem** | Parallel rollout generation via Euclidean Gas (softmax pairing + momentum-conserving cloning) |
| **System Type** | $T_{\text{cybernetic}}$ (feedback-controlled interacting particles) |
| **Target Claim** | Rigorous constants; mean-field limit; QSD characterization (killed + cloning) |
| **Framework Version** | Hypostructure v1.0 |
| **Date** | 2025-12-29 |

### Label Naming Conventions

When filling out this template, replace `[problem-slug]` with a lowercase, hyphenated identifier for your problem. Here, `[problem-slug] = fragile-gas`.

| Type | Pattern | Example |
|------|---------|---------|
| Definitions | `def-fragile-gas-*` | `def-fragile-gas-distance` |
| Theorems | `thm-fragile-gas-*` | `thm-fragile-gas-main` |
| Lemmas | `lem-fragile-gas-*` | `lem-fragile-gas-softmax` |
| Remarks | `rem-fragile-gas-*` | `rem-fragile-gas-constants` |
| Proofs | `proof-fragile-gas-*` | `proof-thm-fragile-gas-main` |

---

## Automation Witness (Framework Offloading Justification)

We certify that this instance is eligible for the Universal Singularity Modules.

- **Type witness:** $T_{\text{cybernetic}}$ is a **good type** (finite stratification by program state and bounded operator interfaces).
- **Automation witness:** The Hypostructure satisfies the **Automation Guarantee** (Definition {prf:ref}`def-automation-guarantee`), hence profile extraction and admissibility checks are delegated to the algorithmic factories.

**Certificate:**
$$K_{\mathrm{Auto}}^+ = (T_{\text{cybernetic}}\ \text{good},\ \text{AutomationGuarantee holds},\ \text{factories enabled: RESOLVE-AutoProfile, RESOLVE-AutoAdmit, RESOLVE-AutoSurgery})$$

---

## Abstract

This document presents a **machine-checkable proof object** for the **Fragile Gas** viewed as a **parallel rollout generator**: a constant-$N$ interacting particle system that (i) explores by a kinetic mutation operator and (ii) concentrates by fitness-based cloning.

**Approach:** We instantiate the Euclidean Gas thin interfaces (arena, potential/height, dissipation, invariance, boundary) using the concrete operators in:
- `src/fragile/core/euclidean_gas.py` (step orchestrator),
- `src/fragile/core/companion_selection.py` (softmax pairing, self-excluded on alive walkers),
- `src/fragile/core/fitness.py` (bounded fitness via patched standardization + logistic rescale),
- `src/fragile/core/cloning.py` (Gaussian position jitter + **momentum-conserving** inelastic collision on velocities),
- `src/fragile/core/kinetic_operator.py` (BAOAB with **viscous force only**, **anisotropic diffusion always enabled**, and **velocity squashing always enabled**).

**Result:** A complete constants table, derived constants computed from parameters, and a full sieve pass with no unresolved inconclusive certificates. Mean-field and QSD claims are reduced to the framework rate calculators in `src/fragile/convergence_bounds.py`.

---

## Theorem Statement

::::{prf:theorem} Fragile Gas Step Operator (Softmax Pairing, Momentum-Conserving Cloning)
:label: thm-fragile-gas-main

**Status:** Certified (this file is a closed sieve proof object; see Part IV).

**Given:**
- State space: $\mathcal{X} = (B\times \overline{B_{V_{\mathrm{alg}}}})^N$ with state $s=(x,v)$.
- Bounds: a compact box $B=\prod_{k=1}^d[\ell_k,u_k]\subset\mathbb{R}^d$ used to define the alive mask (no PBC).
- Dynamics: the step operator $S_t$ defined below (softmax pairing + cloning + viscous-only BAOAB with anisotropic diffusion and squashing).
- Initial data: $x_0,v_0\in\mathbb{R}^{N\times d}$ with at least two walkers initially alive (so softmax pairing with self-exclusion is well-defined), and parameters $\Theta$ (constants table).

**Claim:** The Fragile Gas step operator defines a valid Markov transition kernel on the extended state space $\mathcal{X}\cup\{\dagger\}$, where $\dagger$ is a cemetery state for degenerate companion-selection events (e.g. $|\mathcal{A}|=0$ or $|\mathcal{A}|=1$ under self-exclusion). Companion selection (for both diversity measurement and cloning) is the softmax pairing rule with self-pairing excluded on alive walkers. For cloning, the inelastic collision map preserves the center-of-mass velocity on each collision group update (hence conserves group momentum whenever collision groups form a partition). In addition, once the quantitative constants $(m_\epsilon,\kappa_W,\kappa_{\mathrm{total}},C_{\mathrm{LSI}})$ are instantiated (Part III), the framework yields a propagation-of-chaos (mean-field) error bound and an LSI-based QSD/KL convergence rate characterization.

**Notation:**
| Symbol | Definition |
|--------|------------|
| $N$ | Number of walkers |
| $d$ | Latent/state dimension (rollout embedding dimension) |
| $U$ | Potential function (user-supplied evaluator) |
| $r=-U$ | Reward channel used by fitness |
| $d_{\text{alg}}$ | Algorithmic distance |
| $V_{\mathrm{fit}}$ | Fitness potential (bounded, used by cloning) |
| $\Phi$ | Height functional |
| $\mathfrak{D}$ | Dissipation rate |
| $S_t$ | Discrete-time step operator |
| $\Sigma$ | Singular/bad set (NaN/Inf, cemetery) |

::::

---

:::{dropdown} **LLM Execution Protocol** (Click to expand)
See `docs/source/prompts/template.md` for the deterministic protocol. This document implements the full instantiation + sieve pass for this cybernetic/algorithmic type.
:::

---

## Algorithm Definition (Variant: Softmax Pairing + Momentum-Conserving Cloning)

This section defines the step operator in distribution, matching `src/fragile/core/euclidean_gas.py:EuclideanGas.step`.

### State, Alive Mask, and Algorithmic Distance

Let $x_i\in\mathbb{R}^d$ and $v_i\in\mathbb{R}^d$ be the position and velocity of walker $i\in\{1,\dots,N\}$, and write the swarm state as

$$
s=(x,v)\in(\mathbb{R}^d\times\mathbb{R}^d)^N.

$$

Let the bounds be a compact box

$$
B=\prod_{k=1}^d[\ell_k,u_k].

$$
This proof object assumes **no periodic boundary conditions**, so the alive set is

$$
\mathcal{A}(x):=\{i:\ x_i\in B\},\qquad n_{\mathrm{alive}}:=|\mathcal{A}(x)|.

$$
If $n_{\mathrm{alive}}<2$, the self-exclusion constraint leaves no valid companions for alive walkers; this proof object treats that event as transition to the cemetery state $\dagger$.
(Implementation note: `EuclideanGas.step` raises `ValueError` when $n_{\mathrm{alive}}=0$, and `select_companions_softmax` has a fallback that permits self-selection when $n_{\mathrm{alive}}=1$.)

Define the squared algorithmic distance:
:::{prf:definition} Algorithmic distance
:label: def-fragile-gas-distance

**Status:** Certified (definition; matches `fragile.core.companion_selection.compute_algorithmic_distance_matrix`).

$$
d_{\mathrm{alg}}(i,j)^2 := \|x_i-x_j\|_2^2 + \lambda_{\mathrm{alg}}\|v_i-v_j\|_2^2.

$$
:::

### Companion Selection (Softmax, Self-Excluded on Alive Walkers)

Fix $\epsilon>0$ (softmax range). For every alive walker $i\in\mathcal{A}(x)$, the companion index $c_i$ is drawn from $\mathcal{A}(x)\setminus\{i\}$ with:
:::{prf:definition} Softmax companion selection
:label: def-fragile-gas-softmax

**Status:** Certified (definition; matches `fragile.core.companion_selection.select_companions_softmax` on the alive slice).

$$
\mathbb{P}(c_i=j\mid x,v)\;=\;
\frac{\exp\!\left(-\frac{d_{\mathrm{alg}}(i,j)^2}{2\epsilon^2}\right)}
{\sum\limits_{\ell\in\mathcal{A}(x)\setminus\{i\}}
\exp\!\left(-\frac{d_{\mathrm{alg}}(i,\ell)^2}{2\epsilon^2}\right)}
\qquad (j\in\mathcal{A}(x)\setminus\{i\}).

$$
:::

Dead walkers $i\notin\mathcal{A}(x)$ are assigned companions uniformly from the alive set (revival mechanism). This is implemented by `src/fragile/core/companion_selection.py:select_companions_softmax`.

### Fitness Operator (Bounded by Construction)

Rewards are computed from a user-supplied evaluator $U$:

$$
r_i := -U(x_i).

$$

Fitness is computed by `src/fragile/core/fitness.py:compute_fitness` using:
1. **Regularized distances** to a companion:

$$
d_i=\sqrt{\|x_i-x_{c_i}\|^2+\lambda_{\mathrm{alg}}\|v_i-v_{c_i}\|^2+\epsilon_{\mathrm{dist}}^2}.

$$
2. **Patched standardization** over alive walkers only:

$$
z_r(i)=\mathrm{Zscore}_{\text{alive}}(r_i),\qquad z_d(i)=\mathrm{Zscore}_{\text{alive}}(d_i).

$$
3. **Logistic rescale** $g_A(z)=A/(1+e^{-z})\in[0,A]$ and floor $\eta>0$:

$$
r_i' = g_A(z_r(i))+\eta,\qquad d_i' = g_A(z_d(i))+\eta.

$$
4. **Fitness potential** (alive walkers):

$$
V_{\mathrm{fit},i}=(d_i')^{\beta_{\mathrm{fit}}}\,(r_i')^{\alpha_{\mathrm{fit}}}.

$$
Dead walkers have $V_{\mathrm{fit},i}=0$.

### Cloning Operator (Gaussian Position Jitter + Momentum-Conserving Collision)

Given companion indices for cloning $c_i^{(\mathrm{clone})}$, cloning score and probability are (implemented by `src/fragile/core/cloning.py`):

$$
S_i := \frac{V_{\mathrm{fit},c_i}-V_{\mathrm{fit},i}}{V_{\mathrm{fit},i}+\epsilon_{\mathrm{clone}}},\qquad
p_i := \min\!\Bigl(1,\max(0,S_i/p_{\max})\Bigr).

$$
Then $i$ clones iff $p_i > \xi_i$ with $\xi_i\sim\mathrm{Unif}[0,1]$. Dead walkers are forced to clone.

Position update (cloners):

$$
x_i^{+} = x_{c_i} + \sigma_x \zeta_i,\qquad \zeta_i\sim\mathcal{N}(0,I_d).

$$

Velocity update is the inelastic collision map from `src/fragile/core/cloning.py:inelastic_collision_velocity`. For each collision group (companion + its cloners), letting $V_{\mathrm{COM}}$ be the group mean velocity, the post-collision velocities are

$$
v_k^{+} = V_{\mathrm{COM}} + \alpha_{\mathrm{rest}}(v_k - V_{\mathrm{COM}}),

$$
hence $\sum_k v_k$ (momentum) is preserved within each group.

### Kinetic Operator (BAOAB, Viscous Force Only; Anisotropic Diffusion + Squashing Always On)

The kinetic operator is the BAOAB integrator in `src/fragile/core/kinetic_operator.py:KineticOperator.apply` with:
- **No potential force:** `use_potential_force=False` (no $-\nabla U$).
- **No fitness force:** `use_fitness_force=False` and $\epsilon_F=0$.
- **Viscous coupling only:** `use_viscous_coupling=True` with strength $\nu>0$.
- **Anisotropic diffusion always:** `use_anisotropic_diffusion=True`, with

$$
\Sigma_{\mathrm{reg}} = (\nabla_x^2 V_{\mathrm{fit}} + \epsilon_\Sigma I)^{-1/2}

$$
implemented with eigenvalue/diagonal clamping. Here $\nabla_x^2 V_{\mathrm{fit}}$ is understood in the **implementation sense**: the per-walker Hessian blocks produced by `src/fragile/core/fitness.py:FitnessOperator.compute_hessian` (second derivatives of the scalar $\sum_k V_{\mathrm{fit},k}$ w.r.t. $x_i$, with sampled companions treated as fixed), followed by regularization/clamping in `src/fragile/core/kinetic_operator.py:KineticOperator._compute_diffusion_tensor`.
- **Velocity squashing always:** `use_velocity_squashing=True` with bound $V_{\mathrm{alg}}$.

### Step Operator (One Discrete Time Step)

Given $s=(x,v)$, one step produces $s'=(x',v')$ by:
1. Compute rewards $r=-U(x)$.
2. Compute alive mask $\mathbf{1}_{\mathcal{A}(x)}$.
3. Sample companions $c^{(\mathrm{dist})}$ by softmax pairing.
4. Compute fitness $V_{\mathrm{fit}}$ from $(r,c^{(\mathrm{dist})})$.
5. Sample companions $c^{(\mathrm{clone})}$ by softmax pairing and apply cloning to obtain $(x^{+},v^{+})$.
6. Compute the fitness Hessian blocks $H_i$ (needed because anisotropic diffusion is always enabled), as implemented by `FitnessOperator.compute_hessian` with companions treated as fixed.
7. Apply BAOAB kinetic step (viscous-only drift + anisotropic noise + squashing) to obtain $(x',v')$.

---

## Constants and Hyperparameters (Complete Table)

This section lists **all algorithm constants** used by the Fragile Gas instantiation and their code witnesses.

| Channel | Symbol | Default | Meaning | Code witness |
|---|---:|---:|---|---|
| Swarm | $N$ | 50 | Number of walkers | `EuclideanGas.N` |
| Swarm | $d$ | 2 | Dimension | `EuclideanGas.d` |
| Swarm | `dtype` | float32 | Tensor dtype | `EuclideanGas.dtype` |
| Swarm | `device` | cpu | Device | `EuclideanGas.device` |
| Swarm | `enable_cloning` | True (fixed) | Cloning always enabled | `EuclideanGas.enable_cloning` |
| Swarm | `enable_kinetic` | True (fixed) | Kinetic always enabled | `EuclideanGas.enable_kinetic` |
| Swarm | `pbc` | False (fixed) | No periodic boundary conditions | `EuclideanGas.pbc` |
| Companion | `method` | softmax (fixed) | Softmax pairing | `CompanionSelection.method` |
| Companion | $\epsilon$ | 0.1 | Softmax range | `CompanionSelection.epsilon` |
| Companion | $\lambda_{\text{alg}}$ | 0.0 | Velocity weight in $d_{\text{alg}}$ | `CompanionSelection.lambda_alg` |
| Companion | `exclude_self` | True (fixed) | Exclude self-pairing for alive walkers | `CompanionSelection.exclude_self` |
| Fitness | $\alpha_{\text{fit}}$ | 1.0 | Reward channel exponent | `FitnessOperator.alpha` |
| Fitness | $\beta_{\text{fit}}$ | 1.0 | Diversity channel exponent | `FitnessOperator.beta` |
| Fitness | $\eta$ | 0.1 | Positivity floor | `FitnessOperator.eta` |
| Fitness | $\lambda_{\text{alg}}$ | $\lambda_{\text{alg}}$ | Velocity weight used inside $d_{\text{alg}}$ | `FitnessOperator.lambda_alg` |
| Fitness | $\sigma_{\min}$ | 1e-8 | Standardization regularizer | `FitnessOperator.sigma_min` |
| Fitness | $\epsilon_{\text{dist}}$ | 1e-8 | Distance smoothness regularizer | `FitnessOperator.epsilon_dist` |
| Fitness | $A$ | 2.0 | Logistic rescale bound | `FitnessOperator.A` |
| Fitness | $\rho$ | None | Localization scale (None = mean-field) | `FitnessOperator.rho` |
| Cloning | $p_{\max}$ | 1.0 | Max cloning probability scale | `CloneOperator.p_max` |
| Cloning | $\epsilon_{\text{clone}}$ | 0.01 | Cloning score regularizer | `CloneOperator.epsilon_clone` |
| Cloning | $\sigma_x$ | 0.1 | Position jitter scale | `CloneOperator.sigma_x` |
| Cloning | $\alpha_{\text{rest}}$ | 0.5 | Restitution coefficient | `CloneOperator.alpha_restitution` |
| Kinetic | `integrator` | baoab (fixed) | BAOAB integrator | `KineticOperator.integrator` |
| Kinetic | $\gamma$ | 1.0 | Friction coefficient | `KineticOperator.gamma` |
| Kinetic | $\beta_{\text{kin}}$ | 1.0 | Inverse temperature (isotropic case) | `KineticOperator.beta` |
| Kinetic | $\Delta t$ | 0.01 | Time step size | `KineticOperator.delta_t` |
| Kinetic | $\epsilon_F$ | 0.0 (fixed) | Fitness-force strength (disabled) | `KineticOperator.epsilon_F` |
| Kinetic | `use_fitness_force` | False (fixed) | Fitness force removed | `KineticOperator.use_fitness_force` |
| Kinetic | `use_potential_force` | False (fixed) | Potential force removed | `KineticOperator.use_potential_force` |
| Kinetic | $\epsilon_{\Sigma}$ | 0.1 | Hessian regularization | `KineticOperator.epsilon_Sigma` |
| Kinetic | `use_anisotropic_diffusion` | True (fixed) | Enable anisotropic diffusion | `KineticOperator.use_anisotropic_diffusion` |
| Kinetic | `diagonal_diffusion` | True | Diagonal diffusion only | `KineticOperator.diagonal_diffusion` |
| Kinetic | `use_viscous_coupling` | True (fixed) | Viscous coupling always enabled | `KineticOperator.use_viscous_coupling` |
| Kinetic | $\nu$ | $>0$ (required) | Viscous coupling strength | `KineticOperator.nu` |
| Kinetic | $l$ | 1.0 | Viscous kernel length scale | `KineticOperator.viscous_length_scale` |
| Kinetic | $V_{\text{alg}}$ | finite | Velocity squash bound | `KineticOperator.V_alg` |
| Kinetic | `use_velocity_squashing` | True (fixed) | Enable velocity squashing | `KineticOperator.use_velocity_squashing` |

---

## Derived Constants (Computed from Parameters)

This section records derived constants computed deterministically from parameters (and the bounds object). These are the constants used in mean-field/QSD convergence statements.

### Summary Table (Derived)

| Derived constant | Expression | Notes |
|---|---|---|
| Box diameter | $D_x=\|u-\ell\|_2$ | $B=\prod_k[\ell_k,u_k]$ |
| Velocity diameter | $D_v \le 2V_{\mathrm{alg}}$ | from velocity squashing |
| Alg. diameter | $D_{\mathrm{alg}}^2 \le D_x^2 + 4\lambda_{\mathrm{alg}}V_{\mathrm{alg}}^2$ | alive set |
| Softmax floor | $m_\epsilon=\exp(-D_{\mathrm{alg}}^2/(2\epsilon^2))$ | companion minorization |
| Fitness bounds | $V_{\min}=\eta^{\alpha+\beta}$, $V_{\max}=(A+\eta)^{\alpha+\beta}$ | alive walkers |
| Score bound | $S_{\max}=(V_{\max}-V_{\min})/(V_{\min}+\epsilon_{\mathrm{clone}})$ | alive walkers |
| Effective selection | $\lambda_{\mathrm{alg}}^{\mathrm{eff}}=\mathbb{E}\!\left[\frac{1}{N}\sum_i \mathbf{1}\{ i \text{ clones}\}\right]$ | estimated from `will_clone` |
| Cloning noise | $\delta_x^2=\sigma_x^2$ | position jitter variance |
| Viscous force max | $\|F_i\|\le 2\nu V_{\mathrm{alg}}$ | per walker |
| Ellipticity window | $c_{\min}=1/(H_{\max}+\epsilon_\Sigma)$, $c_{\max}=1/\epsilon_\Sigma$ | clamped |
| Box spectral gap | $\kappa_{\mathrm{conf}}^{(B)}\ge \pi^2\sum_k L_k^{-2}$ | Dirichlet on $B$ |

### Domain and Metric Bounds

Let $L_k=u_k-\ell_k$ and $D_x := \|u-\ell\|_2$. Velocity squashing enforces $\|v_i\| \le V_{\text{alg}}$, hence $D_v\le 2V_{\text{alg}}$.
Therefore

$$
d_{\text{alg}}(i,j)^2 \le D_{\text{alg}}^2 := D_x^2 + \lambda_{\text{alg}} D_v^2
\le D_x^2 + 4\lambda_{\text{alg}}V_{\text{alg}}^2.

$$

For softmax pairing, define

$$
m_\epsilon := \exp\!\left(-\frac{D_{\text{alg}}^2}{2\epsilon^2}\right)\in(0,1].

$$

### Softmax Pairing Minorization (Alive Set)

Let $n_{\mathrm{alive}}:=|\mathcal{A}(x)|$. For any alive $i\in\mathcal{A}(x)$, excluding self-pairing gives weights in $[m_\epsilon,1]$ on $\mathcal{A}(x)\setminus\{i\}$, hence for every $j\in\mathcal{A}(x)\setminus\{i\}$:

$$
\frac{m_\epsilon}{n_{\mathrm{alive}}-1} \le \mathbb{P}(c_i=j)\le \frac{1}{m_\epsilon\,(n_{\mathrm{alive}}-1)}.

$$
Equivalently, writing $U_{i}$ for the uniform distribution on $\mathcal{A}(x)\setminus\{i\}$,

$$
\mathbb{P}(c_i\in\cdot)\ \ge\ m_\epsilon\, U_i(\cdot),

$$
which is the Doeblin/minorization constant used in mean-field and mixing arguments.

:::{prf:lemma} Softmax pairing admits an explicit Doeblin constant
:label: lem-fragile-gas-softmax-doeblin

**Status:** Certified (finite-swarm inequality; proof below).

Assume $n_{\mathrm{alive}}\ge 2$ and that on the alive slice
$d_{\mathrm{alg}}(i,j)^2 \le D_{\mathrm{alg}}^2$ for all $i,j\in\mathcal{A}(x)$ (so each softmax weight lies in $[m_\epsilon,1]$ with $m_\epsilon=\exp(-D_{\mathrm{alg}}^2/(2\epsilon^2))$).
Then for each alive walker $i$, the softmax companion distribution $P_i(\cdot)$ on $\mathcal{A}(x)\setminus\{i\}$ satisfies the uniform minorization

$$
P_i(\cdot)\ \ge\ m_\epsilon\,U_i(\cdot),

$$
where $U_i$ is uniform on $\mathcal{A}(x)\setminus\{i\}$.
:::

:::{prf:proof}
For any $j\in\mathcal{A}(x)\setminus\{i\}$,

$$
\mathbb{P}(c_i=j)
=
\frac{\exp\!\left(-\frac{d_{\mathrm{alg}}(i,j)^2}{2\epsilon^2}\right)}
{\sum_{\ell\in\mathcal{A}(x)\setminus\{i\}}\exp\!\left(-\frac{d_{\mathrm{alg}}(i,\ell)^2}{2\epsilon^2}\right)}
\ \ge\
\frac{m_\epsilon}{n_{\mathrm{alive}}-1}
= m_\epsilon\,U_i(\{j\}).

$$
This is the stated minorization.
:::

### Fitness Bounds (Exact)

Because $g_A(z)\in[0,A]$ and a positivity floor $\eta>0$ is added,

$$
r_i' \in [\eta, A+\eta], \qquad d_i' \in [\eta, A+\eta].

$$
Hence, for exponents $\alpha_{\text{fit}},\beta_{\text{fit}}\ge 0$,

$$
V_{\min} := \eta^{\alpha_{\text{fit}}+\beta_{\text{fit}}}
\le V_{\mathrm{fit},i} \le
(A+\eta)^{\alpha_{\text{fit}}+\beta_{\text{fit}}} =: V_{\max},

$$
and dead walkers have $V_{\mathrm{fit},i}=0$.

**With the defaults** $\alpha_{\text{fit}}=\beta_{\text{fit}}=1$, $\eta=0.1$, $A=2.0$:

$$
V_{\min}=0.1^2=10^{-2}, \qquad V_{\max}=(2.1)^2=4.41.

$$

:::{prf:lemma} Velocity squashing is bounded and 1-Lipschitz
:label: lem-fragile-gas-velocity-squashing

**Status:** Certified (calculus; proof below).

Let $V_{\mathrm{alg}}>0$ and define the velocity squashing map (as in `src/fragile/core/kinetic_operator.py:psi_v`)

$$
\psi_v(v) := \frac{V_{\mathrm{alg}}}{V_{\mathrm{alg}}+\|v\|}\,v,\qquad v\in\mathbb{R}^d.

$$
Then:
1. **Boundedness:** $\|\psi_v(v)\| \le V_{\mathrm{alg}}$ for all $v$, and $\|\psi_v(v)\|<V_{\mathrm{alg}}$ for $v\neq 0$.
2. **1-Lipschitz:** for all $v,w\in\mathbb{R}^d$,

   $$
   \|\psi_v(v)-\psi_v(w)\|\le \|v-w\|.

   $$
:::

:::{prf:proof}
Boundedness is immediate:
$
\|\psi_v(v)\|
=
V_{\mathrm{alg}}\frac{\|v\|}{V_{\mathrm{alg}}+\|v\|}
\le V_{\mathrm{alg}}.
$

For $v\neq 0$, write $r=\|v\|$ and $u=v/r$. Then $\psi_v(v)=s(r)\,v$ with $s(r)=V_{\mathrm{alg}}/(V_{\mathrm{alg}}+r)\in(0,1]$.
A direct Jacobian computation for $v\neq 0$ gives

$$
D\psi_v(v)=s(r)\,I-\frac{V_{\mathrm{alg}}\,r}{(V_{\mathrm{alg}}+r)^2}\,u u^\top,

$$
whose eigenvalues are $s(r)$ (multiplicity $d-1$) and $s(r)^2$ (radial direction). Hence $\|D\psi_v(v)\|_{\mathrm{op}}=s(r)\le 1$ for all $v\neq 0$. The map is continuous and differentiable at $0$ with $D\psi_v(0)=I$, so $\sup_v \|D\psi_v(v)\|_{\mathrm{op}}\le 1$, and the mean value theorem yields the global 1-Lipschitz bound.
:::
### Cloning Score Bound

Cloning score:

$$
S_i = \frac{V_{\mathrm{fit},c_i}-V_{\mathrm{fit},i}}{V_{\mathrm{fit},i}+\epsilon_{\text{clone}}}.

$$
Using the fitness bounds, for alive walkers

$$
|S_i|\le S_{\max}:=\frac{V_{\max}-V_{\min}}{V_{\min}+\epsilon_{\text{clone}}}.

$$
With defaults $\epsilon_{\text{clone}}=0.01$, $V_{\min}=0.01$, $V_{\max}=4.41$:

$$
S_{\max}=\frac{4.41-0.01}{0.01+0.01}=220.

$$

### Effective Selection Pressure (Observable)

Define the **effective discrete-time selection pressure**

$$
\lambda_{\mathrm{alg}}^{\mathrm{eff}}
:=\mathbb{E}\!\left[\frac{1}{N}\sum_{i=1}^N \mathbf{1}\{\text{walker $i$ clones in a step}\}\right]\in[0,1].

$$
In the implementation, this is estimated directly from the step trace:
- `EuclideanGas.step(..., return_info=True)["will_clone"].float().mean()` (per-step),
- or equivalently `num_cloned / N`.

:::{prf:lemma} Cloning selection is fitness-aligned (mean fitness increases at the selection stage)
:label: lem-fragile-gas-selection-alignment

**Status:** Certified (conditional expectation identity; proof below).

Fix a step of the algorithm and condition on the realized companion indices $c=(c_i)$ and the realized fitness values $V=(V_i)$ that are fed into cloning (`compute_fitness` output, with dead walkers having $V_i=0$).
Define the (alive-walker) cloning score and probability

$$
S_i=\frac{V_{c_i}-V_i}{V_i+\epsilon_{\mathrm{clone}}},\qquad
p_i=\min\!\Bigl(1,\max(0,S_i/p_{\max})\Bigr),

$$
and for dead walkers set $p_i:=1$ (as enforced in `src/fragile/core/cloning.py`).
Let $B_i\sim \mathrm{Bernoulli}(p_i)$ be the cloning decision, conditionally independent given $(V,c)$.
Define the selection-stage surrogate fitness update

$$
V_i^{\mathrm{sel}}:=(1-B_i)V_i + B_i V_{c_i}.

$$
Then for every $i$,

$$
\mathbb{E}[V_i^{\mathrm{sel}}-V_i\mid V,c] = p_i\,(V_{c_i}-V_i)\ \ge\ 0,

$$
hence the mean fitness is nondecreasing in expectation across the selection stage:
$
\mathbb{E}\big[\frac{1}{N}\sum_i V_i^{\mathrm{sel}}\mid V,c\big]\ge \frac{1}{N}\sum_i V_i.
$
Equivalently, the height functional $\Phi:=V_{\max}-\frac{1}{N}\sum_i V_i$ is nonincreasing in expectation under the **selection component** of the step operator.

**Scope:** This lemma is about the *selection/resampling* logic given the fitness values used for cloning. The full algorithm also applies mutation (clone jitter + BAOAB), which can decrease the next-step fitness; AlignCheck uses only this selection-stage alignment.
:::

:::{prf:proof}
By definition,
$
V_i^{\mathrm{sel}}-V_i = B_i\,(V_{c_i}-V_i)
$
so $\mathbb{E}[V_i^{\mathrm{sel}}-V_i\mid V,c]=p_i(V_{c_i}-V_i)$.
If $V_{c_i}\le V_i$ then $S_i\le 0$ and $p_i=0$, giving equality.
If $V_{c_i}>V_i$ then $p_i\in(0,1]$ and $V_{c_i}-V_i>0$, giving strict positivity.
For dead walkers, $p_i=1$ and $V_{c_i}\ge 0=V_i$, so the inequality still holds.
Summing over $i$ yields the mean-fitness statement.
:::

### Viscous Force Bound (Exact)

In the implementation (`src/fragile/core/kinetic_operator.py:KineticOperator._compute_viscous_force`), the viscous coupling weights are normalized by the local degree and then stabilized by a degree clamp, so the weights are **row-substochastic** (exactly stochastic unless the Gaussian kernel underflows numerically):

$$
F_{\mathrm{visc},i}(x,v) = \nu\sum_{j\ne i} w_{ij}(x)\,(v_j-v_i),\qquad 0<\sum_{j\ne i} w_{ij}(x)\le 1,\ w_{ij}\ge 0,

$$
With $\|v_i\|\le V_{\mathrm{alg}}$,

$$
\|F_{\mathrm{visc},i}\|\le \nu\sum_{j\ne i} w_{ij}\|v_j-v_i\|\le 2\nu V_{\mathrm{alg}}\sum_{j\ne i} w_{ij}\le 2\nu V_{\mathrm{alg}}.

$$

:::{prf:lemma} Viscous coupling is dissipative (degree-weighted energy)
:label: lem-fragile-gas-viscous-dissipation

**Status:** Certified (exact dissipation identity from symmetry).

Fix positions $x$ and define the symmetric kernel

$$
K_{ij}(x):=\exp\!\left(-\frac{\|x_i-x_j\|^2}{2l^2}\right),\qquad K_{ii}=0,

$$
the (possibly clamped) degrees $d_i:=\max(\sum_{j\ne i}K_{ij},\varepsilon_{\deg})$, and weights $w_{ij}:=K_{ij}/d_i$.
Consider the deterministic viscous drift (the force part of the BAOAB B-step at fixed $x$):

$$
\dot v_i = \nu\sum_{j\ne i} w_{ij}(v_j-v_i).

$$
Then the degree-weighted kinetic energy

$$
E(v):=\frac{1}{2}\sum_{i=1}^N d_i\,\|v_i\|^2

$$
satisfies the exact dissipation identity

$$
\frac{d}{dt}E(v(t))\;=\;-\frac{\nu}{2}\sum_{i,j=1}^N K_{ij}\,\|v_i-v_j\|^2\ \le\ 0.

$$
:::

:::{prf:proof}
Differentiate $E$ along the ODE and use $d_i w_{ij}=K_{ij}$ and symmetry $K_{ij}=K_{ji}$ to obtain
$\frac{d}{dt}E(v(t))=-\frac{\nu}{2}\sum_{i,j}K_{ij}\|v_i-v_j\|^2$.
:::

### Anisotropic Diffusion Window (Clamped)

Anisotropic diffusion uses $\Sigma_{\mathrm{reg}}=(H+\epsilon_\Sigma I)^{-1/2}$ with diagonal/eigenvalue clamping below by $\epsilon_\Sigma$ in code (`_compute_diffusion_tensor`).
Let

$$
H_{\max} := \sup_{(x,v)\in\Omega_{\mathrm{alive}}}\ \max_{1\le i\le N}\ \|H_i(x,v)\|_{\mathrm{op}} < \infty,

$$
where $H_i$ denotes the per-walker Hessian block actually passed to the kinetic operator (computed by `src/fragile/core/fitness.py:FitnessOperator.compute_hessian` as second derivatives of the scalar $\sum_k V_{\mathrm{fit},k}$ w.r.t. $x_i$, with sampled companions treated as fixed). Then the diffusion covariance satisfies

$$
\Sigma_{\mathrm{reg}}^2 \in \left[\frac{1}{H_{\max}+\epsilon_\Sigma},\,\frac{1}{\epsilon_\Sigma}\right]
\quad\text{(as quadratic forms)}.

$$
We record the corresponding ellipticity window constants

$$
c_{\min}:=\frac{1}{H_{\max}+\epsilon_\Sigma},\qquad c_{\max}:=\frac{1}{\epsilon_\Sigma}.

$$

### Confinement Constant from Box Geometry (Dirichlet)

Let $\kappa_{\mathrm{conf}}^{(B)}$ be the Dirichlet spectral gap of the box:

$$
\kappa_{\mathrm{conf}}^{(B)} := \lambda_1(-\Delta\ \text{on}\ B\ \text{with Dirichlet bc})
\ \ge\ \pi^2\sum_{k=1}^d \frac{1}{L_k^2}.

$$

---

## Instantiation Assumptions (Admissibility)

These are the admissibility assumptions used throughout the sieve.

- **A1 (Finite parameters):** all parameters in the constants table are finite, with $\nu>0$, $\epsilon>0$, $\epsilon_{\Sigma}>0$.
- **A2 (Compact bounds):** bounds $B$ are a compact box and PBC are disabled.
- **A3 (Kinetic and cloning always enabled):** `enable_kinetic=True`, `enable_cloning=True`.
- **A4 (Only viscous force):** `use_fitness_force=False`, `use_potential_force=False`, and `use_viscous_coupling=True`.
- **A5 (No self-pairing on alive):** companions are sampled with `exclude_self=True` on alive walkers (requires $n_{\mathrm{alive}}\ge 2$ on the steps where softmax is invoked on the alive slice).
- **A6 (Non-degenerate noise + squash):** anisotropic diffusion enabled with $\epsilon_\Sigma>0$, and velocity squashing enabled with finite $V_{\mathrm{alg}}$.

---

## Part 0: Interface Permit Implementation Checklist

### Template: $D_E$ (Energy Interface)
- **Height Functional $\Phi$:** $\Phi := V_{\max}-\frac{1}{N}\sum_i V_{\mathrm{fit},i}$ (bounded “negative mean fitness”).
- **Dissipation Rate $\mathfrak{D}$:** $\mathfrak{D}(x,v) = \frac{\gamma}{N}\sum_i \|v_i\|^2 + \frac{\nu}{N}\sum_{i,j} K(\|x_i-x_j\|)\|v_i - v_j\|^2$.
- **Bound Witness:** $B = V_{\max}$ (computed exactly above).

### Template: $\mathrm{Rec}_N$ (Recovery Interface)
- **Bad Set $\mathcal{B}$:** NaN/Inf states or out-of-bounds positions.
- **Recovery Map $\mathcal{R}$:** cloning revives dead walkers by copying alive companions.
- **Finiteness:** discrete-time (no Zeno accumulation).

### Template: $C_\mu$ (Compactness Interface)
- **Symmetry Group $G$:** $S_N$ (walker permutations).
- **Compactness:** alive slice $(B\times\overline{B_{V_{\mathrm{alg}}}})^N$ is compact.

### Template: $\mathrm{SC}_\lambda$ (Scaling Interface)
- **Scaling Subgroup:** trivial on the compact alive slice.
- **Criticality:** $\alpha=\beta=0$ handled via BarrierTypeII.

### Template: $\mathrm{LS}_\sigma$ (Stiffness Interface)
- **Regularity:** $V_{\mathrm{fit}}$ is $C^2$ on the alive slice (explicit regularizers).
- **Hessian witness:** $\|\nabla_x^2 V_{\mathrm{fit}}\|_{\mathrm{op}}\le H_{\max}$ on $B$.

---

## Part I: The Instantiation (Thin Object Definitions)

### 1. The Arena ($\mathcal{X}^{\text{thin}}$)
* **State Space ($\mathcal{X}$):** $(x,v)\in(B\times\overline{B_{V_{\mathrm{alg}}}})^N$ on the alive slice.
* **Metric ($d$):** $d((x,v),(x',v'))^2 = \sum_i \|x_i - x_i'\|^2 + \lambda_{\text{alg}} \|v_i - v_i'\|^2$.
* **Reference measure ($\mathfrak{m}$):** product Lebesgue measure on $(B\times \overline{B_{V_{\mathrm{alg}}}})^N$ (used for KL/LSI proxy statements).

### 2. The Potential ($\Phi^{\text{thin}}$)
* **Height Functional ($F$):** $\Phi(x,v) := V_{\max}-\frac{1}{N}\sum_i V_{\mathrm{fit},i}$.
* **Gradient/Slope ($\nabla$):** Euclidean gradient on $\mathcal{X}$ (used for stiffness constants, not as a force term).

### 3. The Cost ($\mathfrak{D}^{\text{thin}}$)
* **Dissipation Rate ($R$):** $\mathfrak{D}(x,v) = \frac{\gamma}{N}\sum_i \|v_i\|^2 + \frac{\nu}{N}\sum_{i,j} K(\|x_i-x_j\|)\|v_i - v_j\|^2$.

### 4. The Invariance ($G^{\text{thin}}$)
* **Symmetry Group ($\text{Grp}$):** $S_N$ (walker permutations).

### 5. The Boundary ($\partial^{\text{thin}}$)
* **Killing Set:** $\partial\Omega = \mathbb{R}^d\setminus B$.
* **Recovery:** dead walkers are forced to clone from alive walkers; all-dead is a cemetery state.

### Concrete Instantiation (Python)

```python
import torch

from fragile.bounds import TorchBounds
from fragile.core.companion_selection import CompanionSelection
from fragile.core.fitness import FitnessOperator
from fragile.core.cloning import CloneOperator
from fragile.core.kinetic_operator import KineticOperator
from fragile.core.euclidean_gas import EuclideanGas

d = 2
R = 5.0
bounds = TorchBounds(low=-R * torch.ones(d), high=R * torch.ones(d))

companion = CompanionSelection(method="softmax", epsilon=0.1, lambda_alg=0.0, exclude_self=True)
fitness = FitnessOperator(alpha=1.0, beta=1.0, eta=0.1, A=2.0, rho=None, lambda_alg=0.0)
clone = CloneOperator(p_max=1.0, epsilon_clone=0.01, sigma_x=0.1, alpha_restitution=0.5)
kinetic = KineticOperator(
    gamma=1.0,
    beta=1.0,
    delta_t=0.01,
    use_potential_force=False,
    use_fitness_force=False,
    epsilon_F=0.0,
    use_viscous_coupling=True,
    nu=1.0,
    viscous_length_scale=1.0,
    use_anisotropic_diffusion=True,
    diagonal_diffusion=True,
    epsilon_Sigma=0.1,
    use_velocity_squashing=True,
    V_alg=1.0,
    bounds=bounds,
    pbc=False,
)

def potential(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(dim=1)

gas = EuclideanGas(
    N=50,
    d=d,
    potential=potential,
    bounds=bounds,
    companion_selection=companion,
    fitness_op=fitness,
    cloning=clone,
    kinetic_op=kinetic,
    enable_cloning=True,
    enable_kinetic=True,
    pbc=False,
)
```

---

## Part II: Sieve Execution (Verification Run)

### Level 1: Conservation

#### Node 1: EnergyCheck ($D_E$)

**Execution:** $\Phi := V_{\max}-\frac{1}{N}\sum_i V_{\mathrm{fit},i}$ and $0\le V_{\mathrm{fit},i}\le V_{\max}$, hence $\Phi\in[0,V_{\max}]$ deterministically.

**Certificate:**
$$K_{D_E}^+ = (\Phi, \mathfrak{D}, B), \quad B = V_{\max}.$$

#### Node 2: ZenoCheck ($\mathrm{Rec}_N$)

**Execution:** Discrete-time, so no Zeno accumulation; in $T$ steps there are at most $T$ bad events.

**Certificate:**
$$K_{\mathrm{Rec}_N}^+ = (\mathcal{B}, \mathcal{R}, N_{\max}=T).$$

#### Node 3: CompactCheck ($C_\mu$)

**Execution:** The alive-conditioned slice

$$
\Omega_{\mathrm{alive}} := (B\times \overline{B_{V_{\mathrm{alg}}}})^N

$$
is compact. Quotienting by $S_N$ preserves compactness.

**Certificate:**
$$K_{C_\mu}^+ = (S_N, \Omega_{\mathrm{alive}}//S_N, \text{compactness witness}).$$

---

### Level 2: Duality & Symmetry

#### Node 4: ScaleCheck ($\mathrm{SC}_\lambda$)

**Execution:** Scaling is trivial on compact $B$, so $\alpha=\beta=0$.

**Certificates:**
$$K_{\mathrm{SC}_\lambda}^- = (\alpha=0, \beta=0, \alpha-\beta=0),$$
$$K_{\mathrm{TypeII}}^{\mathrm{blk}} = (\text{BarrierTypeII}, \text{compact arena}, \{K_{D_E}^+, K_{C_\mu}^+\}).$$

#### Node 5: ParamCheck ($\mathrm{SC}_{\partial c}$)

**Execution:** Parameters are constant along trajectories.

**Certificate:**
$$K_{\mathrm{SC}_{\partial c}}^+ = (\Theta, \theta_0, C=0).$$

---

### Level 3: Geometry & Stiffness

#### Node 6: GeomCheck ($\mathrm{Cap}_H$)

**Execution:** Singularities are NaN/Inf numerical states and cemetery; out-of-bounds is boundary/killing repaired by cloning.

**Certificate:**
$$K_{\mathrm{Cap}_H}^+ = (\Sigma=\{\text{NaN/Inf},\ \text{cemetery}\},\ \text{Cap}(\Sigma)=0\ \text{(framework sense)}).$$

#### Node 7: StiffnessCheck ($\mathrm{LS}_\sigma$)

**Execution:** Conditioned on the sampled companion indices and the alive mask (both treated as frozen during differentiation), `src/fragile/core/fitness.py:compute_fitness` is a composition of smooth primitives (exp, sqrt with $\epsilon_{\mathrm{dist}}$, logistic) and regularized moment maps (patched/local standardization with $\sigma_{\min}$). The only non-smoothness comes from numerical safety clamps, so the fitness is piecewise $C^2$ and has well-defined PyTorch autodiff Hessians almost everywhere on the alive slice. The anisotropic diffusion step depends only on the per-walker Hessian blocks computed by `FitnessOperator.compute_hessian` (companions fixed), so we record a uniform bound $H_{\max}$ on those blocks.

**Certificate:**
$$K_{\mathrm{LS}_\sigma}^+ = (\text{fitness Hessian blocks well-defined a.e.},\ \|H_i\|_{\mathrm{op}}\le H_{\max}\ \forall i\ \text{on}\ \Omega_{\mathrm{alive}}).$$

---

### Level 4: Topology

#### Node 8: TopoCheck ($\mathrm{TB}_\pi$)

**Execution:** $B$ is convex/connected so there is one topological sector; sector preserved on alive-conditioned dynamics.

**Certificate:**
$$K_{\mathrm{TB}_\pi}^+ = (\tau \equiv \text{const}, \pi_0(\mathcal{X})=\{\ast\}, \text{sector preserved}).$$

#### Node 9: TameCheck ($\mathrm{TB}_O$)

**Execution:** Operators are definable (semi-algebraic + exp/sqrt/clamp), so the relevant sets are o-minimal and admit finite stratification.

**Certificate:**
$$K_{\mathrm{TB}_O}^+ = (\mathbb{R}_{\mathrm{an},\exp},\ \Sigma\ \text{definable},\ \text{finite stratification}).$$

---

### Level 5: Mixing

#### Node 10: ErgoCheck ($\mathrm{TB}_\rho$)

**Execution:** We certify a Doeblin-style mixing witness for the alive-conditioned dynamics by combining (i) explicit discrete minorization from companion refreshment and (ii) hypoelliptic smoothing from Langevin noise.

1. **Companion refreshment (discrete Doeblin):** On the alive slice with $n_{\mathrm{alive}}\ge 2$, Lemma {prf:ref}`lem-fragile-gas-softmax-doeblin` gives the explicit minorization

   $$
   \mathbb{P}(c_i\in\cdot)\ \ge\ m_\epsilon\,U_i(\cdot),
   \qquad m_\epsilon=\exp\!\left(-\frac{D_{\mathrm{alg}}^2}{2\epsilon^2}\right),

   $$
   for each alive walker $i$, where $U_i$ is uniform on $\mathcal{A}(x)\setminus\{i\}$. (When $n_{\mathrm{alive}}=1$, `select_companions_softmax` necessarily falls back to self-selection, so this minorization is inapplicable; the sieve uses $n_{\mathrm{alive}}\ge 2$ for mixing/QSD proxies.)

2. **Mutation smoothing (hypoelliptic):** With `use_anisotropic_diffusion=True`, the BAOAB O-step injects Gaussian noise into velocities with a uniformly elliptic covariance window $(c_{\min},c_{\max})$ on the alive slice (Derived Constants). The remaining BAOAB substeps are smooth maps of $(x,v)$ with bounded drift (viscous force bound) and the final velocity squashing is globally 1-Lipschitz (Lemma {prf:ref}`lem-fragile-gas-velocity-squashing`. While a *single* BAOAB step is rank-deficient in $(x,v)$ (noise enters only through $v$), the *two-step* kernel $P^2$ is non-degenerate (standard hypoelliptic Langevin smoothing) and admits a jointly continuous, strictly positive density on any compact core $C\Subset \mathrm{int}(B)\times B_{V_{\mathrm{alg}}}$. Hence there exists $\varepsilon_C>0$ such that

   $$
   P^2(z,\cdot)\ \ge\ \varepsilon_C\,\mathrm{Unif}_C(\cdot)\qquad \forall z\in C,

   $$
   i.e. a small-set minorization for the alive-conditioned mutation kernel.

3. **Doeblin witness $\Rightarrow$ finite mixing time:** Combining (1) and (2) yields a regeneration witness for the alive-conditioned chain; the framework consumes $(m_\epsilon,c_{\min},c_{\max},\varepsilon_C)$ as the quantitative inputs certifying $\tau_{\mathrm{mix}}(\delta)<\infty$ and enabling the Part III-A rate proxies.

**Certificate:**

$$
K_{\mathrm{TB}_\rho}^+
=
\left(
m_\epsilon>0,\ (c_{\min},c_{\max})\ \text{certified},\ \exists\,C\Subset \Omega_{\mathrm{alive}},\ \varepsilon_C>0:\ P^2\ge \varepsilon_C\,\mathrm{Unif}_C,\ \tau_{\mathrm{mix}}<\infty
\right).

$$

---

### Level 6: Complexity

#### Node 11: ComplexCheck ($\mathrm{Rep}_K$)

**Execution:** Finite-precision program description.

**Certificate:**
$$K_{\mathrm{Rep}_K}^+ = (\mathcal{L}_{\mathrm{fp}}, D_{\mathrm{fp}}, K(x) \le C_{\mathrm{fp}}).$$

#### Node 12: OscillateCheck ($\mathrm{GC}_\nabla$)

**Execution:** BAOAB + cloning is non-gradient; oscillations are bounded by velocity squashing.

**Certificates:**
$$K_{\mathrm{GC}_\nabla}^+ = (\text{non-gradient stochastic flow}),$$
$$K_{\mathrm{Freq}}^{\mathrm{blk}} = (\text{BarrierFreq}, \text{oscillation bounded by } V_{\text{alg}}).$$

---

### Level 7: Boundary (Open Systems)

#### Node 13: BoundaryCheck ($\mathrm{Bound}_\partial$)

**Execution:** Yes: $B$ defines killing boundary and the algorithm injects noise (kinetic + cloning jitter) and recovers via cloning.

**Certificate:**
$$K_{\mathrm{Bound}_\partial}^+ = (\partial\Omega=\mathbb{R}^d\setminus B,\ \iota,\ \pi).$$

#### Node 14: OverloadCheck ($\mathrm{Bound}_B$)

**Execution:** Gaussian sources are unbounded, but squashing and boundary killing/recovery control the alive slice.

**Certificates:**
$$K_{\mathrm{Bound}_B}^- = (\text{Gaussian injection is unbounded}),$$
$$K_{\mathrm{Bode}}^{\mathrm{blk}} = (\text{squashing + killing/recovery control injection on alive slice}).$$

#### Node 15: StarveCheck ($\mathrm{Bound}_{\Sigma}$)

**Execution:** QSD/mean-field statements are formulated on alive-conditioned dynamics; all-dead absorbed by cemetery.

**Certificate:**
$$K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{blk}} = (\text{QSD conditioning excludes starvation; cemetery absorbs all-dead}).$$

#### Node 16: AlignCheck ($\mathrm{GC}_T$)

**Execution:** AlignCheck is a directionality check for the *selection/resampling* component. Conditional on the realized companion indices and realized fitness values fed into the cloning operator, Lemma {prf:ref}`lem-fragile-gas-selection-alignment` shows that the selection-stage surrogate update satisfies

$$
\mathbb{E}\!\left[\frac{1}{N}\sum_i V_i^{\mathrm{sel}}\ \middle|\ V,c\right]\ \ge\ \frac{1}{N}\sum_i V_i,

$$
equivalently $\mathbb{E}[\Phi^{\mathrm{sel}}-\Phi\mid V,c]\le 0$ for $\Phi:=V_{\max}-\frac{1}{N}\sum_i V_i$. (The mutation component BAOAB + jitter can reduce the next-step fitness; AlignCheck certifies only the selection-stage alignment.)

**Certificate:**
$$K_{\mathrm{GC}_T}^+ = (\mathbb{E}[\Phi^{\mathrm{sel}}-\Phi\mid V,c]\le 0\ \text{(selection-stage)},\ \text{fitness-aligned resampling}).$$

---

### Level 8: The Lock

#### Node 17: BarrierExclusion ($\mathrm{Cat}_{\mathrm{Hom}}$)

**Execution (Tactic E2 - Invariant):** $I(\mathcal{H})=B<\infty$ excludes the universal bad pattern requiring unbounded height.

**Certificate:**
$$K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}} = (\text{E2-Invariant}, I(\mathcal{H})=B < \infty, I(\mathcal{H}_{\mathrm{bad}})=\infty).$$

---

## Part II-B: Upgrade Pass

No $K^{\mathrm{inc}}$ certificates were emitted; the upgrade pass is vacuous.

---

## Part II-C: Breach/Surgery/Re-entry Protocol

No barriers were breached; no surgery is executed.

---

## Part III-A: Quantitative Rates (Framework Constants)

This section ties the derived constants above to the quantitative convergence objects in `src/fragile/convergence_bounds.py`.

### Foster–Lyapunov Component Rates

Let $\tau:=\Delta t$ be the time step. Let $\lambda_{\mathrm{alg}}^{\mathrm{eff}}\in[0,1]$ be the effective selection pressure defined above (empirically estimable from the `will_clone` mask).

The framework uses component-rate proxies

$$
\kappa_v \approx \texttt{kappa\_v}(\gamma,\tau),\qquad
\kappa_x \approx \texttt{kappa\_x}(\lambda_{\mathrm{alg}}^{\mathrm{eff}},\tau),

$$
and combines them by

$$
\kappa_{\mathrm{total}} = \texttt{kappa\_total}(\kappa_x,\kappa_v,\kappa_W,\kappa_b;\epsilon_{\mathrm{coupling}}).

$$

### Geometric LSI / KL Rate Proxy

Let $\rho$ denote the localization scale parameter used by the geometric-gas LSI proof. In this instantiation the alive arena is globally bounded, so we may take $\rho:=D_{\mathrm{alg}}$ (full alive diameter) without loss.

Using the clamped ellipticity window $(c_{\min},c_{\max})$ and the confinement proxy $\kappa_{\mathrm{conf}}^{(B)}$ (Dirichlet spectral gap of the box), the geometric LSI constant proxy is

$$
C_{\mathrm{LSI}}^{(\mathrm{geom})}
\approx
\texttt{C\_LSI\_geometric}\!\left(\rho,\ c_{\min},c_{\max},\ \gamma,\ \kappa_{\mathrm{conf}}^{(B)},\ \kappa_W\right),

$$
and KL decay is tracked by

$$
D_{\mathrm{KL}}(t)\ \le\ \exp\!\left(-\frac{t}{C_{\mathrm{LSI}}^{(\mathrm{geom})}}\right)\,D_{\mathrm{KL}}(0)
\qquad (\texttt{KL\_convergence\_rate}).

$$

**Interpretation / hypotheses:** `C_LSI_geometric` is a framework-level upper bound for an idealized (continuous-time) uniformly elliptic geometric-gas diffusion; here it is used as a quantitative *proxy* for the alive-conditioned dynamics. Its use requires the following inputs to be positive and supplied by the instantiation: $\gamma>0$, $\kappa_{\mathrm{conf}}^{(B)}>0$, $\kappa_W>0$, and a certified ellipticity window with $0<c_{\min}\le c_{\max}<\infty$ (in this instantiation, $c_{\max}=1/\epsilon_\Sigma$ is the clamped upper bound induced by the implementation).

The continuous-time QSD rate proxy used by the framework is

$$
\kappa_{\mathrm{QSD}} = \texttt{kappa\_QSD}(\kappa_{\mathrm{total}},\tau)\approx \kappa_{\mathrm{total}}\tau.

$$

---

## Part III-B: Mean-Field Limit (Propagation of Chaos)

Let $Z_i^N(k)=(x_i(k),v_i(k))$ and define the empirical measure

$$
\mu_k^N := \frac{1}{N}\sum_{i=1}^N \delta_{Z_i^N(k)}.

$$
Because companion selection and patched standardization depend on swarm-level statistics, the $N$-particle chain is an interacting particle system of McKean–Vlasov/Feynman–Kac type.

At fixed $\Delta t$, the mean-field limit is most naturally expressed as a nonlinear one-step map on measures obtained by composing:
1. the **pairwise selection/resampling operator** induced by softmax companion choice + Bernoulli cloning (Appendix A, Equation defining $\mathcal{S}$), and
2. the **mutation/killing operator** (BAOAB with boundary killing at $\partial B$).

In weak-selection continuous-time scalings (cloning probabilities $=O(\Delta t)$), this nonlinear map linearizes into a mutation–selection/replicator-type evolution with an *effective* selection functional induced by the pairwise rule; the proof object controls this only through explicit bounded ranges and minorization constants (rather than asserting $\tilde V\equiv V_{\mathrm{fit}}$ as an identity).

When a Wasserstein contraction rate $\kappa_W>0$ is certified, the framework uses the generic propagation-of-chaos proxy

$$
\mathrm{Err}_{\mathrm{MF}}(N,T)\ \lesssim\ \frac{e^{-\kappa_W T}}{\sqrt{N}}
\qquad (\texttt{mean\_field\_error\_bound}(N,\kappa_W,T)).

$$

### How Fitness and Cloning Enter

Fitness and cloning affect the mean-field/QSD regime through:
1. **Minorization / locality:** $\epsilon$ and $D_{\mathrm{alg}}$ determine $m_\epsilon$, hence the strength of the companion-selection Doeblin constant.
2. **Selection pressure:** $(\alpha_{\mathrm{fit}},\beta_{\mathrm{fit}},A,\eta,\epsilon_{\mathrm{clone}},p_{\max})$ determine $V_{\min},V_{\max},S_{\max}$ and therefore the range of clone probabilities; this controls $\lambda_{\mathrm{alg}}^{\mathrm{eff}}$ and $\kappa_x$.
3. **Noise regularization:** $\sigma_x$ injects positional noise at cloning; this prevents genealogical collapse and enters the KL/LSI constants via $\delta_x^2=\sigma_x^2$.

---

## Part III-C: Quasi-Stationary Distribution (QSD) Characterization

### Killed Kernel and QSD Definition (Discrete Time)

Let $Q$ be the **sub-Markov** one-step kernel of the single-walker mutation dynamics on $E:=B\times\overline{B_{V_{\mathrm{alg}}}}$ with cemetery $^\dagger$, where exiting $B$ is killing (sent to $^\dagger$). A QSD is a probability measure $\nu$ and a scalar $\alpha\in(0,1)$ such that

$$
\nu Q = \alpha\,\nu.

$$

### Fleming–Viot / Feynman–Kac Interpretation

For pure boundary killing, the “kill + resample from survivors” mechanism is the classical Fleming–Viot particle system and provides an empirical approximation of the conditioned law/QSD of $Q$.

The implemented Fragile Gas additionally performs bounded fitness-based resampling among alive walkers (pairwise cloning). This is still a Del Moral interacting particle system (Appendix A): the mean-field evolution is a normalized nonlinear semigroup whose fixed points play the role of QSD/eigenmeasure objects for the killed/selection-corrected dynamics.

In the idealized special case where selection is a classical Feynman–Kac weighting by a potential $G$ (Appendix A.2), the continuous-time analogue characterizes the stationary object as the principal eigenmeasure of the twisted generator:

$$
(\mathcal{L}+G)^* \nu \;=\; \lambda_0 \nu,

$$
with $\nu$ normalized to be a probability measure.

---

## Part III-D: Fitness/Cloning Sensitivity (What Moves the Rates)

The constants make the dependence explicit:

1. **Exponents $\alpha_{\mathrm{fit}},\beta_{\mathrm{fit}}$:** increasing $\alpha+\beta$ increases $V_{\max}/V_{\min}=\left(\frac{A+\eta}{\eta}\right)^{\alpha+\beta}$, pushing clone probabilities toward the clip ($0$ or $1$). This typically increases selection pressure but increases genealogical concentration, making $\sigma_x$ more important.
2. **Floors $\eta,\epsilon_{\mathrm{clone}}$:** increasing either reduces $S_{\max}$, reducing selection pressure.
3. **Softmax range $\epsilon$:** larger $\epsilon$ increases $m_\epsilon$ (stronger minorization, better mixing) but makes pairing less local.
4. **Cloning jitter $\sigma_x$:** larger $\sigma_x$ increases regularization (better KL/LSI constants) but increases equilibrium variance; too small $\sigma_x$ risks collapse.
5. **Diffusion regularization $\epsilon_\Sigma$:** larger $\epsilon_\Sigma$ improves ellipticity (reduces $c_{\max}/c_{\min}$) and improves KL rates, at the cost of injecting larger kinetic noise.
6. **Noise threshold regime:** the KL/LSI metatheorem encodes an explicit noise floor $\delta>\delta_*$ (see `src/fragile/convergence_bounds.py:delta_star`). In this instantiation the cloning noise scale is $\delta=\sigma_x$, so too-small $\sigma_x$ can move the system out of the provable LSI/KL regime.

---

## Part III-E: Obligation Ledger

No obligations were introduced in this run.

**Ledger Status:** EMPTY (no $K^{\mathrm{inc}}$ emitted).

---

## Part IV: Final Certificate Chain

### 4.1 Validity Checklist

- [x] All 12 core nodes executed
- [x] Boundary nodes executed (Nodes 13–16)
- [x] Lock executed (Node 17)
- [x] Upgrade pass completed (vacuous)
- [x] Obligation ledger is EMPTY
- [x] No unresolved $K^{\mathrm{inc}}$

**Validity Status:** SIEVE CLOSED (0 inc certificates)

### 4.2 Certificate Accumulation Trace

```
Node 1:  K_{D_E}^+
Node 2:  K_{Rec_N}^+
Node 3:  K_{C_mu}^+
Node 4:  K_{SC_lambda}^- -> K_{TypeII}^{blk}
Node 5:  K_{SC_∂c}^+
Node 6:  K_{Cap_H}^+
Node 7:  K_{LS_sigma}^+
Node 8:  K_{TB_pi}^+
Node 9:  K_{TB_O}^+
Node 10: K_{TB_rho}^+
Node 11: K_{Rep_K}^+
Node 12: K_{GC_nabla}^+ -> K_{Freq}^{blk}
Node 13: K_{Bound_∂}^+
Node 14: K_{Bound_B}^- -> K_{Bode}^{blk}
Node 15: K_{Bound_Σ}^{blk}
Node 16: K_{GC_T}^+
---
Node 17: K_{Cat_Hom}^{blk}
```

---

## Appendix A: Derivation of the Quasi-Stationary Distribution (Feynman–Kac / Fleming–Viot)

This appendix records the standard QSD / interacting-particle interpretation used by the framework: (i) boundary killing defines a sub-Markov kernel with a quasi-stationary distribution (QSD), and (ii) constant-$N$ cloning is a Fleming–Viot/Del-Moral-style particle approximation of the corresponding normalized (conditioned) flow. The implemented fitness-based cloning is *Feynman–Kac-like* (a resampling/selection mechanism driven by bounded fitness values) but is not literally “multiply by $V_{\mathrm{fit}}$”; we therefore state the exact pairwise selection operator and then relate it to the classical normalized Feynman–Kac form as an idealized special case.

### A.0 Quantitative Inputs (From This Proof Object)

The abstract results in the QSD/Feynman–Kac literature reduce, in this instantiation, to the following explicit inputs:

- **Bounded selection potential:** $0\le V_{\mathrm{fit}}\le V_{\max}$, with $V_{\max}=(A+\eta)^{\alpha_{\mathrm{fit}}+\beta_{\mathrm{fit}}}$ computed exactly above.
- **Companion minorization:** on the alive slice, softmax pairing has an explicit Doeblin constant $m_\epsilon=\exp(-D_{\mathrm{alg}}^2/(2\epsilon^2))$.
- **Non-degenerate mutation noise:** anisotropic diffusion is uniformly elliptic on the alive slice once $H_{\max}$ is certified, with window $(c_{\min},c_{\max})$.

### A.1 Exact Discrete-Time Selection Operator (Implemented Pairwise Cloning)

Let $E:=B\times \overline{B_{V_{\mathrm{alg}}}}$ and write $z=(x,v)\in E$. At mean-field level, companion selection induces a (nonlinear) Markov kernel $P_{\epsilon,\mu}(z,dz')$ obtained by softmax-tilting the current law $\mu$ by the algorithmic distance:

$$
P_{\epsilon,\mu}(z,dz')
\propto
\exp\!\left(-\frac{d_{\mathrm{alg}}(z,z')^2}{2\epsilon^2}\right)\,\mu(dz'),

$$
with the finite-$N$ implementation additionally excluding self-pairing for alive walkers.

Given a pair $(z,z')$, define the cloning probability from the implemented score+clip rule (`src/fragile/core/cloning.py`):

$$
p(z,z')
=
\pi\!\left(
\frac{V_{\mathrm{fit}}(z')-V_{\mathrm{fit}}(z)}{(V_{\mathrm{fit}}(z)+\epsilon_{\mathrm{clone}})\,p_{\max}}
\right),
\qquad
\pi(s)=\min(1,\max(0,s)).

$$
Let $K_{\mathrm{clone}}(z,z',d\tilde z)$ be the clone-update kernel implementing:
1. **Position jump with jitter:** $\tilde x = x' + \sigma_x \xi$ with $\xi\sim\mathcal{N}(0,I_d)$ (Definition `clone_position`),
2. **Momentum-conserving velocity update:** the inelastic collision rule `inelastic_collision_velocity` (which is a bounded measurable map on the finite swarm; in mean-field it induces a well-defined marginal Markov kernel on $v$).

Then the selection-stage mean-field operator $\mathcal{S}$ is the nonlinear map on measures defined by, for any bounded measurable $f$,

$$
(\mathcal{S}\mu)(f)
=
\int_E \mu(dz)\int_E P_{\epsilon,\mu}(z,dz')
\Bigl[
(1-p(z,z'))\,f(z)
\;+\;
p(z,z')\int_E f(\tilde z)\,K_{\mathrm{clone}}(z,z',d\tilde z)
\Bigr].

$$
This operator is fitness-aligned at the selection stage: Lemma {prf:ref}`lem-fragile-gas-selection-alignment` shows it increases mean fitness (equivalently decreases $\Phi$) in conditional expectation, holding the companion choices fixed.

### A.2 Relation to Classical Normalized Feynman–Kac (Idealized Special Case)

Classical Feynman–Kac particle filters use *multiplicative* weighting by a bounded potential $G\ge 0$ followed by normalization:

$$
\gamma_{k+1}(f)=\gamma_k(M(Gf)),\qquad \mu_k=\frac{\gamma_k}{\gamma_k(1)},

$$
so

$$
\mu_{k+1}(f)=\frac{\mu_k(M(Gf))}{\mu_k(M(G))}.

$$
The implemented pairwise cloning rule is not literally of this form; however it plays the same structural role (a bounded selection/resampling correction applied to a mutation/killing kernel), and the framework uses QSD/Feynman–Kac theory as the reference envelope in which the thin-interface constants (boundedness, minorization, ellipticity, effective selection pressure) control mean-field and long-time behavior.

### A.3 QSD Definition (Killed Kernel)

Let $(X_k)_{k\ge 0}$ be a Markov chain on $E$ with cemetery state $^\dagger$ and killing time $\tau_\dagger:=\inf\{k\ge 0: X_k=^\dagger\}$. The associated **sub-Markov kernel** $Q$ on $E$ is

$$
Q(x,A):=\mathbb{P}_x(X_1\in A,\ X_1\neq {}^\dagger),\qquad A\subseteq E.

$$
A quasi-stationary distribution is $(\nu,\alpha)$ with $\nu$ a probability measure and $\alpha\in(0,1)$ such that

$$
\nu Q = \alpha \nu.

$$
Equivalently, if $X_0\sim \nu$ then $\mathcal{L}(X_k\mid k<\tau_\dagger)=\nu$ for all $k\ge 0$.

### A.4 Fleming–Viot Particle Approximation (Constant Population)

For pure boundary killing (walkers exiting $B$), the “kill + instantaneous resample from survivors” mechanism is the classical constant-$N$ Fleming–Viot particle system, which approximates the conditioned law and converges to the QSD of $Q$ under standard drift/minorization hypotheses.

The implemented Fragile Gas additionally performs bounded fitness-based resampling even among alive walkers (pairwise cloning). This places it in the broader class of Del Moral interacting particle systems approximating a normalized nonlinear semigroup. Under standard hypotheses ensuring tightness and mixing (the kind of inputs tracked by $D_E$, $C_\mu$, and the Doeblin witness in Node 10), propagation of chaos holds and the empirical measure $\mu_k^N$ tracks the mean-field flow; stationary points correspond to QSD/eigenmeasure objects for the associated killed/selection-corrected dynamics.

### A.5 References

1. Del Moral, P. (2004). *Feynman–Kac Formulae: Genealogical and Interacting Particle Systems with Applications*. Springer.
2. Villemonais, D. (2014). General lower bounds for discrete-time QSD existence. *ESAIM: Probability and Statistics*.
3. Asselah, A. et al. (2011). Quasi-stationary distributions for Fleming–Viot particle systems.
