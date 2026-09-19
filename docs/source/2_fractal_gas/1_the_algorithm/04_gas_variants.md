# Variants of the Fractal Gas

(sec-variants-tldr)=
## 0. TLDR

:::{div} feynman-prose
All the algorithms of this volume are the same step operator with different parts plugged in. Fixing those parts — state space, companion laws, distance, standardizer, fitness map, clone decision, clone transform, kinetic operator, boundary, reward and geometry stage — gives a **gas variant**, and a variant is exactly its twelve components.

This chapter fixes the vocabulary and writes down the tuples. The **Fractal Gas** is the family of all such variants. The **Fragile Gas** is the abstract Markov chain and its axioms. **Algorithmic Gas** is the Rust engine that executes a variant, not a variant itself. The named variants are the Euclidean, Viscous Euclidean, Einstein–Hilbert, Geometric, Latent Fractal and Environment gases. Mean-field and continuum limits are limits of a named variant.

The chapter also says, variant by variant, which theorems of this volume actually apply and which hypotheses are still owed. Only the Euclidean Gas carries the convergence program's conclusions; for the others what is established is stated, and what is not established is stated just as plainly.
:::

(sec-variants-intro)=
## 1. Introduction

:::{div} feynman-prose
Several names in this subject sit very close together. A reader meets "Fractal Gas" as the subject of the whole volume, "Fragile Gas" as an abstract chain with axioms, "Algorithmic Gas" as a piece of Rust, and then a list of gases with adjectives in front of them. Each name is clear on its own page. But a name alone cannot answer the only question that matters when a theorem is quoted: *does this result apply to the thing I am running?*

The answer is a matter of components, not of names. Write the step operator as a composition of stages and a variant is the list of choices made at those stages. Two variants are equal when all twelve entries are equal, and a proof written for one transfers to the other only after every changed entry has been checked against that proof's hypotheses. This is a deliberately strict convention, and it is what makes the rest of the chapter short: the Viscous Euclidean Gas differs from the Euclidean Gas in one entry, the viscous coupling inside the kinetic operator, and that single difference is enough to invalidate every estimate whose proof assumed that the kinetic stage acts on one row at a time.

Five levels keep the vocabulary straight. A **family** collects all variants. A **framework** supplies an abstract chain and the axioms an instance must satisfy before the framework's estimates may be used. An **engine** executes instances. A **variant** is a component tuple. A **limit** — mean-field in $N$, continuum in $h$ — is always taken from a named variant and inherits its hypotheses.
:::

```{mermaid}
flowchart TB
    F["Family: Fractal Gas<br/>all gas variants"] --> FW["Framework: Fragile Gas<br/>abstract chain and axioms"]
    FW --> E["Engine: Algorithmic Gas<br/>executes a GasConfig instance"]
    E --> V["Variants: component tuples"]
    V --> V1["Euclidean Gas"]
    V --> V2["Viscous Euclidean Gas"]
    V --> V3["Einstein-Hilbert Gas"]
    V --> V4["Geometric Gas"]
    V --> V5["Latent Fractal Gas"]
    V --> V6["Environment Gas"]
    V1 --> L["Limits of a named variant"]
    V4 --> L
    L --> L1["Mean-field limit, N to infinity at fixed h"]
    L --> L2["Continuum limit, scaling in h"]
```

:::{div} feynman-prose
The second half of each variant section is an audit. For the Euclidean Gas it lists the transition, operator, finite-$N$ quasi-stationary and mean-field results proved in the convergence program, together with the conditions each one carries. For the Viscous Euclidean, Einstein–Hilbert, Latent Fractal and Environment gases it says that no convergence, quasi-stationary or mean-field theorem is established here, and enumerates what a future proof would have to supply — a confining mechanism, time homogeneity at a fixed gate phase, regularity of a tessellation that jumps across degenerate configurations, a minorization for a hypoelliptic map. For the Geometric Gas the results exist but are conditional on named axioms that are not verified for a general instance.

The closing sections compare the tuples side by side and settle a question that only becomes askable once variants are separated: which field-theoretic observables of {doc}`../2_fractal_set/04_standard_model` a given variant can even define. A colour state divides by the viscous force, so it is undefined when that force is zero; a mutual pairing makes exchange-odd frame means vanish identically; a Euclidean-time axis exists only where a component distinguishes a coordinate. Those are consequences of the component tuple, computed once, here.
:::

(sec-variants-taxonomy)=
## 2. Taxonomy: family, framework, engine, variant, limit

Every algorithm of this volume is obtained by fixing the components of one step operator. This section fixes the vocabulary used for those choices. The notation follows {doc}`../convergence_program/02_euclidean_gas`: $N\ge1$ is the number of walker slots, $a_i\in\{0,1\}$ is the alive mark of slot $i$, $\mathcal A=\{i:a_i=1\}$ is the alive set, $M=|\mathcal A|$, $h>0$ is the time step, and $n\in\{1,2,\ldots\}$ is the index of the step being computed.

:::{div} feynman-prose
Here is the picture I want in your head before you read the next definition. Imagine a machine with twelve dials on the front. One dial says what a walker *is* — just a position, or a position and a velocity, or an opaque simulator snapshot. Another says how a walker picks the neighbor it measures itself against. Another says how far apart two walkers count as being. Another says how hard you push the walkers around between decisions. Set all twelve dials and you have not described a family of algorithms, you have described exactly one algorithm. That setting is what we are going to call a variant.

Why bother being this pedantic? Because of the question that keeps coming up and keeps getting answered badly: *does this theorem apply to the thing I am running?* If a variant is a name, the answer is a matter of taste and nobody wins the argument. If a variant is a twelve-entry tuple, the answer is mechanical. You put the two tuples side by side, you find the entries that differ, and for each difference you go and read the proof to see whether it used that entry. If it did, the theorem does not transfer until somebody redoes the work.

That sounds harsh, and it is meant to. You will see in {ref}`sec-variants-viscous-euclidean` that changing *one* entry — switching on a coupling that is zero in the Euclidean Gas — costs us every long-time theorem in the convergence program. Not because the results are believed to be false. Because their proofs assumed the kinetic stage handles one walker at a time, and with the coupling on, it doesn't.
:::

:::{prf:definition} Gas variant
:label: def-gas-variant

A **gas variant** is a tuple

$$
\mathcal V=\bigl(\mathcal W,\ \mathsf C^{D},\ \mathsf C^{C},\ d_{\mathrm{alg}},\ \mathsf Z,\ \mathsf g,\ \mathsf A,\ \mathsf T,\ \mathsf K,\ \mathsf B,\ \mathsf R,\ \mathsf G\bigr)
$$

with the following components.

| # | Symbol | Component | Content |
|---|---|---|---|
| 1 | $\mathcal W$ | State space and fields | A measurable single-walker state space with its named fields (for example position $x$ and velocity $v$). The marked swarm space is $\Sigma_N=(\mathcal W\times\{0,1\})^N$ |
| 2 | $\mathsf C^{D}$ | Distance-companion law | A probability kernel from $\Sigma_N$ to companion maps $c^{D}:\mathcal A\to\mathcal A$; $c^{D}(i)$ supplies the separation measurement of row $i$ |
| 3 | $\mathsf C^{C}$ | Cloning-companion law | A probability kernel from $\Sigma_N$ to companion maps $c^{C}$ assigning one candidate donor in $\mathcal A$ to every row that may clone |
| 4 | $d_{\mathrm{alg}}$ | Algorithmic distance | A measurable map $\mathcal W\times\mathcal W\to[0,\infty)$ together with a floor $\delta_D\ge0$; the separation of row $i$ is $d_i=\sqrt{d_{\mathrm{alg}}(i,c^{D}(i))^2+\delta_D^2}$ |
| 5 | $\mathsf Z$ | Standardizer | For each channel $y\in\{r,d\}$, a location $m_y$ and a scale $s_y>0$ computed from the alive measurements, giving $z_i^{y}=(y_i-m_y)/s_y$ |
| 6 | $\mathsf g$ | Positive map | Maps $g_r,g_d:\mathbb R\to(0,\infty)$ and exponents $\alpha,\beta\ge0$ with $\alpha+\beta>0$; the fitness is $V_{\mathrm{fit},i}=g_r(z_i^{r})^{\alpha}g_d(z_i^{d})^{\beta}$ |
| 7 | $\mathsf A$ | Clone decision | The acceptance probability $p_i(n)\in[0,1]$ of an alive row as a function of $(V_{\mathrm{fit},i},V_{\mathrm{fit},c^{C}(i)})$, with saturation $p_{\max}>0$, regularizer $\varepsilon_{\mathrm{clone}}\ge0$, and cloning period $q\ge1$: the gate is open only when $n\equiv0\pmod q$. It also fixes the decision of dead rows |
| 8 | $\mathsf T$ | Clone transform | The fields copied from donor to recipient, the position jitter amplitude $\sigma_{\mathrm{clone}}\ge0$, and the collision rule: restitution $\alpha_{\mathrm{restitution}}\in[0,1]$ and the law of the orthogonal matrix $R_C$ applied to each collision group $C$ |
| 9 | $\mathsf K$ | Kinetic operator | The integrator and its step $h$; the conservative acceleration $F=-\nabla U$; the thermostat $(\gamma,\sigma_v)$; the viscous coupling $F^{\mathrm{visc}}$ (possibly zero); the curl rotation (possibly absent); the position diffusion $\sigma_x\ge0$; and the velocity cap $V_{\mathrm{alg}}\in(0,\infty]$ |
| 10 | $\mathsf B$ | Boundary and revival | The valid domain $D$, the stages at which marks are reclassified, the revival rule of dead rows, and the extinction convention |
| 11 | $\mathsf R$ | Reward source | The map producing the raw reward $r_i$ of every row and its orientation (maximized or minimized) |
| 12 | $\mathsf G$ | Geometry stage | A map from the swarm to geometric data (neighbor graph, edge weights, metric, volume element, curvature) consumed by $\mathsf R$ or $\mathsf K$, with its evaluation schedule; $\mathsf G=\varnothing$ when no component consumes such data |

A component may be left as a named free parameter. An **instance** of $\mathcal V$ consists of $\mathcal V$, a value of $N$, a value of every free parameter, and the external data (objective, potential, environment). An instance determines a one-step probability kernel on $\Sigma_N$ by composing its components in the stage order of {prf:ref}`def-fg-step-operator`: measurement and fitness, clone decision and clone transform, kinetic operator, boundary classification. When $q>1$ the kernel depends on $n$ through the residue $n\bmod q$ only, so the chain is time-homogeneous on $\Sigma_N\times\mathbb Z/q\mathbb Z$.

Two variants are equal exactly when all twelve components are equal. A statement proved for one variant applies to another variant only after every changed component has been checked against the hypotheses of that statement.
:::

:::{prf:definition} Family, framework, engine, variant, limit
:label: def-gas-taxonomy

1. **Family.** The **Fractal Gas** is the family of all gas variants of {prf:ref}`def-gas-variant`. Its minimal member has an arbitrary metric state space $(\mathcal X,d)$, $d_{\mathrm{alg}}=d$, an arbitrary measurable reward $r:\mathcal X\to\mathbb R$, a position-only kinetic operator, no velocity field, and $\mathsf G=\varnothing$.
2. **Framework.** The **Fragile Gas** is the abstract Markov chain $\mathcal S_{t+1}\sim\Psi_{\mathcal F}(\mathcal S_t,\cdot)$ of {prf:ref}`def-fragile-gas-algorithm`, and a **Fragile Swarm** is the instantiated tuple $\mathcal F$ of {prf:ref}`def-fragile-swarm-instantiation`. The framework consists of that chain together with the axioms of {doc}`../convergence_program/01_fragile_gas_framework`; its estimates apply to an instance only when those axioms have been verified for it.
3. **Engine.** **Algorithmic Gas** is the Rust engine specified in {doc}`../architecture/01_algorithmic_gas` ({prf:ref}`def-algorithmic-gas-execution-contract`). It executes the instance described by a `GasConfig` value together with a reward source, an optional gradient provider, and a domain adapter. The engine is not a variant: it executes any variant whose components it implements, and a named constructor of `GasConfig` fixes a variant up to the constructor's arguments.
4. **Variant.** The named variants of this chapter are the **Euclidean Gas** ({prf:ref}`def-variant-euclidean`), the **Viscous Euclidean Gas** ({prf:ref}`def-variant-viscous-euclidean`), the **Einstein–Hilbert Gas** ({prf:ref}`def-variant-einstein-hilbert`), the **Geometric Gas** ({prf:ref}`def-variant-geometric`), the **Latent Fractal Gas** ({prf:ref}`def-variant-latent`), and the **Environment Gas** ({prf:ref}`def-variant-environment`).
5. **Limit.** A **mean-field limit** ($N\to\infty$ at fixed $h$) or a **continuum limit** (a scaling limit in $h$) is always the limit of a specified variant, written for example "the mean-field limit of the Euclidean Gas". A limit is not a variant, and a result about a limit is a result about the variant from which it is taken.
:::

:::{div} feynman-prose
The item in that list that people trip over is the third one, so let me take it slowly. **Algorithmic Gas is an engine, not a variant.** It is a Rust program. You hand it a `GasConfig` value, a reward source, maybe a gradient, and it runs the step operator. It does not *have* twelve components; it *offers arms* for twelve components — a menu of kernels, integrators, boundary policies, collision rules — and the `GasConfig` you hand it picks one arm from each menu. Picking them all is what produces a variant.

So the sentence "we proved a theorem about the Algorithmic Gas" is a category error, the same kind of error as "we proved a theorem about NumPy." There is nothing there to prove a theorem *about*. What you can prove theorems about is `GasConfig::euclidean(d, h)`, because that constructor pins every arm down, and {prf:ref}`rem-variant-euclidean-rust` is the table that lets you check, line by line, that the thing the compiler runs is the thing the theorem is stated for.

Now you might push back: the engine does constrain things. It refuses viscosity unless the integrator is BAOAB; it stores positions and velocities in a particular layout. True. The engine is a *sublanguage* — it can express some variants and not others, and {prf:ref}`def-variant-geometric` is one it cannot express at all. But "the set of variants this program can execute" is still a set, not a member of it.

The same discipline kills the other confusions. A mean-field limit is not a seventh gas sitting beside the six; it is something you do *to* one of them, and it drags that one's hypotheses along with it.
:::

:::{prf:remark} Names denoting the same objects
:label: rem-variants-terminology

In this volume *Fragile Gas* and *Fragile Swarm* have only the framework meaning of {prf:ref}`def-gas-taxonomy`. The reinforcement-learning instantiation of the family, whose reward is an environment signal, is the Environment Gas. The name *Abstract Fractal Gas* denotes the minimal member of the family described in item 1 of {prf:ref}`def-gas-taxonomy`. The name *Adaptive Gas*, used in the literature for a gas with a fitness-adapted force and noise, denotes the Geometric Gas. The equation of {prf:ref}`def-fractal-set-sde` writes the dynamics of the Geometric Gas with a different normalization of the weights and of the adaptive force; the conventions of the two displayed equations are compared in {prf:ref}`def-variant-geometric`.
:::

:::{div} feynman-added
**A pocket card for the five words.** If you remember nothing else from this section, remember which question each word answers.

| Word | What kind of thing it is | The question it answers | Can a theorem be *about* it? |
|---|---|---|---|
| Fractal Gas | a set of tuples | "What is the subject of this volume?" | Only if the proof uses nothing but the shared skeleton |
| Fragile Gas / Fragile Swarm | an abstract chain plus axioms | "What must I verify before borrowing the framework's estimates?" | Yes — conditionally on its axioms holding for your instance |
| Algorithmic Gas | a program | "What will actually execute on my machine?" | No. It executes variants; it is not one |
| Euclidean Gas, Geometric Gas, … | one tuple each | "Exactly which algorithm is this?" | Yes. This is the level theorems live at |
| mean-field / continuum limit | an operation on a variant | "What happens as $N\to\infty$ or $h\to0$ *for which gas?*" | Yes, but the answer inherits that gas's hypotheses |

The trap is the middle row. An engine feels concrete — you can run it, profile it, watch it print numbers — and a tuple feels abstract, so it is tempting to attach results to the concrete-feeling thing. Resist that. The engine is the *least* specific object in the table.
:::

(sec-variants-euclidean)=
## 3. Euclidean Gas

The Euclidean Gas is the variant analyzed by the convergence program. Its transition is {prf:ref}`alg-euclidean-gas`; the definition below records it as a component tuple.

### 3.1 Component tuple

:::{div} feynman-prose
Every subject needs one worked example that has been pushed all the way through, and in this volume that example is the Euclidean Gas. Walkers are points with velocities in a box. The reward is minus whatever function you are trying to minimize. Between decisions a BAOAB Langevin integrator shakes them around at a fixed temperature. Nothing exotic.

The reason it is the analyzed one is not that it is the most interesting. It is that every entry in its tuple was chosen so that some estimate downstream would survive. Look at the table below with that in mind and it stops reading like a pile of arbitrary constants. The floor $\delta_D=10^{-3}$ on the separation is there so nothing divides by zero when two walkers coincide. The $+0.1$ in the logistic map $g$ keeps the fitness bounded away from zero, so the acceptance ratio has a denominator you can bound. The $\sigma_{\min}=0.1$ in the standardizer stops the scale collapsing when the swarm happens to agree. The position diffusion $\sigma_x=0.1$ is the last thing that happens in the step, which is exactly what the Feller proof needs: a Gaussian smeared on after everything else. The velocity cap $V_{\mathrm{alg}}=2$ makes the moment bounds trivial instead of delicate.

So the constants are not tuning. They are the visible residue of the proofs. When you change one you are not adjusting a knob, you are reopening a lemma — and you should go find out which one.
:::

:::{prf:definition} Euclidean Gas
:label: def-variant-euclidean

The **Euclidean Gas** in dimension $d$ ($1\le d\le256$) with time step $h>0$ is the gas variant with the following components. Numerical values are those fixed by the constructor `GasConfig::euclidean(d, h)` and agree with {prf:ref}`def-eg-canonical-rust`. Lengths, times and rewards are in the normalized units of the objective.

| # | Component | Specification | Values |
|---|---|---|---|
| 1 | $\mathcal W$ | $(x,v)\in\mathbb R^d\times\mathbb R^d$; all coordinates of dead rows are retained | — |
| 2 | $\mathsf C^{D}$ | Independent draws, one per alive row, from the Gaussian law $P_D^N(i,j)\propto\exp[-d_{\mathrm{alg}}(i,j)^2/(2\epsilon_D^2)]$ on $\mathcal A\setminus\{i\}$ ({prf:ref}`def-eg-frozen-measurements`) | $\epsilon_D=2$ [length] |
| 3 | $\mathsf C^{C}$ | Independent draws from the same law with width $\epsilon_C$; a dead row draws from all of $\mathcal A$ with its retained coordinates | $\epsilon_C=2$ [length] |
| 4 | $d_{\mathrm{alg}}$ | $d_{\mathrm{alg}}(i,j)^2=\lVert \psi_x(x_i)-\psi_x(x_j)\rVert ^2+\lambda_v\lVert \psi_v(v_i)-\psi_v(v_j)\rVert ^2$ with the squashing maps of {prf:ref}`lem-squashing-properties-generic` | $R_x=2$ [length], $R_v=2$ [length/time], $\lambda_v=1$ [time$^2$], $\delta_D=10^{-3}$ [length] |
| 5 | $\mathsf Z$ | Alive-population mean and variance, $s_y=\sqrt{\operatorname{Var}_y+\sigma_{\min,y}^2}$ | $\sigma_{\min,r}=\sigma_{\min,d}=0.1$ |
| 6 | $\mathsf g$ | $g_r(z)=g_d(z)=A/(1+e^{-z})+\eta$ | $A=2$, $\eta=0.1$, $\alpha=\beta=1$ [dimensionless] |
| 7 | $\mathsf A$ | $p_i=\min\{1,[V_{\mathrm{fit},c^{C}(i)}-V_{\mathrm{fit},i}]_+/(p_{\max}(V_{\mathrm{fit},i}+\varepsilon_{\mathrm{clone}}))\}$; dead rows accept with probability one ({prf:ref}`def-eg-component-collision`) | $p_{\max}=1$, $\varepsilon_{\mathrm{clone}}=10^{-6}$, $q=1$ |
| 8 | $\mathsf T$ | Position copy plus Gaussian jitter; one Haar matrix $R_C\in O(d)$ per connected component of the accepted graph, $\widetilde v_i=\bar v_C+\alpha_{\mathrm{restitution}}R_C(v_i-\bar v_C)$ | $\sigma_{\mathrm{clone}}=0.1$ [length], $\alpha_{\mathrm{restitution}}=0.5$ |
| 9 | $\mathsf K$ | BAOAB, final position diffusion and smooth radial cap of {prf:ref}`def-eg-baoab-canonical`; $F^{\mathrm{visc}}\equiv0$; no curl rotation | $\gamma=1$ [1/time], $\sigma_v=1$ [length/time$^{3/2}$], $\sigma_x=0.1$ [length/time$^{1/2}$], $V_{\mathrm{alg}}=2$ [length/time] |
| 10 | $\mathsf B$ | Absorbing box $D$, classified once at the end of the step; revival at the cloning stage through $\mathsf C^{C}$ ({prf:ref}`lem-eg-scheduled-revival`); the all-dead state is absorbing | $D=[-2,2]^d$ |
| 11 | $\mathsf R$ | $R(x,v)=R_{\mathrm{pos}}(x)-\lambda_{\mathrm{vel}}\lVert v\rVert ^2$ with $R_{\mathrm{pos}}=-U$ for a minimized objective $U$ supplied by the caller, together with its gradient | caller-supplied; $\lambda_{\mathrm{vel}}=0$ in the preset |
| 12 | $\mathsf G$ | $\varnothing$ | — |

The frozen OU stage has stationary velocity variance $\sigma_v^2/(2\gamma)=1/2$ per coordinate before the cap. The time step $h$ and the dimension $d$ are arguments of the constructor; the quantitative theorems of {doc}`../convergence_program/09_propagation_chaos` name the step for which they are stated.
:::

:::{prf:remark} Euclidean Gas: component and configuration correspondence
:label: rem-variant-euclidean-rust

The constructor is `GasConfig::euclidean` in `algorithmic-gas/crates/algorithmic-gas/src/variants/euclidean.rs`. Fields not listed keep the values of `GasConfig::default()`. The reference instance `RunConfig::euclidean()` takes $N=64$, $d=2$, $h=0.04$ and the quadratic objective $U(x)=\lVert x\rVert^2/2$, and starts the walkers uniformly in $[-1,1]^2$ at rest.

| Component | `GasConfig` field | Value or enum arm |
|---|---|---|
| $\mathcal W$ | population fields `positions`, `velocities`; `precision` | `Precision::F64` |
| $\mathsf C^{D}$ | `distance_donors` | `DonorModule { law: SamplingLaw::Independent, kernel: Kernel::Gaussian { width: 2 }, count: 1, allow_self: false, history_window: 0 }` |
| $\mathsf C^{C}$ | `cloning_donors` | the same `DonorModule` value |
| $d_{\mathrm{alg}}$ | `DonorModule.distance`, `fitness.distance_floor`, `reducer` | `Distance::SquashedPhaseSpace { position_radius: 2, velocity_radius: 2, lambda: 1 }`, `1e-3`, `CompanionReducer::Mean` |
| $\mathsf Z$ | `fitness.reward_standardizer`, `fitness.diversity_standardizer` | `Standardizer::Global { sigma_min: 0.1 }` |
| $\mathsf g$ | `fitness.reward_map`, `fitness.diversity_map`, `fitness.reward_exponent`, `fitness.diversity_exponent`, `fitness.direction` | `PositiveMap::Logistic { amplitude: 2, floor: 0.1 }`, `1`, `1`, `ObjectiveDirection::Minimize` |
| $\mathsf A$ | `clone_decision` | `CloneDecision { epsilon: 1e-6, saturation: 1, revival_from_companion: true, every: 1 }` |
| $\mathsf T$ | `clone_transform` | `CloneTransform { jitter: Some(Noise::default()), jitter_amplitude: 0.1, restitution: Some(0.5), collision_rotation: CollisionRotation::Haar }` |
| $\mathsf K$ | `kinetic.integrator`, `kinetic.noise.geometry`, `kinetic.position_diffusion`, `kinetic.velocity_cap` | `KineticKind::Baoab { dt: h, friction: 1 }`, `NoiseGeometry::Isotropic` with scale `1`, `0.1`, `Some(2)` |
| $F^{\mathrm{visc}}$, curl | `qft` | `QftExecutionConfig::default()`: `viscosity: None`, `graph_viscosity: None`, `curl: None` |
| $\mathsf B$ | `boundary`, `kinetic.boundary_schedule` | `BoundaryPolicy::AbsorbingBox` on `[-2, 2]^d`, `KineticBoundarySchedule::EndOfStep` |
| $\mathsf R$ | `GasBuilder::new(population, reward)` and `GasBuilder::gradient` | caller-supplied |
| $\mathsf G$ | `geometry` | `None` |
:::

### 3.2 Established results and hypotheses

The following results are proved in this volume for the Euclidean Gas of {prf:ref}`def-variant-euclidean`. Each holds under the hypotheses of its own statement, which are not repeated here.

1. **Transition.** {prf:ref}`thm-eg-canonical-kernel` (time-homogeneous Markov kernel, collision-stage momentum conservation, permutation equivariance), {prf:ref}`thm-eg-component-balances` (component momentum, restitution and shared covariance), and {prf:ref}`thm-euclidean-feller` (Feller continuity for $\sigma_x>0$ and Lebesgue-null $\partial D$).
2. **Operator estimates.** The cloning, Wasserstein and kinetic estimates of {doc}`../convergence_program/03_cloning`, {doc}`../convergence_program/04_wasserstein_contraction` and {doc}`../convergence_program/05_kinetic_contraction`, and the drift and killed-block criteria {prf:ref}`thm-foster-lyapunov-main` and {prf:ref}`thm-main-convergence`. These are conditional statements; the landscape condition {prf:ref}`axiom-non-deceptive` is an additional hypothesis wherever it is used, and {prf:ref}`remark-eg-convergence-hypotheses` lists what kernel existence does not supply.
3. **Finite-$N$ quasi-stationary law.** {prf:ref}`thm-chaos-canonical-finite-n-qsd` gives a unique quasi-stationary distribution (QSD) $\nu_N$ and geometric convergence in total variation for the canonical absorbing-box update with $U(x)=|x|^2/2$ and $h\ne2$, with constants that may depend on $N$. {prf:ref}`thm-hypocoercive-canonical-discrete-entropy` gives the corresponding relative-entropy decay.
4. **Mean-field limit of the Euclidean Gas.** {prf:ref}`thm-mean-field-equation` (the fixed-step population map $\mu_{n+1}=\mathcal F_h(\mu_n)$), {prf:ref}`thm-mean-field-stationary-existence` (existence of a stationary marked law for the absorbing-box map), and {prf:ref}`thm-chaos-finite-time-consistency` (finite-horizon propagation of chaos).

Uniqueness or attraction of a stationary point of $\mathcal F_h$, and the identification of limits of $\nu_N$ as $N\to\infty$ with such a point, are subject to the obligations stated in {prf:ref}`remark-cemetery-state` and in {doc}`../convergence_program/09_propagation_chaos`. Every result above is stated for $F^{\mathrm{visc}}\equiv0$ and $\mathsf G=\varnothing$.

:::{div} feynman-prose
Read that list carefully, because it is shorter than it looks. There is a real ladder here: the update is a Markov kernel, the kernel is Feller, the cloning and kinetic stages contract in the right metrics, the killed chain has a quasi-stationary law it converges to geometrically, and the population map has a stationary point. Four rungs, each resting on the one below.

But notice two things that are *not* on the ladder. First, most of the operator estimates are conditional — the landscape condition {prf:ref}`axiom-non-deceptive` is a hypothesis you supply about your objective, not a fact we proved about the algorithm. Point the gas at an adversarial landscape and the conclusion is simply unavailable; the theorem hasn't failed, you just never met its hypothesis. Second, the constants in the finite-$N$ quasi-stationary result may depend on $N$, and the problem there is the quadratic potential $U(x)=|x|^2/2$. That is a long way from "we proved the optimizer works."

And the last sentence of the section is the one to tattoo somewhere: every one of these results assumes $F^{\mathrm{visc}}\equiv0$ and $\mathsf G=\varnothing$. Zero viscosity, no geometry stage. The next four sections are what happens when you give up one of those, and the honest accounting is that you give up the whole ladder with it.
:::

(sec-variants-viscous-euclidean)=
## 4. Viscous Euclidean Gas

The Viscous Euclidean Gas changes one component of the Euclidean Gas: the viscous coupling inside $\mathsf K$. It is the Euclidean variant on which the colour state of {prf:ref}`thm-sm-su3-emergence` is defined.

### 4.1 Component tuple

:::{div} feynman-prose
Now we turn one dial. In the Euclidean Gas, each walker is kicked by the gradient of the objective and by nothing else; it does not know, at the kinetic stage, that the other walkers exist. Here we add a term that says: if a nearby walker is moving differently from you, you get dragged toward its velocity. The strength of the drag falls off with distance as a Gaussian of width $\rho$, and the overall size is $\nu$. That is the whole change.

The physical picture is honest but limited. It really is the discrete version of what viscosity does in a fluid: neighboring parcels that shear against each other exchange momentum until they move together. So the gas acquires something it did not have before — a local notion of *which way the crowd around me is going*, and a force that points from my velocity toward that crowd's. That force is the object {ref}`sec-variants-measurability` needs, and it is the only reason this variant exists.

Where the fluid analogy breaks: there is no continuum here, no conserved mass field, no pressure, and — this matters — with the row-normalized version there is not even momentum conservation. Each walker divides by the total weight of *its own* neighbors, so walker $i$ pulls on $j$ with a different strength than $j$ pulls on $i$, and the pair no longer cancels. The eligible-count version does conserve momentum, because there the coefficient depends on nothing but $N$. You must declare which one you are running; {prf:ref}`prop-variant-viscous-kernel` will tell you what you bought.
:::

:::{prf:definition} Viscous Euclidean Gas
:label: def-variant-viscous-euclidean

Let $\nu\ge0$ [1/time] be a coupling strength and $\rho>0$ [length] a bandwidth, and let $K_\rho(x,y)=\exp[-\|x-y\|^2/(2\rho^2)]$ be the Gaussian kernel of {prf:ref}`def-fractal-set-viscous-force`, evaluated on physical (unsquashed) positions. For a population $(x,v)$ with eligible set $E$, $|E|=M_E$, define one of the two normalizations

$$
F_i^{\mathrm{visc}}(x,v)=\frac{\nu}{M_E}\sum_{j\in E\setminus\{i\}}K_\rho(x_i,x_j)(v_j-v_i)
\qquad\text{(eligible-count normalization)},
$$

$$
F_i^{\mathrm{visc}}(x,v)=\nu\sum_{j\in E\setminus\{i\}}\omega_{ij}(v_j-v_i),
\quad
\omega_{ij}=\frac{K_\rho(x_i,x_j)}{\sum_{l\in E\setminus\{i\}}K_\rho(x_i,x_l)}
\qquad\text{(row normalization)},
$$

with $F_i^{\mathrm{visc}}=0$ when $i\notin E$ or the normalizer vanishes. The **Viscous Euclidean Gas** with parameters $(d,h,\nu,\rho)$ and a declared normalization is the gas variant whose components 1–8 and 10–12 are those of {prf:ref}`def-variant-euclidean`, and whose kinetic operator $\mathsf K$ is {prf:ref}`def-eg-baoab-canonical` with the two B stages replaced by

$$
v_1=v+\tfrac h2\bigl[F(x)+F^{\mathrm{visc}}(x,v)\bigr],
\qquad
v_3=v_2+\tfrac h2\bigl[F(x_2)+F^{\mathrm{visc}}(x_2,v_2)\bigr].
$$

Here $(x,v)$ is the whole post-collision population at the first kick and $(x_2,v_2)$ is the whole population after the second drift; the coupling is evaluated once per kick and held fixed during it. The A and O stages, the position diffusion, the cap and the terminal classification are unchanged.

In the canonical schedule every row is alive after the cloning stage and no mark changes before the terminal classification, so $E=\{1,\ldots,N\}$ and $M_E=N$ at both kicks. With eligible-count normalization the force is therefore the total pairwise viscous force of {prf:ref}`def-fractal-set-viscous-force` with coupling $\nu/N$; with row normalization it has the weights $\omega_{ij}$ of {prf:ref}`def-latent-fractal-gas-viscous-force`. At $\nu=0$ the variant is the Euclidean Gas.
:::

:::{prf:remark} Viscous Euclidean Gas: component and configuration correspondence
:label: rem-variant-viscous-euclidean-rust

The instance is the configuration `GasConfig::euclidean(d, h)` with one changed field; the named constructor is `GasConfig::viscous_euclidean(d, h, viscosity)` in `algorithmic-gas/crates/algorithmic-gas/src/variants/viscous_euclidean.rs`. The engine requires a BAOAB integrator for this field and rejects it together with `qft.graph_viscosity`.

| Component | `GasConfig` field | Value or enum arm |
|---|---|---|
| $F^{\mathrm{visc}}$ in $\mathsf K$ | `qft.viscosity` | `Some(ViscousForceConfig { coefficient: ν, bandwidth: ρ, row_normalized })`; `row_normalized: false` selects eligible-count normalization |
| all other components | as in {prf:ref}`rem-variant-euclidean-rust` | unchanged |

The constructor takes $\nu$, $\rho$ and the normalization as its `viscosity` argument; they are free parameters of the variant. The reference instance `RunConfig::viscous_euclidean()` takes $N=200$, $d=3$, $h=0.04$, the quadratic objective $U(x)=\lVert x\rVert^2/2$, and the coupling `reference_viscosity()`: $\nu=0.3$, $\rho=1$, eligible-count normalization.
:::

:::{prf:proposition} Transition and momentum balance of the Viscous Euclidean Gas
:label: prop-variant-viscous-kernel

Let $N\ge1$, $\nu\ge0$, $\rho>0$, and fix either normalization of {prf:ref}`def-variant-viscous-euclidean`.

1. Under the hypotheses of {prf:ref}`thm-eg-canonical-kernel`, the Viscous Euclidean Gas defines a time-homogeneous Markov kernel on the full marked state space, and this kernel is permutation equivariant.
2. Under the hypotheses of {prf:ref}`thm-euclidean-feller`, this kernel is Feller.
3. With eligible-count normalization, $\sum_{i\in E}F_i^{\mathrm{visc}}(x,v)=0$ for every population. With row normalization the sum need not vanish.
:::

:::{prf:proof}
**Step 1. Regularity of the coupling.** With $E=\{1,\ldots,N\}$, each $K_\rho(x_i,x_j)$ is a smooth positive function of the positions. The eligible-count normalizer is the constant $N$. For $N\ge2$ the row normalizer $\sum_{l\ne i}K_\rho(x_i,x_l)$ is smooth and strictly positive; for $N=1$ the force is zero. Hence $(x,v)\mapsto F^{\mathrm{visc}}(x,v)$ is continuous on $(\mathbb R^d\times\mathbb R^d)^N$, and it is equivariant under relabeling of rows because it is a sum over the other rows of a function of the pair.

**Step 2. Kernel and equivariance.** The measurement, decision and collision stages are those of {prf:ref}`alg-euclidean-gas` and are covered by the proof of {prf:ref}`thm-eg-canonical-kernel`. The modified kinetic stage is a composition of the continuous kicks of Step 1, the linear drifts, the Gaussian OU and position innovations, the cap, and the measurable terminal classification. Its composition with the finite draws is a probability kernel. Relabeling the input, the innovations and the components relabels the output, since every stage, including the coupled kicks, is equivariant.

**Step 3. Feller continuity.** The proof of {prf:ref}`thm-euclidean-feller` uses two properties of the kinetic stage: for fixed innovations it is a continuous map of the post-collision population, and the final position innovation $\sigma_x\sqrt h\,\xi_{x,i}$ is an independent Gaussian vector added after all other operations. Both hold here by Step 1 and by {prf:ref}`def-variant-viscous-euclidean`. The remainder of that proof applies without change; it does not use independence of the output rows.

**Step 4. Momentum.** With eligible-count normalization,

$$
\sum_{i\in E}F_i^{\mathrm{visc}}
=\frac{\nu}{M_E}\sum_{\substack{i,j\in E\\ i\ne j}}K_\rho(x_i,x_j)(v_j-v_i),
$$

and the summand changes sign under $i\leftrightarrow j$ because $K_\rho$ is symmetric; the sum is zero. For row normalization take $N=3$, $v_1=v_2=0$ and $v_3=u\ne0$. Then $\sum_iF_i^{\mathrm{visc}}=\nu u(\omega_{13}+\omega_{23}-1)$, which tends to $-\nu u$ as $x_3$ moves away from $x_1$ and $x_2$. $\square$
:::

:::{div} feynman-prose
Items 1 and 2 are the cheap part, and I want you to see *why* they were cheap. Being a Markov kernel and being Feller are one-step properties: you compose continuous maps with Gaussian noise and you are done. The viscous kick is smooth in the positions and linear in the velocities, so it slots into the existing proof of {prf:ref}`thm-euclidean-feller` without a fight. Nothing about that proof cared whether the rows were coupled.

Item 3 is where something real happens, and the counterexample in Step 4 is worth sitting with. Take three walkers, two of them at rest, one of them moving with velocity $u$, and drag the moving one far away. With row normalization every walker's weights are forced to add up to one — including the lonely one's. So the runaway feels the full drag $-\nu u$ toward a pair of walkers it can barely see, while those two feel almost nothing back, because *their* weight on the runaway has gone to zero. Momentum leaks out of the system. Newton's third law has been broken, not by an approximation error, but on purpose, by the act of dividing each row by its own normalizer.

Is that bad? It depends entirely on what you want. As an optimizer, row normalization is well behaved — the drag doesn't collapse when a walker is isolated. As physics, it is a momentum sink, and any argument that tried to use momentum conservation as a conserved quantity is dead. Pick deliberately, and write down which you picked.
:::

### 4.2 Established results and hypotheses

For $\nu>0$ the only results established in this volume for the Viscous Euclidean Gas are {prf:ref}`prop-variant-viscous-kernel` and the collision identities of {prf:ref}`thm-eg-component-balances`, which concern an unchanged stage. **No convergence theorem, no existence or uniqueness theorem for a quasi-stationary distribution, and no mean-field limit theorem is established for $\nu>0$.** The form that the population force takes is recorded in {prf:ref}`remark-separation-kinetic-death`, which also states that a theorem for the zero-viscosity configuration is not a theorem for a viscous one.

The results of {ref}`sec-variants-euclidean` were proved for a kinetic stage that acts on each row separately given the post-collision population. With $\nu>0$ the two kicks couple all rows. To assert any of those results for this variant, the following must be verified for the coupled kinetic map:

1. the kinetic moment, Lipschitz and boundary estimates of {doc}`../convergence_program/05_kinetic_contraction` with the additional velocity-dependent force, including the bound $\|F_i^{\mathrm{visc}}\|\le2\nu\max_j\|v_j\|$ and its dependence on the uncapped intermediate velocities;
2. the compactness, minorization and two-sided block bounds used in {prf:ref}`thm-chaos-canonical-finite-n-qsd` and {prf:ref}`thm-hypocoercive-canonical-discrete-entropy`;
3. one-step consistency and conditional concentration of the population map of {doc}`../convergence_program/08_mean_field` and {doc}`../convergence_program/09_propagation_chaos` with the law-dependent force of {prf:ref}`remark-separation-kinetic-death`;
4. for row normalization, the degree comparison recorded in {prf:ref}`axiom-gg-viscous-kernel`, since the alignment dissipation of {prf:ref}`lem-gg-viscous-dissipative` is degree-weighted.

:::{div} feynman-prose
I expect this to feel like an overreaction. One parameter went from $0$ to something small, the trajectories look almost the same on a plot, and we have just thrown away every long-time theorem in the book. Surely by continuity the results survive for small $\nu$?

They might. But "might" is not a theorem, and here is the concrete thing that breaks. Nearly all of the Euclidean machinery is built on a factorization: condition on the population after cloning, and then the $N$ rows evolve *independently*. That is what lets you bound one row at a time, couple two copies of the chain row by row, and build a minorization out of a single Gaussian. Switch on $\nu$ and the kick at row $i$ reads every velocity in the swarm. The conditional law no longer factorizes, and every one of those arguments needs a new proof, not a continuity remark.

There is a second, sharper problem hiding in item 1 of the list. The velocity cap is applied at the *end* of the kinetic stage, but the viscous force is evaluated on the intermediate velocities, before the cap. The bound $\|F_i^{\mathrm{visc}}\|\le2\nu\max_j\|v_j\|$ is therefore in terms of uncapped quantities, and a force that grows with the largest velocity in the swarm is exactly the shape that can defeat a Lyapunov argument. Small $\nu$ helps, but you have to do the estimate to find out how small, and nobody has.
:::

(sec-variants-einstein-hilbert)=
## 5. Einstein–Hilbert Gas

The Einstein–Hilbert Gas is a free gas: no potential force acts on the walkers. Its reward is each walker's share of the Einstein–Hilbert action of the geometry estimated from the walker positions ({prf:ref}`def-tessellation-rust-representation`). The reward enters the dynamics through fitness and cloning only.

### 5.1 Component tuple

:::{div} feynman-prose
This one is strange, so let me tell you what it is doing before you meet the table.

There is no objective function. $U\equiv0$: nothing pushes the walkers anywhere. What the walkers do instead is *be* a space. At each step you take their positions, throw away the last coordinate, build a Delaunay triangulation of what's left, and read off — from the spread of each walker's neighbors — a local metric $g_i$. From the metric you get a volume element $\sqrt{\det g_i}$, and from how the volume element varies across the graph you get a scalar curvature $R_i$. A walker's reward is $R_i\sqrt{\det g_i}$: its personal share of the Einstein–Hilbert action of the cloud it is part of.

So the gas is trying to make a geometry whose curvature-times-volume is large, by the only means it has — killing off walkers in bad neighborhoods and copying walkers in good ones. The dropped coordinate is left out of the triangulation on purpose; it is the Euclidean-time axis, and correlations along it are what {ref}`sec-variants-measurability` will let you ask about.

Now, I have to be blunt about what this is not. Nothing here solves the Einstein field equations. Nothing derives them. $R_i$ and $\sqrt{\det g_i}$ are estimators computed from a point cloud by a specified recipe, and {prf:ref}`def-tessellation-rust-representation` is that recipe. Whether they converge to anything as $N\to\infty$ is not addressed in this volume. Take the names as labels for formulas, and the enterprise stays honest.
:::

:::{prf:definition} Einstein–Hilbert Gas
:label: def-variant-einstein-hilbert

Let $T>0$ [length$^2$/time$^2$] be a temperature and $h>0$ [time] a time step. Let $d\ge1$ be the position dimension and let $d'$ be the tessellated dimension: $d'=d-1$ if $d\ge3$ and $d'=d$ otherwise, with $1\le d'\le3$. Write $x_i=(\bar x_i,t_i)$ with $\bar x_i\in\mathbb R^{d'}$ when $d\ge3$; the last coordinate $t_i$ is the **Euclidean-time coordinate**. The **Einstein–Hilbert Gas** is the gas variant with the following components. Numerical values are those fixed by the constructor `GasConfig::einstein_hilbert(T, h)`.

| # | Component | Specification | Values |
|---|---|---|---|
| 1 | $\mathcal W$ | $(x,v)\in\mathbb R^d\times\mathbb R^d$. The per-walker outputs of $\mathsf G$ (volume element, curvature, diffusion factor) are stored as additional fields; under the schedule of component 12 they are functions of the current positions | — |
| 2 | $\mathsf C^{D}$ | A uniformly distributed perfect matching of $\mathcal A$: shuffle $\mathcal A$ by the Fisher–Yates algorithm and pair consecutive entries. If $M$ is odd the remaining walker is its own companion. Thus $c^{D}\circ c^{D}=\mathrm{id}_{\mathcal A}$ | uniform kernel |
| 3 | $\mathsf C^{C}$ | A second perfect matching with the same law, drawn independently of $c^{D}$ given the swarm; $c^{C}\circ c^{C}=\mathrm{id}_{\mathcal A}$ | uniform kernel |
| 4 | $d_{\mathrm{alg}}$ | $d_{\mathrm{alg}}(i,j)=\lVert x_i-x_j\rVert $ on all $d$ position coordinates; no squashing and no velocity term | $\delta_D=10^{-30}$ [length] |
| 5 | $\mathsf Z$ | Sample statistics: $\bar y=M^{-1}\sum_{i\in\mathcal A}y_i$, $s_y^2=\max\{M-1,1\}^{-1}\sum_{i\in\mathcal A}(y_i-\bar y)^2$, $z_i^{y}=(y_i-\bar y)/(s_y+\varepsilon_{\mathrm{std}})$. A constant channel standardizes to zero | $\varepsilon_{\mathrm{std}}=10^{-30}$ |
| 6 | $\mathsf g$ | $g_r(z)=g_d(z)=A/(1+e^{-z})$, no additive floor; the reward is maximized | $A=2$, $\eta=0$, $\alpha=\beta=1$ |
| 7 | $\mathsf A$ | $p_i(n)=\min\{1,[V_{\mathrm{fit},c^{C}(i)}-V_{\mathrm{fit},i}]_+/V_{\mathrm{fit},i}\}$ if $n\equiv0\pmod{20}$, and $p_i(n)=0$ otherwise. Dead rows accept with probability one at every step | $p_{\max}=1$, $\varepsilon_{\mathrm{clone}}=0$, $q=20$ |
| 8 | $\mathsf T$ | An accepted row copies every field of its donor. No jitter. Collision on the connected components of the accepted graph with the identity rotation: $\widetilde v_i=\bar v_C+\alpha_{\mathrm{restitution}}(v_i-\bar v_C)$ | $\sigma_{\mathrm{clone}}=0$, $\alpha_{\mathrm{restitution}}=1$, $R_C=I$ |
| 9 | $\mathsf K$ | BAOAB with $F=-\nabla U\equiv0$. Each B stage is the graph viscous kick with Boris curl rotation of {prf:ref}`alg-einstein-hilbert-gas`. OU thermostat $v\leftarrow cv+\sqrt{T(1-c^2)}\,\xi$, $c=e^{-\gamma h}$. No position diffusion and no velocity cap | $\gamma=1$ [1/time], $\sigma_v=\sqrt{2\gamma T}$, $\nu=3$ [1/time], $\beta_{\mathrm{curl}}=1$, $\sigma_x=0$, $V_{\mathrm{alg}}=\infty$ |
| 10 | $\mathsf B$ | $D=\mathbb R^d$. A row is dead only if one of its fields is non-finite; marks are checked after the clone transform and after every kinetic substage. A dead row is revived at the cloning stage from a donor drawn uniformly from $\mathcal A$, at every step. The engine halts when $\mathcal A=\varnothing$ | unbounded |
| 11 | $\mathsf R$ | $r_i=R_i\sqrt{\det g_i}$, the walker's share of the Einstein–Hilbert action, with $R_i$ the scalar curvature and $\sqrt{\det g_i}$ the volume element produced by $\mathsf G$ | scale $\lambda=1$; curvature field `ricci_scalar` |
| 12 | $\mathsf G$ | Delaunay tessellation of the projected sites $\bar x_i$ (all of $x_i$ if $d<3$); neighbor-covariance metric $g_i$, the ridge-regularized inverse of the covariance of the displacements to the neighbors, with clamped spectrum; volume element $\sqrt{\max\{\det g_i,10^{-12}\}}$; conformal-Laplacian scalar curvature $R_i=-2(d'-1)\sum_{j\sim i}w^{R}_{ij}(u_j-u_i)$ with $u_i=\log\max\{\det g_i,10^{-12}\}/(2d')$; two row-normalized edge-weight families $w^{R}$ and $w^{\mathrm{visc}}$ ({prf:ref}`def-tessellation-rust-representation`). Evaluated whenever positions have changed before a reward evaluation | metric ridge $10^{-5}$, eigenvalue floor $10^{-6}$; kernel length $\ell=1$ [length] |

The edge weights are, before normalization over the neighbors $j\sim i$ of each walker,

$$
w^{R}_{ij}\propto\frac{1}{\sqrt{\max\{d_g(i,j)^2,10^{-8}\}}+10^{-8}},
\qquad
w^{\mathrm{visc}}_{ij}\propto\exp\!\left[-\frac{d_g(i,j)^2}{2\ell^2}\right]\sqrt{\max\{\det g_j,10^{-12}\}},
$$

where $d_g(i,j)^2=\Delta\bar x_{ij}^{\mathsf T}\tfrac12(g_i+g_j)\Delta\bar x_{ij}$ is the metric edge length. The normalized weight is the raw weight divided by $\max\{\sum_{l\sim i}w^{\mathrm{raw}}_{il},10^{-12}\}$. Hence $w_{ij}\ge0$ and $\sum_{j\sim i}w_{ij}\le1$, with equality whenever the raw row sum is at least $10^{-12}$; a walker without neighbors has an empty row. In general $w_{ij}\ne w_{ji}$.

The reference instance `RunConfig::einstein_hilbert()` takes $N=500$, $d=3$ (so $d'=2$), $T=0.33$, $h=0.002$, and starts every walker at the origin at rest.
:::

:::{prf:remark} Einstein–Hilbert Gas: component and configuration correspondence
:label: rem-variant-einstein-hilbert-rust

The constructor is `GasConfig::einstein_hilbert` in `algorithmic-gas/crates/algorithmic-gas/src/variants/einstein_hilbert.rs`. It is built with `GeometryReward::default()` as the reward source and `ZeroPotential` as the gradient provider. Fields not listed keep the values of `GasConfig::default()`.

| Component | `GasConfig` field | Value or enum arm |
|---|---|---|
| $\mathcal W$ | population fields `positions`, `velocities`, `geometry.volume_element`, `geometry.curvature.ricci_scalar`, `geometry.diffusion` | — |
| $\mathsf C^{D}$ | `distance_donors` | `DonorModule { kernel: Kernel::Uniform, law: SamplingLaw::FisherYates, odd: OddPolicy::SelfCompanion, count: 1 }` |
| $\mathsf C^{C}$ | `cloning_donors` | the same `DonorModule` value; the two roles use separate random streams |
| $d_{\mathrm{alg}}$ | `DonorModule.distance`, `fitness.distance_floor`, `reducer` | `Distance::Euclidean { field: "positions", squared: false }`, `1e-30`, `CompanionReducer::Mean` |
| $\mathsf Z$ | `fitness.reward_standardizer`, `fitness.diversity_standardizer` | `Standardizer::LegacySample { epsilon: 1e-30 }` |
| $\mathsf g$ | `fitness.reward_map`, `fitness.diversity_map`, exponents, `fitness.direction` | `PositiveMap::Logistic { amplitude: 2, floor: 0 }`, `1`, `1`, `ObjectiveDirection::Maximize` |
| $\mathsf A$ | `clone_decision` | `CloneDecision { epsilon: 0, saturation: 1, revival_from_companion: false, every: 20 }` |
| $\mathsf T$ | `clone_transform` | `CloneTransform { position_field: None, jitter: None, velocity_field: Some("velocities"), restitution: Some(1), collision_rotation: CollisionRotation::Identity }` |
| $\mathsf K$ | `kinetic.integrator`, `kinetic.noise` | `KineticKind::Baoab { dt: h, friction: 1 }`; `InnovationLaw::Gaussian` with `NoiseGeometry::Isotropic` of scale $\sqrt{2\gamma T}$; `position_diffusion: 0`, `velocity_cap: None` |
| $F^{\mathrm{visc}}$ | `qft.graph_viscosity` | `Some(GraphViscosityConfig { coefficient: 3, weights: "riemannian_kernel_volume" })` |
| curl rotation | `qft.curl` | `Some(CurlRotationConfig { beta_curl: 1 })` |
| $\mathsf B$ | `boundary`, `kinetic.boundary_schedule` | `BoundaryPolicy::Unbounded`, `KineticBoundarySchedule::Substeps` |
| $\mathsf R$ | reward source | `GeometryReward { curvature: "ricci_scalar", allocation: RewardAllocationKind::EinsteinHilbertDensity { scale: 1 } }` |
| $\mathsf G$ | `geometry` | `Some(GeometryStageConfig { schedule: GeometrySchedule::EveryStage, .. })` with `Projection::DropLast { min_ambient: 3 }`, `MetricKind::NeighborCovariance`, `VolumeKind::SqrtDetMetric { det_floor: 1e-12 }`, weights `[InverseRiemannianDistance, RiemannianKernelVolume]`, `CurvatureKind::ConformalLaplacian { weights: "inverse_riemannian_distance", det_floor: 1e-12 }` |
:::

### 5.2 One update step

:::{prf:algorithm} Einstein–Hilbert Gas Update
:label: alg-einstein-hilbert-gas

**Input.** The step index $n\ge1$ and the marked swarm $S=((x_i,v_i,a_i))_{i=1}^N$ with alive set $\mathcal A$, $M=|\mathcal A|\ge1$. Let $\mathcal G(x)$ denote the output of the geometry stage at positions $x$: the Delaunay neighbor graph of the projected sites, the weights $w^{R}$ and $w^{\mathrm{visc}}$, and the per-walker fields $g_i$, $\sqrt{\det g_i}$ and $R_i$. All coordinates used in Steps 1–5 are frozen input coordinates.

1. **Reward.** Set $r_i=R_i(x)\sqrt{\det g_i(x)}$ for every row, with the floors of {prf:ref}`def-variant-einstein-hilbert`.
2. **Distance companions and separation.** Draw a uniformly distributed perfect matching $c^{D}$ of $\mathcal A$; for odd $M$ the unmatched walker has $c^{D}(i)=i$. Set $d_i=\sqrt{\|x_i-x_{c^{D}(i)}\|^2+\delta_D^2}$ with all $d$ coordinates of the positions.
3. **Fitness.** For $y\in\{r,d\}$ compute the sample mean $\bar y$ and the sample standard deviation $s_y$ over $\mathcal A$, set $z_i^{y}=(y_i-\bar y)/(s_y+\varepsilon_{\mathrm{std}})$, and

   $$
   V_{\mathrm{fit},i}=\frac{2}{1+e^{-z_i^{r}}}\cdot\frac{2}{1+e^{-z_i^{d}}},\qquad i\in\mathcal A.
   $$

4. **Cloning companions and decisions.** Draw a second, independent, uniformly distributed perfect matching $c^{C}$ of $\mathcal A$ with the same odd-walker rule, and independent $U_i\sim\operatorname{Unif}[0,1]$. For $i\in\mathcal A$ set

   $$
   p_i=\begin{cases}\min\{1,[V_{\mathrm{fit},c^{C}(i)}-V_{\mathrm{fit},i}]_+/V_{\mathrm{fit},i}\},&n\equiv0\pmod{20},\\0,&\text{otherwise},\end{cases}
   \qquad A_i=\mathbf 1_{\{U_i<p_i\}}.
   $$

   Every dead row draws a donor uniformly from $\mathcal A$ and has $A_i=1$, at every $n$.
5. **Clone transform.** Every row with $A_i=1$ copies all fields of its donor; in particular $\widetilde x_i=x_{c^{C}(i)}$, with no jitter. On each connected component $C$ of the accepted graph apply the elastic identity-rotation collision

   $$
   \widetilde v_i=\bar v_C+(v_i-\bar v_C)=v_i,\qquad \bar v_C=|C|^{-1}\sum_{j\in C}v_j,
   $$

   computed from the frozen input velocities. Rows outside the accepted graph keep $(x_i,v_i)$. Reclassify the marks.
6. **Geometry and reward refresh.** Evaluate $\mathcal G(\widetilde x)$ and the rewards at $\widetilde x$. Denote by $\mathcal G_\star=\mathcal G(\widetilde x)$ the graph and the weights $w=w^{\mathrm{visc}}$ used by both B stages below.
7. **B stage** (duration $h/2$), applied to the whole population $(x,v)$ with neighbors $j\sim i$ in $\mathcal G_\star$, restricted to alive rows:

   $$
   \begin{aligned}
   F_i(v)&=\nu\sum_{j\sim i}w_{ij}(v_j-v_i),\\
   v_i^{(1)}&=v_i+\tfrac h4F_i(v),\\
   \Omega_i&=\tfrac12(J_i-J_i^{\mathsf T}),\qquad J_i(\Xi_i+\varrho_iI)=\Phi_i,\\
   \Phi_i&=\sum_{j\sim i}w_{ij}\,(F_j(v)-F_i(v))(x_j-x_i)^{\mathsf T},\qquad
   \Xi_i=\sum_{j\sim i}w_{ij}\,(x_j-x_i)(x_j-x_i)^{\mathsf T},\\
   v_i^{(2)}&=(I-\Theta_i)^{-1}(I+\Theta_i)\,v_i^{(1)},\qquad \Theta_i=\tfrac{\beta_{\mathrm{curl}}h}{4}\,\Omega_i,\\
   v_i^{(3)}&=v_i^{(2)}+\tfrac h4F_i(v^{(2)}).
   \end{aligned}
   $$

   Here $x_j-x_i$ has all $d$ coordinates, $\varrho_i=\max\{\sqrt{\epsilon_{\mathrm{mach}}}\operatorname{tr}\Xi_i/d,\ \varrho_{\min}\}>0$ is a ridge, with $\epsilon_{\mathrm{mach}}$ the machine epsilon and $\varrho_{\min}$ the smallest positive normal number of the run precision, and $J_i$ is the weighted least-squares Jacobian of the force field at walker $i$. The output velocity is $v^{(3)}$.
8. **A stage.** $x_i\leftarrow x_i+\tfrac h2v_i$.
9. **O stage.** With independent $\xi_i\sim\mathcal N(0,I_d)$ and $c=e^{-\gamma h}$, set $v_i\leftarrow cv_i+\sqrt{T(1-c^2)}\,\xi_i$.
10. **A stage.** Repeat Step 8.
11. **B stage.** Repeat Step 7 at the current positions and velocities, with the graph and weights of $\mathcal G_\star$.
12. **Commit.** Reclassify the marks, evaluate $\mathcal G(x^{+})$ and the rewards at the output positions, and return $S^{+}=((x_i^{+},v_i^{+},a_i^{+}))_{i=1}^N$ with step index $n+1$.

The marks are also reclassified after each of Steps 7–11; a row that becomes dead takes no further substage in that step. The reward $r_i$ enters Steps 3–4 only. No force depends on it.
:::

:::{div} feynman-prose
Twelve steps is a lot to hold at once, so here is the shape of it. Steps 1–5 are selection: measure the geometry, score everybody, pair everybody off, let the losers jump onto the winners. Steps 6–12 are motion: rebuild the geometry at the new positions and run BAOAB — kick, drift, thermostat, drift, kick — with the graph frozen at $\mathcal G_\star$ for the whole sweep. Freezing the graph matters. If you rebuilt the Delaunay complex in the middle of a kick, the force would jump discontinuously whenever a triangle flipped, and you would not have an integrator at all.

The part worth staring at is Step 7. The first line is ordinary viscous drag along graph edges. Then something less ordinary: we fit a matrix $J_i$ that best explains how the drag force changes as you step to each neighbor — a least-squares Jacobian of the force field at walker $i$ — take its antisymmetric part $\Omega_i$, and rotate the velocity by it. That is a curl. The gas measures how much the local force field swirls, and then swirls the walker to match.

Why a Cayley transform, $(I-\Theta)^{-1}(I+\Theta)$, rather than just adding $\Theta v$? Because that expression is *exactly* orthogonal for any skew $\Theta$, at any step size — Item 3 will prove it in two lines. A naive rotation would pump energy in or out at $O(h^2)$, and you would spend the rest of your life arguing with the thermostat about where the heat came from.
:::

:::{prf:proposition} Elementary identities of the Einstein–Hilbert update
:label: prop-variant-eh-identities

For the update of {prf:ref}`alg-einstein-hilbert-gas`:

1. **Collision.** The clone transform leaves every velocity unchanged: $\widetilde v_i=v_i$ for all $i$. An accepted row receives the position of its donor and keeps its own velocity.
2. **Accepted graph.** At most one member of each pair $\{i,c^{C}(i)\}$ of alive walkers is accepted, a self-companion is never accepted, and the accepted components among alive walkers are pairs.
3. **Rotation.** For every skew-symmetric $\Theta$, the matrix $Q=(I-\Theta)^{-1}(I+\Theta)$ exists and is orthogonal. Hence $\|v_i^{(2)}\|=\|v_i^{(1)}\|$ in Step 7.
4. **Thermostat.** The O stage has the unique stationary law $\mathcal N(0,TI_d)$ for each velocity; in particular its stationary velocity variance is $T$ per coordinate.
5. **Translations and the time split.** Call a configuration *Delaunay-generic* when the Delaunay complex of its distinct projected sites is unique; this holds, for example, when the sites affinely span $\mathbb R^{d'}$ and no $d'+2$ of them lie on a common sphere. Assume that every configuration at which $\mathcal G$ is evaluated in the step is Delaunay-generic. Then the one-step kernel is equivariant under a simultaneous translation $x_i\mapsto x_i+b$ of all positions, and, for $d\ge3$, under the simultaneous maps $(x_i,v_i)\mapsto(Ox_i,Ov_i)$ with $O=\operatorname{diag}(O',\pm1)$, $O'\in O(d-1)$. At the remaining configurations the tessellator selects one Delaunay complex, and equivariance holds there exactly when that selection commutes with the map; this is not established.
6. **Survival.** Every map of the update is finite-valued on finite inputs. For finite initial data, $M=N$ at every step.
:::

:::{prf:proof}
**Item 1.** Substitute $\alpha_{\mathrm{restitution}}=1$ and $R_C=I$ in the collision formula of {prf:ref}`def-eg-component-collision`. The field copy overwrites the recipient velocity with the donor velocity, and the collision then writes $\bar v_C+(v_i-\bar v_C)=v_i$ from the frozen inputs to every member of $C$, including the recipient.

**Item 2.** On an open gate the numerators of $p_i$ and $p_{c^{C}(i)}$ are $[V_j-V_i]_+$ and $[V_i-V_j]_+$ with $j=c^{C}(i)$, and the fitness values are strictly positive because $2/(1+e^{-z})>0$. At most one numerator is positive, and both vanish when $j=i$. Accepted edges among alive walkers therefore belong to distinct pairs of the matching. This is the argument of {prf:ref}`cor-sm-physics-paired-cloning`.

**Item 3.** For real skew $\Theta$ and $u\in\mathbb R^d$, $u^{\mathsf T}(I-\Theta)u=\|u\|^2$, so $I-\Theta$ is injective and hence invertible. The matrices $I\pm\Theta$ commute, as do their inverses, and $(I\pm\Theta)^{\mathsf T}=I\mp\Theta$. Therefore

$$
Q^{\mathsf T}Q=(I-\Theta)(I+\Theta)^{-1}(I-\Theta)^{-1}(I+\Theta)=I.
$$

The matrix $\Omega_i$ is skew by construction, so $\Theta_i$ is skew.

**Item 4.** If $v\sim\mathcal N(0,TI_d)$ is independent of $\xi$, then $cv+\sqrt{T(1-c^2)}\xi$ is centered Gaussian with covariance $[c^2T+T(1-c^2)]I_d=TI_d$, so this law is stationary. For an arbitrary initial $v_0$, iterating the stage $m$ times with independent innovations gives $v_m=c^mv_0+\zeta_m$ with $\zeta_m\sim\mathcal N(0,T(1-c^{2m})I_d)$ independent of $v_0$. Since $0<c<1$, $c^mv_0\to0$ almost surely and $\zeta_m\Rightarrow\mathcal N(0,TI_d)$. Every initial law therefore converges weakly to $\mathcal N(0,TI_d)$, which is the unique stationary law.

**Item 5.** Positions enter Steps 1–7 only through differences $x_j-x_i$: the distances, the Delaunay complex of the projected sites, the metric $g_i$, the edge lengths $d_g$, and the matrices $\Phi_i$ and $\Xi_i$. The drifts of Steps 8 and 10 commute with translations. For the orthogonal maps, $O$ preserves the splitting $\mathbb R^{d-1}\oplus\mathbb R$, so the projected sites are rotated by $O'$. The empty-sphere property is preserved by translations and by $O'$, so the image of a Delaunay complex is a Delaunay complex of the image sites; on a Delaunay-generic configuration it is the unique one, which is the only place where the genericity assumption is used. Hence the neighbor graph is invariant, $g_i\mapsto O'g_iO'^{\mathsf T}$, and $d_g$, $\det g_i$, $R_i$, the weights and the pairing law are invariant. The force transforms as $F\mapsto OF$, so $\Phi_i\mapsto O\Phi_iO^{\mathsf T}$, $\Xi_i\mapsto O\Xi_iO^{\mathsf T}$, the ridge $\varrho_i$ is unchanged, and $J_i\mapsto OJ_iO^{\mathsf T}$; hence $\Theta_i\mapsto O\Theta_iO^{\mathsf T}$ and the Cayley rotation is covariant. The Gaussian innovations have an $O(d)$-invariant law.

**Item 6.** The fitness is a product of logistic values in $(0,2)$, the acceptance ratio has a positive denominator, $\Xi_i+\varrho_iI$ is positive definite, and $I-\Theta_i$ is invertible by Item 3. The geometry stage is defined for every finite configuration by the duplicate and rank rules of {prf:ref}`def-tessellation-rust-representation`, and the metric and curvature floors of {prf:ref}`def-variant-einstein-hilbert` keep $g_i$, $\det g_i$ and $u_i$ finite. A finite population is mapped to a finite population for every finite innovation, so no row is classified dead. $\square$
:::

:::{div} feynman-prose
Six small facts, and together they tell you what kind of animal this is.

Items 1 and 2 say the cloning stage is much gentler than it looks. With $\alpha_{\mathrm{restitution}}=1$ and $R_C=I$, the collision formula collapses to $\widetilde v_i=v_i$: a clone teleports to its donor's position and keeps its own velocity, and nobody else is touched. Combine that with Item 2 — walkers are matched into pairs, and within a pair only the loser can be accepted — and cloning here is nothing but resampling in position space, one pair at a time.

Item 6 is the one that reorients everything. $M=N$ at every step, always. There is no boundary, no killing, no cemetery. Every quasi-stationary argument in the convergence program is about a chain that *dies*, and this chain cannot die, so that machinery is not weakened here, it is simply about a different object.

Which leaves Item 5 holding the bad news. Away from the degenerate configurations where the triangulation is ambiguous, the update commutes with translating the whole swarm. There is no potential and no box. So if any law were stationary, translating it would give another stationary law, and nothing in the update pulls the centre of mass back toward any particular place. The question is not "does it converge slowly" — wherever that symmetry holds, it rules out a unique stationary law for the positions. Any honest stationarity statement for this gas has to be made about the translation-reduced process, the shape of the cloud rather than its location, or else somebody has to add a confining mechanism and pay for it with moment bounds.
:::

### 5.3 Established results and hypotheses

**No convergence theorem, no existence or uniqueness theorem for an invariant or quasi-stationary law, and no mean-field or continuum limit theorem is established in this volume for the Einstein–Hilbert Gas.** The established statements are the identities of {prf:ref}`prop-variant-eh-identities`, the pairing arguments of {prf:ref}`cor-sm-physics-paired-cloning` and {prf:ref}`cor-sm-paired-doublet-cancellation`, whose proofs use only a uniform mutual-pair companion law and the pair collision rule, and the cancellation {prf:ref}`prop-exchange-odd-cancellation`. The estimators of $\mathsf G$ are discrete constructions; by {prf:ref}`def-tessellation-rust-representation` none of them imposes a field equation.

The hypotheses of the Euclidean Gas results fail structurally for this variant, so none of those results transfers. To establish a long-time statement the following must be supplied.

1. **A state space on which stationarity is possible.** By Item 5 of {prf:ref}`prop-variant-eh-identities` the update is translation equivariant on Delaunay-generic configurations, $D=\mathbb R^d$, and $U\equiv0$: there is no confining envelope and no Safe Harbor estimate. A stationarity statement must be formulated for a translation-reduced process or must add and analyze a confinement mechanism, together with moment and tail bounds.
2. **Time homogeneity.** With $q=20$ the chain is homogeneous only on $\Sigma_N\times\mathbb Z/20\mathbb Z$. The phase observable of {prf:ref}`rem-sm-actual-step-and-clock` excludes a strict mixing estimate on the phase-augmented space; a convergence statement must use the $20$-step kernel at a fixed phase.
3. **Regularity of the geometry stage.** The Delaunay complex is a discontinuous function of the positions across degenerate configurations (coincident, affinely dependent or cospherical sites). With $\sigma_{\mathrm{clone}}=0$ a clone coincides exactly with its donor, so degenerate configurations occur with positive probability, and with $\sigma_x=0$ no final position innovation makes them null. Measurability of $\mathcal G$ with its duplicate and rank rules, and a Feller or strong Feller property of the composed kernel, have to be proved; the proof of {prf:ref}`thm-euclidean-feller` does not apply.
4. **Accessibility.** Noise enters the velocities only. A minorization must come from a multi-step hypoelliptic argument for a kinetic map whose force is defined on a state-dependent graph and is discontinuous across changes of that graph.
5. **Coupled rows and asymmetric weights.** The B stages couple all rows, and the row-normalized weights are not symmetric, so the viscous kick need not conserve momentum and the degree-weighted dissipation of {prf:ref}`lem-gg-viscous-dissipative` requires the degree comparison of {prf:ref}`axiom-gg-viscous-kernel` for the tessellation weights.
6. **Population limit.** The mutual-pair companion law, the periodic gate and a tessellation-defined force are outside the canonical regime of {doc}`../convergence_program/08_mean_field` and {doc}`../convergence_program/09_propagation_chaos`. A population map for this variant has not been defined.

:::{div} feynman-prose
Six obligations, and I want to single out the third, because it is the one that a reader used to smooth dynamics will walk straight past.

A Delaunay triangulation is not a continuous function of the point positions. Slide four points until they sit on a common circle and the diagonal of the quadrilateral flips; an instant before and an instant after, the neighbor lists differ, and so does every force built on them. Usually one shrugs: degenerate configurations form a measure-zero set, generic point clouds are fine, move on.

You cannot shrug here, and the reason is right there in the tuple. The jitter is $\sigma_{\mathrm{clone}}=0$, so when a walker clones it lands on its donor *exactly* — same floating-point position, zero separation. And $\sigma_x=0$, so no position noise is sprinkled on at the end of the step to smear it off again. Degenerate configurations are not a null set for this chain. They happen with positive probability, by construction, every time the gate opens. The duplicate and rank rules of {prf:ref}`def-tessellation-rust-representation` say what the code does when it meets one, but saying what the code does is not the same as proving the resulting kernel is measurable, let alone Feller.

That is the honest state of affairs. This variant is a well-specified, runnable, reproducible algorithm about which no long-time theorem is claimed here — and item 3 is a decent guess at where the first hard lemma will have to go.
:::

(sec-variants-geometric)=
## 6. Geometric Gas

The Geometric Gas adapts the force and the noise covariance to the measured fitness landscape. Its defining chapter is {doc}`../convergence_program/17_geometric_gas`.

### 6.1 Component tuple

:::{div} feynman-prose
The Geometric Gas asks a greedy question: if the swarm is already measuring a fitness field, why not use its *derivatives* too?

Two things come out of that. First, an extra force $F_i=\epsilon_F\nabla_{x_i}V_i$ — walkers are pushed uphill in fitness directly, instead of only being selected for it. Second, and more interesting, the noise is shaped by the second derivative: $\Sigma_i=(H_i+\epsilon_\Sigma I)^{-1/2}$, with $H_i$ the Hessian of the fitness. Think about what that does. Along a direction where fitness curves sharply, $H$ is large, $\Sigma$ is small, and the walker is barely jiggled. Along a flat direction, $\Sigma$ is large and the walker is flung. The gas explores cheaply where exploration is cheap and steps carefully where the landscape is steep — it is doing something close to preconditioning, with the preconditioner measured on the fly by the swarm itself.

That is also where the trouble comes from. You cannot invert a square root of a matrix that might be indefinite, and the fitness Hessian certainly can be. The ridge $\epsilon_\Sigma$ is what keeps $H_i+\epsilon_\Sigma I$ positive, and the condition $\epsilon_\Sigma-\Lambda_->0$ in {prf:ref}`axiom-gg-ueph` is the precise statement that the ridge beats the most negative eigenvalue the fitness can produce. Every ellipticity, well-posedness and ergodicity result downstream leans on that single inequality. It is an assumption about your landscape, not a property of the algorithm.

One bookkeeping note before the table: this variant is written as an SDE and has no engine constructor. You cannot run it by naming it.
:::

:::{prf:definition} Geometric Gas
:label: def-variant-geometric

The **Geometric Gas** is the gas variant with the following components. It has no fixed numerical preset; every listed symbol is a free parameter constrained by the assumptions of {ref}`sec-gg-axioms`.

| # | Component | Specification |
|---|---|---|
| 1 | $\mathcal W$ | $(x,v)\in\mathbb R^d\times\mathbb R^d$ |
| 2–3 | $\mathsf C^{D}$, $\mathsf C^{C}$ | The companion laws contained in the specified jump kernel $r_N(S,dS')$ of {prf:ref}`def-gg-sde`; the reference choice is the soft Gaussian kernel of {prf:ref}`def-fg-soft-companion-kernel` |
| 4 | $d_{\mathrm{alg}}$ | The phase-space distance of {prf:ref}`def-fg-algorithmic-distance` |
| 5 | $\mathsf Z$ | The $\rho$-localized moments of {prf:ref}`def-gg-rho-moments` with kernel {prf:ref}`def-gg-localization-kernel` and floor $s_*>0$: $Z_\rho=(d-\mu_\rho)/\sqrt{s_\rho^2+s_*^2}$ |
| 6 | $\mathsf g$ | The exponential field $V_i=\eta^{\alpha+\beta}\exp[\alpha Z_\rho(R,x_i)+\beta Z_\rho(d_{\mathrm{alg}},x_i)]$ of {prf:ref}`def-gg-fitness-potential`, $\eta>0$ |
| 7–8 | $\mathsf A$, $\mathsf T$ | The complete cloning update contained in $r_N(S,dS')$, or the discrete cloning operator $P_{\mathrm{clone}}$, as declared ({prf:ref}`axiom-gg-cloning`) |
| 9 | $\mathsf K$ | The Stratonovich dynamics of {prf:ref}`def-gg-sde`: $dv_i=[-\nabla U(x_i)+F_i(S)-\gamma v_i+\nu\sum_{j\ne i}W_{ij}(X)(v_j-v_i)]dt+\Sigma_i(S)\circ dW_i$ with adaptive force $F_i=\epsilon_F\nabla_{x_i}V_i$, row-normalized viscous weights $W_{ij}=K_{ij}/\sum_{l\ne i}K_{il}$, and $\Sigma_i=(H_i+\epsilon_\Sigma I)^{-1/2}$, $H_i=\nabla_{x_i}^2V_i$. No curl rotation |
| 10 | $\mathsf B$ | Killing by an interior rate or an absorbing boundary, represented separately from the jump kernel ({prf:ref}`def-gg-sde`) |
| 11 | $\mathsf R$ | A reward measurement $R$ and a confining potential $U$ ({prf:ref}`axiom-gg-confining-potential`) |
| 12 | $\mathsf G$ | The fitness-Hessian metric $g_i=H_i+\epsilon_\Sigma I$, evaluated with the declared differentiation convention; it is consumed by the noise factor of $\mathsf K$ |

The equation of {prf:ref}`def-fractal-set-sde` has the same structure: conservative force, adaptive force, viscous coupling, friction, and fitness-adapted Stratonovich noise. It is written with the unnormalized weights $K_\rho(x_i,x_j)$ and the adaptive force $-\nabla V_{\mathrm{fit}}$, whereas {prf:ref}`def-gg-sde` uses the row-normalized weights $W_{ij}$ and $F_i=\epsilon_F\nabla_{x_i}V_i$. The Geometric Gas is defined by {prf:ref}`def-gg-sde`; the results of {doc}`../convergence_program/17_geometric_gas` refer to that convention.

The continuous jump realization and the discrete update are different operators ({prf:ref}`def-gg-sde`). An instance must declare which of them it is. No `GasConfig` constructor of the Algorithmic Gas engine fixes this variant.
:::

### 6.2 Established results and hypotheses

The results of {doc}`../convergence_program/17_geometric_gas` apply to the Geometric Gas under the assumptions named in each statement: confinement ({prf:ref}`axiom-gg-confining-potential`), positive friction ({prf:ref}`axiom-gg-friction`), a cloning estimate for the chosen functional ({prf:ref}`axiom-gg-cloning`), the spectral margin $\epsilon_\Sigma-\Lambda_->0$ ({prf:ref}`axiom-gg-ueph`), and the viscous kernel and degree comparison ({prf:ref}`axiom-gg-viscous-kernel`).

1. **Coefficients.** {prf:ref}`lem-gg-adaptive-force-bounded`, {prf:ref}`thm-gg-ueph-construction` and {prf:ref}`cor-gg-well-posedness` give bounded adaptive forces, uniform velocity ellipticity and strong solutions, under normalized measurement bounds and the spectral margin.
2. **Regularity of the fitness.** {prf:ref}`thm-c3-regularity` in {doc}`../convergence_program/14_a_geometric_gas_c3_regularity` and {prf:ref}`thm-cinf-regularity-zscore-full` in {doc}`../convergence_program/14_b_geometric_gas_cinf_regularity_full` supply the derivative bounds that the coefficient calculus consumes.
3. **Moments and mixing.** {prf:ref}`thm-gg-foster-lyapunov-drift` transfers a backbone drift when every added term has a proved bound and the net rate is positive. {prf:ref}`thm-gg-geometric-ergodicity` gives Harris convergence of a conservative skeleton that satisfies the drift, minorization and aperiodicity hypotheses; for a killed process it refers to the QSD criteria of {doc}`../convergence_program/06_convergence`.
4. **Entropy.** {prf:ref}`thm-gg-lsi-main` and {prf:ref}`prop-gg-entropy-fisher-gap` concern a specified joint law that satisfies one of the structural criteria named there.
5. **Mean-field limit of the Geometric Gas.** {prf:ref}`prop-gg-propagation-chaos` is a finite-time comparison conditional on a differential inequality, and {prf:ref}`thm-gg-mean-field-lsi` passes a uniform LSI to an identified stationary marginal limit.

These are conditional statements. The spectral margin, the $N$-uniformity of the measurement bounds, the cloning estimate for the declared cloning operator, the minorization, and the identification of the stationary law are hypotheses; they are not verified in this volume for a general instance, and each must be checked for the instance under study. A discrete-time instance must use bounds on the actual kernel difference, as stated in {prf:ref}`thm-gg-foster-lyapunov-drift`.

:::{div} feynman-prose
This is a different situation from the previous two sections, and the difference is worth naming precisely. For the Einstein–Hilbert Gas there are no long-time results. For the Geometric Gas there are plenty of them — drift, ergodicity, a log-Sobolev inequality, a propagation-of-chaos comparison — and every single one is an implication whose antecedent is an axiom nobody has verified for a concrete instance.

That is a perfectly respectable way to do mathematics. The theorems say: *if* the potential confines, *if* the friction is positive, *if* the ridge beats the negative curvature, *if* the cloning operator satisfies its estimate, *if* the viscous degrees compare, *then* here is your geometric ergodicity. What it is not is a licence to quote the conclusion. Between "we proved $A\Rightarrow B$" and "we have $B$" sits the entire job of checking $A$, and for a general instance that job is open.

There is a trap in the last sentence of the section, too. This variant is written as a continuous-time SDE, but anything you actually run is a discrete update, and those are different operators — {prf:ref}`def-gg-sde` says so explicitly. A drift inequality proved for the generator is not automatically a drift inequality for the one-step kernel; you need bounds on the actual difference between them. Declare which object you mean, then prove things about *that* one.
:::

(sec-variants-latent)=
## 7. Latent Fractal Gas

The Latent Fractal Gas runs on a latent chart with a Riemannian metric and a velocity-dependent reward. Its defining chapter is {doc}`02_fractal_gas_latent`.

### 7.1 Component tuple

:::{div} feynman-prose
Everything so far has run in a space where the coordinates mean something and Euclidean distance is the right distance. Now suppose they don't. You have learned a latent chart — the code of a trained model, say — and distances in raw coordinates are meaningless, because the chart stretches some directions and squashes others. The fix is the obvious one: carry a metric $G(z)$ with you and measure everything through it. Inner products, the velocity cap, the noise factor, the reward — all of them are computed with $G$ rather than with $\delta_{ij}$.

The second change is more of a surprise. The reward is not a function of where you are, it is a function of where you are *going*: $r_i=\langle\mathcal R(z_i),v_i\rangle_G$, the pairing of a 1-form with your velocity. You are not paid for sitting in a good place. You are paid for moving in a good direction. And once the reward is a 1-form, it has an exterior derivative $\mathcal F=d\mathcal R$, which is a 2-form, which acts on velocities as a rotation — that is the Boris curl term in component 9. A curl appears here not because anybody added a magnetic field, but because the reward was given an orientation and geometry did the rest.

Be careful with one thing. Component 1 insists that anything affecting termination lives in the state. A latent code alone is usually not Markov; if an episode clock or a learner's parameters can end a walker's life, they are part of the state, or none of the kernel statements hold.
:::

:::{prf:definition} Latent Fractal Gas
:label: def-variant-latent

The **Latent Fractal Gas** is the gas variant with the following components. The reference parameter values are those of {prf:ref}`def-latent-fractal-gas-parameters`; they are reference choices and are not fixed by a constructor of the Algorithmic Gas engine.

| # | Component | Specification |
|---|---|---|
| 1 | $\mathcal W$ | $(z,v)\in T\mathcal Z$ in a specified chart of dimension $d_z$, with positive-definite metric $G(z)$ and momentum $p=G(z)v$ ({prf:ref}`def-latent-fractal-gas-state`). Environment, time or learner variables that affect termination belong to the state |
| 2–3 | $\mathsf C^{D}$, $\mathsf C^{C}$ | Independent soft Gaussian draws with bandwidth $\epsilon$ on $\mathcal A\setminus\{i\}$, with fresh randomness for each role; dead walkers draw uniformly from $\mathcal A$ ({prf:ref}`def-latent-fractal-gas-companions`) |
| 4 | $d_{\mathrm{alg}}$ | $d_{\mathrm{alg}}(i,j)^2=\lVert z_i-z_j\rVert ^2+\lambda_{\mathrm{alg}}\lVert v_i-v_j\rVert ^2$ in chart coordinates, with regularizer $\epsilon_{\mathrm{dist}}>0$ |
| 5 | $\mathsf Z$ | Alive-only statistics, global or localized at scale $\rho$, with $\sigma'=\sqrt{\sigma^2+\sigma_{\min}^2}$ ({prf:ref}`def-latent-fractal-gas-fitness`) |
| 6 | $\mathsf g$ | $g_A(u)+\eta$ with $g_A(u)=A/(1+e^{-u})$, exponents $\alpha_{\mathrm{fit}},\beta_{\mathrm{fit}}$ |
| 7 | $\mathsf A$ | $p_i=\min\{1,\max\{0,S_i/p_{\max}\}\}$ with $S_i=(V_{c^{C}(i)}-V_i)/(V_i+\varepsilon_{\mathrm{clone}})$; dead walkers clone with probability one; $q=1$ ({prf:ref}`def-latent-fractal-gas-cloning`) |
| 8 | $\mathsf T$ | Position copy with jitter $\sigma_x\zeta_i$ in the chart; recipient-group inelastic collision $v_j'=V_{\mathrm{COM}}+\alpha_{\mathrm{rest}}(v_j-V_{\mathrm{COM}})$ without rotation, in the declared recipient order |
| 9 | $\mathsf K$ | The Boris-BAOAB sequence of {prf:ref}`def-latent-fractal-gas-kinetic`: force $-\nabla\Phi_{\mathrm{eff}}$, row-normalized viscous force of {prf:ref}`def-latent-fractal-gas-viscous-force` with $(\nu_{\mathrm{visc}},\ell_{\mathrm{visc}})$, Boris rotation associated with $\beta_{\mathrm{curl}}G^{-1}\mathcal F$, $\mathcal F=d\mathcal R$; OU stage with friction $\gamma$, temperature $T_c$ and noise factor $\Sigma_{\mathrm{reg}}$ ({prf:ref}`def-latent-fractal-gas-diffusion`); metric velocity cap $\psi_v$ with radius $V_{\mathrm{alg}}$ ({prf:ref}`def-latent-velocity-squashing`) |
| 10 | $\mathsf B$ | Alive mask $\mathcal A=\{i:z_i\in B\}$; revival at the cloning stage; fewer than two alive walkers sends the swarm to the cemetery state $\dagger$ ({prf:ref}`def-latent-fractal-gas-step`) |
| 11 | $\mathsf R$ | $r_i=\langle\mathcal R(z_i),v_i\rangle_G=\mathcal R_{z_i}(v_i)$ for the reward 1-form $\mathcal R$ of {prf:ref}`def-reward-1-form` |
| 12 | $\mathsf G$ | The latent metric $G$ and the clamped fitness-Hessian factor $\Sigma_{\mathrm{reg}}=H_{\mathrm{reg}}^{-1/2}$, consumed by $\mathsf K$ |

The reference implementation is the Python package `src/fragile/fractalai/core`, as cited in the defining chapter.
:::

### 7.2 Established results and hypotheses

1. **Transition.** {prf:ref}`thm-latent-fractal-gas-main` proves that the update of {prf:ref}`def-latent-fractal-gas-step` is a Markov kernel on the state space enlarged by $\dagger$, under its measurability and specification hypotheses. It asserts no smoothness, recurrence or entropy inequality.
2. **Identities and bounds.** The collision, selection-alignment, viscous, Boris-rotation and thermostat lemmas of {doc}`02_fractal_gas_latent` (for example {prf:ref}`lem-latent-fractal-gas-boris-energy` and {prf:ref}`lem-latent-fractal-gas-ou-moments`) hold as stated there.
3. **Conditional statements.** {prf:ref}`prop-latent-fractal-gas-drift-iteration` derives moment bounds from an assumed drift inequality. {prf:ref}`prop-latent-fractal-gas-conditional-qsd` derives a unique QSD and its convergence rate from an assumed uniform survival bound and an assumed contraction of the full conditioned block.

**The hypotheses of these conditional statements are not established for the general latent model, so no QSD existence, uniqueness or convergence theorem is established for the Latent Fractal Gas.** The population map of {prf:ref}`def-latent-fractal-gas-mean-field` is a candidate; no mean-field limit theorem is proved for it. The conditions that an instance must verify are the five items of {prf:ref}`rem-latent-fractal-gas-analytic-hypotheses`: geometry and regularity, coercive drift on unbounded spaces, full-transition accessibility, survival control, and quantitative uniformity. The Euclidean estimates transfer only after the latent metric, the velocity-dependent reward, the adaptive diffusion, the cap and the force normalization of {prf:ref}`rem-latent-fractal-gas-splitting-normalization` have been included.

(sec-variants-environment)=
## 8. Environment Gas

The Environment Gas is the reinforcement-learning member of the family: the kinetic operator is one transition of an external environment and the reward is the environment's reward signal.

### 8.1 Component tuple

:::{div} feynman-prose
Here is the variant that makes the twelve-component bookkeeping earn its keep, because it is the one where a walker is not a point at all.

A walker is an entire snapshot of a simulator: the game state, the physics state, the environment's random-number generator, whatever the policy is remembering. You cannot add two snapshots. You cannot take a gradient of one. You cannot jitter one, because there is no coordinate to jitter. So look at what happens to the tuple: the clone transform $\mathsf T$ degenerates to a literal copy — no jitter, no collision, no restitution — because those operations require arithmetic on the state and the state is opaque. The kinetic operator $\mathsf K$ is not an integrator at all; it is "ask the policy for an action, hand it to the environment, take whatever comes back." There is no thermostat and no viscosity, and the engine enforces that: viscosity is only accepted with a BAOAB integrator.

What survives is the part of the algorithm that never needed coordinates. You can still measure a dissimilarity between *observations*, still standardize, still compute a fitness from reward and diversity, still decide who copies whom. That is the whole selection machinery, untouched.

And it explains where the exploration comes from here. In the Euclidean Gas, two walkers that clone onto the same point separate because of Gaussian jitter. Here there is no jitter. Two identical snapshots separate only if the environment or the policy is stochastic — and if neither is, they never separate at all.
:::

:::{prf:definition} Environment Gas
:label: def-variant-environment

Let $(\mathcal S_{\mathrm{env}},\mathcal B_{\mathrm{env}})$ be a standard Borel space of complete environment snapshots, $\mathcal U$ a measurable action space, $P_{\mathrm{env}}(ds'\mid s,u)$ a probability kernel on $\mathcal S_{\mathrm{env}}$, $\phi:\mathcal S_{\mathrm{env}}\to\mathbb R^{d_o}$ a measurable feature map, $\pi(du\mid o)$ a measurable policy kernel, $\varrho:\mathcal S_{\mathrm{env}}\to\mathbb R$ a measurable reward read from the snapshot, and $\mathrm{term}:\mathcal S_{\mathrm{env}}\to\{0,1\}$ a measurable termination flag. The **Environment Gas** is the gas variant with the following components. It has no fixed numerical preset.

| # | Component | Specification |
|---|---|---|
| 1 | $\mathcal W$ | $s\in\mathcal S_{\mathrm{env}}$, with observation $o=\phi(s)$. There is no velocity field. Every quantity that affects the transition, the reward or the termination, including the environment's random-number state and any policy memory, is part of $s$ |
| 2–3 | $\mathsf C^{D}$, $\mathsf C^{C}$ | Two declared companion laws on $\mathcal A$; free |
| 4 | $d_{\mathrm{alg}}$ | A declared distance or dissimilarity between observations $o_i,o_j$; free |
| 5–6 | $\mathsf Z$, $\mathsf g$ | A declared standardizer and positive map; free |
| 7 | $\mathsf A$ | A declared acceptance rule of the form of {prf:ref}`def-fg-cloning-decision`; terminated rows clone with probability one |
| 8 | $\mathsf T$ | Literal copy of the complete donor snapshot $s_{c^{C}(i)}$, its observation and its reward. No jitter and no collision: an opaque snapshot admits no coordinate edit |
| 9 | $\mathsf K$ | One environment transition per row: $u_i\sim\pi(\cdot\mid o_i)$, $s_i'\sim P_{\mathrm{env}}(\cdot\mid s_i,u_i)$. No thermostat, no viscous coupling, no curl rotation, no cap |
| 10 | $\mathsf B$ | $a_i=1-\mathrm{term}(s_i)$, together with finiteness of the observation; revival at the cloning stage by copying an alive donor |
| 11 | $\mathsf R$ | $r_i=\varrho(s_i)$, for example the cumulative episode reward, maximized |
| 12 | $\mathsf G$ | $\varnothing$ |
:::

:::{prf:remark} Environment Gas: component and configuration correspondence
:label: rem-variant-environment-rust

No `GasConfig` constructor fixes this variant. The engine components that realize it are listed below; a documented instance is the opaque-simulator profile in {ref}`sec-algorithmic-gas-validation`.

| Component | Engine element | Value or enum arm |
|---|---|---|
| $\mathcal W$ | opaque `StateStore` snapshots and the observation field produced by the `DomainAdapter` | domain-defined |
| $\mathsf K$ | `kinetic.integrator` | `KineticKind::Environment`, which calls `DomainAdapter::transition` on the eligible rows and consumes no numerical noise source |
| $F^{\mathrm{visc}}$, curl | `qft.viscosity`, `qft.graph_viscosity`, `qft.curl` | each must be `None`: the engine accepts viscosity only with `KineticKind::Baoab`, and curl rotation only with graph viscosity |
| $\mathsf T$ | `clone_transform` | `CloneTransform::default()`: no jitter and no restitution |
| $\mathsf B$ | `boundary` | `BoundaryPolicy::ExternalTermination`, optionally composed with other policies |
| $\mathsf R$ | reward source | a `RewardSource` reading the environment reward |
| $\mathsf G$ | `geometry` | `None` |
:::

### 8.2 Established results and hypotheses

**No convergence theorem, no QSD theorem and no mean-field limit theorem is established in this volume for the Environment Gas.** Under the measurability hypotheses written into {prf:ref}`def-variant-environment`, and when the declared companion laws, standardizer, positive map and acceptance rule are measurable, the update is a composition of finite-valued companion and acceptance kernels with the product kernel $\prod_iP_{\mathrm{env}}(\cdot\mid s_i,u_i)\pi(du_i\mid\phi(s_i))$, and is therefore a Markov kernel on the marked swarm space; this is the only general statement available.

The kinetic, hypocoercive and entropy chapters of the convergence program assume a position–velocity state with Langevin noise and do not apply. The applicable route is the axiomatic framework of {doc}`../convergence_program/01_fragile_gas_framework`. For a given environment the following must be verified:

1. the environment snapshot is a complete Markov state, as required in item 1 of {prf:ref}`def-variant-environment`, and the policy is fixed or its learner state is included;
2. the framework axioms for the chosen $d_{\mathrm{alg}}$, reward and perturbation kernel: bounded algorithmic diameter ({prf:ref}`axiom-bounded-algorithmic-diameter`), reward regularity, and continuity and non-degeneracy of the perturbation measure $P_{\mathrm{env}}$ in the chosen metric ({prf:ref}`def-fragile-swarm-instantiation`);
3. accessibility and a minorization for the composed kernel, which depend entirely on $P_{\mathrm{env}}$; a deterministic environment under a deterministic policy supplies none, and literal copies then continue identically;
4. survival control for the termination rule, as in {prf:ref}`thm-main-convergence`.

(sec-variants-comparison)=
## 9. Variant comparison

:::{div} feynman-prose
Now put them all in one table and read down the columns, because the pattern that emerges is not the one the names suggest.

Look at the "Convergence or QSD theorem" row first. One column says established. One says conditional. Four say not established. That is the actual state of the subject, and no amount of family resemblance between the tuples changes it.

Then look at the rows that differ. Every other variant differs from the Euclidean Gas in the *kinetic* rows, and most differ in the *geometry* row. The fitness and cloning rows change in their details — a matching instead of independent draws, an exponential instead of a logistic — but never in their pattern. That is not an accident: selection is the robust part of this algorithm. Measuring reward and diversity, standardizing, squashing through a logistic, letting the loser copy the winner — that pattern survives being moved to a latent chart, to a graph, to an opaque simulator. What does not survive is motion. Change how walkers move and you change what you can prove, because everything in the convergence program that was hard was about the kinetic stage.

One more row deserves a second look: "Companion map involutive." Only the Einstein–Hilbert Gas says yes, and that single entry is worth an entire section — {ref}`sec-variants-measurability` — because a mutual pairing silently annihilates a whole class of measurements. A yes in that row is the difference between a statistic that reports something and a statistic that reports zero no matter what the gas does.
:::

:::{prf:remark} Component comparison of the named variants
:label: rem-variants-comparison

Abbreviations: EG Euclidean Gas, VEG Viscous Euclidean Gas, EHG Einstein–Hilbert Gas, GG Geometric Gas, LFG Latent Fractal Gas, EnvG Environment Gas. "Free" means a parameter or a declared choice of the instance.

| Component | EG | VEG | EHG | GG | LFG | EnvG |
|---|---|---|---|---|---|---|
| $\mathcal W$ | $(x,v)\in\mathbb R^{2d}$ | as EG | $(x,v)\in\mathbb R^{2d}$, last coordinate Euclidean time for $d\ge3$ | $(x,v)\in\mathbb R^{2d}$ | $(z,v)\in T\mathcal Z$ | snapshot $s$; no velocity |
| $\mathsf C^{D}$ | independent Gaussian, $\epsilon_D=2$ | as EG | uniform perfect matching | declared; reference soft Gaussian | independent soft Gaussian, $\epsilon$ | free |
| $\mathsf C^{C}$ | independent Gaussian, $\epsilon_C=2$ | as EG | independent uniform perfect matching | declared | independent soft Gaussian, $\epsilon$ | free |
| Companion map involutive | no | no | yes | no | no | depends on the law |
| $d_{\mathrm{alg}}$ | squashed phase space | as EG | Euclidean on positions | phase space | chart phase space | on observations |
| $\mathsf Z$ | global, $\sqrt{\operatorname{Var}+0.1^2}$ | as EG | sample deviation $+10^{-30}$ | $\rho$-localized, floor $s_*$ | global or $\rho$-localized, $\sigma_{\min}$ | free |
| $\mathsf g$ | $2/(1+e^{-z})+0.1$ | as EG | $2/(1+e^{-z})$ | $\eta\,e^{Z}$ per channel | $A/(1+e^{-z})+\eta$ | free |
| Cloning period $q$ | $1$ | $1$ | $20$ | declared | $1$ | free |
| $\varepsilon_{\mathrm{clone}}$, $p_{\max}$ | $10^{-6}$, $1$ | as EG | $0$, $1$ | declared | free (reference $0.01$, $1$) | free |
| Jitter $\sigma_{\mathrm{clone}}$ | $0.1$ | $0.1$ | $0$ | declared | free (reference $0.1$) | none |
| Collision | $\alpha=0.5$, Haar $R_C$, connected components | as EG | $\alpha=1$, $R_C=I$: velocities unchanged | declared | $\alpha_{\mathrm{rest}}$, no rotation, recipient groups | none |
| Integrator | BAOAB | BAOAB | BAOAB with Boris B stages | SDE of {prf:ref}`def-gg-sde` | Boris-BAOAB on $(T\mathcal Z,G)$ | environment transition |
| Conservative force | $-\nabla U$ | $-\nabla U$ | $0$ | $-\nabla U$ | $-\nabla\Phi_{\mathrm{eff}}$ | — |
| Adaptive force | none | none | none | $\epsilon_F\nabla V_i$ | none | — |
| Thermostat | $\gamma=1$, $\sigma_v=1$; variance $1/2$ | as EG | $\gamma=1$; variance $T$ | $\gamma$, $\Sigma_i=(H_i+\epsilon_\Sigma I)^{-1/2}$ | $\gamma$, $T_c$, $\Sigma_{\mathrm{reg}}$ | — |
| Viscous coupling | $0$ | Gaussian kernel $K_\rho$, coupling $\nu$ | tessellation graph, $\nu=3$ | row-normalized kernel, $\nu$ | row-normalized kernel, $\nu_{\mathrm{visc}}$ | — |
| Curl rotation | none | none | $\beta_{\mathrm{curl}}=1$, curl of $F^{\mathrm{visc}}$ | none | $\beta_{\mathrm{curl}}$, $\mathcal F=d\mathcal R$ | — |
| $\sigma_x$, $V_{\mathrm{alg}}$ | $0.1$, $2$ | as EG | $0$, $\infty$ | not part of the SDE | $0$, free | — |
| $\mathsf B$ | box $[-2,2]^d$, terminal | as EG | $\mathbb R^d$, unbounded | killing rate or absorbing boundary | alive mask $B$, cemetery for $M<2$ | termination flag |
| $\mathsf R$ | $-U(x)-\lambda_{\mathrm{vel}}\lVert v\rVert ^2$ | as EG | $R_i\sqrt{\det g_i}$ | $R(x)$ | $\mathcal R_z(v)$ | environment reward |
| $\mathsf G$ | $\varnothing$ | $\varnothing$ | Delaunay tessellation | fitness-Hessian metric | $G$ and fitness-Hessian factor | $\varnothing$ |
| Engine constructor | `GasConfig::euclidean` | `GasConfig::viscous_euclidean` | `GasConfig::einstein_hilbert` | none | none | none (`KineticKind::Environment`) |
| Convergence or QSD theorem | established under the hypotheses of {ref}`sec-variants-euclidean` | not established for $\nu>0$ | not established | conditional on {ref}`sec-gg-axioms` | not established; conditional criteria only | not established |
:::

:::{admonition} Which variant should I run?
:class: feynman-added tip

The formal tables above say what each variant *is*. This one is a practical crib sheet for picking one; it makes no claim not already made in the sections it points to.

| If your job is… | Run | Because | And the price is |
|---|---|---|---|
| minimize a function you can evaluate (and ideally differentiate) on $\mathbb R^d$ | **Euclidean Gas** | it is the only variant carrying proved finite-$N$ and mean-field results, and the only one with a preset that fixes every constant | the box $[-2,2]^d$ and the normalization of your objective are part of the algorithm, not of the problem |
| the same, but you need a colour observable | **Viscous Euclidean Gas**, $\nu>0$ | it is the smallest change that makes $F^{\mathrm{visc}}$ nonzero, so the colour state of {prf:ref}`thm-sm-su3-emergence` is defined | every long-time theorem of {ref}`sec-variants-euclidean` goes away ({ref}`sec-variants-viscous-euclidean`) |
| study emergent geometry, curvature, or a Euclidean-time axis | **Einstein–Hilbert Gas** | it is the only variant producing a graph record and a distinguished coordinate | no confinement, a gate every 20 steps, exchange-odd frame means identically zero, and no long-time theorem |
| use fitness curvature to precondition exploration | **Geometric Gas** | adaptive force and Hessian-shaped noise | no engine constructor, and every result conditional on {ref}`sec-gg-axioms` |
| search in a learned latent chart with a direction-valued reward | **Latent Fractal Gas** | metric-aware distances, cap and noise; reward as a 1-form | conditional criteria only; you must supply the metric and verify its regularity |
| drive an external simulator or RL environment | **Environment Gas** | the only tuple that accepts opaque, non-arithmetic states | no jitter, no collisions, no thermostat; exploration comes entirely from the environment or the policy |

Two entries that are *never* the answer to "which should I run": *Algorithmic Gas*, which is the engine that runs whichever you picked, and *mean-field limit*, which is something you prove about one, not something you execute.
:::

(sec-variants-measurability)=
## 10. Field-theoretic measurability

The direct observables of {doc}`../2_fractal_set/04_standard_model` are functions of recorded fields. Which of them is defined for a variant is decided by its component tuple. This section states the requirements, proves the cancellation that mutual pairings impose, and tabulates the outcome.

:::{div} feynman-prose
Now we come to the part I think is the most useful in the chapter, because it settles arguments before they start.

The Standard Model chapter, {doc}`../2_fractal_set/04_standard_model`, defines a pile of field-theoretic observables — colour contractions, $U(1)$ amplitudes and phases, $SU(2)$ doublets, correlators along a Euclidean-time axis. Somebody will eventually want to compute one of them from a run. And the question "can I measure this?" has, for once, a purely mechanical answer, because each observable is a formula in certain recorded quantities, and whether those quantities exist is decided by the tuple you chose. Not by how long you run. Not by how many walkers you use. By the tuple.

So the definition below does a boring, valuable thing: it lists the *records* — velocity, colour, fitness-and-companion, graph, time axis — and then says which records each family of observables needs. After that, availability is a lookup.

Keep one caution in mind, and the definition states it too: "available" is a statement about existence, not about meaning. It says the formula does not divide by zero and does not reference a field that isn't there. It does not say the number you compute is interesting, or that it deserves the name printed on it. A quantity can be perfectly well-defined and still be a constant, or noise, or an artifact of the companion law — and in fact we are about to meet a case where an available observable is identically zero for reasons that have nothing to do with physics.
:::

:::{prf:definition} Records and measurement families
:label: def-variant-measurement-families

For a gas variant $\mathcal V$ define the following **records** of one step.

1. **Velocity record**: $\mathcal W$ has a velocity field $v_i\in\mathbb R^d$.
2. **Colour record**: a velocity record together with the viscous force $F_i^{\mathrm{visc}}$ evaluated at a B stage and paired with the velocity at which it was evaluated. The colour state $c_i$ of {prf:ref}`thm-sm-su3-emergence` is defined on the rows with $F_i^{\mathrm{visc}}\ne0$.
3. **Fitness and companion records**: $V_{\mathrm{fit},i}$, $c^{D}$, $c^{C}$ and the clone decisions $A_i$. Every variant has them.
4. **Graph record**: a neighbor graph with edge weights produced by $\mathsf G$.
5. **Euclidean-time axis**: a position coordinate that the tuple distinguishes from the others.

The **measurement families** and their requirements are:

| Family | Observables | Required records |
|---|---|---|
| Colour channels | colour contractions and their correlators ({prf:ref}`def-sm-direct-color-contractions`) | colour record and the companion role or roles entering the contraction |
| $U(1)$ | companion amplitudes and phases ({prf:ref}`thm-sm-u1-emergence`) | fitness record and $c^{D}$ |
| $SU(2)$ and chirality | companion doublets ({prf:ref}`def-sm-direct-companion-doublet`) and walker roles ({prf:ref}`def-sm-walker-chirality`) | fitness record, $c^{C}$ and the clone decisions |
| Multiscale and graph | observables summed over graph neighborhoods or graph distances | graph record |
| Euclidean-time axis | correlators in the separation $t_i-t_j$ along a position coordinate | Euclidean-time axis |

A family is **available** for $\mathcal V$ when its required records exist and are not identically degenerate, and **unavailable** otherwise. Availability is a property of the tuple; it does not assert that a statistic has a particular value or a physical interpretation.
:::

:::{div} feynman-prose
Before the statement, let me give you the picture, because the algebra below is three lines and the content is entirely in the picture.

Suppose you want to know something about how walkers differ from their companions. The natural thing to write down is a frame average: at each step, for each walker $i$, compute some quantity comparing $i$ to its companion $c(i)$, add them all up, divide by $N$, and plot the series. People do this constantly. It is the obvious estimator.

Now suppose the quantity you chose is *exchange-odd*: swapping the two walkers flips its sign. Almost everything you would naturally reach for is exchange-odd — the velocity difference $v_j-v_i$, the position difference, the fitness gap $V_j-V_i$, and any odd function of those.

And suppose the companion law is a *mutual pairing*: walkers are matched two by two, so that if $i$'s companion is $j$, then $j$'s companion is $i$. That is exactly what the Fisher–Yates matching in the Einstein–Hilbert Gas does.

Then your sum contains both $O_{ij}$ and $O_{ji}$, which are negatives of each other. They cancel. Every pair cancels. The average is zero — not on average, not to leading order in $1/N$, but exactly, for every realization, at every step, whatever the gas is doing. You have built a thermometer that reads zero because of how it was wired, and if you don't know that, you will spend a week interpreting the reading.
:::

:::{prf:proposition} Exchange-odd frame sums vanish on an involutive companion map
:label: prop-exchange-odd-cancellation

Let $I$ be a finite index set, $c:I\to I$ a map with $c(c(i))=i$ for every $i\in I$, $\mathbb K\in\{\mathbb R,\mathbb C\}$, $\mathbb V$ a vector space over $\mathbb K$, and $O:I\times I\to\mathbb V$ an exchange-odd operator: $O_{ji}=-O_{ij}$ for all $i,j\in I$. Then for every weight $w:I\to\mathbb K$,

$$
\sum_{i\in I}w_iO_{i\,c(i)}=\frac12\sum_{i\in I}\bigl(w_i-w_{c(i)}\bigr)O_{i\,c(i)}.
$$

In particular, if the weight is pair-symmetric, $w_i=w_{c(i)}$ for every $i$, then

$$
\sum_{i\in I}w_iO_{i\,c(i)}=0.
$$

Each fixed point $i=c(i)$ contributes $O_{ii}=0$.
:::

:::{prf:proof}
**Step 1. Fixed points.** Setting $j=i$ in $O_{ji}=-O_{ij}$ gives $2O_{ii}=0$, hence $O_{ii}=0$.

**Step 2. Reindexing.** Since $c\circ c=\mathrm{id}_I$, the map $c$ is a bijection of $I$. Put $\Sigma=\sum_{i\in I}w_iO_{i\,c(i)}$ and substitute $i=c(j)$:

$$
\Sigma=\sum_{j\in I}w_{c(j)}O_{c(j)\,c(c(j))}=\sum_{j\in I}w_{c(j)}O_{c(j)\,j}=-\sum_{j\in I}w_{c(j)}O_{j\,c(j)}.
$$

**Step 3. Conclusion.** Adding the definition of $\Sigma$ to the last expression gives $2\Sigma=\sum_{i\in I}(w_i-w_{c(i)})O_{i\,c(i)}$, which is the first identity. If $w_i=w_{c(i)}$ every term vanishes and $\Sigma=0$. $\square$
:::

:::{div} feynman-prose
Notice what the proof did *not* use. No probability. No largeness of $N$. No property of the gas, the reward, the potential or the noise. Two facts: $c$ is its own inverse, and $O$ flips sign under exchange. That is the entire input. So the conclusion is correspondingly brutal — this is not "the signal is small," it is "the estimator is identically zero and always was."

I want to be careful not to overclaim in the other direction, though, because there are three real escape hatches and they are all in {prf:ref}`rem-exchange-odd-scope`.

First, the weight. The general identity is $\Sigma=\frac12\sum_i(w_i-w_{c(i)})O_{i\,c(i)}$, so what is actually killing you is not the pairing, it is *pair-symmetric weighting* — counting both members of each pair the same way. Break that symmetry and the signal comes back. Count only the walker whose companion is fitter, say; the pair now contributes once instead of twice, and $\Sigma$ is a genuine sum over oriented pairs.

Second, the operator. Exchange-odd is a strong condition and plenty of interesting quantities fail it. The normalized score $S_i=(V_{c(i)}-V_i)/(V_i+\varepsilon_{\mathrm{clone}})$ looks antisymmetric in the numerator, but the denominator carries $i$ and not $c(i)$, so $S_{c(i)}\ne-S_i$ and nothing cancels.

Third, the companion law. Draw companions independently, as the Euclidean, Viscous Euclidean, Geometric and Latent gases do, and $c$ is not an involution at all — $i$ picks $j$ while $j$ picks somebody else entirely. The hypothesis fails and the proposition says nothing whatsoever. It constrains exactly one of our six variants.

So the rule to carry away is narrow and sharp: *mutual pairing plus exchange-odd operator plus unweighted frame average equals zero.* Break any one of the three and you are back in business.
:::

:::{prf:remark} Scope of the cancellation
:label: rem-exchange-odd-scope

1. **Where it applies.** In the Einstein–Hilbert Gas both $c^{D}$ and $c^{C}$ are involutions of $\mathcal A$ and $\mathcal A=\{1,\ldots,N\}$ ({prf:ref}`prop-variant-eh-identities`). An unsplit frame average has $w_i=1$. Hence the frame mean of every exchange-odd operator is identically zero at every step and for every realization: for example $V_{\mathrm{fit},j}-V_{\mathrm{fit},i}$, $v_j-v_i$, $x_j-x_i$, and $\sin[(V_{\mathrm{fit},j}-V_{\mathrm{fit},i})/\hbar_{\mathrm{eff}}]$. All autocorrelations of such a frame-mean series vanish. {prf:ref}`cor-sm-paired-doublet-cancellation` is the case $O_{ij}=a_i-a_j$.
2. **Where it does not apply.** The normalized score $S_i(j)=(V_{\mathrm{fit},j}-V_{\mathrm{fit},i})/(V_{\mathrm{fit},i}+\varepsilon_{\mathrm{clone}})$ is not exchange-odd, because the two denominators differ. Exchange-even operators are unaffected. With independent companion draws (Euclidean, Viscous Euclidean, Geometric and Latent variants) the map $c$ is in general not an involution and the proposition gives no constraint.
3. **What remains available on a mutual pairing.** By the first identity, a weight that is not pair-symmetric retains the signal: the oriented weight $w_i=\mathbf 1_{\{V_{\mathrm{fit},c(i)}>V_{\mathrm{fit},i}\}}$ gives $\Sigma=\sum_{i:V_{c(i)}>V_i}O_{i\,c(i)}$, and role masks act in the same way. Correlators that pair an operator at one step with an operator at a later step through a fixed source row are not frame sums of the above form and are not constrained.
:::

:::{div} feynman-prose
Before the summary table, let me walk through its most consequential entry — the one that says the Euclidean Gas has no colour.

Go back to {prf:ref}`thm-sm-su3-emergence` and look at what the colour state *is*. You take the viscous force on walker $i$, divide it by its own norm to get a unit vector, and attach a phase to each component. The whole construction is a *direction*: it says which way the crowd is pulling this walker. The magnitude is deliberately thrown away, which is what makes the object a unit vector.

Now read component 9 of {prf:ref}`def-variant-euclidean`: $F^{\mathrm{visc}}\equiv0$. Not small. Not usually zero. Identically zero, on every walker, at every step, by definition of the variant. So the encoding reads $0/\|0\|$. There is no direction to extract, because there is no vector. The colour state is not weak or noisy in the Euclidean Gas; it does not exist.

And you cannot rescue it by substituting some other force. The conservative force $-\nabla U$ is available, but it is a gradient of a function on configuration space — it knows nothing about the other walkers and nothing about velocities, so whatever you encoded would describe the landscape and not the crowd; it is not the object the theorem defines. The softened denominator $\sqrt{\|F\|^2+\delta^2}$ doesn't help either: the theorem points out that this gives a vector of norm *at most* one, not a unit vector, and at $F=0$ it returns the zero vector, which carries no direction at all. The construction needs a genuinely nonzero force, and only a coupling between walkers supplies one.

So there are exactly two ways to get colour, and both cost something. Turn on $\nu>0$ and you have the Viscous Euclidean Gas — one changed component, and by {ref}`sec-variants-viscous-euclidean` no convergence theorem survives. Or run the Einstein–Hilbert Gas, whose graph viscous force is nonzero by construction — and give up confinement, time homogeneity and every long-time result at once.

Even then, be careful. A nonzero coupling constant does not guarantee a nonzero force. Reason 3 in the list below makes the point concretely: the Einstein–Hilbert force $\nu\sum_j w_{ij}(v_j-v_i)$ vanishes identically on a population whose velocities all agree — which is exactly the reference initial condition, every walker at the origin at rest. It also vanishes on any walker the tessellation left without neighbors. Those rows have no colour state and must be masked out, not silently treated as zero. Availability is a property of the variant; validity is a property of the sample.
:::

:::{prf:remark} Availability of the measurement families by variant
:label: rem-variants-measurability

A denotes available and U unavailable in the sense of {prf:ref}`def-variant-measurement-families`. The bracketed numbers refer to the structural reasons listed below the table.

| Variant | Colour channels | $U(1)$ | $SU(2)$ and chirality | Multiscale and graph | Euclidean-time axis |
|---|---|---|---|---|---|
| Euclidean Gas | U [1] | A | A | U [5] | U [6] |
| Viscous Euclidean Gas | A for $\nu>0$ [2] | A | A | U [5] | U [6] |
| Einstein–Hilbert Gas | A [3] | A [7] | A, with exchange-odd frame means identically zero [8] | A [9] | A for $d\ge3$ [10] |
| Geometric Gas | A for $\nu>0$ [2] | A | A | U [5] | U [6] |
| Latent Fractal Gas | A for $\nu_{\mathrm{visc}}>0$; U at the reference value $\nu_{\mathrm{visc}}=0$ [1, 2] | A | A | U [5] | U [6] |
| Environment Gas | U [4] | A | A | U [5] | U [6] |

Structural reasons:

1. $F^{\mathrm{visc}}\equiv0$. By {prf:ref}`thm-sm-su3-emergence` the colour encoding divides by $\|F_i^{\mathrm{visc}}\|$ and is undefined at zero force, so the colour state is undefined on every row at every step.
2. The viscous coupling is nonzero for $\nu>0$. The colour state is defined on the rows with $F_i^{\mathrm{visc}}\ne0$; rows with zero force, for example in a population with equal velocities, are excluded, and any replacement at zero force must be declared. The kernel is invariant under simultaneous orthogonal maps of positions and velocities, as the covariance statement of {prf:ref}`thm-sm-su3-emergence` requires.
3. The graph viscous force with $\nu=3$ is the recorded force. It vanishes on a population at rest, in particular at the reference initial condition, and on rows without neighbors; those rows are excluded. For $d\ge3$ the weights are invariant under $O(d-1)\times O(1)$ only, and on Delaunay-generic configurations (Item 5 of {prf:ref}`prop-variant-eh-identities`), so the covariance hypothesis of {prf:ref}`thm-sm-su3-emergence` holds for that subgroup and not for $O(d)$.
4. $\mathcal W$ has no velocity field and $\mathsf K$ has no B stage, so neither factor of the colour encoding exists.
5. $\mathsf G=\varnothing$, or $\mathsf G$ is a per-walker metric without a neighbor graph. No graph record is produced by the tuple.
6. No component distinguishes a position coordinate: every component treats the $d$ coordinates identically, or $\mathcal W$ has no position coordinates. A Euclidean-time correlator requires an axis declared by the analysis, which is then not a property of the variant.
7. The companion law is uniform, so the amplitude moduli of {prf:ref}`thm-sm-u1-emergence` carry no information: the matching law is invariant under relabeling of $\mathcal A$, so for even $M$ the marginal is $P_i(k)=1/(M-1)$ for $k\ne i$, and for odd $M$ each walker is the unmatched one with probability $1/M$ and $P_i(k)=1/M$ for every $k\in\mathcal A$ including $k=i$. The phases remain available. Self-companion rows are excluded from pair observables.
8. {prf:ref}`prop-exchange-odd-cancellation` and {prf:ref}`rem-exchange-odd-scope`. In addition the gate is closed unless $n\equiv0\pmod{20}$, so the clone decisions, and the roles of {prf:ref}`def-sm-walker-role-partition` that depend on them, are nontrivial only on those steps; on all other steps $\Delta_t=\mathrm{SR}_t=\varnothing$.
9. $\mathsf G$ produces the Delaunay neighbor graph with the weights $w^{R}$ and $w^{\mathrm{visc}}$ and the metric edge lengths.
10. For $d\ge3$ the last position coordinate is excluded from the tessellation and is the Euclidean-time coordinate of {prf:ref}`def-variant-einstein-hilbert`. For $d<3$ no coordinate is distinguished and reason 6 applies.
:::

:::{div} feynman-prose
Read that table and you find an uncomfortable trade running right through it. The variant with the richest measurement structure — colour, a graph, a time axis, all available — is the Einstein–Hilbert Gas, and it is also the variant about which we can prove the least. The variant with the theorems is the Euclidean Gas, and three of its five columns say U. Richness of observables and strength of results are, at the moment, anticorrelated. That is a fact about where the work has been done, not a law of nature, but you should plan around it.

The other thing the table teaches is how many different ways an observable can be empty. There is "the field isn't there" (reason 4: no velocities, so no colour factor at all). There is "the field is there but identically zero" (reason 1: the Euclidean viscous force). There is "the record exists but the statistic is constant" (reason 7: a uniform matching law makes every $U(1)$ amplitude modulus the same number, though the phases still carry information). There is "the estimator cancels by construction" (reason 8). And there is "nothing in the tuple distinguishes the coordinate, so the analyst would have to declare an axis by hand, and then it is the analyst's choice and not a property of the gas" (reason 6). Five different failures, and only one of them looks like a missing field.

That last distinction is the one I would most like you to take away. If you pick a time axis yourself in a variant that treats all $d$ coordinates identically, you will get numbers out. They will be reproducible. They will not be measuring a property of the algorithm — they will be measuring your choice. The virtue of doing this bookkeeping once, up front, from the component tuple, is that it tells you which of your numbers are about the gas and which are about you.
:::
