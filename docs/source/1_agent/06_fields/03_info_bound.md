(sec-causal-information-bound)=
# The Causal Information Bound

## TLDR

- Under an explicit capacity permit, stable internal information is summarized by an **interface-area diagnostic** (an
  “area law” analogue).
- Under the separate spherical and overdamped hypotheses, approaching a selected
  radial horizon can slow that radial component (**conditional causal stasis**).
- This chapter defines the operational capacity convention and turns it into a measurable diagnostic.
- Practical implication: under the stated radial ansatz, approaching the selected capacity horizon can slow radial
  updates; this is a conditional diagnostic, not a universal freezing theorem.
- Use the bound to size models, tune horizons, and justify when ontology expansion is necessary.

## Roadmap

1. State the bound and the physical/geometry analogy (area law).
2. State the conditional causal-stasis consequence near a selected horizon.
3. Diagnostics and implementation guidance for monitoring proximity to the bound.

:::{div} feynman-prose
Now I want to tell you about a fundamental limit---perhaps the most important limit in the whole theory. It's the kind of thing that, once you understand it, changes how you think about intelligence, memory, and computation.

Here's the question: How much can an agent stably represent given its finite interface with the world?

The proposed answer is an area-law capacity: under an explicit capacity permit, the chosen boundary area and resolution define an operational $I_{\max}$. That is a modeling convention until the channel argument, units, and any field equation connecting information to geometry have been supplied. It is not a universal statement that intelligence is determined by area alone.

This might remind you of the Bekenstein--Hawking bound. The resemblance is useful as a physical analogy, but it does not import black-hole physics into an agent. The coefficient, resolution, and boundary measure still belong to the model here.

There is a second conditional idea: a singular radial metric can slow radial updates near a selected horizon. We call that Causal Stasis. The formal result controls a radial component under its stated ansatz and force/overdamped hypotheses; it does not by itself say that every overparameterized model freezes.
:::

*Abstract.* Under an explicit capacity permit, the maximum stable information is modeled by the interface area
measured in units of a declared resolution length, the **Levin Length**. The resulting expression is an operational
capacity diagnostic. A separate spherical metric ansatz gives a conditional radial slowdown near a selected horizon;
it does not establish a universal area law or a general freezing theorem. The former derivation is retained in
{ref}`sec-appendix-a-area-law` as a record of the assumptions and normalization choices.

(rb-sensor-bandwidth)=
:::{admonition} Researcher Bridge: The Sensor Bandwidth Ceiling
:class: important
The sensor channel supplies a natural capacity diagnostic for **Model Overload**. Under the explicit permit and
radial ansatz below, a selected radial update can slow near a horizon; this does not prove that every over-parameterized
model stops learning.
:::

(pi-bekenstein-bound)=
::::{admonition} Physics Isomorphism: Bekenstein-Hawking Entropy Bound
:class: note

**In Physics:** The Bekenstein-Hawking entropy of a black hole is $S_{BH} = A/(4\ell_P^2)$ where $A$ is horizon area and $\ell_P$ is the Planck length. Information inside a region cannot exceed its boundary area in Planck units {cite}`bekenstein1973black,hawking1975particle`.

**In Implementation:** The maximum information $I_{\max}$ an agent can stably represent is bounded by its interface area:

$$
I_{\max} = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}}

$$
where $\ell_L$ is the Levin length (Definition {prf:ref}`def-levin-length`) and $\nu_D$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`). For $D=2$: $I_{\max} = \text{Area}/(4\ell_L)$.

The correspondence below is a structural analogy and a unit-matching convention. It does not derive a black-hole
entropy law or identify the agent's capacity with a physical horizon entropy.

**Correspondence Table:**

| Physics | Agent |
|:--------|:------|
| Horizon area $A$ | Interface bandwidth $\text{Area}(\partial\mathcal{Z})$ |
| Planck length $\ell_P$ | Levin length $\ell_L$ |
| Bekenstein-Hawking coefficient $1/4$ | Holographic Coefficient $\nu_D$ |
| Black hole entropy $S_{BH}$ | Representational capacity $I_{\max}$ |
| Horizon singularity ($g_{rr} \to \infty$) | Conditional radial stasis ($G_{rr} \to \infty$, $v^r \to 0$) |
::::

*Cross-references:* This section extends the Capacity-Constrained Metric Law (Theorem
{prf:ref}`thm-capacity-constrained-metric-law`), the Boundary Capacity Definition
({prf:ref}`def-boundary-capacity-area-law-at-finite-resolution`), and the Equation of Motion (Definition
{prf:ref}`def-bulk-drift-continuous-flow`). The remediation connects to Ontological Fusion
({ref}`sec-ontological-fusion-concept-consolidation`).

*Literature:* Holographic bounds {cite}`thooft1993holographic,susskind1995world`; Fisher information geometry
{cite}`amari2016information`; Levin complexity {cite}`levin1973universal`.



(sec-holographic-coefficient)=
## The Holographic Coefficient

:::{div} feynman-prose
Before we can write down the information bound, we need to understand a curious fact: the efficiency of boundary storage depends on dimension.

Think about it this way. In 2D, the boundary of a disk is a circle. In 3D, the boundary of a ball is a sphere. In higher dimensions, the boundary becomes increasingly exotic. The storage efficiency can change substantially with dimension, so the coefficient deserves its own calculation.

You might have expected the opposite. More dimensions, more room, more capacity, right? The reality is subtler. For the coefficient defined below, $\nu_D$ is *non-monotonic*: it rises from $D=2$, peaks around $D \approx 9$, and only then declines toward zero as $D \to \infty$.

This gives a dimensional trend for that normalization. Calling it a "curse of dimensionality" or an efficiency limit requires a separate task and representation model; the coefficient alone does not establish performance.

For $D = 2$, the chosen normalization gives $\nu_2 = 1/4$. It is convenient to compare that number with the Bekenstein--Hawking coefficient, but the numerical match is an analogy or convention, not a derivation from black-hole geometry.
:::

Before defining the Levin Length, we establish the dimension-dependent coefficient that governs holographic capacity.

:::{prf:definition} Holographic Coefficient
:label: def-holographic-coefficient

The **Holographic Coefficient** $\nu_D$ for a $D$-dimensional latent manifold with $(D-1)$-sphere boundary is:

$$
\nu_D := \frac{(D-1)\,\Omega_{D-1}}{8\pi}

$$

where $\Omega_{D-1} = \frac{2\pi^{D/2}}{\Gamma(D/2)}$ is the surface area of the unit $(D-1)$-sphere.

| $D$ | Boundary | $\Omega_{D-1}$ | $\nu_D$ | Numerical |
|-----|----------|----------------|---------|-----------|
| 2   | Circle ($S^1$) | $2\pi$ | $1/4$ | 0.250 |
| 3   | Sphere ($S^2$) | $4\pi$ | $1$ | 1.000 |
| 4   | Glome ($S^3$) | $2\pi^2$ | $3\pi/4$ | 2.356 |
| 5   | 4-sphere ($S^4$) | $8\pi^2/3$ | $4\pi/3$ | 4.189 |
| 6   | 5-sphere ($S^5$) | $\pi^3$ | $5\pi^2/8$ | 6.169 |
| $D \gg 1$ | Hyper-sphere | $\to 0$ | $\to 0$ | Capacity collapse |

*Remark (Dimensional pressure).* The coefficient $\nu_D$ is non-monotonic: it increases from $D=2$ to a peak near $D \approx 9$ ($\nu_9 \approx 9.45$), then decays to zero as $D \to \infty$. The curse of dimensionality applies to this high-dimensional tail. Dimensional reduction pressure arises beyond the peak; $D \approx 3$ lies on the rising portion of the curve.

*Remark (Physics correspondence).* For $D=2$, we recover the Bekenstein-Hawking coefficient $\nu_2 = 1/4$, making the Causal Information Bound $I_{\max} = \text{Area}/(4\ell_L)$ directly analogous to black hole entropy $S = A/(4\ell_P^2)$.

*Units:* $[\nu_D] = \text{dimensionless}$.

:::



(sec-levin-length)=
## The Levin Length

:::{div} feynman-prose
Now we need to define the "pixel size" of thought. How small a distinction can you make?

Every physical measurement has a resolution limit. A camera has pixels; below that scale, you can't see finer detail. A thermometer has precision; below that, temperature differences are meaningless. What's the analogous limit for internal representations?

We call it the Levin Length, after Leonid Levin, who pioneered the theory of algorithmic information. In this volume it is a declared implementation resolution---the information-theoretic "pixel" of the latent model. A one-nat cell is a normalization choice, and its dimension must match the boundary measure being used; the phrase $\ell_L^2$ should not be silently applied in every dimension.

Why does this matter? Once a capacity permit has fixed the units, the area formula can be evaluated in Levin Lengths. A smaller declared resolution can increase the resulting capacity, but that conclusion belongs to the permit and encoding model, not to the name of the length itself.

This gives us a useful engineering question: should capacity be increased by expanding the interface, changing the resolution, or changing the representation? The answer depends on which of those quantities the model holds fixed.
:::

We define a characteristic length scale that represents the minimal resolvable distinction in the latent manifold---the information-theoretic floor of the agent's representational capacity.

:::{prf:definition} Levin Length
:label: def-levin-length

Let $\eta_\ell$ be the boundary $(D-1)$-volume per nat at resolution $\ell$
(Definition {prf:ref}`def-boundary-capacity-area-law-at-finite-resolution`).
For a declared latent dimension $D\ge2$, define the **Levin Length** by

$$
\ell_L := (\nu_D\eta_\ell)^{1/(D-1)}.
$$

This convention makes $\ell_L^{D-1}$ the boundary volume per nat after the
dimension-dependent normalization $\nu_D$ is fixed.

Units: $[\ell_L]=[z]$ when the boundary measure is expressed in the
corresponding normalized coordinate units.

*Interpretation.* A boundary cell of $(D-1)$-volume
$\ell_L^{D-1}$ carries one nat under this operational normalization. In the
two-dimensional Poincaré-disk convention this reads
$C_\partial=\operatorname{Area}/(4\ell_L)$ because $\nu_2=1/4$.

*Remark (Naming).* The name honors Leonid Levin's foundational work on algorithmic information theory and the universal distribution {cite}`levin1973universal`. The Levin Length represents the floor below which distinctions cannot be computationally meaningful.

:::



(sec-saturation-limit)=
## The Saturation Limit

:::{div} feynman-prose
Now let's talk about what happens when you're full.

Imagine filling a balloon with water. At first, it's easy---the balloon stretches, accommodates more. As you approach its capacity, the rubber becomes taut. That is a good picture for the equality $I_{\text{bulk}}=C_\partial$ defined here.

But keep the bookkeeping straight. Saturation of the DPI capacity and the quantity called $I_{\max}$ are the same threshold only after an explicit identification. The capacity-constrained metric law is sourced by its declared risk tensor; it does not automatically turn bulk information density into a uniform stress.

The Schwarzschild-style radial expression below is therefore an exploratory ansatz. At a zero of its denominator, the radial inverse metric may vanish in that ansatz. A full metric divergence, a freeze of all update directions, or a link to information saturation requires the additional field equation, coupling, boundary conditions, and regularity estimates.
:::

We characterize the regime where the agent's representational capacity is fully utilized.

:::{prf:definition} Saturation Limit
:label: def-saturation-limit

The agent is at the **Saturation Limit** when the bulk information volume (Definition {prf:ref}`def-a-bulk-information-volume`) equals the boundary capacity (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`):

$$
I_{\text{bulk}} = C_\partial.

$$
At this limit, the DPI constraint $I_{\text{bulk}} \le C_\partial$ is satisfied with equality.

:::

:::{prf:remark} Formal Spherical Saturation Ansatz
:label: lem-metric-divergence-at-saturation

The following Schwarzschild-style expression is a formal radial ansatz for exploring a
capacity-saturation regime:

$$
A(r) = \left( 1 - \frac{2\mu(r)}{(n-2)r^{n-2}}
- \frac{\Lambda_{\mathrm{eff}}r^2}{n(n-1)} \right)^{-1}.
$$
It is not a consequence of the capacity-constrained metric law without an independent
spherically symmetric field equation and boundary-value calculation. In particular, the
Poincare-disk boundary and the $n=2$ case require separate analysis. Treat $G^{rr}\to0$ at a
zero of the displayed denominator as a diagnostic ansatz, not as a theorem about the learned
metric.
:::





(sec-area-law-derivation)=
## Operational area-law normalization

:::{div} feynman-prose
All right, now let's state the capacity convention clearly. This is where all the pieces can fit together, but only after we declare the permits.

**Step 1 (Capacity permit):** Choose the boundary measure, resolution, and coefficient that define the operational capacity. A bulk-to-boundary identity is an additional statement; it is not a general consequence of integrating the Einstein tensor in arbitrary dimension.

**Step 2 (Spherical ansatz, if used):** A radial denominator and a selected horizon can be studied for a specified spherically symmetric field equation. The ansatz is not automatically a solution of the metric law or a statement about a general boundary.

**Step 3 (Normalization):** If Fisher geometry is used to set the scale, the coupling and units must be fixed consistently. That is a modeling permit until the corresponding derivation is supplied.

Under those declarations, the operational expression is:

$$
I_{\max} = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}}

$$

This is an area-law diagnostic under the chosen convention. It becomes a proved bound only if the stated channel, field-equation, dimensional, and boundary hypotheses establish it.
:::

We now state the operational capacity convention used by the diagnostic.

:::{prf:definition} Conditional Causal Information Capacity
:label: thm-causal-information-bound

Under an explicit capacity permit that identifies stable representational information with
boundary area at resolution $\ell_L$, define the operational capacity

$$
I_{\max}:=\nu_D\,\frac{\operatorname{Area}(\partial\mathcal Z)}{\ell_L^{D-1}}.
$$
Here $\nu_D$ is the dimension-dependent coefficient defined above and the boundary area is
computed in the selected induced metric. This formula is a modeling convention/diagnostic
normalization; the current metric law does not by itself prove the bulk-to-boundary identity,
the spherical saturation solution, or the Fisher normalization used in the former derivation.

For the $D=2$ normalized convention, $\nu_2=1/4$ and the formula reads
$I_{\max}=\operatorname{Area}(\partial\mathcal Z)/(4\ell_L)$.
Any use of this expression as a theorem must state the additional field equation, boundary
conditions, and dimensional normalization that establish the permit.
:::





(sec-causal-stasis)=
## Causal Stasis

:::{div} feynman-prose
Now we come to the most striking conditional consequence: a selected radial update can slow near a singular radial metric.

The black-hole picture gives useful intuition about a horizon, but it is only an analogy. Under the formal proposition's radial ansatz, bounded radial force, selected horizon, and overdamped drift, one obtains

$$
v^r=-G^{rr}\partial_r\Phi_{\mathrm{eff}}\longrightarrow0.
$$

That is a statement about the radial component at that horizon. It does not imply $\|v\|_G\to0$ when angular components remain, does not freeze the interior, and does not follow from $I_{\text{bulk}}\to I_{\max}$ until a coupling between those quantities has been proved or assumed.

If an implementation shows a slowdown, treat it as a diagnostic to investigate: check the metric component, force bounds, momentum or overdamped regime, and the capacity definition. Pruning concepts or expanding the interface are possible interventions, not consequences that the formal ansatz has already proved.
:::

We record the conditional consequence of the ansatz: a radial update can slow at the selected horizon.

:::{prf:proposition} Conditional Radial Causal Stasis
:label: thm-causal-stasis

Assume the conditional capacity formula above, the formal spherical ansatz, bounded radial force,
and $G^{rr}\to0$ at the selected horizon. Then the radial component of an overdamped drift
satisfies

$$
v^r=-G^{rr}\partial_r\Phi_{\mathrm{eff}}\longrightarrow0.
$$
This conclusion controls the radial component in that ansatz. It does not imply
$\|v\|_G\to0$ for the full tensor, nor does it follow from $I_{\mathrm{bulk}}\to I_{\max}$
without the additional hypotheses.
:::


:::{prf:remark} Formal Saturation-Velocity Scaling
:label: cor-saturation-velocity-tradeoff

Let $\eta_{\text{Sch}} := I_{\text{bulk}}/I_{\max}$ be the saturation ratio. If the model additionally identifies
$\eta_{\text{Sch}}=\mu/\mu_{\max}$ at fixed horizon radius, the radial update scales as:

$$
|v^r| \sim (1 - \eta_{\text{Sch}})^{1/2}.

$$
*Scope.* This square-root scaling is a consequence only of the formal radial ansatz and a
specific relation between the saturation ratio and the radial denominator; it is not established
for a general learned metric.

*Former proof sketch.* If one additionally assumes $\eta_{\text{Sch}}=\mu/\mu_{\max}$ and a linear radial denominator,
then $G^{rr}\sim1-\eta_{\text{Sch}}$ and the displayed square-root scaling follows for the selected radial component. This
identification is a modeling assumption, not a consequence of the capacity definition.

At 90% saturation ($\eta_{\text{Sch}} = 0.9$), the radial component is $\sim 32\%$ of its
reference value; at 99% it is $\sim 10\%$. These percentages do not describe angular motion or a general learned metric.

:::



(sec-diagnostic-node-56)=
## Diagnostic Node 56: CapacityHorizonCheck

:::{div} feynman-prose
How do you know if you're approaching the bound? You need a warning light.

That's what Diagnostic Node 56 is meant to provide: a warning light for the declared saturation ratio $\eta_{\text{Sch}}$. Treat it like a fuel gauge whose calibration must first be checked. If the bulk estimate, area normalization, and resolution do not match, the number is only a heuristic.

The 50%, 90%, and 99% values are engineering setpoints. They can organize monitoring and trigger a study of utilization trends, but they are not universal safe, warning, or critical probabilities. In particular, 90% does not prove degraded velocity and 99% does not prove imminent Causal Stasis without the radial and coupling hypotheses above.

The subscript "Sch" records the Schwarzschild analogy. It does not make $\eta_{\text{Sch}}=1$ a horizon or establish a singularity in the learned metric.
:::

Following the diagnostic node convention ({ref}`sec-diagnostics-stability-checks`), we define a monitor for proximity to the Causal Information Bound.

(node-56)=
**Node 56: CapacityHorizonCheck**

| **#**  | **Name**                 | **Component** | **Type**   | **Interpretation** | **Proxy**                                                                                 | **Cost** |
|--------|--------------------------|---------------|------------|--------------------|-------------------------------------------------------------------------------------------|----------|
| **56** | **CapacityHorizonCheck** | Memory        | Saturation | Is capacity safe?  | $\eta_{\text{Sch}} := I_{\text{bulk}} / I_{\max}$ | $O(B)$   |

:::{prf:definition} Capacity Horizon Diagnostic
:label: def-capacity-horizon-diagnostic

Compute the **Saturation Ratio**:

$$
\eta_{\text{Sch}}(s) := \frac{I_{\text{bulk}}(s)}{I_{\max}} = \frac{I_{\text{bulk}}(s)}{\nu_D \cdot \text{Area}(\partial\mathcal{Z}) / \ell_L^{D-1}},

$$
where:
- $I_{\text{bulk}}(s) = \int_{\mathcal{Z}} \iota_{\mathrm{bulk}}(z,s) \, d\mu_G$ per Definition {prf:ref}`def-a-bulk-information-volume`; any empirical proxy must be calibrated to this quantity
- $\nu_D$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`)
- $D$ is the latent manifold dimension

*Special case (Poincare disk, $D=2$):* $\eta_{\text{Sch}} = 4\ell_L \cdot I_{\text{bulk}} / \text{Area}(\partial\mathcal{Z})$.

*Interpretation:*
- $\eta_{\text{Sch}} < 0.5$: Safe operating regime. Ample capacity headroom.
- $0.5 \le \eta_{\text{Sch}} < 0.9$: Elevated utilization. Monitor for growth trends.
- $0.9 \le \eta_{\text{Sch}} < 0.99$: **Warning setpoint.** Test the radial-stasis hypotheses and monitor the measured update components.
- $\eta_{\text{Sch}} \ge 0.99$: **Critical setpoint.** Investigate the radial-stasis hypotheses and consider a
  conservative remediation; this threshold does not prove that stasis is imminent.

*Cross-reference:* Complements the metric-law CapacitySaturationCheck ({ref}`sec-diagnostic-node-capacity-saturation`)
by providing the velocity-degradation interpretation and connecting to ontological remediation.
:::

**Trigger Conditions:**
- **$\eta_{\text{Sch}} > 0.9$:** Near-saturation setpoint. Test the radial and
  information-to-risk hypotheses before considering **Ontological Fusion**
  ({ref}`sec-ontological-fusion-concept-consolidation`) to prune the
  macro-register $\mathcal{K}$.
- **Velocity drop detected:** If a measured radial component decreases while
  $\eta_{\text{Sch}}$ increases, record the association and test the stated
  metric, force, and overdamped hypotheses; the correlation alone does not
  establish causation.
- **Persistent high $\eta_{\text{Sch}}$ after fusion:** The interface capacity
  $C_\partial$ may be the bottleneck. Consider hardware/bandwidth scaling
  after checking the estimator and units.

**Remediation:**
1. **Ontological Fusion** ({ref}`sec-ontological-fusion-concept-consolidation`): Merge redundant charts to reduce $I_{\text{bulk}}$.
2. **Chart Pruning**: Remove charts whose measured utility fails the
   codebook-liveness criterion ({prf:ref}`node-codebook-liveness-check`).
3. **Interface Expansion**: Increase boundary bandwidth (sensor resolution, communication channels).
4. **Depth Reduction**: Decrease TopoEncoder depth to reduce latent dimensionality.

**Operational computational estimator.** Estimate the bulk information from
the empirical joint law of the macro register and nuisance coordinates, then
divide by the declared area-law capacity:

$$
\widehat I_{\text{bulk}}
:= \widehat H(K)+\sum_k \widehat P(K=k)\,\widehat H(z_n\mid K=k),
\qquad
\widehat I_{\max}
:= \nu_D\,\frac{\widehat{\operatorname{Area}}(\partial\mathcal Z)}
                 {\ell_L^{D-1}},
\qquad
\widehat\eta_{\text{Sch}}:=
\frac{\widehat I_{\text{bulk}}}{\widehat I_{\max}},

$$
Here $\widehat H(K)$ and the conditional entropies are computed from the
same batch or EMA, and the area estimate uses the same induced metric and
resolution convention as $I_{\max}$. This estimator is distinct from the
capacity ratio $I_{\text{bulk}}/C_\partial$ unless the declared Levin-length
normalization identifies the two.



(sec-summary-geometry-bounded-intelligence)=
## Summary: The Geometry of Bounded Intelligence

:::{div} feynman-prose
Let me step back and say precisely what we have learned here.

With an explicit capacity convention, interface area and resolution give a measurable capacity diagnostic. With an additional spherical metric ansatz and an information-to-risk coupling, a radial inverse metric can provide a conditional stasis signal. Those are useful hypotheses to test; they are not a universal area law or a theorem that all over-parameterized agents become paralyzed.

There is still no free lunch in representation: interface bandwidth, resolution, data, and compute all constrain what can be learned. If an agent slows or learning plateaus, the saturation ratio is one diagnostic among several. Check its units and estimator first, then test whether the metric and dynamics satisfy the hypotheses before choosing fusion or interface expansion.

The physical analogies help us remember the structure. The mathematics tells us exactly where the analogy stops: at the declared capacity permit, the selected geometry, and the estimates that connect them.
:::

**Table 33.6.1 (Causal Information Bound Summary).**

| Concept                      | Definition/Reference                                                                                  | Units         | Diagnostic |
|:-----------------------------|:------------------------------------------------------------------------------------------------------|:--------------|:-----------|
| **Holographic Coefficient**  | $\nu_D = (D-1)\Omega_{D-1}/(8\pi)$ (Def {prf:ref}`def-holographic-coefficient`)                       | dimensionless | —          |
| **Levin Length**             | $\ell_L = (\nu_D\eta_\ell)^{1/(D-1)}$ (Def {prf:ref}`def-levin-length`)                              | $[z]$         | —          |
| **Saturation Limit**         | $I_{\text{bulk}} = C_\partial$ (Def {prf:ref}`def-saturation-limit`)                                  | nat           | Capacity check |
| **Causal Information Bound** | $I_{\max} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$ (Def {prf:ref}`thm-causal-information-bound`) | nat           | —          |
| **Saturation Ratio**         | $\eta_{\text{Sch}} = I_{\text{bulk}}/I_{\max}$ (Def {prf:ref}`def-capacity-horizon-diagnostic`)       | dimensionless | Node 56    |
| **Causal Stasis**            | $v^r \to 0$ at the selected horizon under the radial ansatz (Prop {prf:ref}`thm-causal-stasis`)             | —             | Node 56    |

**Key Results:**

1. **The Holographic Coefficient** (Definition {prf:ref}`def-holographic-coefficient`) determines how efficiently information can be stored on a boundary of dimension $D$. For $D=2$: $\nu_2 = 1/4$. For $D=3$: $\nu_3 = 1$.

2. **The Levin Length** (Definition {prf:ref}`def-levin-length`) sets the minimal scale of representational distinction. One nat of information occupies $(D-1)$-dimensional volume $\ell_L^{D-1}$.

3. **The Causal Information Capacity** (Definition {prf:ref}`thm-causal-information-bound`) defines an operational
   area-normalized capacity: $I_{\max} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$. For the
   Poincare disk ($D=2$): $I_{\max} = \text{Area}/(4\ell_L)$ under the chosen normalization.

4. **Causal Stasis** (Proposition {prf:ref}`thm-causal-stasis`) controls only the radial component at the selected
   horizon, under the spherical ansatz and bounded-force hypotheses.

5. **Remediation options** include reducing bulk information (Ontological Fusion), expanding the interface, or
   changing the representation. Their effectiveness is an engineering question outside the conditional statements above.

**Conclusion.** Under the declared capacity permit, the expression gives an
auditable interface-normalized capacity for an agent whose internal state is
grounded through that interface. It is a modeling convention and diagnostic,
not a universal bound on intelligence. Whether an implementation approaches
the diagnostic depends on the chosen estimator, representation, boundary
channel, and the hypotheses of the conditional radial result.



(sec-unified-notation-table-and-cross-section-connectivity)=
## Unified Notation Table and Cross-Section Connectivity

This section provides a consolidated reference for the key symbols introduced across Sections 17-32.

(sec-core-symbols)=
## Core Symbols (Sections 17-32)

| Symbol                         | Name                            | Definition                                                                                                         | Units             | Section        |
|--------------------------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------|-------------------|----------------|
| $G_{ij}(z)$                    | Latent metric tensor            | Capacity-constrained Riemannian metric                                                                             | $[z]^{-2}$        | 2.5, 18.2      |
| $\Gamma^k_{ij}$                | Christoffel symbols             | Levi-Civita connection of $G$                                                                                      | $[z]^{-1}$        | 2.5.1, 22.2.1a |
| $\iota_{\mathrm{bulk}}(z,t)$  | Bulk information density       | Relative-information density used in Definition {prf:ref}`def-a-bulk-information-volume`                           | nat$/[z]^n$       | 18.1.2         |
| $C_\partial$                   | Boundary capacity               | Area-law capacity of interface                                                                                     | nat               | 18.1.3         |
| $\nu_{\text{cap}}$             | Capacity saturation             | $I_{\text{bulk}}/C_\partial$                                                                                       | dimensionless     | 18.3.1         |
| $\lambda$                      | WFR length-scale                | Transport-vs-reaction crossover                                                                                    | $[z]$             | 20.2.1, 20.3.1 |
| $\kappa_{\text{screen}}$       | Screening mass                  | $\sqrt{(-\ln\gamma)/(T_c\Delta t)}$                                                                                  | $[z]^{-1}$        | 24.2.4         |
| $\kappa_{\text{metric}}$       | Metric-law coupling             | Risk-tensor coupling in the capacity-constrained metric law                                                            | model-dependent   | 18.2           |
| $\ell_{\text{screen}}$         | Screening length                | $1/\kappa_{\text{screen}}$; reward correlation length                                                               | $[z]$             | 24.2.4         |
| $U(z)$                         | Hyperbolic potential            | $-2\operatorname{artanh}(\lvert z\rvert)$                                                                          | nat               | 21.1.4         |
| $V(z)$                         | Value/Critic                    | Solution to Helmholtz equation (conservative case)                                                                 | nat               | 2.7, 24.3      |
| $\mathcal{R}$                  | Reward 1-form                   | General reward field; $r_t = \mathcal{R}[v]$                                                                       | nat$/[z]$         | 24.1           |
| $\mathcal{F}$                  | Value Curl                      | $d\mathcal{R}$; measures non-conservative structure                                                                | nat$/[z]^2$       | 24.2           |
| $\Phi$                         | Scalar Potential                | Hodge gradient component of $\mathcal{R}$                                                                          | nat               | 24.2           |
| $\Psi$                         | Vector Potential                | Hodge solenoidal component of $\mathcal{R}$                                                                        | nat$\cdot[z]^2$   | 24.2           |
| $\eta_{\mathrm{harm}}$         | Harmonic Flux                   | Hodge harmonic component of $\mathcal{R}$                                                                            | nat$/[z]$         | 24.2           |
| $\mathbf{A}$                   | Vector Potential (WFR)          | $d\mathbf{A} = \mathcal{F}$; appears in generalized WFR action                                                     | nat$/[z]$         | 20.2           |
| $\beta_{\text{curl}}$          | Curl Coupling                   | Lorentz force strength                                                                                             | dimensionless     | 22.2           |
| $J$                            | Probability Current             | $\rho v - D\nabla\rho$; non-zero in NESS                                                                           | $1/\text{step}$   | 24.4           |
| $\Phi_{\text{eff}}$            | Effective potential             | $\alpha U + (1-\alpha)\Phi + \gamma_{\text{risk}}\Psi_{\text{risk}}$                                               | nat               | 22.3.1         |
| $u_\pi$                        | Control field                   | Policy-induced tangent vector                                                                                      | $[z]/\text{step}$ | 21.2.2         |
| $T_c$                          | Cognitive temperature           | Exploration parameter                                                                                              | nat               | 22.4           |
| $\Omega(z)$                    | Conformal factor                | $1 + \alpha_{\text{conf}}\lVert\nabla^2 V\rVert$                                                                   | dimensionless     | 24.4.1         |
| $\omega$                       | Symplectic form                 | $\sum_i dq^i \wedge dp_i$                                                                                          | nat               | 23.1.1         |
| $\mathcal{L}$                  | Legendre transform              | $T\mathcal{Q} \to T^*\mathcal{Q}$; $p = G\dot{q}$                                                                  | —                 | 23.2.3         |
| $\mathcal{M}_\Theta$           | Parameter manifold              | Space of agent parameters                                                                                          | —                 | 26.2           |
| $\Psi$                         | Constraint evaluation map       | $\theta \mapsto [C_1(\theta), \ldots, C_K(\theta)]$                                                                | —                 | 26.3           |
| $\pi_{\mathfrak{G}}$           | Governor policy                 | $s_{t:t-H} \mapsto \Lambda_t$                                                                                      | —                 | 26.3           |
| $V_{\mathfrak{L}}$             | Training Lyapunov               | $\mathcal{L} + \sum_k \frac{\mu_k}{2}\max(0,C_k)^2$                                                                | nat               | 26.5           |
| $\gamma_{\text{viol}}$         | Violation penalty               | Constraint violation weight                                                                                        | dimensionless     | 26.4           |
| $\Lambda_t$                    | Control vector                  | $(\eta_t, \vec{\lambda}_t, T_{c,t})$                                                                               | mixed             | 26.3           |
| $\Xi_T$                        | Memory screen                   | $\int_0^T \alpha(t') \delta_{\gamma(t')} dt'$                                                                      | nat               | 27.1.2         |
| $H_\tau(z, z')$                | Heat kernel                     | Memory kernel (fundamental soln to heat eqn)                                                                       | $[z]^{-d}$        | 27.2.1         |
| $\tau$                         | Diffusion time                  | Memory smoothing scale                                                                                             | $[z]^2$           | 27.2.1         |
| $\Psi_{\text{mem}}$            | Memory potential                | $-\int H_\tau(z, z') d\Xi_T(z')$                                                                                   | nat               | 27.2.2         |
| $\Omega_{\text{mem}}$          | Non-locality ratio              | $\lVert\nabla_G \Psi_{\text{mem}}\rVert_G / \lVert\nabla_G \Phi_{\text{eff}}\rVert_G$                              | dimensionless     | 27.5.1         |
| $\mathcal{Z}^{(N)}$            | N-agent product manifold        | $\prod_{i=1}^N \mathcal{Z}^{(i)}$                                                                                  | $[z]$             | 29.1           |
| $\mathcal{B}_{ij}$             | Bridge manifold                 | Interaction submanifold between agents $i,j$                                                                       | $[z]$             | 29.2           |
| $\Phi_{ij}$                    | Strategic potential             | Interaction kernel from agent $j$                                                                                  | nat               | 29.3           |
| $\mathcal{G}_{ij}^{kl}$        | Game Tensor                     | $\partial^2 V^{(i)} / \partial z^{(j)}_k \partial z^{(j)}_l$                                                       | nat$/[z]^2$       | 29.4           |
| $\tilde{G}^{(i)}$              | Game-augmented metric           | $G^{(i)} + \alpha_{\text{adv}} \mathcal{G}_{ij}$                                                                   | $[z]^{-2}$        | 29.4           |
| $\epsilon_{\text{Nash}}$       | Nash residual                   | Max gradient deviation from equilibrium                                                                            | nat$/[z]$         | 29.6           |
| $\emptyset$                    | Semantic Vacuum                 | Fiber over origin $z=0$; maximal $SO(D)$ symmetry                                                                  | —                 | 30.1           |
| $\Xi$                          | Ontological Stress              | $I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, K^{\text{act}}_t)$                                                   | nat               | 30.2           |
| $\Xi_{\text{crit}}$            | Fission threshold               | Critical stress for chart bifurcation                                                                              | nat               | 30.3           |
| $\mathcal{L}_{\text{center}}$  | Centering loss                  | $\lVert\sum q_i\rVert^2 + \sum\lVert\sum e_{i,c}\rVert^2$                                                          | $[z]^2$           | 30.1           |
| $\mathcal{L}_{\text{Ricci}}$   | Ricci flow loss                 | $\lVert R_{ij} - \frac{1}{2}RG_{ij} + \Lambda G_{ij} - \kappa T_{ij}\rVert_F^2 + \nu^2\lVert\nabla^2\Xi\rVert_F^2$ | $[z]^{-4}$        | 30.5           |
| $\Upsilon_{ij}$                | Ontological Redundancy          | $\exp(-[d_{\text{WFR}} + D_{\mathrm{KL}} + \lVert V_i - V_j\rVert^2])$                                             | dimensionless     | 30.8           |
| $G_\Delta$                     | Discrimination Gain             | $I(X; \{K_i, K_j\}) - I(X; K_{i \cup j})$                                                                          | nat               | 30.8           |
| $\Upsilon_{\text{crit}}$       | Fusion threshold                | Critical redundancy for chart merger                                                                               | dimensionless     | 30.9           |
| $\epsilon_{\text{hysteresis}}$ | Hysteresis constant             | Fission/Fusion asymmetry term                                                                                      | nat               | 30.9           |
| $\sigma_k^2$                   | Intra-Symbol Variance           | $\mathbb{E}[\lVert z_e - e_k\rVert^2 \mid K=k]$                                                                    | $[z]^2$           | 30.12          |
| $\mathcal{D}_f$                | Functional Indistinguishability | $D_{\mathrm{KL}}(\pi_1 \lVert \pi_2) + \lVert V_1 - V_2\rVert$                                                     | nat               | 30.12          |
| $\mathcal{V}_k$                | Voronoi cell                    | $\{z : d_G(z, e_k) \le d_G(z, e_j)\}$                                                                              | —                 | 30.12          |
| $\mathcal{D}_k$                | Local Distortion                | $\int_{\mathcal{V}_k} d_G(z, e_k)^2 p(z) d\mu_G$                                                                   | $[z]^2$           | 30.12          |
| $U_k$                          | Symbol Utility                  | $P(k) \cdot I(K=k; A) + P(k) \cdot I(K=k; K_{t+1})$                                                                | nat               | 30.12          |
| $\dot{\mathcal{M}}(s)$         | Metabolic flux                  | WFR action rate (transport + reaction cost)                                                                        | nat/step          | 31.1           |
| $\Psi_{\text{met}}(s)$         | Metabolic potential             | Cumulative dissipation $\int_0^s \dot{\mathcal{M}} \, du$                                                          | nat               | 31.2           |
| $\mathcal{S}_{\text{delib}}$   | Deliberation action             | $-\langle V \rangle_{\rho_S} + \Psi_{\text{met}}(S)$                                                               | nat               | 31.2           |
| $S^*$                          | Optimal computation budget      | Deliberation stopping time                                                                                         | step              | 31.3           |
| $\Gamma(s)$                    | Value-Improvement Rate          | $\lVert d\langle V \rangle/ds\rVert$                                                                               | nat/step          | 31.3           |
| $\sigma_{\text{tot}}$          | Total entropy production        | $\dot{H} + \dot{\mathcal{M}}/T_c \ge 0$                                                                            | nat/step          | 31.4           |
| $\eta_{\text{thought}}$        | Efficiency of thought           | $-T_c \dot{H}/\dot{\mathcal{M}} \le 1$                                                                             | dimensionless     | 31.4           |
| $\mathfrak{I}$                 | Interventional operator         | Pearl's $do(\cdot)$ surgery                                                                                        | —                 | 32.1           |
| $\Psi_{\text{causal}}$         | Causal information potential    | EIG for transition parameters                                                                                      | nat               | 32.2           |
| $\Delta_{\text{causal}}$       | Causal deficit                  | $D_{\text{KL}}(P_{\text{int}} \lVert P_{\text{obs}})$                                                              | nat               | 32.2           |
| $\mathbf{f}_{\text{exp}}$      | Curiosity force                 | $G^{-1}\nabla\Psi_{\text{causal}}$                                                                                 | $[z]$/step        | 32.3           |
| $\beta_{\text{exp}}$           | Exploration coefficient         | Curiosity vs. exploitation balance                                                                                 | dimensionless     | 32.3           |
| $\nu_D$                        | Holographic Coefficient         | $(D-1)\Omega_{D-1}/(8\pi)$; dim-dependent capacity factor                                                          | dimensionless     | 33.0           |
| $\ell_L$                       | Levin Length                    | $(\nu_D\eta_\ell)^{1/(D-1)}$; minimal distinction scale                                                            | $[z]$             | 33.1           |
| $I_{\max}$                     | Causal Information Bound        | $\nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$                                                        | nat               | 33.3           |
| $\eta_{\text{Sch}}$            | Saturation Ratio                | $I_{\text{bulk}}/I_{\max}$                                                                                         | dimensionless     | 33.5           |
| $r_h$                          | Horizon radius                  | Critical radius where $G_{rr} \to \infty$                                                                          | $[z]$             | 33.2           |

(sec-boundary-conditions)=
## Boundary Conditions ({ref}`sec-the-boundary-interface-symplectic-structure`)

| Type      | Symbol                                               | Interpretation              | Physics             |
|-----------|------------------------------------------------------|-----------------------------|---------------------|
| Dirichlet | $\rho\lvert_{\partial} = \delta(q - q_{\text{obs}})$ | Position clamped by sensors | Environment → Agent |
| Neumann   | $\nabla_n\rho = j_{\text{motor}}$                    | Flux clamped by motors      | Agent → Environment |
| Source    | $J_r$                                                | Reward flux on boundary     | Reward signal       |

(sec-cross-section-connectivity-map)=
## Cross-Section Connectivity Map (Sections 17-32)

```
{ref}`sec-summary-unified-information-theoretic-control-view` (Summary)
     |
     v
{ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` (Capacity Law) ─────────────────────────────────────────┐
     |                                                              |
     | $\iota_{\mathrm{bulk}}$, $C_\partial$, $T_{ij}$              |
     v                                                              |
Section 19 (Conclusion) ←───────────────────────────────────────────────────────┐  |
     |                                                           |  |
     v                                                           |  |
{ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` (WFR Geometry) ──────────────────────────────────┐   |  |
     |                                                       |   |  |
     | $\lambda$, $(v, r)$, WFR metric                       |   |  |
     v                                                       |   |  |
{ref}`sec-radial-generation-entropic-drift-and-policy-control` (Holographic Generation {cite}`thooft1993holographic,susskind1995world`) ────────────────┐       |   |  |
     |                                               |       |   |  |
     | $U(z)$, $u_\pi$, SO(D) breaking              |       |   |  |
     v                                               v       v   |  |
{ref}`sec-the-equations-of-motion-geodesic-jump-diffusion` (Equations of Motion) ←──────────────────┴───────┴───┘  |
     |                                                              |
     | $\Phi_{\text{eff}}$, geodesic SDE, BAOAB                    |
     v                                                              |
{ref}`sec-the-boundary-interface-symplectic-structure` (Holographic Interface) ←────────────────────────────────┤
     |                                                              |
     | Symplectic structure, Legendre transform, $(q, p)$          |
     v                                                              |
{ref}`sec-the-reward-field-value-forms-and-hodge-geometry` (Scalar Field) ←─────────────────────────────────────────┘
     |
     | $V$ as Helmholtz solution, conformal coupling $\Omega$
     v
{ref}`sec-supervised-topology-semantic-potentials-and-metric-segmentation` (Supervised Topology)
     |
     | Classification as geodesic relaxation
     v
{ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller` (Meta-Stability) ←─────────────────────── {ref}`sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration`
     |                                               (Adaptive Multipliers)
     | $\pi_{\mathfrak{G}}$, $V_{\mathfrak{L}}$, bilevel optimization
     v
{ref}`sec-section-non-local-memory-as-self-interaction-functional` (Non-Local Memory) ←──────────────────── {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`, 22, 24
     |                                               (WFR, EoM, Scalar Field)
     | $\Xi_T$, $H_\tau$, $\Psi_{\text{mem}}$, $\Omega_{\text{mem}}$
     v
{ref}`sec-section-hyperbolic-active-retrieval-geodesic-search-and-semantic-pull-back` (Hyperbolic Retrieval) ←────────────────── {ref}`sec-radial-generation-entropic-drift-and-policy-control` (Poincare metric)
     |                                               {ref}`sec-section-non-local-memory-as-self-interaction-functional` (Memory potential)
     | $\Phi_{\text{ret}}$, Geodesic search, WFR sources
     v
{ref}`sec-symplectic-multi-agent-field-theory` (Multi-Agent SMFT) ←──────────────────── {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits`, 21, 23
     |                                               (Metric, Symplectic, Capacity)
     | $\mathcal{G}_{ij}$, Strategic potential, Nash equilibrium
     v
{ref}`sec-ontological-expansion-topological-fission-and-the-semantic-vacuum` (Ontological Expansion) ←───────────────── {ref}`sec-radial-generation-entropic-drift-and-policy-control` (Pitchfork bifurcation)
     |                                               {ref}`sec-tier-the-attentive-atlas` (Attentive Atlas)
     | $\Xi$, $\emptyset$, Query Fission, Ricci flow   {ref}`sec-main-result` (Metric law)
     v
Appendices (Derivations, Units, WFR Tensor)
```

(sec-diagnostic-node-registry)=
## Diagnostic Node Registry (Complete)

| #  | Name                                              | Section | Key Formula                                                                                                      |
|----|---------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------|
| 1  | [CostBoundCheck](#sec-the-stability-checks)       | 3.5     | $\max(0, V(z) - V_{\text{max}})^2$                                                                               |
| 2  | [ZenoCheck](#sec-the-stability-checks)            | 3.5     | $D_{\mathrm{KL}}(\pi_t \Vert \pi_{t-1})$                                                                         |
| 3  | [CompactCheck](#sec-the-stability-checks)         | 3.5     | $H(q(K \mid x))$                                                                                                 |
| 4  | [ScaleCheck](#sec-the-stability-checks)           | 3.5     | $\lVert \nabla \theta \rVert / \lVert \Delta S \rVert$                                                           |
| 5  | [ParamCheck](#sec-the-stability-checks)           | 3.5     | $\lVert \nabla_t S_t \rVert^2$                                                                                   |
| 6  | [GeomCheck](#sec-the-stability-checks)            | 3.5     | $\mathcal{L}_{\text{contrastive}}$ (InfoNCE)                                                                     |
| 7  | [StiffnessCheck](#sec-the-stability-checks)       | 3.5     | $\max(0, \epsilon - \lVert \nabla_A V \rVert)$                                                                     |
| 8  | [TopoCheck](#sec-the-stability-checks)            | 3.5     | $T_{\text{reach}}(z_{\text{goal}})$                                                                              |
| 9  | [TameCheck](#sec-the-stability-checks)            | 3.5     | $\lVert \nabla^2 S_t \rVert$                                                                                     |
| 10 | [ErgoCheck](#sec-the-stability-checks)            | 3.5     | $-H(\pi)$                                                                                                        |
| 11 | [ComplexCheck](#sec-the-stability-checks)         | 3.5     | $H(K)/\log\lvert\mathcal{K}\rvert$                                                                               |
| 12 | [OscillateCheck](#sec-the-stability-checks)       | 3.5     | $\lVert z_t - z_{t-2} \rVert$                                                                                    |
| 13 | [BoundaryCheck](#sec-the-stability-checks)        | 3.5     | $I(X;K)$                                                                                                         |
| 14 | [InputSaturationCheck](#sec-the-stability-checks) | 3.5     | $\mathbb{I}(\lvert x \rvert > x_{\text{max}})$                                                                   |
| 15 | [SNRCheck](#sec-the-stability-checks)             | 3.5     | $\text{SNR} < \epsilon$                                                                                          |
| 16 | [AlignCheck](#sec-the-stability-checks)           | 3.5     | $\lvert V_{\text{proxy}} - V_{\text{true}} \rvert$                                                               |
| 17 | [Lock](#sec-the-stability-checks)                 | 3.5     | $\mathbb{I}(\text{Unsafe}) \cdot \infty$                                                                         |
| 18 | [SymmetryCheck](#sec-the-stability-checks)        | 3.5     | $\mathbb{E}_{g\sim G}[D_{\mathrm{KL}}(q(K\mid x)\Vert q(K\mid g\cdot x))]$                                       |
| 19 | [DisentanglementCheck](#sec-the-stability-checks) | 3.5     | $\lVert\mathrm{Cov}(z_{\text{macro}},z_n)\rVert_F^2$                                                             |
| 20 | [LipschitzCheck](#sec-the-stability-checks)       | 3.5     | $\max_\ell \sigma(W_\ell)$                                                                                       |
| 21 | [SymplecticCheck](#sec-the-stability-checks)      | 3.5     | $\lVert J_S^\top J J_S - J\rVert_F^2$                                                                            |
| 22 | [MECCheck](#sec-the-stability-checks)             | 3.5     | $\lVert(\widetilde\varrho_{t+1}-\varrho_t)/\Delta t - \mathcal{L}_{\text{GKSL}}(\varrho_t)\rVert_F^2$                      |
| 23 | [NEPCheck](#sec-the-stability-checks)             | 3.5     | $\mathrm{ReLU}(D_{\mathrm{KL}}(p_{t+1}\Vert \widetilde p_{t+1})-\widehat I_{t+1})^2$                                                  |
| 24 | [QSLCheck](#sec-the-stability-checks)             | 3.5     | $\mathrm{ReLU}(d_G(z_{t+1},z_t)-v_{\max})^2$                                                                     |
| 25 | [HoloGenCheck](#node-25)                          | 21.4    | $\mathbf{1}(\lVert z\rVert \geq R_{\text{cutoff}})$                                                              |
| 26 | [GeodesicCheck](#node-26)                         | 22.6    | $\lVert\ddot{z}+\gamma\dot z+\Gamma(\dot z,\dot z)+G^{-1}\nabla\Phi_{\mathrm{eff}}-\gamma u_\pi-\beta_{\mathrm{curl}}G^{-1}\mathcal{F}\dot z\rVert_G$ |
| 27 | [OverdampedCheck](#node-27)                       | 22.6    | $\chi_{\mathrm{in}}:=m\lVert\ddot z\rVert_G/(\gamma\lVert\dot z\rVert_G+m\lVert\ddot z\rVert_G+\varepsilon)$ |
| 28 | [JumpConsistencyCheck](#node-28)                  | 22.6    | $\lVert m_{\text{pre}} - m_{\text{post}}\eta\rVert$                                                              |
| 29 | [TextureFirewallCheck](#node-29)                  | 22.6    | $\lVert\partial_{z_{\text{tex}}} \dot{z}\rVert$                                                                  |
| 30 | [SymplecticBoundaryCheck](#node-30)               | 23.8    | $\lVert E_\phi(x) - q_{\text{clamp}}\rVert_G$                                                                    |
| 31 | [DualAtlasConsistencyCheck](#node-31)             | 23.8    | $\lVert D_A(E_A(a)) - a\rVert$                                                                                   |
| 32 | [MotorTextureCheck](#node-32)                     | 23.8    | $H(z_{\text{tex,motor}} \mid A, z_{n,\text{motor}})$                                                             |
| 33 | [ThermoCycleCheck](#node-33)                      | 23.8    | $\lVert\Delta S_{\text{cycle}}\rVert$                                                                            |
| 34 | [ContextGroundingCheck](#node-34)                 | 23.8    | $I(c; z)$                                                                                                        |
| 35 | [HelmholtzResidualCheck](#node-35)                | 24.7    | $\lVert-\Delta_G V + \kappa^2 V - \rho_r\rVert$                                                                  |
| 36 | [GreensFunctionDecayCheck](#node-36)              | 24.7    | $\lVert V(z) - V(z')\rVert \cdot e^{\kappa d_G(z,z')}$                                                           |
| 37 | [BoltzmannConsistencyCheck](#node-37)             | 24.7    | $D_{\mathrm{KL}}(P_{\text{empirical}} \lVert P_{\text{Boltzmann}})$                                              |
| 38 | [ConformalBackReactionCheck](#node-38)            | 24.7    | $\text{Var}(\Omega)$                                                                                             |
| 39 | [ValueMassCorrelationCheck](#node-39)             | 24.7    | $\text{corr}(m_t, V(z_t))$                                                                                       |
| 40 | [CapacitySaturationCheck](#sec-diagnostic-node-capacity-saturation) | 18.3 | $I_{\text{bulk}}/C_\partial$ |
| 41 | [SupervisedTopologyChecks](#node-41)              | 25.4    | (See {ref}`sec-the-supervised-topology-loss`)                                                                                               |
| 42 | [GovernorStabilityCheck](#node-42)                | 26.9    | $\Delta V_{\mathfrak{L}} = V_{\mathfrak{L}}(\theta_{t+1}) - V_{\mathfrak{L}}(\theta_t)$                          |
| 43 | [MemoryBalanceCheck](#node-43)                    | 27.5    | $\Omega_{\text{mem}} = \lVert\nabla_G\Psi_{\text{mem}}\rVert_G / \lVert\nabla_G\Phi_{\text{eff}}\rVert_G$        |
| 44 | [HyperbolicAlignmentCheck](#node-44)              | 28.6    | $\Delta_{\text{align}} := \mathbb{E}[\lVert d_{\mathbb{D}}^{\text{int}} - d_{\mathbb{D}}^{\text{ext}}\rVert]$    |
| 45 | [RetrievalFirewallCheck](#node-45)                | 28.6    | $\Gamma_{\text{leak}} := \lVert\nabla_{z_{\text{int}}} (\partial \pi / \partial z_{\text{tex,ext}})\rVert$       |
| 46 | [GameTensorCheck](#node-46)                       | 29.6    | $\lVert\mathcal{G}_{ij}\rVert_F$                                                                                 |
| 47 | [NashResidualCheck](#node-47)                     | 29.6    | $\epsilon_{\text{Nash}} := \max_i \lVert(G^{(i)})^{-1}\nabla \Phi_{\text{eff}}^{(i)}\rVert_{G^{(i)}}$            |
| 48 | [SymplecticBridgeCheck](#node-48)                 | 29.6    | $\Delta_\omega := \lVert\int_{\mathcal{B}_{ij}} \omega_{ij}(t) - \omega_{ij}(0)\rVert$                           |
| 49 | [OntologicalStressCheck](#node-49)                | 30.6    | $\Xi := I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, K^{\text{act}}_t)$                                          |
| 50 | [FissionReadinessCheck](#node-50)                 | 30.6    | $\mathbb{I}(\Xi > \Xi_{\text{crit}}) \cdot \mathbb{I}(\Delta V_{\text{proj}} > \mathcal{C}_{\text{complexity}})$ |
| 51 | [MetabolicEfficiencyCheck](#node-51)              | 31.5    | $\eta_{\text{ROI}} := \lvert\Delta\langle V\rangle\rvert / \Psi_{\text{met}}(S)$                                 |
| 52 | [EntropyProductionCheck](#node-52)                | 31.5    | $\sigma_{\text{tot}} := \dot{H} + \dot{\mathcal{M}}/T_c \ge 0$                                                   |
| 53 | [CausalEnclosureCheck](#node-53)                  | 32.6    | $\Delta_{\text{causal}} < \delta_{\text{causal}}$                                                                |
| 54 | {prf:ref}`node-fusion-readiness-check`                  | 30.11   | $\max_{i \neq j} \Upsilon_{ij} > \Upsilon_{\text{crit}}$                                                         |
| 55 | {prf:ref}`node-codebook-liveness-check`                 | 30.11   | $\min_k P(K=k) < \epsilon_{\text{dead}}$                                                                         |
| 56 | [CapacityHorizonCheck](#node-56)                  | 33.5    | $\eta_{\text{Sch}} := I_{\text{bulk}} / I_{\max}$                                                                |
| 57 | [CoherenceCheck](#node-57)                        | 34.1    | $\delta_{\text{coh}}$                                                                                              |
| 58 | [EntropyProductionCheck](#node-58)                | 34.1    | $\dot S_{\mathrm{vN}}$                                                                                             |
| 59 | [UncertaintyPrincipleCheck](#node-59)             | 34.1    | $\eta_{\mathrm{unc}}$                                                                                              |
| 60 | [TunnelingRateMonitor](#node-60)                 | 34.1    | $\Gamma_{\mathrm{tunnel}}$                                                                                        |
| 61 | [ValueCurlCheck](#node-61)                        | 24.8    | $\oint_\gamma \delta_{\text{TD}} \approx \int\lVert\nabla\times\mathcal{R}\rVert$                                |

Here $v := \dot{z}$ and $\mathcal{M}_\gamma^{-1} = \gamma I - \beta_{\text{curl}} G^{-1}\mathcal{F}$.
