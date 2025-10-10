
# The Standard Model on the Fractal Set (CST + IG)
*fragile series — formal style (definition–theorem–proof)*

> **Purpose.** We define the **Standard Model (SM)** — gauge group $\mathrm{SU}(3)_c\times\mathrm{SU}(2)_L\times \mathrm{U}(1)_Y$ with fermion generations and a complex Higgs doublet — **entirely on the discrete fractal set** $\mathcal F=T\cup G$ produced by the relativistic gas. We give:
> (i) gauge bundles and Wilson actions on irregular cycles;
> (ii) color‑ and electroweak‑covariant Dirac operators for all fermions (with Pauli exclusion via the fermionic layer);
> (iii) a discrete Higgs field, covariant kinetic term, potential and **Yukawa couplings**;
> (iv) **spontaneous symmetry breaking** (SSB) $\mathrm{SU}(2)_L\times \mathrm{U}(1)_Y\to \mathrm{U}(1)_{\rm em}$ with $W^\pm, Z, \gamma$;
> (v) **CKM/PMNS mixing**;
> (vi) anomaly cancellation and BRST‑independent gauge invariance;
> (vii) **consistency and convergence** theorems showing the discrete action and propagators converge to the continuum SM in the manifoldlike limit.

## 1. Discrete substrate and oriented edge set

:::{admonition} 🌌 The Physical Stage
:class: note
Think of this section as setting up the **stage** where all particle physics will play out. Instead of continuous spacetime, we're building physics on a discrete network that emerges from the underlying fractal dynamics. It's like replacing a smooth canvas with a pointillist painting—from far away it looks continuous, but up close you see the discrete structure.
:::

### 1.1 Episodes, CST, IG (recap)
Episodes $e\in\mathcal E$ arise as walker worldline segments (birth–death). The **CST** $T=(\mathcal E,\to)$ records genealogical causation with proper‑time edge weights; the **IG** $G=(\mathcal E,\sim)$ connects temporally overlapping, interacting episodes. As in the fermion/QCD layers, we work on the oriented 1‑skeleton

$$
\mathcal E^{\Rightarrow}=\{\,e\to e'\text{ (CST)}\,\}\;\cup\;\{\,\overrightarrow{e\,e'}\text{ (an orientation of each IG edge)}\,\}.
$$

:::{admonition} 💡 Arrows on the Network
:class: tip
Why do we need oriented edges? Think of it as giving each connection a direction—like one-way streets in a city. This orientation is crucial for defining how gauge fields "live" on the edges and how particles "hop" between nodes. The CST edges represent causal connections (parent-child), while IG edges represent interactions between contemporaries.
:::

### 1.2 Local frames and geometric weights

:::{admonition} Intuition: From Network to Spacetime
:class: dropdown
The key insight here is that our discrete network isn't just an abstract graph—it's embedded in actual spacetime! The local tetrads are like coordinate frames at each node, telling us "which way is up" in 4D spacetime. The area bivectors $\Sigma^{\mu\nu}$ measure the "size" of tiny loops in spacetime, crucial for computing field strengths.

Think of it as building a house of cards where each card knows its orientation relative to the table. The kernel-based volume estimates ensure that when we sum over many small pieces, we get the right answer—just like how many small rectangles can approximate the area under a curve.
:::

Faithful embedding and standardization yield local tetrads and area bivectors $\Sigma^{\mu\nu}(C)$ for fundamental cycles built from IG edges and CST paths. Kernel‑based volume estimates provide local 4‑volume elements used to **calibrate discrete sums into Riemann sums**.

## 2. Gauge bundles $\mathrm{SU}(3)_c\times\mathrm{SU}(2)_L\times \mathrm{U}(1)_Y$

:::{admonition} 🎭 The Three Forces of Nature
:class: important
:class: dropdown
The Standard Model describes three of the four fundamental forces through gauge symmetries:
- **SU(3)**: The strong force (color) that binds quarks into protons and neutrons
- **SU(2)**: The weak force responsible for radioactive decay
- **U(1)**: Related to electromagnetism (after symmetry breaking)

Think of these as three different types of "charges" that particles can carry, each with its own force field. The genius of gauge theory is that these forces arise naturally from demanding that physics looks the same under certain transformations—like how conservation of energy comes from time translation symmetry.
:::

### 2.1 Link variables and gauge transformations

:::{admonition} 🔗 What Are Link Variables?
:class: note
Imagine you're a particle sitting at node $e$. To hop to node $e'$, you need to know how your internal properties (color, weak charge, hypercharge) change during the journey. Link variables are like **rotation instructions** that tell you how to transform as you travel along an edge. Each gauge group has its own set of instructions:
- $U^{(3)}$: How your color rotates in color space
- $U^{(2)}$: How your weak isospin rotates
- $u^{(1)}$: How your phase shifts due to hypercharge
:::

:::{prf:definition} SM gauge bundle on the graph
:label: def-sm-gauge-bundle
:nonumber:

For each oriented edge $a=e\to e'\in\mathcal E^{\Rightarrow}$ assign link variables

$$
(U^{(3)}_a,\; U^{(2)}_a,\; u^{(1)}_a)\in \mathrm{SU}(3)\times\mathrm{SU}(2)\times \mathrm{U}(1).
$$

A node‑wise gauge transformation $g(e)=(g_3(e),g_2(e),e^{i\alpha_Y(e)})$ acts by

$$
\boxed{\;U^{(3)}_{e\to e'}\mapsto g_3(e)U^{(3)}_{e\to e'}g_3(e')^\dagger,\quad
U^{(2)}_{e\to e'}\mapsto g_2(e)U^{(2)}_{e\to e'}g_2(e')^\dagger,\quad
u^{(1)}_{e\to e'}\mapsto e^{i\alpha_Y(e)}\,u^{(1)}_{e\to e'}\,e^{-i\alpha_Y(e')}\;}
$$

:::{admonition} 💡 The Sandwich Formula
:class: tip
Notice the beautiful pattern: $U \mapsto gUg^\dagger$. This "sandwich" transformation ensures that when a particle hops from node to node and back, the total transformation is consistent. It's like saying: "rotate your perspective at the start, follow the original instructions, then rotate back at the end." This guarantees that closed loops give gauge-invariant results!
:::

and on matter fields $\psi_R(e)$ in representation $R$ with hypercharge $Y$,

$$
\psi_R(e)\mapsto g_3^{(\rho_c)}(e)\,g_2^{(\rho_w)}(e)\,e^{i Y\alpha_Y(e)}\,\psi_R(e).
$$
:::

Here $g^{(\rho)}$ denotes the representation matrix appropriate for $\rho$ (fundamental, singlet, etc.).

:::{admonition} 🎨 Gauge Symmetry Analogy
:class: hint
:class: dropdown
Gauge transformations are like choosing different coordinate systems at each point. Imagine painting a sphere—you can rotate your perspective at each point without changing the actual painting. Similarly, gauge transformations are "local rotations" in an abstract space of internal symmetries. The requirement that physics doesn't change under these rotations (gauge invariance) is what generates the forces!
:::

### 2.2 Fundamental cycles and Wilson actions

:::{admonition} 🔄 Why Cycles Matter
:class: tip
Cycles (closed loops) are special because they reveal the presence of gauge fields! When you parallel transport around a loop, you might not return to your original state—the amount of "twist" tells you about the field strength. It's like walking around a mountain: even if you stay at constant altitude, you've climbed due to the curvature. Wilson loops measure this "holonomy" around cycles.
:::

Let $T$ be a spanning tree; each oriented IG edge $\overrightarrow{u\,v}$ closes a **fundamental cycle**

$$
\boxed{\;C(\overrightarrow{u\,v})=\overrightarrow{u\,v}\circ P_T(v,u)\;}
$$

forming a cycle basis. For a loop $C$, define holonomies

:::{admonition} Intuition: The Memory of a Journey
:class: simple
A holonomy is the "memory" of a complete journey around a loop. Imagine carrying a gyroscope around a curved path—when you return, it might point in a different direction! The holonomy $\mathcal{U}(C)$ captures this total rotation. In flat space with no fields, you'd return unchanged ($\mathcal{U} = I$). But with gauge fields present, you accumulate a non-trivial transformation—this deviation from identity measures the field strength!
:::

$$
\mathcal U^{(3)}(C)=\prod_{a\in C} U^{(3)}_a,\quad \mathcal U^{(2)}(C)=\prod_{a\in C} U^{(2)}_a,\quad \mathcal u^{(1)}(C)=\prod_{a\in C} u^{(1)}_a.
$$

:::{prf:definition} Gauge actions
:label: def-sm-gauge-actions
:nonumber:

With weights $w_\varepsilon(C)>0$ and $\beta^{(k)}_\varepsilon$ ($k=3,2,1$),

$$
\boxed{\;
\begin{aligned}
S^{(3)}_g &= \frac{\beta^{(3)}_\varepsilon}{2N_c}\sum_{C} w_\varepsilon(C)\Big(1-\frac{1}{N_c}\operatorname{Re}\operatorname{Tr}\mathcal U^{(3)}(C)\Big),\\
S^{(2)}_g &= \frac{\beta^{(2)}_\varepsilon}{2N_w}\sum_{C} w_\varepsilon(C)\Big(1-\frac{1}{N_w}\operatorname{Re}\operatorname{Tr}\mathcal U^{(2)}(C)\Big),\\
S^{(1)}_g &= \frac{\beta^{(1)}_\varepsilon}{2}\sum_{C} w_\varepsilon(C)\Big(1-\operatorname{Re}\,\mathcal u^{(1)}(C)\Big),
\end{aligned}
}
$$

with $N_c=3$, $N_w=2$.
:::
:::{prf:theorem} Gauge invariance
:label: thm-sm-gauge-invariance
:nonumber:

Each $S^{(k)}_g$ is invariant under its corresponding gauge transformations.
:::

:::{admonition} 🔑 The Magic of Traces
:class: note
Why does taking the trace make things gauge invariant? When you conjugate a matrix $M \to gMg^{-1}$, its trace doesn't change because $\text{Tr}(gMg^{-1}) = \text{Tr}(Mg^{-1}g) = \text{Tr}(M)$. This is like saying that the sum of eigenvalues is independent of which basis you use to write the matrix!
:::

:::{prf:proof}

:::{admonition} 🔍 Why This Works
:class: hint
The proof is elegant: when you transform $U_a \mapsto g(e)U_a g(e')^\dagger$ for each edge, the product around a closed loop becomes $g(e_0)\mathcal{U}(C)g(e_0)^\dagger$ (since intermediate $g$'s cancel). Taking the trace kills the conjugation because $\text{Tr}(gMg^{-1}) = \text{Tr}(M)$. It's like rearranging beads on a closed necklace—the pattern stays the same!
:::

Holonomies transform by conjugation (or a phase for U(1)); traces and real parts are invariant.
:::

### 2.3 Small‑loop expansion and continuum limit

:::{admonition} 🔬 From Discrete to Continuous
:class: important
:class: dropdown
This is where the magic happens! When cycles become very small (as $\varepsilon \to 0$), the discrete Wilson action converges to the familiar Yang-Mills action of continuous gauge theory. It's like how a discrete sum becomes an integral: $\sum_i f(x_i)\Delta x \to \int f(x)dx$. The holonomy around a tiny loop encodes the field strength tensor $F_{\mu\nu}$—the curl of the gauge field.
:::

Assuming manifoldlike scaling, for small cycles

$$
\mathcal U^{(k)}(C)=\exp\big(i g_k F^{(k)}_{\mu\nu}(x_C)\Sigma^{\mu\nu}(C)+O(\varepsilon^3)\big),
$$

hence $\operatorname{Re}\operatorname{Tr}\mathcal U^{(k)}(C)=N_k-\tfrac{g_k^2}{2}\operatorname{Tr}(F_{\mu\nu}F_{\rho\sigma})\Sigma^{\mu\nu}\Sigma^{\rho\sigma}+O(\varepsilon^6)$.

:::{admonition} 🎯 The Heart of Field Theory
:class: important
:class: dropdown
This equation is where discrete meets continuous! For tiny loops, the holonomy $\mathcal{U}(C) \approx I + i g F_{\mu\nu}\Sigma^{\mu\nu}$. The deviation from identity (measured by $1 - \frac{1}{N}\text{Re}\text{Tr}\mathcal{U}$) is proportional to the field strength squared—exactly what appears in the Yang-Mills action!

It's like measuring the curvature of space by walking around tiny squares: the failure to return to your starting orientation tells you about the local curvature.
:::

:::{prf:theorem} Consistency of gauge sector
:label: thm-sm-gauge-consistency
:nonumber:

With a geometric calibration of $w_\varepsilon(C)$ and $\beta^{(k)}_\varepsilon$,

$$
\boxed{\;S^{(3)}_g+S^{(2)}_g+S^{(1)}_g\;\xrightarrow{\varepsilon\to 0}\; \int \Big(\tfrac{1}{4g_3^2}\mathrm{Tr}G_{\mu\nu}G^{\mu\nu}+\tfrac{1}{4g_2^2}\mathrm{Tr}W_{\mu\nu}W^{\mu\nu}+\tfrac{1}{4g_1^2}B_{\mu\nu}B^{\mu\nu}\Big)\sqrt{-g}d^4x.}
$$
:::
:::{prf:proof}

Small‑loop holonomy expansion + Riemann‑sum argument as in the QCD layer; weights are chosen so the sum over cycle areas reproduces the 4‑volume integral.
:::


## 3. Fermion content and covariant Dirac operators

:::{admonition} 🎪 The Cast of Characters
:class: attention
:class: dropdown
The Standard Model is like a play with a specific cast of fundamental particles. Each "generation" contains:
- **Quarks**: Come in two types (up/down) and three colors, feel all forces
- **Leptons**: The electron and neutrino, no color charge so immune to strong force
- **Left vs Right**: Nature is left-handed! Only left-handed particles feel the weak force

The numbers in parentheses tell you: (color representation, weak representation)$_{hypercharge}$. Think of these as different "membership cards" that determine which forces affect each particle.
:::

### 3.1 Field content (one generation)

:::{admonition} Intuition: The Particle Zoo's Organization
:class: dropdown
Nature organizes fermions in a very specific way. Left-handed particles always come in pairs (doublets)—like dance partners. The up and down quarks dance together, as do the electron and neutrino. Right-handed particles are loners (singlets).

The notation $(\mathbf{3}, \mathbf{2})_{+1/6}$ is like an address:
- First number: apartment in the color building (3 = has color, 1 = colorless)
- Second number: room in the weak force hotel (2 = doublet room, 1 = single)
- Subscript: your electric bill (hypercharge)

This specific arrangement isn't arbitrary—it's the ONLY way to make all anomalies cancel!
:::

Left doublets: quarks $Q_L=(u_L,d_L)\sim (\mathbf 3,\mathbf 2)_{+1/6}$, leptons $L_L=(\nu_L,e_L)\sim (\mathbf 1,\mathbf 2)_{-1/2}$.
Right singlets: $u_R\sim(\mathbf 3,\mathbf 1)_{+2/3}$, $d_R\sim(\mathbf 3,\mathbf 1)_{-1/3}$, $e_R\sim(\mathbf 1,\mathbf 1)_{-1}$.
(Optionally) $\nu_R\sim(\mathbf 1,\mathbf 1)_0$ for Dirac neutrino masses.

### 3.2 Transport and discrete covariant derivatives

:::{admonition} 🚂 Parallel Transport on the Network
:class: note
When a particle hops from node to node, it needs to know how to "carry" its properties. This is parallel transport—like carrying a vector along a curved surface while keeping it as "straight" as possible. The combined transport $\mathbb{U}$ accounts for:
1. How spin rotates (from curved spacetime)
2. How color changes (strong force)
3. How weak isospin changes (weak force)
4. How phase shifts (electromagnetic)

All these happen simultaneously as the particle moves!
:::

Let $\mathcal U^{\mathrm{spin}}_{e\to e'}\in \mathrm{SL}(2,\mathbb C)$ be spin transport (from local tetrads). For a representation $R$ with hypercharge $Y$, define **combined transport**

$$
\mathbb U^{(R)}_{e\to e'}=\mathcal U^{\mathrm{spin}}_{e\to e'}\otimes U^{(3)}_{e\to e'}\big|_{\rho_c(R)}\otimes U^{(2)}_{e\to e'}\big|_{\rho_w(R)}\cdot \big(u^{(1)}_{e\to e'}\big)^{Y}.
$$

:::{prf:definition} Dirac operators for SM fermions
:label: def-sm-dirac-operators
:nonumber:

For $\Psi_R:\mathcal E\to \mathbb C^{4}\otimes \rho_c(R)\otimes \rho_w(R)$, set
$$
\boxed{\;
(D^{R}_{\varepsilon,N}\Psi_R)(e)=\frac{i}{\tau_e}\sum_{e'\in \mathrm{nb}_T(e)} \mathbf n_{e\to e'}\!\cdot\!\boldsymbol\gamma \,\mathbb U^{(R)}_{e\to e'}\Psi_R(e')\;+\;\frac{i}{\tau_e}\sum_{\overrightarrow{e\,e''}\in \mathrm{nb}_{\mathrm{IG}}(e)} \eta^{(R)}_{e\to e''}\,\mathbb U^{(R)}_{e\to e''}\Psi_R(e'')\;-\;m_R\,\Psi_R(e)\;}
$$
with $\eta^{(R)}$ interaction phases (bounded) and $m_R$ bare masses (to be generated by Yukawas below; set to 0 before SSB).

:::{admonition} 🌊 The Discrete Dirac Equation
:class: hint
:class: simple
This is the discrete version of the famous Dirac equation! The first sum handles "parent-child" hops (CST edges), the second handles "sibling" interactions (IG edges), and the mass term keeps particles from spreading infinitely. Before symmetry breaking, $m_R = 0$—all particles are massless and travel at light speed. The phases $\eta$ encode quantum interference between different paths.
:::
:::

:::{prf:theorem} Gauge covariance and consistency
:label: thm-sm-dirac-covariance
:nonumber:

Each $D^{R}_{\varepsilon,N}$ is gauge‑covariant and, for smooth test spinors,

$$
(D^{R}_{\varepsilon,N}\Psi_R)|_{\mathcal E}\;\to\;(i\gamma^\mu\nabla_\mu + i g_3 \gamma^\mu G_\mu^a T^a + i g_2 \gamma^\mu W_\mu^i \tfrac{\sigma^i}{2} + i g_1 Y \gamma^\mu B_\mu - m_R)\Psi_R.
$$
:::
:::{prf:proof}

:::{admonition} 📐 The Small-Step Expansion
:class: note
For small steps, link variables are almost identity: $U \approx I + ig A_\mu \Delta x^\mu$. This is like saying "for tiny rotations, $\sin\theta \approx \theta$." When you substitute this into the discrete Dirac operator and take the continuum limit, you recover the familiar covariant derivative $\partial_\mu + igA_\mu$. The miracle is that gauge covariance is preserved at every step—the discrete and continuous theories transform the same way!
:::

Transport expansion $U^{(k)}=I+i g_k A_\mu^{(k)}\Delta x^\mu+O(\varepsilon^2)$ gives the covariant derivative; gauge covariance is by intertwining of nodewise transforms.
:::

### 3.3 Pauli exclusion and fermionic measure

:::{admonition} ⚠️ No Two Fermions in the Same State
:class: warning
:class: dropdown
The Pauli exclusion principle is fundamental: no two identical fermions can occupy the same quantum state. On our discrete network, this is enforced through anticommuting Grassmann variables and determinantal measures. Think of it as a cosmic "social distancing" rule for fermions—they literally cannot be in the same place with the same quantum numbers. This is why matter is stable and doesn't collapse!
:::

Pauli exclusion holds via the **fermionic microcell/determinant** construction (see fermionic layer). The grand‑canonical measure over admissible configurations yields Fermi–Dirac occupancies and antisymmetric Slater weights, independent of gauge choice.


## 4. Higgs field, SSB, and Yukawa couplings

:::{admonition} 🏔️ The Higgs Mechanism: Mass from Symmetry Breaking
:class: important
:class: dropdown
The Higgs mechanism is perhaps the most beautiful idea in the Standard Model. Imagine a perfectly symmetric Mexican hat potential. A ball at the peak is symmetric but unstable. It rolls down to the valley, "choosing" a direction and breaking the symmetry.

The Higgs field does this throughout space: it "chooses" a value everywhere, breaking electroweak symmetry. Particles get mass by interacting with this omnipresent field—like moving through molasses. The stronger the interaction (Yukawa coupling), the more massive the particle.

Without the Higgs, all particles would be massless and zip around at light speed. The universe would be very different!
:::

### 4.1 Discrete Higgs field and covariant kinetic term

:::{admonition} Intuition: The Field That Fills the Universe
:class: dropdown
The Higgs field is unique—it's the only fundamental scalar (spin-0) field we know. Unlike fermions which can be present or absent, the Higgs field has a non-zero value everywhere in space, even in "empty" vacuum.

Think of it as an invisible ocean that fills all of space. Particles gain mass by how strongly they interact with this ocean. Photons don't interact with it (massless), electrons interact weakly (light mass), while top quarks interact strongly (heavy mass). The Higgs boson is like a wave in this ocean—proof that the ocean exists!
:::

:::{prf:definition} Higgs field on nodes
:label: def-sm-higgs-field
:nonumber:

Attach a complex doublet $\Phi(e)\in \mathbb C^2$ to each episode $e$ (spin‑0 scalar, weak doublet, hypercharge $+1/2$, color singlet). Define covariant edge‑difference

$$
\nabla_{e\to e'}\Phi:= U^{(2)}_{e\to e'}\,\Phi(e) \cdot \big(u^{(1)}_{e\to e'}\big)^{+1/2} - \Phi(e').
$$

:::{admonition} 💡 The Covariant Difference
:class: tip
This isn't just $\Phi(e') - \Phi(e)$! The gauge factors $U^{(2)}$ and $u^{(1)}$ ensure we're comparing "apples to apples." It's like comparing temperatures in different cities—you need to account for different scales (Celsius vs Fahrenheit). Here, we're accounting for different gauge orientations at each node. Without these factors, the difference wouldn't be gauge-invariant.
:::
Higgs action:

$$
\boxed{\;S_H=\sum_{a=e\to e'} \kappa_\varepsilon(a)\,\|\nabla_{e\to e'}\Phi\|^2 \;+\; \sum_{e} \lambda_\varepsilon\!\left(\|\Phi(e)\|^2-\tfrac{v_\varepsilon^2}{2}\right)^{\!2}\;}
$$

with positive edge weights $\kappa_\varepsilon(a)$ calibrated by geometry and $v_\varepsilon,\lambda_\varepsilon>0$.
:::

:::{admonition} 💡 The Mexican Hat Potential
:class: tip
The term $(\|\Phi\|^2 - v^2/2)^2$ creates a potential shaped like a Mexican hat. The minimum isn't at $\Phi = 0$ but at $\|\Phi\| = v/\sqrt{2}$—a circle of minima! The field must "choose" a point on this circle, spontaneously breaking the symmetry. This choice gives mass to the W and Z bosons while leaving the photon massless.
:::

:::{prf:theorem} Gauge invariance and continuum limit
:label: thm-sm-higgs-invariance
:nonumber:

$S_H$ is gauge‑invariant. With calibration of $\kappa_\varepsilon,\lambda_\varepsilon,v_\varepsilon$,

$$
S_H \xrightarrow{\varepsilon\to 0} \int \Big((D_\mu\Phi)^\dagger(D^\mu\Phi) + \lambda(\Phi^\dagger\Phi-\tfrac{v^2}{2})^2\Big)\sqrt{-g}d^4x.
$$
:::
:::{prf:proof}

:::{admonition} 🔄 Gauge Blindness of the Potential
:class: hint
The potential $V(\Phi) = \lambda(|\Phi|^2 - v^2/2)^2$ depends only on the magnitude $|\Phi|$, not its phase or orientation. This is crucial—it means the potential "doesn't care" about gauge transformations, only about how far the field is from its preferred value $v/\sqrt{2}$. The kinetic term, however, must be gauge-covariant to ensure particles interact correctly with gauge bosons.
:::

The edge‑difference is covariant; the potential is gauge‑blind. Small‑edge expansion produces the continuum kinetic term; volume calibration gives the integral.
:::

### 4.2 Yukawa couplings and masses

:::{admonition} 🔗 How Particles Get Their Mass
:class: note
Yukawa couplings are the "handshake" between fermions and the Higgs field. Each fermion has its own coupling strength $y$—this determines how strongly it interacts with the Higgs. After symmetry breaking, when the Higgs gets a vacuum expectation value $v$, these couplings become masses: $m = yv/\sqrt{2}$.

The top quark has $y \approx 1$ (strong coupling), while the electron has $y \approx 10^{-6}$ (weak coupling). Nobody knows why these numbers are what they are—it's one of the great mysteries!
:::

:::{prf:definition} Gauge‑invariant Yukawa terms on nodes
:label: def-sm-yukawa-terms
:nonumber:

At each node $e$, let $\widetilde{\Phi}(e)=i\sigma^2\Phi^*(e)$. For generation indices $i,j$,

$$
\boxed{\;
\begin{aligned}
S_Y=&\sum_e \Big[\; \overline{Q_L^i}(e)\,\Phi(e)\,y^{d}_{ij}\,d_R^j(e)\;+\;\overline{Q_L^i}(e)\,\widetilde{\Phi}(e)\,y^{u}_{ij}\,u_R^j(e)\\
&\qquad+\;\overline{L_L^i}(e)\,\Phi(e)\,y^{e}_{ij}\,e_R^j(e)\;+\;(\overline{L_L^i}(e)\,\widetilde{\Phi}(e)\,y^{\nu}_{ij}\,\nu_R^j(e)+\text{h.c.})\;\Big].
\end{aligned}
}
$$
:::
:::{prf:theorem} SSB and mass generation
:label: thm-sm-ssb-mass
:nonumber:

:::{admonition} 🎯 The Moment of Symmetry Breaking
:class: hint
:class: dropdown
When the Higgs field "chooses" its vacuum value, several dramatic things happen simultaneously:
1. The $W^\pm$ and $Z$ bosons eat three of the four Higgs components (Goldstone bosons) and become massive
2. One Higgs component remains as the physical Higgs boson
3. All fermions acquire masses proportional to their Yukawa couplings
4. The photon remains massless because U(1)$_{em}$ symmetry is preserved

It's like a phase transition—suddenly the universe goes from having massless particles to having the mass spectrum we observe!
:::

In a gauge where $\Phi$ attains $\langle\Phi\rangle=(0,v/\sqrt2)^T$, the Dirac operators acquire mass matrices $M_f=\tfrac{v}{\sqrt2}y^f$ for $f=u,d,e,\nu$; the electroweak gauge links mix into massive $W^\pm,Z$ and massless $A$ with $\tan\theta_W=g_1/g_2$.
:::
:::{prf:proof}

:::{admonition} 🎭 The Unfolding Drama of Symmetry Breaking
:class: important
:class: dropdown
This proof encodes one of physics' most beautiful mechanisms! Here's what happens step by step:

1. **Choosing a vacuum**: The Higgs "chooses" $\langle\Phi\rangle = (0, v/\sqrt{2})^T$, breaking the symmetry
2. **Goldstone consumption**: Three would-be massless bosons (Goldstones) get "eaten" by $W^+, W^-, Z$, giving them mass
3. **The photon escapes**: A special combination $A_\mu = \sin\theta_W W^3_\mu + \cos\theta_W B_\mu$ remains massless—this is the photon!
4. **Weinberg angle**: The mixing angle $\theta_W$ determines how the original $W^3$ and $B$ fields combine into photon and $Z$

It's like a dance where four partners (the gauge bosons) suddenly change—three gain weight while one stays light and nimble!
:::

Standard completion of the square at each node; unitary gauge removes Goldstones; the edge‑kinetic term yields $m_W,m_Z$ from the quadratic part; the hypercharge–isospin mixing produces $A_\mu=\sin\theta_W W^3_\mu+\cos\theta_W B_\mu$ massless, $Z_\mu$ massive.
:::

### 4.3 Flavor mixing (CKM/PMNS)

:::{admonition} 🔄 The Flavor Puzzle
:class: attention
:class: dropdown
Nature's quarks and leptons come in three "flavors" (generations), and here's the twist: the states that have definite mass are NOT the same as the states that interact via the weak force!

It's like having two different ID cards—one for weighing yourself (mass eigenstate) and one for weak interactions (flavor eigenstate). The CKM matrix for quarks and PMNS matrix for leptons are the "translation guides" between these two identities. This mismatch is why we see phenomena like neutrino oscillations and CP violation!
:::

:::{prf:definition} Unitary flavor rotations
:label: def-sm-flavor-rotations
:nonumber:

Let $U_{u_L},U_{d_L},U_{e_L},U_{\nu_L}$ be unitary matrices diagonalizing $M_f$. Redefine fields $\psi_L\mapsto U_{\psi_L}\psi_L$. The charged‑current link term acquires **CKM** $V_{\rm CKM}=U_{u_L}^\dagger U_{d_L}$ and **PMNS** $U_{\rm PMNS}=U_{e_L}^\dagger U_{\nu_L}$ factors on the $W^\pm$ edges.
:::
:::{prf:theorem} Gauge invariance and locality
:label: thm-sm-flavor-invariance
:nonumber:

Flavor rotations act only on generation indices at nodes, commute with gauge transport, and preserve all gauge actions; mixing appears solely in charged currents.
:::
:::{prf:proof}

:::{admonition} 🎲 The Origin of Mixing
:class: note
Flavor mixing only appears in charged currents (W boson interactions) because only the W couples left-handed up-type to down-type quarks. The Z and photon couple to particles of the same type, so no mixing there. This asymmetry is why we see CP violation—nature has a slight preference for matter over antimatter, possibly explaining why we exist!
:::

Nodewise unitaries leave holonomies unchanged; only left‑doublet charged currents mix, reproducing the SM structure.
:::


## 5. Anomalies, ghosts (optional), and BRST‑free invariance

:::{admonition} 🚨 Anomalies: When Symmetries Break Down
:class: danger
:class: dropdown
Anomalies are the bane of gauge theories—they occur when a classical symmetry fails at the quantum level. Imagine a perfectly balanced seesaw that tilts when you add quantum corrections. For the Standard Model to be consistent, all gauge anomalies must cancel exactly.

Miraculously, with the exact particle content we observe (including the right hypercharges), all anomalies cancel generation by generation! Change even one particle's charge slightly, and the theory becomes inconsistent. This is a profound constraint on possible physics.
:::

### 5.1 Anomaly cancellation
:::{prf:theorem} Gauge anomalies cancel generation‑wise
:label: thm-sm-anomaly-cancellation
:nonumber:

Let $\mathcal A^{abc}\propto \sum_R \operatorname{Tr}\big(\{T^a_R,T^b_R\}T^c_R\big)$ with hypercharges; on the discrete cycle basis, the triangle contributions sum to zero for the SM assignments per generation.
:::
:::{prf:proof} Algebraic verification

:::{admonition} Intuition: The Miraculous Cancellation
:class: dropdown
This is perhaps the most miraculous "coincidence" in physics! Take all the particles in one generation and add up their hypercharges: you get exactly zero. Do the same with $Y^3$: zero again. These aren't small numbers that are approximately zero—they're EXACTLY zero.

It's like having a complex equation with dozens of terms that just happens to equal zero. Change even one particle's charge by a tiny amount, and the theory becomes mathematically inconsistent. This suggests the Standard Model's particle content isn't random—there's a deep principle at work that we don't fully understand yet.
:::

Same representation‑theory cancellation as in continuum: $\mathrm{SU}(3)^3,\mathrm{SU}(2)^3$ vanish by (anti)fundamental pairing; mixed and $\mathrm{U}(1)^3$ vanish due to $\sum Y= \sum Y^3=0$ over the listed fields. The discrete triangle weights reproduce these traces by small‑loop expansion.
:::

### 5.2 Gauge fixing and ghosts (optional)

:::{admonition} 👻 Faddeev-Popov Ghosts
:class: note
Ghosts aren't physical particles—they're mathematical tools that appear when you fix a gauge. Think of gauge symmetry as having "too much freedom" in your description. When you remove this redundancy (gauge fixing), you need to compensate with ghost fields to maintain unitarity. On our discrete lattice, the Wilson formulation cleverly avoids this complication by maintaining manifest gauge invariance!
:::

A Wilson‑style gauge‑invariant formulation needs no gauge fixing for nonperturbative simulations; if perturbative expansions are desired, one may add discrete BRST‑exact terms and ghost link fields on cycles, which converge to Faddeev–Popov structures in the limit.

:::{admonition} 💡 The Advantage of Manifest Gauge Invariance
:class: tip
Our discrete formulation maintains gauge invariance at every step—a huge advantage! In the continuum theory, you often need to "fix a gauge" (choose specific values for gauge freedoms) and then add ghost fields to maintain unitarity. It's like taking a photo: you need to choose a viewpoint (gauge fixing), but then must correct for the distortion this causes (ghosts). Our Wilson formulation avoids this complication entirely for non-perturbative calculations.
:::


## 6. Full discrete SM action and generating functional

:::{admonition} 🎼 The Complete Symphony
:class: important
:class: dropdown
The total Standard Model action is like a grand symphony with multiple movements:
- **Gauge terms** ($S_g$): The dynamics of force carriers (gluons, W/Z, photon)
- **Higgs terms** ($S_H$): The scalar field that breaks symmetry and generates mass
- **Yukawa terms** ($S_Y$): The interactions that give fermions their masses
- **Fermion kinetic terms**: How matter particles propagate and interact

Each term is gauge invariant on its own, and together they describe all known particle physics (except gravity). This single formula encodes centuries of experimental discoveries!
:::

:::{prf:definition} Total action
:label: def-sm-total-action
:nonumber:

$$
\boxed{\; S_{\rm SM}^{(\varepsilon)}[U^{(3)},U^{(2)},u^{(1)},\Phi,\{\bar\Psi,\Psi\}]\;=\;S^{(3)}_g+S^{(2)}_g+S^{(1)}_g+S_H+S_Y+ \sum_R \overline{\Psi_R}\,D^{R}_{\varepsilon,N}\,\Psi_R\;}
$$

with sums over all generations and species $R$.
:::
Define the generating functional (integrating out fermions if desired):

$$
\mathcal Z_\varepsilon=\int \mathcal D U^{(3)}\mathcal D U^{(2)}\mathcal D u^{(1)}\mathcal D\Phi\,\prod_R \mathcal D\bar\Psi_R\mathcal D\Psi_R\; e^{-S_{\rm SM}^{(\varepsilon)}}.
$$

:::{prf:theorem} Gauge invariance and existence on finite graphs
:label: thm-sm-existence
:nonumber:

On any finite $\mathcal E^{\Rightarrow}$, the measure and action are well‑defined and gauge‑invariant; fermion inverses exist almost surely for small steps (mass/diagonal dominance).
:::
:::{prf:proof}

:::{admonition} 🎯 Why Everything Is Well-Defined
:class: hint
Three key facts ensure our theory makes mathematical sense:
1. **Haar measures**: These are the unique "fair" probability distributions on groups—like uniform distribution on a circle
2. **Polynomial potentials**: These grow at most like $\Phi^4$, ensuring integrals converge
3. **Gershgorin bounds**: These tell us when matrices are invertible—crucial for fermion propagators

Together, these ensure we can actually compute physical quantities without encountering infinities or undefined operations.
:::

Product Haar measures on compact groups; polynomial scalar potential; Gershgorin bounds for Dirac blocks.
:::


## 7. Continuum limit: convergence to the SM Lagrangian

:::{admonition} 🌉 Bridging Discrete and Continuous
:class: tip
This section proves that our discrete formulation isn't just an approximation—it converges exactly to the standard continuum theory as the lattice spacing goes to zero. It's like showing that a digital photo, with enough pixels, becomes indistinguishable from the continuous scene it represents. This convergence validates that we're doing "real" physics on the discrete network.
:::

Assume **faithful embedding, local finiteness, bounded geometry**, standardization/aggregation continuity, and **stability** (uniform bounds on inverses for test spaces).

:::{prf:theorem} Action‑level convergence
:label: thm-sm-action-convergence
:nonumber:

With geometric calibrations of $w_\varepsilon,\beta^{(k)}_\varepsilon,\kappa_\varepsilon,\lambda_\varepsilon,v_\varepsilon$,

$$
\boxed{\;S_{\rm SM}^{(\varepsilon)} \;\xrightarrow{\varepsilon\to 0}\; \int\! d^4x\,\sqrt{-g}\,\Big[\mathcal L_{\rm YM}+\mathcal L_{\rm Higgs}+\mathcal L_{\rm Yukawa}+\sum_R \bar\Psi_R(i\slashed D - m_R)\Psi_R\Big]\;}
$$

i.e. the full continuum Standard Model (with optional Dirac neutrino masses).
:::
:::{prf:proof}

:::{admonition} 🎊 The Grand Convergence
:class: important
:class: dropdown
This theorem is the culmination of our construction! Each piece of the discrete theory converges to its continuum counterpart:
- Wilson loops → Yang-Mills field strength
- Discrete differences → Covariant derivatives
- Node sums → Spacetime integrals
- Discrete propagators → Continuum Green's functions

It's like showing that a pointillist painting, when viewed from the right distance, perfectly reproduces the Mona Lisa. Our discrete Standard Model isn't just an approximation—in the limit, it becomes exactly the theory tested in every particle physics experiment!
:::

Gauge sector: Section 2.3; Higgs: Section 4.1; Yukawa: Section 4.2; fermions: Section 3.2. Riemann‑sum and small‑edge expansions give termwise convergence; SSB analysis induces the standard mass spectrum and mixing.
:::

:::{prf:theorem} Propagator and correlator convergence
:label: thm-sm-correlator-convergence
:nonumber:

For smooth test sources, two‑point functions of gauge, Higgs, and fermion fields computed on the graph converge to their continuum SM counterparts; LSZ‑type limits of amputated correlators reproduce scattering amplitudes.
:::
:::{prf:proof}

:::{admonition} 🔬 From Propagators to Predictions
:class: note
The Lax equivalence theorem is a powerful principle: consistency + stability = convergence. Here it means:
- **Consistency**: Our discrete operators approximate the right continuous ones
- **Stability**: Small changes in input give small changes in output
- **Convergence**: Together, these guarantee we get the right physics!

The "amputated correlators" are correlation functions with external legs removed—these directly give scattering amplitudes that we compare with collider experiments.
:::

Follows from operator consistency and stability (Lax‑type arguments) plus known causal‑set/path‑sum convergence for propagators; amputated limits track convergent inverses and vertex insertions from local Yukawa/gauge couplings.
:::


## 8. Observables and algorithms

:::{admonition} 📊 What We Can Measure
:class: note
Physics is about predictions! On our discrete network, we can compute:
- **Confinement**: Do quarks remain forever bound? (Wilson loops test this)
- **Masses**: Extract particle masses from correlation functions
- **Decay rates**: How quickly particles transform
- **Scattering**: What happens when particles collide

These observables connect our abstract mathematical framework to real experimental measurements at the LHC and other facilities.
:::

- **QCD:** Wilson/Polyakov loops, string tension, hadron spectroscopy via color‑singlet bilinears.
- **Electroweak:** $m_W,m_Z,\sin^2\theta_W$ from two‑point functions; $G_F$ from four‑fermion limits; vector boson scattering.
- **Flavor:** CKM/PMNS extraction from charged‑current correlators; CP‑violating phases from Jarlskog‑type invariants on nodewise Yukawas.
- **Photon & QED:** After SSB, the residual U(1)$_{\rm em}$ link variables define compact QED on the fractal set; fine‑structure constant and running can be measured via scattering and bound‑state spectra.

**Algorithms.**

:::{admonition} Intuition: Computational Strategies
:class: simple
These algorithms are the workhorses of lattice field theory:
- **Heatbath/Metropolis**: Statistical sampling methods that explore configuration space like a random walk that prefers low-energy states
- **Hybrid Monte Carlo**: Combines molecular dynamics with random steps—like simulating planetary motion with occasional random kicks
- **Krylov solvers**: Efficient methods for solving $(D + m)\psi = \eta$ by building up solutions in expanding subspaces
- **Smearing**: Averaging over nearby points to reduce noise—like applying a blur filter to see large-scale patterns
:::

Heatbath/Metropolis on irregular cycles for each gauge factor; hybrid Monte‑Carlo if desired. Krylov solvers for Dirac inverses; multi‑shift for many masses. Measurement by gauge‑invariant smearings defined via CST geodesics and IG neighborhoods.


## 9. Extensions

:::{admonition} 🚀 Beyond the Standard Model
:class: seealso
:class: dropdown
The Standard Model, while incredibly successful, isn't the final theory. These extensions explore how to incorporate:
- **Neutrino masses**: We know neutrinos have mass (from oscillations), but the SM doesn't explain it
- **Gravity**: The fourth force, still resisting quantization
- **Grand Unification**: Could all forces merge at high energy into one superforce?
- **Dark matter/energy**: The SM only describes 5% of the universe!

Our discrete framework provides a natural laboratory for exploring these extensions.
:::

- **Neutrino masses:** Either add $\nu_R$ (Dirac) as above, or discrete Weinberg operator $\tfrac{1}{\Lambda}(\overline{L_L}\widetilde\Phi)(\widetilde\Phi^\dagger L_L)$ as a nodewise 4‑field coupling.
- **Curved backgrounds:** Native support via tetrads; minimal coupling already built in.
- **Finite temperature/density:** Use time‑cycle Polyakov loops and chemical potentials via temporal link phases on U(1)$_B$ and lepton number (approximate).
- **Grand unification (exploratory):** Replace product links by a single GUT link on a larger compact group; recover SM by breaking via nodewise adjoint scalars.


### Appendix A — SM field content and charges (per generation)

:::{admonition} 📋 Reading the Particle Table
:class: hint
This table is the "ID card" for every fundamental particle. The columns show which forces affect each particle:
- **SU(3)**: Bold 3 means "feels strong force" (quarks), 1 means "doesn't" (leptons)
- **SU(2)**: Bold 2 means "weak doublet" (left-handed), 1 means "singlet" (right-handed)
- **Y**: Hypercharge, related to electric charge by $Q = T^3 + Y$

Notice the pattern: all left-handed particles are doublets, all right-handed are singlets. This left-right asymmetry is built into nature!
:::

$$
\begin{array}{c|c|c|c}
\text{Field} & \mathrm{SU}(3)_c & \mathrm{SU}(2)_L & Y \\ \hline
Q_L & \mathbf 3 & \mathbf 2 & +\tfrac{1}{6} \\
u_R & \mathbf 3 & \mathbf 1 & +\tfrac{2}{3} \\
d_R & \mathbf 3 & \mathbf 1 & -\tfrac{1}{3} \\
L_L & \mathbf 1 & \mathbf 2 & -\tfrac{1}{2} \\
e_R & \mathbf 1 & \mathbf 1 & -1 \\
\nu_R\;(\text{optional}) & \mathbf 1 & \mathbf 1 & 0 \\
\Phi & \mathbf 1 & \mathbf 2 & +\tfrac{1}{2}
\end{array}
$$

**Charge operator:** $Q = T^3 + Y$.

:::{admonition} 💡 The Electric Charge Formula
:class: tip
This simple formula $Q = T^3 + Y$ explains all electric charges! $T^3$ is the third component of weak isospin (±1/2 for doublets, 0 for singlets), and $Y$ is hypercharge. For example:
- Electron: $T^3 = -1/2$, $Y = -1/2$, so $Q = -1$ ✓
- Up quark: $T^3 = +1/2$, $Y = +1/6$, so $Q = +2/3$ ✓
- Neutrino: $T^3 = +1/2$, $Y = -1/2$, so $Q = 0$ ✓

After symmetry breaking, the original $W^3$ and $B$ fields mix at angle $\theta_W$ to give us the photon and Z boson!
:::

After SSB: $A_\mu = \sin\theta_W\, W^3_\mu + \cos\theta_W\, B_\mu$, $Z_\mu = \cos\theta_W\, W^3_\mu - \sin\theta_W\, B_\mu$.

### Appendix B — Geometric calibration of cycle weights

:::{admonition} ⚖️ Getting the Weights Right
:class: note
The weights $w_\varepsilon(C)$ ensure that discrete sums converge to continuous integrals. Think of it as choosing the right "pixel size" for each region of space so that the discrete picture accurately represents the continuous one. This calibration is crucial for recovering the correct continuum physics.
:::

Select $w_\varepsilon(C)$ so that $\sum_{C\subset\mathcal N_x} w_\varepsilon(C)\,A(C)^2 \approx \Delta V(\mathcal N_x)$. With $\beta^{(k)}_\varepsilon$ matched, the Wilson sums approximate $\int \mathrm{Tr}F^2$.

### Appendix C — Anomaly checks in the discrete setting

:::{admonition} 🔺 The Triangle Test
:class: caution
:class: dropdown
Anomaly cancellation is so important it gets its own appendix! The "triangle diagrams" refer to quantum loops with three gauge bosons. In theories with anomalies, these loops would violate gauge invariance and destroy the theory's consistency.

The miracle: with exactly the particle content we observe in nature, all these dangerous terms cancel. The sum $\sum Y = 0$ means electric charge is conserved, $\sum Y^3 = 0$ ensures no gauge anomalies. It's like a complex equation that just happens to equal zero—suggesting deep underlying structure.
:::

Triangle diagrams correspond to 3‑cycle concatenations on the cycle basis. Their small‑loop traces reproduce the same group theory coefficients as in the continuum; with the table in Appendix A one verifies $\sum Y=\sum Y^3=\sum Y\,\mathrm{Tr}T^aT^b=0$ per generation.

### Appendix D — LSZ and scattering on the graph

:::{admonition} 💥 From Correlations to Collisions
:class: tip
The LSZ (Lehmann-Symanzik-Zimmermann) reduction formula is the bridge between quantum field correlators and actual scattering experiments. It tells us how to extract what happens when particles collide (S-matrix) from the correlation functions we compute. On our discrete network, this same machinery works, allowing us to predict collision outcomes at particle accelerators!
:::

Define amputated connected correlators by inverting discrete two‑point kernels and removing external transports; in the limit $\varepsilon\to 0$, the discrete LSZ reduction yields continuum S‑matrix elements.

:::{admonition} Intuition: Computing What Detectors See
:class: simple
The LSZ formula connects our abstract math to what particle detectors actually measure. "Amputated" means we remove the external particle lines (they're not part of the interaction). "Connected" means we only consider processes where all particles actually interact (not just fly past each other). The S-matrix elements tell us probabilities: if particles A and B collide, what's the chance of producing particles C and D? This is what the LHC measures millions of times per second!
:::

*End of document.*
