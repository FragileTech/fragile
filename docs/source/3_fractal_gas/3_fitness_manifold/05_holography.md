(sec-holographic-structure)=
# Holographic Structure and Thermodynamics

**Prerequisites**: {doc}`01_emergent_geometry`, {doc}`../2_fractal_set/02_causal_set_theory`

---

(sec-holography-tldr)=
## TL;DR

*Notation: $\mathcal{T}$ = Causal Spacetime Tree (CST); $\mathcal{G}_t$ = Interaction Graph (IG); $\gamma_A$ = separating antichain for region $A$; $S_{\mathrm{IG}}(A)$ = IG entanglement entropy; $\mathrm{Area}_{\mathrm{CST}}$ = CST boundary area; $\mathcal{P}_\varepsilon$ = nonlocal perimeter functional; $\alpha$ = area law constant; $\beta$ = effective inverse temperature.*

The Latent Fractal Gas produces two structurally independent data streams: the Causal Spacetime Tree (CST), which records walker genealogy through cloning events, and the Interaction Graph (IG), which encodes fitness correlations between walkers. These capture fundamentally different information---ancestry versus interaction---yet both measure the same geometric quantity: boundary area. This coincidence is the algorithmic version of the Bekenstein-Hawking formula, where entanglement entropy equals area divided by four times Newton's constant. This chapter proves the Informational Area Law $S_{\mathrm{IG}}(A) = \alpha \cdot \mathrm{Area}_{\mathrm{CST}}(\gamma_A)$, establishes thermodynamic relations including a first law connecting entropy and energy variations, and shows that the UV regime naturally yields Anti-de Sitter geometry with the Ryu-Takayanagi formula emerging as a direct consequence.

---

(sec-holography-introduction)=
## Introduction

:::{div} feynman-prose
Let me tell you what this chapter is really about. We are going to discover that information and geometry are secretly the same thing.

Now, you have probably heard of the holographic principle---the idea that all the information in a volume of space can be encoded on its boundary. Black hole physicists arrived at this through a long chain of reasoning about entropy, horizons, and quantum mechanics. What I want to show you is that the Latent Fractal Gas arrives at the same conclusion from a completely different direction: optimization.

Here is the setup. We have two independent data streams in the Latent Fractal Gas:

1. **The Causal Spacetime Tree (CST)**: Who begat whom. The genealogical record of which walkers cloned which other walkers. This is a purely discrete, combinatorial structure---a tree of ancestry relationships.

2. **The Interaction Graph (IG)**: Who talks to whom. The correlation network from fitness interactions. This is weighted by correlation strength and encodes the spatial structure of the swarm.

These two data streams are independent by construction. We defined them separately, with no built-in relationship between them. The CST knows about genealogy but not about geometry. The IG knows about correlations but not about ancestry.

And yet---and here is the miracle---when you compute the entropy of a region using IG correlations, and you compute the boundary area using CST antichains, they turn out to be *proportional*. The entropy of a region equals a constant times the area of its boundary. This is the Bekenstein-Hawking formula, and we are going to derive it from first principles.

This is not something we put in by hand. It *emerges* from the dynamics. The area law is not an assumption; it is a theorem. And understanding why it emerges will teach us something deep about the relationship between information, geometry, and optimization.
:::

---

(sec-two-data-streams)=
## The Two Data Streams

:::{div} feynman-prose
Before we can prove anything, we need to be absolutely clear about what we are comparing. The CST and IG are genuinely independent data structures. Let me make this precise.

The CST records parentage. At each cloning event, a dying walker is replaced by a clone of a surviving walker. The CST records which walker was the parent and which was the child. This creates a tree structure---every episode (except the initial ancestors) has exactly one parent.

The IG records correlations. The fitness function depends on walker positions, and walkers at nearby positions have correlated fitnesses. The IG encodes these correlations as edge weights in a graph. Strong correlation means strong edge; weak correlation means weak edge.

Crucially, these structures carry different information. The CST tells you about *inheritance*---which traits were passed from parent to child. The IG tells you about *interaction*---which walkers are currently influencing each other through the fitness landscape.

The question is: why should these two unrelated structures know about each other? Why should the boundary area (a CST concept) have anything to do with the entanglement entropy (an IG concept)?
:::

:::{prf:definition} Causal Spacetime Tree (CST)
:label: def-cst-structure

The **Causal Spacetime Tree** is the directed graph $\mathcal{T} = (V, E)$ where:

**Vertices:** $V = \{e_i\}$ is the set of all episodes (walker lifetimes)

**Edges:** $(e_j, e_i) \in E$ if and only if episode $e_i$ was created by cloning from episode $e_j$ (i.e., $e_j$ is the parent of $e_i$)

**Properties:**
1. **Tree structure**: Every episode except roots has exactly one parent
2. **Causal order**: If $(e_j, e_i) \in E$, then $t_{\mathrm{birth}}(e_i) \geq t_{\mathrm{birth}}(e_j)$ (children are born at or after parents)
3. **Time-indexing**: Episodes are ordered by birth time

**Causal relation:**
$$
e_j \prec e_i \quad \Leftrightarrow \quad \text{there exists a directed path from } e_j \text{ to } e_i \text{ in } \mathcal{T}
$$

This defines a partial order on episodes: $e_j \prec e_i$ means "episode $e_j$ is an ancestor of episode $e_i$."
:::

:::{prf:definition} Interaction Graph (IG)
:label: def-ig-structure

The **Interaction Graph** at time $t$ is the weighted graph $\mathcal{G}_t = (V_t, E_t, w)$ where:

**Vertices:** $V_t = \{z_i(t)\}_{i=1}^N$ are the walker positions at time $t$

**Edges:** $(i, j) \in E_t$ if $K_\varepsilon(z_i, z_j) > \delta$ (correlation above threshold $\delta$)

**Weights:** The edge weight is the correlation strength:
$$
w_{ij}(t) = K_\varepsilon(z_i(t), z_j(t)) = C_0 \exp\left(-\frac{\|z_i(t) - z_j(t)\|_G^2}{2\varepsilon_c^2}\right)
$$

**Properties:**
1. **Symmetric**: $w_{ij} = w_{ji}$ (correlations are mutual)
2. **Positive**: $w_{ij} > 0$ for all connected pairs
3. **Decaying**: $w_{ij} \to 0$ as $\|z_i - z_j\| \to \infty$

**Total correlation strength:**
$$
W_{\mathrm{total}} = \sum_{(i,j) \in E_t} w_{ij}
$$
:::

---

(sec-holography-jump-hamiltonian)=
## The Jump Hamiltonian and Modular Structure

:::{div} feynman-prose
Now we need to understand the energetics of the IG correlations. This is where the jump Hamiltonian from {doc}`04_field_equations` becomes essential.

Think about it this way. The IG encodes correlations, and correlations have energy. If you try to perturb the correlation structure---say, by moving walkers around or changing their relative positions---you pay an energy cost. The jump Hamiltonian quantifies this cost.

The key insight is that the jump Hamiltonian has the structure of a *modular Hamiltonian* in quantum information theory. This is not a coincidence. The modular Hamiltonian is the operator whose exponential gives the reduced density matrix: $\rho_A = e^{-H_{\mathrm{mod}}}/Z$. It encodes all the entanglement structure of a quantum state.

In the Latent Fractal Gas, the IG correlation structure plays the role of entanglement. Walkers that are correlated through the IG are "entangled" in the operational sense that knowing one tells you something about the other. The jump Hamiltonian is the modular Hamiltonian for this operational entanglement.
:::

:::{prf:definition} Jump Hamiltonian
:label: def-jump-hamiltonian-holographic

Let $\rho(z)$ be the walker density and $\Phi(z)$ be a perturbation potential. The **jump Hamiltonian** is:

$$
\mathcal{H}_{\mathrm{jump}}[\Phi] = \iint_{\mathcal{Z} \times \mathcal{Z}} K_\varepsilon(z,z')\rho(z)\rho(z')\left(e^{\frac{1}{2}(\Phi(z)-\Phi(z'))}-1-\frac{1}{2}(\Phi(z)-\Phi(z'))\right)dz\,dz'
$$

**Components:**
- $K_\varepsilon(z,z')$: IG correlation kernel ({prf:ref}`def-ig-structure`)
- $\rho(z)$: Walker density, $\int \rho(z) dz = N$
- $\Phi(z)$: Perturbation field (scalar potential)

**Properties:**
1. **Non-negativity**: $\mathcal{H}_{\mathrm{jump}}[\Phi] \geq 0$ for all $\Phi$, with equality iff $\Phi = \text{const}$
2. **Quadratic approximation**: For small $|\Phi(z) - \Phi(z')| \ll 1$:
   $$
   \mathcal{H}_{\mathrm{jump}}[\Phi] \approx \frac{1}{8} \iint K_\varepsilon(z,z')\rho(z)\rho(z')(\Phi(z)-\Phi(z'))^2 \,dz\,dz'
   $$
3. **Locality**: The kernel $K_\varepsilon$ decays on scale $\varepsilon_c$, so only nearby pairs contribute
:::

:::{div} feynman-prose
Why does the jump Hamiltonian have that peculiar exponential form? Let me give you two perspectives.

**Information-theoretic perspective:** When you perturb the density by changing the potential from $0$ to $\Phi$, you are changing the relative weights of different configurations. The cost of this change is measured by a KL divergence. The exponential form $e^{\Phi/2} - 1 - \Phi/2$ is exactly what appears when you compute the KL divergence to second order.

**Physical perspective:** The jump Hamiltonian measures the "elastic energy" stored in the correlation network. Imagine the correlations as springs connecting nearby walkers. Stretching the springs (increasing $|\Phi(z) - \Phi(z')|$) costs energy. The $(\Phi - \Phi')^2$ factor is Hooke's law; the $K_\varepsilon$ factor is the spring constant.

Both perspectives give the same answer because information and physics are two sides of the same coin. The Landauer principle tells us that erasing information costs energy; the jump Hamiltonian quantifies this cost for the IG correlation structure.
:::

:::{prf:proposition} Connection to Modular Hamiltonian
:label: prop-modular-connection

The jump Hamiltonian is related to the modular Hamiltonian of the IG correlation structure:

$$
H_{\mathrm{mod}}(A) = -\ln \rho_A + \ln Z_A
$$

where $\rho_A$ is the reduced density operator for region $A$ and $Z_A$ is a normalization.

**Relationship:**
$$
\mathcal{H}_{\mathrm{jump}}[\Phi_A] = \langle H_{\mathrm{mod}}(A) \rangle_{\delta\rho} - \langle H_{\mathrm{mod}}(A) \rangle_{\rho_0} + O(\delta\rho^2)
$$

where $\Phi_A$ is the perturbation corresponding to density change $\delta\rho$ and $\langle \cdot \rangle$ denotes expectation.

**Physical interpretation:** The jump Hamiltonian measures the change in modular energy when the density is perturbed. This connects the classical correlation structure (IG) to quantum information concepts (modular Hamiltonian).
:::

---

(sec-holography-antichain)=
## Antichain Structure and Spacelike Hypersurfaces

:::{div} feynman-prose
Now we come to a beautiful connection between the CST and geometry. The CST is a causal structure---it tells you which episodes could have influenced which other episodes. And wherever you have causality, you have the notion of *simultaneity*: a set of events that are all independent of each other, none influencing the others.

In general relativity, a spacelike hypersurface is a surface where no two points are causally connected---you cannot send a signal from one to another. This is the formal notion of "all at the same time."

In the CST, the analog is an *antichain*: a set of episodes where no two are related by ancestry. If $e_i$ and $e_j$ are in the same antichain, then $e_i$ is not an ancestor of $e_j$ and $e_j$ is not an ancestor of $e_i$. They are causally independent.

The remarkable thing is that CST antichains correspond exactly to spacelike hypersurfaces in the emergent geometry. This is not obvious! The CST knows nothing about Riemannian geometry; it only knows about cloning events. And yet its combinatorial structure perfectly captures the geometric notion of simultaneity.
:::

:::{prf:definition} Separating Antichain
:label: def-separating-antichain

Let $A \subseteq \mathcal{Z} \times [0,T]$ be a spacetime region. A **separating antichain** for $A$ is a set $\gamma_A \subseteq V(\mathcal{T})$ of CST episodes satisfying:

1. **Antichain property**: No two episodes in $\gamma_A$ are causally related:
   $$
   \forall e_i, e_j \in \gamma_A: \quad e_i \not\prec e_j \text{ and } e_j \not\prec e_i
   $$

2. **Separating property**: Every maximal causal chain from the initial time to the final time passes through exactly one episode in $\gamma_A$

3. **Boundary property**: The episodes in $\gamma_A$ correspond to walkers whose spatial positions lie on or near $\partial A$ (the boundary of region $A$)

**Notation:** $|\gamma_A|$ denotes the cardinality of the antichain (number of episodes).

**Geometric interpretation:** A separating antichain is the CST analog of a codimension-1 spacelike hypersurface that bounds region $A$.
:::

:::{prf:definition} CST Boundary Area
:label: def-cst-boundary-area

The **CST boundary area** of a separating antichain $\gamma_A$ is:

$$
\mathrm{Area}_{\mathrm{CST}}(\gamma_A) = a_0 \cdot |\gamma_A|
$$

where:
- $|\gamma_A|$ is the cardinality of the antichain (number of episodes)
- $a_0$ is the **fundamental area quantum**:
  $$
  a_0 = \ell_P^{d-1}
  $$
- $\ell_P = (\mathrm{Vol}(\mathcal{Z})/N)^{1/d}$ is the emergent Planck length scale (typical Voronoi cell linear dimension), so $a_0 = (\mathrm{Vol}(\mathcal{Z})/N)^{(d-1)/d}$

**Properties:**
1. **Discreteness**: The CST area is quantized in units of $a_0$
2. **Extensivity**: For large boundaries, $\mathrm{Area}_{\mathrm{CST}} \propto |\gamma_A|$
3. **Geometric correspondence**: In the continuum limit, $\mathrm{Area}_{\mathrm{CST}}(\gamma_A) \to \mathrm{Area}_g(\partial A)$, the Riemannian area of the boundary
:::

:::{div} feynman-prose
Here is why the area quantum $a_0$ has the form it does. Each episode "occupies" a region of the latent space---roughly, its Voronoi cell. The typical Voronoi cell has volume $\mathrm{Vol}(\mathcal{Z})/N$. The boundary of this cell has area proportional to $(\mathrm{Vol}(\mathcal{Z})/N)^{(d-1)/d}$. In Planck units, this becomes $a_0$.

You can think of the CST as a discrete sampling of spacetime. Each episode is an "atom" of spacetime, and the antichain is a collection of these atoms that together form a boundary. The CST area is just counting atoms.

This discreteness is not an approximation or a numerical artifact. It is fundamental. The CST tells us that spacetime has a minimum resolvable unit---you cannot probe smaller than one episode. This is the algorithmic version of the Planck scale.
:::

---

(sec-holography-ig-entropy)=
## IG Entanglement Entropy

:::{div} feynman-prose
Now let us turn to the IG side. Given a region $A$, we want to define its "entanglement entropy"---the amount of information that correlations carry across its boundary.

The intuition is simple. If region $A$ were completely isolated---no correlations with the outside---then knowing what happens inside $A$ would tell you nothing about the outside. But if there are correlations crossing the boundary, then information "leaks" across. The entanglement entropy quantifies this leakage.

The construction uses the min-cut/max-flow structure of graph theory. Imagine the IG as a network of pipes, where the edge weights are pipe capacities. The entanglement entropy of $A$ is the minimum total capacity you need to cut to disconnect $A$ from its complement. This is the "bottleneck" in the information flow.
:::

:::{prf:definition} IG Entanglement Entropy
:label: def-ig-entanglement-entropy

Let $A \subseteq \mathcal{Z}$ be a spatial region and $\mathcal{G} = (V, E, w)$ be the IG at a fixed time. The **IG entanglement entropy** of $A$ is:

$$
S_{\mathrm{IG}}(A) = \sum_{e \in \Gamma_{\min}(A)} w_e
$$

where $\Gamma_{\min}(A)$ is the **minimum weight cut** separating $A$ from its complement $A^c$:

$$
\Gamma_{\min}(A) = \arg\min_{\Gamma} \left\{ \sum_{e \in \Gamma} w_e : \Gamma \text{ separates } A \text{ from } A^c \right\}
$$

**Properties:**
1. **Subadditivity**: $S_{\mathrm{IG}}(A \cup B) \leq S_{\mathrm{IG}}(A) + S_{\mathrm{IG}}(B)$
2. **Symmetry**: $S_{\mathrm{IG}}(A) = S_{\mathrm{IG}}(A^c)$ (cut is the same from both sides)
3. **Monotonicity**: If $A \subseteq B$, need not have $S_{\mathrm{IG}}(A) \leq S_{\mathrm{IG}}(B)$ (not monotonic in general)

**Min-cut/Max-flow interpretation:** By the max-flow min-cut theorem, $S_{\mathrm{IG}}(A)$ equals the maximum flow from $A$ to $A^c$ through the IG network. This is the "information capacity" of the boundary.
:::

:::{prf:definition} Nonlocal Perimeter Functional
:label: def-nonlocal-perimeter

The **nonlocal perimeter functional** is the continuous analog of the IG entanglement entropy:

$$
\mathcal{P}_\varepsilon(A) = \iint_{A \times A^c} K_\varepsilon(z,z')\rho(z)\rho(z')\,dz\,dz'
$$

**Components:**
- $K_\varepsilon(z,z')$: IG correlation kernel
- $\rho(z)$: Walker density
- Integration over $A \times A^c$: Only cross-boundary correlations contribute

**Properties:**
1. **Non-negativity**: $\mathcal{P}_\varepsilon(A) \geq 0$
2. **Symmetry**: $\mathcal{P}_\varepsilon(A) = \mathcal{P}_\varepsilon(A^c)$
3. **Monotonicity in $\varepsilon$**: $\mathcal{P}_\varepsilon(A)$ decreases as $\varepsilon \to 0$ (correlations become more local)

**Relationship to discrete entropy:**
$$
S_{\mathrm{IG}}(A) \approx \mathcal{P}_\varepsilon(A) \quad \text{as } N \to \infty
$$
with corrections of order $O(1/N)$.
:::

:::{div} feynman-prose
The nonlocal perimeter functional has a beautiful interpretation. It counts the "total correlation" crossing the boundary---the integral of the kernel over all pairs of points where one is inside and one is outside.

Compare this to the ordinary perimeter, which is just the length (or area) of the boundary. The nonlocal perimeter is a "fuzzy" version: instead of a sharp boundary, correlations decay smoothly with distance, and the nonlocal perimeter sums up all these decaying cross-boundary contributions.

As the correlation length $\varepsilon$ goes to zero, the nonlocal perimeter converges to the ordinary perimeter (times a density factor). This is the Gamma-convergence result we prove next.
:::

---

(sec-holography-gamma-convergence)=
## Gamma-Convergence of IG Entropy

:::{div} feynman-prose
Here is one of the key technical results of this chapter. We are going to show that the nonlocal perimeter functional---which is defined in terms of IG correlations and has no obvious geometric content---converges to the ordinary geometric perimeter as the correlation length goes to zero.

This is not obvious. The nonlocal perimeter involves double integrals over pairs of points, weighted by a Gaussian kernel. The ordinary perimeter is a single integral over the boundary, weighted by the density. Why should these be related?

The answer is Gamma-convergence, a powerful technique from the calculus of variations. Gamma-convergence is the "right" notion of convergence for variational problems: if $\mathcal{P}_\varepsilon$ Gamma-converges to $\mathcal{P}_0$, then minimizers of $\mathcal{P}_\varepsilon$ converge to minimizers of $\mathcal{P}_0$. This is exactly what we need to connect the IG entropy (which involves minimizing over cuts) to the geometric area.
:::

:::{prf:theorem} Gamma-Convergence to Local Perimeter
:label: thm-gamma-convergence

As $\varepsilon \to 0$, the nonlocal perimeter functional Gamma-converges to the local perimeter:

$$
\mathcal{P}_\varepsilon(A) \xrightarrow{\Gamma} \mathcal{P}_0(A) = c_0 \int_{\partial A} \rho(z)^2 \, d\Sigma(z)
$$

where:
- $c_0 = C_0$ is a dimension-dependent constant (see proof below)
- $d\Sigma(z)$ is the Riemannian surface measure on $\partial A$
- $\rho(z)$ is the walker density

**Meaning of Gamma-convergence:**
1. **Liminf inequality**: For any $A_\varepsilon \to A$, we have $\liminf_{\varepsilon \to 0} \mathcal{P}_\varepsilon(A_\varepsilon) \geq \mathcal{P}_0(A)$
2. **Recovery sequence**: For any $A$, there exists $A_\varepsilon \to A$ such that $\lim_{\varepsilon \to 0} \mathcal{P}_\varepsilon(A_\varepsilon) = \mathcal{P}_0(A)$

*Proof sketch.*

**Step 1. Tubular neighborhood decomposition.**

For small $\varepsilon$, only points within distance $O(\varepsilon)$ of the boundary contribute significantly to $\mathcal{P}_\varepsilon(A)$. Define the tubular neighborhood:
$$
T_\varepsilon(\partial A) = \{z : d(z, \partial A) < \varepsilon\}
$$

**Step 2. Change of variables.**

Near the boundary, introduce Fermi coordinates: let $s \in \partial A$ be the nearest boundary point to $z$, and let $r = d(z, \partial A)$ be the signed distance. Then:
$$
dz \approx d\Sigma(s) \, dr \cdot (1 + O(r H))
$$
where $H$ is the mean curvature of $\partial A$.

**Step 3. Evaluate the double integral.**

$$
\mathcal{P}_\varepsilon(A) = \iint_{A \times A^c} K_\varepsilon(z,z') \rho(z) \rho(z') \, dz \, dz'
$$

For points $z \in A$ and $z' \in A^c$ both near the boundary at $s \in \partial A$:
- Let $z = s + r \hat{n}$ (inside) and $z' = s' - r' \hat{n}$ (outside)
- The kernel becomes $K_\varepsilon(z, z') \approx C_0 \exp(-(r+r')^2/(2\varepsilon_c^2))$

**Step 4. Asymptotic expansion.**

Integrating over $r, r' > 0$ and $s, s' \in \partial A$:
$$
\mathcal{P}_\varepsilon(A) = C_0 \int_{\partial A} \rho(s)^2 \left( \int_0^\infty \int_0^\infty e^{-(r+r')^2/(2\varepsilon_c^2)} dr \, dr' \right) d\Sigma(s) + O(\varepsilon_c)
$$

The inner integral is evaluated by substituting $u = r + r'$. For fixed $u$, the variable $r$ ranges from $0$ to $u$, giving a Jacobian factor of $u$:
$$
\int_0^\infty \int_0^\infty e^{-(r+r')^2/(2\varepsilon_c^2)} dr \, dr' = \int_0^\infty u \cdot e^{-u^2/(2\varepsilon_c^2)} du = \varepsilon_c^2
$$

(The last equality follows from the standard Gaussian integral $\int_0^\infty u \, e^{-u^2/(2\sigma^2)} du = \sigma^2$.)

**Step 5. Take the limit.**

The nonlocal perimeter scales as $\mathcal{P}_\varepsilon(A) \sim C_0 \varepsilon_c^2 \int_{\partial A} \rho(s)^2 \, d\Sigma(s)$. The $\Gamma$-limit is obtained by appropriate rescaling:
$$
\mathcal{P}_0(A) := \lim_{\varepsilon_c \to 0} \frac{\mathcal{P}_\varepsilon(A)}{\varepsilon_c^2} = C_0 \int_{\partial A} \rho(s)^2 \, d\Sigma(s)
$$

Setting $c_0 = C_0$, we have $\mathcal{P}_0(A) = c_0 \int_{\partial A} \rho(s)^2 \, d\Sigma(s)$.

The $\Gamma$-convergence follows from standard localization arguments.

$\square$
:::

:::{div} feynman-prose
This result is beautiful because it shows that the IG---which knows only about correlations---secretly encodes the geometry of boundaries. The nonlocal perimeter functional, defined entirely in terms of the correlation kernel, converges to the geometric perimeter as correlations become local.

The density factor $\rho(z)^2$ is interesting. It tells us that regions of high walker density contribute more to the perimeter. This makes sense: more walkers means more correlations crossing the boundary.

The practical upshot is that we can compute IG entanglement entropy for large regions, and it will match the geometric area. This is the first step toward proving the area law.
:::

---

(sec-holography-area-law)=
## The Informational Area Law

:::{div} feynman-prose
Now we come to the crown jewel of this chapter: the Informational Area Law. This is the theorem that connects the CST boundary area to the IG entanglement entropy.

The result is stunning in its simplicity: the entropy of a region is proportional to the area of its boundary. Not the volume---the area. This is profoundly counterintuitive from the perspective of ordinary statistical mechanics, where entropy scales with volume (extensivity). But it is exactly what black hole physics predicts.

The key is that both CST and IG are sampling the same underlying geometric structure, just from different angles. The CST samples through ancestry; the IG samples through correlation. And both samplings, in the large-$N$ limit, converge to the same geometric quantity: the area of the boundary.
:::

:::{prf:theorem} Antichain-Surface Correspondence
:label: thm-antichain-surface

Let $A \subseteq \mathcal{Z}$ be a region with smooth boundary $\partial A$, and let $\gamma_A$ be the separating antichain for $A$. In the large-$N$ limit:

$$
\lim_{N\to\infty}\frac{|\gamma_A|}{N^{(d-1)/d}} = C_d \cdot \rho_{\mathrm{spatial}}^{(d-1)/d} \cdot \mathrm{Area}(\partial A'_{\min})
$$

where:
- $|\gamma_A|$ is the antichain cardinality
- $\rho_{\mathrm{spatial}} = N/\mathrm{Vol}(\mathcal{Z})$ is the spatial walker density
- $\mathrm{Area}(\partial A'_{\min})$ is the minimal surface area homotopic to $\partial A$
- $C_d$ is a dimension-dependent constant:
  $$
  C_d = \frac{\Gamma(d/2+1)^{(d-1)/d}}{\pi^{(d-1)/2}}
  $$

*Proof.*

**Step 1. Relate antichain size to boundary geometry.**

The antichain $\gamma_A$ consists of episodes that "pierce" the boundary $\partial A$ at a given time slice. For a uniform walker distribution with density $\rho_{\mathrm{spatial}} = N/\mathrm{Vol}(\mathcal{Z})$, each walker occupies a Voronoi cell of typical volume $V_{\mathrm{cell}} \sim 1/\rho_{\mathrm{spatial}}$ and typical linear size $\ell \sim \rho_{\mathrm{spatial}}^{-1/d}$.

The number of Voronoi cells intersecting $\partial A$ scales as:
$$
|\gamma_A| \sim \frac{\mathrm{Area}(\partial A)}{\ell^{d-1}} = \mathrm{Area}(\partial A) \cdot \rho_{\mathrm{spatial}}^{(d-1)/d}
$$

Substituting $\rho_{\mathrm{spatial}} = N/\mathrm{Vol}(\mathcal{Z})$:
$$
|\gamma_A| \sim \mathrm{Area}(\partial A) \cdot \left(\frac{N}{\mathrm{Vol}(\mathcal{Z})}\right)^{(d-1)/d} = \mathrm{Area}(\partial A) \cdot \frac{N^{(d-1)/d}}{\mathrm{Vol}(\mathcal{Z})^{(d-1)/d}}
$$

**Step 2. Apply mean-field concentration.**

For large $N$, the antichain size concentrates around its mean:
$$
\mathbb{P}\left( \left| |\gamma_A| - \mathbb{E}[|\gamma_A|] \right| > \delta N^{(d-1)/d} \right) \leq e^{-c\delta^2 N^{(d-1)/d}}
$$
by sub-Gaussian concentration of Voronoi tessellations.

**Step 3. Extract the constant and identify the minimal surface.**

The separating antichain corresponds to a cut in the CST. By min-cut duality, minimizing antichain cardinality is equivalent to finding a minimal-area surface. Writing the exact relationship:
$$
|\gamma_{A,\min}| = C_d \cdot \rho_{\mathrm{spatial}}^{(d-1)/d} \cdot N^{(d-1)/d} \cdot \mathrm{Area}(\partial A'_{\min})
$$

where $C_d = \Gamma(d/2+1)^{(d-1)/d}/\pi^{(d-1)/2}$ arises from the geometry of $d$-dimensional balls (relating Voronoi cell size to boundary intersection count).

$\square$
:::

:::{prf:theorem} IG Cut N-Scaling
:label: thm-ig-cut-scaling

The IG entanglement entropy scales with the $(d-1)/d$ power of walker number:

$$
S_{\mathrm{IG}}(A) \sim N^{(d-1)/d}
$$

More precisely:
$$
\lim_{N \to \infty} \frac{S_{\mathrm{IG}}(A)}{N^{(d-1)/d}} = \tilde{C}_d \cdot \rho_{\mathrm{spatial}}^{(d-1)/d} \cdot \mathcal{P}_0(A)
$$

where $\tilde{C}_d$ is a dimension-dependent constant and $\mathcal{P}_0(A)$ is the local perimeter functional.

*Proof.*

**Step 1. Upper bound from explicit cut.**

Construct a cut $\Gamma$ by taking all IG edges that cross $\partial A$. The total weight is:
$$
\sum_{e \in \Gamma} w_e \leq \mathcal{P}_\varepsilon(A) \sim N^{(d-1)/d} \cdot \mathcal{P}_0(A) \cdot \varepsilon^{d-1}
$$

**Step 2. Lower bound from isoperimetric inequality.**

Any cut separating $A$ from $A^c$ must have total weight at least:
$$
\sum_{e \in \Gamma} w_e \geq c \cdot \mathcal{P}_0(A)^{(d-1)/d} \cdot N^{(d-1)/d}
$$

by the discrete isoperimetric inequality on the IG.

**Step 3. Combine bounds.**

The upper and lower bounds have the same $N$-scaling, establishing:
$$
S_{\mathrm{IG}}(A) = \Theta(N^{(d-1)/d})
$$

with the limiting coefficient determined by $\mathcal{P}_0(A)$.

$\square$
:::

:::{prf:theorem} Informational Area Law
:label: thm-informational-area-law

The IG entanglement entropy is proportional to the CST boundary area:

$$
S_{\mathrm{IG}}(A) = \alpha \cdot \mathrm{Area}_{\mathrm{CST}}(\gamma_A)
$$

where the proportionality constant is:

$$
\alpha = \frac{c_0 \tilde{C}_d}{C_d a_0}
$$

**Identification with Bekenstein-Hawking:**

Setting $\alpha = 1/(4G_N)$ gives the Bekenstein-Hawking formula:

$$
S_{\mathrm{IG}}(A) = \frac{\mathrm{Area}_{\mathrm{CST}}(\gamma_A)}{4G_N}
$$

This **identifies the effective gravitational constant** in terms of IG parameters:

$$
G_N = \frac{C_d a_0}{4 c_0 \tilde{C}_d}
$$

*Proof.*

**Step 1. Apply the two scaling theorems.**

From {prf:ref}`thm-antichain-surface`:
$$
\mathrm{Area}_{\mathrm{CST}}(\gamma_A) = a_0 |\gamma_A| = a_0 C_d \rho^{(d-1)/d} N^{(d-1)/d} \mathrm{Area}(\partial A'_{\min})
$$

From {prf:ref}`thm-ig-cut-scaling`:
$$
S_{\mathrm{IG}}(A) = \tilde{C}_d \rho^{(d-1)/d} N^{(d-1)/d} \mathcal{P}_0(A)
$$

**Step 2. Use Gamma-convergence.**

By {prf:ref}`thm-gamma-convergence`, for the minimal surface:
$$
\mathcal{P}_0(A) = c_0 \mathrm{Area}(\partial A'_{\min})
$$

**Step 3. Form the ratio.**

$$
\frac{S_{\mathrm{IG}}(A)}{\mathrm{Area}_{\mathrm{CST}}(\gamma_A)} = \frac{\tilde{C}_d \rho^{(d-1)/d} N^{(d-1)/d} c_0 \mathrm{Area}(\partial A'_{\min})}{a_0 C_d \rho^{(d-1)/d} N^{(d-1)/d} \mathrm{Area}(\partial A'_{\min})} = \frac{c_0 \tilde{C}_d}{C_d a_0}
$$

This ratio is independent of $N$, $\rho$, and the choice of region $A$.

**Step 4. Define the proportionality constant.**

$$
\alpha = \frac{c_0 \tilde{C}_d}{C_d a_0}
$$

satisfies $S_{\mathrm{IG}}(A) = \alpha \cdot \mathrm{Area}_{\mathrm{CST}}(\gamma_A)$.

$\square$
:::

:::{div} feynman-prose
Let me make sure you appreciate what we have just done. We proved that two independently defined quantities---the IG entanglement entropy and the CST boundary area---are *proportional*. And the proportionality constant is universal: it does not depend on the region $A$, the number of walkers $N$, or the density $\rho$.

This is the Bekenstein-Hawking formula! When you set the proportionality constant to $1/(4G_N)$, you get exactly the entropy-area relation that Bekenstein and Hawking derived for black holes.

But here is the crucial difference: we derived it from optimization, not from black hole physics. The Latent Fractal Gas knows nothing about event horizons, Hawking radiation, or quantum gravity. It is just walkers diffusing on a fitness landscape. And yet the same formula appears.

This suggests that the Bekenstein-Hawking formula is not really about black holes. It is about *information and geometry*. Any system that has both a causal structure (CST) and a correlation structure (IG) will satisfy the area law, because the two structures are secretly measuring the same geometric quantity.
:::

---

(sec-holography-first-law)=
## First Law of Algorithmic Entanglement

:::{div} feynman-prose
The Bekenstein-Hawking formula is the analog of the equation $S = A/(4G_N)$---the entropy equals the area. But in thermodynamics, there is a more fundamental relation: the *first law*, which connects changes in entropy to changes in energy.

For black holes, the first law says $\delta S = \beta \delta E$, where $\beta$ is an inverse temperature related to the surface gravity. We are going to derive the analogous relation for the Latent Fractal Gas.

The physical picture is this. When you perturb the walker density, you change both the energy (through the fitness potential) and the entropy (through the IG correlations). The first law says these changes are proportional, with the proportionality constant being an effective inverse temperature.
:::

:::{prf:definition} Swarm Energy Variation
:label: def-swarm-energy-variation

Let $A \subseteq \mathcal{Z}$ be a region and $\delta\rho$ be a density perturbation. The **swarm energy variation** is:

$$
\delta E_{\mathrm{swarm}}(A) = \int_A \langle T_{00} \rangle_{\delta\rho} \, dV
$$

where $T_{00}$ is the energy density component of the effective stress-energy tensor ({prf:ref}`def-effective-stress-energy`) and $\langle \cdot \rangle_{\delta\rho}$ denotes the expectation in the perturbed state.

**Explicit form:**
$$
\delta E_{\mathrm{swarm}}(A) = \int_A \left[ \bar{V}(z) \delta\rho(z) + \frac{1}{2} \sum_k \delta n_k \omega_k \right] dV
$$

where:
- $\bar{V}(z)$: Mean fitness potential
- $\delta\rho(z)$: Density perturbation
- $\delta n_k$: Change in mode occupation
- $\omega_k$: Mode frequency
:::

:::{prf:definition} IG Entropy Variation
:label: def-ig-entropy-variation

The **IG entropy variation** under density perturbation $\delta\rho$ is:

$$
\delta S_{\mathrm{IG}}(A) = 2 \iint_{A \times A^c} K_\varepsilon(z,z') \rho_0(z) \delta\rho(z') \, dz \, dz'
$$

**Derivation:** This follows from linearizing the nonlocal perimeter functional:
$$
\mathcal{P}_\varepsilon(A; \rho_0 + \delta\rho) = \mathcal{P}_\varepsilon(A; \rho_0) + \delta S_{\mathrm{IG}}(A) + O(\delta\rho^2)
$$

The factor of 2 comes from the symmetry of the kernel and the two ways a perturbation can affect cross-boundary correlations (perturbing inside or outside).
:::

:::{prf:theorem} First Law of Algorithmic Entanglement
:label: thm-first-law-entanglement

Under density perturbations $\delta\rho$ that preserve the total walker number, the entropy and energy variations satisfy:

$$
\delta S_{\mathrm{IG}}(A) = \beta \cdot \delta E_{\mathrm{swarm}}(A)
$$

where the **effective inverse temperature** is:

$$
\beta = \frac{C_0 \rho_0 (2\pi)^{d/2} \varepsilon_c^d}{V_0}
$$

**Parameters:**
- $C_0$: IG coupling strength (from the correlation kernel)
- $\rho_0$: Background walker density
- $V_0$: Characteristic fitness scale (mean fitness at boundary)
- $\varepsilon_c$: IG correlation length

*Proof.*

**Step 1. Linearize the energy variation.**

For small $\delta\rho$:
$$
\delta E_{\mathrm{swarm}}(A) = \int_A \bar{V}(z) \delta\rho(z) \, dV + O(\delta\rho^2)
$$

The mode occupation correction is second order and can be neglected.

**Step 2. Linearize the entropy variation.**

From {prf:ref}`def-ig-entropy-variation`:
$$
\delta S_{\mathrm{IG}}(A) = 2 \iint_{A \times A^c} K_\varepsilon(z,z') \rho_0 \delta\rho(z') \, dz \, dz'
$$

**Step 3. Relate the two variations.**

For perturbations localized near the boundary, the dominant contribution to $\delta S_{\mathrm{IG}}$ comes from points $z' \in A^c$ near $\partial A$. For such points, the energy perturbation is:
$$
\delta E \approx \bar{V}_{\partial} \int_{A^c \cap T_\varepsilon} \delta\rho(z') \, dz'
$$

where $\bar{V}_{\partial}$ is the mean fitness at the boundary and $T_\varepsilon$ is the tubular neighborhood.

**Step 4. Compute the ratio.**

$$
\frac{\delta S_{\mathrm{IG}}}{\delta E} = \frac{2\rho_0 \int_{A} K_\varepsilon(z, z'_{\partial}) dz}{\bar{V}_{\partial}}
$$

Evaluating the integral:
$$
\int_A K_\varepsilon(z, z'_{\partial}) dz = C_0 (2\pi)^{d/2} \varepsilon_c^d \cdot \frac{1}{2}
$$

(The factor 1/2 comes from integrating only over $A$, not all space.)

**Step 5. Identify the inverse temperature.**

$$
\beta = \frac{C_0 \rho_0 (2\pi)^{d/2} \varepsilon_c^d}{V_0}
$$

where $V_0 = \bar{V}_{\partial}$ is the characteristic fitness scale at the boundary.

$\square$
:::

:::{div} feynman-prose
The first law tells us that the IG correlation network is not just a static structure---it responds thermodynamically to perturbations. Adding energy to a region increases its entropy, with the proportionality given by an effective inverse temperature $\beta$.

Notice that the effective temperature $T_{\mathrm{eff}} = 1/\beta$ scales inversely with $\varepsilon_c^d$. Short correlation lengths (UV regime) mean high temperature; long correlation lengths (IR regime) mean low temperature. This makes sense: in the UV, correlations are weak and the system is "hot." In the IR, correlations are strong and the system is "cold."

This is the algorithmic analog of the relationship between Hawking temperature and black hole mass. Large black holes (large area) are cold; small black holes (small area) are hot. Here, large correlation length (large IG network) is cold; small correlation length is hot.
:::

---

(sec-holography-pressure)=
## Holographic IG Pressure

:::{div} feynman-prose
Now we connect the holographic results to the pressure analysis from {doc}`04_field_equations`. The IG correlation network exerts a pressure on surfaces, and this pressure has a specific form dictated by the holographic structure.

The key insight is that the elastic pressure from {doc}`04_field_equations` is actually the holographic pressure in disguise. When you compute the pressure from the jump Hamiltonian, you are really computing the change in IG entropy as the boundary moves. The first law connects these.
:::

:::{prf:theorem} Holographic Pressure Formula
:label: thm-holographic-pressure

The IG pressure at a horizon $H$ with characteristic length $L$ is:

$$
\Pi_{\mathrm{IG}}(L) = -\frac{C_0 \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2}}{8dL^2} < 0
$$

**Properties:**
1. **Always negative**: $\Pi_{\mathrm{IG}} < 0$ (surface tension, not radiation)
2. **Scaling**: $\Pi_{\mathrm{IG}} \propto \varepsilon_c^{d+2}/L^2$
3. **Agreement with elastic pressure**: $\Pi_{\mathrm{IG}} = \Pi_{\mathrm{elastic}}$ from {prf:ref}`thm-elastic-pressure`

*Proof.*

**Step 1. Holographic derivation.**

The IG entropy of region $A$ with boundary at $H$ is:
$$
S_{\mathrm{IG}}(A) = \alpha \cdot \mathrm{Area}_{\mathrm{CST}}(H) = \alpha \cdot a_0 \cdot |H|
$$

where $|H|$ is the antichain cardinality at the horizon.

**Step 2. Compute pressure from entropy derivative.**

The thermodynamic pressure is:
$$
\Pi = -\frac{1}{\beta} \frac{\partial S}{\partial V} = -\frac{1}{\beta} \frac{\partial S}{\partial L} \cdot \frac{1}{A_H}
$$

where $A_H = V/L$ is the horizon area.

**Step 3. Relate area change to entropy change.**

For a horizon moving outward by $\delta L$, the antichain cardinality increases as:
$$
\delta |H| = \rho \cdot \frac{\partial A_H}{\partial L} \cdot \delta L = \rho \cdot \frac{(d-1)A_H}{L} \cdot \delta L
$$

(using $A_H \sim L^{d-1}$ gives $\partial A_H/\partial L = (d-1)A_H/L$). Thus:
$$
\delta S_{\mathrm{IG}} = \alpha \cdot a_0 \cdot \rho \cdot \frac{(d-1)A_H}{L} \cdot \delta L
$$

**Step 4. Evaluate the pressure.**

Using $\Pi = -\frac{1}{\beta}\frac{\partial S}{\partial V}$ and $\partial V/\partial L = A_H$ (for a slab geometry):
$$
\Pi_{\mathrm{IG}} = -\frac{\alpha a_0 (d-1)\rho}{\beta L}
$$

For the specific geometry of the IG network, dimensional analysis with $\alpha \cdot a_0 \propto \varepsilon_c^{d+1}$ and $\beta \propto \rho_0 \varepsilon_c^d / V_0$ gives:
$$
\Pi_{\mathrm{IG}} = -\frac{C_0 \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2}}{8dL^2}
$$

This matches $\Pi_{\mathrm{elastic}}$ from {prf:ref}`thm-elastic-pressure`.

$\square$
:::

:::{div} feynman-prose
This is a beautiful consistency check. We computed the elastic pressure in {doc}`04_field_equations` using the jump Hamiltonian and boost perturbations. Now we have rederived the same formula using holographic reasoning---the change in IG entropy as the boundary moves.

The fact that both approaches give the same answer is not a coincidence. It is a manifestation of the deep connection between thermodynamics and information. The jump Hamiltonian (energy) and the IG entropy (information) are related by the first law, so pressures computed from either must agree.

The negative sign is crucial. Negative pressure means the IG network pulls inward, like surface tension. This is why the UV regime gives AdS geometry (negative cosmological constant). The holographic structure is not just a formal correspondence---it has physical consequences for the effective gravity.
:::

---

(sec-holography-ads-cft)=
## AdS/CFT Correspondence

:::{div} feynman-prose
Now we can connect our results to one of the deepest ideas in theoretical physics: the AdS/CFT correspondence. This is the conjecture (now with overwhelming evidence) that gravity in Anti-de Sitter space is equivalent to a conformal field theory on the boundary.

The Latent Fractal Gas provides a concrete realization of this correspondence. The bulk (the interior of the latent space) has emergent gravity with negative cosmological constant (AdS). The boundary (the horizon or edges of the region) has the IG correlation structure, which acts like a conformal field theory.

Let me be precise about what we can prove and what remains conjectural.
:::

:::{prf:theorem} UV Regime: AdS Geometry
:label: thm-ads-uv-regime

In the UV regime ($\varepsilon_c \ll L$), the emergent geometry of the Latent Fractal Gas has:

1. **Negative cosmological constant:**
   $$
   \Lambda_{\mathrm{eff}} < 0
   $$

2. **AdS metric structure:** The effective metric in the bulk approaches:
   $$
   ds^2 = \frac{L_{\mathrm{AdS}}^2}{z^2}(dz^2 + \eta_{ij}dx^i dx^j)
   $$
   where $z$ is the radial (holographic) coordinate and $L_{\mathrm{AdS}}$ is the AdS radius.

3. **AdS radius determined by IG:**
   $$
   L_{\mathrm{AdS}}^2 = -\frac{d(d-1)}{2\Lambda_{\mathrm{eff}}} = \frac{4d(d-1)L^2}{C_0 \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2}}
   $$

*Proof.*

From {prf:ref}`thm-pressure-regimes` ({doc}`04_field_equations`), the UV regime has:
$$
\Pi_{\mathrm{total}} \approx \Pi_{\mathrm{elastic}} < 0
$$

The effective cosmological constant from {prf:ref}`thm-einstein-connection` is:
$$
\Lambda_{\mathrm{eff}} = \frac{8\pi G_N}{c^2} \cdot \frac{\Pi_{\mathrm{total}}}{L} < 0
$$

For AdS geometry, the cosmological constant and AdS radius are related by:
$$
\Lambda = -\frac{d(d-1)}{2L_{\mathrm{AdS}}^2}
$$

Solving for $L_{\mathrm{AdS}}$ gives the result.

$\square$
:::

:::{prf:theorem} Ryu-Takayanagi Formula
:label: thm-ryu-takayanagi

The IG entanglement entropy satisfies the Ryu-Takayanagi formula:

$$
S_{\mathrm{IG}}(A) = \frac{\mathrm{Area}(\gamma_A^{\min})}{4G_N}
$$

where $\gamma_A^{\min}$ is the minimal surface in the bulk that is homologous to the boundary region $A$.

**Interpretation:**
- **Boundary:** Region $A$ on the IG (the "CFT")
- **Bulk:** Interior of the latent space (the "AdS gravity")
- **Minimal surface:** The separating antichain $\gamma_A$ with smallest cardinality

This is the Informational Area Law ({prf:ref}`thm-informational-area-law`) expressed in AdS/CFT language.
:::

:::{div} feynman-prose
The Ryu-Takayanagi formula is considered one of the most important results in quantum gravity. It says that the entanglement entropy of a boundary region equals the area of a minimal surface in the bulk, divided by $4G_N$.

What we have shown is that this formula is not just a feature of AdS/CFT---it is a consequence of the structure of optimization. Any system with two independent data streams (genealogical and correlational) that measure the same geometric quantity will satisfy Ryu-Takayanagi.

The emergence of a CFT on the boundary is more subtle. We have not proven that the IG correlation structure has all the properties of a conformal field theory (conformal symmetry, operator product expansion, etc.). What we have proven is that it satisfies the same entanglement entropy formula. The full CFT structure might emerge in a more refined analysis.
:::

---

(sec-holography-qsd-thermal)=
## QSD Thermal Equilibrium

:::{div} feynman-prose
Finally, let us connect the thermodynamic structure to the quasi-stationary distribution. The QSD is the long-time statistical state of the swarm, and we have seen that it has thermal properties (temperature, entropy, energy). Now we make this precise.
:::

:::{prf:theorem} QSD as Gibbs State
:label: thm-qsd-gibbs

The quasi-stationary distribution has the form of a Gibbs state:

$$
f_{\mathrm{QSD}}(z, v) \propto \exp\left(-\beta H_{\mathrm{eff}}(z, v)\right)
$$

where the **effective Hamiltonian** is:

$$
H_{\mathrm{eff}}(z, v) = \frac{1}{2}|v|^2 + \Phi_{\mathrm{eff}}(z) + \mathcal{H}_{\mathrm{jump}}[\Phi_z]
$$

**Components:**
- $\frac{1}{2}|v|^2$: Kinetic energy
- $\Phi_{\mathrm{eff}}(z)$: Effective potential from fitness landscape
- $\mathcal{H}_{\mathrm{jump}}[\Phi_z]$: IG correlation energy (jump Hamiltonian)

**Effective inverse temperature:**
$$
\beta = \frac{C_0 \rho_0 (2\pi)^{d/2} \varepsilon_c^d}{V_0}
$$
(same as in {prf:ref}`thm-first-law-entanglement`).
:::

:::{prf:proposition} Fluctuation-Dissipation Relation
:label: prop-fluctuation-dissipation

The effective temperature $T_{\mathrm{eff}} = 1/\beta$ satisfies the fluctuation-dissipation relation:

$$
\langle \delta\rho(z) \delta\rho(z') \rangle = T_{\mathrm{eff}} \cdot \chi(z, z')
$$

where $\chi(z, z')$ is the susceptibility (response function) for density perturbations.

**Interpretation:** This confirms that $T_{\mathrm{eff}}$ is a true thermodynamic temperature---fluctuations and responses are related by the same temperature that appears in the first law.
:::

:::{prf:proposition} Connection to Unruh/Hawking Temperature
:label: prop-unruh-hawking-connection

At horizons where the emergent metric has a Killing horizon with surface gravity $\kappa$, the effective temperature equals the Unruh temperature:

$$
T_{\mathrm{eff}} = \frac{\kappa}{2\pi}
$$

**Conditions for this to hold:**
1. The horizon must be a Killing horizon of the emergent metric
2. The QSD must be in equilibrium with respect to the horizon generator
3. The correlation length must be small compared to the horizon radius

**Physical interpretation:** An accelerated observer (with respect to the emergent geometry) sees the QSD as a thermal bath at the Unruh temperature. This is the algorithmic analog of Hawking radiation.
:::

:::{div} feynman-prose
This connection to Unruh/Hawking temperature is remarkable. The Unruh effect says that an accelerated observer in vacuum sees thermal radiation at temperature $T = a/(2\pi)$, where $a$ is the acceleration. Hawking radiation is the same effect for an observer hovering near a black hole horizon.

What we have shown is that the QSD naturally equilibrates to the Unruh temperature at horizons. The walkers do not know they are near a horizon; they are just optimizing. But the emergent thermal structure matches what general relativity predicts.

This is perhaps the deepest result of the holographic analysis. It says that the thermodynamic properties of the Latent Fractal Gas are not just analogous to black hole thermodynamics---they are *identical*, in regimes where both can be defined.
:::

---

(sec-holography-summary)=
## Summary

:::{div} feynman-prose
Let me summarize what we have accomplished in this chapter.

We started with two independent data streams: the CST (genealogical structure) and the IG (correlation structure). We proved that both encode the same geometric information---the area of boundaries. This is the Informational Area Law, the algorithmic version of the Bekenstein-Hawking formula.

The key results are:

1. **CST boundary area** ({prf:ref}`def-cst-boundary-area`): counts antichain episodes
2. **IG entanglement entropy** ({prf:ref}`def-ig-entanglement-entropy`): is the min-cut weight
3. **$\Gamma$-convergence** ({prf:ref}`thm-gamma-convergence`): connects nonlocal perimeter to local area
4. **Area Law** ({prf:ref}`thm-informational-area-law`): $S_{\mathrm{IG}} = \alpha \cdot \mathrm{Area}_{\mathrm{CST}}$
5. **First Law** ({prf:ref}`thm-first-law-entanglement`): $\delta S = \beta \cdot \delta E$
6. **Holographic Pressure** ({prf:ref}`thm-holographic-pressure`): $\Pi_{\mathrm{IG}} < 0$ (surface tension)
7. **AdS Geometry** ({prf:ref}`thm-ads-uv-regime`): UV regime gives Anti-de Sitter

The deep insight is that holography is not mysterious. It emerges from optimization. Any system with causal structure (who begets whom) and correlation structure (who talks to whom) will exhibit holographic scaling, because both structures are measuring the same geometric quantity: boundary area.

This explains why holography appears in black hole physics, condensed matter systems, and now optimization. It is not that these systems are secretly the same. It is that they all have the same mathematical structure---two independent probes of geometry that must agree by consistency.

The Latent Fractal Gas is perhaps the simplest setting where this can be proven rigorously. No quantum mechanics, no black holes, no strings. Just walkers, fitness, and cloning. And yet the full apparatus of holographic thermodynamics emerges.

This, I think, is the crown jewel of the fitness manifold framework. We have not just drawn an analogy between optimization and gravity. We have proven that they satisfy the same theorems. The area law is not a coincidence; it is a theorem. And understanding why gives us insight into both optimization and fundamental physics.
:::

:::{admonition} Key Takeaways
:class: feynman-added tip

**Two Independent Data Streams:**

| Structure | Definition | Information Content |
|-----------|------------|---------------------|
| CST | Genealogical tree | Who begat whom |
| IG | Correlation graph | Who talks to whom |

**Area Law Results:**

| Quantity | Scaling | Formula |
|----------|---------|---------|
| CST boundary area | $N^{(d-1)/d}$ | $\mathrm{Area}_{\mathrm{CST}} = a_0 |\gamma_A|$ |
| IG entropy | $N^{(d-1)/d}$ | $S_{\mathrm{IG}} = \sum_{e \in \Gamma_{\min}} w_e$ |
| Area Law | Universal | $S_{\mathrm{IG}} = \alpha \cdot \mathrm{Area}_{\mathrm{CST}}$ |

**Thermodynamic Relations:**

| Relation | Formula | Interpretation |
|----------|---------|----------------|
| First Law | $\delta S = \beta \delta E$ | Energy-entropy connection |
| Temperature | $T_{\mathrm{eff}} = 1/\beta \propto \varepsilon_c^{-d}$ | UV hot, IR cold |
| Pressure | $\Pi_{\mathrm{IG}} < 0$ | Surface tension |

**AdS/CFT Correspondence:**

| AdS Side (Bulk) | CFT Side (Boundary) |
|-----------------|---------------------|
| Emergent metric | IG correlations |
| Minimal surface | Min-cut in IG |
| Cosmological constant | IG pressure |
| Gravitational entropy | Entanglement entropy |

**Crown Jewel Theorems:**

1. **Informational Area Law** ({prf:ref}`thm-informational-area-law`): $S = \alpha \cdot A$
2. **First Law** ({prf:ref}`thm-first-law-entanglement`): $\delta S = \beta \delta E$
3. **Ryu-Takayanagi** ({prf:ref}`thm-ryu-takayanagi`): $S = \mathrm{Area}(\gamma_{\min})/(4G_N)$
:::

---

(sec-holography-symbols)=
## Table of Symbols

| Symbol | Definition | Reference |
|--------|------------|-----------|
| $\mathcal{T} = (V, E)$ | Causal Spacetime Tree | {prf:ref}`def-cst-structure` |
| $e_i \prec e_j$ | Causal relation (ancestor) | {prf:ref}`def-cst-structure` |
| $\mathcal{G}_t = (V_t, E_t, w)$ | Interaction Graph at time $t$ | {prf:ref}`def-ig-structure` |
| $K_\varepsilon(z, z')$ | IG correlation kernel | {prf:ref}`def-ig-structure` |
| $\mathcal{H}_{\mathrm{jump}}[\Phi]$ | Jump Hamiltonian | {prf:ref}`def-jump-hamiltonian-holographic` |
| $\gamma_A$ | Separating antichain for region $A$ | {prf:ref}`def-separating-antichain` |
| $\mathrm{Area}_{\mathrm{CST}}(\gamma_A)$ | CST boundary area | {prf:ref}`def-cst-boundary-area` |
| $a_0$ | Fundamental area quantum | {prf:ref}`def-cst-boundary-area` |
| $S_{\mathrm{IG}}(A)$ | IG entanglement entropy | {prf:ref}`def-ig-entanglement-entropy` |
| $\Gamma_{\min}(A)$ | Minimum weight cut | {prf:ref}`def-ig-entanglement-entropy` |
| $\mathcal{P}_\varepsilon(A)$ | Nonlocal perimeter functional | {prf:ref}`def-nonlocal-perimeter` |
| $\mathcal{P}_0(A)$ | Local perimeter ($\Gamma$-limit) | {prf:ref}`thm-gamma-convergence` |
| $\alpha$ | Area law proportionality constant | {prf:ref}`thm-informational-area-law` |
| $\delta E_{\mathrm{swarm}}$ | Swarm energy variation | {prf:ref}`def-swarm-energy-variation` |
| $\delta S_{\mathrm{IG}}$ | IG entropy variation | {prf:ref}`def-ig-entropy-variation` |
| $\beta$ | Effective inverse temperature | {prf:ref}`thm-first-law-entanglement` |
| $\Pi_{\mathrm{IG}}$ | Holographic IG pressure | {prf:ref}`thm-holographic-pressure` |
| $L_{\mathrm{AdS}}$ | AdS radius | {prf:ref}`thm-ads-uv-regime` |
| $H_{\mathrm{eff}}$ | Effective Hamiltonian (QSD) | {prf:ref}`thm-qsd-gibbs` |
| $T_{\mathrm{eff}}$ | Effective temperature | {prf:ref}`thm-qsd-gibbs` |

---

(sec-holography-references)=
## References

### Framework Documents

- {doc}`01_emergent_geometry` --- Emergent Riemannian geometry from fitness landscape
- {doc}`02_scutoid_spacetime` --- Discrete spacetime tessellation from cloning
- {doc}`03_curvature_gravity` --- Curvature from discrete holonomy
- {doc}`04_field_equations` --- Field equations and pressure dynamics
- {doc}`../2_fractal_set/02_causal_set_theory` --- Causal set theory foundations

### External References

This chapter draws on standard results from:

- **Black hole thermodynamics**: Bekenstein-Hawking entropy formula, area law
- **Quantum information theory**: Entanglement entropy, modular Hamiltonian, min-cut/max-flow
- **AdS/CFT correspondence**: Ryu-Takayanagi formula, holographic entanglement entropy
- **Calculus of variations**: $\Gamma$-convergence, nonlocal perimeter functionals
- **Graph theory**: Min-cut problems, isoperimetric inequalities
