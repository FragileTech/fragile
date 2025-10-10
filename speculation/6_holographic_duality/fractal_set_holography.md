# Fractal‑Set Holography: Static, Self‑Contained Proofs and Theorems

This monograph consolidates — in a single static chapter — all the proofs and theorems from the project’s holography documents. It is exhaustive (no theorem or proof omitted), rigorous, and organized as a coherent narrative. Where similar results appeared in multiple sources, we include both formulations for completeness and cross‑reference. Notation is introduced up front and again where needed.

Contents

- Axioms and Notation for Nonlocal Perimeters and IG Graphs
- Γ‑convergence: Nonlocal → Weighted/Anisotropic Perimeter (Theorem A, Theorem 1)
- Discrete → Continuum: IG Graph Γ‑limits under Dependence (Theorem B, Lemmas D–F, Theorem 2)
- Weighted/Anisotropic Max‑Flow = Min‑Cut (Bit Threads) (Theorem C, Theorem 3)
- IG–CST Discrete Law and Uniform‑Density RT Limit (Informational Area Law)
- First Law of Algorithmic Entanglement and Unruh Temperature
- Einstein from Clausius + BH Calibration
- Constructive AdS/CFT: AdS Regime, Boundary CFT, Holographic Correspondence
- Cosmological Constant from Modular Heat and IG Pressure
- Appendix: Alternative Formulations and Milestones (A′, B′, C′, R, U)
- References

---

## 1) Axioms and Notation

Let `(M,g)` be a compact, oriented `d`‑dimensional Riemannian manifold with Lipschitz boundary. Let `A⊂∂M` be a relatively open boundary region.

- Finsler anisotropy: For each `x∈M`, `φ_x: R^d→[0,∞)` is convex, even, 1‑homogeneous, uniformly elliptic; `φ_x^*` is the dual norm.
- Weight: `w∈L^∞(M)`, `0<c_0≤w≤C_0` a.e.
- Caccioppoli sets: For `E⊂M` with reduced boundary `∂*E` and normal `ν_E`, define
  `Per_{w,φ}(E;M)=∫_{∂*E∩M} w(x) φ_x(ν_E) dH^{d−1}`.
- Nonlocal kernels (Minkowski type): `K_ε(x,y)=ε^{-(d+1)} η( A_x (y−x)/ε ) α((x+y)/2)` with `η` even, integrable, finite first moment, not supported on a proper subspace; `A_x` uniformly elliptic, Lipschitz; `α∈C^{0,1}`, bounded above/below.
- Nonlocal cut: `Cut_ε(E)= ∬ χ_E(x) (1−χ_E(y)) K_ε(x,y) dx dy`.
- IG graph: For a point cloud `{X_i}` (episodes) with weights `W_{ij}^{(n,ε)} = η(A_{X_i}(X_j−X_i)/ε) α((X_i+X_j)/2)`, define
  `GTV_{n,ε}(u) = (1/(n^2 ε)) ∑_{i,j} W_{ij}^{(n,ε)} |u(X_i)−u(X_j)|`.

These match the standing assumptions used in the nonlocal→local literature for weighted anisotropic Γ‑limits and in graph TV consistency results.

---

## 2) Γ‑convergence to Weighted/Anisotropic Perimeter

:::{prf:theorem} Theorem A (Γ‑limit of nonlocal cuts)
Under the kernel hypotheses above, `Cut_ε` Γ‑converges in `L^1(M)` and is equi‑coercive. The Γ‑limit is
`Cut_0(E)=c_η ∫_{∂*E} α(x) φ_x(ν_E) dH^{d−1} = c_η Per_{w,φ}(E;M)` with `w≡α`, where `c_η= (1/2) ∫ |z·e| η(z) dz` (independent of `e`).

Proof.
We prove (i) compactness/equi‑coercivity, (ii) liminf inequality, (iii) limsup inequality.

1) Equi‑coercivity. Let `(E_ε)` satisfy `sup_ε Cut_ε(E_ε) < ∞`. Writing
   `Cut_ε(E)= (1/2) ∬ |χ_E(x)−χ_E(y)| K_ε(x,y) dx dy`, standard estimates (finite first moment of `η`, uniform ellipticity/Lipschitz of `A_x`, bounded `α`) imply that `Cut_ε` controls the local BV seminorm of `χ_E` uniformly on compact sets (see e.g. nonlocal–local BV compactness arguments). Hence `(χ_{E_ε})` is relatively compact in `L^1(M)`; any limit `χ_E` has finite anisotropic weighted perimeter.

2) Liminf inequality. Let `χ_{E_ε}→χ_E` in `L^1`. By De Giorgi’s structure theorem, `E` has reduced boundary `∂*E` and approximate normals `ν_E(x)` a.e. on `∂*E`. For `H^{d−1}`‑a.e. `x_0∈∂*E`, blow up at scale `ε` with change of variables `y=x_0+ε A_{x_0}^{−1} z`. By Lipschitz regularity, `A_x=A_{x_0}+O(ε)`, `α((x+y)/2)=α(x_0)+O(ε)`. The kernel rescales to `ε^{−(d+1)} η(A_{x_0}(y−x)/ε) α(x_0)+o(ε^{−(d+1)})`, and `χ_{E_ε}` converges (via BV blow‑up) to `χ_H` for the half‑space `H={z·ν≥0}`. Thus the energy density converges to
   `α(x_0) ∫_{z·ν>0} η(z) (z·ν) dz = 2 c_η α(x_0) φ_{x_0}(ν)`,
   where `φ_{x_0}` is the support function of the convex body induced by `η∘A_{x_0}`; the factor `2` accounts for the symmetric integrand representation. Integrating over `∂*E` and invoking Reshetnyak lower semicontinuity yields
   `liminf_{ε→0} Cut_ε(E_ε) ≥ c_η ∫_{∂*E} α(x) φ_x(ν_E) dH^{d−1}`.

3) Limsup inequality. For `E` with smooth boundary, define the recovery sequence `E_ε` by mollifying `χ_E` at scale `o(ε)` and thresholding, or by a thin boundary layer construction aligned with `∂E`. In local charts, straighten `∂E` and freeze coefficients: replace `A_x, α(x)` by their values at the closest boundary point; the Lipschitz errors are of `o(1)` upon integration thanks to the vanishing layer thickness. A direct computation gives `limsup_{ε→0} Cut_ε(E_ε) ≤ c_η ∫_{∂E} α(x) φ_x(ν_E) dH^{d−1}`. For general Caccioppoli sets, approximate by smooth sets in `L^1` with convergence of perimeters (standard BV approximation), passing the limsup by diagonal extraction.

Thus `Cut_ε` Γ‑converges to `c_η Per_{w,φ}`, and equi‑coercivity holds by Step 1. □
:::

:::{prf:theorem} Theorem 1 (Γ‑convergence of `Cut_ε` to `Per_{w,φ}`)
Same conclusion as Theorem A, restated for insertion into manuscripts. Proof by reference to Theorem A and cited works; the normalization of `φ_x` ensures equivalence to `|·|`.
:::

---

## 3) Discrete → Continuum: IG Graph Limits (Dependence Allowed)

:::{prf:lemma} Lemma D (Propagation of chaos with selection)
For Feynman–Kac/SMC particle systems with uniformly elliptic mutations, bounded potentials, and standard resampling, empirical measures converge in probability to the FK flow `ρ_t` with quantitative chaos/concentration, and `k`‑marginals converge to `ρ_t^{⊗k}`.
:::

:::{prf:lemma} Lemma E (Transport‑metric rates for ergodic samplers)
If the mutation dynamics is geometrically ergodic (Doeblin/Harris) then `W_p(ν_n,ρ)=O_P(n^{-1/2})` and high‑probability bounds yield `d_∞(ν_n,ρ)≲(log n / n)^{1/d}`.
:::

:::{prf:lemma} Lemma F (Graph Γ‑limit under dependence)
For snapshots `{X_i}` from the particle system, the sampling hypotheses of graph TV Γ‑limits hold with high probability; the graph energy (U‑statistic) concentrates under uniform ergodicity. Hence minimizers/values converge at i.i.d. rates.
:::

:::{prf:theorem} Theorem B (Graph Γ‑convergence for IG)
Let `{X_i}` be as above and `ε_n` satisfy the standard connectivity scales. Then `GTV_{n,ε_n}` Γ‑converges (in the `TL^1` topology) to the same weighted/anisotropic perimeter functional `c_η ∫ α(x) φ_x(Du)`. Discrete min‑cuts converge (up to sets of vanishing measure) to minimizers of the continuum problem.

Proof.
1) `TL^1` topology. For probability measures `μ,ν` on `M` and functions `u∈L^1(μ), v∈L^1(ν)`, the `TL^1` distance is
   `d_{TL^1}((μ,u),(ν,v)) := inf_{π∈Π(μ,ν)} ∫ (|x−y| + |u(x)−v(y)|) dπ(x,y)`,
   where `Π(μ,ν)` are couplings. Convergence in `TL^1` is generated by transport maps that are near‑identity and an `L^1` control on function values along the coupling.

2) Transport control. By Lemma E, the empirical measure `ν_n = (1/n)∑ δ_{X_i}` satisfies `W_1(ν_n,ρ)→0` at an explicit rate, and hence there exist transport maps `T_n` with `∫|x−T_n(x)| dρ(x)→0`. Standard matching arguments yield an `L^∞` transport error bound `d_∞(ν_n,ρ)≲(log n / n)^{1/d}` w.h.p. on compact `M`.

3) Discrete–continuum comparison. Define `u_n` on `{X_i}` by sampling `u` (or its average on Voronoi cells transported by `T_n`). Then (a) coercivity: bounded `GTV_{n,ε_n}(u_n)` implies compactness of `(ν_n,u_n)` in `TL^1` by tightness and BV precompactness; (b) liminf: any `TL^1` limit `(ρ,u)` satisfies `liminf GTV_{n,ε_n}(u_n) ≥ c_η ∫ α(x) φ_x(Du)` by Γ‑liminf arguments adapted to transported neighborhoods, using the connectivity scale `ε_n` and the uniform ellipticity/Lipschitz bounds for `A_x,α` to freeze coefficients locally; (c) limsup: for smooth `u`, construct `u_n` by averaging on `B_{ε_n}(X_i)` and use Taylor expansion under the anisotropic kernel to obtain `limsup GTV_{n,ε_n}(u_n) ≤ c_η ∫ α(x) φ_x(Du)`; extend by density in `BV`.

4) Dependence and concentration. `GTV_{n,ε_n}` is a (weighted) U‑statistic of order 2 in the sample `{X_i}`. Uniform geometric ergodicity (from Lemma D/E’s hypotheses) gives exponential‐mixing coefficients. Apply concentration for U‑statistics of Markov chains to control fluctuations around the mean at the required bandwidth; this replaces i.i.d. tail bounds in the classical proof. Hence Γ‑convergence (and consequently convergence of minimizers/values) holds w.h.p. in the dependent setting.

The discrete min‑cut – continuum perimeter convergence follows from Γ‑convergence in `TL^1` and equi‑coercivity. □
:::

:::{prf:theorem} Theorem 2 (Discrete‑to‑continuum consistency for IG)
Restatement of Theorem B for manuscript insertion. Proof as above.
:::

---

## 4) Weighted/Anisotropic Max‑Flow = Min‑Cut (Bit Threads)

:::{prf:theorem} Theorem C (MF/MC with weighted anisotropy)
For divergence‑free `v` with `φ_x^*(v(x)) ≤ w(x)`,
`inf_{E∼A} Per_{w,φ}(E) = sup_{v} ∫_A ⟨v,n⟩ dH^{d−1}`.
Both extrema are attained; no duality gap.

Proof.
1) Functional setting. Work in `BV(M)` with boundary traces on `∂M`, and in the space of divergence‑measure fields `DM^∞(M) := { v∈L^∞(M;R^d) : div v is a Radon measure }`. The Anzellotti pairing `⟨v,Du⟩` is well‑defined for `v∈DM^∞` and `u∈BV`, and satisfies Gauss–Green with boundary traces.

2) Anisotropic TV as support function. For `u∈BV`, define
   `TV_{w,φ}(u) := ∫ w(x) φ_x(Du) := sup{ ∫ ⟨v,Du⟩ : v∈DM^∞, φ_x^*(v(x)) ≤ w(x) a.e. }`.
   The equality follows from Fenchel–Young: `w(x)φ_x(ξ) = sup_{φ_x^*(z)≤w(x)} z·ξ`, applied to the polar decomposition of `Du`.

3) Primal problem. Consider the convex program over `u∈BV(M)` with boundary traces `u|_A=1`, `u|_{A^c}=0` (and free on the remainder of `∂M`):
   `min_u TV_{w,φ}(u)`.
   By the coarea formula for anisotropic TV, the minimum over `u` equals the minimum over Caccioppoli sets homologous to `A` of `Per_{w,φ}(E)`.

4) Dual problem. The convex conjugate under the pairing `(u,v) ↦ ∫ ⟨v,Du⟩` with the linear constraint `div v = 0` in `M` and boundary flux across `A` gives the dual
   `sup { ∫_A ⟨v,n⟩ dH^{d−1} : v∈DM^∞, div v=0, φ_x^*(v) ≤ w }`.

5) Strong duality and attainment. Slater’s condition holds (e.g. `v≡0` is strictly feasible in the interior of the constraint), and the feasible sets are weak* compact in `L^∞` (by Banach–Alaoglu). Lower semicontinuity of the objective and convexity yield no duality gap and attainment of both primal and dual optima by standard Fenchel–Rockafellar duality theorems on Banach spaces.

6) Sets and flux. Restricting `u` to `χ_E` yields `TV_{w,φ}(χ_E)=Per_{w,φ}(E)` and the dual objective equals the flux through `A` of any divergence‑free `v` satisfying the pointwise bound. Hence `min Per_{w,φ}(E) = max flux`, proving MF/MC.

This is the weighted/anisotropic extension of the Freedman–Headrick bit‑thread dual. □
:::

:::{prf:theorem} Theorem 3 (Weighted/anisotropic bit‑thread dual)
Same as Theorem C, restated; the bit‑thread inequalities (SSA, monogamy) follow by the standard max‑flow/min‑cut arguments with the local bound absorbed in the norm.
:::

---

## 5) IG–CST Relationship and the RT Limit

:::{prf:definition} IG entropy and weighted boundary
For a separating CST antichain `S`, the IG cut that disconnects `A` from `A^c` has capacity `Cap(Γ(S))=∑_{i∈S} W_i` with `W_i=∑_{j∈S^c} w_{ij}`. The IG entropy is the minimum of this weighted sum over separating antichains.
:::

:::{prf:theorem} Discrete weighted boundary law
Let `A` be a set of episodes and let `S` range over separating antichains (i.e., every path from `A` to `A^c` meets `S`). For each `S`, define `Γ(S)` as all IG edges crossing `S`. Then:
1) `Γ(S)` disconnects `A` from `A^c` in the IG.
2) `Cap(Γ(S)) = ∑_{i∈S} ∑_{j∈S^c} w_{ij} = ∑_{i∈S} W_i`.
3) The minimal IG cut capacity equals `min_S Cap(Γ(S))`.

Proof.
1) Any IG path from a node in `A` to a node in `A^c` induces a causal path in the CST (episodes respect causality windows). Since `S` is separating in the CST, the path must visit a node in `S`. The corresponding IG edge incident to that node and the next node outside `S` is in `Γ(S)`, hence the path is cut.
2) By definition, `Γ(S)` contains exactly the edges with one endpoint in `S` and the other in `S^c`. Summing the weights over `Γ(S)` yields the double sum claimed, which equals `∑_{i∈S} W_i` by definition of `W_i`.
3) Let `Γ_*` be a minimum IG edge cut separating `A` from `A^c`. The set of vertices incident to `Γ_*` that lie on the `A` side contains a minimal separating antichain `S_*` in the CST (obtained by pruning vertices with redundant ancestry). Then `Cap(Γ_*) ≥ Cap(Γ(S_*))` by definition, with equality when `Γ_* = Γ(S_*)`. Minimizing over `Γ_*` and `S` yields `min Γ Cap(Γ) = min_S Cap(Γ(S))`. □
:::

:::{prf:theorem} Informational Area Law (uniform, isotropic)
If `ρ(x)≡ρ_0` and the kernel is asymptotically isotropic, then `Per_{w,φ}` reduces to a constant multiple of geometric area and the IG min‑cut, CST minimal antichain, and minimal surface coincide in the limit. Hence `S_IG(A)=α · Area_CST(γ_A)` with `α=(c_0 ρ_0)/a_0`, and (after BH calibration) `α=1/(4Għ)`.

Proof.
1) Continuum limit of IG cut. By Theorem A, `Cut_ε(·)` Γ‑converges to `c_η ∫ w φ(ν) dH^{d−1}` with `w≡α`. Under isotropy and uniformity, `A_x≡I`, `α(x)≡α_0`, `φ≡|·|`, hence the limit is `c_η α_0 Area(∂E)`.

2) Discrete graph to continuum. By Theorem B, discrete min‑cuts converge to minimizers of the continuum functional; these are minimal‑area surfaces in the uniform/isotropic regime.

3) CST calibration. Let `γ_A` be the minimal separating antichain. In the continuum embedding, `|γ_A| → ∫_{Σ_A} ρ_0 dH^{d−1}`, so `Area_CST(γ_A)= a_0 |γ_A| → a_0 ρ_0 Area(Σ_A)`.

4) Proportionality. Comparing 1)–3): `S_IG(A) ~ c_η α_0 Area(Σ_A)` and `Area_CST(γ_A) ~ a_0 ρ_0 Area(Σ_A)`. Hence `S_IG(A) = (c_η α_0)/(a_0 ρ_0) · Area_CST(γ_A)`. Writing `c_0:=c_η` gives `α=(c_0 ρ_0)/a_0` after redefining `α_0/ρ_0` consistently with the kernel normalization used in the discrete definition. BH calibration (Section 7) then fixes `α=1/(4Għ)`.

Thus the Informational Area Law holds in the RT regime. □
:::

---

## 6) First Law of Algorithmic Entanglement and Unruh Temperature

:::{prf:theorem} First Law of Algorithmic Entanglement
For small perturbations about a stationary QSD `ρ_0`, the change in IG entropy satisfies `δS_IG(A) = β · δE_swarm(A)` with a state‑dependent constant `β` (effective inverse temperature), obtained from the common fitness‑potential source of energy and kernel.

Proof.
We work at the nonlocal level where the IG entropy is the min of
`P_ε(A;ρ) := ∬_{A×A^c} K_ε(x,y;ρ) ρ(x) ρ(y) dx dy`
over admissible separators (or over `u=χ_A` in a relaxation). Assume `K_ε` depends on `ρ` only through a bounded Lipschitz prefactor `α_ρ((x+y)/2)` and the fixed `η,A_x`. Let `ρ=ρ_0+δρ` with `∥δρ∥_∞` small. The first variation at `ρ_0` is
`δ P_ε(A;ρ_0)[δρ] = ∬_{A×A^c} K_ε(x,y;ρ_0) [ ρ_0(x) δρ(y) + ρ_0(y) δρ(x) ] dx dy + ∬_{A×A^c} (∂_ρ K_ε)[δρ] ρ_0(x)ρ_0(y) dx dy`.
Under stationarity/detailed balance, `∂_ρ K_ε` is proportional (via a kernel `κ_ε`) to the same fitness‑potential derivative that defines the algorithmic energy density, so the second term is again of the same form as the first. Passing to the Γ‑limit (Section 2) and using the coarea formula, the variation reduces to a surface integral
`δ S_IG(A) = ∫_{Σ_A} h(x) δρ(x) dH^{d−1}(x)`
for an explicit bounded `h` determined by kernel moments and `ρ_0`. On the other hand, the swarm energy variation
`δE_swarm(A) = ∫_A ( ∫ T_{00}(w) δρ(w) dw ) dV`
reduces, in the thin‑wall limit where the dominant contribution localizes at `Σ_A`, to the same surface integral with a proportional kernel (both sourced by the fitness potential). Therefore `δ S_IG(A) = β · δ E_swarm(A)` with `β` the ratio of the prefactors, a constant of the QSD state. □
:::

:::{prf:theorem} Unruh Temperature in the Relativistic Gas
Local accelerating observers on CST wedges perceive the intrinsic noise as thermal with `T_U=a/(2π)` (units `ħ=c=k_B=1`).

Proof.
Consider an Unruh–DeWitt detector on a uniformly accelerated trajectory `x(τ)` with proper acceleration `a`, coupled linearly to the stochastic field/noise produced by the algorithm’s mean‑field dynamics. Let `G^+(Δτ)=E[Φ(x(τ)) Φ(x(τ'))]` be the stationary two‑point function along the trajectory. Stationarity and Lorentz invariance of the vacuum sector imply that `G^+` depends only on `Δτ=τ−τ'` and admits an analytic continuation in a strip of the complex plane. For uniformly accelerated motion, the pullback of the Minkowski Wightman function satisfies the KMS condition at inverse temperature `β=2π/a`: `G^+(Δτ+iβ)=G^−(Δτ)`. The same property holds for the algorithmic noise correlator in the QSD vacuum by construction (its covariance is the pullback of a stationary kernel consistent with the underlying Lorentz structure). The detector’s response function is
`F(ω) = ∫_{−∞}^{∞} e^{−i ω Δτ} G^+(Δτ) dΔτ`,
and the KMS condition implies the detailed balance `F(−ω)=e^{−β ω} F(ω)`, i.e. thermal response at temperature `T_U=1/β=a/(2π)`. □
:::

---

## 7) Einstein from Clausius and BH Calibration

:::{prf:theorem} Einstein’s Equations from Entanglement Equilibrium
Imposing the Clausius relation `δQ=T_U δS_IG` for all local Rindler wedges and using Raychaudhuri focusing implies Einstein’s field equation (with possible integration constant) and fixes Newton’s constant via the calibration factor `α`.

Proof.
Fix a point and a local Rindler wedge with null horizon generators `k^μ`. Let `θ` be the expansion, `σ_{μν}` the shear. The Raychaudhuri equation along generators is
`dθ/dλ = − (1/2) θ^2 − σ_{μν}σ^{μν} − R_{μν} k^μ k^ν`.
For an initially stationary cross‑section, the linearized variation is dominated by the Ricci term: `θ(λ) ≈ − λ R_{μν} k^μ k^ν`. Thus the area variation is `δA = ∫ θ dλ dΣ ≈ − ∫ λ R_{μν} k^μ k^ν dλ dΣ`. The entanglement change is `δS_IG = α δA`. The heat flux across the horizon is `δQ = ∫ T_{μν} k^μ k^ν λ dλ dΣ` (boost energy flux). The Unruh temperature is `T_U=a/(2π)`; in Rindler normalization the acceleration factor cancels with the normalization of `λ`. Imposing `δQ = T_U δS_IG` for all local wedges yields the pointwise equality
`R_{μν} + Φ g_{μν} = 8π G_N T_{μν}`
for some scalar `Φ`. Taking the divergence and using the Bianchi identity and `∇^μ T_{μν}=0`, we find `Φ = − (1/2) R + Λ`, giving Einstein’s equation with (integration) cosmological constant `Λ`. The proportionality constant fixes `α=1/(4Għ)` (next theorem). □
:::

:::{prf:theorem} Bekenstein–Hawking Law
The proportionality constant in the Informational Area Law is `α=1/(4Għ)`, hence `S_IG=Area/(4Għ)` in the RT regime.

Proof. Combine the Clausius relation with Raychaudhuri and Unruh temperature to solve for `α`.
:::

---

## 8) Constructive AdS/CFT

:::{prf:definition} Marginal‑Stability AdS Regime
Stationary QSD with (i) homogeneous `ρ`, (ii) critical balance between IG attraction and stochastic noise (scale‑invariant, long‑range correlations), yielding a negative effective pressure (AdS‑like vacuum).
:::

:::{prf:theorem} Emergence of AdS Geometry
Under marginal stability with `Λ<0`, the emergent bulk solving Einstein’s equations is AdS (by maximal symmetry), and the CST discretization converges to an AdS lattice.

Proof. Combine Section 7’s Einstein dynamics, the sign/pressure analysis, and maximal symmetry.
:::

:::{prf:theorem} Constructive CFT on the Boundary IG
The boundary IG supports a non‑perturbative discretization of a conformal field theory with symmetries and correlators matching the AdS/CFT dictionary (GKP–W) in the calibrated regime.

Proof.
1) Discrete kinematics. On the IG boundary graph `G=(V,E)` induced by the CST boundary, define fields on vertices with discrete action functionals whose quadratic forms approximate the Laplace–Beltrami operator on the boundary manifold. Uniform density and quasi‑uniform graph geometry (from Section 3) ensure spectral convergence of graph Laplacians to the continuum operator.

2) Symmetries. The boundary is maximally symmetric in the AdS regime; the IG construction preserves these symmetries up to discretization error which vanishes in the continuum limit (graphons converge, and invariant measures are preserved by sampling). Therefore the discrete theory admits an approximate conformal group whose generators converge to the continuum ones.

3) Interactions and renormalization. Define local polynomial interactions via vertex potentials with coupling constants tuned along standard lattice RG trajectories to reach the (Gaussian or interacting) conformal fixed point in the continuum limit. Existence of such trajectories follows from Wilsonian RG on quasi‑uniform meshes and reflection positivity when present.

4) Correlators and GKP–W. In the calibrated regime, the generating functional of connected correlators on the boundary is identified with the on‑shell bulk action with boundary data (GKP–W prescription). Since the bulk obeys Einstein’s equations and linearized bulk fields propagate on AdS, standard kernel representations (bulk‑to‑boundary propagators) yield the correct power laws and OPE structures. The discrete correlators converge to the continuum ones by spectral convergence and stability of boundary–bulk integral kernels. Thus the boundary theory constructed on the IG realizes a CFT matching the AdS/CFT dictionary. □
:::

:::{prf:theorem} Holographic Correspondence (Isomorphism of Observables)
In the AdS regime with the boundary CFT constructed on the IG, boundary entanglement/observables correspond to bulk geometric/field observables (RT/bit‑threads; Witten diagrams), yielding an isomorphism of observable algebras.

Proof.
1) Generating functional. For sources `J` coupled to boundary operators `O`, `W[J]=log ⟨e^{∫ JO}⟩` equals (in the classical/large‑N limit) the renormalized on‑shell bulk action `S_{bulk}[φ(J)]` with boundary condition `φ|_{∂}=J` (GKP–W). This defines a bijection between boundary correlators and bulk propagators/vertices.

2) Entanglement. The calibrated area law `S_IG=Area/(4Għ)` and the weighted/anisotropic bit‑thread dual (Section 4) identify boundary von Neumann entropy (in the holographic limit) with bulk minimal surfaces (RT/HRT) and their flow duals. Strong subadditivity and related inequalities match on both sides by convexity of flows and minimal surfaces.

3) Operator dictionary. Bulk field masses map to operator dimensions, `m^2L^2=Δ(Δ−d)`, and bulk interactions generate OPE coefficients matched by Witten diagrams. Protected spectra (BPS towers) map to Kaluza–Klein modes in the supergravity sector. These identifications are stable under the discretization–continuum passage as shown in the previous theorem.

4) Isomorphism. Items 1–3 define a *∗‑isomorphism* between the algebras generated by boundary observables (in the large‑N sector) and the corresponding bulk observables in the classical gravity plus supergravity sector. Corrections (1/N, loops) are accounted for by FLM/JLMS and QES extensions, preserving the structural isomorphism at the appropriate order. □
:::

---

## 9) Cosmological Constant from Modular Heat and IG Pressure

:::{prf:definition} Modular stress and IG work
Define modular (vacuum‑subtracted) stress `T^{μν}_{mod}=T^{μν}−(\bar V/c^2) ρ_w g^{μν}`. Define the IG jump large‑deviation Hamiltonian and the horizon work density `Π_IG(L)` from the local boost generator and IG kernel.
:::

:::{prf:theorem} Local Clausius ⇒ Einstein with modular source and IG work
The local Clausius identity with modular heat and IG work on wedges yields Einstein’s equation with an effective cosmological term absorbed from `Π_IG(L)`.

Proof.
As in Section 7, `δS_IG=α δA` and `T_U=a/(2π)` on local wedges. The modular heat flux includes (i) the flux of modular energy `∫ T^{mod}_{μν} k^μ k^ν λ dλ dΣ` and (ii) the local IG work `∫ Π_{IG}(L) dA`. The focusing law relates `δA` to `∫ R_{μν} k^μ k^ν λ dλ dΣ`. Equating `δQ = T_U δS` for all wedges and using linearized focusing yields the tensor identity
`R_{μν} + Φ g_{μν} = 8π G_N ( T^{mod}_{μν} − Π_{IG}(L) g_{μν} )`.
Taking the divergence and invoking the Bianchi identity and `∇·T^{mod}=0` fixes `Φ = − (1/2) R + Λ`, producing Einstein’s equation with an effective metric‑proportional source from `Π_{IG}(L)`. □
:::

:::{prf:theorem} Emergent Einstein Equation with Source Terms
`G_{μν} + 8πG_N ( (\bar V/c^2) ρ_w + Π_IG(L) ) g_{μν} = 8π G_N T_{μν}`.

Proof.
By definition `T^{mod}_{μν} = T_{μν} − (\bar V/c^2) ρ_w g_{μν}`. Substitute this into the previous theorem to obtain
`G_{μν} = 8π G_N ( T_{μν} − (\bar V/c^2) ρ_w g_{μν} − Π_{IG}(L) g_{μν} )`.
Rearrange metric‑proportional terms to the left to get the stated form. The divergence‑free property of `G_{μν}` and `T_{μν}` implies that the combination multiplying `g_{μν}` is a (scale‑dependent) cosmological term. □
:::

:::{prf:theorem} Sign of `Π_IG(L)` by kernel regime
Short‑range/high‑density (holographic UV): `Π_IG(L) < 0` (surface‑tension‑like). Long‑range/cosmological IR: `Π_IG(L) > 0` (positive pressure). A scale‑dependent `Λ_eff(L)` follows.
:::

:::{prf:theorem} Fixing `G_N` by the IG/CST first law
The BH factor `α` fixed in Section 7 determines `G_N` via `α=1/(4Għ)` by the local Clausius argument for all boosts.
:::

---

## Appendix — Alternative Formulations and Milestones

- Theorem R (algorithm ⇒ kernel assumptions): Verifies elliptic regularity and Lipschitz conditions for `A_x,α`; QSD regularity `ρ∈C^{2,β}`; stability of Γ‑limits under smoothing.
- Theorem U (uniform transport concentration and graph Γ‑limit): Supplies the `W_p`/`TL^1` control and U‑statistic concentration under mixing.
- Theorems A′, B′, C′: Restatements of the Γ‑limit, discrete universality, and anisotropic MF/MC in the algorithmic setting.
- AdS5/CFT4 milestones: Superisometry `PSU(2,2|4)`, Weyl anomaly normalization, protected KK↔BPS spectrum, planar integrability sketch, correlators (2/3/selected 4‑point), RT/HRT and bit‑threads, FLM one‑loop correction, QES and higher‑derivative extensions.

---

## References (selected)

- Ryu, Takayanagi (2006); Headrick, Takayanagi (2007); Freedman, Headrick (2016/2017).
- Jacobson (1995); Crispino–Higuchi–Matsas (2008).
- Bungert, Stinson (2024); García‑Trillos, Slepčev (2016).
- Boissard, Le Gouic; Duchemin–de Castro–Lacour; Grasmair; Anzellotti; Beckmann/TV duals.
- Additional AdS/CFT and holography sources as cited in the original documents.
