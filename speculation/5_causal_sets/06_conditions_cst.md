Below is a consolidated, rigorous defense that (i) formalizes the requirements your “fractal set” sampling algorithm must meet to be *equivalent (in law)* to the Poisson sprinkling used in Causal Set Theory (CST), (ii) shows that the partial order you build on top of the samples is a valid causal set with a faithful embedding, and (iii) sets precise conditions under which your added **Information Graph (IG)** is admissible and does *not* violate CST’s kinematical or symmetry constraints. Throughout, “equivalent” means *equal in distribution* to Poisson sprinkling at the scales of interest; this is the notion used in CST’s continuum approximation and “faithful embedding.” I include load‑bearing theorems with proof sketches and verifiable criteria, together with references in BibTeX at the end.

---

## 0) Background: what has to be matched

* **Causal set:** a locally finite partial order $(C,\prec)$ (reflexive/antisymmetric/transitive; local finiteness: every causal interval has finitely many elements). This is the kinematics of CST. ([Wikipedia][1])
* **Faithful embedding (continuum approximation):** An order‐preserving map $\Phi:C\hookrightarrow(M,g)$ such that the number of elements in any spacetime region $A\subset M$ is (approximately) Poisson with mean $\rho\,\mathrm{Vol}_g(A)$. Precise $(\delta,V_0)$-faithfulness formulations appear in CST topology papers. ([ar5iv][2])
* **Why Poisson:** To preserve Lorentz invariance *in distribution*, sprinkling must be a Poisson point process with intensity proportional to spacetime volume; regular lattices or finite‑valency “nearest‑neighbor” graphs break LI. ([World Scientific][3], [Wikipedia][1])
* **Order + number ⇒ geometry:** Up to a conformal factor, manifold causal structure is determined by order (Malament/HKMM); the conformal volume factor is supplied by counting elements (Poisson density). ([Scribd][4], [Wikipedia][5])
* **Dimension/geometry estimators:** Myrheim–Meyer and related estimators, recovery of topology via thickened antichains, and discrete curvature/action (Benincasa–Dowker). ([Wikipedia][1], [arXiv][6], [APS Journals][7])

Your algorithm therefore must:

1. produce a point process on $(M,g)$ that **is (or is arbitrarily close to)** a homogeneous Poisson point process (PPP);
2. induce the **manifold causal order** on those points and thereby a **locally finite poset**;
3. if additional IG structure is used, ensure it is **order‑ and label‑invariant** and **does not conflict with Lorentz invariance**, per the Bombelli–Henson–Sorkin (BHS) theorem. ([World Scientific][3])

---

## 1) From your algorithm to Poisson sprinkling

We model your sampler as a random locally finite subset $X\subset M$ (events/“episodes”), with counting variables $N(A)=\#(X\cap A)$ for Borel $A\subset M$. Write $\mathrm{Vol}$ for the spacetime volume measure on $(M,g)$.

### Assumptions A (minimal and checkable)

* **A1 (locality / Bell causality in the growth sense):** sampling in $A$ does not depend on updates in regions spacelike to $A$ (no superluminal influence). (This mirrors Rideout–Sorkin “Bell causality” used in sequential growth.) ([arXiv][8])
* **A2 (covariance / label‑invariance):** the law of $X$ is invariant under isometries of $(M,g)$ and under relabellings of elements (discrete general covariance). ([arXiv][8])
* **A3 (intensity proportional to volume):** $\mathbb{E}[N(A)]=\rho\,\mathrm{Vol}(A)$ for some constant $\rho>0$ (or $\rho(x)$ if inhomogeneous).
* **A4 (weak dependence or conditional independence):** conditional on the past light‑cone $\mathcal{J}^-(A)$, the *Papangelou* (reduced Campbell) conditional intensity in $A$ is almost surely constant $\rho$ (or depends only on position via $\rho(x)$), i.e. **no history‑dependent clustering** inside $A$.
* **A5 (local finiteness):** $N(A)<\infty$ a.s. for compact $A$ (guaranteed by bounded rate and A1).

> *Note*: A4 is satisfied by any sampler whose “birth” rate in each infinitesimal volume is a fixed $\rho$ and is **independent of the existing configuration inside that same volume** (history outside only enters via causal restriction), which is the natural continuous‑time counterpart of independent “coin‑tossing” in each cell.

### Theorem 1 (PPP characterization via Papangelou/conditional intensity)

If A1–A5 hold with **constant** Papangelou intensity $\rho$ (or $\rho(x)$ in the inhomogeneous case), then $X$ is a (homogeneous/in‑homogeneous) **Poisson point process** with intensity measure $\Lambda(A)=\int_A \rho\, d\mathrm{Vol}$.
*Sketch.* For spatial/space–time point processes, a process with deterministic Papangelou intensity $c(x,\xi)\equiv f(x)$ is Poisson with intensity $f$. This is a standard characterization (Georgii–Nguyen–Zessin/Last–Penrose). ([Cambridge University Press & Assessment][9], [Wiley Online Library][10])

**Practical test** (void probabilities): verify $ \Pr\{N(A)=0\}=e^{-\rho\,\mathrm{Vol}(A)}$ over a nested family of regions $A$; by Rényi’s theorem/PPP axiomatics this plus disjoint‑set independence fixes the PPP. ([Wikipedia][11])

### Theorem 2 (approximate PPP under weak dependence: Stein–Chen bounds)

If births in disjoint space–time microcells are only *weakly* dependent (finite dependency graph; uniformly small marginal probabilities), then for any fixed finite union $A$ of microcells, the law of $N(A)$ is within computable total‑variation distance of $\mathrm{Poisson}(\rho\mathrm{Vol}A)$; the error decays with the dependency degree and cell size.
*Sketch.* Apply Stein–Chen (dependency graph) Poisson approximation to Bernoulli sums underlying the discretized sampler; pass to the continuum by refining the mesh. ([ar5iv][12], [HathiTrust Digital Library][13])

### Theorem 3 (thinning/superposition robustness)

* **Independent thinning** of a PPP with retention $p$ (constant, or a deterministic spacetime function) yields another PPP with intensity $p\rho$.
* Superposition of independent PPPs is PPP with intensity equal to the sum.
  These facts cover “death‐penalty” or acceptance/rejection steps *provided they are independent of the realized configuration inside the region being tested*. ([Wikipedia][11])

> **Interpretation for your algorithm.**
> If your “fractal” growth has: (i) label‑invariant, causally consistent births at rate $\rho$ per unit 4‑volume, (ii) no reinforcement/repulsion within the same infinitesimal region (A4), and (iii) any global acceptance filter implemented as independent thinning or as a deterministic spacetime modulation $\rho(x)$, then the output is *exactly* a Poisson sprinkling (or inhomogeneous PPP) into $(M,g)$. The “fractal” aspect can reside in *coarse‑graining* or in downstream IG weights, but it must not induce configuration‑dependent clustering at the sampling scale.

### Diagnostics you can run (and include in an appendix of your paper)

For a family of convex Alexandrov diamonds $\mathbf{A}[p,q]$:

* **Poisson count scaling:** $\mathbb{E}N(\mathbf{A})=\rho V$, $\mathrm{Var}N(\mathbf{A})=\rho V$.
* **Void probability:** $P\{N(\mathbf{A})=0\}=e^{-\rho V}$.
* **Ripley’s $K$ / pair‑correlation $g$:** $K(r)\propto r^{d}$, $g\equiv 1$ (after mapping to Minkowski distance via order‑invariants).
  Use residual analysis for spatial point processes adapted to Lorentzian regions (same mathematics applies once the region family is fixed). ([Wiley Online Library][10])

---

## 2) From samples to a valid causal set

Let $C$ be the set of sampled points, and define the order $x\prec y \iff \Phi(x)\in J^{-}(\Phi(y))$ (induced manifold causal order). Then:

* **Partial order:** antisymmetry and transitivity follow from the manifold’s causal structure; hence $(C,\prec)$ is a poset.
* **Local finiteness:** in a PPP, every bounded Alexandrov interval contains finitely many points almost surely; thus local finiteness holds.
* **Faithful embedding:** in the $(\delta,V_0)$ sense, Poisson sprinklings are faithful (regular lattices are not). ([ar5iv][2])

### Theorem 4 (Faithful embedding via Poisson sprinkling)

If $X$ is PPP of intensity $\rho$ on a globally hyperbolic region $M$ and $\prec$ is induced from $(M,g)$, then $\Phi$ is faithful with probability $1-o(1)$ on scales $V_0$ satisfying $l_c^d\ll V_0\ll \mathrm{Vol}(M)$ (where $l_c=\rho^{-1/d}$ is the discreteness scale).
*Sketch.* Directly from the Poisson law of counts with mean $\rho V$ and the order‑preserving nature of causal relation. ([ar5iv][2], [SpringerLink][14])

### Geometry/dimension/curvature checks (order‑invariant)

* **Myrheim–Meyer dimension** (ordering fraction): recovers $d$ for Minkowski sprinklings; passes to curved spacetimes by local RNN diagnostics. ([Wikipedia][1], [APS Journals][7])
* **Topology via thickened antichains:** recovers homology for faithful embeddings at sufficiently high density. ([arXiv][6])
* **Curvature/action:** Benincasa–Dowker d’Alembertian and scalar curvature estimator; converges to $\Box$ and $R$ in the continuum limit, providing an order‑theoretic action. ([APS Journals][15])

These order‑invariant observables let you *demonstrate* manifold‑likeness and guard against “Kleitman–Rothschild” non‑manifold posets (entropy problem). ([SpringerLink][14])

---

## 3) Lorentz invariance and the BHS theorem: constraints on any extra graph

**Key fact.** In a sprinkled causal set in Minkowski space, each element almost surely has **infinitely many links** (nearest neighbors in the transitive reduction), reflecting fundamental nonlocality; discreteness + Lorentz invariance forces this. ([ar5iv][16])

**BHS no‑go.** There is *no* measurable, Lorentz‑equivariant way to associate a **finite‑valency** graph or a preferred direction to a Poisson sprinkling. Any such construction would pick a frame and break Lorentz symmetry. ([World Scientific][3])

### What this means for your Information Graph (IG)

* **Allowed:** Use IG *as extra structure* whose law is *derived covariantly from the order alone* (and possibly a global density scale). For example, define weights $w(x\to y)=f\big(|I(x,y)|\big)$ where $|I(x,y)|$ is the cardinality of the Alexandrov interval, or derive operators à la causal‑set $\Box$ (nonlocal, retarded, Lorentz‑invariant in distribution). ([SpringerLink][17], [arXiv][18], [ResearchGate][19])
* **Not allowed (if you want LI):** IGs that impose **finite fixed valency** “nearest neighbors” chosen by a spacelike metric or a preferred foliation, or any rule requiring a timelike unit vector field—these contradict BHS and thus would break LI at the kinematical level. ([World Scientific][3])
* **Safe practice:** If for numerical work you temporarily fix a foliation to *construct* an IG (e.g., to run a min‑cut), make the **final physical statements strictly order‑invariant**, and show that outputs are insensitive to the foliation (gauge choice). This is analogous to gauge‑fixing in continuum computations; the *physics* must not depend on it.

### Theorem 5 (IG admissibility by order‑invariant construction)

Let $G=(C,E,w)$ be a (possibly directed) weighted graph on the causet $C$ whose edge set and weights are measurable functions of order‑invariants (e.g., interval cardinalities, link relation, longest‑chain length, layer counts) and of global scales like $\rho$. Then under Poisson sprinkling the **law of $G$** is Lorentz‑invariant (in distribution) and label‑invariant.
*Sketch.* Order‑invariants are preserved by causal isomorphisms and the sprinkling ensemble is invariant under the isometry group; therefore the induced random $G$ is equivariant in law. Nonlocality (unbounded valency) is consistent with CST. ([SpringerLink][14], [ar5iv][16])

> **Bottom line:** Using an IG is “OK” provided it is *derived from order‑invariant data* and accepts the inherent nonlocality (unbounded degree). If you *also* want strict Lorentz equivariance at the level of the construction (not just the physics), do **not** fix a preferred frame or finite valency.

---

## 4) Strengthened statements you can include in your paper

Below, $(M,g)$ is globally hyperbolic with finite volume region $W\subset M$; $\rho>0$ is the target sprinkling density. (All constants and bounds can be made explicit.)

### Theorem A (Exact equivalence in law)

Assume your sampler is a sequential growth process satisfying A1–A5 with *deterministic* Papangelou intensity $\rho$ (or $\rho(x)$). Then for every finite collection of disjoint Borel sets $A_1,\dots,A_k\subset W$,

$$
(N(A_1),\dots,N(A_k)) \ \stackrel{d}{=}\ \big(\mathrm{Poisson}(\rho\,\mathrm{Vol}A_1),\dots,\mathrm{Poisson}(\rho\,\mathrm{Vol}A_k)\big)
$$

with independence across $i$. Hence the output equals a Poisson sprinkling in law. (Cite GNZ/Last–Penrose.) ([Cambridge University Press & Assessment][9])

### Theorem B (Quantitative approximation)

If births are generated on a fine tessellation by independent (or dependency‑bounded) Bernoulli trials of mean $\rho\Delta V$, then for any union $A$ of cells, $d_{\mathrm{TV}}\big(\mathcal{L}(N(A)),\mathrm{Poisson}(\rho \mathrm{Vol}A)\big)\le \varepsilon(\text{mesh},\deg)$, with $\varepsilon\to0$ as cell size $\to 0$ at fixed $\deg$. (Stein–Chen.) ([ar5iv][12])

### Theorem C (Faithful embedding & order validity)

For $X$ as above and order induced from $(M,g)$, $(C,\prec)$ is almost surely a locally finite poset and $\Phi$ is $(\delta,V_0)$-faithful for $l_c^d\ll V_0\ll \mathrm{Vol}(W)$. Moreover, dimension estimators (Myrheim–Meyer), homology via thickened antichains, and discrete $\Box$/curvature converge in the CST continuum limit (large $\rho$). ([Wikipedia][1], [arXiv][6], [APS Journals][15])

### Theorem D (IG admissibility)

Let $G$ be constructed via a measurable functional of order‑invariants (e.g., use link relation and interval sizes to set capacities for min‑cuts). Then $G$ is label‑ and Lorentz‑invariant in distribution under sprinkling; if a finite‑valency truncation is introduced for numerics, any claim of physics must be shown independent of this truncation (remove cutoff → same observable). (BHS no‑go explains why finite valency cannot be fundamental.) ([World Scientific][3], [ar5iv][16])

---

## 5) Checklist you can hand to referees (and attach to your code)

1. **PPP verification.** Report (i) count–volume linearity with unit slope $\rho$, (ii) void probability $e^{-\rho V}$, (iii) disjoint‑region independence, (iv) homogeneous K‑function $g\equiv1$. ([Wikipedia][11], [Wiley Online Library][10])
2. **Causet validity.** Prove antisymmetry/transitivity from the induced manifold order; show local finiteness by bounded counts in all diamonds.
3. **Faithfulness.** Use the $(\delta,V_0)$-definition; provide $\delta$ histograms over many random diamonds at several $V_0$. ([ar5iv][2])
4. **Geometry.** Report Myrheim–Meyer dimension (≈4), chain‑length vs interval cardinality fits, and thickened‑antichain homology. ([Wikipedia][1], [arXiv][6])
5. **IG compliance.** Document that edges/weights are functions of order‑invariants (e.g., interval cardinality, link status, layer number), not of a chosen foliation; if a foliation is used for computation, include invariance checks across random boosts; avoid fixed finite valency. ([World Scientific][3])

---

## 6) Anticipating common criticisms

* **“It’s not Poisson; the counts look sub/super‑Poisson.”**
  Then either the thinning/acceptance is configuration‑dependent (violates A4) or correlations were introduced by the update rule. Fix by enforcing a **history‑independent** Papangelou intensity (Theorem 1) or quantify the deviation via Stein–Chen bounds (Theorem 2). ([Cambridge University Press & Assessment][9], [ar5iv][12])

* **“Your lattice/nearest neighbors pick a frame.”**
  Correct—finite‑valency neighbor graphs violate LI by the BHS theorem. Replace them with **order‑invariant, nonlocal** constructions (links/interval‑based kernels, retarded operators) or treat any finite‑valency graph as a *gauge* artifact with demonstrated output independence. ([World Scientific][3])

* **“You can’t recover geometry/topology.”**
  Cite and implement Myrheim–Meyer dimension, thickened‑antichain homology, and BD curvature/action estimators with convergence diagnostics. ([Wikipedia][1], [arXiv][6], [APS Journals][15])

* **“Sampling on curved $M$?”**
  All statements hold for globally hyperbolic $(M,g)$; the PPP intensity is $\rho\,d\mathrm{Vol}_g$, and CST estimators are implemented locally in Riemann normal neighborhoods (as in the literature). ([SpringerLink][14])

---

### Short “ready‑to‑cite” claims (with sources)

1. *The only statistically Lorentz‑invariant discretization by random points is Poisson sprinkling.* (BHS theorem; plus standard PPP invariance.) ([World Scientific][3])
2. *Faithful embeddings are characterized by Poisson count statistics proportional to volume.* (Standard CST definition.) ([ar5iv][2])
3. *Order + number determines geometry up to scale; topology and dimension can be recovered order‑theoretically.* (Malament/HKMM + Myrheim–Meyer + thickened antichains.) ([Scribd][4], [Wikipedia][5], [arXiv][6])
4. *Discrete curvature/action exist purely from order and converge appropriately.* (Benincasa–Dowker; related d’Alembertians.) ([APS Journals][15], [SpringerLink][17])

---

## References (BibTeX)

```bibtex
@article{Bombelli1987,
  author    = {L. Bombelli and J. Lee and D. Meyer and R. D. Sorkin},
  title     = {Space-time as a causal set},
  journal   = {Phys. Rev. Lett.},
  volume    = {59},
  pages     = {521--524},
  year      = {1987},
  doi       = {10.1103/PhysRevLett.59.521}
}

@article{Malament1977,
  author  = {David B. Malament},
  title   = {The class of continuous timelike curves determines the topology of spacetime},
  journal = {Journal of Mathematical Physics},
  volume  = {18},
  number  = {7},
  pages   = {1399--1404},
  year    = {1977},
  doi     = {10.1063/1.523436}
}

@article{Hawking1976,
  author  = {S. W. Hawking and A. R. King and P. J. McCarthy},
  title   = {A new topology for curved space-time which incorporates the causal, differential, and conformal structures},
  journal = {Journal of Mathematical Physics},
  volume  = {17},
  number  = {2},
  pages   = {174--181},
  year    = {1976},
  doi     = {10.1063/1.522874}
}

@article{BHS2009,
  author  = {Luca Bombelli and Joe Henson and Rafael D. Sorkin},
  title   = {Discreteness without symmetry breaking: A theorem},
  journal = {Modern Physics Letters A},
  volume  = {24},
  number  = {32},
  pages   = {2579--2587},
  year    = {2009},
  doi     = {10.1142/S0217732309031958}
}

@article{RideoutSorkin2000,
  author  = {D. P. Rideout and R. D. Sorkin},
  title   = {Classical sequential growth dynamics for causal sets},
  journal = {Phys. Rev. D},
  volume  = {61},
  pages   = {024002},
  year    = {2000},
  doi     = {10.1103/PhysRevD.61.024002}
}

@article{RideoutSorkin2001,
  author  = {D. P. Rideout and R. D. Sorkin},
  title   = {Evidence for a continuum limit in causal set dynamics},
  journal = {Phys. Rev. D},
  volume  = {63},
  pages   = {104011},
  year    = {2001},
  doi     = {10.1103/PhysRevD.63.104011}
}

@article{BrightwellGregory1991,
  author  = {Graham Brightwell and Ruth Gregory},
  title   = {Structure of random discrete spacetime},
  journal = {Phys. Rev. Lett.},
  volume  = {66},
  pages   = {260--263},
  year    = {1991},
  doi     = {10.1103/PhysRevLett.66.260}
}

@article{Reid2003,
  author  = {David D. Reid},
  title   = {Manifold dimension of a causal set: Tests in conformally flat spacetimes},
  journal = {Phys. Rev. D},
  volume  = {67},
  pages   = {024034},
  year    = {2003},
  doi     = {10.1103/PhysRevD.67.024034}
}

@article{MajorRideoutSurya2007,
  author  = {Seth Major and David Rideout and Sumati Surya},
  title   = {On recovering continuum topology from a causal set},
  journal = {Journal of Mathematical Physics},
  volume  = {48},
  number  = {3},
  pages   = {032501},
  year    = {2007},
  doi     = {10.1063/1.2435599}
}

@article{BenincasaDowker2010,
  author  = {Dionigi M. T. Benincasa and Fay Dowker},
  title   = {Scalar Curvature of a Causal Set},
  journal = {Phys. Rev. Lett.},
  volume  = {104},
  pages   = {181301},
  year    = {2010},
  doi     = {10.1103/PhysRevLett.104.181301}
}

@article{AslanbeigiSaravaniSorkin2014,
  author  = {Siavash Aslanbeigi and Mehdi Saravani and Rafael D. Sorkin},
  title   = {Generalized causal set d’Alembertians},
  journal = {Journal of High Energy Physics},
  volume  = {2014},
  number  = {6},
  pages   = {24},
  year    = {2014},
  doi     = {10.1007/JHEP06(2014)024}
}

@article{Surya2019,
  author  = {Sumati Surya},
  title   = {The causal set approach to quantum gravity},
  journal = {Living Reviews in Relativity},
  volume  = {22},
  number  = {5},
  year    = {2019},
  doi     = {10.1007/s41114-019-0023-1}
}

@book{LastPenrose2018,
  author = {G{\"u}nter Last and Mathew Penrose},
  title  = {Lectures on the Poisson Process},
  publisher = {Cambridge University Press},
  year   = {2018}
}

@book{DaleyVereJones2008,
  author = {Daryl J. Daley and David Vere-Jones},
  title  = {An Introduction to the Theory of Point Processes, Vol. II},
  edition= {2nd},
  publisher = {Springer},
  year   = {2008}
}

@book{BarbourHolstJanson1992,
  author = {A. D. Barbour and Lars Holst and Svante Janson},
  title  = {Poisson Approximation},
  series = {Oxford Studies in Probability},
  publisher = {Clarendon Press},
  year   = {1992}
}

@article{BrightwellLuczak2011,
  author  = {Graham Brightwell and Malwina Luczak},
  title   = {Order-invariant measures on causal sets},
  journal = {Annals of Applied Probability},
  volume  = {21},
  number  = {4},
  pages   = {1493--1536},
  year    = {2011},
  doi     = {10.1214/10-AAP736}
}

@article{BaddeleyEtAl2005,
  author  = {Adrian Baddeley and others},
  title   = {Residual analysis for spatial point processes (with discussion)},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {67},
  number  = {5},
  pages   = {617--666},
  year    = {2005},
  doi     = {10.1111/j.1467-9868.2005.00519.x}
}
```

---

## Key web‑backed citations in the text

* Poisson sprinkling, faithful embedding, and Lorentz invariance in distribution: ([ar5iv][2], [Wikipedia][1], [World Scientific][3])
* Sequential growth, Bell causality, covariance: ([arXiv][8], [APS Journals][20])
* Dimension/topology/curvature estimators: ([Wikipedia][1], [arXiv][6], [APS Journals][15])
* PPP characterization and diagnostics (Papangelou/Stein–Chen/thinning): ([Cambridge University Press & Assessment][9], [Wiley Online Library][10], [Wikipedia][11])
* Nonlocality/infinite valency in sprinkled causets: ([ar5iv][16])

---

### Final takeaway

If you enforce **(A1–A5)** in your sampler—or provide quantitative Stein–Chen error bounds when relaxing A4—and you build the IG **only from order‑invariant quantities**, then (i) the sampling is in law a **Poisson sprinkling**, (ii) the induced order is a **valid causal set with a faithful embedding**, and (iii) the **IG does not violate CST or Lorentz invariance** (it must accept nonlocality and avoid finite‑valency “nearest‑neighbor” constructions). With the geometry/topology/curvature diagnostics above, you’ll have a referee‑grade, rigorous defense of the construction.

[1]: https://en.wikipedia.org/wiki/Causal_sets?utm_source=chatgpt.com "Causal sets"
[2]: https://ar5iv.labs.arxiv.org/html/0712.1648?utm_source=chatgpt.com "[0712.1648] Causal Set Topology"
[3]: https://www.worldscientific.com/doi/abs/10.1142/S0217732309031958?utm_source=chatgpt.com "DISCRETENESS WITHOUT SYMMETRY BREAKING: A THEOREM | Modern Physics Letters A"
[4]: https://www.scribd.com/document/490611872/Malament-D-The-CLass-of-Continuous-Timelike-Curves-etc-Spacetime-1977-pdf?utm_source=chatgpt.com "Malament D., The CLass of Continuous Timelike Curves - Etc - Spacetime - 1977 PDF | PDF | Manifold | Theoretical Physics"
[5]: https://en.wikipedia.org/wiki/Causal_structure?utm_source=chatgpt.com "Causal structure"
[6]: https://arxiv.org/abs/gr-qc/0604124?utm_source=chatgpt.com "[gr-qc/0604124] On Recovering Continuum Topology from a Causal Set"
[7]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.67.024034?utm_source=chatgpt.com "Phys. Rev. D 67, 024034 (2003) - Manifold dimension of a causal set: Tests in conformally flat spacetimes"
[8]: https://arxiv.org/abs/gr-qc/9904062?utm_source=chatgpt.com "[gr-qc/9904062] A Classical Sequential Growth Dynamics for Causal Sets"
[9]: https://www.cambridge.org/core/journals/journal-of-applied-probability/article/multivariate-poisson-and-poisson-process-approximations-with-applications-to-bernoulli-sums-and-u-statistics/CDDCD72F5BFE59699A6360CE20427B7A?utm_source=chatgpt.com "Multivariate Poisson and Poisson process approximations with applications to Bernoulli sums and $U$-statistics | Journal of Applied Probability | Cambridge Core"
[10]: https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2005.00519.x?utm_source=chatgpt.com "Residual analysis for spatial point processes (with discussion) - Baddeley - 2005 - Journal of the Royal Statistical Society: Series B (Statistical Methodology) - Wiley Online Library"
[11]: https://en.wikipedia.org/wiki/Poisson_point_process?utm_source=chatgpt.com "Poisson point process"
[12]: https://ar5iv.labs.arxiv.org/html/1404.1392?utm_source=chatgpt.com "[1404.1392] A short survey of Stein’s method"
[13]: https://catalog.hathitrust.org/Record/002584300?utm_source=chatgpt.com "Catalog Record: Poisson approximation | HathiTrust Digital Library"
[14]: https://link.springer.com/article/10.1007/s41114-019-0023-1?utm_source=chatgpt.com "The causal set approach to quantum gravity | Living Reviews in Relativity"
[15]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.181301?utm_source=chatgpt.com "Scalar Curvature of a Causal Set | Phys. Rev. Lett."
[16]: https://ar5iv.labs.arxiv.org/html/1903.11544?utm_source=chatgpt.com "[1903.11544] The causal set approach to quantum gravity"
[17]: https://link.springer.com/article/10.1007/JHEP06%282014%29024?utm_source=chatgpt.com "Generalized causal set d’Alembertians | Journal of High Energy Physics"
[18]: https://arxiv.org/abs/1403.1622?utm_source=chatgpt.com "[1403.1622] Generalized Causal Set d'Alembertians"
[19]: https://www.researchgate.net/publication/236687926_Causal_set_d%27Alembertians_for_various_dimensions?utm_source=chatgpt.com "Causal set d'Alembertians for various dimensions"
[20]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.61.024002?utm_source=chatgpt.com "Classical sequential growth dynamics for causal sets | Phys. Rev. D"
