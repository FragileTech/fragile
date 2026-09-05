---
title: "Lectures on Algorithmic Geometrodynamics"
subtitle: "A Two-Volume Study of Bounded Intelligence and Interacting Particles"
author: "Guillem Duran-Ballester"
---

(sec-algorithmic-geometrodynamics)=
# Lectures on Algorithmic Geometrodynamics

**A Two-Volume Study of Bounded Intelligence and Interacting Particles**

by *Guillem Duran-Ballester and Sergio Hernández Cerezo*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18237451.svg)](https://doi.org/10.5281/zenodo.18237451)

:::{div} feynman-prose
An agent has limited memory, incomplete observations, and a finite budget for
computation. It must decide what to represent, how to predict, and where to spend
its next step of search. These lectures study mathematical models of those
choices and the geometry used to describe them.

**Volume I, Fragile Mechanics**, develops an agent architecture through
representation, belief dynamics, control, and runtime diagnostics.
**Volume II, The Fractal Gas**, develops interacting-particle algorithms, their
convergence analysis, and geometric constructions from their histories.

You can begin with the question that interests you. The agent chapters explain
how representations and control fit together. The particle chapters start from
an explicit update rule and ask what can be established about the resulting
population. Each volume has its own definitions, reading guide, and detailed
mathematical arguments.
:::

- {doc}`Volume I: Fragile Mechanics <source/1_agent/intro_agent>`
- {doc}`Volume II: The Fractal Gas <source/2_fractal_gas/intro_fractal_gas>`

(sec-what-is-ag)=
## What this book studies

:::{div} feynman-prose
A geometry tells us which states are close and what it costs to move between
them. For an agent, those choices affect representation, prediction, and control.
For a swarm, they affect companion selection, motion, and the interpretation of
its recorded interactions. The term *algorithmic geometrodynamics* names the
study of these relationships between computation, dynamics, and geometry.

A concrete example is the separation between moving a walker and copying one.
Motion transports an existing state. Cloning changes the population allocated
to that state. A mathematical description of the swarm must account for both.
Likewise, an agent's belief update can change both the location of a hypothesis
and the probability assigned to it. Transport and information geometry provide
ways to express these changes precisely.
:::

The book contains definitions, model-specific theorems, proposed constructions,
and numerical studies. Each claim has the scope of its stated assumptions.
Connections with field theory require the variables, dynamics, and limiting
regime specified in the relevant chapter. The Fractal Gas
{doc}`introduction <source/2_fractal_gas/intro_fractal_gas>` distinguishes
finite-particle relaxation, mean-field approximation, and geometric continuum
limits before presenting those connections.

(sec-two-volumes)=
## The two volumes

(sec-vol1-overview)=
### Volume I: Fragile Mechanics

*On Geometry, Thermodynamics, and Bounded Intelligence*

Volume I develops a proposed architecture for agents under partial observability
and finite capacity. Its components include a structured latent representation,
a world model, belief updates, a critic, a policy, and runtime monitoring. The
**Sieve** organizes diagnostics and interventions around conditions such as
stability, capacity, and grounding; the associated analyses state the conditions
under which each diagnostic or intervention applies.

::::{div} feynman-added
| Topic | Entry point |
|---|---|
| State, observations, and the control loop | {doc}`Foundations <source/1_agent/01_foundations/01_definitions>` |
| Runtime diagnostics and interventions | {doc}`The Sieve <source/1_agent/02_sieve/01_diagnostics>` |
| Representation and implementation architecture | {doc}`Architecture at a glance <source/1_agent/03_architecture/00_architecture_at_a_glance>` |
| Exploration and belief dynamics | {doc}`Control and belief <source/1_agent/04_control/01_exploration>` |
| Metrics, transport, and equations of motion | {doc}`Geometric dynamics <source/1_agent/05_geometry/01_metric_law>` |
| Boundary interfaces and reward fields | {doc}`Holography and field theory <source/1_agent/06_fields/01_boundary_interface>` |
| Memory, adaptation, and other cognitive extensions | {doc}`Cognitive extensions <source/1_agent/07_cognition/01_supervised_topo>` |
| Multi-agent interactions and gauge formulations | {doc}`Multi-agent gauge theory <source/1_agent/08_multiagent/01_gauge_theory>` |
| Proof of Useful Work | {doc}`Economics <source/1_agent/09_economics/01_pomw>` |
| Encoder, world model, and planning implementation | {doc}`Implementation <source/1_agent/11_implementation/01_encoder>` |
::::

The {doc}`volume introduction <source/1_agent/intro_agent>` provides the full map.
Its {doc}`FAQ <source/1_agent/10_appendices/04_faq>`,
{doc}`derivations <source/1_agent/10_appendices/01_derivations>`, and
{doc}`loss reference <source/1_agent/10_appendices/06_losses>` support detailed
reading and implementation.

(sec-vol2-overview)=
### Volume II: The Fractal Gas

*Interacting Particles, Convergence, and Emergent Geometry*

Volume II starts with populations of walkers that select companions, compute
fitness, clone, and undergo kinetic motion. The analysis follows the specified
transition kernel through operator estimates, conditional relaxation,
mean-field limits, and stronger entropy or regularity bounds. Later chapters
construct the Fractal Set from walker histories and examine geometric and field
models associated with that record.

::::{div} feynman-added
| Part | Subject | Central task |
|---|---|---|
| {doc}`I <source/2_fractal_gas/parts/01_foundations>` | Algorithms and foundations | Define the framework, Euclidean and latent variants, and single-particle dynamics |
| {doc}`II <source/2_fractal_gas/parts/02_convergence>` | Finite-particle convergence | Combine cloning, Wasserstein, and kinetic estimates for the full update |
| {doc}`III <source/2_fractal_gas/parts/03_mean_field>` | Mean-field limits and equilibrium | Relate population laws, marginals, quasi-stationarity, and exchangeability |
| {doc}`IV <source/2_fractal_gas/parts/04_entropy_regularity>` | Entropy, regularity, and bounds | Establish stronger estimates with explicit parameter and population dependence |
| {doc}`V <source/2_fractal_gas/parts/05_continuum>` | Fractal Set and continuum limits | Construct the discrete record and examine conditions for geometric limits |
| {doc}`VI <source/2_fractal_gas/parts/06_fields>` | Fields and emergent physics | Develop field constructions and their conditional physical interpretations |
| {doc}`VII <source/2_fractal_gas/parts/07_experiments>` | Computation and experiments | Define observables, inspect simulations, and perform calibration |
| {doc}`Reference <source/2_fractal_gas/parts/08_reference>` | FAQ and literature | Locate explanations, historical sources, and related studies |
::::

The {doc}`volume introduction <source/2_fractal_gas/intro_fractal_gas>` explains
the dependencies between these parts. The analytic arguments appear as chapters
in the main reading sequence.

(sec-how-to-read-lectures)=
(sec-reading-paths)=
## Choose a reading path

:::{div} feynman-prose
Start with a model you can describe operationally. For the agent, trace one
observation through the representation, belief update, and action. For the gas,
trace one swarm update through companion selection, cloning, and motion. Once
you know what changes in a step, the estimates have something concrete to refer
to.

The **Full Mode** view includes explanatory prose and examples. **Expert Mode**
hides the marked explanatory additions so you can concentrate on definitions,
statements, and proofs. The mathematical hypotheses are the same in both views.
:::

::::{div} feynman-added
| Your interest | Suggested route |
|---|---|
| Agent architecture | Volume I foundations, architecture, control, and runtime diagnostics; then the implementation chapters |
| Optimization and sampling | Volume II algorithm introduction and Euclidean model; then finite-particle convergence and parameter constraints |
| Probability and analysis | Volume II Parts I–IV, keeping track of the state space, conditioning, metric, and order of limits |
| Geometry and fields | Volume I geometry and multi-agent chapters; Volume II Parts V–VI together with their analytic prerequisites |
| Numerical assessment | Volume II Part VII, returning to the definitions of the measured quantities and the hypotheses of the comparison |
::::

(sec-quick-links)=
### Direct entry points

- {doc}`Agent architecture overview <source/1_agent/03_architecture/00_architecture_at_a_glance>`
- {doc}`Wasserstein–Fisher–Rao geometry <source/1_agent/05_geometry/02_wfr_geometry>`
- {doc}`Fractal Gas algorithm introduction <source/2_fractal_gas/1_the_algorithm/01_algorithm_intuition>`
- {doc}`Finite-particle convergence <source/2_fractal_gas/convergence_program/06_convergence>`
- {doc}`Fractal Set construction <source/2_fractal_gas/2_fractal_set/01_fractal_set>`
- {doc}`Computation and experiments <source/2_fractal_gas/parts/07_experiments>`

(sec-volume-connections)=
## Using the volumes together

:::{div} feynman-prose
One possible integration uses the agent's learned representation as the space
in which a swarm searches. The agent supplies a model, a metric, or a score; the
swarm produces candidate trajectories and measurements. Those outputs can then
inform planning or learning.

This connection has mathematical consequences. A changing learned metric changes
the swarm's transition rule. A model-generated trajectory carries the errors of
that model. To apply a convergence result to the combined system, identify which
quantities are held fixed, which are updated, and whether the estimates remain
valid during those updates.
:::

Volume I supplies the agent's representation and control setting. Volume II
supplies particle models whose assumptions can be checked in that setting.
A result established for a fixed Euclidean transition kernel applies to an
adaptive latent-space implementation only when the required hypotheses are
verified for that implementation.

(sec-key-results)=
## Following a mathematical claim

The linked chapters contain the formal statements and arguments. To assess a
claim, identify four things:

1. **The object.** Determine whether the claim concerns an agent, an entire
   swarm, a one-particle marginal, an empirical measure, or a continuum field.
2. **The hypotheses.** Locate assumptions about the transition rule, boundary,
   noise, regularity, survival, and parameter regime.
3. **The conclusion.** Record the precise topology, metric, rate, or observable
   controlled by the result.
4. **The dependencies.** Follow the estimates used in the proof and check their
   scope before transferring the conclusion to another model.

For the Fractal Gas, convergence at fixed population size, propagation of chaos,
and convergence of geometric operators are distinct statements. Uniformity in
population size or time requires its own estimates. Conditional results retain
their hypotheses, and conjectural physical identifications require further
argument and empirical assessment.

(sec-landing-faq)=
### Common questions

**Where should I start if I want the proofs?**

For particle dynamics, begin with
{doc}`Volume II foundations <source/2_fractal_gas/parts/01_foundations>` and
{doc}`finite-particle convergence <source/2_fractal_gas/parts/02_convergence>`.
For agent geometry, begin with the definitions supporting the
{doc}`metric law <source/1_agent/05_geometry/01_metric_law>` and
{doc}`WFR geometry <source/1_agent/05_geometry/02_wfr_geometry>`.

**How does the agent relate to reinforcement learning?**

Volume I formulates representation, belief dynamics, and control together with
capacity and geometric constraints. Its
{doc}`introduction <source/1_agent/intro_agent>` discusses comparisons and limits
connecting this formulation to reinforcement-learning models.

**What do the physics connections establish?**

Their scope depends on the specified construction and argument. A symmetry or a
geometric observable must be defined before its dynamics or physical
interpretation can be assessed. Volume II's
{doc}`field chapters <source/2_fractal_gas/parts/06_fields>` should be read with
their continuum conditions and mathematical status in view.

**Where are detailed objections and limitations discussed?**

Consult the {doc}`Fragile Mechanics FAQ <source/1_agent/10_appendices/04_faq>` and
the {doc}`Fractal Gas FAQ <source/2_fractal_gas/reference/faq>`, then follow the
references to the relevant definitions and results.

(sec-llm-exploration)=
## Downloads and assisted reading

The **Download prompt** menu exports a volume for offline reading or use as
context in an assistant. Its choices are:

::::{div} feynman-added
| Setting | Choices |
|---|---|
| Volume | **Vol 1 - Agent** (Fragile Mechanics) or **Vol 2 - Fractal Gas** |
| Proofs | **With proofs** or **Without proofs** |
| Format | **Markdown** (`.md`) or **Text** (`.txt`) |
::::

Choose **With proofs** when following an argument. The shorter export is suited
to orientation and locating topics; consult the chapter or full export when the
proof is needed. File sizes depend on the generated version.

:::{div} feynman-prose
An assistant is easiest to check when you give it a bounded task. Supply the
relevant chapter and ask it to identify a definition, explain one estimate, or
trace the hypotheses of a particular theorem. Keep the source open beside the
answer so that you can compare the explanation with the statement.

For example:

- “Explain one Fractal Gas update using the definitions in this chapter.”
- “Which hypotheses are used when the cloning and kinetic estimates are
  combined?”
- “Distinguish the finite-particle QSD from the mean-field stationary law.”
- “Identify the geometric and sampling assumptions in this continuum result.”
- “For this calibration report, separate the fixed inputs from the quantities
  being compared.”
:::

(sec-landing-citation)=
## Citation

The DOI and citation below identify the project's published record.

:::{admonition} How to Cite This Work
:class: note dropdown

**DOI:** [10.5281/zenodo.18237451](https://doi.org/10.5281/zenodo.18237451)

**Online version:** [https://fragiletech.github.io/fragile/](https://fragiletech.github.io/fragile/)

**Preferred citation:**
> Duran Ballester, G. (2025). *Lectures on Algorithmic Geometrodynamics*. Zenodo. https://doi.org/10.5281/zenodo.18237451

**BibTeX:**
```bibtex
@book{duranballester2025lag,
  author    = {Duran Ballester, Guillem and Hernández Cerezo, Sergio },
  title     = {Lectures on Algorithmic Geometrodynamics},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18237451},
  url       = {https://fragiletech.github.io/fragile/}
}
```

**Note on naming:** The author's name follows Spanish naming conventions. *Duran Ballester* is the complete surname (paternal + maternal). Please index under **D** for Duran, not B for Ballester. The hyphenated form *Duran-Ballester* is used in English contexts to prevent parser errors. The same logic applies to Sergio Hernández Cerezo.
:::
