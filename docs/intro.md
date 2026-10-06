---
title: "Fragile Mechanics"
subtitle: "On Geometry, Thermodynamics, and Bounded Intelligence"
author: "Guillem Duran-Ballester"
---

(sec-algorithmic-geometrodynamics)=
# Fragile Mechanics

**On Geometry, Thermodynamics, and Bounded Intelligence**

by *Guillem Duran-Ballester and Sergio Hernández Cerezo*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18237451.svg)](https://doi.org/10.5281/zenodo.18237451)

:::{div} feynman-prose
An agent has limited memory, incomplete observations, and a finite budget for
computation. It must decide what to represent, how to predict, and where to spend
its next step of search. These lectures study mathematical models of those
choices and the geometry used to describe them.

The central question is practical: how can an agent remain intelligible when
its information and computational resources are bounded? The answer developed
here combines structured representations, belief dynamics, control, geometric
models, and runtime diagnostics. Each piece gives us something concrete to
inspect when the agent succeeds—or fails.
:::

- {doc}`Begin Fragile Mechanics <source/1_agent/intro_agent>`

:::{div} feynman-prose
The hands-on <a href="../lab/">Control Laboratory guide</a> is a separate documentation site
with its own navigation. It accompanies the lectures and connects the theory to
implementation exercises.
:::

(sec-what-is-ag)=
## What this book studies

:::{div} feynman-prose
A geometry tells us which states are close and what it costs to move between
them. For an agent, those choices affect representation, prediction, and control.
The term *algorithmic geometrodynamics* names the study of these relationships
between computation, dynamics, and geometry.

Here is the picture to keep in mind. An observation arrives, but the agent
cannot preserve every detail. It must compress the observation into a useful
state, update what it believes, and choose an action. The geometry determines
which changes count as small; the information budget limits what can be kept;
the controller decides how to spend the next step. These are different jobs,
and the architecture keeps them separate enough to test.
:::

The book contains definitions, model-specific theorems, proposed constructions,
and numerical studies. Each claim has the scope of its stated assumptions.
Connections with field theory require the variables, dynamics, and limiting
regime specified in the relevant chapter.

(sec-book-at-a-glance)=
## The book at a glance

Fragile Mechanics develops a proposed architecture for agents under partial
observability and finite capacity. Its components include a structured latent
representation, a world model, belief updates, a critic, a policy, and runtime
monitoring. The **Sieve** organizes diagnostics and interventions around
conditions such as stability, capacity, and grounding; the associated analyses
state the conditions under which each diagnostic or intervention applies.

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

The {doc}`book introduction <source/1_agent/intro_agent>` provides the full map.
Its {doc}`FAQ <source/1_agent/10_appendices/04_faq>`,
{doc}`derivations <source/1_agent/10_appendices/01_derivations>`, and
{doc}`loss reference <source/1_agent/10_appendices/06_losses>` support detailed
reading and implementation.

(sec-how-to-read-lectures)=
(sec-reading-paths)=
## Choose a reading path

:::{div} feynman-prose
Start with a process you can describe operationally: trace one observation
through the representation, belief update, and action. Once you know what
changes in a step, the equations have something concrete to refer to.

The **Full Mode** view includes explanatory prose and examples. **Expert Mode**
hides the marked explanatory additions so you can concentrate on definitions,
statements, and proofs. The mathematical hypotheses are the same in both views.
:::

::::{div} feynman-added
| Your interest | Suggested route |
|---|---|
| Agent architecture | Foundations, architecture, control, and runtime diagnostics; then the implementation chapters |
| Probability and information | Foundations, belief dynamics, WFR geometry, and the information bounds |
| Geometry and fields | Metric law, WFR geometry, boundary interfaces, and multi-agent gauge theory |
| Building and testing agents | Architecture overview, implementation chapters, the Sieve, and the Control Laboratory guide |
| Economics and consensus | Multi-agent foundations followed by Proof of Useful Work |
::::

(sec-quick-links)=
### Direct entry points

- {doc}`Agent architecture overview <source/1_agent/03_architecture/00_architecture_at_a_glance>`
- {doc}`Runtime diagnostics <source/1_agent/02_sieve/01_diagnostics>`
- {doc}`Wasserstein–Fisher–Rao geometry <source/1_agent/05_geometry/02_wfr_geometry>`
- {doc}`Boundary interfaces <source/1_agent/06_fields/01_boundary_interface>`
- {doc}`Multi-agent gauge theory <source/1_agent/08_multiagent/01_gauge_theory>`
- {doc}`Implementation <source/1_agent/11_implementation/01_encoder>`

(sec-key-results)=
## Following a mathematical claim

The linked chapters contain the formal statements and arguments. To assess a
claim, identify four things:

1. **The object.** Determine whether the claim concerns a representation, a
   belief, a policy, a diagnostic, or an interacting collection of agents.
2. **The hypotheses.** Locate assumptions about capacity, regularity, geometry,
   observability, stability, and the operating regime.
3. **The conclusion.** Record the precise quantity, metric, rate, or observable
   controlled by the result.
4. **The dependencies.** Follow the estimates used in the proof and check their
   scope before transferring the conclusion to another model.

:::{div} feynman-prose
The important habit is to carry the assumptions with the result. A statement
about a fixed metric does not automatically describe a metric that is learning
at the same time. A capacity bound does not by itself guarantee good control.
When the model changes, ask which part of the argument survives.
:::

(sec-landing-faq)=
### Common questions

**Where should I start if I want the proofs?**

Begin with the definitions in {doc}`foundations <source/1_agent/01_foundations/01_definitions>`,
then follow the prerequisites of the result you care about. For agent geometry,
start with the definitions supporting the
{doc}`metric law <source/1_agent/05_geometry/01_metric_law>` and
{doc}`WFR geometry <source/1_agent/05_geometry/02_wfr_geometry>`.

**How does the agent relate to reinforcement learning?**

Fragile Mechanics formulates representation, belief dynamics, and control
together with capacity and geometric constraints. Its
{doc}`introduction <source/1_agent/intro_agent>` discusses comparisons and limits
connecting this formulation to reinforcement-learning models.

**What do the physics connections establish?**

Their scope depends on the specified construction and argument. A symmetry or a
geometric observable must be defined before its dynamics or physical
interpretation can be assessed. Read each identification with its mathematical
conditions and stated status in view.

**Where are detailed objections and limitations discussed?**

Consult the {doc}`Fragile Mechanics FAQ <source/1_agent/10_appendices/04_faq>`,
then follow its references to the relevant definitions and results.

(sec-llm-exploration)=
## Downloads and assisted reading

The **Download prompt** menu exports the book for offline reading or use as
context in an assistant. Choose **With proofs** when following an argument. The
shorter export is suited to orientation and locating topics; consult the chapter
or full export when the proof is needed.

:::{div} feynman-prose
An assistant is easiest to check when you give it a bounded task. Supply the
relevant chapter and ask it to identify a definition, explain one estimate, or
trace the hypotheses of a particular theorem. Keep the source open beside the
answer so that you can compare the explanation with the statement.

For example:

- “Trace one observation through the representation, belief update, and action.”
- “Which assumptions are used in this stability argument?”
- “Distinguish the geometric definition from its physical interpretation.”
- “For this diagnostic, identify the measured quantity and the intervention.”
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
  author    = {Duran Ballester, Guillem and Hernández Cerezo, Sergio},
  title     = {Lectures on Algorithmic Geometrodynamics},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18237451},
  url       = {https://fragiletech.github.io/fragile/}
}
```

**Note on naming:** The author's name follows Spanish naming conventions. *Duran Ballester* is the complete surname (paternal + maternal). Please index under **D** for Duran, not B for Ballester. The hyphenated form *Duran-Ballester* is used in English contexts to prevent parser errors. The same logic applies to Sergio Hernández Cerezo.
:::
