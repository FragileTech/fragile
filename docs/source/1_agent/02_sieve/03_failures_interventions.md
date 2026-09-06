(sec-failure-modes)=
# Failure Modes (Observed Pathologies)

## TLDR

- Enumerate a “periodic table” of **observable failure patterns** (collapse, oscillation, Zeno/chatter, overfitting,
  paralysis, fragility) and map each to a responsible component.
- Treat failures as **structured diagnostics**: identify the signature, then apply the corresponding intervention
  (damping, regularization, projection, reset, curriculum changes).
- Use the table to distinguish **fundamental limits** (barriers) from **implementation/optimization issues**
  (fixable by schedule/architecture changes).
- Provide a shared vocabulary for debugging: “what failed” is named, and “where it lives” is explicit.
- This chapter pairs with `Diagnostics` and `Barriers`: together they define what to monitor, what limits exist, and what
  to do when something triggers.

## Roadmap

1. Failure taxonomy and how to read the table.
2. Canonical interventions and what signals should trigger them.

:::{div} feynman-prose
Here is something fascinating about learning systems: they fail in characteristic ways. Not randomly, but in patterns that repeat across wildly different domains. Whether you are training a robot to walk or a neural network to recognize faces, the same breakdowns keep appearing.

Why? Because many learning systems face recurring challenges: balancing exploration against exploitation, compressing the world into
internal representations, and updating behavior from feedback. When one of these goes wrong, related failure patterns can appear,
but the signature and remedy still depend on the model and data.

What follows is a "periodic table" of failure modes. Just as chemists organized elements by their properties and could predict how unknown elements would behave, we can organize learning failures by their signatures and predict how to fix them. The key insight: each failure traces back to a specific component, and that component tells us exactly where to look for the repair.
:::

(rb-rl-pathologies)=
:::{admonition} Researcher Bridge: RL Pathologies, Named and Localized
:class: info
If you have seen mode collapse, oscillation, overfitting, or deadlock in RL, this table is the same landscape but made explicit. Each failure is tied to a component and a diagnostic signature, so it can be detected and corrected rather than discovered post hoc.
:::

:::{div} feynman-prose
Before diving into the table, let me give you a feel for its structure.

Each mode has a two-letter mnemonic code. The code is a stable label for cross-references; its letters are not a formal coordinate system.

The "Failed Component" column is crucial. It tells you where the disease lives. Is it in the Policy? The Shutter? The World Model? The Critic? Knowing this immediately narrows your debugging search.

Here is the beautiful thing: once you know the component and failure type, the intervention almost writes itself. A policy that oscillates needs damping. A world model that overfits needs regularization. Like medicine, the diagnosis determines the treatment.
:::

When Limits are breached or Interfaces fail, the agent exhibits specific pathologies.

| Mode    | Standard Name       | Failed Component     | Fragile (Pathology) Name      | Description                                                                     |
|---------|---------------------|----------------------|-------------------------------|---------------------------------------------------------------------------------|
| **D.D** | Dispersion-Decay    | **All (Optimal)**    | **Success (Convergence)**     | Agent solves task; error drops to a stable floor.                               |
| **S.C** | Stability-Critic     | **Critic**           | **Target Chasing**       | Bootstrapped value targets drift faster than the critic can track them. |
| **S.E** | Subcritical-Equilib | **Policy**           | **Curriculum Stumble**        | Task difficulty increases faster than adaptation rate.                          |
| **C.D** | Conc-Dispersion     | **Policy/Shutter**   | **Mode Collapse / Obsession** | Policy concentrates on a single mode, neglecting remaining state space.         |
| **C.E** | Conc-Escape         | **Policy/Critic**    | **Divergence / Blow-up**      | Gradients/activations diverge; optimization becomes unstable.                   |
| **T.E** | Topo-Extension      | **Shutter/WM**       | **Wrong Paradigm**            | Architecture is topologically insufficient.                                     |
| **S.D** | Struct-Dispersion   | **Shutter**          | **Symmetry Blindness**        | Fails to exploit available symmetries.                                          |
| **C.C** | Event Accumulation  | **Policy/WM**        | **Decision Paralysis**        | Input happens faster than decision loop (Zeno).                                 |
| **T.D** | Glassy Freeze       | **Policy**           | **Learned Helplessness**      | Policy converges to suboptimal fixed point with zero gradient.                  |
| **D.E** | Oscillatory         | **Policy**           | **Pilot-Induced Oscillation** | Overcorrection causes increasing instability.                                   |
| **T.C** | Labyrinthine        | **World Model**      | **Overfitting to Noise**      | WM models noise instead of signal.                                              |
| **D.C** | Semantic Horizon    | **Shutter/WM**       | **Ungrounded inference**      | Distribution shift causes internal rollouts to decouple from boundary evidence. |
| **B.O** | Boundary Overload  | **Boundary**         | **Injection / Overload** | Interface inflow exceeds the declared input capacity.                           |
| **B.E** | Sensitivity Expl.   | **Critic**           | **Fragility**                 | Optimization for a single condition induces high sensitivity to perturbations.  |
| **B.D** | Resource Depletion  | **Boundary/Shutter** | **Starvation**                | Input resources are depleted; internal information volume decays (catastrophic forgetting).                                              |
| **B.C** | Control Deficit     | **Critic / Policy**  | **Value-boundary misalignment / requisite-variety deficit** | The value head can disagree with boundary feedback (AlignCheck), or the policy action repertoire can be too small for the disturbance process (BarrierVariety). |

:::{div} feynman-prose
A few of these deserve special attention.

**D.D (Success)** is not a failure at all. It is what happens when everything works: errors decay to a stable floor. I include it as the reference point. When debugging, you need to know what "healthy" looks like.

**C.D (Mode Collapse)** is among the most common failures in deep learning. The policy becomes obsessed with one solution and ignores everything else. Picture a robot that learns to stand still because standing still is safe, even though its task is to walk. Probability mass concentrates on one mode; exploration vanishes.

**T.E (Wrong Paradigm)** is the most subtle failure. The system is not learning poorly; it *cannot* represent the solution. This is like trying to model a spiral staircase with a flat piece of paper. No matter how cleverly you fold it, the topology is wrong. You need to change the architecture, not tune the parameters.

**D.E (Pilot-Induced Oscillation)** is the opposite of helplessness. The agent is *too* responsive, overcorrecting each error and making things worse. Pilots call this "PIO," and it crashes airplanes. The cure is not more learning but more damping.
:::

(sec-interventions)=
## Interventions (Mitigations)

:::{div} feynman-prose
Now the practical part: what do you *do* when things go wrong?

Interventions are not magic. Each is a targeted surgery for a specific pathology. You would not give antibiotics for a broken bone, and you would not apply gradient clipping to fix mode collapse. The failure mode determines the intervention.

Think of these as a doctor's toolkit. When diagnostics detect a failure signature, they prescribe the corresponding treatment. This differs from the usual ML approach of randomly trying tricks until something works. Here, we are systematic: diagnosis first, then treatment.
:::

(rb-heuristic-fixes)=
:::{admonition} Researcher Bridge: Heuristic Fixes as Typed Surgeries
:class: tip
These interventions correspond mathematically to common RL stabilizers: target networks, clipping, entropy tuning, replay, and resets. Each intervention is triggered by a specific diagnostic condition rather than manual hyperparameter tuning.
:::

:::{div} feynman-prose
Here is how to read the intervention table.

**Surgery ID**: Each intervention is named "Surg" plus the failure mode code. SurgCE is the surgery for C.E (Divergence).

**Target Mode**: Which failure triggers this intervention. When diagnostics see this failure, consider this surgery.

**Target Component**: Which part of the system you are operating on. Change the minimum necessary to fix the problem.

**Fragile (Upgrade) Translation**: The conceptual category. Is it a limiter? A reset? A regularizer? This tells you the *type* of intervention.

**Mechanism**: The actual implementation, what to change and how.
:::

Interventions are external mitigations to restore stability, re-ground the representation, or reduce unsafe update rates.

| Surgery ID     | Target Mode        | Target Component     | Fragile (Upgrade) Translation | Mechanism                                                                                                                                                                                                                                                                                                                               |
|----------------|--------------------|----------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SurgCE**     | C.E (Divergence)   | **Policy/Critic**    | **Limiter / trust region**    | **Gradient Clipping / Trust Region:** Clamp outputs; enforce $D_{\mathrm{KL}}(\pi_{\mathrm{old}}\Vert\pi_{\mathrm{new}})<\delta$.                                                                                                                                                                                                                              |
| **SurgCC**     | C.C (Zeno)         | **WM/Policy**        | **Time-boxing / Rate Limit**  | **Skip-Frame / Latency:** Force fixed $\Delta t$; ignore inputs during cool-down.                                                                                                                                                                                                                                                       |
| **SurgCD_Alt** | C.D (policy-entropy collapse; Nodes 7b/10)    | **Policy**           | **Reset / Reshuffling**       | **Re-initialization:** Reset parameters of the obsession-locked sub-module to random.                                                                                                                                                                                                                                                   |
| **SurgSE**     | S.E (Stumble)      | **Policy / curriculum**      | **Curriculum Ease-off**       | **Curriculum Learning:** Reduce Task Difficulty or Rewind to earlier level.                                                                                                                                                                                                                                                             |
| **SurgSC**     | S.C (Target Chasing)  | **Critic**           | **Parameter Freezing**        | **Target Network Freeze:** Stop updating Target V; switch to slower exponential moving average.                                                                                                                                                                                                                                         |
| **SurgCD**     | C.D (codebook/dead-fibre collapse; Nodes 3/11)     | **Shutter**          | **Feature Pruning**           | **Dead Code Pruning:** Identify and excise unused macro symbols / dead fibres.                                                                                                                                                                                                                                                          |
| **SurgSD**     | S.D (Blindness)    | **Shutter**          | **Augmentation / Ghost Vars** | **Domain Randomization:** Inject noise into $x$ to force the shutter to learn robust macrostates.                                                                                                                                                                                                                                       |
| **SurgTE**     | T.E (Paradigm)     | **Shutter/WM**       | **Architecture Search**       | **Neural Architecture Search (NAS):** Modify shutter+WM class to match topology (e.g., add hierarchy / memory).                                                                                                                                                                                                                         |
| **SurgTC**     | T.C (Overfit)      | **WM**               | **Regularization**            | **Weight Decay / Dropout:** Increase $\lambda \lVert\theta\rVert^2$ penalty.                                                                                                                                                                                                                                                            |
| **SurgTD**     | T.D (Helplessness) | **Policy**           | **Noise Injection**           | **Parameter Space Noise:** Add $\xi \sim \mathcal{N}(0, \Sigma)$ to Policy weights.                                                                                                                                                                                                                                                     |
| **SurgDC**     | D.C (Ungrounded)   | **Shutter/WM**       | **Smoothing / fallback**      | **OOD rejection:** use Node 13 BoundaryCheck ($I(X;K)>0$) and the operational coupling window ({prf:ref}`thm-information-stability-window-operational`) as primary D.C checks; as a per-sample fallback, if nuisance surprisal exceeds $\tau_n$, texture surprisal exceeds $\tau_{\mathrm{tex}}$, or macro surprisal exceeds $\tau_K$, trigger safe stop. |
| **SurgDE**     | D.E (Oscillate)    | **Policy**           | **Damping**                   | **Triggered by OscillateCheck / HolonomyCheck:** reduce policy step size (lower LR), decrease Adam $\beta_1$, increase batch size, or temporarily freeze policy updates until the critic signal is stable.                                                                                                                              |
| **SurgBE**     | B.E (Fragile)      | **Critic**           | **Gain / Lipschitz bound**  | **Spectral Normalization:** Constrain Lipschitz constant of $V(z)$.                                                                                                                                                                                                                                                                     |
| **SurgBD**     | B.D (Starve)       | **Boundary/Shutter** | **Replay Buffer / Reservoir** | **Experience Replay:** Train on historical buffers to prevent catastrophic forgetting.                                                                                                                                                                                                                                                  |
| **SurgBC**     | B.C (Deficit)      | **Critic / Policy** | **Interface re-grounding / controller expansion** | If AlignCheck fails, re-ground the value head against boundary feedback; if BarrierVariety fails, expand the Policy action repertoire (for example with Net2Net). |

:::{div} feynman-prose
Several patterns in these interventions are worth noticing.

First, many interventions involve *slowing down* or *constraining* the system. SurgCE limits how far the policy can move. SurgCC enforces time-boxing. SurgSC freezes target networks. SurgDE reduces learning rates. This reflects a deep principle: when a dynamical system is unstable, try damping first. Let it settle before pushing further.

Second, several interventions involve *adding noise*. SurgTD adds parameter noise. SurgSD adds domain randomization. SurgCD_Alt
resets parameters randomly. This seems counterintuitive: why would chaos help a struggling system? In some cases these failures
reflect a local trap or lost diversity, and noise can provide an escape direction. It can also damage an already well-calibrated
model, so the perturbation, schedule, and recovery signal need to be checked.

Third, consider SurgDC (Out-of-Distribution rejection). The primary boundary signal is Node 13's $I(X;K)>0$ non-collapse sanity check, together with the coupling-window diagnostics; per-sample nuisance, texture, or macro surprisal can serve as a fallback trigger when those quantities spike. The system then stops trusting the internal model and falls back to safe behavior. This is epistemic humility encoded into the control loop.

Finally, only SurgBC (Controller Expansion) and SurgTE (Architecture Search) actually grow the system. All others work within the existing architecture. Most problems can be fixed without architectural changes, but some genuinely require more capacity. The diagnostic system helps you distinguish these cases.
:::

:::{note}
:class: feynman-added
The relationship between failures and interventions is not always one-to-one. Some failures may require multiple interventions applied in sequence, and some interventions may help with multiple failure modes. The table gives the primary mapping, but clinical judgment is still required.
:::
