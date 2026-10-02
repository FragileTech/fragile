SHARED_DYNAMICS_INTRODUCTION = r'''
\section*{Introduction: a common dynamics for geometry and fields}
\addcontentsline{toc}{section}{Introduction: a common dynamics for geometry and fields}
\label{sec:shared-dynamics}

The organizing idea of this program is to construct geometry, causal history,
and field observables from one interacting stochastic population. The gas is
the primitive mathematical object. Its recorded evolution supplies the causal
complex; its fitness supplies an adaptive spatial geometry; its interactions
supply the field readouts. Their common origin fixes a joint probability law
and gives a setting in which their quantitative effects can be balanced.
The proof strategy is to establish control of this dynamics, transport that
control to its recorded observables, and retain it through the continuum and
physical reconstruction.

The gravitational extension follows two routes through this same construction:
fitness derivatives determine curvature directly, while the Fractal Set
supplies convergent geometric operators and a calibrated curvature action.
Section~\ref{sec:quantum-gravity} develops both routes, their comparison,
singular regimes and their common geometry--matter fluctuation hierarchy.

\paragraph{A joint construction.}
Let \(\omega\) denote a complete recorded history and \(P\) its specified
law. Write schematically
\[
\mathscr D(\omega)
 =\bigl(\mathcal C(\omega),g(\omega),\Phi(\omega)\bigr),
\qquad \nu=\mathscr D_*P,
\]
where \(\mathcal C\) is the recorded causal complex, \(g\) the configured
fitness metric, and \(\Phi\) the collection of field observables. The selected
equilibrium history law determines the corresponding equilibrium field law.
This construction retains correlations between the population, geometry and
fields, including those carried by companion choices and cloning events.
Separate marginal laws would leave these correlations undetermined.
Compatibility is therefore addressed at the level of the full transition
and its history, before the observable projections are taken.

\paragraph{The mechanisms share the work of control.}
Selection and cloning redistribute the population through the actual donor
flux. Their signed drift enters the spatial moment and tail estimates on
noncompact spaces. Kinetic dissipation controls velocity, while noise and
transport contribute to exploration and spatial spread. The confinement
argument combines these contributions, together with the specified force
and boundary terms, in a common estimate for the executed update.
Stages~\ref{stage:2}--\ref{stage:4} describe the collision identities,
confinement budgets and long-time laws used for this purpose.

The analytic principle is visible in the combined Lyapunov functional
\[
V_{\mathrm{total}}
 =W_h^2+c_VV_{\mathrm{Var}}+c_BW_b.
\]
Its terms account for coupled population error, internal dispersion and
boundary exposure. A substep can increase one term while reducing another.
The component estimates are evaluated for the same update and assembled
with weights that give the required total drift. Thus the proof uses the
relations between mechanisms as well as estimates for each mechanism.
The moment, Poincar\'e and logarithmic Sobolev estimates developed later
are attached to the identified laws of this coupled system.

\paragraph{Geometry is part of the feedback.}
On the identified positive Hessian branch, the conditional fitness field
determines
\[
g_i=H_i+\epsilon_\Sigma I,
\qquad \Sigma_i\Sigma_i^{\mathsf T}=g_i^{-1},
\]
where \(\Sigma_i\) is the normalized noise-shape factor; the thermostat
supplies its amplitude. Population-dependent fitness therefore shapes the
directions of exploration, and the resulting motion changes the population
from which fitness is computed. Metric derivatives determine the associated
connection and curvature. The configured positive regularization or clipping
rule provides the stated metric control. Regularity, density correction and
finite-step metric balances keep the geometric readout tied to this same
sampling process; their roles are developed in
Stages~\ref{stage:9}, \ref{stage:15} and \ref{stage:16}.

The recorded complex likewise follows the realized interactions and
population distribution. Its spatial sampling responds to the configured
fitness and interaction scales, while causal edges follow increasing
algorithmic time. The spacetime construction uses three spatial coordinates
and this recorded time. The quantitative continuum estimates specify how
the resulting sampling geometry approximates the intended operators.

\paragraph{Interactions and the field action come from the same record.}
The color readout uses the recorded interacting viscous force and its
associated velocity, together with the specified complex representation
and determinant structure. This assigns a precise field role to quantities
that already participate in the population dynamics. The representation and
transport identifications are made in the observable construction of
Stage~\ref{stage:11}.

The complete transition law also determines the history likelihood.
Integrating that likelihood over histories with the same field descriptor
produces the effective action of Stage~\ref{stage:12}. In the notation of
that stage, for a reference history law \(R\) and \(Y=\mathscr D(\omega)\),
\[
S_{\mathrm{alg}}(y)
=-\log\mathbb E_R\!\left[\frac{dP}{dR}\,\middle|\,Y=y\right].
\]
This conditional integration preserves the probability and normalization
of unresolved records. The resulting field law, action and temporal
correlations consequently refer to the same underlying evolution; see the
\hyperref[src:thm-ym-recorded-action-emergence]{recorded-action theorem}.

\paragraph{How the common foundation guides the continuum proof.}
The lossless record representation transports established variance,
entropy and Dirichlet-form estimates with their constants; see the
\hyperref[src:prop-fractal-set-analytic-transfer]{analytic-transfer proposition}.
Mean-field control supplies the population background, fluctuation estimates
retain the relevant random observables, and reconstruction bounds quantify
the resolution error. Their constants are then followed through the
specified population, bandwidth, cutoff and volume families. The final
physical identifications connect these estimates to the equilibrium field
hierarchy and its transfer Hamiltonian.

The strategic claim is that a common interacting dynamics provides both
the observable objects and the balance relations needed to control them
together. The lattice-like complex and the proposed quantum field theory
are successive constructions from that dynamics. The program pursues this
route by proving its compatibility relations and their persistence under
limits. The diagram on the following pages displays how these dependencies
organize the eighteen stages.
\clearpage
'''
