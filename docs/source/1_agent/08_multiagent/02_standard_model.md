(sec-standard-model-cognition)=
# The Standard Model of Cognition: Gauge-Theoretic Formulation

## TLDR

- Calculate local connection covariance for the displayed phase, mode, and feature representations.
- Check which representation freedoms preserve the established channels, boundary operators, and observables.
- Build the scalar potential from the proved deterministic chart-fission drift and compute masses with explicit normalization.
- Use the previous chapter's exact polar representation for belief dynamics.
- Distinguish these identities from the interacting quantum reconstruction; the displayed chiral matter content has a gauge-anomaly obstruction.

## Roadmap

1. Identify representation freedoms and calculate their connection laws.
2. Check the proposed matter representation and scalar couplings.
3. Compare the resulting objects with the established dynamics and reconstruction criteria.

:::{div} feynman-prose
A connection answers a concrete question: how do we compare two vectors written in different local bases? The preceding gauge chapter gives the transformation law and the curvature calculation. Here we apply that machinery to utility phases, update representations, and feature coordinates.

There are two calculations to keep in view. One checks that a proposed field equation transforms covariantly. The other checks that the proposed transformation preserves the agent's actual update maps, boundary data, decoder, and observables. Writing a familiar matrix group completes neither calculation by itself. The chapter records the identities we can establish and tests their compatibility with the proposed matter fields.
:::

*Abstract.* This chapter develops the connection calculus for the candidate representation
$SU(N_f)_C\times SU(r)_L\times U(1)_Y$ and checks its relation to the established agent dynamics.
The update-channel construction determines Kraus rank, while the actual decoder, boundary operators, and update maps determine which represented transformations preserve the agent.
The scalar potential is obtained from the deterministic chart-fission drift. Curvature and mass matrices are computed for the displayed fields and conventions.
The proposed chiral spinor content has an uncancelled color anomaly at $N_f=3,r=2$ and does not furnish the claimed quantum gauge theory.
The non-Dirac belief representation remains the exact polar construction of the preceding chapter.
The final reconstruction ledger distinguishes those established identities from properties of a full interacting field state.

*Cross-references:* This chapter synthesizes:
- {ref}`sec-the-belief-wave-function-schrodinger-representation` (Belief density, phase, and exact polar representation)
- {ref}`sec-the-boundary-interface-symplectic-structure` (Holographic Interface: Dirichlet/Neumann Boundary Conditions)
- {ref}`sec-ontological-expansion-topological-fission-and-the-semantic-vacuum` (Ontological Expansion: Pitchfork Bifurcation, Chart Fission)
- {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` (Capacity-Constrained Metric Law)
- {ref}`sec-the-reward-field-value-forms-and-hodge-geometry` (Helmholtz Equation, Value Field)



(sec-gauge-principle-derivation)=
## The Gauge Principle: Derivation of the Symmetry Group $G_0$

:::{div} feynman-prose
Start with a change of basis that preserves the represented objects. A derivative of that basis produces an extra term. The connection cancels this term, so differentiation respects the change of description. This is the gauge-covariance calculation proved in the preceding chapter.

We examine a phase factor, a complex mode space of dimension $r$, and a complex feature space of dimension $N_f$. Their proposed unitary actions provide concrete matrices on which to perform the calculation. To identify an actual symmetry of the agent, we also check preservation of the update and observation maps. Tensor factors let their actions commute; the kernel of the combined representation determines which transformations act identically.
:::

The preceding chapter establishes the connection transformation law. Here we apply it to the displayed phase, mode, and feature representations, then compare those representations with the agent's actual maps. The derivative calculation uses {prf:ref}`rem-local-gauge-template`; preservation of the boundary and update operators is checked separately.

:::{prf:proposition} Connection covariance with the fixed sign convention
:label: rem-local-gauge-template

Use Hermitian generators, $D_\mu=\partial_\mu-igA_\mu$, and $\Phi'=U\Phi$.
For nonzero coupling $g$, covariance fixes
$$
A'_\mu=UA_\mu U^{-1}-\frac{i}{g}(\partial_\mu U)U^{-1}.
$$
*Proof.* Expand the required identity on an arbitrary section:
$$
D'_\mu(U\Phi)-UD_\mu\Phi
=\big[(\partial_\mu U)-igA'_\mu U+igUA_\mu\big]\Phi.
$$
Its vanishing gives the displayed transformation. For
$D=\partial-igqB$ and $U=e^{iq\alpha}$ this reads
$B'=B+g^{-1}d\alpha$. For $U=1+i\theta^aT_a+O(\theta^2)$,
$[T_b,T_c]=if^{bc}{}_aT_a$ gives
$$
\delta A^a_\mu=g^{-1}\partial_\mu\theta^a-f^{bc}{}_a\theta^b A^c_\mu.
$$
The field representation must already be specified. This identity
constructs covariant derivatives for that representation; the operational
symmetry group is defined separately in
{prf:ref}`def-agent-symmetry-group-operational`. $\square$
:::
### A. $U(1)_Y$: Phase Covariance and Utility Shifts

:::{div} feynman-prose
Adding the same constant to every score preserves their ordering. In the polar representation, a constant phase rotation also preserves the density and phase-gradient current. These are precise invariances of specified quantities.

A position-dependent change is different: its derivative contributes to the current. We can calculate that contribution and the compensating connection transformation explicitly. Whether the original control problem permits the corresponding change of value and reward data is a separate identity to check in its equations.
:::

A constant shift preserves value differences. Potential-based reward shaping has its own policy-invariance identity {cite}`ng1999policy`. The local phase calculation below concerns the polar amplitude and its connection; it is not obtained by replacing a constant shift with an arbitrary function in the original control equations.

:::{prf:definition} Global phase and the scalar transport current
:label: def-utility-gauge-freedom

For the established scalar amplitude $\psi=\sqrt\rho e^{iV/\sigma}$,
$D_i=\partial_i-iA^{\rm ext}_i/\sigma$ gives the spatial transport current
$$
j^i=\sigma\operatorname{Im}(\bar\psi G^{ij}D_j\psi)
=\rho G^{ij}(\partial_jV-A^{\rm ext}_j).
$$
This is the canonical current in {prf:ref}`thm-madelung-transform`.
It is spatial; the corresponding density is $\rho$, not a raised temporal
component of this expression. A constant shift of $V$ changes only the
global phase and preserves both $\rho$ and $j$. For a local shift its
extra derivative is computed in {prf:ref}`ax-local-utility-invariance`.
The source of $V$ is the scalar value problem in
{prf:ref}`thm-the-hjb-helmholtz-correspondence`; $A^{\rm ext}$ remains
separate from the internal matrix comparison connection.
:::
:::{div} feynman-prose
The polar amplitude stores density in its modulus and value in its phase. A constant phase rotation changes neither the density nor the phase gradient. A varying rotation changes the phase gradient by an exact one-form. The following connection calculation accounts for that extra term.
:::

:::{prf:proposition} Local phase compensation and the operational current
:label: ax-local-utility-invariance

The amplitude of {prf:ref}`def-belief-wave-function` admits the local
coordinate change
$$
V'=V+\sigma q\alpha,\qquad
\psi'=e^{iq\alpha}\psi,\qquad B'=B+g_1^{-1}d\alpha,
\qquad D=\partial-ig_1qB.
$$
It preserves $|\psi|^2$ and $\operatorname{Im}(\bar\psi D\psi)$.
*Proof.* The first identity is immediate; the second follows by applying
{prf:ref}`rem-local-gauge-template` and canceling the unit phases.
At fixed connection, instead,
$$
\operatorname{Im}(\bar\psi' D\psi')
=\operatorname{Im}(\bar\psi D\psi)+q\rho\,d\alpha.
$$
Thus a local shift changes the current unless its connection is transformed.
Finite propagation constrains communication; it supplies no cancellation
of this derivative term. For global real $V$, the map from additive
baselines to phases has kernel $2\pi\sigma\mathbb Z$ when written as
$V\mapsto e^{iV/\sigma}$; the phase representation is a quotient of the
additive baseline group. $\square$
:::
:::{div} feynman-prose
Finite propagation controls when information can arrive. It does not alter which transformations preserve a reward function or boundary condition. Local phase covariance is verified by transforming the field and connection together and substituting them in the derivative. This gives an exact statement about the represented fields.
:::

:::{prf:theorem} Abelian compensation in the belief action
:label: thm-emergence-opportunity-field

Write $q=Y/2$. For the phase representation in
{prf:ref}`ax-local-utility-invariance`, set
$D_\mu=\partial_\mu-ig_1qB_\mu$. The first-order action density
$$
\mathcal L_{\rm kin}=\frac{i\sigma}{2}
 (\bar\psi D_t\psi-\overline{D_t\psi}\,\psi)
-\frac{\sigma^2}{2}G^{ij}\overline{D_i\psi}D_j\psi
$$
is gauge invariant on a fixed spatial metric.

*Proof.* Each $D_\mu\psi$ transforms by the same phase as $\psi$, so
both contractions are invariant. To see the terms being compensated,
put $B=0$ and transform only $\psi$. With
$j_i=\operatorname{Im}(\bar\psi\partial_i\psi)$,
$$
\delta\mathcal L_{\rm kin}
=-\sigma q\rho\,\partial_t\alpha
-\sigma^2qG^{ij}j_i\partial_j\alpha
-\frac{\sigma^2q^2}{2}\rho|d\alpha|_G^2.
$$
These follow by multiplying
$\partial_i\psi'=e^{iq\alpha}(\partial_i\psi+iq\psi\partial_i\alpha)$.
The connection transformation cancels all three contributions.
The curvature is $dB$, invariant because $d^2\alpha=0$.
The first-order kinetic term remains distinct from a relativistic scalar
kinetic term; local covariance alone does not equate their dynamics.
The external reward form $A^{\rm ext}$ and $B$ are separate data.
A Hodge decomposition applies to a spatial form in the realization of
{prf:ref}`thm-hodge-decomposition`; it neither equates these two forms
nor makes a connection flat. $\square$
:::
:::{div} feynman-prose
Differentiating a locally rotated amplitude produces a derivative of the rotation. The transformation of $B_\mu$ cancels it. The resulting covariant derivative therefore transforms in the same way as the amplitude.

The curvature $dB$ measures the local failure of this connection to be exact. Interpreting its circulation as accumulated reward requires identifying $B$ with the relevant reward one-form. The covariance identity itself does not establish that identification.
:::

:::{admonition} Why "Opportunity Field"?
:class: feynman-added note

The name refers to the proposed interpretation of the phase connection as a representation of the external value one-form. Its established transformation law compensates local phase changes.

When the connection is identified with that value one-form, its circulation describes the corresponding path dependence. The identification is checked through the model's value equations, rather than inferred from covariance alone.
:::



### B. $SU(r)_L$: Mode Representations and Boundary Operators

:::{div} feynman-prose
Observation and action play different roles at the boundary. We can express that difference through the data imposed on their components. A matrix that mixes the components must preserve those boundary data to qualify as a symmetry.

For example, rotating a nonzero component into one constrained to vanish generally violates the constraint. Counting an input role and an output role therefore does not produce a two-dimensional unitary symmetry or a spacetime chirality.
:::

The boundary interface distinguishes prescribed sensor values and motor fluxes. The mode-space construction below records that distinction and the channel's representation dimension. An $SU(r)$ matrix acts covariantly on the displayed multiplet; preservation of the fixed boundary conditions requires an intertwining identity with their operators.

:::{prf:proposition} Boundary asymmetry and its stabilizer
:label: ax-cybernetic-parity-violation

The sensor and motor boundary data are those of
{prf:ref}`def-dirichlet-boundary-condition-sensors` and
{prf:ref}`def-neumann-boundary-condition-motors`. Their different roles
do not identify Lorentz chirality or a full mode-mixing symmetry.

*Proof.* Even for homogeneous two-component model conditions
$f_1|_\partial=0$, $\partial_nf_2|_\partial=0$, take $f_1=0,f_2=1$.
A constant unitary with $U_{12}\ne0$ gives
$(Uf)_1|_\partial=U_{12}\ne0$. Hence it does not preserve the domain.
A passive change of basis can transform the boundary projectors as well:
$P_D'=UP_DU^{-1}$, $P_N'=UP_NU^{-1}$. Then transformed fields obey
transformed conditions. At fixed projectors the admissible internal group
is their stabilizer, further restricted by the actual update and readout
maps. A spacetime Weyl representation is separate from this boundary
linear algebra. $\square$
:::
:::{prf:definition} Rank of a specified update operation
:label: def-mode-rank-parameter

For a nonzero linear CP operation $\mathcal E$ on the finite belief-operator
space of {prf:ref}`def-belief-operator`, define
$$
r(\mathcal E)=\operatorname{rank}J(\mathcal E),\qquad
J(\mathcal E)=\sum_{ij}|i\rangle\langle j|\otimes
\mathcal E(|i\rangle\langle j|).
$$
This equals the minimal number of Kraus operators. Indeed,
$J=\sum_a|K_a\rangle\!\rangle\langle\!\langle K_a|$ for a Kraus
representation, and spectral decomposition of $J\succeq0$ gives a
representation with exactly $\operatorname{rank}J$ operators.
For a finite family take the maximum of these ranks to obtain a common
padded environment. The zero operation has rank zero.
The matrix comparison uses $r\ge2$ and $N_f\ge2$. A mode representation of dimension $r$ used below is a separately specified
internal fiber; identifying it with this environment requires the maps
between the two spaces. The case $r=2$ labels the doublet comparison.
:::
:::{prf:proposition} Operations, channels, and environment changes of basis
:label: rem-mode-rank-stinespring

The existing GKSL model ({prf:ref}`def-gksl-generator`) supplies a CPTP
semigroup on the finite belief-operator space. An outcome operation has
$$
\mathcal E_y(\rho)=\sum_aK_{ya}\rho K_{ya}^\dagger,
\quad\sum_aK_{ya}^\dagger K_{ya}\preceq I,
\quad p_y(\rho)=\operatorname{Tr}\mathcal E_y(\rho).
$$
For a fixed chosen control, summing over the outcomes gives a channel
when $\sum_{y,a}K_{ya}^\dagger K_{ya}=I$. Averaging controls uses their
probabilities as well. Define $V_yu=\sum_aK_{ya}u\otimes|a\rangle$.
Then $\mathcal E_y(\rho)=\operatorname{Tr}_E(V_y\rho V_y^\dagger)$ and
$V_y^\dagger V_y\preceq I$; $V_y$ is generally a contraction, not an
isometry. Combining outcomes produces the channel isometry, which extends
to a unitary on a larger system-plus-environment space. A selected outcome
is recovered by an environment measurement, not by an unconditional trace.
For $p_y>0$, the normalized state is $\mathcal E_y(\rho)/p_y(\rho)$;
this update is generally nonlinear. Kraus mixing
$K'_a=\sum_bu_{ab}K_b$ with $u\in U(r)$ leaves the CP map unchanged,
since $\sum_a u_{ab}\bar u_{ac}=\delta_{bc}$. Its common phase cancels
from the channel; it is not an observed utility phase. The dilation and Kraus identities are the finite-dimensional constructions in [Watrous, Chapter 2](https://cs.uwaterloo.ca/~watrous/TQI/TQI.2.pdf).
:::
:::{div} feynman-prose
Dirichlet data prescribe a value; Neumann data prescribe a normal derivative. Changing coordinates in their joint representation also changes how these boundary operators are written. Keeping the operators fixed restricts the allowed transformations.

The mode dimension used below belongs to the chosen update representation. Its minimal dilation dimension is determined by the rank of the channel's Choi operator. It is not obtained by counting the words “observation” and “action.”
:::

:::{prf:definition} Specified chiral comparison multiplets
:label: def-cognitive-isospin-multiplet

The comparison uses $\Psi_L$ in the fundamental internal $SU(r)$
representation and $\Psi_R$ in its singlet, with spacetime bundles as in
{prf:ref}`def-cognitive-spinor`. For $r=2$, write
$\Psi_L=(\psi_1,\psi_2)^T$, with each component left Weyl, and one
independent right Weyl field $\Psi_R$. The labels observation, intent and
commitment may name these components in a chosen frame. Their identification
with algorithmic channels is not a linear intertwiner supplied by the
boundary definitions; {prf:ref}`ax-cybernetic-parity-violation` computes
why fixed Dirichlet/Neumann conditions are not preserved by general mixing.
The matrix representation used in the ensuing covariance calculation is
fully specified by these multiplet and singlet actions.
:::
:::{prf:remark} Mode-Rank Generalization
:label: rem-mode-rank-generalization

For general mode rank $r$ (Definition {prf:ref}`def-mode-rank-parameter`), the left-handed field is an $r$-plet in the
fundamental representation of $SU(r)_L$. The doublet comparison sets $r=2$; its generators are $	au_a/2$.

:::

:::{div} feynman-prose
The displayed multiplet and singlet specify different representations: a mode matrix acts on the multiplet and acts trivially on the singlet. This makes their transformation laws explicit.

Calling these fields left and right is notation for this comparison model. Identifying them with Weyl spinors requires the spacetime representation, and identifying them with boundary channels requires preservation of the boundary operators. Neither identification follows from the number of components.
:::

:::{prf:definition} Gauge-Covariant Action Commitment
:label: def-gauge-covariant-action-commitment

The scalar field selects a commitment direction in the specified $\Psi_L$ mode fiber. A frame change acts simultaneously on the scalar and the multiplet. To make action commitment gauge-covariant, we use the ontological order parameter
to define a unit multiplet $n(x) \in \mathbb{C}^r$:

$$
n(x) := \frac{\phi(x)}{\|\phi(x)\|}, \qquad n(x)^\dagger n(x) = 1

$$
where $\phi$ is the ontological order parameter (Definition {prf:ref}`def-ontological-order-parameter`), and $n$ is
defined only when $\phi \neq 0$.

The gauge-covariant **Commitment Projection** is:

$$
\psi_{\text{act}}^{\text{proj}}(x) := n(x)^\dagger \Psi_L(x)

$$

where the projection operator is:

$$
\Pi_n = n n^\dagger, \qquad \Pi_n \Psi_L = n(n^\dagger \Psi_L)

$$

The committed action singlet $\Psi_R$ remains an independent right-handed field; the Yukawa term
couples $\Psi_R$ to the projected amplitude $\psi_{\text{act}}^{\text{proj}}$ through the Hermitian contraction in {prf:ref}`def-decision-coupling`; relaxation does not follow from this coupling alone.

*Justification:* The unit multiplet $n$ encodes the local ontological split and makes the commitment projection intrinsic
to the scalar sector, not an arbitrary choice of basis. Under local $SU(r)$ transformations $\Psi_L \to U(x)\Psi_L$ and
$n \to U(x)n$, so $\psi_{\text{act}}^{\text{proj}} = n^\dagger \Psi_L$ is invariant and $\Pi_n \to U \Pi_n U^\dagger$,
ensuring the projected component is $SU(r)$-covariant. Under $U(1)_Y$, $n$ carries charge $Y_\phi$, so
$\psi_{\text{act}}^{\text{proj}}$ transforms with charge $Y_L - Y_\phi$, matching $\Psi_R$ by Definition
{prf:ref}`def-rep-covariant-derivatives`.

*Remark:* At $\phi=0$, the normalized direction $n$ is undefined, corresponding to decision ambiguity. The agent requires a nonzero ontological split to define a preferred commitment projection.

:::

:::{div} feynman-prose
The projection $n^\dagger\Psi_L$ is a useful exact construction. When both $n$ and $\Psi_L$ transform by the same unitary mode matrix, the two matrices cancel in their inner product. The remaining transformation is determined by their other charges.

The normalized direction exists where $\phi\ne0$. At a zero of $\phi$, the unnormalized contraction $\phi^\dagger\Psi_L$ remains defined, while the normalized projection does not. The Yukawa term couples this selected direction to the singlet; it does not couple every orthogonal mode.
:::

:::{prf:theorem} Mode covariance and the limits of the rank identification
:label: thm-emergence-error-field

For the specified $SU(r)$ representation on the active internal fiber,
$W_\mu=W_\mu^aT_a$ defines
$$
D_\mu\Psi_L=(\partial_\mu-ig_2W_\mu-ig_1Y_LB_\mu/2)\Psi_L.
$$
It is covariant under simultaneous frame and connection transformations
of {prf:ref}`rem-local-gauge-template`. Its curvature is
$F_W=dW-ig_2W\wedge W$.

*Proof.* The product rule gives the connection transformation already
proved there. Expanding the operator commutator on a test section gives
$$
[D_\mu,D_\nu]_{SU(r)}
=-ig_2\{\partial_\mu W_\nu-\partial_\nu W_\mu-ig_2[W_\mu,W_\nu]\},
$$
so $F^a_{W,\mu\nu}=\partial_\mu W_\nu^a-\partial_\nu W_\mu^a
+g_2f^{bc}{}_aW_\mu^bW_\nu^c$.
For $r=2$, $T_a=\tau_a/2$ gives three independent connection components.
This is a matrix connection, with parallel transport in $SU(2)$.

The identification with update rank has a concrete counterexample:
$\mathcal E(\rho)=\operatorname{Tr}(\rho)I_2/2$ has
$J(\mathcal E)=I_4/2$, hence minimal rank four. Two sensor/motor roles
therefore do not determine rank two. Also a unitary acting only on the
environment has
$\operatorname{Tr}_E[(I\otimes u)(\rho\otimes|0\rangle\langle0|)
(I\otimes u^\dagger)]=\rho$; it cannot implement a nontrivial channel.
These distinctions preserve the exact CP construction while preventing
its environment basis freedom from being identified with a physical
weak interaction without an intertwining map. $\square$
:::
:::{div} feynman-prose
An outcome operation is a linear completely positive map before normalization. Its trace is the probability of that outcome. Dividing by that probability produces the conditioned state and generally makes the update nonlinear.

A dilation represents the linear operation using a larger system and an outcome selection. For a trace-preserving channel, an isometry into system times environment suffices. A change of Kraus basis acts on the environment index and leaves the channel unchanged. That freedom is an exact representation redundancy. A physical mode connection requires a further identification with the fields and observables on which it acts.
:::

:::{admonition} Non-Abelian Structure: Order Matters
:class: feynman-added note

Two mode generators can fail to commute, so successive represented rotations can depend on order. Their commutator enters the connection curvature.

This identity concerns the represented rotations. To identify them with actual observation updates, evaluate the update channel under those rotations. In particular, normalized conditioning is generally nonlinear and cannot be replaced by a unitary rotation on the belief space.
:::



:::{prf:definition} Feature representation dimension
:label: def-feature-dimension-parameter

$N_f$ is the complex dimension of the feature fiber used in this chapter's
matrix-field comparison. Its value is obtained from that representation.
Real spatial dimension, the number of sensor channels, and the number of
fermion families are separate quantities. Choosing $N_f=3$ gives the
fundamental representation of $SU(3)$, with eight Lie-algebra generators;
it is not a derivation of that choice from RGB or spatial coordinates.
:::
:::{div} feynman-prose
The dimension $N_f$ specifies the complex feature space in this representation. Three measured channels do not determine an internal $SU(3)$ action: their decoder and update maps decide which changes of coordinates preserve the represented observations.

For a chosen $N_f$-dimensional complex fiber, the traceless Hermitian generators number $N_f^2-1$. This is a dimension count for that matrix algebra, not a derivation of the environment's symmetry.
:::

### C. $SU(N_f)_C$: Feature Representations and Invariant Observables

:::{div} feynman-prose
Feature binding asks how several internal components contribute to one represented object. A decoder gives this question a mathematical form: which changes of the components leave the decoded object unchanged?

Permutations, real rotations, and complex unitary matrices are different candidate answers. We must evaluate the decoder and its update maps under each proposed action. Once a unitary feature action has been identified, its connection compares feature coordinates at neighboring points.
:::

The hierarchical atlas supplies a decoder from internal features to represented concepts ({ref}`sec-stacked-topoencoders-deep-renormalization-group-flow`). Its exact feature symmetries are the transformations preserving that decoder and the update maps. The proposed complex feature representation below makes the unitary connection calculation explicit.

:::{prf:proposition} Macro readout and dynamical confinement
:label: ax-feature-confinement

The established firewall ({prf:ref}`ax-bulk-boundary-decoupling`) removes
texture from the planning variables. In the presence of a specified compact
internal action $R$, the Haar average $P=\int R(u)\,du$ projects onto its
invariant vectors: invariance of Haar measure gives $P^2=P=P^\dagger$.
For invariant readout $O$, $O(R(u)z)=O(z)$ expresses observational
redundancy. Neither identity specifies a field probability measure or a
large-loop expectation. Projecting out a charged component can be done
for every value of the gauge coupling, including zero; hence this
projection alone gives no lower bound on a confining coupling.
:::
:::{div} feynman-prose
Restricting the observable algebra to invariant combinations makes individual charged components unavailable as observables in that algebra. This is a precise restriction on what is measured.

Dynamical confinement is a different calculation: it concerns the state, energy, or Wilson-loop expectations of the interacting system. A boundary restriction on observables does not supply those expectations.
:::

:::{prf:definition} Feature frame group and operational symmetries
:label: def-feature-color-space

The Hermitian feature fiber is $\mathbb C^{N_f}$. Its orthonormal frames
have group $U(N_f)$; frames preserving a specified complex volume element
have group $SU(N_f)$. Real orthonormal frames instead have group $O(N)$,
and permutations give a finite subgroup. These follow respectively from
$U^\dagger U=I$, $\det U=1$, and $R^TR=I$.
For an encoder $E$ and decoder $D$, an operational action also obeys their
intertwining identities, for example $D(R(u)z)=D(z)$ for invariant readout.
These are the symmetries of {prf:ref}`def-agent-symmetry-group-operational`.
The matrix-field calculations below use the displayed $SU(N_f)$ action;
the frame-group calculation does not establish those decoder identities.
:::
:::{div} feynman-prose
A unitary feature basis change preserves the Hermitian inner product. A special-unitary change also preserves the complex volume form. These statements explain which tensor contractions remain invariant.

A particular decoder can preserve a smaller group. The feature representation must therefore be checked against that decoder, rather than identified with all coordinate changes.
:::

:::{prf:theorem} Feature connection and screening calculation
:label: thm-emergence-binding-field

With $t_a=\lambda_a/2$, $\operatorname{tr}(t_at_b)=\delta_{ab}/2$,
$D_\mu=\partial_\mu-ig_sG_\mu^at_a$ has curvature
$$
F^a_{G,\mu\nu}=\partial_\mu G_\nu^a-\partial_\nu G_\mu^a
+g_sf^{bc}{}_aG_\mu^bG_\nu^c.
$$
*Proof.* The commutator expansion is the calculation in
{prf:ref}`thm-emergence-error-field` with the feature generators. The
quadratic commutator produces non-Abelian interaction terms in the
specified Yang--Mills action.
For the attention weight $w(A)=e^{-\sigma A}$ used in
{prf:ref}`thm-texture-confinement-area-law`, the exact threshold is
$$
w(A)\le\varepsilon
\quad\Longleftrightarrow\quad
\sigma A\ge\log(1/\varepsilon),\qquad 0<\varepsilon<1.
$$
Thus a positive area by itself does not ensure strong suppression.
This evaluates that attention weight, not a Wilson-loop expectation.
The proposed infrared theorem {prf:ref}`thm-ir-binding-constraint` cites
this binding theorem as a premise and cannot supply an independent proof
of its confinement conclusion. Neither result is used here to derive the
other. The definition $\mu\,dg_s/d\mu=\beta(g_s)$ fixes notation;
the sign of $\beta$ depends on the quantum theory and its matter content,
and is not obtained from this classical curvature calculation. $\square$
:::
:::{div} feynman-prose
The non-Abelian curvature includes a commutator of connection matrices. Squaring it in the field action produces interaction terms among connection components. That algebraic self-interaction is explicit.

The sign of a renormalization beta function depends on the full matter content and its representations. Wilson-loop decay depends on the field state. Neither quantity is fixed merely by exhibiting a nonzero commutator. The established attention-screening calculation retains its meaning as a screening calculation for its specified kernel.
:::

:::{admonition} The Binding Problem Solved?
:class: feynman-added note

Invariant contractions describe how several represented feature components can contribute to a basis-independent observable. This addresses the representation of a bound feature combination.

Dynamical binding additionally concerns the state and its evolution. A restriction to invariant observables or an imposed attention-screening kernel does not calculate the Wilson-loop expectation of an interacting gauge theory.
:::

:::{prf:proposition} Product representation and its faithful quotient
:label: cor-standard-model-symmetry

The specified mode and feature actions define a representation of
$$
G_0=SU(N_f)\times SU(r)\times U(1)
$$
on their tensor products. The faithful acting group is $G_0/\ker R$,
where $R$ is the combined representation on all fields and readout data.
*Proof.* Actions on distinct factors commute. The representation
homomorphism theorem gives $\operatorname{im}R\simeq G_0/\ker R$.
A center can cancel another center on a tensor product:
$(\zeta I)\otimes(\zeta^{-1}I)=I$. For example
$(S,z)\mapsto zS$ maps $SU(r)\times U(1)$ onto $U(r)$ with kernel
$\{(\zeta I,\zeta^{-1}):\zeta^r=1\}$. Hence a direct product of
frame actions does not prove faithful direct-product symmetry.
Charges of a compact $U(1)$ representation must be characters of its
specified period; an arbitrary real sensitivity only specifies a Lie
algebra action until this period is fixed. The parameters $Y/2$ below
use one common charge normalization. At $N_f=3,r=2$ the Lie algebra is
$\mathfrak{su}(3)\oplus\mathfrak{su}(2)\oplus\mathfrak u(1)$.
This calculation identifies the algebra of the comparison model, rather
than deriving its ranks from communication constraints. $\square$
:::
:::{div} feynman-prose
We now have a candidate tensor-product representation and its covariant derivatives. Setting $N_f=3$ and $r=2$ gives the familiar three Lie-algebra factors.

To identify the represented gauge group, calculate the common central kernel. To identify a symmetry of the algorithm, verify the intertwining identities for its maps. The matter calculation below gives another independent test: the displayed chiral content must satisfy the quantum gauge-anomaly identities.
:::



(sec-matter-sector-chiral-spinors)=
## The Matter Sector: Chiral Spinor Comparison and Anomaly Test

:::{div} feynman-prose
The preceding gauge chapter already supplies a scalar representation of belief density and phase. This section compares it with a chiral spinor field model. Its spinor and internal indices specify new mathematical objects whose relation to belief dynamics must be established by a map between their state spaces.

Finite signal speed and unequal boundary roles do not provide that map. We can nevertheless calculate the spinor model's covariance and test its quantum consistency. The anomaly calculation is particularly decisive for the matter content displayed here.
:::

The scalar belief representation is established in {ref}`sec-the-belief-wave-function-schrodinger-representation`. The following chiral spinor construction is a comparison model on the spin backgrounds defined by {prf:ref}`def-loc-spin-g`. Its representation content and operator identities are examined explicitly. No lift from scalar WFR states to this chiral field space is supplied by boundary asymmetry.

### A. Spinor Sections and Their Representation Content

The latent metric is the object constructed by {prf:ref}`thm-capacity-constrained-metric-law`. The spinor comparison uses the spacetime and bundle data of {prf:ref}`def-loc-spin-g`; the following definitions specify its sections and their current pairing.

:::{prf:definition} Chiral comparison fields and Cauchy data
:label: def-cognitive-spinor

On the four-dimensional spin background of {prf:ref}`def-loc-spin-g`, the
comparison fields are sections of
$$
(S_L\otimes E_L)\oplus(S_R\otimes E_R),\quad
E_L=\mathbb C^r\otimes\mathbb C^{N_f},\quad E_R=\mathbb C^{N_f}.
$$
Their complex ranks are $2rN_f$ and $2N_f$. They form a chiral multiplet;
there is no common internal bundle identifying every left component with
a right component. They can be embedded into
$S\otimes(E_L\oplus E_R)$ with the unused chiral components set to zero.
The kinetic pairing is defined separately on each physical Weyl summand.

Use signature $(-+++)$ throughout. For the particle-physics convention
$i\gamma^\mu D_\mu$, take $\{\gamma^\mu,\gamma^\nu\}=-2g^{\mu\nu}$;
in an orthonormal frame $(\gamma^{\hat0})^2=I$ and
$\bar\Psi=\Psi^\dagger\gamma^{\hat0}$.
The positive one-particle density is the contraction of the conserved
current with the future Cauchy normal:
$$
\|\Psi\|_\Sigma^2=-\int_\Sigma n_\mu j^\mu\,d\Sigma,
\qquad j^\mu=\bar\Psi\gamma^\mu\Psi.
$$
In an orthonormal frame adapted to $\Sigma$, its integrand is
$\Psi^\dagger\Psi$. It is not generally the coordinate component $j^0$.
The divergence theorem proves surface independence from
$\nabla_\mu j^\mu=0$ and zero side flux in the domain considered.
Spacetime $L^2$ is not the Cauchy-data Hilbert space: a nonzero stationary
solution on an infinite time interval has divergent spacetime norm.
The scalar amplitude of {prf:ref}`def-belief-wave-function` remains a
separate representation; no spinor isomorphism follows from adjoining
these components.
:::
:::{div} feynman-prose
The tensor factors keep the counting transparent. In four spacetime dimensions, a left Weyl sector has two complex spin components, $r$ mode components, and $N_f$ feature components. The right sector has two spin components and $N_f$ feature components. The total is $2(r+1)N_f$.

This count also exposes an imbalance: the left sector contains $r$ color fundamentals, while the right sector contains one. Their cubic anomaly contributions have opposite chirality signs. For $N_f=3$ and $r=2$, the remaining coefficient is nonzero. The displayed content therefore does not define the claimed quantum gauge model.
:::

:::{prf:proposition} First-order operator identity in the comparison sector
:label: ax-cognitive-dirac-equation

Write $P=i\gamma^\mu D_\mu$ on a fixed spinor bundle with a compatible
spin and internal connection, using the Clifford convention in
{prf:ref}`def-cognitive-spinor`. Its covariant connection wave operator is
$\Box_D=g^{\mu\nu}(D_\mu D_\nu-\Gamma^\lambda_{\mu\nu}D_\lambda)$.
For a constant scalar mass on the same bundle,
$$
(P-m)(P+m)=\Box_D-m^2
-\frac14[\gamma^\mu,\gamma^\nu][D_\mu,D_\nu].
$$
*Proof.* Evaluate in a normal frame at a point. Compatibility differentiates
no gamma matrix there. Split $\gamma^\mu\gamma^\nu$ into its symmetric
and antisymmetric parts to obtain
$$
P^2=-\tfrac12\{\gamma^\mu,\gamma^\nu\}D_\mu D_\nu
-\tfrac14[\gamma^\mu,\gamma^\nu][D_\mu,D_\nu].
$$
Restore the Christoffel contraction to express the identity covariantly.
Since $[P,m]=0$, the product is $P^2-m^2$.
The commutator includes the spin curvature and
$-i\sum_a g_a F^a_{\mu\nu}T_a$. Even on a flat base, a nonzero internal
curvature contributes a spin-dependent term. With both curvatures zero,
the dispersion is $\omega^2=|k|^2+m^2$ in units $c_{\rm info}=1$.
This calculation neither identifies a scalar wave with a spinor nor
selects a chiral matter representation. In particular a scalar bare mass
cannot pair the unequal bundles of {prf:ref}`def-cognitive-spinor`;
the Yukawa contraction below pairs only its specified components.
The Bellman generator of {prf:ref}`thm-hjb-klein-gordon` does not supply
this first-order spinor equation. $\square$
:::
:::{prf:remark} Covariant derivatives on the chiral bundles
:label: rem-curved-dirac-operator

All spinor occurrences of $\partial_\mu$ in the internal-connection
notation stand for $\nabla^{\rm spin}_\mu$. The scalar multiplet has no
spin connection. The identity
$[D_\mu,D_\nu]=R^{\rm spin}_{\mu\nu}-i\sum_a g_aF^a_{\mu\nu}T_a$
separates spacetime spin curvature from the internal curvatures of
{prf:ref}`thm-three-cognitive-forces`. The Weyl equations and the full
Dirac comparison use the sign convention of
{prf:ref}`def-cognitive-spinor` consistently.
:::
:::{div} feynman-prose
Gamma matrices implement a Clifford algebra on spinors. Squaring a Dirac operator uses that algebra and introduces the curvature of the spin and gauge connections. It therefore yields an operator on spinors with additional curvature terms.

An equality involving that square does not identify a scalar belief amplitude with a spinor. For the algorithm's density and phase dynamics, the exact polar calculation in the preceding chapter supplies the established representation.
:::

### B. The Strategic Connection (Covariant Derivative)

:::{div} feynman-prose
The covariant derivative attaches a connection to each represented tensor factor. On a curved spacetime it also contains the spin connection for a spinor field. Each term acts on its own index, which makes the transformation calculation explicit.
:::

For the represented fields, $D_\mu$ compares neighboring sections using the specified spin and internal connections. Its transformation law is checked on each tensor factor.

:::{prf:definition} The Universal Covariant Derivative
:label: def-universal-covariant-derivative

The operator moving the belief spinor through the latent manifold is:

$$
D_\mu = \underbrace{\partial_\mu}_{\text{Change}} - \underbrace{ig_1 \frac{Y}{2} B_\mu}_{U(1)_Y \text{ (Value)}} - \underbrace{ig_2 T^a W^a_\mu}_{SU(r)_L \text{ (Error)}} - \underbrace{ig_s \frac{\lambda^a}{2} G^a_\mu}_{SU(N_f)_C \text{ (Binding)}}

$$

where $T^a$ ($a = 1, \ldots, r^2 - 1$) are the generators of $SU(r)$ in the fundamental representation (for $r=2$,
$T^a = \tau^a/2$), and $\lambda^a$ ($a = 1, \ldots, N_f^2 - 1$) are the generators of $SU(N_f)$, and:
- **$B_\mu$ (Opportunity Field):** Adjusts the belief for local shifts in the value baseline and path-dependent opportunity
- **$W_\mu$ (Error Field):** Adjusts the belief for the rotation between Prior and Posterior
- **$G_\mu$ (Binding Field):** Adjusts the belief for the permutation of sub-symbolic features

For the right-handed singlet $\Psi_R$, the $SU(r)_L$ generators act trivially, so the $W_\mu$ term drops.

**Operational Interpretation:** The quantity $D_\mu \Psi$ measures the deviation from parallel transport. When $D_\mu \Psi = 0$, the belief state is covariantly constant along the direction $\mu$---all changes are accounted for by the gauge connection. When $D_\mu \Psi \neq 0$, the section varies covariantly in that direction; force is determined by the equations of motion.

:::

:::{prf:definition} Representation-Specific Covariant Derivatives
:label: def-rep-covariant-derivatives

Let $Y_L$, $Y_R$, and $Y_\phi$ denote the $U(1)_Y$ hypercharges of $\Psi_L$, $\Psi_R$, and $\phi$.
Then the covariant derivatives used in {prf:ref}`def-cognitive-lagrangian` are:

$$
\begin{aligned}
D_\mu \Psi_L &= \left(\partial_\mu - i g_1 \frac{Y_L}{2} B_\mu - i g_2 T^a W^a_\mu - i g_s \frac{\lambda^a}{2} G^a_\mu \right)\Psi_L, \\
D_\mu \Psi_R &= \left(\partial_\mu - i g_1 \frac{Y_R}{2} B_\mu - i g_s \frac{\lambda^a}{2} G^a_\mu \right)\Psi_R, \\
D_\mu \phi &= \left(\partial_\mu - i g_1 \frac{Y_\phi}{2} B_\mu - i g_2 T^a W^a_\mu \right)\phi.
\end{aligned}
$$

Gauge invariance of the Yukawa term $\bar{\Psi}_L \phi \Psi_R$ requires
$
Y_R = Y_L - Y_\phi.
$

:::

:::{div} feynman-prose
A covariant derivative compares a field with its transported neighbor. A zero covariant derivative along a path means the field follows that parallel transport. A nonzero derivative measures the difference from it.

This is a geometric comparison. An evolution law requires an action or generator, and an interpretation as prediction error requires the corresponding map to the agent's update variables.
:::

:::{prf:theorem} Anomaly obstruction for the displayed chiral multiplet
:label: thm-smoc-chiral-anomaly-obstruction

For $N_f=3$, the displayed matter multiplets have a nonzero perturbative
color gauge anomaly whenever $r>1$. A copy with $r=2,N_f=3$ also has
an odd number of weak doublets.

*Proof.* The four-dimensional chiral anomaly is proportional to the symmetric generator trace ([Bilal, Lectures on Anomalies](https://arxiv.org/abs/0802.0634)). Treat right-handed fundamentals as left-handed conjugates.
Their cubic symmetric traces have the opposite sign. Thus, per family,
$$
\operatorname{tr}_{L}T^{(a}T^bT^{c)}
-\operatorname{tr}_{R}T^{(a}T^bT^{c)}
=(r-1)\operatorname{tr}_{\mathbf3}T^{(a}T^bT^{c)}.
$$
For $T_8=\operatorname{diag}(1,1,-2)/(2\sqrt3)$,
$\operatorname{tr}T_8^3=-1/(4\sqrt3)\ne0$.
For $r=2$ the difference is already nonzero. Repeating the displayed
family multiplies this trace; changing hypercharges does not alter it.
Writing $q_L=Y_L/2$ and $q_R=Y_R/2$, other traces include
$$
\mathcal A_{SU(N_f)^2U(1)}=\tfrac12(rq_L-q_R),\quad
\mathcal A_{SU(r)^2U(1)}=\tfrac{N_f}{2}q_L,
$$
$$
\mathcal A_{U(1)^3}=N_f(rq_L^3-q_R^3),\qquad
\mathcal A_{{\rm grav}^2U(1)}=N_f(rq_L-q_R).
$$
For $r=2$, there are $N_f$ left weak doublets per family. The usual
four-dimensional $SU(2)$ global obstruction applies when their total
number is odd ([Wang, Wen and Witten](https://arxiv.org/abs/1810.00844)). These are quantum consistency obstructions for the
stated Weyl theory; the classical covariant action remains a definable
functional. No anomaly cancellation is established by the utility-charge
relation or by the scalar mass matrix. A quantum reconstruction therefore
cannot use this multiplet as an anomaly-free matter sector. $\square$
:::


### C. The Yang-Mills Curvature

:::{div} feynman-prose
The commutator of covariant derivatives measures the leading change around an infinitesimal loop. It gives one curvature tensor for each connection.

A finite loop also depends on its path and on global topology. Even a flat connection can have nontrivial holonomy around a noncontractible loop. The local curvature calculation and the global transport calculation answer different questions.
:::

The curvature is computed from the connection by the commutator below; a nonzero connection potential may still be flat.

:::{prf:theorem} Field Strength Tensors
:label: thm-three-cognitive-forces

The commutator of the covariant derivatives $[D_\mu, D_\nu]$ generates three distinct curvature tensors corresponding to each gauge factor.

*Proof.* Computing $[D_\mu, D_\nu]\Psi$ and extracting contributions from each gauge sector:

1. **$U(1)_Y$ Curvature:**

   $$
   B_{\mu\nu} = \partial_\mu B_\nu - \partial_\nu B_\mu

   $$
When $B_{\mu\nu} \neq 0$, the internal opportunity 1-form is non-conservative (Value Curl; Definition
   {prf:ref}`def-value-curl`). The resulting Lorentz-type force generates cyclic dynamics.

2. **$SU(r)_L$ Curvature:**

   $$
   W_{\mu\nu}^a = \partial_\mu W_\nu^a - \partial_\nu W_\mu^a + g_2 f^{abc} W_\mu^b W_\nu^c

   $$
Contracting $W_{\mu\nu}$ with the oriented area of an infinitesimal loop gives the leading internal transport rotation. This is a connection calculation, not an identification with a Bayesian update. Here $f^{abc}$ are the $SU(r)$ structure constants ($\epsilon^{abc}$ for $r=2$).

3. **$SU(N_f)_C$ Curvature:**

   $$
   G_{\mu\nu}^a = \partial_\mu G_\nu^a - \partial_\nu G_\mu^a + g_s f^{abc} G_\mu^b G_\nu^c

   $$
   Binding curvature is a matrix-valued geometric observable. Ontological stress $\Xi$ is a conditional information quantity, compared explicitly in {prf:ref}`lem-binding-curvature-ontological-stress`. The established fission criterion uses $\Xi > \Xi_{\text{crit}}$
   ({ref}`sec-ontological-expansion-topological-fission-and-the-semantic-vacuum`).

$\square$

:::

:::{div} feynman-prose
The three curvatures are computed by the same commutator identity on different representation factors. Their proposed cognitive names identify the connection being discussed; they do not replace the calculation relating that connection to an observable.

In particular, a mutual information requires a joint probability law. Curvature alone specifies no such law. Independent isotropic residuals remain independent after a fixed unitary rotation, so transport by itself need not produce ontological stress.
:::

:::{admonition} Path Dependence and Holonomy
:class: feynman-added note

Parallel transport around an infinitesimal loop detects the curvature contracted with that loop's oriented area. Nonzero curvature somewhere need not change a chosen vector along every path.

A flat connection can still have nontrivial holonomy around a noncontractible loop. Local curvature and global holonomy therefore require separate calculations. Neither determines mutual information without the joint law of the transported variables.
:::

:::{prf:proposition} Transport and conditional texture information
:label: lem-binding-curvature-ontological-stress

Ontological stress is the conditional mutual information of
{prf:ref}`def-ontological-stress`. If $C=(K_t,z_{n,t},K_t^{\rm act})$,
its exact expression is
$$
\Xi=\int D_{\rm KL}\big(P_{X,Y\mid C=c}\Vert
P_{X\mid C=c}\otimes P_{Y\mid C=c}\big)\,P_C(dc),
\quad X=z_{{\rm tex},t},\quad Y=z_{{\rm tex},t+1}.
$$
It vanishes exactly for conditional independence (up to null conditioning
values). A connection matrix alone does not determine this joint law.

*Proof.* This is the conditional relative-entropy definition and the
zero case of Gibbs' inequality. For a concrete counterexample, take
independent isotropic Gaussian residuals $\eta_t,\eta_{t+1}$ at a fixed
macro state and any nonidentity unitary $U$. Put $X=\eta_t$ and
$Y=U\eta_{t+1}$. The conditional law factors, so $\Xi=0$ despite the
nontrivial matrix. In contrast, $Y=UX$ retains information when $X$ is
nondegenerate; it is a different transition law. The firewall specifies
which variables enter planning, not this persistent-transport identity.
Nonzero curvature determines infinitesimal loop transport, not the
transport of every selected path. An open-path transporter also transforms
at its two endpoints, so being the identity is not itself gauge invariant.
A flat connection can have nontrivial holonomy on noncontractible loops.
Thus curvature and $\Xi$ must be computed from their respective geometric
and probabilistic data. $\square$
:::
:::{prf:proposition} Variation of the specified gauge action
:label: cor-gauge-invariant-action

For the product representation, the action
$$
S_g=-\frac14\int\big(B_{\mu\nu}B^{\mu\nu}
+W^a_{\mu\nu}W^{a\mu\nu}+G^a_{\mu\nu}G^{a\mu\nu}\big)d\mu_g
$$
is invariant under its internal frame transformations.
*Proof.* Each curvature transforms by conjugation and its invariant
quadratic contraction is unchanged. For a compactly supported variation
$a_\nu$, $\delta F_{\mu\nu}=\mathcal D_\mu a_\nu-\mathcal D_\nu a_\mu$.
Integration by parts yields
$\delta S_g=\int(\mathcal D_\mu F^{\mu\nu})^a a^a_\nu\,d\mu_g$.
Coupling matter gives $\mathcal D_\mu F^{\mu\nu}=J^\nu$ with
$J^{\nu,a}=-\delta\mathcal L_m/\delta A^a_\nu$, using the curved
divergence of {prf:ref}`thm-yang-mills-equations`.
Flatness gives locally trivial transport on contractible neighborhoods,
not global path independence or a probabilistic stability theorem.
Gauge covariance proves invariance of this action; it does not select
it uniquely from all invariant functionals. $\square$
:::
:::{div} feynman-prose
With the stated Lorentzian signature, the curvature term in the action is not a positive squared norm. Its variation gives a stress tensor whose electric and magnetic energy densities are positive under the conventions proved in the preceding chapter.

This energy statement concerns the field model. Flatness means vanishing local curvature; it does not imply globally trivial transport or a particular level of predictive accuracy.
:::



(sec-scalar-sector-symmetry-breaking)=
## The Scalar Sector: Radial Fission Dynamics and Gauge Masses

:::{div} feynman-prose
The established chart-fission equation gives a concrete starting point for the scalar sector: its deterministic radial drift can be integrated to obtain a potential. We can then compute the potential's stationary points and their stability.

A spacetime kinetic term and a mode-space representation contain additional information beyond this radial equation. The calculations below keep their contributions visible, so the resulting masses refer to the stated field action and normalization.
:::

The deterministic radial drift in {ref}`sec-symmetry-breaking-and-chart-birth` determines the potential used below. The scalar representation and kinetic term then specify the field model in which its gauge mass matrix is calculated.

### A. The Ontological Scalar Field

:::{prf:definition} The Ontological Order Parameter
:label: def-ontological-order-parameter

Let the local chart structure at spacetime point $x$ be described by a complex $SU(r)_L$ multiplet field
$\phi(x) \in \mathbb{C}^r$ (doublet for the $r=2$ comparison):

$$
\phi(x) = r(x)\,n(x), \qquad r(x) := \|\phi(x)\|

$$

where:
1. **Modulus $r(x) \ge 0$:** Represents the **Metric Separation** between daughter queries $\{q_+, q_-\}$ in the Attentive Atlas (Definition {prf:ref}`def-query-fission`).
   - $r=0$: Coalescence (Single Chart / Vacuum)
   - $r>0$: Fission (Distinct Concepts)

2. **Unit multiplet $n(x)$:** Encodes the **Orientation** of the split in the $SU(r)_L$ fiber (the specific feature
   axis along which differentiation occurs), with $n^\dagger n = 1$.

The field $\phi$ transforms in the fundamental representation under the gauge group $SU(r)_L$, coupling it to the
inference spinor.

:::

:::{prf:remark} Gauge-fixed scalar form
:label: rem-ontological-order-parameter-gauge

On a local region where $\phi\ne0$, a gauge fixing its $SU(r)_L$ orientation to a constant unit vector $n_0$ reduces the order parameter to
$\phi(x) = r(x) n_0$ (with $r \ge 0$ after using $U(1)_Y$). In the $r=2$ comparison this is equivalent to the scalar
parametrization $\phi(x) = r(x) e^{i\theta(x)} n_0$ used in the intuitive discussion.

:::

:::{div} feynman-prose
Writing $\phi=\lVert\phi\rVert n$ separates a magnitude from a unit direction wherever $\phi$ is nonzero. The radial chart-separation variable can supply the magnitude in the proposed representation.

For $\phi\in\mathbb C^r$, the unit directions form $S^{2r-1}$. A single phase parameter describes only a circle inside that space. A local gauge choice can simplify the displayed direction, but it is not a global parametrization through zeros or across arbitrary bundle topology.
:::

### B. Derivation of the Scalar Potential

:::{div} feynman-prose
Integrate the negative deterministic radial drift to obtain its potential. Differentiating the result recovers the same drift, providing a direct check of coefficients and signs. This establishes the radial energy landscape used below.
:::

We derive the potential $V(\phi)$ from the stability analysis of the Topological Fission process ({ref}`sec-symmetry-breaking-and-chart-birth`).

:::{prf:theorem} Integration of the established radial drift
:label: thm-complexity-potential

Write $a=\Xi-\Xi_{\rm crit}$ and use the drift
$b(r)=ar-\alpha r^3$ of
{prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`.
At fixed $a,\alpha$ its radial potential, up to an additive constant, is
$$
\mathcal V(r)=-\frac a2r^2+\frac\alpha4r^4
=-\mu^2r^2+\lambda r^4,
\quad\mu^2=a/2,\quad\lambda=\alpha/4.
$$
*Proof.* Differentiate: $-\mathcal V'(r)=ar-\alpha r^3$.
The radial lift $r=\|\phi\|$ gives an invariant quartic on the specified
Hermitian multiplet. The drift fixes this radial function, not its
spacetime kinetic term or its mobility. In particular the complex gradient
is $\partial_{\bar\phi}\mathcal V=(-\mu^2+2\lambda\|\phi\|^2)\phi$;
matching the real radial drift with fixed orientation uses
$\dot\phi=-2\partial_{\bar\phi}\mathcal V$.
The noise term in the antecedent is still part of its stochastic process.
For a realization $dr=b(r)ds+\eta\,dW_s$, its interior generator is
$b\partial_r+\eta^2\partial_r^2/2$, rather than $b\partial_r$.
For example Itô's formula gives
$d\mathbb E[r^2]/ds=2a\mathbb E[r^2]-2\alpha\mathbb E[r^4]+\eta^2$
before any boundary-local-time contribution. Deterministic critical
points therefore do not determine stochastic expectations. $\square$
:::
:::{div} feynman-prose
The linear term determines whether a small separation grows or shrinks. The cubic term limits its growth and produces finite stationary separation above threshold. Integrating these two terms gives a quadratic-plus-quartic radial potential.

The stochastic chart equation also contains noise. Its expectation involves higher moments, so the deterministic equilibrium is not automatically the mean of the stochastic process. The potential calculation identifies the deterministic drift without discarding that distinction.
:::

:::{admonition} The Mexican Hat Potential
:class: feynman-added note

A radial cross-section shows why the quadratic and quartic terms can produce a nonzero equilibrium radius. Along that cross-section, zero separation becomes unstable and the quartic term stabilizes a finite radius.

For a complex $r$-component field, the equal-radius set is $S^{2r-1}$; for a doublet it is $S^3$. The familiar circular brim is the one-complex-component picture. Gauge-equivalent directions represent the same physical configuration.
:::

:::{prf:proposition} Classical radial minima and their orbit
:label: cor-ontological-ssb

For the established $\alpha>0$, hence $\lambda>0$, the minimum is at
$\phi=0$ for $\mu^2\le0$. For $\mu^2>0$, put
$$
v^2=\frac{\mu^2}{2\lambda}=\frac{\Xi-\Xi_{\rm crit}}{\alpha}.
$$
*Proof.* Complete the square:
$\mathcal V=\lambda(\|\phi\|^2-v^2)^2-\lambda v^4$.
The minima are $vS^{2r-1}\subset\mathbb C^r$; for $r=2$ this is $S^3$.
$SU(r)$ is transitive on unit vectors, with stabilizer $SU(r-1)$,
so the orbit has real dimension $2r-1$.
The number $v$ is a classical minimizer radius; it is not, from this
calculation, an expectation under a stochastic or quantum field law.
Gauge-invariant configurations identify the locally gauge-equivalent
orientations. At zeros or in nontrivial bundle sectors, a global
constant-orientation gauge need not exist. $\square$
:::
:::{div} feynman-prose
Above the deterministic threshold, the stable radial separation grows as the square root of the excess control parameter. Below it, the stable radial equilibrium is zero.

Embedding this radius in a complex multiplet produces a sphere of equal-potential directions. Which directions are physically distinct is determined by the represented gauge action and its stabilizer.
:::

### C. Mass Generation

:::{div} feynman-prose
Insert a constant nonzero scalar configuration into its covariant kinetic term. The terms quadratic in the connection give a mass matrix. Its null directions are exactly the generators that annihilate the scalar configuration.

This is a direct representation calculation. The result depends on the scalar's charges, its norm, and the normalization of the generators.
:::

We derive the mass terms for the gauge fields from the covariant kinetic term of the scalar field.

:::{prf:theorem} Gauge mass matrix at the classical scalar minimum
:label: thm-semantic-inertia

Use $\mathcal L_\phi=-(D_\mu\phi)^\dagger D^\mu\phi-\mathcal V(\phi)$
with signature $(-+++)$ and $\phi_0=vn_0$, $n_0^\dagger n_0=1$.
For the combined Hermitian generators
$Q_A=(g_2T_a,g_1Y_\phi I/2)$, the real vector mass matrix is
$$
(M^2)_{AB}=v^2n_0^\dagger\{Q_A,Q_B\}n_0,
\qquad\mathcal L_{\rm mass}=-\tfrac12(M^2)_{AB}A_\mu^A A^{B\mu}.
$$
*Proof.* At a constant vacuum $D_\mu\phi_0=-ivA_\mu^AQ_An_0$.
Symmetrizing the product $A^AA^B$ gives the formula. For real $u$,
$u^TM^2u=2v^2\|(\sum_Au_AQ_A)n_0\|^2\ge0$; its kernel is precisely
the Lie-algebra stabilizer of $\phi_0$.
For $r=2$, $n_0=(0,1)^T$, and $T_a=\tau_a/2$,
$$
M^2=\frac{v^2}{2}
\begin{pmatrix}
g_2^2&0&0&0\\0&g_2^2&0&0\\
0&0&g_2^2&-g_1g_2Y_\phi\\
0&0&-g_1g_2Y_\phi&g_1^2Y_\phi^2
\end{pmatrix}
$$
in the order $(W^1,W^2,W^3,B)$. Therefore
$$
M_W=|g_2|v/\sqrt2,\quad
M_Z=v\sqrt{g_2^2+g_1^2Y_\phi^2}/\sqrt2,
$$
and the neutral vector proportional to $(g_1Y_\phi,g_2)$ is massless
when the denominator is nonzero. Zero couplings are handled directly
by the matrix. The conventional electroweak parameter is
$v_{\rm EW}=\sqrt2v$, giving $M_W=|g_2|v_{\rm EW}/2$.
With $\phi=(v+h/\sqrt2)n_0$, $h$ has kinetic term
$-\tfrac12(\partial h)^2$ and $m_h^2=4\lambda v^2=2\mu^2$.
These are quadratic masses of this classical action. Neither ordering
of latent metric eigenvalues nor a full interacting spectral gap follows
from the capacity-constrained metric variation. $\square$
:::
:::{div} feynman-prose
For the displayed doublet with $\phi_0=vn_0$ and $\lVert n_0\rVert=1$, the charged mass is $g_2v/\sqrt2$. The neutral matrix has one massive combination and one null combination. Using $vn_0/\sqrt2$ instead would change the meaning of $v$ and produce the familiar factor $1/2$.

The mass matrix assigns an energy cost to physical connection fluctuations around this configuration. A simultaneous gauge change of the fields remains a change of description and does not acquire an energy cost.
:::

:::{prf:remark} Orbit directions and texture variables
:label: rem-goldstone-texture

The single complex fundamental has $2r$ real components. Its fixed-radius
orbit has $2r-1$ tangent directions, leaving one radial scalar locally.
The kernel calculation in {prf:ref}`thm-semantic-inertia` counts the
unbroken vector directions. In a local nonzero-vacuum gauge the orbit
directions are removed from the scalar coordinates by the gauge action.
The texture variable in {prf:ref}`ax-bulk-boundary-decoupling` is a
stochastic boundary residual. No bijection between that residual space
and this compact orbit is supplied by the component count. The firewall
and the gauge orbit retain their separate established definitions.
:::
:::{div} feynman-prose
Directions along a gauge orbit describe equivalent scalar configurations. The local field decomposition places the corresponding longitudinal degrees of freedom in the massive vector modes.

The texture variable belongs to the previously defined latent decomposition. Identifying it with a gauge-orbit coordinate requires a map that preserves its observables and dynamics. The gauge-orbit calculation alone does not prove that identification or the texture firewall.
:::



(sec-interaction-terms)=
## The Interaction Terms

:::{div} feynman-prose
We can now calculate invariant couplings among the displayed fields. Contracting their representation indices determines which terms are covariant, and conjugating the interaction determines whether the action is real.

The Yukawa term selects a mode direction through the scalar field. The external term couples a specified current to a prescribed one-form. Their relation to belief transport is checked separately against the established polar equations.
:::

The following contractions couple the displayed scalar, spinor, and connection fields. Their invariance, conjugation, and quadratic mass maps can be checked directly from the representations already specified.

### A. Yukawa Coupling and the Selected Mode

:::{prf:definition} Hermitian Yukawa contraction
:label: def-decision-coupling

For the specified chiral comparison fields and scalar, define
$$
\mathcal L_Y=-\sum_{ij}\left[
Y_{ij}\bar\Psi_{L,i}^{\,a}\phi_a\Psi_{R,j}
+\bar Y_{ij}\bar\Psi_{R,j}\phi_a^\dagger\Psi_{L,i}^{\,a}\right].
$$
Color indices contract with the invariant Hermitian pairing. The second
term is the Hermitian conjugate of the first, including the coefficient.
Its hypercharge phase is
$e^{i(-Y_L+Y_\phi+Y_R)\alpha/2}$, so invariance gives
$Y_R=Y_L-Y_\phi$. This verifies classical covariance and Hermiticity;
the anomaly trace in {prf:ref}`thm-smoc-chiral-anomaly-obstruction`
remains nonzero for the displayed color multiplets.
:::
:::{div} feynman-prose
At $\phi=vn_0$, the Yukawa contraction selects $n_0^\dagger\Psi_L$. The family matrix couples this projected component to $\Psi_R$. Its Hermitian conjugate contains the complex-conjugate family matrix.

This describes precisely which components interact. The orthogonal left-handed modes receive no mass from this term. A dynamical claim about decision commitment would also require identifying these components with the agent's update variables.
:::

:::{prf:theorem} Rank and singular values of the Yukawa mass map
:label: thm-cognitive-mass

At $\phi_0=vn_0$, let $\chi_{L,i}=n_0^\dagger\Psi_{L,i}$.
The Yukawa mass map on family indices is $M=vY$ between these projected
left fields and the right fields. Its nonzero masses are its singular
values.
*Proof.* Substitute the vacuum:
$\mathcal L_Y=-\bar\chi_LM\Psi_R-\bar\Psi_RM^\dagger\chi_L$.
For $M=U_L\operatorname{diag}(m_k)U_R^\dagger$, unitary changes of family
basis diagonalize the kinetic pairings and give $m_k\ge0$.
The full map from the left multiplet has a kernel containing
$(I-n_0n_0^\dagger)\Psi_L$, of dimension $r-1$ per color and family.
Those directions acquire no mass from this single Yukawa contraction.
For one family the paired mass is $v|Y|$; a phase redefinition can make
that one coefficient real. The fluctuation
$\phi=(v+h/\sqrt2)n_0$ couples with coefficient $Y/\sqrt2$.
This quadratic calculation does not show relaxation into a committed
action; unitary mixing can oscillate without asymptotic alignment.
$\square$
:::
:::{div} feynman-prose
For several families, changing orthonormal family bases reduces the mass matrix to its singular values. The nonnegative masses of the coupled modes are therefore $v$ times the singular values of $Y$.

Increasing $v$ increases these masses at fixed $Y$. This is a statement about the displayed quadratic field operator. It does not establish a relaxation rate or a psychological measure of commitment.
:::

### B. External Current Coupling and Exact Polar Belief Dynamics

:::{div} feynman-prose
An external drive is specified independently of the fields being varied. Coupling it to a current defines how that drive enters the field action. The sign, charge, and current normalization then determine its contribution to the equations.

For the agent, the earlier value and WFR equations already state how rewards affect evolution. We compare with those equations directly.
:::

We pair the external reward one-form with the current of the displayed spinor model, then compare its evolution with the previously established polar belief equations.

:::{prf:definition} The Value 1-Form (External Drive)
:label: def-value-1-form-external-drive

We model the external drive as a fixed background 1-form
$A^{\text{ext}}_\mu(z) = (A^{\text{ext}}_0(z), A^{\text{ext}}_i(z))$, encoding both conservative
and non-conservative components of the reward signal (Definition {prf:ref}`def-effective-potential`).
Concretely, $A^{\text{ext}}_0 = -\Phi_{\text{eff}}$ is the conservative potential, while
$A^{\text{ext}}_i$ captures the non-conservative (curl) component.

$$
A^{\text{ext}}_\mu(z) = (A^{\text{ext}}_0(z), A^{\text{ext}}_i(z))

$$

This is an **external background field**, distinct from the internal gauge field $B_\mu$.

**Special case (scalar drive):** If the external reward 1-form is purely temporal, then
$A^{\text{ext}}_\mu(z) = (-\Phi_{\text{eff}}(z), \vec{0})$.

:::

:::{prf:definition} External current pairing in the comparison action
:label: ax-minimal-value-coupling

The comparison action contains $\mathcal L_{\rm drive}=j^\mu A^{\rm ext}_\mu$
with $j^\mu=\sum_{\chi=L,R}\bar\Psi_\chi\gamma^\mu\Psi_\chi$.
Varying $\bar\Psi_\chi$ contributes
$\gamma^\mu A^{\rm ext}_\mu\Psi_\chi$ to its Euler--Lagrange equation.
In an adapted unit-lapse local inertial coordinate frame, a purely
scalar drive $A^{\rm ext}=(-\Phi,0)$ gives
$\mathcal L_{\rm drive}=-\Psi^\dagger\Psi\Phi$.
On a general slice the density is $-n_\mu j^\mu$, as in
{prf:ref}`def-cognitive-spinor`; $j^0$ alone depends on the coordinates.
A time-independent potential does not break time translations just because
its value equation includes a discount parameter. An explicitly varying
background preserves only its actual symmetry subgroup.
:::
:::{div} feynman-prose
The external term is a current–one-form pairing. Varying it gives the corresponding source in the field equations. A Lorentzian action is stationary on solutions; treating every term as a loss to be minimized would change this variational principle.

Transport toward value is established through the actual density and phase equations, including their signs and reaction term.
:::

:::{prf:theorem} Exact scalar representation of the established WFR equations
:label: thm-recovery-wfr-drift

Use the fixed-metric polar construction of {prf:ref}`thm-madelung-transform`.
On a positive-density chart put $a=\sqrt\rho$, $\psi=ae^{iV/\sigma}$,
$p=dV-B$, $v=G^{-1}p$, $D_i=\partial_i-iB_i/\sigma$, and
$Q=-\sigma^2\Delta_Ga/(2a)$. The Hamilton--Jacobi and mass equations
$$
\partial_sV+\tfrac12|p|_G^2+\Phi=0,\qquad
\partial_s\rho+\operatorname{div}_G(\rho v)=r\rho
$$
are equivalent on this chart to
$$
i\sigma\partial_s\psi=
\left[-\frac{\sigma^2}{2}\Delta_B+\Phi-Q+\frac{i\sigma r}{2}\right]\psi.
$$
Here $r$ is the reaction rate, not the internal mode dimension.

*Proof.* Direct differentiation yields
$$
\frac{\Delta_B\psi}{\psi}
=\frac{\Delta_Ga}{a}-\frac{|p|_G^2}{\sigma^2}
+\frac{i}{\sigma}\left(2\langle d\log a,p\rangle_G+
\operatorname{div}_G v\right),\quad
\frac{i\sigma\partial_s\psi}{\psi}
=-\partial_sV+i\sigma\partial_s\log a.
$$
Equating real parts cancels $Q$; equating imaginary parts gives
$\partial_s\rho=-\operatorname{div}_G(\rho v)+r\rho$.
Conversely these equations give the amplitude identity. The inverse is
$\rho=|\psi|^2$, $V=\sigma\arg\psi$ locally, with phase branches differing
by $2\pi\sigma\mathbb Z$. At zeros the density equations remain the
primary description; a global phase lift obeys the circulation constraints
already discussed in {prf:ref}`thm-madelung-transform`.
This is the non-Dirac representation actually established here, with its
state-dependent $-Q$ term. It is not a linear Schrödinger equation.
For a real initial amplitude, $dV=0$, the current is zero when $B=0$
even if $d\Phi\ne0$. Thus an external scalar potential cannot imply
$v=-\nabla\Phi$ by a nonrelativistic reduction. The spinor drive
$\bar\Psi\gamma^\mu A^{\rm ext}_\mu\Psi$ belongs to the separate
comparison action; identifying its dynamics with this scalar system would
require equality of the represented currents and generators. $\square$
:::
:::{div} feynman-prose
The preceding gauge chapter gives an exact polar representation of the WFR equations. Its velocity is determined by the phase gradient and connection. Its amplitude equation contains the reaction term, and the nonlinear wave representation contains the compensating quantum potential.

These terms matter. A real initial amplitude has zero phase current even in a varying external potential. Replacing that current immediately by a potential gradient does not reproduce the same evolution. Using the established polar identity keeps the density, phase, and generator matched.
:::



(sec-cognitive-lagrangian-density)=
## The Classical Comparison Action and Quantum Reconstruction

:::{div} feynman-prose
The action below collects the displayed gauge, scalar, and spinor comparison terms. Its classical variations and transformation laws can be checked directly. The anomaly calculation already rules out interpreting the displayed chiral matter content as the claimed quantum gauge theory.

For the actual belief dynamics, the prior chapter's scalar polar representation remains the established construction. The following ledger keeps its operator results separate from the correlation functions required for an interacting field reconstruction.
:::

We collect the specified local terms into a classical action and calculate their variational consequences.

$$
\mathcal S_{\rm cmp}=\int d^4x\sqrt{-g}\,\mathcal L_{\rm cmp}

$$

:::{prf:definition} Classical comparison action and its quantum obstruction
:label: def-cognitive-lagrangian

The specified matrix and chiral fields define the classical density
$$
\begin{aligned}
\mathcal L_{\rm cmp}={}&-\tfrac14B_{\mu\nu}B^{\mu\nu}
-\tfrac14W^a_{\mu\nu}W^{a\mu\nu}-\tfrac14G^a_{\mu\nu}G^{a\mu\nu}\\
&+\sum_{\chi=L,R}\frac i2\left[
\bar\Psi_\chi\gamma^\mu D_\mu\Psi_\chi
-(D_\mu\bar\Psi_\chi)\gamma^\mu\Psi_\chi\right]\\
&-(D_\mu\phi)^\dagger D^\mu\phi-\mathcal V(\phi)
+\mathcal L_Y+\sum_{\chi=L,R}\bar\Psi_\chi\gamma^\mu A^{\rm ext}_\mu\Psi_\chi.
\end{aligned}
$$
The action is $\int\mathcal L_{\rm cmp}\,d\mu_g$. The symmetric spinor
kinetic term differs from the integrated one-sided form by a boundary
term, using compatibility and the divergence theorem. All contractions
use {prf:ref}`def-cognitive-spinor` and
{prf:ref}`def-rep-covariant-derivatives`; $\mathcal L_Y$ includes its
conjugated matrix coefficients. Units in this comparison are
$c_{\rm info}=\sigma=1$; the WFR identity retains both scales explicitly.
For a homogeneous scalar in a local inertial frame the kinetic term is
$|\partial_t\phi|^2$, and its Hamiltonian density is
$|\partial_t\phi|^2+|\nabla\phi|^2+\mathcal V$.
This checks the relative kinetic sign.

The density is a classical covariant comparison functional. Its chiral
matter has the obstruction in {prf:ref}`thm-smoc-chiral-anomaly-obstruction`;
it is not an established quantum field law. The established agent dynamics
used here are the scalar polar equations and the separately defined
finite-dimensional CP updates, with their proved representation maps.
:::
:::{div} feynman-prose
An action determines a variational problem once its fields and domain are fixed. A quantum expectation also needs a state or measure on the corresponding observables. Gauge covariance of the action and self-adjointness of the specified scalar operator establish their respective identities; neither identifies the full interacting field measure.
:::

:::{div} feynman-prose
Each sector supplies a concrete calculation: curvature variation, scalar Hessian, representation contraction, or current coupling. A simulator can use these formulas only with the same conventions, field content, and evolution for which they were proved.

The reconstruction ledger records the mathematical objects needed to assign quantum correlations. Keeping those objects fixed prevents a scalar spectral estimate from being transferred to a different gauge–fermion model.
:::

**Terms in the comparison action:**

| Sector | Role of the term | Reference |
|:-------|:-----------------|:----------|
| Gauge | Curvature contribution with the stated Lorentzian sign | {prf:ref}`thm-three-cognitive-forces` |
| Spinor comparison | First-order operator on the specified spinor representation | {prf:ref}`ax-cognitive-dirac-equation` |
| Scalar | Negative covariant kinetic contraction for $(-+++)$ and the radial potential | {prf:ref}`thm-complexity-potential` |
| Yukawa | Representation contraction plus its Hermitian conjugate | {prf:ref}`thm-cognitive-mass` |
| External | Pairing of the specified current with an external one-form | {prf:ref}`thm-recovery-wfr-drift` |

### A. Quantum Reconstruction Criteria and Established Operator Results

#### Established Constructions and Reconstruction Dependencies

The local action and its gauge transformation laws are explicit constructions. The scalar kinetic operator has a specified self-adjoint realization. Their established consequences are recorded in {prf:ref}`thm-fragile-constructive-axioms`; the reconstruction dependencies are recorded in {prf:ref}`thm-constructive-specialization-os-wightman`.

The Causal Information Bound controls the stated information functional. The corrected spectral calculation in {prf:ref}`cor-mass-gap-existence` concerns its specified scalar operator. Neither calculation identifies the full interacting gauge-field measure. In particular, covariance of a formula under a simultaneous change of metric and coordinates does not prove invariance of a probability law on a fixed background.

The WFR action ({prf:ref}`def-the-wfr-action`) and the field action ({prf:ref}`def-cognitive-lagrangian`) retain their respective variational meanings. Their laws, states, and generators must be compared explicitly before a conclusion about one is transferred to the other.

#### Reconstruction Ledger

| Property | Established meaning and dependency | Reference |
|:---------|:-----------------------------------|:----------|
| Internal gauge covariance | The connection transformation makes the covariant derivative transform with its field | {prf:ref}`prop-gauge-transformation-connection` |
| Scalar self-adjoint evolution | The stated scalar kinetic form has its specified self-adjoint realization | {prf:ref}`prop-laplace-beltrami-self-adjointness` |
| W0 / OS0: distributional bounds | Smearing specifies observables; bounds must be established for their actual correlation functions | {prf:ref}`thm-constructive-specialization-os-wightman` |
| W1 / OS1: spacetime covariance | The action has geometric covariance; fixed-background invariance of the Schwinger law is a separate identity | {prf:ref}`def-cognitive-lagrangian`, {prf:ref}`thm-constructive-specialization-os-wightman` |
| W2: spectral condition | The reconstructed Hamiltonian has the spectral property stated by the applicable reconstruction theorem | {prf:ref}`thm-smoc-poincare-reconstruction` |
| W3: locality | Local dependence in the action specifies its field equations; operator microcausality concerns the resulting observable algebra | {prf:ref}`def-cognitive-lagrangian`, {prf:ref}`def-lc-aft` |
| W4: cyclicity | The OS/GNS construction generates its Hilbert space from the chosen observable algebra and state | {prf:ref}`thm-smoc-poincare-reconstruction` |
| OS2: reflection positivity | The reflected quadratic form must be evaluated for the same field law and observable algebra | {prf:ref}`thm-smoc-os2-construction` |
| OS3: clustering | A gap bounds centered matrix elements for the same transfer operator; the scalar estimate is not a gauge-sector gap | {prf:ref}`thm-smoc-os3-construction` |
| OS4: graded symmetry | The correlation functions must realize the specified boson/fermion grading | {prf:ref}`def-os-axioms` |

:::{prf:definition} Axiomatic Field Theory (AFT)
:label: def-aft

An **Axiomatic Field Theory (AFT)** is a relativistic quantum field theory whose vacuum correlation
functions satisfy the Wightman axioms (Definition {prf:ref}`def-wightman-axioms`) {cite}`wightman1956quantum`.
Equivalently, if its Euclidean Schwinger functions satisfy the Osterwalder-Schrader axioms
(Definition {prf:ref}`def-os-axioms`), then the OS reconstruction theorem yields a Wightman QFT
{cite}`osterwalder1973axioms,osterwalder1975axioms`.

:::

:::{prf:definition} Wightman Axioms (W0-W4)
:label: def-wightman-axioms

For the Wightman comparison, let $\Phi_A(x)$ be operator-valued tempered distributions on a positive Hilbert space with a common invariant dense domain, and let
$|\Omega\rangle$ be the vacuum. The Wightman functions are
$W_n(x_1,\ldots,x_n) := \langle \Omega | \Phi_{A_1}(x_1)\cdots\Phi_{A_n}(x_n) | \Omega \rangle$.
The axioms {cite}`wightman1956quantum` are:

1. **W0 Temperedness:** Each $W_n$ is a tempered distribution in $\mathcal{S}'((\mathbb{R}^4)^n)$.
2. **W1 Poincare Covariance:** There exists a unitary representation $U(a,\Lambda)$ of the proper
   orthochronous Poincare group with
   $U(a,\Lambda)\,\Phi_A(x)\,U(a,\Lambda)^{-1} = S_A{}^B(\Lambda)\,\Phi_B(\Lambda x + a)$ and
   $U(a,\Lambda)|\Omega\rangle = |\Omega\rangle$.
3. **W2 Spectral Condition:** The joint spectrum of translation generators $P^\mu$ lies in the closed
   forward light cone, and $P^\mu|\Omega\rangle=0$.
4. **W3 Locality (Microcausality):** For spacelike separation $(x-y)^2>0$ in signature $(-+++)$,
   $[\Phi_A(x),\Phi_B(y)]_\pm = 0$, with graded commutator chosen by spin-statistics.
5. **W4 Vacuum Cyclicity:** The set of vectors generated by polynomials in smeared fields acting on
   $|\Omega\rangle$ is dense in the Hilbert space.

:::

:::{prf:definition} Osterwalder-Schrader Axioms (OS0-OS4)
:label: def-os-axioms

For a specified Euclidean correlation family $S_n$, the following labels summarize the reconstruction properties. This list does not assert that the comparison action defines such a family. The
Osterwalder-Schrader axioms {cite}`osterwalder1973axioms,osterwalder1975axioms` are:

1. **OS0 Temperedness:** Each $S_n$ is a tempered distribution in $\mathcal{S}'((\mathbb{R}^4)^n)$.
2. **OS1 Euclidean Covariance:** $S_n$ is invariant under the Euclidean group $E(4)$.
3. **OS2 Reflection Positivity:** For any polynomial $F$ of smeared fields with support in positive
   Euclidean time, $\langle \Theta F \cdot F \rangle_E \ge 0$, where $\Theta$ is time reflection.
4. **OS3 Cluster Property:** $S_{m+n}(x_1,\ldots,x_m,x_{m+1}+a,\ldots,x_{m+n}+a) \to
   S_m(x_1,\ldots,x_m)\,S_n(x_{m+1},\ldots,x_{m+n})$ as $|a|\to\infty$.
5. **OS4 Symmetry:** $S_n$ is symmetric under permutations (graded symmetry for fermions).

The full reconstruction theorem also includes growth control on the correlation family; the index OS0 here includes that requirement when the theorem is invoked. Vacuum uniqueness is the vacuum-sector property associated with clustering.

:::

(sec-smoc-generalized-aft)=
#### A.0 Generalized AFT (Locally Covariant/Algebraic)

:::{prf:definition} The Background Category $\mathrm{Loc}_{\mathrm{Spin},G}$
:label: def-loc-spin-g

Fix the specified compact comparison group $G=G_0$. The category $\mathrm{Loc}_{\mathrm{Spin},G}$ has objects
$(\mathcal{M}, g, \mathfrak{o}, \mathfrak{t}, \mathcal{S}, P_G, A^{\text{ext}})$ where:
1. $(\mathcal{M}, g)$ is a 4D globally hyperbolic Lorentzian manifold with orientation
   $\mathfrak{o}$ and time orientation $\mathfrak{t}$.
2. $\mathcal{S}$ is a spin structure on $(\mathcal{M}, g)$.
3. $P_G$ is a principal $G$-bundle over $\mathcal{M}$ (fixed topology).
4. $A^{\text{ext}}$ is a fixed background 1-form (the external drive).

Morphisms $\chi:(\mathcal{M}, g, \mathfrak{o}, \mathfrak{t}, \mathcal{S}, P_G, A^{\text{ext}})
\to (\mathcal{M}', g', \mathfrak{o}', \mathfrak{t}', \mathcal{S}', P_G', A^{\text{ext}\prime})$
are smooth isometric embeddings with causally convex image that preserve $\mathfrak{o}$ and
$\mathfrak{t}$, admit a lift to the spin bundles, and are covered by a bundle morphism
$\tilde{\chi}:P_G \to P_G'$ with $\chi^*A^{\text{ext}\prime} = A^{\text{ext}}$.
Internal gauge connections are dynamical fields; only the underlying bundle $P_G$ is background data.

:::

:::{prf:remark} Fixed Bundle, Dynamical Connection
:label: rem-loc-spin-g-connection

Fixing $P_G$ selects the topological sector for the gauge fields; the connection 1-forms are
sections of the affine bundle of connections on $P_G$ and remain dynamical fields. Connections themselves are gauge dependent; physical observables are their gauge-invariant combinations. The LC-AFT
assignment is the functor $\mathcal{A}:\mathrm{Loc}_{\mathrm{Spin},G} \to *\mathrm{Alg}$,
so morphisms act by pullback on background data and by *-homomorphisms on algebras.

:::

:::{prf:definition} Locally Covariant AFT (LC-AFT)
:label: def-lc-aft

A **Locally Covariant AFT** is a covariant functor
$\mathcal{A}:\mathrm{Loc}_{\mathrm{Spin},G} \to *\mathrm{Alg}$ that assigns to each object
$(\mathcal{M}, g, \mathfrak{o}, \mathfrak{t}, \mathcal{S}, P_G, A^{\text{ext}})$ a *-algebra
$\mathcal{A}(\mathcal{M})$ of gauge-invariant observables, together with a net of subalgebras
$\mathcal{A}_{\mathcal{M}}(O) \subset \mathcal{A}(\mathcal{M})$ for causally convex regions
$O \subset \mathcal{M}$, such that {cite}`haag1992local,brunetti2003locally`:

1. **Isotony:** If $O_1 \subset O_2$, then $\mathcal{A}_{\mathcal{M}}(O_1) \subset \mathcal{A}_{\mathcal{M}}(O_2)$.
2. **Locality:** If $O_1$ and $O_2$ are spacelike separated, then
   $[\mathcal{A}_{\mathcal{M}}(O_1),\mathcal{A}_{\mathcal{M}}(O_2)]_\pm = 0$.
3. **Local Covariance:** For any morphism $\chi$ in $\mathrm{Loc}_{\mathrm{Spin},G}$, the induced
   *-homomorphism $\alpha_\chi := \mathcal{A}(\chi)$ is injective and satisfies
   $\alpha_\chi(\mathcal{A}_{\mathcal{M}}(O)) = \mathcal{A}_{\mathcal{M}'}(\chi(O))$, with
   $\alpha_{\chi_2 \circ \chi_1} = \alpha_{\chi_2} \circ \alpha_{\chi_1}$ and
   $\alpha_{\mathrm{id}} = \mathrm{id}$.
4. **Time-Slice:** If $O$ contains a Cauchy surface of $\mathcal{M}$, then $\mathcal{A}_{\mathcal{M}}(O)$ generates
   $\mathcal{A}(\mathcal{M})$.
5. **Gauge Invariance:** The physical algebra is the subalgebra invariant under vertical
   automorphisms of $P_G$; a constrained realization specifies its constraint quotient before assigning physical states.
6. **State Regularity (Microlocal Spectrum):** Physical states are positive linear functionals
   with the microlocal regularity appropriate to their represented fields. For basic free KG/Dirac fields this is the Hadamard two-point condition; composite fields require their own distributional products and bounds. No such products are supplied by this definition.

:::

:::{prf:proposition} Field reconstruction versus a net of algebras
:label: thm-lc-aft-special-cases

A locally covariant net as defined in {prf:ref}`def-lc-aft` specifies
algebras and their maps. Its definition alone supplies neither a preferred
vacuum nor tempered point fields. To apply a field reconstruction theorem,
the correlation distributions, their positivity, covariance, spectral
or Euclidean regularity, and the required growth conditions must be
verified for one and the same field family. This is the meaning of the
reconstruction criterion in {prf:ref}`thm-smoc-poincare-reconstruction`.

The distinction has an elementary state-level example. A direct sum of
two positive vacuum sectors, with a convex-mixture vacuum state, retains
locality and positive energy. For the central projection $P$ onto one
summand with vacuum weight $0<p<1$,
$\omega(P\alpha_a(P))-\omega(P)^2=p(1-p)$ for every translation $a$.
Clustering therefore does not follow just from locality and positive
energy. A time-independent external background also need not preserve
spatial translations or Lorentz boosts. These properties must be checked
on its actual stabilizer rather than inferred from stationarity.
:::
:::{prf:remark} Use of the reconstruction theorem
:label: cor-aft-validity-yang-mills

The OS theorem is applied to a specified Schwinger family satisfying the
full regularity, symmetry, covariance and reflected-positivity requirements
of its chosen version. It reconstructs the corresponding Hilbert space,
fields and positive-energy representation; it does not prove that a
formal action supplies those Schwinger functions. In this chapter the
classical comparison action, finite CP maps, and scalar operator
realization are distinct constructed objects. The records below give
no OS verification for an interacting chiral gauge measure associated
with {prf:ref}`def-cognitive-lagrangian`.
:::
:::{prf:remark} Symmetries of the background
:label: rem-aft-scope

Poincare covariance requires a background and state invariant under that
group. A generic fixed spatially varying drive is not translation invariant,
even if time independent. A curved background is described by its actual
isometries or the local-covariance comparison category. Defining that
category does not construct its interacting field functor.
:::
(sec-constructive-aft-axioms)=
#### A.0b Algebraic and Analytic Construction Criteria

The following definitions organize locality, invariant observables, resolution, propagation, gluing, and stability for a proposed construction. Their interpretation as properties of a field theory requires the same specified observable algebra, state, and evolution throughout. The verified operator identities and their scope are recorded in {prf:ref}`thm-fragile-constructive-axioms`.

:::{prf:definition} Local-net comparison criterion
:label: ax-constructive-locality

For each oriented Riemannian manifold $(\mathcal{M}, g)$ (boundary allowed), there is a net of
local observable *-algebras $\mathcal{A}_{\mathcal{M}}(\mathcal{O})$ for open regions
$\mathcal{O} \subset \mathcal{M}$ with isotony:
$
\mathcal{O}_1 \subset \mathcal{O}_2 \Rightarrow
\mathcal{A}_{\mathcal{M}}(\mathcal{O}_1) \subset \mathcal{A}_{\mathcal{M}}(\mathcal{O}_2).
$
Algebras of causally disjoint regions commute (graded for fermions) with causal separation defined
by Definition {prf:ref}`def-causal-interval`.
:::

:::{prf:definition} Gauge-invariant observable subalgebra
:label: ax-constructive-gauge-physical

There is a compact gauge group $G$ acting locally on fields. The physical observable algebra is
the gauge-invariant subalgebra:
$
\mathcal{A}^{\mathrm{phys}}_{\mathcal{M}}(\mathcal{O}) =
\mathcal{A}_{\mathcal{M}}(\mathcal{O})^{G}.
$
Only gauge-invariant elements represent physical observables.
:::

:::{prf:remark} Operational resolution and correlation distributions
:label: ax-constructive-finite-resolution

The positive Levin length is the operational resolution scale already
defined in {prf:ref}`def-levin-length`. It specifies distinguishability
of observations. Temperedness of a correlation distribution instead means
continuity on Schwartz test functions, with seminorm estimates for that
distribution. The former definition alone gives neither these estimates
nor uniform bounds on all $n$-point distributions. Such bounds are not
added to the resolution definition or used as proved consequences here.
:::
:::{prf:axiom} Finite Propagation
:label: ax-constructive-finite-propagation

There exists a maximum information speed $c_{\mathrm{info}}$; causal influence is restricted to the
causal interval determined by $c_{\mathrm{info}}$ (Definition {prf:ref}`def-causal-interval`).
:::

:::{prf:definition} Local-action comparison criterion
:label: ax-constructive-local-action

The dynamics are generated by a local action functional
$\mathcal{S} = \int_{\mathcal{M}} \mathcal{L}(\Phi, D\Phi, g)\,d\mathrm{vol}_g$
with $\mathcal{L}$ a local density built from covariant fields and derivatives. For disjoint
subregions, the action decomposes additively and the induced dynamics glue consistently. The local
algebra is generated by (smeared) field polynomials supported in $\mathcal{O}$.
:::

:::{prf:remark} Three uses of positivity
:label: ax-constructive-positivity

Positive belief matrices belong to {prf:ref}`def-belief-operator`.
The scalar closed quadratic form supplies a self-adjoint operator and
its spectral semigroup. A Hilbert-space completion of a *-algebra uses a
positive functional $\omega$, via
$\langle a,b\rangle=\omega(a^*b)$ and its null quotient.
These are different constructions. Merely specifying a map on an
algebra as a positivity-preserving semigroup does not define this
functional or the full field Hilbert space. Reflection positivity is the
additional reflected correlation identity explicitly tested above.
:::
:::{prf:remark} Curvature and interaction diagnostics
:label: ax-constructive-nontriviality

A nonzero curvature observable records a nonflat connection. It does not
by itself prove a non-Gaussian quantum interaction: a free abelian field
can have nonzero curvature fluctuations. Nontriviality of a reconstructed
field law is assessed on its correlation functions and observable
algebra. The commutator in the classical non-Abelian curvature explicitly
produces nonlinear terms in the specified classical action.
:::
:::{prf:remark} Thermodynamic vs. Resolution Limit
:label: rem-thermo-vs-levin-length-smoc

The continuum limit used in this volume is the **population/thermodynamic limit** (large $N$ with
empirical measures converging to a density) at **fixed** Levin length $\ell_L>0$
({ref}`sec-mean-field-metric-law`). The Levin length is an operational resolution bound (Axiom
{prf:ref}`ax-constructive-finite-resolution`), not a regulator to be sent to zero. Taking
$\ell_L \to 0$ would exit the framework by violating the Causal Information Bound and is **not**
required for validity.

:::

#### A.0c Established Constructions and Their Scope

:::{prf:remark} Construction identities and analytic realization
:label: thm-fragile-constructive-axioms

The architecture declares its local action, internal gauge action, and
finite-resolution observations. Gauge covariance follows from the
connection identities of {prf:ref}`prop-gauge-transformation-connection`.
The chosen scalar kinetic form has the self-adjoint realization of
{prf:ref}`prop-laplace-beltrami-self-adjointness`. These facts do not verify
every constructive QFT axiom for the interacting continuum field law:
self-adjointness of a spatial scalar operator does not prove reflection
positivity of another path law, and the former information-based gap
argument is corrected in {prf:ref}`thm-mass-gap-constructive`.
The assertions of this record are the stated construction identities;
it supplies no unconditional OS or gauge mass-gap theorem.
:::
:::{prf:proposition} Linear background-field Green operators
:label: lem-smoc-green-hyperbolic

For the smooth globally hyperbolic backgrounds of
{prf:ref}`def-loc-spin-g`, a fixed smooth KG connection-wave operator is
normally hyperbolic. The square identity in
{prf:ref}`ax-cognitive-dirac-equation` shows that a fixed compatible Dirac
operator is prenormally hyperbolic. The standard Cauchy theorem for these
linear operators gives advanced and retarded Green operators on compactly
supported sections. Their supports lie in the respective causal future
and past of the source. Smooth fixed mass and curvature endomorphisms
are lower-order terms and preserve the principal symbol.
This checks the linear-operator setting of the Cauchy theorem. When the
connection and matter evolve together through the nonlinear interacting
action, they are not fixed coefficients of that theorem; the conclusion
here concerns the fixed background operators only. The Green-operator theorem and its square-root property are established in [Bär, Green-hyperbolic operators](https://arxiv.org/abs/1310.0738).
:::
:::{prf:theorem} Time-slice identity for the linear equation quotient
:label: lem-smoc-time-slice

For a fixed linear Green-hyperbolic operator $P$ from
{prf:ref}`lem-smoc-green-hyperbolic`, the equation quotient is generated
by test sections in a neighborhood $O$ of a Cauchy surface.

*Proof.* Let $G_{\rm ret}$ and $G_{\rm adv}$ be the Green operators and
$f$ a compactly supported test section. Choose a smooth time cutoff $\chi$
which is zero to the past and one to the future, with transition in $O$.
Put $h=(1-\chi)G_{\rm ret}f+\chi G_{\rm adv}f$.
Causal support and global hyperbolicity make $h$ compactly supported
(the time transition is chosen between two Cauchy surfaces inside $O$).
Using $PG_{\rm ret}f=PG_{\rm adv}f=f$ gives
$$
f-Ph=[P,\chi](G_{\rm ret}-G_{\rm adv})f.
$$
The right side has support in the transition region in $O$. Thus $f$
and an $O$-supported test section agree modulo the field equation.
For a free CCR/CAR algebra constructed on this quotient, its generators
therefore obey the time-slice property. This proves the quotient identity;
it does not construct the interacting SMoC algebra. $\square$
:::
:::{prf:remark} Scope of the free-field state result
:label: lem-smoc-hadamard-existence

The Hadamard condition concerns the short-distance wavefront structure
of the two-point distribution of a specified field state. Its familiar
free KG/Dirac existence results concern the linear background operators
of {prf:ref}`lem-smoc-green-hyperbolic` and their corresponding CCR/CAR
algebras. They do not produce the higher correlation functions or
renormalized products of the interacting comparison action. In particular
a free Hadamard covariance is not a construction of a chiral gauge state,
and its ultraviolet wavefront set supplies no infrared mass-gap estimate.
The present reconstruction record uses no interacting-state existence
conclusion from this free-field comparison.
:::
:::{prf:remark} Which objects the comparison criteria describe
:label: rem-constructive-axiom-relations

The preceding net and action criteria describe the data of an algebraic
field theory. The operational speed and resolution come from the earlier
agent definitions. The scalar form and CP maps have the constructions
stated in {prf:ref}`thm-fragile-constructive-axioms`. These facts remain
separate until explicit maps identify their algebras, states and evolution.
The OS and AQFT records below therefore report construction dependencies,
not additional premises asserted for the algorithm.
:::
:::{prf:remark} OS reconstruction dependency record
:label: thm-constructive-specialization-os-wightman

The flat stationary field sector fixes the geometry and the candidate
Schwinger functions. The reconstruction theorem
{prf:ref}`thm-smoc-poincare-reconstruction` applies to functions satisfying
its stated OS requirements. The architecture and finite resolution alone
do not verify them. In particular the prior clustering route used
{prf:ref}`thm-mass-gap-constructive`; that result now supplies only a
fixed compact scalar gap and does not establish a Yang--Mills gap.
Thus this record is not a verification of OS0--OS4 for the full interacting
action. The finite algebraic and operator constructions retain their own
proved statements; a gauge-field reconstruction cannot use clustering
derived from the very spectral assertion it is meant to justify.
:::
:::{prf:remark} Role of the construction record
:label: rem-constructive-axioms-use

The record fixes which geometry, observable algebra, state and generator
belong to each comparison. It prevents a theorem about the finite belief
operator, scalar spatial Hamiltonian, or linear wave equation from being
used for a different interacting field law. The full-field conclusions
are restricted by the explicit anomaly and reflected-positivity
calculations, while the finite and scalar identities retain their proved
content.
:::
:::{prf:remark} Dependency Map (Constructive → OS/Wightman)
:label: rem-constructive-dependency-map

```{mermaid}
graph TD
  A[Specified action and gauge transformation] --> B[Gauge covariance identities]
  C[Specified scalar quadratic form] --> D[Scalar self-adjoint operator]
  D --> E[Operator-specific spectral estimate]
  F[Specified field law and observable algebra] --> G[Check the OS identities for this law]
  G --> H{OS requirements verified}
  H -->|Yes| I[Apply OS reconstruction]
  I --> J[Reconstructed Hilbert space and fields]
  K[Construction dependency record] --> G
  E --> L[Decay for the same scalar operator]
```
:::

:::{prf:remark} Status of the reconstruction comparison
:label: rem-os-wightman-hypotheses-checked

The corrected dependency record is
{prf:ref}`thm-constructive-specialization-os-wightman`. In particular,
{prf:ref}`thm-smoc-os3-construction` does not establish clustering of the
interacting field law from the compact scalar gap. Metric covariance of
an action, positivity of a measure, and reflection invariance are distinct
from positivity of all reflected Gram matrices. Each OS identity refers
to the same Schwinger family. The previous declaration of complete
verification is therefore not retained as an antecedent of the gauge
chapter's spectral conclusions.
:::
:::{prf:remark} Algebraic properties of the established constructions
:label: thm-haag-kastler-constructive

The earlier results supply a finite belief-operator algebra with CPTP
maps ({prf:ref}`def-gksl-generator`), a scalar self-adjoint form realization
({prf:ref}`prop-laplace-beltrami-self-adjointness`), and the linear time-slice
quotient of {prf:ref}`lem-smoc-time-slice`. Each conclusion refers to its
own space and generator. A functor satisfying
{prf:ref}`def-lc-aft` would additionally specify the local observable
algebras, embeddings and their composition for the interacting fields.
Those maps are not constructed by naming the category or writing a local
action. Consequently the earlier declaration of an interacting
Haag--Kastler construction is not a consequence of these results.
The finite algebra and linear quotient retain the identities proved above.
:::
:::{prf:remark} Dependency record for local observables
:label: rem-haag-kastler-hypotheses-checked

The construction record {prf:ref}`thm-fragile-constructive-axioms`
establishes the indicated gauge identities and scalar realization.
The time-slice calculation establishes a linear equation quotient.
Neither statement is a definition or proof of the full interacting
local observable net. The algebraic requirements in
{prf:ref}`def-lc-aft` are comparison criteria, and are not imported as
extra properties of the agent. This keeps the direction of dependence
from explicitly constructed algebras to their verified properties.
:::
(sec-smoc-os2-os3-poincare)=
#### A.1 Reflection Positivity and the Specified State

:::{prf:theorem} Reflection positivity for a constructed reversible path law
:label: thm-smoc-os2-construction

There is an exact positive result for the existing finite-state
reversible Markov sector discussed in {prf:ref}`def-gksl-generator`.
Let $P_t=e^{tL}$ be its transition semigroup with stationary law $\pi$ and
detailed balance $\pi_xP_t(x,y)=\pi_yP_t(y,x)$. For the stationary two-sided
path law and a bounded cylinder functional $F$ of positive times, define
$\Theta F$ by complex conjugation and time reflection. Then
$$
\mathbb E[\Theta F\,F]
=\sum_x\pi_x\left|\mathbb E[F\mid X_0=x]\right|^2\ge0.
$$
*Proof.* The Markov property makes future and past conditionally independent
given $X_0$. Detailed balance identifies the conditional reversed-past
law with the future law. Conditional expectation therefore factors into
conjugate factors, and averaging gives the displayed identity. For
$F=f(X_t)$ it becomes $\|P_tf\|_{L^2(\pi)}^2$.
The scalar form in Appendix E also has its own spectral semigroup; its
relation to a path law is through its proved Feynman--Kac realization.
This path-law identity is not an identification of either construction
with the Wilson-loop or chiral-field functional of
{prf:ref}`def-cognitive-lagrangian`. No such identification is assumed.
$\square$
:::
:::{prf:theorem} Capacity and reflected positivity are different inequalities
:label: thm-os2-closure-semigroup

Reflection symmetry and finite information capacity alone do not imply
reflection positivity. Moreover positivity of a scalar semigroup does
not identify a separate field measure with that semigroup.

*Proof.* On two spins $x,y\in\{-1,1\}$ take
$p_J(x,y)=e^{-Jxy}/(4\cosh J)$ with $J>0$ and reflection exchanging
$x$ and $y$. This is a strictly positive, reflection-invariant law on
four states, of entropy at most $\log4$. For the positive-side function
$F(y)=y$,
$$
\mathbb E[\Theta F\,F]
=\frac{2e^{-J}-2e^J}{2e^{-J}+2e^J}=-\tanh J<0.
$$
This disproves the capacity-plus-reflection inference. For any already
constructed self-adjoint $H$ bounded below and any vector $u$,
$\langle u,e^{-tH}u\rangle=\|e^{-tH/2}u\|^2\ge0$.
To use this identity for a path functional one must first derive the
map $F\mapsto u_F$ and its correlation identity, as was done in
{prf:ref}`thm-smoc-os2-construction`. Self-adjointness of an unrelated
scalar operator supplies neither map. In particular its compact-resolvent
proof does not establish the gauge--fermion partition function or the
positivity of the Grassmann functional. $\square$
:::
:::{prf:remark} Wilson observables and the reflected Gram matrix
:label: rem-os2-gauge-fixing-wilson

For a matrix connection define
$$
W_R(C)=\operatorname{tr}_R\mathcal P\exp
\left(i g\oint_C A_\mu\,dx^\mu\right).
$$
The generators and coupling belong inside the transporter. A closed-loop
transport transforms by conjugation at its base point, proving trace
invariance. This geometric identity does not evaluate its expectation.
For a specified field law and positive-time functionals $F_i$, reflection
positivity is the matrix inequality
$\sum_{ij}\bar c_i c_j\,\mathbb E[\Theta F_iF_j]\ge0$ for all $c$.
Restricting to gauge-invariant loops does not by itself prove that matrix
is positive. The positive path-law result above applies to its own
cylinder algebra; there is no established Wilson-law identification here.
:::
#### A.2 Clustering for the Specified Transfer Operator

The spectral estimate in {ref}`sec-mass-gap` applies to its specified scalar operator. The following record explains exactly which clustering conclusion follows for that operator and why it cannot be transferred to another field law without an operator identification.

:::{prf:remark} Clustering and spectral support
:label: thm-smoc-os3-construction

For an already constructed self-adjoint transfer operator, its spectral
gap bounds centered transfer matrix elements by Cauchy--Schwarz and the
spectral semigroup estimate in {prf:ref}`cor-mass-gap-existence`.
This argument applies to that same Hilbert space, state, and operator.
The compact scalar gap does not establish this estimate for the full
gauge-invariant field sector. Moreover
$S_{m+n}-S_mS_n$ is a cluster difference, not generally the fully connected
$(m+n)$-point cumulant; lower connected partitions also contribute.
The previous proof used an unestablished full-field spectral gap and
cluster-expansion control, so it does not verify OS3 for the stated
interacting Schwinger functions. Those conclusions are not used as
antecedents in the corrected gauge chapter.
:::
#### A.3 Poincare/Unitarity Setup (OS Reconstruction)

:::{prf:remark} OS reconstruction as a correlation-family criterion
:label: thm-smoc-poincare-reconstruction

The Osterwalder--Schrader reconstruction theorem applies to a specified
Euclidean correlation family with its full OS regularity and growth
requirements, Euclidean covariance, permutation or graded symmetry, and
reflection positivity. The abbreviated list in {prf:ref}`def-os-axioms`
is an index of these properties; pointwise temperedness for each $n$
alone must not replace the growth requirement in the theorem used.
Clustering concerns the vacuum sector of the same family.

The construction proceeds as follows. On positive-time test sequences
set $(F,G)=S(\Theta F\,G)$. Reflection positivity permits quotienting by
the null space and completion. Positive Euclidean time translations
then yield a contraction semigroup $e^{-tH}$ with $H\ge0$ on this
reconstructed space. Spatial translations and rotations act unitarily.
Euclidean time translation is a semigroup, not a unitary representation
of the full Euclidean group on this Hilbert space. The reconstruction
and analytic-continuation theorem supplies the Lorentzian positive-energy
representation and fields from the same correlation family
{cite}`osterwalder1973axioms,osterwalder1975axioms`.

This describes the mathematical reconstruction operation. The chapter's
finite CP maps, reversible Markov identity and compact scalar estimates
do not verify its premises for the interacting comparison action; in
addition that action's chiral multiplet has
{prf:ref}`thm-smoc-chiral-anomaly-obstruction`. No unconditional Poincare
or Wightman construction for that action is concluded here.
:::
(sec-isomorphism-dictionary)=
## Summary: Representations and Established Identities

:::{div} feynman-prose
The table records formulas and their mathematical scope. A shared notation or a matching local term is not yet an isomorphism of theories. Such an isomorphism must map states and observables and intertwine their evolution.

The established results here are useful precisely because they are explicit: we can transform a derivative, compute its curvature, integrate the radial drift, and diagonalize the stated mass matrices. The same calculations expose the anomaly and the mismatches that prevent the displayed comparison from being a complete quantum reconstruction.
:::

The table records the proved calculations and the objects to which they apply.

| Object | Established calculation | Reference |
|:-------|:------------------------|:----------|
| Phase connection | Compensation of a local phase derivative | {prf:ref}`thm-emergence-opportunity-field` |
| Mode representation | Channel dilation and represented connection covariance | {prf:ref}`thm-emergence-error-field` |
| Feature connection | Curvature and invariant contractions for the chosen representation | {prf:ref}`thm-emergence-binding-field` |
| Candidate product action | Commuting factor actions and their representation kernel | {prf:ref}`cor-standard-model-symmetry` |
| Chiral comparison fields | Representation count and quantum anomaly obstruction | {prf:ref}`def-cognitive-spinor` |
| Scalar potential | Integration of the deterministic radial drift | {prf:ref}`thm-complexity-potential` |
| Gauge masses | Quadratic form at the stated scalar configuration | {prf:ref}`thm-semantic-inertia` |
| Yukawa masses | Singular values on the coupled projected modes | {prf:ref}`thm-cognitive-mass` |
| Belief transport | Exact polar representation of the established density and phase equations | {prf:ref}`thm-recovery-wfr-drift` |
| Quantum reconstruction | Properties to be checked for the same state, algebra, and evolution | {prf:ref}`thm-constructive-specialization-os-wightman` |

:::{div} feynman-prose
The preceding constructions provide concrete objects to implement: specified channels, represented connections, radial dynamics, and the exact polar belief equations. Their defining identities give direct numerical checks.

A claimed equivalence with an interacting quantum theory must preserve the same states, observables, and generators. The chapter's calculations identify which identities hold and which proposed identifications fail, so subsequent implementations can build on the established mathematics.
:::
