# Independent audit of the reference first-provider estimates

## 1. Reviewed revision and conclusion

The complete source `36_reference_first_provider.md` passes independent
review at SHA-256
`c8d149c20280c18101db6e18d61bf918aa9dd057846fcb55c435c3bbfb5d1948`.
This record checks the actual harmonic count-population branch at the
reference parameters, including its component preparation, full Gaussian
tails, native cap and actual deterministic first and joint second
providers. No formal source or algorithm was edited during the review.

| Interface | Conclusion | Retained condition |
|---|---|---|
| Source--velocity products | Correct full-tail weighted bounds | Fresh recipient jitter is independent of the complete source/component/Haar plan |
| Provider field differences | Correct uniform bounds from any stated coupling | Actual donor velocity second moment, with joint position--velocity correlation retained |
| Burn-aware cap feedback | Correct complete fixed-root RMS comparison | Root preparation is common; post-burn moment belongs to all slots |
| Signed first-kick balance | Correct lifted pair and quadratic identities | Actual endpoint own providers; intermediate laws are comparison laws |
| Source-aware spatial defect | Correct conditional formula and constants | Coupled plans precede a shared fresh recipient jitter |
| Joint second-stage differences | Exact covariance identities | Original independent OU innovation is shared between the two root kernels |
| Complete default law block | Remains open | Rooted preparation change, signed B2/cap consumption and marked feedback still need closure |

## 2. Actual source products and cap feedback

Conditional on the complete plan and Haar marks, $S,I,P$ are fixed and
the fresh Gaussian recipient jitter is independent. Its cross term in
$|S+IJ|^2|P|^2$ therefore vanishes exactly. The remaining conditional
moment is bounded by $X_2^2|P|^2$. Integrating gives the stated
$\||X||P|\|_2\le X_2r$ without assuming that the source choice and
prepared component velocity are independent.

Every interpolated deterministic first provider produces a convex
velocity average bounded by $V_c$. Minkowski then proves both the
unweighted and velocity-weighted first-drift position bounds. Applying
these inequalities to
$|H|\le A_0+B_0|P|$ gives the complete weighted product in (RFP.4).
Its displacement remains correlated with the actual source and first
drift.

At a fixed query, the count kernel has global gradient norm at most
$\ell=e^{-1/2}$. Splitting the vector numerator into a velocity
difference and a kernel difference gives exactly
$d_P+\ell r d_X$. Cauchy--Schwarz is applied on the actual coupling;
it does not factor a donor's velocity from its displacement. The same
proof applies to the joint second-stage provider, whose uncapped velocity
moment is explicitly bounded.

The second-kick/cap pointwise derivative coefficients are affine
nonnegative functions of $|x_1|$. Multiplying them by the actual
first-provider perturbation and using (RFP.4) proves the two weighted
Jacobian products. This step supplies the precise condition needed to
replace the original bounded-speed charge $A_0+V_cB_0$ by the sharper
averaged charge $A_0+rB_0$. It never pulls an averaged Jacobian outside
an arbitrary correlated $L^2$ displacement.

The first-drift, OU-mean and landing-position interpolation derivatives
are $taH$, $caH$ and $baH$, respectively. Their full-tail products
give the stated phase RMS coefficient. Changing the second provider with
root and first provider fixed then uses the already reviewed radial
estimate. Minkowski and the upper quadratic norm comparison prove
(RFP.9). The two root kernels retain their exact independent Gaussian
marginals under the shared innovation coupling.

The bound $r=.55$ is the preparation's all-slot averaged moment after
the proved population burn. A common root at an intermediate frozen
provider need not satisfy its own collision contraction; its declared
source product and velocity moment are the hypotheses used here. No
alive-normalized moment, reduced individual cap or finite empirical
provider bound is inferred.

## 3. Lifted signed first-kick identities

On the root-coupling probability space, the interpolated count kernel is
symmetric and lies in $[0,1]$. Its lifted pair form is nonnegative and
bounded by the complete-kernel variance form. Hence
$0\le L_\theta\le I$. Differentiating the actual endpoint comparison
is legitimate in $L^2$: kernel gradients are bounded, velocities are
bounded, and root displacements have finite second moments.

The derivative is exactly
$(I-aL_\theta)\delta P-aL_{\dot k_\theta}P_\theta$.
The kernel derivative remains symmetric under interchanging the two
independent roots. Weighted conditional Cauchy--Schwarz gives
$\|B_\theta\|_2^2\le2S_\theta$; pair symmetrization gives
$|\langle\delta P,B_\theta\rangle|\le\sqrt{D_\theta S_\theta}$.
The positive contraction property also gives
$\|L_\theta\delta P\|_2^2\le D_\theta$.

These identities check the cross coefficient
$(1+a\sqrt2)\sqrt{D_\theta S_\theta}$ and the complete Young
allowance in (RFP.12). Jensen is applied only after integrating the
actual comparison derivative from zero to one. The two endpoint
providers remain their own prepared-law fields.

Expanding $Q_\beta(R,\dot U_\theta)$ preserves the negative velocity
alignment and the exact mixed term $-2\beta aE_\theta$. Convexity in
the velocity argument then gives (RFP.16). The same pair
symmetrization proves its spatial forcing estimate
$|\langle R,B_\theta\rangle|\le\sqrt{D_{X,\theta}S_\theta}$.
The upper bound on $D_{X,\theta}$ is the independent-copy variance
identity. Every displacement/velocity correlation remains in these
forms.

## 4. Conditional source defect and actual joint stages

The identity $|\nabla K(r)|^2/K(r)\le2/e$ and the factor one-half in
$S_\theta$ give the coefficient $1/e$ in (RFP.13). Conditional on
two independent coupled plans, the shared within-pair jitter and its
independent copy have zero cross terms. Their contribution is exactly
$d\sigma_J^2[(\delta I)^2+(\delta I')^2]$.

Expanding the two independent-copy sums after the elementary squared
difference bounds gives the stated coefficients $8/e$ and
$4d\sigma_J^2/e$ in (RFP.14). The mixed local factors
$|\delta S|^2|P_\theta|^2$ and $(\delta I)^2|P_\theta|^2$ remain
inside expectations. Only products across independent root plans
factor. Minkowski gives $r_\theta\le.55$ from the two declared
post-burn preparation moments.

Sharing an OU innovation independent of both preparations cancels its
two differences without changing either own marginal. The affine stage
identities are then exactly
$\Delta y=a_xR+bE$ and $\Delta w=c(E-tR)$. Squaring proves the
two covariance formulas in (RFP.15), including both signs of the
mixed term. The provider fields are never conditioned on that innovation.

## 5. Numeric and structural checks; remaining consumer

Exact Python `Fraction` arithmetic verifies the claimed
$C_0<.00578$ from the displayed rational bounds on $b,c,\overline K_w$
and $\overline K_x$. It also verifies
$\sqrt{1.04}\,C_0<.00590$. The reviewed source has nine unique formal
labels, seventeen unique equation tags, balanced formal directives and
display-math delimiters, and no trailing whitespace.

The RMS provider estimates are upper bounds supplied by a valid complete
fixed-root population coupling. Taking an infimum can therefore bound
transport between those specified frozen-root kernels. This conclusion
does not include the preparation-law change needed for full nonlinear
population transport and does not identify a finite noisy empirical
provider with a deterministic field.

The remaining full-block consumer must combine the rooted preparation
comparison, its source-local forcing, the signed first-kick quadratic,
the joint-stage covariance and the actual B2/native-cap/terminal-mark
account. The source states these remaining obligations and claims no
combined coefficient below one. The audit finds no mathematical blocker
in the proved interfaces.

## 6. Zero-displacement precision revision

The accepted source was subsequently revised to SHA-256
`be58078b8f130f635658d5660c3edb9e826617f4eaedd77cdf8c3929b71136fe`.
The final product bound in (RFP.8) now reads
$\le .00578[d_P+1.1\ell d_X]$ instead of a strict inequality.
This includes the zero-displacement case. The scalar coefficient
certificate $C_0<.00578$, proof, carrier and scope are unchanged.
The original accepted SHA-256 above remains the historical audit record;
this precise non-strict endpoint repair passes review as well.
