# An evaluated population-uniform regional bound in the same signed ledger

:::{prf:theorem} Evaluated regional bound with the original Keystone source pressure
:label: thm-kur-evaluated-regional-bound

Use the native Rastrigin force, unchanged reference parameters and
actual complete preparation of {prf:ref}`thm-kul-centered-regional-attraction`.
Condition the entering configuration on having every eligible frozen source
in one declared core $Q_z$; average its entire sampled measurement, source,
acceptance, mandatory revival, component-Haar and jitter laws.
The original Gaussian noise remains unrestricted after this conditioning
on the entering sources. Write $S$ for the complete entering array,
$\mu$ for its actual zero-jitter source array, and $\bar A$ for the
realized accepted fraction. Set

$$
\lambda=\frac{1+\rho_J^2}{2},\qquad
\varepsilon=\sqrt{\lambda/\rho_J^2}-1,\qquad
D_J=A(1-a_J)\sin(\omega r_*),
$$
$$
C_{\rm reg}=(1+\varepsilon)
 \left[(1+\varepsilon^{-1})dD_J^2+dK_J\right]
 +(1+\varepsilon^{-1})b^2V_c^2+d\tau^2.
\tag{KUR.1}
$$

The constants are independent of the particle population. Both actual
viscous normalizations obey the complete positional bound

$$
\boxed{\quad
\mathbb E W_N(X^+)\le
\lambda\,\mathbb E W_N(\mu)+C_{\rm reg}.
\quad}
\tag{KUR.2}
$$

For the displayed reference,

$$
\lambda=.7989420407893824,\qquad
\varepsilon=.15597686364299568,\qquad
D_J=.0040927561398052745,\qquad
C_{\rm reg}=.20613357968212598.
\tag{KUR.3}
$$

This is a conservative bound. The sharper signed expression is (KUL.12),
which retains the gate-refresh and velocity covariances and the actual
accepted fraction. For all-alive entering arrays with $N\ge2$ satisfying the already
stated original signed Keystone source hypotheses, insert exactly its computed source variance

$$
\begin{aligned}
\mathbb E W_N(\mu)\le{}&
W_N(S)-\theta k_{\rm key}W_N(S)^p
+{\theta E_{\max}}/{N^2}
+\mathbb E_{\mathbf F}\Gamma_\theta\\
&-\mathbb E_{\mathbf F}|\bar t|^2
-N^{-2}\mathbb E_{\mathbf F}\sum_i\sigma_{i,\rm source}^2
\end{aligned}
\tag{KUR.4}
$$

in (KUR.2), with $p=5+4d$. Thus the computed full bound retains
$-\lambda\theta k_{\rm key}W_N(S)^p$, the signed incoming-donor
excess, both centering corrections and the original $N^{-2}$
term; it does not assume a negative copying increment.
The two-swarm centered pressure remains the separate original
bound (KULP.4); its $W$ is not replaced by this one-population variance.

If only some eligible sources are in the core, split the exact
source integral by the actual observed source-phase labels, apply
the regional estimate on its declared domain, and retain the
remaining integrals with the global moment budget. Between-phase
root separations and their cross terms remain (KUL.6)--(KUL.7),
and actual crossings are {prf:ref}`thm-kulp-complete-phase-flux`
and {prf:ref}`thm-kuan-jitter-phase-flux`.
No population-wide good-noise event is required. For terminal-alive
variance use the exact random-denominator integral (KUL.21),
rather than dividing (KUR.2) by an expected number of survivors.
Consequently (KUR.2) is not asserted as a conditional closed-core
kernel or as contraction between distinct attainable phases.
:::

:::{prf:proof}
Condition first on the entire actual source/acceptance pattern and
component rotations. In (KUL.11), every coordinate of the refresh
array $d$ has magnitude at most $D_J$, since its source lies
within $r_*$ of its corresponding integer center. Therefore
$W_N(d)\le dD_J^2$. Young's inequality and (KUL.9) give

$$
\mathbb E W_N(g_h(X))
\le(1+\varepsilon)\rho_J^2\mathbb E W_N(\mu)
+ (1+\varepsilon^{-1})dD_J^2+dK_J\mathbb E\bar A.
\tag{KUR.5}
$$

Both positive first-kick matrices are stochastic at the
unchanged $t\nu=.006$ and hence
$|(W_XV^C)_i|\le V_c$, even though the matrix depends on every
actual jitter. Thus $W_N(W_XV^C)\le V_c^2$ pathwise.
Apply Young's inequality to
$W_N(g_h(X)+bW_XV^C)$, then use (KUL.5):

$$
\mathbb E W_N(X^+)
\le(1+\varepsilon)\mathbb E W_N(g_h(X))
+ (1+\varepsilon^{-1})b^2V_c^2
+ (1-N^{-1})d\tau^2.
\tag{KUR.6}
$$

Since $(1+\varepsilon)^2\rho_J^2=\lambda$, inserting (KUR.5),
$\bar A\le1$ and $1-N^{-1}\le1$ proves (KUR.1)--(KUR.2).
All source, jitter and Haar correlations were preserved in the
conditioning; no independence of the viscous graph from the jitter
was used. The source-variance identity and its original pressure
bound give (KUR.4) with the same measured law. Substitute it in
(KUR.2); the nonnegative multiplier $\lambda$ preserves every sign.
For a mixed phase population, splitting the original integral
is an identity, and the displayed existing phase and Gaussian
formulas evaluate its remaining terms. The terminal formula
uses the actual random alive normalization and was proved
by its conditional Bernoulli generating function. This establishes
the asserted scope without an absorbing phase boundary. $\square$
:::

