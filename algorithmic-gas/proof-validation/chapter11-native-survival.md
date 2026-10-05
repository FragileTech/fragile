# Chapter 11 native survival calibration

The Rust validator keeps the native update fixed. Successful revival and the native shared collision produce a prepared configuration. For terminal-boundary BAOAB, no geometric correction or shared kinetic interaction, and constant isotropic Gaussian velocity noise, conditional survival probabilities can be computed for the complete kinetic step.

Write $h$ for the step, $\gamma$ for friction, $B$ for the native isotropic velocity diffusion factor, and $\sigma_x$ for final position diffusion. Put

$$
c=e^{-\gamma h},\qquad
s_h^2=\begin{cases}h,&\gamma=0,\\(1-e^{-2\gamma h})/(2\gamma),&\gamma>0.\end{cases}
$$

For each prepared position and velocity $(X_i,V_i)$, the first kick gives $V_i^-=V_i-h\nabla U(X_i)/2$. The first transport, OU stage, and second transport therefore give the terminal position before classification

$$
X_i'=a_i+\frac h2 Bs_h\xi_i+\sqrt h\sigma_x\eta_i,
\qquad
a_i=X_i+\frac h2(1+c)V_i^-.
$$

The second kick and radial velocity cap do not change position. The native Gaussian vectors $\xi_i,\eta_i$ are independent between rows after conditioning on the complete prepared configuration. Consequently the complete kinetic position is Gaussian with scalar variance

$$
\sigma_{\mathrm{eff}}^2=\frac{h^2}{4}B^2s_h^2+h\sigma_x^2.
$$

The validator reconstructs every realized terminal position using the actual recorded first force, OU innovations, final position innovations, and physical inputs. For an absorbing box $\prod_{k=1}^d[\ell_k,u_k]$, its conditional survival probability is

$$
p_i=\prod_{k=1}^d\left[\Phi\!\left(\frac{u_k-a_{ik}}{\sigma_{\mathrm{eff}}}\right)-\Phi\!\left(\frac{\ell_k-a_{ik}}{\sigma_{\mathrm{eff}}}\right)\right].
$$

On the unbounded domain $p_i=1$. Thus the actual alive fraction has conditional mean $b=N^{-1}\sum_i p_i$ and conditional variance $v=N^{-2}\sum_i p_i(1-p_i)\le1/(4N)$. Its complete conditional count distribution is the Poisson-binomial law; its polynomial coefficients are retained with the native operands. The validator also computes a second, separately scoped conditional law after the recorded second kick, when only final position innovations remain.

For either conditioning scheme, define the actual innovation $Z_j=m_j-b_j$ and predictable conditional variance $v_j$. The stopped sums $M_T=\sum_{j\le T}Z_j$, $V_T=\sum_{j\le T}v_j$ satisfy

$$
\mathbb E M_T=0,\qquad \mathbb E M_T^2=\mathbb E V_T.
$$

This follows from conditional centering and vanishing cross-time martingale increments. Extinction makes all future increments zero. The data records the first physical transition with zero alive mass, including a final requested step; an exception on the next attempted update is not its killing time.

The Monte Carlo comparison resamples complete independently seeded trajectories within each fixed configuration and conditioning scheme. It retains the samples of $M_T,V_T$, all resampling indices, and bootstrap intervals for the mean and the second-moment difference. Time steps and walker rows are not treated as independent replicates.

For $R$ independent trajectories with deterministic planned horizon $T$, conditional Hoeffding gives the two-sided mean bound

$$
\mathbb P\left(\left|R^{-1}\sum_{r=1}^R M_T^{(r)}\right|>a\right)
\le2\exp\left(-\frac{2NRa^2}{T}\right).
$$

A variance-sensitive bound uses an exponential supermartingale rather than substituting an observed random variance into a fixed-variance tail inequality. For a centered Bernoulli variable $Y=I-p$, $|Y|\le1$, $\mathbb EY=0$, and $|\mathbb E Y^k|\le p(1-p)$ for $k\ge2$. Taylor expansion and $\log(1+u)\le u$ give

$$
\log\mathbb E e^{tY}\le p(1-p)(e^{|t|}-1-|t|).
$$

Independent rows scaled by $1/N$ consequently give

$$
\mathbb E\left[\exp\left(\lambda Z_j-N^2v_j\psi(|\lambda|/N)\right)\mid\mathcal F_j\right]\le1,
\qquad\psi(u)=e^u-1-u.
$$

Conditional iteration, stopping, and independent trajectories preserve this supermartingale. Markov's inequality bounds each fixed positive or negative $\lambda$. The helper pays a union bound for a predeclared grid of eight positive $\lambda$ values, both signs, every configuration/conditioning group, and the three separate statistical families. It adds explicit mean and variance allowances for the normal-CDF approximation. The complete family uses probability budget $0.01$.

These checks concern the actual finite-time survival innovations. They do not infer a global mass-contraction coefficient, a stationary alive law, a QSD, or an LSI from a finite collection of trajectories. All errors use the alive fraction normalized by $N$; dimension enters the Gaussian product and its CDF error allowance explicitly.
