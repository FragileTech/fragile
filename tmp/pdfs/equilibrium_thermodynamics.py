"""Equilibrium-law synopsis for the Volume 2 research prospectus."""

EQUILIBRIUM_THERMODYNAMICS = r"""
\section*{Equilibrium thermodynamics and the field state}
\phantomsection\label{sec:equilibrium-thermodynamics}
\addcontentsline{toc}{section}{Equilibrium thermodynamics and the field state}
The equilibrium construction fixes both the population statistics and the temporal law from which the field theory is derived. Its successive objects are the thermal reference, the selected law of the complete population update, the stationary mean field and its fluctuations, and the induced gauge and matter history. The following chain connects these objects to the physical reconstruction in Stages~\ref{stage:17}--\ref{stage:18}.

\subsection*{Thermal balance and the complete population law}
For the conservative kinetic Langevin model, the gradient force, velocity friction and matching Gaussian noise give
\[
\theta=\frac{\sigma_v^2}{2\gamma},\qquad
m_U(dx\,dv)=Z_U^{-1}
\exp\!\left[-\frac{U(x)+|v|^2/2}{\theta}\right]dx\,dv.
\]
Confinement makes the partition function finite. Hamiltonian transport preserves the kinetic energy plus potential, while friction and noise balance at temperature \(\theta\). This is the explicit kinetic Gibbs reference @@def-gibbs-kinetic@@. The kinetic LSI theorem incorporates its Gaussian velocity factor and gives \(C_0=\max\{C_x,\theta\}\) @@thm-kinetic-lsi@@.

The complete configured update determines the population equilibrium through
\[
\pi_NP_{N,h}=\pi_N,
\qquad
\nu_NQ_{N,h}=\alpha_N\nu_N,
\quad 0<\alpha_N\le1.
\]
The first identity selects the conservative invariant law; the second selects the killed process's quasi-stationary law. A Gibbs profile for either law is identified by inserting that profile into its complete equation, with cloning, collision, cap and boundary stages retained. The source supplies a quantitative version: for a proposed probability \(g\), a killed block \(Q\), \(b=gQ1>0\), and a normalized block map contracting total variation by \(r<1\),
\[
\|g-\nu_N\|_{\rm TV}
\le\frac{\|gQ-bg\|_{\rm TV}}{b(1-r)}.
\]
Thus exact balance identifies the Gibbs law, and the evaluated balance residual controls an approximate Gibbs identification @@thm-qsd-gibbs@@. These equations connect the thermal reference to the selected population ensemble of Stage~\ref{stage:4}.

\subsection*{Population thermodynamics supplies uniform coercivity}
For an identified joint Gibbs law \(d\pi_N=Z_N^{-1}e^{-V_N}dz\), uniform curvature \(\nabla^2V_N\succeq\rho_*I\) gives \(C_*=1/\rho_*\). A concrete strongly convex specialization in the book is
\begin{align*}
V_N(z)&=\sum_iV_0(z_i)
 +\frac{\epsilon_{\rm int}}{N}\sum_{i<j}W(z_i,z_j),\\
\rho_*&=\rho_0-\epsilon_{\rm int}L_W>0,
\end{align*}
where \(\nabla^2V_0\succeq\rho_0I\) and the pair Hessian has norm at most \(L_W\). The normalized interaction preserves this curvature independently of \(N\) @@prop-kl-joint-interaction-curvature@@. Product, bounded joint tilt and contractive-flow criteria provide the other population-uniform routes @@cor-n-uniform-lsi@@.

For the continuous coordinates of that same law, these estimates imply
\begin{align*}
\Ent_{\pi_N}(f^2)&\le2C_*\int|\nabla f|^2\,d\pi_N,\\
\Var_{\pi_N}(F)&\le C_*\int|\nabla F|^2\,d\pi_N,\\
\Var_{\pi_N}(L_N\varphi)&\le
 \frac{C_*\operatorname{Lip}(\varphi)^2}{N}.
\end{align*}
The full gradient retains every position and velocity coordinate; discrete alive/dead marks carry the additional status entropy of Stage~\ref{stage:7}. The empirical variance estimate controls an interacting population directly @@cor-quantitative-lsi-final@@.

The evolution uses this coercivity through the modified kinetic entropy. The common-target theorem gives a proved route when the cloning kernel preserves the kinetic equilibrium and its Fisher amplification satisfies the stated bound @@thm-main-kl-convergence@@. The full-law theorem instead combines the actual stationary law, complete cloning and killing source, and a negative modified-entropy derivative; its conclusion is
\[
D_{\rm KL}(\mu_t\Vert\nu_N)
\le\Phi_{G_N}(h_0)e^{-r_Nt},
\qquad
r_N=\frac{\delta_N}{C_N/2+g_{+,N}}.
\]
Uniform bounds on \(C_N\), \(g_{+,N}\) and \(\delta_N^{-1}\) retain a positive population-independent relaxation rate @@thm-kl-convergence-euclidean@@. The selected discrete update uses the full-step entropy estimates described in Stage~\ref{stage:8}.

\subsection*{The stationary mean field and its controlled fluctuations}
The nonlinear map \(\mathcal F_h\) is computed from the same complete update. Under the stationary-chaos hypotheses of Stage~\ref{stage:6}, the selected population law has a deterministic macroscopic limit,
\[
\mu_*=\mathcal F_h(\mu_*),\qquad
L_N\varphi\longrightarrow\mu_*\varphi
\quad\text{in probability and every finite }L^p
\]
for bounded continuous \(\varphi\) @@thm-thermodynamic-limit@@. The marginal LSI passes to the limiting law on its stated test-function domain @@cor-kl-lsi-mean-field-limit@@. Smooth normalized mean-field integrals then supply the metric, force and reconstruction coefficients in Stage~\ref{stage:9}.

Retaining the centered fluctuations gives
\[
\Xi_N(\varphi)=\sqrt N
\bigl(L_N\varphi-\mathbb E_{\pi_N}L_N\varphi\bigr),
\qquad
\mathbb E|\Xi_N(\varphi)|^2
\le C_*\operatorname{Lip}(\varphi)^2.
\]
Their drift and covariance are calculated from the actual kernel, including the shared collision rotation and cloning writes. The mean field controls the background coefficients; the fluctuation hierarchy retains the random field. Moment bounds, normalized derivative estimates and the two continuum estimator bounds of Stage~\ref{stage:15} control the passage to the continuum on the selected observation law.

\subsection*{The equilibrium history induces the field state and action}
Start the conservative record at \(\pi_N\), or use the specified QSD window or Doob history. The complete transition determines every temporal correlation. For an invariant law and centered channel observables \(\widetilde f_a\), the recorded-time correlator is
\[
C_{ab}(\ell)=
\langle\widetilde f_a,P_{N,h}^{\ell}\widetilde f_b\rangle_{L^2(\pi_N)},
\qquad t_\ell=\ell\Delta t.
\]
This is the equilibrium correlator identity used in Stage~\ref{stage:17} @@thm-effective-twistor-spectral-meaning@@. The spatial coordinates remain in \(\mathbb R^3\), and the temporal separation is measured by the recorded algorithmic clock.

Push the complete history law \(P^{\rm hist}\) and its specified reference \(R^{\rm hist}\) through the Fractal Set descriptor \(Y=\mathscr D(\omega)\). With \(\lambda=\mathscr D_*R^{\rm hist}\), the action theorem gives
\begin{gather*}
e^{-S_{\rm alg}(y)}=
\mathbb E_{R^{\rm hist}}\!\left[
\frac{dP^{\rm hist}}{dR^{\rm hist}}\middle|Y=y\right],
\qquad d\nu=e^{-S_{\rm alg}}d\lambda,\\
\varpi(O)=\mathbb E_{P^{\rm hist}}O(Y)=\int O\,d\nu.
\end{gather*}
The induced state \(\varpi\), action, source derivatives and temporal correlations all come from this record likelihood @@thm-ym-recorded-action-emergence@@. Geometry-fiber disintegration retains the normalization needed to recover the same joint law @@thm-ym-native-geometry-fiber-action@@. Stage~\ref{stage:12} computes its predictive evolution and source response, and Stage~\ref{stage:16} carries the sourced action and correlations through the controlled limits.

Thermodynamic response is also explicit. For bounded observables \(A,B\) and the specified equilibrium tilt \(d\pi_s=Z(s)^{-1}e^{\beta sB}d\pi\), the susceptibility formula is @@prop-fluctuation-dissipation@@
\[
\left.\frac{d}{ds}\pi_s(A)\right|_{s=0}
=\beta\operatorname{Cov}_{\pi}(A,B)
\]
A perturbation of the actual transition has the stationary Poisson-response formula already given in Stage~\ref{stage:12} @@thm-algorithmic-stationary-response@@. Each response therefore specifies both the perturbed law and the evolution used to compute it.

\subsection*{Physical reconstruction and survival of the gap}
Stages~\ref{stage:17}--\ref{stage:18} use the limiting equilibrium hierarchy just constructed. The temporal reflection is \(\Theta:(t,x)\mapsto(-t,x)\) in its recorded clock. Identify its positive reflection pairing and transfer representation on the retained gauge and matter observables; the reconstruction then supplies the physical Hilbert space, vacuum and self-adjoint Hamiltonian. For a centered self-adjoint channel in the positive-transfer representation with a unique vacuum and complete orthonormal energy basis, the source gives @@cor-effective-twistor-positive-transfer@@
\[
C_{aa}(\ell)=\sum_{E_j>0}
|\langle j|\widehat f_a|0\rangle|^2
e^{-E_j\ell\Delta t}
\]
Verify the accompanying spacetime covariance and field-axiom hypotheses on this same limiting hierarchy.

To turn population coercivity into the physical mass gap, identify the physical Hamiltonian form and its norm so that the transported estimates give
\begin{gather*}
\mathcal E_a(\psi,\psi)=\|H_a^{1/2}\psi\|^2
\ge\lambda_*\|\psi\|^2,
\qquad\lambda_*>0,\\
\psi\in\operatorname{Dom}(H_a^{1/2})\cap\Omega_a^\perp,
\end{gather*}
uniformly along the chosen cutoff and volume family. This is the specified form comparison between the thermodynamic control and the physical transfer Hamiltonian. The source's gap-survival theorem @@thm-mass-gap-rg-fixed-point@@ then passes the estimate through the strong semigroup and vacuum limits of Stage~\ref{stage:18}, giving
\[
\Spec(H)\subset\{0\}\cup[\lambda_*,\infty),
\qquad \ker H=\mathbb C\Omega
\]
The selected population law defines the field state, its evolution and its quantitative continuum controls. The final representation and form identifications place those controls on the physical Yang--Mills theory.

"""
