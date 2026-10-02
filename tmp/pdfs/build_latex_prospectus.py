from pathlib import Path
import ast
import re
from equilibrium_thermodynamics import EQUILIBRIUM_THERMODYNAMICS
from proof_pipeline import PROOF_PIPELINE
from shared_dynamics_introduction import SHARED_DYNAMICS_INTRODUCTION
from quantum_gravity_routes import QUANTUM_GRAVITY_ROUTES
from chapter_contributions import (
    ADDITIONS,
    CHAPTER_GROUPS,
    CONSTRUCTION_SECTION,
    OVERRIDE_PARAGRAPHS,
    OVERRIDE_SKETCHES,
)

ROOT=Path('/home/guillem/fragile')
BASE=ROOT/'docs/source/2_fractal_gas'
catalog={}
for path in BASE.rglob('*.md'):
    text=path.read_text()
    for m in re.finditer(r'^:+\{prf:([^}]+)\} ([^\n]+)\n:label: ([^\n]+)',text,re.M):
        catalog[m[3]]=dict(kind=m[1],title=m[2],file=str(path.relative_to(BASE)),line=text[:m.start()].count('\n')+1)

def tex(text):
    text=re.sub(r'<[^>]+>','',text)
    math_fragments=[]
    def retain_math(match):
        math_fragments.append(match.group(1))
        return 'MATHFRAGMENTTOKEN'+str(len(math_fragments)-1)+'ENDTOKEN'
    text=re.sub(r'\$([^$]+)\$',retain_math,text)
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','²':r'\(^{2}\)','⁴':r'\(^{4}\)','→':r'\(\longrightarrow\)','≤':r'\(\leq\)','≥':r'\(\geq\)','∫':r'\(\int\)','∪':r'\(\cup\)','∞':r'\(\infty\)'}
    text=''.join(replacements.get(c,c) for c in text)
    for old,new in [
        ('Poincare',"Poincar\\'e"),('Bakry-Emery',"Bakry--\\'Emery"),
        ('S = ((x\\_i, v\\_i, a\\_i))',r'\(S=((x_i,v_i,a_i))_{i=1}^N\)'),
        ('C\\_* L\\(^{2}\\)/N',r'\(C_*L^2/N\)'),('C\\_*',r'\(C_*\)'),
        ('1/N',r'\(N^{-1}\)'),('C3',r'\(C^3\)'),('C\\textasciicircum{}n',r'\(C^n\)'),
        ('C\\textasciicircum{}\\{n\\}',r'\(C^n\)'),('C\\textasciicircum{}\\{3\\}',r'\(C^3\)'),
        ('C\\textasciicircum{}\\{infty\\}',r'\(C^\infty\)'),
        ('N-uniform',r'\(N\)-uniform'),('N-independent',r'\(N\)-independent'),
        ('square-root population scaling',r'\(\sqrt N\) fluctuation scaling'),
        ('H\\_a',r'\(H_a\)'),('lambda\\_*',r'\(\lambda_*\)'),
        ('J\\_a',r'\(J_a\)'),
    ]: text=text.replace(old,new)
    for i,fragment in enumerate(math_fragments):
        text=text.replace('MATHFRAGMENTTOKEN'+str(i)+'ENDTOKEN',r'\('+fragment+r'\)')
    return text

equations={
1:r'''E=\mathbb R^d\times\overline B_V\times\{0,1\},\qquad S\in E^N,\qquad L_N=\frac1N\sum_{i=1}^N\delta_{z_i},\quad z_i=(x_i,v_i,a_i).''',
2:r'''\widetilde v_i=\overline v_C+\alpha R_C(v_i-\overline v_C),\qquad R_C\sim\mathrm{Haar}(O(d)),
\quad \sum_{i\in C}\widetilde v_i=\sum_{i\in C}v_i.''',
3:r'''P_NW_p\le r_pW_p+B_p+a_pE_p,\qquad r_p<1,\\
\mathbb EW_p(S_n)\le r_p^n\mathbb EW_p(S_0)+\frac{B_p+a_p\overline e_p}{1-r_p}(1-r_p^n).''',
4:r'''\pi_NP_{N,h}=\pi_N,\qquad \nu_NQ_{N,h}=\alpha_N\nu_N,\quad 0<\alpha_N\le1.''',
5:r'''\mu_{n+1}=\mathcal F_h(\mu_n),\qquad m(\mu)=\mu(a=1),\qquad \rho_\mu=\frac{\mu^a}{m(\mu)}.''',
6:r'''\left|\mathbb E[L'_N\varphi\mid S]-\mathcal F_h(L_N(S))\varphi\right|
\le\frac{2\|\varphi\|_\infty B_*}{\sqrt N},\\
\mathbb E\!\left[|L'_N\varphi-\mathcal F_h(L_N(S))\varphi|^2\mid S\right]
\le\frac{A_\varphi+4\|\varphi\|_\infty^2B_*^2}{N}.''',
7:r'''\operatorname{Ent}_{\pi_N}(f^2)\le 2C_*\int\sum_{i=1}^N
\bigl(|\nabla_{x_i}f|^2+|\nabla_{v_i}f|^2\bigr)\,d\pi_N.''',
8:r'''\operatorname{Var}_{\pi_N}(F)\le C_*\int\sum_i|\nabla_iF|^2\,d\pi_N,\qquad
\operatorname{Var}_{\pi_N}\!\left(\frac1N\sum_i\varphi(z_i)\right)\le\frac{C_*\operatorname{Lip}(\varphi)^2}{N}.''',
9:r'''A_\mu(x)=\int a(x,y)\,\mu(dy),\qquad
F_\mu(x)=\frac{\int a(x,y)m(x,y)\,\mu(dy)}{A_\mu(x)},\\
\mu_N\Rightarrow\mu,\quad \inf_K A_\mu>0
\quad\Longrightarrow\quad F_{\mu_N}\longrightarrow F_\mu\ \text{in }C^n(K),''',
10:r'''\mathscr D_*P=\nu,\qquad
\mathbb E_P\bigl[O(\mathscr D(\omega))\bigr]=\int O(y)\,\nu(dy).''',
11:r'''U_{\partial f}=\prod_{e\in\partial f}^{\longrightarrow}U_e,\qquad
W_f=\frac{\operatorname{Re}\operatorname{tr}U_{\partial f}}{\dim R},\qquad
U_{ij}\mapsto g_iU_{ij}g_j^{-1}.''',
12:r'''Y=\mathscr D(\omega),\quad \lambda=\mathscr D_*R,\quad
a(y)=\mathbb E_R\!\left[\frac{dP}{dR}\middle|Y=y\right],\\
S_{\rm alg}(y)=-\log a(y),\qquad d\nu=e^{-S_{\rm alg}}\,d\lambda,\\
\mathcal Z(J)=\mathbb E_P\exp\!\left(i\sum_rJ_rO_r(Y)\right).''',
13:r'''\{a(f),a(g)\}=0,\qquad \{a(f),a^*(g)\}=\langle f,g\rangle I,
\qquad \mathcal C_t:\operatorname{CAR}(\mathcal K)\to\operatorname{CAR}(\mathcal K).''',
14:r'''\Xi_N(\varphi)=\sqrt N\bigl(L_N\varphi-\mathbb E[L_N\varphi]\bigr),\qquad
\mathbb E|\Xi_N(\varphi)|^2\le C_*\operatorname{Lip}(\varphi)^2.''',
15:r'''\mathbb E\left|\widehat L_{M,\varepsilon}f(p)-\Box_gf(p)\right|^2
\le C_{\rm b}^2\varepsilon^4+
\frac{C_{\rm mix}C_{\rm v}}{M\varepsilon^{D+2}},\\
\varepsilon_M=\ell_0M^{-\beta},\quad 0<\beta<\frac1{D+4},\qquad
\beta=\frac1{D+6}\ \Longrightarrow\ \mathrm{MSE}=O(M^{-4/(D+6)}).''',
16:r'''S_W\longrightarrow \frac1{4g^2}\int_{\mathbb R_t\times\mathbb R_x^3}
F_{\mu\nu}^aF_{\mu\nu}^a\,dt\,d^3x,\\
\mathbb E|\widehat\varphi_N-\rho\varphi|
\le b_N+\sqrt{v_{N,h}}+d_h+m_{N,h,n}.''',
17:r'''(F,G)_{\rm OS}=\mathbb E\bigl[\overline{\Theta F}\,G\bigr],\qquad
(F,F)_{\rm OS}\ge0\quad(F\in\mathcal A_+).''',
18:r'''\|e^{-tH_a}(I-P_{\Omega_a})\|\le e^{-\lambda_*t},\qquad \lambda_*>0,\\
J_ae^{-tH_a}J_a^*\xrightarrow{\rm strong}e^{-tH},\qquad
\operatorname{spec}(H)\subset\{0\}\cup[\lambda_*,\infty).'''
}

sketches={
1:r'''Construct the companion and donor kernels by normalizing strictly positive weights against the alive population. The regularized standard deviations keep the score maps defined on constant populations. Condition on the frozen marks and accepted-edge graph, then compose the collision, Gaussian, deterministic kinetic, cap, and classification kernels. Integrating these conditional kernels gives \(P_{N,h}\). Its path law is the initial law times the ordered product of the complete transition kernels. Record coverage is fixed at this point, so later likelihood calculations integrate the randomness that actually produced the path.''',
2:r'''For a component \(C\), the centered velocities satisfy \(\sum_{i\in C}(v_i-\overline v_C)=0\). Orthogonality preserves their squared norm and Haar symmetry gives zero mean for the rotated component. Taking traces in the rotationally invariant covariance identity gives
\[\operatorname{Cov}(\widetilde v_i,\widetilde v_j\mid C)
=\frac{\alpha^2}{d}\bigl[(v_i-\overline v_C)\cdot(v_j-\overline v_C)\bigr]I_d.\]
For the population limit, truncate the rooted accepted-edge component, compare its finite exploration law with the limiting rooted law, and remove the truncation using the proved component moments. The two-root calculation controls the empirical variance. These steps identify the collision term without replacing component rotations by independent row noise.''',
3:r'''Evaluate the copying increment \(|x_j|^p-|x_i|^p\) under accepted donor flux. Lower bounds on inward core flux and upper bounds on outward tail flux give a signed selection drift. Add cloning jitter through Gaussian radial moments, then propagate through the actual kinetic position update using the force-growth and cap bounds. The remainder \(E_p\) records coverage or class failures. Iterating the resulting affine recurrence gives the displayed moment budget. Markov's inequality turns higher moments into quantitative tail localization; the entropy-floor calculation gives additional tail and population-fraction estimates for the normalized joint positional law.''',
4:r'''Iterate the weighted drift to enter the controlled region, then apply the appropriate common-mass or surviving-block comparison. In the conservative case, the coupling contracts a weighted distance and yields an invariant probability. In the killed case, normalized block iteration yields the selected eigenmeasure. For a QSD window, multiply the killed transitions and divide by the window survival probability. Successive survival factors cancel in the selected-history identity. Comparing an actual relaxed window with this eigenmeasure window introduces the explicit survival amplification, bounded in the chapter by \(2\alpha_N^{-w}\delta_n\) for a \(w\)-step window.''',
5:r'''Disintegrate a representative row into its measurement mark, frozen fitness, donor gate, rooted collision component, and independent kinetic innovations. Integrate the root output against this product of conditional laws to define \(\mathcal F_h\). Positivity and total mass follow from kernel composition. Gaussian final position noise makes terminal box classification continuous after averaging and supplies positive alive mass. Polynomial growth of reward is handled by the propagated moments. The weak mass and field balances follow by testing the same root input-output coupling; each term belongs to a declared stage of the complete update.''',
6:r'''Condition on the input array and estimate the realized measurement-normalization fluctuations. Couple the resulting donor and gate laws, use rooted-component estimates for the collision response, and apply conditional concentration to independent kinetic innovations. This proves the displayed one-step bias and mean-square bounds at the actual empirical input. Continuity of \(\mathcal F_h\) and a countable determining class give finite-horizon iteration. For stationary limits, combine one-step consistency with population attraction and tightness. In the weighted-contraction branch, expand two iterates and verify the source condition
\[q_w=1-\epsilon_2/2+2B_wL_{\rm rem}+L_{\rm rem}^2<1.\]
The growing-horizon branch instead propagates its explicit error recursion and high-moment budget to a diverging observation horizon.''',
7:r'''Write \(\pi(dx)=Z^{-1}e^{-V(x)}dx\), assume \(\nabla^2V\succeq-KI\), and use the quadratic Lyapunov estimate. Choose \(R\) so \(\ell=cR^2-b>0\). The exterior weighted-energy estimate and interior Poincar\'e inequality give
\[C_P=\left(1+\frac b\ell\right)C_{\rm loc}+\frac1\ell.\]
The entropy--transport--Fisher estimate and the moment bound yield a defective LSI. Centering by Rothaus' inequality removes the defect using \(C_P\). Tensorization then incorporates Gaussian velocities. At the joint level, the chosen product, bounded-tilt, curvature, or flow-contraction criterion controls the constant independently of \(N\). The selected invariant law or QSD must be the law appearing in that criterion.''',
8:r'''Substitute \(f=1+\epsilon F\) in LSI and compare the second-order terms to obtain Poincar\'e. For \(F_N=N^{-1}\sum_i\varphi(z_i)\), the squared gradient sum is bounded by \(\operatorname{Lip}(\varphi)^2/N\). For entropy decay, differentiate the modified kinetic functional and absorb its cross-gradient terms by the chapter's quadratic-form inequalities. Add the cloning contribution and the conditioned normalization term. A closed negative derivative bound followed by Gr\"onwall gives the stated relaxation estimate. Projection to fixed marginals preserves LSI; bounded continuous test integrands pass the inequality to their weak limits.''',
9:r'''Start with \(A_\mu w_\mu=a\). Differentiating this identity isolates the highest derivative of the normalized weight; integrating its norm gives the all-order recurrence. The same Leibniz calculus handles measurement moments and regularized variance. On a compact parameter set, a finite equicontinuity net upgrades weak convergence of the constituent integrals to uniform convergence of each derivative. A positive limiting denominator gives a common lower bound for the empirical denominators. Induction through the quotient identity then proves \(C^n\) convergence. The result retains dependence on bandwidth and regularization parameters, which must be tracked along any changing-scale family.''',
10:r'''Apply the lossless decoder to a covered record, reconstruct the stage variables, and substitute them into the complete kernel. This identifies finite histories and their observable expectations. For invariant coordinates, the orbit theorem resolves the frame equivalence and the measure theorem pushes forward both probability and readouts. The transition intertwining follows by applying the same kernel to pullbacks of descriptor observables. If a descriptor is compressed, conditional elimination gives its exact memory term. Closing the original bounded insertion algebra under repeated transition action gives the prediction-complete descriptor; finite conditional partitions approximate its transition matrices.''',
11:r'''Under an endpoint frame change, adjacent factors cancel in the ordered product around a face, leaving conjugation at the base point; the trace is invariant. The direct Gram and determinant coordinates retain the full declared color invariant algebra. For a companion doublet, compare its value with the transported neighboring doublet, separating amplitude and angular contributions. The interaction triangle retains the mismatch between IA and IG transports. This supplies the native observable whose curvature expansion and source response are analyzed downstream. The group representation, face incidence, and observable normalization are part of its definition.''',
12:r'''For bounded \(F\), conditional expectation gives
\[\int F(y)a(y)\,\lambda(dy)=\mathbb E_R[F(Y)\,dP/dR]=\mathbb E_PF(Y).\]
Thus the descriptor action reproduces the original measure and all its bounded correlations. Disintegrate \(\nu\) over geometry; the conditional gauge denominator is retained, so recombining the fibers recovers the same native expectation. Introduce sources through the executed force or metric update and differentiate its likelihood on its stated domain. This produces the native response current and weak variation identity. Gauge variations of the discrete face functional give the graph Ward identity. Matching these identities to the limiting Yang--Mills law is an explicit downstream identification.''',
13:r'''Lift the recorded one-particle mode space to its exterior algebra and represent alternating insertions by creation and annihilation operators. The insertion norm proves faithfulness. Transport the recorded contraction through the prescribed isometric dilation and compression to obtain a unital completely positive CAR channel. Its stated two-point and higher-word prescriptions retain the recorded correlations. The native generator follows by differentiating the transported evolution on its declared domain. Regional covariance coefficients quantify the locality comparison used by the CAR local-net application. This stage fixes the operator algebra whose physical spacetime realization remains part of the final reconstruction task.''',
14:r'''Apply the joint-law LSI concentration bound to the centered Lipschitz empirical fields. Higher moments follow from the same law-specific bounds. The chapter's test-space construction gives tightness and a common subsequence for the distributional hierarchy. For dynamics, use the generator product rule and conditional variance decomposition to compute drift and noise from the complete update. For gauge words, combine bounded normalized readouts with the explicit uniform-integrability and mark-reconstruction estimates. A diagonal extraction over the stated observable/test classes places the words and reflected products in one native hierarchy, preserving their common sourced-law interpretation.''',
15:r'''In normal coordinates \(\Psi_p\), expand \(f\circ\Psi_p\) to fourth order and the volume Jacobian to third order. Verified kernel moments remove the first-order and unwanted odd terms and select the intended second-order signature. The remainder is \(O(\varepsilon^2)\) after normalization. For samples with density \(q\), divide each summand by \(q\); its second moment scales as \(\varepsilon^{-D-2}\). Summing the chapter's covariance bound gives the displayed variance. Balance bias and variance through the bandwidth schedule. Finally prove the recorded operator comparison at this normalization and a joint weighted quadrature estimate for action evaluation on the interacting records.''',
16:r'''Expand the actual small-face transport and control the remainder by the curvature and geometric bounds. Weighted face quadrature yields the displayed classical action under the chapter's hypotheses. For temporal error, use the telescoping identity
\[P_h^n-T_h^n=\sum_{j=0}^{n-1}P_h^j(P_h-T_h)T_h^{n-1-j}.\]
A local defect of order \(h^{p+1}\) and stability give a finite-time defect of order \(h^p\). Stationary perturbation sums the defect against the relaxation resolvent; the QSD version compares normalized surviving blocks. Combine the resulting bias, sampling, stationary, and mixing terms using the total-observable-error theorem. Uniform integrability and the native source/fiber identities retain expectations along the common hierarchy.''',
17:r'''Treat the limiting physical hierarchy as the reconstruction input. Establish positivity of every finite reflected matrix: for \(F=\sum_jc_jF_j\), the requirement is \(\sum_{ij}\overline c_ic_j\mathbb E[\overline{\Theta F_i}F_j]\ge0\). Together with the required symmetry and regularity, quotient the future algebra by null vectors and complete it in the reflection inner product. Physical Euclidean-time translation induces the positive contraction semigroup, whose self-adjoint generator is the transfer Hamiltonian. Complete the Euclidean-to-relativistic reconstruction and verify physical covariance, locality, spectrum, vacuum, and nontriviality. The chapter's native reflected matrices, translation identities, and normalization tests specify the concrete family on which these tasks must be resolved.''',
18:r'''Let \(J_a:\mathcal H_a\to\mathcal H\) be isometries with \(J_aJ_a^*\to I\) strongly and \(J_a\Omega_a\to\Omega\). For \(f\in\mathcal H\), put \(f_a=J_a^*f\). The embedded gap bound gives
\[\|J_ae^{-tH_a}(f_a-P_{\Omega_a}f_a)\|\le e^{-\lambda_*t}\|f_a\|.\]
Pass to the strong semigroup limit and the vacuum limit. The resulting estimate holds on \(\Omega^\perp\); the spectral theorem excludes spectrum in \((0,\lambda_*)\) and any further zero-energy vector. Multiplication by a fixed \(\hbar_{\rm eff}\) converts the bound to physical energy. The program must identify this self-adjoint family with the gauge transfer Hamiltonians and retain the positive bound through cutoff removal and physical volume growth.'''
}

tree=ast.parse((ROOT/'tmp/pdfs/build_roadmap.py').read_text())
stages=[]
for node in tree.body:
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='stage':
        args=[eval(compile(ast.Expression(arg),'<own-source>','eval'),{'catalog':catalog}) for arg in node.value.args]
        stages.append(args)

preamble=r'''\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=27mm,headheight=15pt]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools,mathrsfs}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{tikz,pdflscape}
\usetikzlibrary{arrows.meta,calc}
\usepackage{enumitem}
\usepackage{booktabs,longtable,array}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\definecolor{ink}{HTML}{17384A}
\definecolor{teal}{HTML}{087F8C}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries\color{ink}}{\thesection}{.7em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{teal}}{\thesubsection}{.7em}{}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\emergencystretch}{2em}
\setlist{nosep,leftmargin=1.5em}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\footnotesize Fractal Gas: a Yang--Mills proof program}
\fancyhead[R]{\footnotesize Research prospectus}
\fancyfoot[C]{\thepage}
\newtheorem{criterion}{Completion criterion}
\newcommand{\Ent}{\operatorname{Ent}}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\Spec}{\operatorname{spec}}
\hypersetup{pdftitle={From the Euclidean Gas to Quantum Yang-Mills: A Research Prospectus},pdfauthor={Compiled from Fragile Volume 2}}
\begin{document}
\hypersetup{pageanchor=false}
\begin{titlepage}
\vspace*{18mm}
{\small\scshape Fragile / Volume 2}\par\vspace{12mm}
{\LARGE\bfseries\color{ink} From the Euclidean Gas\\[4pt] to Quantum Yang--Mills\\[4pt] and a Mass Gap}\par\vspace{8mm}
{\large A research prospectus with proof sketches and theorem dependencies}\par\vspace{13mm}
\textbf{Abstract.} This prospectus organizes the constructive program in Volume 2 around one stochastic transition and its recorded history law. A marked interacting gas supplies a nonlinear population evolution. Confinement, Poincar\'e and logarithmic Sobolev inequalities, hypocoercivity, propagation of chaos, and normalized regularity estimates control its long-time and population limits. A lossless Fractal Set representation transports the same law to gauge and matter observables. Native likelihood, source-response, fluctuation, and continuum-consistency results then supply the field construction. The final tasks identify a nontrivial physical Euclidean Yang--Mills hierarchy, reconstruct its transfer Hamiltonian, and carry a uniform self-adjoint gap through the cutoff and volume limits.

The document gives the mechanism of each stage and identifies the theorem statements in the source. It follows the existing population, mechanical, metric, gauge, and matter constructions, including the derived sampling normalizations and temporal correlations. The field-theoretic reconstruction statements retain their explicit hypotheses on the same selected equilibrium law.
\vfill
\textbf{Source version:} repository snapshot, 1 October 2026.\\
\textbf{Purpose:} mathematical discussion of the complete construction program.\\
\textbf{Source corpus:} \texttt{docs/source/2\_fractal\_gas/}.
\end{titlepage}
\hypersetup{pageanchor=true}\pagenumbering{roman}
\begingroup\setlength{\parskip}{0pt}\footnotesize\tableofcontents\endgroup
\clearpage
\pagenumbering{arabic}
\section*{Objects, target, and logical structure}
\addcontentsline{toc}{section}{Objects, target, and logical structure}
The construction begins with three spatial coordinates and algorithmic time:
\[
x_i(n),v_i(n)\in\mathbb R^3,\qquad t_n=n\Delta t,\qquad
e_{n,i}\mapsto(t_n,x_i(n)).
\]
Here \(\Delta t\) is the configured recording spacing with its fixed unit calibration. CST edges inherit increasing algorithmic time. Equilibrium supplies the selected history law in which the additional field-theoretic properties are developed. The target is a nontrivial quantum Yang--Mills theory on this \(3+1\) reconstruction, with a unique vacuum and a positive physical energy gap. The construction proceeds through
\[
\begin{gathered}\text{gas}\ \longrightarrow\ \text{population control}\ \longrightarrow\ \text{recorded field law}\\
\longrightarrow\ \text{continuum hierarchy}\ \longrightarrow\ (\mathcal H,\Omega,H).\end{gathered}
\]
\paragraph{Law ledger.} Write \(P_{N,h}\) for the complete conservative transition and \(Q_{N,h}\) for its specified killed counterpart. Their selected laws are respectively \(\pi_N\) and \(\nu_N\). The nonlinear population update is \(\mathcal F_h\). A history descriptor \(\mathscr D\) induces the field law \(\mathscr D_*P\). Every transfer of an estimate uses an explicit identification of these laws, their conditionings, or their observables.
\paragraph{Scale ledger.} Here \(N\) counts walkers, \(M\) counts observations in a continuum estimator, \(h\) is the algorithm step, \(n\) is the update index, \(\varepsilon\) is a reconstruction bandwidth, \(a\) is a field cutoff, and \(L\) is a physical volume scale. The continuum chapter uses \(N\) for its sample count; this prospectus writes \(M\) to distinguish it from population size. This roadmap takes \(d=3\) and \(D=d+1=4\). The fourth coordinate is the calibrated recorded time. A phase-space fluctuation test also retains \(v\in\mathbb R^3\).
\paragraph{Reading convention.} ``Source results'' name statements and proofs already present in Volume 2, with their hypotheses retained. ``Proof mechanism'' sketches the argument of those results and their composition. ``Stage output'' identifies the object or estimate passed to the next stage. The final assembly criterion organizes these dependencies.
\paragraph{The main interface.} Population-uniform coercivity and relaxation are transported through the recorded observable law. The physical transfer Hamiltonian must then be identified and controlled uniformly in the cutoff and volume family. That identification connects the gas estimates to the Yang--Mills mass gap.
'''

preamble=preamble.replace(
    r'\section*{Objects, target, and logical structure}',
    SHARED_DYNAMICS_INTRODUCTION + r'\section*{Objects, target, and logical structure}',
    1,
)
parts=[preamble,PROOF_PIPELINE]
used=[]

def render_source_links(content):
    def source_link(match):
        label = match.group(1)
        if label not in catalog:
            raise KeyError(f'Unknown source label: {label}')
        if label not in used:
            used.append(label)
        return '\\hyperref[src:'+label+']{[R'+str(used.index(label)+1)+']}'
    return re.sub(r'@@([a-zA-Z0-9_-]+)@@', source_link, content)

parts.append(render_source_links(EQUILIBRIUM_THERMODYNAMICS))

def append_source_results(refs, heading='Source results'):
    if not refs:
        return
    parts.append('\\subsection*{'+heading+'}\n\\begin{itemize}\n')
    for label,role in refs:
        if label not in used:
            used.append(label)
        item=catalog[label]
        parts.append('\\item \\textbf{'+tex(item['title'])+'}~\\hyperref[src:'+label+']{[R'+str(used.index(label)+1)+']}. '+tex(role)+'\n')
    parts.append('\\end{itemize}\n')

for number,title,paragraphs,old_eq,tools,refs,handoff in stages:
    title={5:'Derive the population law and exact field equations',10:'Reconstruct the causal 3+1 Fractal Set',17:'Reconstruct fields from the equilibrium history'}.get(number,title)
    parts.append('\n\\section{'+tex(title)+'}\n\\label{stage:'+str(number)+'}\n')
    if number in (7,12):
        bridges = {
            7: r'The \hyperref[sec:equilibrium-thermodynamics]{equilibrium thermodynamics section} identifies the Gibbs reference, the complete population balance and the explicit joint-curvature specialization used in this stage.',
            12: r'The \hyperref[sec:equilibrium-thermodynamics]{equilibrium thermodynamics section} connects this likelihood construction to the selected population state, its temporal law and its susceptibility formulas.',
        }
        parts.append(bridges[number]+'\n\n')
    if number in OVERRIDE_PARAGRAPHS:
        parts.append(OVERRIDE_PARAGRAPHS[number])
    elif number == 15:
        parts.append(r'''Identify the reconstructed product geometry and its sampling measure \(q\,d\mathrm{vol}_g\). Smoothed coefficient and spectral bounds supply geometric regularity; the geometry lemma gives a sufficient global-hyperbolicity condition. Normalize sampling using the selected history law and density weights.

Verify the kernel moments in local normal coordinates. The Taylor estimate gives bias at most \(C_{\rm b}\varepsilon^2\). The covariance estimate controls the density-corrected local summands and yields variance at most \(C_{\rm mix}C_{\rm v}/(M\varepsilon^{D+2})\).

Choose \(\varepsilon_M=\ell_0M^{-\beta}\), with \(0<\beta<1/(D+4)\), as in the chapter. Both squared bias and variance vanish. The balanced exponent \(\beta=1/(D+6)\) gives the displayed mean-square rate. Compare the reconstructed episode operator with this estimator, controlling metric, density, neighborhood, and kernel errors after normalization.
''')
    else:
        for index, paragraph in enumerate(paragraphs):
            if number == 18 and index == 0:
                parts.append(r'''Carry the functional-inequality and evolution constants of the \hyperref[sec:equilibrium-thermodynamics]{equilibrium construction} along the selected cutoff and volume family. Poincar\'e supplies coercivity of its associated form; hypocoercivity supplies relaxation for the kinetic evolution. Complete the physical form comparison with the reconstructed gauge transfer Hamiltonian.

''')
            else:
                parts.append(tex(paragraph)+'\n\n')
    parts.append('\n\\begin{gather*}\n'+equations[number]+'\n\\end{gather*}\n')
    parts.append('\\subsection*{Proof mechanism}\n'+OVERRIDE_SKETCHES.get(number,sketches[number])+'\n')
    parts.append('\\subsection*{Analytic inputs}\n'+tex(tools)+'\n')
    append_source_results(refs)
    if number in ADDITIONS:
        added_text,added_refs=ADDITIONS[number]
        parts.append(added_text)
        append_source_results(added_refs,'Results supplying these constructions')
    if number==17:
        handoff='The same equilibrium observable hierarchy, recorded-time correlators, native local algebras, and the stated physical representation criteria. These identify the field law and transfer family on which the gap-survival theorem is applied.'
    parts.append('\\paragraph{Stage output.} '+tex(handoff)+'\n')
    if number in (9,15):
        parts.append(r'\paragraph{Gravitational application.} Section~\ref{sec:quantum-gravity} reuses these estimates for both the fitness-Hessian and Fractal Set curvature/action routes, retaining their common law and scale dependencies.'+'\n')

parts.append(render_source_links(QUANTUM_GRAVITY_ROUTES))
parts.append(CONSTRUCTION_SECTION)
parts.append(r'''
\clearpage
\section{The estimates that control the continuum}
\small
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{.23\textwidth}>{\raggedright\arraybackslash}p{.35\textwidth}>{\raggedright\arraybackslash}p{.35\textwidth}@{}}
\toprule Tool & Controlled quantity & Constant or identification retained\\\midrule\endhead
Moment and tail drift & Escape, tightness, uniform integrability & Moment order, drift coefficient, coverage defect\\
Poincar\'e inequality & Empirical variance and form coercivity & Joint-law \(C_*\), appropriate Dirichlet form\\
Joint LSI & Entropy, concentration, fluctuation moments & Full gradient, status term, law identification\\
Hypocoercivity & Kinetic relaxation & Modified norm, prefactor, positive rate\\
Population comparison & Particle and time approximation & Initial moments, bias, horizon, attraction branch\\
Normalized \(C^n\) calculus & Smooth reconstructed coefficients & Kernel widths, floors, denominator lower bound\\
Local kernel calculus & Differential-operator bias & Geometry, moments, normal-coordinate bounds\\
Covariance estimate & Shrinking-neighborhood variance & \(C_{\rm mix}/(M\varepsilon^{D+2})\)\\
Direct joint-LSI estimate & Continuum error for dependent smooth observations & \(C_*C_\nabla/(M\varepsilon^{D+4})\)\\
Sampling density correction & Geometric integration from probability samples & Actual \(q\), weight \(q^{-1}\), selected law\\
Recorded operator comparison & Actual graph reconstruction & Error after \(\varepsilon^{-D-2}\) normalization\\
Temporal and ensemble perturbation & Weak, stationary, conditioned errors & Stability, resolvent rate, survival weights\\
Native likelihood and source identities & Gauge action and correlations & Descriptor support, geometry-fiber normalization\\
Physical reflection and symmetry & Quantum Hilbert space and dynamics & Selected spacetime hierarchy and physical clock\\
Strong semigroup convergence & Survival of a positive gap & \(J_a\), \(\Omega_a\), cutoff/volume-uniform \(\lambda_*\)\\
\bottomrule
\end{longtable}
\normalsize
\paragraph{Joint scale choice.}
Fix the transition parameters for each declared family, choose its observation regime using the population/time estimates, and set a bandwidth for which the local bias and correlated variance vanish. Then control reconstruction and action quadrature at that normalization. If \(h\), algorithmic locality, confinement, or noise changes, retain the corresponding uniform estimates for that changed transition family. Finally pass the sourced hierarchy and physical transfer operators through the same cutoff and volume sequence.

\clearpage
\appendix
\section{Source theorem register}
The references below point to the full statements and proofs in the repository snapshot. File paths are relative to \texttt{docs/source/2\_fractal\_gas/}. Line numbers identify the start of the named directive. The labels provide stable search keys when line numbers change.
''')
for i,label in enumerate(used,1):
    c=catalog[label]
    parts.append('\n\\par\\medskip\\noindent\\begin{minipage}{\\linewidth}\\textbf{[R'+str(i)+'] '+tex(c['title'])+'}\\label{src:'+label+'}\\par\n')
    parts.append('{\\small\\texttt{\\detokenize{'+label+'}}\\par\n\\nolinkurl{'+c['file']+'}, line '+str(c['line'])+'.\\par}\n\\end{minipage}\\par\n')

parts.append('\n\\section{Chapter contribution map}\n')
parts.append('This map records the contributions used in the proof program. Chapter-local configurations, laws, and theorem hypotheses are retained when their results are composed. The experiment and calibration chapters supply observable definitions and diagnostics alongside the analytic results.\n')
for group,files,contribution in CHAPTER_GROUPS:
    parts.append('\\par\\medskip\\noindent\\begin{minipage}{\\linewidth}\n')
    parts.append('\\subsection*{'+tex(group)+'}\n')
    parts.append(tex(contribution)+'\n\n')
    for filename in files.split('; '):
        parts.append('{\\small\\nolinkurl{'+filename+'}\\par}\n')
    parts.append('\\end{minipage}\\par\n')
parts.append('\n\\end{document}\n')
out=ROOT/'output/pdf/volume_2_qft_yang_mills_proof_roadmap.tex'
out.write_text(''.join(parts))
print(out)
print(len(stages),'stages;',len(used),'source results;',len(CHAPTER_GROUPS),'chapter groups')
