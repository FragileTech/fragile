"""Vector overview of the roadmap's eighteen proof stages."""

PROOF_PIPELINE = r"""
\clearpage
\begin{landscape}
\thispagestyle{plain}
\section*{The proof strategy at a glance}
\phantomsection\label{sec:proof-pipeline}
\addcontentsline{toc}{section}{The proof strategy at a glance}
\vspace{-2mm}
{\small One configured gas, one selected history law, three spatial coordinates and recorded algorithmic time. Stage numbers link to the detailed arguments.}
\par\vspace{1mm}
\begin{center}
\begin{tikzpicture}[
  x=1cm,y=1cm,
  every node/.style={font=\fontsize{9}{10.5}\selectfont,align=center},
  box/.style={draw=ink!65,fill=ink!4,rounded corners=2pt,line width=.6pt,
    text width=10.8cm,minimum height=1.22cm,inner sep=5pt},
  full/.style={box,text width=23cm},
  control/.style={box,draw=teal!85!black,fill=teal!6},
  target/.style={box,draw=orange!65!black,fill=orange!7,dashed,line width=.8pt},
  flow/.style={-{Stealth[length=2.2mm,width=1.7mm]},draw=ink!75,line width=.8pt},
  link/.style={draw=ink!75,line width=.8pt},
  note/.style={font=\fontsize{8}{9}\selectfont,text=ink!85},
]
\node[full] (gas) at (0,0) {
  \textbf{THE COMPLETE EUCLIDEAN GAS AND ITS EQUILIBRIUM}\\[2pt]
  \hyperref[stage:1]{\textbf{1} Full transition kernel}\quad$\longrightarrow$\quad
  \hyperref[stage:2]{\textbf{2} Component collisions}\quad$\longrightarrow$\quad
  \hyperref[stage:3]{\textbf{3} Confinement and tails}\quad$\longrightarrow$\quad
  \hyperref[stage:4]{\textbf{4} Selected population law}\\[1pt]
  {\fontsize{8}{9}\selectfont Complete-update balance: $\pi_NP_{N,h}=\pi_N$ or $\nu_NQ_{N,h}=\alpha_N\nu_N$; retain the corresponding history conditioning.}
};

\node[control] (meanfield) at (-6.1,-1.85) {
  \textbf{POPULATION EVOLUTION}\\[2pt]
  \hyperref[stage:5]{\textbf{5} Mean field and exact balances}\quad$\longrightarrow$\quad
  \hyperref[stage:6]{\textbf{6} Propagation of chaos}\\[1pt]
  {\fontsize{8}{9}\selectfont Nonlinear update $\mathcal F_h$; finite-time and stationary population comparison.}
};
\node[box] (record) at (6.1,-1.85) {
  \textbf{THE SAME LAW IN FIELD COORDINATES}\\[2pt]
  \hyperref[stage:10]{\textbf{10} Complete record $\longrightarrow$ causal $3+1$ Fractal Set}\\[1pt]
  {\fontsize{8}{9}\selectfont Lossless descriptors, invariant observables, transported dynamics and memory.}
};

\node[control,minimum height=1.42cm] (tools) at (-6.1,-3.8) {
  \textbf{UNIFORM ANALYTIC CONTROL}\\[2pt]
  \hyperref[stage:7]{\textbf{7} Joint LSI}\quad
  \hyperref[stage:8]{\textbf{8} Poincar\'e, concentration, hypocoercivity}\\[1pt]
  \hyperref[stage:9]{\textbf{9} Normalized $C^n$ coefficient calculus}\\[1pt]
  {\fontsize{8}{9}\selectfont Actual-law criteria; uniform constants for moments, relaxation and regularity.}
};
\node[box,minimum height=1.42cm] (fields) at (6.1,-3.8) {
  \textbf{GAUGE AND MATTER CONSTRUCTION}\\[2pt]
  \hyperref[stage:11]{\textbf{11} Connections, holonomies and matter}\quad
  \hyperref[stage:13]{\textbf{13} CAR algebra}\\[1pt]
  \hyperref[stage:12]{\textbf{12} Native likelihood action, sources and temporal correlations}\\[1pt]
  {\fontsize{8}{9}\selectfont $d\nu=e^{-S_{\rm alg}}d\lambda$; retain geometry fibers and source-response identities.}
};

\coordinate (join) at (0,-4.98);
\node[control,minimum height=1.35cm] (continuum) at (-6.1,-6.05) {
  \textbf{CONTINUUM AND YANG--MILLS IDENTIFICATION}\\[2pt]
  \hyperref[stage:15]{\textbf{15} Geometry, density correction, kernel bias and variance}\\[1pt]
  \hyperref[stage:16]{\textbf{16} Action quadrature and temporal / ensemble limits}\\[1pt]
  {\fontsize{8}{9}\selectfont Geometry and sampling criteria; native-to-YM action correspondence.}
};
\node[box,minimum height=1.35cm] (hierarchy) at (6.1,-6.05) {
  \textbf{ONE COMMON EQUILIBRIUM HIERARCHY}\\[2pt]
  \hyperref[stage:14]{\textbf{14} Fluctuation fields, gauge words and sourced limits}\\[1pt]
  {\fontsize{8}{9}\selectfont Tightness and moments; reflected products on one common subsequence.}\\[1pt]
  {\fontsize{8}{9}\selectfont Carry the same observable normalizations through the continuum estimates.}
};

\node[target,text width=23cm,minimum height=1.3cm] (reconstruction) at (0,-8.05) {
  \textbf{\hyperref[stage:17]{17\quad PHYSICAL RECONSTRUCTION FROM THESE CORRELATIONS}}\\[2pt]
  Establish physical reflection positivity, symmetry and regularity; identify the nontrivial Yang--Mills hierarchy.\\[1pt]
  {\fontsize{8}{9}\selectfont Reconstruct $(\mathcal H,\Omega,H)$ and the local fields; verify physical covariance, locality and spectrum with the calibrated clock.}
};
\node[target,minimum height=1.35cm] (gapinput) at (-6.1,-10.05) {
  \textbf{\hyperref[stage:18]{18\quad PHYSICAL GAP AND OPERATOR LIMITS}}\\[2pt]
  Compare population coercivity with the gauge transfer form.\\[1pt]
  {\fontsize{8}{9}\selectfont Obtain cutoff / volume-uniform $\lambda_*>0$; identify strong semigroup and vacuum limits.}
};
\node[control,minimum height=1.35cm] (conclusion) at (6.1,-10.05) {
  \textbf{\hyperref[stage:18]{18\quad APPLY THE GAP-SURVIVAL THEOREM}}\\[2pt]
  $\Spec(H)\subset\{0\}\cup[\lambda_*,\infty)$\\[1pt]
  {\fontsize{8}{9}\selectfont Under the reconstruction and uniform-limit hypotheses:\\[1pt]nontrivial quantum Yang--Mills, unique vacuum, positive mass gap.}
};

\draw[flow] (gas.south -| meanfield.north) -- (meanfield.north);
\draw[flow] (gas.south -| record.north) -- (record.north);
\draw[flow] (meanfield.south) -- (tools.north);
\draw[flow] (record.south) -- (fields.north);
\draw[link] (tools.south) -- (-6.1,-4.98) -- (join);
\draw[link] (fields.south) -- (6.1,-4.98) -- (join);
\fill[ink!75] (join) circle (1.2pt);
\draw[flow] (-6.1,-4.98) -- (continuum.north);
\draw[flow] (6.1,-4.98) -- (hierarchy.north);
\node[note,fill=white,inner sep=2pt] at (0,-4.98) {combine estimates and recorded law};
\draw[flow] (continuum.south) -- (reconstruction.north -| continuum.south);
\draw[flow] (hierarchy.south) -- (reconstruction.north -| hierarchy.south);
\draw[flow] (reconstruction.south -| gapinput.north) -- (gapinput.north);
\draw[flow] (gapinput.east) -- (conclusion.west);

\node[note,text width=23.5cm] at (0,-12.0) {
  \textbf{Solid boxes:} source constructions and tools, with their stated hypotheses.
  \quad\textbf{Dashed boxes:} final physical identifications to establish.\\[2pt]
  \textbf{Joint scale ledger throughout:} population $N$, observations $M$, algorithm step $h$,
  bandwidth $\varepsilon$, field cutoff $a$, volume $L$.\\[1pt]
  Changing $h$, algorithmic locality or noise requires uniform estimates for that transition family; keep the selected law and physical units fixed in each comparison.\\[1pt]
  \hyperref[sec:quantum-gravity]{\textbf{Quantum-gravity extension:} fitness-Hessian curvature and Fractal Set operator/action convergence share this foundation.}
};
\end{tikzpicture}
\end{center}
\end{landscape}
\clearpage
"""
