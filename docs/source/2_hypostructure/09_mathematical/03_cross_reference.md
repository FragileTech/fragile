---
title: "Foundation-Sieve Cross-Reference"
---

(sec-foundation-sieve-cross-reference)=
## Foundation - Sieve Cross-Reference

The following table provides the complete mapping from Sieve components to their substantiating foundational theorems.

### Kernel Logic Cross-Reference

| **Sieve Component** | **Foundation Theorem** | **Certificate** | **Primary Literature** |
|---------------------|------------------------|-----------------|------------------------|
| {prf:ref}`def-node-lock` | {prf:ref}`mt-krnl-exclusion` | $K_{\text{Lock}}^{\text{blk}}$ | Grothendieck, Mac Lane |
| {prf:ref}`def-node-compact` | Metatheorem KRNL-Trichotomy | Trichotomy | Lions, Kenig-Merle |
| {prf:ref}`def-node-complex` | {prf:ref}`mt-resolve-profile` | $K_{11}^{\text{lib/tame/inc}}$ | van den Dries, Kurdyka |
| Meta-Learning | {prf:ref}`mt-krnl-equivariance` | $K_{\text{SV08}}^+$ | Noether, Cohen-Welling |

### Gate Evaluator Cross-Reference

| **Blue Node** | **Foundation Theorem** | **Predicate** | **Primary Literature** |
|---------------|------------------------|---------------|------------------------|
| {prf:ref}`def-node-scale` | {prf:ref}`mt-lock-tactic-scale` | $\alpha > \beta$ | Merle-Zaag, Kenig-Merle |
| {prf:ref}`def-node-stiffness` | {prf:ref}`mt-lock-spectral-gen` | $\sigma_{\min} > 0$ | Łojasiewicz, Simon |
| {prf:ref}`def-node-ergo` | {prf:ref}`mt-lock-ergodic-mixing` | $\tau_{\text{mix}} < \infty$ | Birkhoff, Sinai |
| {prf:ref}`def-node-oscillate` | {prf:ref}`mt-lock-spectral-dist` | $\|[D,a]\| < \infty$ | Connes |
| {prf:ref}`def-node-boundary` | {prf:ref}`mt-lock-antichain` | min-cut/max-flow | Menger, De Giorgi |

### Barrier Defense Cross-Reference

| **Orange Barrier** | **Foundation Theorem** | **Blocking Mechanism** | **Primary Literature** |
|--------------------|------------------------|------------------------|------------------------|
| {prf:ref}`def-barrier-sat` | {prf:ref}`mt-up-saturation-principle` | $\mathcal{L}\mathcal{V} \leq -\lambda\mathcal{V} + b$ | Meyn-Tweedie, Hairer |
| {prf:ref}`def-barrier-causal` | {prf:ref}`mt-up-causal-barrier` | $d(u) < \infty \Rightarrow t < \infty$ | Bennett, Penrose |
| {prf:ref}`def-barrier-cap` | {prf:ref}`mt-lock-tactic-capacity` | $\text{Cap}(B) < \infty \Rightarrow \mu_T(B) < \infty$ | Federer, Maz'ya |
| {prf:ref}`def-barrier-action` | {prf:ref}`mt-up-shadow` | $\mu(\tau \neq 0) \leq e^{-c\Delta^2}$ | Herbst, Łojasiewicz |
| {prf:ref}`def-barrier-bode` | {prf:ref}`thm-bode` | $\int \log|S| d\omega = \pi \sum p_i$ | Bode, Doyle |
| {prf:ref}`def-barrier-epi` | {prf:ref}`mt-act-horizon` | $I(X;Z) \leq I(X;Y)$ | Cover-Thomas, Landauer |

### Surgery Construction Cross-Reference

| **Purple Surgery** | **Foundation Theorem** | **Construction** | **Primary Literature** |
|--------------------|------------------------|------------------|------------------------|
| {prf:ref}`def-surgery-se` | {prf:ref}`mt-act-lift` | $\mathscr{T} = (T, A, G)$ | Hairer (2014) |
| SurgTE (Tunnel) | {prf:ref}`mt-act-surgery-2` | Excise + Cap | Perelman (2002-03) |
| {prf:ref}`def-surgery-cd` | {prf:ref}`mt-act-projective` | Slack variables | Boyd-Vandenberghe |
| SurgSD (Ghost) | {prf:ref}`mt-act-ghost` | Ghost fields $(c, \bar{c})$ | Faddeev-Popov, BRST |
| SurgBC (Adjoint) | {prf:ref}`mt-act-align` | Lagrange $\lambda$ | Pontryagin |
| {prf:ref}`def-surgery-ce` | {prf:ref}`mt-act-compactify` | Conformal $\Omega$ | Penrose |
