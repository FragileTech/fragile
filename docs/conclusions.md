# Conclusion

The Fragile Agent is a {prf:ref}`def-bounded-rationality-controller` specification: the environment is described by an observation generator $P_{\partial}$, the agent's internal state is a split macro/micro bundle, and stability is enforced by explicit defect functionals rather than implicit optimizer heuristics.

1. Discretizing the macro channel $K$ turns enclosure, capacity, and grounding into well-typed information constraints ($I(X;K)$, $H(K)$, closure cross-entropy) via the {prf:ref}`def-boundary-markov-blanket`.
2. The critic induces a Fisher/Hessian sensitivity geometry via the {prf:ref}`def-mass-tensor`; the policy becomes a regulated flow on a curved manifold, with stability and coupling audited by Gate Nodes and Barriers.
3. Exploration and control are expressed via MaxEnt/KL-control and causal path entropy on $\mathcal{K}$; belief evolution is filtering + projection ({ref}`Section 11 <sec-intrinsic-motivation-maximum-entropy-exploration>`).
4. When representational complexity is constrained by finite interface capacity, the latent metric obeys the Capacity-Constrained Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`); deviations yield computable consistency defects ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`).
5. The hybrid discrete-continuous state space admits a canonical Wasserstein-Fisher-Rao geometry ({prf:ref}`def-the-wfr-action`, {ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`), unifying transport (continuous motion within charts) and reaction (discrete jumps between charts) in a single variational principle.

{ref}`Appendix A <sec-appendix-a-full-derivations>` records the full derivations. {ref}`Appendix B <sec-appendix-b-units-parameters-and-coefficients>` consolidates notation and all regularization losses.



(sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces)=
