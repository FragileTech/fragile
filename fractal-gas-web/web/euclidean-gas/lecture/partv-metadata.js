export const metadata = [
  {
    id: "V-01",
    part: "V",
    title: "A slot is not an ancestor",
    question: "Which history did this walker inherit?",
    prediction:
      "Accepted copies follow the donor ancestry; slot continuity follows the recipient. Rejected copies preserve ancestry.",
    explanation:
      "Accepted copies follow the donor ancestry; slot continuity follows the recipient. Rejected copies preserve ancestry.",
  },
  {
    id: "V-02",
    part: "V",
    title: "Build the three-edge interaction",
    question: "How do CST, IG and IA close a triangle?",
    prediction:
      "The oriented CST + IA − IG boundary closes for eligible same-frame interactions.",
    explanation:
      "The oriented CST + IA − IG boundary closes for eligible same-frame interactions.",
  },
  {
    id: "V-03",
    part: "V",
    title: "Reconstruct the run from its record",
    question: "Can the archived stages reproduce the visible trajectory?",
    prediction:
      "Absolute anchors reconstruct the recorded coordinates; a continuous spinor rotation changes sign at 2π and returns at 4π.",
    explanation:
      "Absolute anchors reconstruct the recorded coordinates; a continuous spinor rotation changes sign at 2π and returns at 4π.",
  },
  {
    id: "V-04",
    part: "V",
    title: "Triangles, plaquettes, and transport",
    question: "What survives a change of local frame?",
    prediction:
      "Supplied connection holonomy transforms by conjugation; its trace is invariant, with rebasing required between triangles.",
    explanation:
      "Supplied connection holonomy transforms by conjugation; its trace is invariant, with rebasing required between triangles.",
  },
  {
    id: "V-05",
    part: "V",
    title: "From fitness curvature to a metric",
    question: "How does fitness curvature shape the metric?",
    prediction:
      "Exact conditional derivatives include population normalization; regularization controls the smallest metric eigenvalue.",
    explanation:
      "Exact conditional derivatives include population normalization; regularization controls the smallest metric eigenvalue.",
  },
  {
    id: "V-06",
    part: "V",
    title: "Predict the next noise cloud",
    question: "Can the metric predict one O-step innovation cloud?",
    prediction:
      "Conditional covariance is T(1 − exp(−2γh))g⁻¹; sampling error decreases with independent innovations.",
    explanation:
      "Conditional covariance is T(1 − exp(−2γh))g⁻¹; sampling error decreases with independent innovations.",
  },
  {
    id: "V-07",
    part: "V",
    title: "Coordinate area versus geometric volume",
    question: "How much geometric volume does each cell hold?",
    prediction:
      "For constant metric g, geometric volume equals coordinate area times √det(g); the cells sum to the clipped region.",
    explanation:
      "For constant metric g, geometric volume equals coordinate area times √det(g); the cells sum to the clipped region.",
  },
  {
    id: "V-08",
    part: "V",
    title: "Relaxation in anisotropic geometry",
    question: "Which covariance does the anisotropic thermostat approach?",
    prediction:
      "The harmonic BAOAB covariance follows an exact transient recurrence and a discrete Lyapunov equation. Sampling uncertainty and stationary timestep bias are separate.",
    explanation:
      "The harmonic BAOAB covariance follows an exact transient recurrence and a discrete Lyapunov equation. Sampling uncertainty and stationary timestep bias are separate.",
  },
  {
    id: "V-09",
    part: "V",
    title: "Cells and their dual graph",
    question: "Which walkers share a cell boundary?",
    prediction:
      "Metric Voronoi cells partition the observation window; coincident sites share a geometric cell while keeping distinct identities.",
    explanation:
      "Metric Voronoi cells partition the observation window; coincident sites share a geometric cell while keeping distinct identities.",
  },
  {
    id: "V-10",
    part: "V",
    title: "Which operation changed the neighbors?",
    question: "Did copying or kinetic motion change this interface?",
    prediction:
      "Comparing recorded pre-clone, post-clone and final sites locates neighbor changes by operation.",
    explanation:
      "Comparing recorded pre-clone, post-clone and final sites locates neighbor changes by operation.",
  },
  {
    id: "V-11",
    part: "V",
    title: "Sweep the cells through time",
    question: "What volume does a moving cell occupy?",
    prediction:
      "A shared spacetime partition closes the observation slab; refinement resolves moving interfaces and clone jumps use one-sided caps.",
    explanation:
      "A shared spacetime partition closes the observation slab; refinement resolves moving interfaces and clone jumps use one-sided caps.",
  },
  {
    id: "V-12",
    part: "V",
    title: "Maintain the mesh and count its activity",
    question: "How much geometric work accompanies each update?",
    prediction:
      "Changed unique interfaces and incident cell counts differ; reconstruction work depends on the realized configuration.",
    explanation:
      "Changed unique interfaces and incident cell counts differ; reconstruction work depends on the realized configuration.",
  },
  {
    id: "V-13",
    part: "V",
    title: "Smooth fields within a fixed stratum",
    question: "What changes continuously when the query point moves?",
    prediction:
      "With alive status and sampled companions frozen, regularized fitness has smooth exact derivatives; changing the stratum is a separate operation.",
    explanation:
      "With alive status and sampled companions frozen, regularized fitness has smooth exact derivatives; changing the stratum is a separate operation.",
  },
  {
    id: "V-14",
    part: "V",
    title: "Normalize the sampled geometry",
    question: "Can a biased cloud integrate volume correctly?",
    prediction:
      "Inverse sampling-density weights recover the known integral; effective sample size reveals weight concentration.",
    explanation:
      "Inverse sampling-density weights recover the known integral; effective sample size reveals weight concentration.",
  },
  {
    id: "V-15",
    part: "V",
    title: "Build a signed wave-operator kernel",
    question: "Which moments make the kernel a wave operator?",
    prediction:
      "The signed kernel must satisfy its actual cutoff-domain moment equations, including the temporal sign.",
    explanation:
      "The signed kernel must satisfy its actual cutoff-domain moment equations, including the temporal sign.",
  },
  {
    id: "V-16",
    part: "V",
    title: "Find the useful bandwidth",
    question: "When does a smaller bandwidth improve the estimate?",
    prediction:
      "Bias falls with bandwidth while sampling variance grows; the useful bandwidth balances the two.",
    explanation:
      "Bias falls with bandwidth while sampling variance grows; the useful bandwidth balances the two.",
  },
  {
    id: "V-17",
    part: "V",
    title: "Recorded order and geometric light cones",
    question: "Which events are related by each order?",
    prediction:
      "Recorded CST reachability and geometric light-cone comparability are distinct relations that can be inspected on the same events.",
    explanation:
      "Recorded CST reachability and geometric light-cone comparability are distinct relations that can be inspected on the same events.",
  },
  {
    id: "V-18",
    part: "V",
    title: "Count events to measure volume",
    question: "What count law belongs to this sampling experiment?",
    prediction:
      "Poisson and fixed-count sampling have different count variances; weighting depends on the specified density.",
    explanation:
      "Poisson and fixed-count sampling have different count variances; weighting depends on the specified density.",
  },
  {
    id: "V-19",
    part: "V",
    title: "Estimate dimension from comparable pairs",
    question: "How does the comparable-pair fraction encode dimension?",
    prediction:
      "Uniform Alexandrov samples approach comparable-pair fractions 1/2, 8/35 and 1/10 in spacetime dimensions 2, 3 and 4, respectively.",
    explanation:
      "Uniform Alexandrov samples approach comparable-pair fractions 1/2, 8/35 and 1/10 in spacetime dimensions 2, 3 and 4, respectively.",
  },
  {
    id: "V-20",
    part: "V",
    title: "Curvature, calibration, and amplified noise",
    question: "How does the curvature signal compete with sampling noise?",
    prediction:
      "Independent kernel-moment calibration recovers R = 2K for the chosen product family as resolution and sample size improve.",
    explanation:
      "Independent kernel-moment calibration recovers R = 2K for the chosen product family as resolution and sample size improve.",
  },
];
