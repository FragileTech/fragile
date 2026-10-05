# Einstein–Hilbert operator validation

Reference h=0.002, T=0.33, viscosity=3, dimension=3, clone period=20, f64. All trajectories start coincident at rest. Seed means and minimum/maximum bands use three complete independent runs; the bands are not confidence intervals.

Checked 40800 native trajectory steps and 768 checkpoint replicas, independent within each ensemble. Exact check failures: 0. Maximum normalized algebraic residual: 1.335e-15.

| N | Final W | Final mean squared speed | Late expected cloning ratio | Negative late gates | Maximum kappa |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.838764 | 0.265035 | 1.001767 | 161/300 | 1.008112 |
| 128 | 2.063589 | 0.313732 | 1.007416 | 91/300 | 1.007567 |
| 500 | 3.768393 | 0.292508 | 1.007620 | 37/300 | 1.009319 |

Late means steps 2001–4000 in the longer report, with a gate every 20 steps. These sampled states do not certify a global contraction hypothesis. The sufficient unweighted velocity margin requires kappa < exp(0.001) = 1.0010005; the maxima in this table exceed it. This failure of a sufficient bound is not evidence that the actual velocity energy grows.

| N | Gate | Independent replicas | Cloning mean / SE | Kinetic mean / SE |
|---:|---:|---:|---:|---:|
| 32 | 40 | 128 | 1.028 | -0.794 |
| 64 | 40 | 128 | 1.029 | 1.395 |
| 128 | 4000 | 256 | -0.763 | 0.132 |
| 500 | 4000 | 256 | -0.392 | -0.189 |

Replica errors subtract each replica's own exact conditional prediction, including its sampled diversity fitness and realized post-cloning geometry. Standard errors use independent completed replicas, never interacting walkers. The error bars in the figure are one estimated standard error. Simulations support the identities; the continued spread growth through time 8 does not establish either stationary convergence or nonconvergence.
