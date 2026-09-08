# libcmaes for Optimization Lab

Upstream: https://github.com/CMA-ES/libcmaes
Commit: 6a53cd562c85dc6df3b8df317be3ecb415240c2d (upstream version 0.10.3).
License: upstream LICENSE offers Apache-2.0 or LGPL-3.0-or-later; retain all notices.

The Lab builds a static subset with Eigen 3.4, no OpenMP, Python, surrogates,
examples, or file logging. Local source patch: BIPOP restart RNG uses the supplied
seed XOR 0x9e3779b9 rather than random_device/time. The Lab adapter maps seed zero
to a fixed nonzero upstream seed and exposes the upstream restart primitives as
an incremental scheduler. Whole-generation budget admission is stricter than the
batch driver's post-generation termination check. Numeric covariance updates and
normal sampling remain upstream. See src/optimization/cma.cpp for integration.

The adapter saves/restores upstream scalar_normal_dist_op<double>::rng around
construction and sampling: upstream shares this generator across instances.
This keeps interleaved Lab sessions independent without changing normal draws.
Lab/native API calls are serialized; the C API is not a concurrent-thread API.
