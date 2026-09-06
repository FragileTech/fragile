"""Read-only numerical probes for the make physics review; no production patches."""

from types import SimpleNamespace
from unittest.mock import patch
import warnings

import gvar
import numpy as np
import torch

from fragile.physics.fractal_gas.cloning import CloneOperator
from fragile.physics.fractal_gas.euclidean_gas import EuclideanGas, SwarmState
from fragile.physics.fractal_gas.fitness import FitnessOperator
from fragile.physics.fractal_gas.kinetic_operator import KineticOperator
from fragile.physics.geometry.delaunai import compute_delaunay_data
from fragile.physics.mass_extraction.config import CovarianceConfig, MassExtractionConfig
from fragile.physics.mass_extraction.data_preparation import correlators_to_gvar, operator_series_to_correlator_samples
from fragile.physics.mass_extraction.pipeline import extract_masses
from fragile.physics.new_channels.correlator_channels import CorrelatorConfig, extract_mass_aic
from fragile.physics.new_channels.meson_phase_channels import compute_meson_phase_correlator_from_color
from fragile.physics.operators.correlators import compute_correlators_batched
from fragile.physics.operators.pipeline import PipelineResult
from fragile.physics.qft_utils.fft import _fft_correlator_batched


def kinetic(**kwargs):
    return KineticOperator(gamma=1.0, beta=1.0, delta_t=0.01, **kwargs)


def gas(**kwargs):
    return EuclideanGas(N=20, d=3, kinetic_op=kinetic(beta_curl=0.0, viscous_neighbor_weighting='riemannian_kernel_volume'),
                        cloning=CloneOperator(), fitness_op=FitnessOperator(), **kwargs)


torch.set_num_threads(1)
torch.manual_seed(42)

# Constant force isolates the B-step time increment without geometry assumptions.
k = kinetic(beta_curl=0.0)
s = SwarmState(torch.zeros(2, 3), torch.zeros(2, 3))
with patch.object(k, '_compute_viscous_force', return_value=torch.ones(2, 3)), patch('torch.randn', side_effect=lambda *a, **kw: torch.zeros(*a, **kw)):
    out = k.apply(s)
expected = 0.5 * k.dt * (1 + float(k.c1))
print('B-step ratio actual/BAOAB:', float(out.v[0, 0]) / expected)

edges = torch.tensor([[0, 1], [1, 0]])
s.v[0] = 1
k = kinetic(use_viscous_coupling=False, beta_curl=0.0)
try:
    k.apply(s, neighbor_edges=edges, edge_weights=None)
except Exception as exc:
    print('Disabled viscosity with graph:', type(exc).__name__, str(exc))
print('Disabled viscosity supplied weights force:', k._compute_viscous_force(s.x, s.v, edges, torch.ones(2)).tolist())

g = gas(clone_every=20)
h = g.run(3, seed=42)
print('Skipped cloning recorded events:', h.num_cloned.tolist())
print('Skipped cloning actual position changes:', (h.x_after_clone-h.x_before_clone[1:]).abs().max().item())
print('Skipped cloning recorded position changes:', h.clone_delta_x.abs().max().item())
pre_fit, _ = g.fitness_op(h.x_before_clone[1], h.rewards[0], h.companions_distance[0])
print('Stored fitness vs original-state fitness max difference:', (h.fitness[0]-pre_fit).abs().max().item())

x = torch.randn(20, 3)
x[1] = x[0]
tess = compute_delaunay_data(x, torch.zeros(20), spatial_dims=2)
degree = torch.bincount(tess.edge_index[0], minlength=20)
print('Exact duplicate zero-degree walkers:', torch.where(degree == 0)[0].tolist())

g = gas()
x = torch.randn(20, 3)
first = g._compute_tessellation(x)
x[:, 2] += 100 * torch.randn(20)
second = g._compute_tessellation(x)
print('Changing third coordinate leaves edges/rewards identical:', torch.equal(first['edges'], second['edges']), torch.equal(first['ricci'], second['ricci']))
g.kinetic_op.viscous_neighbor_weighting = 'riemannian_kernel_volume'
g.kinetic_op.viscous_length_scale = 0.001
first = g._compute_tessellation(x)
g.kinetic_op.viscous_length_scale = 1000
second = g._compute_tessellation(x)
print('Changing kernel length 0.001 -> 1000 leaves weights identical:', torch.equal(first['edge_weights'], second['edge_weights']))

g = gas()
g.kinetic_op.auto_thermostat = True
g.kinetic_op.temperature = 0.33
g.kinetic_op.n_kinetic_steps = 4
h = g.run(2, seed=42)
print('Recorded dt vs elapsed per iteration:', h.delta_t, 4 * h.delta_t)
print('Saved kinetic parameters:', h.params['kinetic'])

T, N = 40, 4
color = torch.randn(T, N, 3, dtype=torch.complex64)
color /= color.norm(dim=-1, keepdim=True)
pair = torch.tensor([1, 0, 3, 2]).expand(T, N)
meson = compute_meson_phase_correlator_from_color(color, torch.ones(T, N, dtype=torch.bool), pair, pair, max_lag=4)
print('Pair pseudoscalar C(0) vs averaged operator max:', float(meson.pseudoscalar[0]), float(meson.operator_pseudoscalar_series.abs().max()))
cfg = MassExtractionConfig(covariance=CovarianceConfig(method='bootstrap', n_bootstrap=20, block_size=5))
converted = correlators_to_gvar({'pseudoscalar': meson.pseudoscalar}, {'pseudoscalar': meson.operator_pseudoscalar_series}, cfg)
print('Pseudoscalar C(0) actually passed to fit:', gvar.mean(converted['pseudoscalar'][0]))

series = torch.randn(100)
for method in ['bootstrap', 'block_jackknife']:
    samples = operator_series_to_correlator_samples(series, 4, method=method, block_size=10, n_bootstrap=100)
    avg = gvar.dataset.avg_data(samples)
    centered = samples - samples.mean(axis=0)
    if method == 'bootstrap':
        reference = samples.std(axis=0, ddof=1)
    else:
        reference = np.sqrt((len(samples)-1)/len(samples)*(centered**2).sum(axis=0))
    print(method, 'reported/reference standard error:', gvar.sdev(avg)[0]/reference[0])

vector = torch.stack([series, -series], dim=-1)
corr = compute_correlators_batched({'vector': vector}, max_lag=4)['vector']
converted = correlators_to_gvar({'vector': corr}, {'vector': vector}, cfg)
print('Vector C(0) before/after covariance conversion:', float(corr[0]), gvar.mean(converted['vector'][0]))
cfg = MassExtractionConfig(covariance=CovarianceConfig(method='block_jackknife', block_size=10))
corr = _fft_correlator_batched(series[None], 4)[0]
joint = correlators_to_gvar({'a': corr, 'b': corr}, {'a': series, 'b': series}, cfg)
print('Identical-channel correlation retained:', gvar.evalcorr([joint['a'][0], joint['b'][0]])[0, 1])

multi = torch.randn(2, 20, 3)
actual = compute_correlators_batched({'v': multi}, 4, n_scales=2)['v']
reference = torch.stack([compute_correlators_batched({'v': row}, 4)['v'] for row in multi])
print('Multiscale vector correlation max error:', float((actual-reference).abs().max()))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    result = extract_masses(PipelineResult(correlators={'scalar': torch.exp(-0.2*torch.arange(12.)).repeat(2, 1)}))
print('Multiscale fitted channels:', list(result.channels), 'data keys:', list(result.data))

c = torch.exp(-2.0 * 0.1 * torch.arange(30))
print('AIC physical mass expected 2; returned:', extract_mass_aic(c, 0.1, CorrelatorConfig(window_widths=[5]))['mass'])

# Jackknife removes a block and joins points that were not adjacent in time.
ramp = torch.arange(8.)
samples = operator_series_to_correlator_samples(ramp, 1, method='block_jackknife', block_size=2, use_connected=False)
kept_adjacent = torch.tensor([0*1, 4*5, 5*6, 6*7], dtype=torch.float32).mean()
print('Delete [2,3] lag-1 mean actual/preserving original lag:', samples[1,1], float(kept_adjacent))

# Scalar mean is changed even for a constant raw correlator.
cfg = MassExtractionConfig(covariance=CovarianceConfig(method='bootstrap', n_bootstrap=10, block_size=5))
converted = correlators_to_gvar({'scalar': torch.full((5,), 9.)}, {'scalar': torch.full((40,), 3.)}, cfg)
print('Raw constant correlator converted C(0), expected 9:', gvar.mean(converted['scalar'][0]))

from fragile.physics.new_channels.dirac_spinors import build_dirac_gamma_matrices, color_to_dirac_spinor, compute_dirac_operators_from_spinors
from fragile.physics.operators.tensor_operators import compute_tensor_operators
from fragile.physics.operators.config import TensorOperatorConfig

prepared = SimpleNamespace(color=color, color_valid=torch.ones(T, N, dtype=torch.bool),
                          companions_distance=pair, companions_clone=pair,
                          device=color.device, scales=None, pairwise_distances=None)
tensor_series = compute_tensor_operators(prepared, TensorOperatorConfig())['tensor']
print('Standard tensor series max for random colors and mutual pairs:', float(tensor_series.abs().max()))
gamma = build_dirac_gamma_matrices()
psi, valid = color_to_dirac_spinor(color)
psi_parity, _ = color_to_dirac_spinor(-color.conj())
expected = torch.einsum('ab,...b->...a', gamma['gamma0'], psi)
print('Color->Dirac parity map max error:', float((psi_parity-expected).abs().max()))
# One directed pair avoids trivial cancellation in the parity test.
sample = torch.zeros(T, 1, dtype=torch.long)
neighbor = torch.ones(T, 1, 1, dtype=torch.long)
alive = torch.ones(T, N, dtype=torch.bool)
op = compute_dirac_operators_from_spinors(psi, valid, sample, neighbor, alive, gamma)
transformed = compute_dirac_operators_from_spinors(expected, valid, sample, neighbor, alive, gamma)
print('Implemented Dirac pseudoscalar parity even/odd residual:',
      float((transformed.pseudoscalar-op.pseudoscalar).abs().max()),
      float((transformed.pseudoscalar+op.pseudoscalar).abs().max()))

h = gas().run(12, seed=42, chunk_size=10)
print('Chunked frame count vs recorded step metadata:', h.n_recorded, h.recorded_steps)

k = kinetic(beta_curl=1.0, nu=0.0, curl_field=lambda x: torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]).expand(x.shape[0], 3, 3))
outputs = []
for mode in ['baoab', 'boris-baoab']:
    k.integrator = mode
    torch.manual_seed(42)
    outputs.append(k.apply(s, neighbor_edges=edges, edge_weights=torch.ones(2)).v)
print('baoab vs boris-baoab exactly equal with nonzero curl:', torch.equal(*outputs))

# Double precision FFT contract is lost before subtracting a large mean.
signal = 1e8 + torch.tensor([[0., 1., 0., -1.]], dtype=torch.float64)
print('Float64 connected C(0), expected 0.5:', float(_fft_correlator_batched(signal, 1)[0,0]))
