(sec-appendix-g-architecture-modules-reference)=
# Appendix G: Architecture Modules Reference

## TLDR

- Centralizes **~22 neural network modules** defined throughout Volume 1
- Organized by domain:
  - **Disentangled VAE** ({ref}`Part III <sec-architecture-the-disentangled-vq-vae-rnn>`)
  - **Supervised Topology** ({ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`)
  - **Lorentzian Memory Attention** ({ref}`Part VII <sec-covariant-memory-attention-architecture>`)
  - **Gauge-Covariant Attention** ({ref}`Part VIII <sec-covariant-cross-attention-architecture>`)
- Each module includes: class signature, key parameters, input/output shapes, purpose, and source reference
- Use as a single reference when implementing the Fragile Agent architecture
- All modules follow the gauge-covariant paradigm with explicit unit tracking

---

## G.1 Disentangled VAE Modules

These modules implement the split-latent VQ-VAE architecture from {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, separating observations into macro state $K$, nuisance $Z_n$, and texture $Z_{\mathrm{tex}}$.

### G.1.1 DisentangledConfig

:::{prf:definition} G.1.1 (DisentangledConfig)
:label: def-g-disentangled-config

**Class signature:**
```python
@dataclass
class DisentangledConfig:
    obs_dim: int = 64 * 64 * 3        # Observation dimension [pixels]
    hidden_dim: int = 256              # Encoder hidden dimension
    macro_embed_dim: int = 32          # Macro embedding dim (code vectors e_k)
    codebook_size: int = 512           # Number of discrete macrostates |K|
    nuisance_dim: int = 32             # Structured nuisance latent dimension
    tex_dim: int = 96                  # Texture latent dimension
    action_dim: int = 4                # Action dimension
    rnn_hidden_dim: int = 256          # Dynamics model RNN hidden
    lambda_closure: float = 1.0        # Causal enclosure weight
    lambda_slowness: float = 0.1       # Temporal smoothness weight
    lambda_nuis_kl: float = 0.01       # Nuisance KL weight
    lambda_tex_kl: float = 0.05        # Texture KL weight
    lambda_vq: float = 1.0             # VQ codebook + commitment weight
    lambda_recon: float = 1.0          # Reconstruction weight
    tex_dropout_prob: float = 0.5      # Texture dropout probability
```

**Purpose:** Configuration dataclass for the split-latent (macro + nuisance + texture) agent. Controls the latent decomposition dimensions and loss weights for training.

**Key parameters:**
- `codebook_size` – Number of discrete macro symbols $|K|$ (typically 512 or 1024)
- `macro_embed_dim` – Dimension of each codebook vector $e_k$ [nat]
- `tex_dropout_prob` – Probability of dropping texture during training (forces macro+nuisance decoding)

**Units:** `obs_dim` in [pixels], latent dimensions in [nat], loss weights dimensionless.

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, line 187.
:::

### G.1.2 Encoder

:::{prf:definition} G.1.2 (Encoder)
:label: def-g-encoder

**Class signature:**
```python
class Encoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: `x` shape `[B, C, H, W]` – RGB image observations
- Output: `h` shape `[B, hidden_dim]` – Encoded features

**Purpose:** Shared convolutional encoder backbone that maps observations to a latent representation. Uses a 4-layer CNN with GELU activations and a final linear projection.

**Architecture:**
```
Conv2d(3→32, 4×4, stride=2) → GELU
Conv2d(32→64, 4×4, stride=2) → GELU
Conv2d(64→128, 4×4, stride=2) → GELU
Conv2d(128→256, 4×4, stride=2) → GELU
Flatten → Linear(4096→hidden_dim)
```

**Key parameters:**
- `obs_dim` – Input observation dimension (for 64×64 RGB: 64×64×3)
- `hidden_dim` – Output feature dimension [nat]

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, line 216.
:::

### G.1.3 VectorQuantizer

:::{prf:definition} G.1.3 (VectorQuantizer)
:label: def-g-vector-quantizer

**Class signature:**
```python
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embed_dim: int, beta: float = 0.25):
        ...

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...
```

**Input/Output:**
- Input: `z_e` shape `[B, D]` – Continuous encoder output
- Output: `(z_q, indices, vq_loss)` where:
  - `z_q` shape `[B, D]` – Quantized embedding (with straight-through gradient)
  - `indices` shape `[B]` – Discrete code indices $K_t \in \{0, \ldots, |K|-1\}$
  - `vq_loss` shape `[]` – VQ loss (commitment + codebook)

**Purpose:** Maps continuous encoder outputs to discrete codes using nearest-neighbor lookup and straight-through estimator for gradient flow.

**Key parameters:**
- `codebook_size` – Number of discrete symbols $|K|$
- `embed_dim` – Dimension of each codebook vector $e_k$ [nat]
- `beta` – Commitment loss weight (default 0.25)

**Mathematical operation:**
$$K_t = \arg\min_{k \in \{1,\ldots,|K|\}} \|z_e - e_k\|^2$$
$$z_q = e_{K_t} + \operatorname{sg}[z_e - e_{K_t}]$$

where $\operatorname{sg}[\cdot]$ is the stop-gradient operator.

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, line 267.
:::

### G.1.4 Decoder

:::{prf:definition} G.1.4 (Decoder)
:label: def-g-decoder

**Class signature:**
```python
class Decoder(nn.Module):
    def __init__(self, macro_dim: int, nuisance_dim: int, tex_dim: int, obs_channels: int = 3):
        ...

    def forward(self, z_macro: torch.Tensor, z_nuis: torch.Tensor, z_tex: torch.Tensor) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input:
  - `z_macro` shape `[B, macro_dim]` – Macro latent (quantized)
  - `z_nuis` shape `[B, nuisance_dim]` – Nuisance latent
  - `z_tex` shape `[B, tex_dim]` – Texture latent
- Output: `x_recon` shape `[B, C, H, W]` – Reconstructed observation

**Purpose:** Reconstructs observations from the concatenated latent triple $(K_t, Z_{n,t}, Z_{\mathrm{tex},t})$. Uses transposed convolutions to upsample.

**Architecture:**
```
Linear(macro+nuis+tex → 4096) → GELU → Reshape(256×4×4)
ConvTranspose2d(256→128, 4×4, stride=2) → GELU
ConvTranspose2d(128→64, 4×4, stride=2) → GELU
ConvTranspose2d(64→32, 4×4, stride=2) → GELU
ConvTranspose2d(32→3, 4×4, stride=2) → Sigmoid
```

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, line 309.
:::

### G.1.5 MacroDynamicsModel

:::{prf:definition} G.1.5 (MacroDynamicsModel)
:label: def-g-macro-dynamics-model

**Class signature:**
```python
class MacroDynamicsModel(nn.Module):
    def __init__(self, macro_embed_dim: int, action_dim: int, hidden_dim: int, codebook_size: int):
        ...

    def forward(self, z_macro: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...
```

**Input/Output:**
- Input:
  - `z_macro` shape `[B, macro_embed_dim]` – Current macro embedding
  - `action` shape `[B, action_dim]` – Action taken
  - `hidden` shape `[B, hidden_dim]` – GRU hidden state
- Output: `(logits, hidden_next, z_pred)` where:
  - `logits` shape `[B, codebook_size]` – Logits over next macro state
  - `hidden_next` shape `[B, hidden_dim]` – Updated hidden state
  - `z_pred` shape `[B, macro_embed_dim]` – Predicted next embedding

**Purpose:** Micro-blind world model that predicts next macro state from current macro state and action. Crucially, this module never sees $Z_n$ or $Z_{\mathrm{tex}}$, enforcing the causal enclosure property.

**Key insight:** By being blind to nuisance and texture, the model forces all predictively relevant information into the macro channel, implementing Definition {prf:ref}`def-causal-enclosure`.

**Architecture:**
```
GRUCell(macro_embed + action → hidden)
MLP(hidden → codebook_size logits)
MLP(hidden → macro_embed prediction)
```

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, Definition {prf:ref}`def-causal-enclosure`, line 352.
:::

### G.1.6 DisentangledAgent

:::{prf:definition} G.1.6 (DisentangledAgent)
:label: def-g-disentangled-agent

**Class signature:**
```python
class DisentangledAgent(nn.Module):
    def __init__(self, config: DisentangledConfig):
        ...

    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def decode(self, z_macro: torch.Tensor, z_nuis: torch.Tensor, z_tex: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, obs: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...
```

**Input/Output:**
- Input:
  - `obs` shape `[B, C, H, W]` – Observation
  - `action` shape `[B, action_dim]` – Action
  - `hidden` shape `[B, hidden_dim]` – Dynamics hidden state
- Output: Dictionary containing:
  - `z_macro` shape `[B, macro_embed_dim]` – Quantized macro latent
  - `z_nuis` shape `[B, nuisance_dim]` – Nuisance latent (mean)
  - `z_tex` shape `[B, tex_dim]` – Texture latent (mean)
  - `indices` shape `[B]` – Discrete macro indices
  - `recon` shape `[B, C, H, W]` – Reconstruction
  - `next_logits` shape `[B, codebook_size]` – Next-state prediction
  - `hidden_next` shape `[B, hidden_dim]` – Updated hidden state
  - `losses` – Dictionary of individual loss terms

**Purpose:** Full split-latent VQ-VAE agent combining encoder, vector quantizer, nuisance/texture heads, decoder, and macro dynamics model. Implements the complete latent decomposition $Z_t = (K_t, Z_{n,t}, Z_{\mathrm{tex},t})$.

**Components:**
- `encoder` – Shared CNN backbone
- `vq` – VectorQuantizer for macro discretization
- `head_macro` – Linear projection to macro embedding
- `head_nuis_mu/logvar` – Nuisance VAE heads
- `head_tex_mu/logvar` – Texture VAE heads
- `decoder` – Reconstruction decoder
- `dynamics` – MacroDynamicsModel (micro-blind)

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, line 412.
:::

### G.1.7 HierarchicalDisentangled

:::{prf:definition} G.1.7 (HierarchicalDisentangled)
:label: def-g-hierarchical-disentangled

**Class signature:**
```python
class HierarchicalDisentangled(nn.Module):
    def __init__(
        self,
        config: DisentangledConfig,
        n_levels: int = 3,
        level_dims: List[int] = [8, 16, 32],
        level_codebook_sizes: List[int] = [64, 128, 256],
        level_update_freqs: List[int] = [8, 4, 1],
    ):
        ...
```

**Purpose:** Multi-scale split-latent architecture where each level operates at a different timescale. Inspired by Clockwork VAE and Hierarchical World Models.

**Key parameters:**
- `n_levels` – Number of hierarchy levels (default 3)
- `level_dims` – Latent dimension at each level [nat]
- `level_codebook_sizes` – Number of discrete codes per level
- `level_update_freqs` – Update frequency (e.g., [8, 4, 1] means level 1 updates every 8 steps)

**Hierarchy structure:**
- **Level 1 (Slowest):** Global game state, long-term goals
- **Level 2 (Medium):** Object positions, velocities
- **Level 3 (Fast):** Fine motor control, reactions
- **Micro:** Noise (textures, particles)

**Source:** {ref}`Section 3.2 <sec-advanced-hierarchical-multi-scale-latents>`, Definition {prf:ref}`def-hierarchical-latent`, line 1052.
:::

---

## G.2 Supervised Topology Modules

These modules implement the supervised topology framework from {ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`, ensuring chart purity and class-consistent transitions.

### G.2.1 SupervisedTopologyLoss

:::{prf:definition} G.2.1 (SupervisedTopologyLoss)
:label: def-g-supervised-topology-loss

**Class signature:**
```python
class SupervisedTopologyLoss(nn.Module):
    def __init__(
        self,
        num_charts: int,
        num_classes: int,
        lambda_purity: float = 0.1,
        lambda_balance: float = 0.01,
        lambda_metric: float = 0.01,
        margin: float = 1.0,
        temperature: float = 1.0,
    ):
        ...

    def forward(
        self,
        chart_assignments: torch.Tensor,  # [B, N_c] soft assignments
        class_labels: torch.Tensor,        # [B] ground truth classes
        embeddings: torch.Tensor,          # [B, D] latent embeddings
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ...
```

**Input/Output:**
- Input:
  - `chart_assignments` shape `[B, N_c]` – Soft chart assignments (router weights)
  - `class_labels` shape `[B]` – Ground truth class labels
  - `embeddings` shape `[B, D]` – Latent embeddings
- Output: `(total_loss, loss_dict)` where `loss_dict` contains individual loss terms

**Purpose:** Enforces that each chart is dominated by a single class (purity), charts are used roughly equally (balance), and same-class samples are metrically closer (separation).

**Key parameters:**
- `num_charts` – Number of atlas charts $N_c$
- `num_classes` – Number of semantic classes $C$
- `lambda_purity` – Weight for chart purity loss (Definition {prf:ref}`def-purity-loss`)
- `lambda_balance` – Weight for chart balance loss
- `lambda_metric` – Weight for metric contrastive loss

**Learnable parameters:**
- `chart_to_class` shape `[N_c, C]` – Logits mapping charts to class probabilities

**Loss components:**
1. **Chart Purity:** $\mathcal{L}_{\text{purity}} = -\sum_k \max_c p(c|k) \log \max_c p(c|k)$
2. **Chart Balance:** $\mathcal{L}_{\text{balance}} = D_{\text{KL}}(\bar{p}(k) \| \text{Uniform})$
3. **Metric Contrastive:** Encourages intra-class proximity, inter-class separation

**Source:** {ref}`Section 25.4 <sec-the-supervised-topology-loss>`, Definition {prf:ref}`def-total-loss`, line 680.
:::

### G.2.2 class_modulated_jump_rate

:::{prf:definition} G.2.2 (class_modulated_jump_rate)
:label: def-g-class-modulated-jump-rate

**Function signature:**
```python
def class_modulated_jump_rate(
    lambda_base: torch.Tensor,     # [N_c, N_c] base jump rates
    chart_to_class: torch.Tensor,  # [N_c, C] learnable logits
    gamma_sep: float = 5.0,        # Separation strength
) -> torch.Tensor:
    ...
```

**Input/Output:**
- Input:
  - `lambda_base` shape `[N_c, N_c]` – Base jump rate matrix
  - `chart_to_class` shape `[N_c, C]` – Chart-to-class mapping logits
  - `gamma_sep` – Separation strength coefficient
- Output: `lambda_sup` shape `[N_c, N_c]` – Class-modulated jump rates

**Purpose:** Computes class-consistent jump rates that suppress transitions between charts of different dominant classes, implementing the class-modulated rate from Definition {prf:ref}`def-class-consistent-jump-rate`.

**Mathematical operation:**
$$\lambda_{kk'}^{\text{sup}} = \lambda_{kk'}^{\text{base}} \cdot \exp(-\gamma_{\text{sep}} \cdot D_{\text{class}}(k, k'))$$

where $D_{\text{class}}(k, k') = 1$ if charts $k$ and $k'$ have different dominant classes, else $0$.

**Key parameters:**
- `gamma_sep` – Controls how strongly cross-class jumps are suppressed (higher = stronger suppression)

**Source:** {ref}`Section 25.3 <sec-metric-segmentation-via-jump-rate-modulation>`, Definition {prf:ref}`def-class-consistent-jump-rate`, line 445.
:::

---

## G.3 Lorentzian Memory Attention Modules

These modules implement the causal memory attention from {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, enforcing light-cone causality in memory retrieval.

### G.3.1 LorentzianConfig

:::{prf:definition} G.3.1 (LorentzianConfig)
:label: def-g-lorentzian-config

**Class signature:**
```python
@dataclass
class LorentzianConfig:
    d_model: int = 256        # Model dimension [nat]
    d_latent: int = 64        # Latent space dimension
    n_heads: int = 4          # Number of attention heads
    c_info: float = 1.0       # Information speed (latent units per timestep)
    T_c: float = 0.1          # Cognitive temperature [nat/step]
    gamma_friction: float = 1.0  # Friction coefficient for O-step
    dt: float = 0.01          # Integration timestep
```

**Purpose:** Configuration for Lorentzian memory attention with causal structure.

**Key parameters:**
- `c_info` – Information speed $c_{\text{info}}$ defining the light cone (Definition {prf:ref}`def-information-speed-recap`)
- `d_latent` – Dimension of the latent manifold $\mathcal{Z}$

**Units:** `d_model` and `d_latent` in [nat], `c_info` in [latent units/timestep], `T_c` in [nat/step].

**Source:** {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, line 864.
:::

### G.3.2 LorentzianMetric

:::{prf:definition} G.3.2 (LorentzianMetric)
:label: def-g-lorentzian-metric

**Class signature:**
```python
class LorentzianMetric(nn.Module):
    def __init__(self, config: LorentzianConfig, epsilon: float = 1e-6):
        ...

    def conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def geodesic_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        ...

    def spacetime_interval(self, z: torch.Tensor, t: torch.Tensor,
                           z_mem: torch.Tensor, t_mem: torch.Tensor) -> torch.Tensor:
        ...

    def temperature(self, z: torch.Tensor, d_k: int) -> torch.Tensor:
        ...
```

**Input/Output:**
- `conformal_factor`: Input `z` shape `[B, d]` → Output `[B, 1]`
- `geodesic_distance`: Input `z1` shape `[B, d]`, `z2` shape `[B, N, d]` → Output `[B, N]`
- `spacetime_interval`: Input positions and times → Output `[B, N]` intervals
- `temperature`: Input `z` shape `[B, d]` → Output `[B, 1]`

**Purpose:** Implements the Lorentzian metric on the memory manifold $\mathcal{M} = \mathbb{R} \times \mathcal{Z}$ with signature $(-,+,\ldots,+)$.

**Key methods:**
- `conformal_factor`: $\lambda(z) = 2/(1-|z|^2)$ (Poincaré disk)
- `geodesic_distance`: $d_G(z, z') = \operatorname{arcosh}(1 + 2|z-z'|^2/((1-|z|^2)(1-|z'|^2)))$
- `spacetime_interval`: $\Delta s^2_{\text{eff}} = -c_{\text{info}}^2(t-t')^2 + d_G^2$ (Definition {prf:ref}`def-spacetime-interval`)
- `temperature`: $\tau(z) = \sqrt{d_k}/\lambda(z)$ (Theorem {prf:ref}`thm-metric-temperature-correspondence`)

**Source:** {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, Definition {prf:ref}`def-lorentzian-memory-manifold`, line 886.
:::

### G.3.3 CausalMask

:::{prf:definition} G.3.3 (CausalMask)
:label: def-g-causal-mask

**Class signature:**
```python
class CausalMask(nn.Module):
    def __init__(self, config: LorentzianConfig):
        ...

    def forward(
        self,
        z: torch.Tensor,       # [B, d] query position
        t: torch.Tensor,       # [B, 1] query time
        z_mem: torch.Tensor,   # [B, N, d] memory positions
        t_mem: torch.Tensor,   # [B, N, 1] memory times
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Query spacetime position $(z, t)$ and memory positions $(z_{\text{mem}}, t_{\text{mem}})$
- Output: `mask` shape `[B, N]` – Binary mask (1 = causal, 0 = acausal)

**Purpose:** Computes the causal mask from the light cone structure, enforcing that attention is zero outside the causal past $J^-(z, t)$.

**Mathematical operation:**
$$M_{\text{causal}}(z, t; z', t') = \mathbf{1}\left[ t' < t \text{ and } d_G(z, z') \leq c_{\text{info}}(t - t') \right]$$

**Key insight:** This is spacetime causality, not just temporal ordering. Events must be both in the past *and* within the light cone defined by the information speed $c_{\text{info}}$.

**Source:** {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, Definition {prf:ref}`def-causal-past-light-cone`, line 978.
:::

### G.3.4 TemporalChristoffelQuery

:::{prf:definition} G.3.4 (TemporalChristoffelQuery)
:label: def-g-temporal-christoffel-query

**Class signature:**
```python
class TemporalChristoffelQuery(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_latent: int):
        ...

    def forward(
        self,
        x: torch.Tensor,        # [B, d_in]
        z: torch.Tensor,        # [B, d_latent]
        t: torch.Tensor,        # [B, 1]
        v_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Features `x`, position `z`, time `t`, optional velocity features
- Output: `Q` shape `[B, d_out]` – Geodesic Query vector

**Purpose:** Extends the geodesic Query projection to include temporal Christoffel terms for the Lorentzian metric.

**Mathematical operation:**
$$Q_{\text{geo}}(x, z, t, v) = W_Q x + W_{Qz} z + W_{Qt} t + W_{Qv} v + W_{Q,\Gamma}(z, z) + W_{Q,t}(t, t) + W_{Q,zt}(z, t)$$

**Christoffel structure:** For the Lorentzian metric $g_{\mu\nu} = \text{diag}(-c^2\lambda^2, \lambda^2 I_d)$:
- Spatial: $\Gamma^k_{ij} = \frac{2}{1-|z|^2}(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k)$
- Time-time-space: $\Gamma^0_{0j} = \frac{2z_j}{1-|z|^2}$
- Space-time-time: $\Gamma^k_{00} = \frac{2c^2 z_k}{1-|z|^2}$

**Source:** {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, Definition {prf:ref}`def-temporal-christoffel-encoding`, line 1021.
:::

### G.3.5 LorentzianMemoryAttention

:::{prf:definition} G.3.5 (LorentzianMemoryAttention)
:label: def-g-lorentzian-memory-attention

**Class signature:**
```python
class LorentzianMemoryAttention(nn.Module):
    def __init__(self, config: LorentzianConfig):
        ...

    def forward(
        self,
        x: torch.Tensor,         # [B, d_model] current state features
        z: torch.Tensor,         # [B, d_latent] current position
        t: torch.Tensor,         # [B, 1] current time
        x_mem: torch.Tensor,     # [B, N, d_model] memory features
        z_mem: torch.Tensor,     # [B, N, d_latent] memory positions
        t_mem: torch.Tensor,     # [B, N, 1] memory times
        v_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

**Input/Output:**
- Input: Current state $(x, z, t)$ and memory bank $(x_{\text{mem}}, z_{\text{mem}}, t_{\text{mem}})$
- Output: `(output, weights)` where:
  - `output` shape `[B, d_model]` – Attended memory representation
  - `weights` shape `[B, N]` – Attention weights (for diagnostics)

**Purpose:** Full Lorentzian memory attention combining covariant self-attention with causal mask. Implements Definition {prf:ref}`def-covariant-self-attention-causal` and Definition {prf:ref}`def-lorentzian-cross-attention`.

**Components:**
- `metric` – LorentzianMetric for conformal factor and geodesic distance
- `causal_mask` – CausalMask for light cone enforcement
- `query` – TemporalChristoffelQuery for geodesic Query
- `wilson_scale` – Learnable Wilson line approximation scale

**Key properties:**
1. **Causality:** Attention weight is zero outside $J^-(z, t)$
2. **Gauge covariance:** Wilson line preprocessing ensures gauge invariance
3. **Metric-encoded temperature:** $\tau(z) = \sqrt{d_k}/\lambda(z)$

**Diagnostic nodes:** Monitor with Nodes 71-73 (CausalMaskCheck, RetardedPotentialCheck, LorentzianSignatureCheck).

**Source:** {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, line 1095.
:::

---

## G.4 Gauge-Covariant Attention Modules

These modules implement the gauge-covariant world model from {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, enforcing $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ symmetry.

### G.4.1 GeodesicConfig

:::{prf:definition} G.4.1 (GeodesicConfig)
:label: def-g-geodesic-config

**Class signature:**
```python
@dataclass
class GeodesicConfig:
    d_model: int = 256         # Model dimension [nat]
    d_latent: int = 64         # Latent space dimension
    n_heads: int = 1           # Number of attention heads per BAOAB step
    T_c: float = 0.1           # Cognitive temperature [nat/step]
    gamma_friction: float = 1.0  # Friction coefficient for O-step
    dt: float = 0.01           # Integration timestep
    g_s: float = 1.0           # Binding coupling strength
    g_2: float = 0.5           # Error field coupling
    g_1: float = 0.3           # Opportunity field coupling
    use_learned_thermostat: bool = False  # Enable learned thermostat residual
    thermostat_residual_scale: float = 0.1  # Scale for learned residual
```

**Purpose:** Configuration for the gauge-covariant geodesic cross-attention world model.

**Key parameters:**
- `g_s` – $SU(N_f)_C$ binding coupling (confinement)
- `g_2` – $SU(2)_L$ error field coupling (chirality)
- `g_1` – $U(1)_Y$ opportunity field coupling (hypercharge)
- `use_learned_thermostat` – If True, adds a learnable thermostat head; otherwise uses closed-form OU

**Units:** Couplings $g_s, g_2, g_1$ are dimensionless.

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, line 1146.
:::

### G.4.2 WilsonLineApprox

:::{prf:definition} G.4.2 (WilsonLineApprox)
:label: def-g-wilson-line-approx

**Class signature:**
```python
class WilsonLineApprox(nn.Module):
    def __init__(self, config: GeodesicConfig, d_k: int):
        ...

    def forward(
        self,
        z_query: torch.Tensor,  # [B, d_latent]
        z_key: torch.Tensor,    # [B, N, d_latent]
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Query position `z_query` and key positions `z_key`
- Output: `U` shape `[B, N, d_k, d_k]` – Transformation matrices for each key

**Purpose:** Computes the linearized Wilson line $U(z, z') \approx I - i A_\mu(z)(z - z')^\mu$ for parallel transport in attention.

**Learnable parameters:**
- `theta_binding` – $SU(N_f)_C$ connection coefficients
- `theta_error` – $SU(2)_L$ connection coefficients
- `theta_opportunity` – $U(1)_Y$ connection coefficient

**Mathematical operation:**
$$U(z, z') \approx I - i\Theta(z) \cdot (z - z')$$

where $\Theta$ encodes the total gauge connection $A_\mu = g_s G_\mu + g_2 W_\mu + g_1 B_\mu$.

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Proposition {prf:ref}`prop-wilson-line-approximation`, line 1176.
:::

### G.4.3 ConformalMetric

:::{prf:definition} G.4.3 (ConformalMetric)
:label: def-g-conformal-metric

**Class signature:**
```python
class ConformalMetric(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        ...

    def conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def metric_inv(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def temperature(self, z: torch.Tensor, d_k: int) -> torch.Tensor:
        ...
```

**Input/Output:**
- `conformal_factor`: `z` shape `[B, d]` → `[B, 1]`
- `metric`: `z` shape `[B, d]` → `[B, d, d]`
- `metric_inv`: `z` shape `[B, d]` → `[B, d, d]`
- `temperature`: `z` shape `[B, d]` → `[B, 1]`

**Purpose:** Computes the Poincaré disk metric and its derived quantities.

**Key formulas:**
- Conformal factor: $\lambda(z) = 2/(1-|z|^2)$
- Metric: $G_{ij}(z) = \lambda(z)^2 \delta_{ij}$
- Inverse metric: $G^{ij}(z) = \lambda(z)^{-2} \delta^{ij}$
- Temperature: $\tau(z) = \sqrt{d_k}/\lambda(z)$

**Boundary behavior:** As $|z| \to 1$, $\lambda \to \infty$ and $\tau \to 0$, making attention infinitely sharp and preventing boundary crossing.

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Definition {prf:ref}`def-poincare-metric-recap`, line 1248.
:::

### G.4.4 ChristoffelQuery

:::{prf:definition} G.4.4 (ChristoffelQuery)
:label: def-g-christoffel-query

**Class signature:**
```python
class ChristoffelQuery(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_latent: int):
        ...

    def forward(
        self,
        x: torch.Tensor,         # [B, d_in] feature vector
        z_geom: torch.Tensor,    # [B, d_latent] position
        v_feat: Optional[torch.Tensor] = None,  # velocity features
        v_geom: Optional[torch.Tensor] = None,  # velocity
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Features `x`, position `z_geom`, optional velocity features
- Output: `Q` shape `[B, d_out]` – Geodesic Query vector

**Purpose:** Implements the geodesic Query projection encoding Christoffel symbols via linear + quadratic terms.

**Mathematical operation:**
$$Q_{\text{geo}}(x, z, v) = W_Q x + W_{Qz} z + W_{Qv} v_{\text{feat}} + W_{Q,\Gamma}(z, z) + W_{Qzv}(z, v)$$

**Learnable parameters:**
- `W_Q` – Feature projection
- `W_Qz` – Position projection (captures linear part of $\Gamma$)
- `W_Qv` – Velocity feature projection
- `W_Q_gamma` – Quadratic tensor for Christoffel encoding
- `W_Qzv` – Position-velocity coupling

**Initialization:** `W_Q_gamma` is initialized with Poincaré-inspired structure to approximate $\Gamma^k_{ij} \propto (\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k)$.

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Definition {prf:ref}`def-geodesic-query-projection`, line 1311.
:::

### G.4.5 ChiralProjector

:::{prf:definition} G.4.5 (ChiralProjector)
:label: def-g-chiral-projector

**Class signature:**
```python
class ChiralProjector(nn.Module):
    def __init__(self, d_latent: int):
        ...

    def forward(
        self,
        psi_doublet: torch.Tensor,  # [B, 2, d] observation-action doublet
        grad_V: torch.Tensor,        # [B, d_latent] value gradient
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Doublet `psi_doublet` shape `[B, 2, d]` and value gradient `grad_V`
- Output: Gated projected doublet shape `[B, 2*d]`

**Purpose:** Implements the $SU(2)_L$ chiral projector that extracts committed actions from the observation-action doublet using the value gradient direction.

**Mathematical operation:**
$$\hat{n}(z) = \frac{P \nabla V}{\|P \nabla V\|}, \quad \Pi_{\text{chirality}} = \frac{1}{2}(I_2 + \hat{n} \cdot \vec{\tau})$$

where $\vec{\tau} = (\tau_1, \tau_2, \tau_3)$ are Pauli matrices.

**Key insight:** The projection extracts the component of the doublet aligned with the value gradient—the direction of improvement. When $\nabla V \approx 0$ (flat landscape), the projector is degenerate, encoding decision ambiguity.

**Gauge covariance:** The commitment strength $c(z) = \Psi_L^\dagger \Pi \Psi_L$ is $SU(2)$-invariant (Theorem {prf:ref}`thm-gauge-covariance-chiral-projection`).

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Definition {prf:ref}`def-chiral-projector-value-gradient`, line 1384.
:::

### G.4.6 AreaLawScreening

:::{prf:definition} G.4.6 (AreaLawScreening)
:label: def-g-area-law-screening

**Class signature:**
```python
class AreaLawScreening(nn.Module):
    def __init__(self, config: GeodesicConfig):
        ...

    def string_area(
        self,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        lambda_z: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def forward(
        self,
        attention: torch.Tensor,  # [B, N] attention scores
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        lambda_z: torch.Tensor,
        level: int = 0,
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Attention weights, positions, conformal factor, hierarchy level
- Output: Screened attention shape `[B, N]`

**Purpose:** Implements $SU(N_f)_C$ area law screening for texture confinement. Suppresses attention between positions at different representation levels.

**Mathematical operation:**
$$\alpha_{\text{screened}} = \alpha \cdot \exp(-\sigma(\ell) \cdot A_{\text{string}})$$

where:
- $\sigma(\ell) = \sigma_0 \cdot e^{-\ell/L}$ is the level-dependent string tension
- $A_{\text{string}} \approx \frac{\lambda^2}{2}|z - z'|^2$ is the minimal string area

**Asymptotic freedom:** At texture level ($\ell = L$), $\sigma \to 0$ and features interact freely. At macro level ($\ell = 0$), $\sigma$ is large and texture is confined.

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Definition {prf:ref}`def-area-law-screening-attention`, Theorem {prf:ref}`thm-texture-confinement-area-law`, line 1438.
:::

### G.4.7 CovariantAttention

:::{prf:definition} G.4.7 (CovariantAttention)
:label: def-g-covariant-attention

**Class signature:**
```python
class CovariantAttention(nn.Module):
    def __init__(
        self,
        config: GeodesicConfig,
        use_chirality: bool = False,
        use_screening: bool = False,
        head_type: str = 'generic',  # 'B', 'A', 'O', or 'generic'
    ):
        ...

    def forward(
        self,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        x_query: torch.Tensor,
        x_key: torch.Tensor,
        x_value: torch.Tensor,
        v_query: Optional[torch.Tensor] = None,
        v_query_geom: Optional[torch.Tensor] = None,
        grad_V: Optional[torch.Tensor] = None,
        level: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

**Input/Output:**
- Input: Query/key positions and features, optional velocity and value gradient
- Output: `(output, attention)` where:
  - `output` shape `[B, d_model]` – Attention output
  - `attention` shape `[B, N]` – Attention weights

**Purpose:** Single covariant attention head combining all gauge structures: Wilson lines, position-dependent temperature, Christoffel Query, chiral projection, and area law screening.

**Components:**
- `query` – ChristoffelQuery
- `wilson` – WilsonLineApprox
- `metric` – ConformalMetric
- `chiral` – ChiralProjector (optional)
- `screening` – AreaLawScreening (optional)

**Attention computation:**
1. Compute $Q$ with geodesic Query projection
2. Compute $K$ and apply Wilson line: $K_{\text{transported}} = U \cdot K$
3. Score: $s = Q^T K_{\text{transported}} / \tau(z)$
4. Softmax and optional screening
5. Weighted sum of $V$, optional chiral projection

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, line 1505.
:::

### G.4.8 GeodesicCrossAttention

:::{prf:definition} G.4.8 (GeodesicCrossAttention)
:label: def-g-geodesic-cross-attention

**Class signature:**
```python
class GeodesicCrossAttention(nn.Module):
    def __init__(self, config: GeodesicConfig):
        ...

    def forward(
        self,
        z: torch.Tensor,             # [B, d_latent] current position
        p: torch.Tensor,             # [B, d_latent] current momentum
        context_z: torch.Tensor,     # [B, N, d_latent] context positions
        context_x: torch.Tensor,     # [B, N, d_model] context features
        context_force: torch.Tensor, # [B, N, d_latent] force/gradient bank
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

**Input/Output:**
- Input: Current phase space state $(z, p)$ and context banks
- Output: `(z_next, p_next)` – Updated position and momentum

**Purpose:** Full geodesic world model implementing Boris-BAOAB integration via four attention heads (B-A-A-B) plus a closed-form OU thermostat (or optional learned thermostat head).

**BAOAB Steps:**
1. **Head 1 (B-step):** First half-kick from force bank
2. **Head 2 (A-step):** First half-drift + attention correction
3. **OU step:** Ornstein-Uhlenbeck thermostat (closed-form, or learned residual)
4. **Head 4 (A-step):** Second half-drift + attention correction
5. **Head 5 (B-step):** Second half-kick from force bank

**OU coefficients:**
$$c_1 = e^{-\gamma h}, \quad c_2 = \sqrt{(1-c_1^2)T_c}$$

**Boltzmann preservation:** Preserves $\rho(z, p) \propto \exp(-\Phi_{\text{eff}}/T_c - \|p\|_G^2/(2T_c))$ to $O(h^2)$ (Theorem {prf:ref}`thm-baoab-attention-boltzmann`).

**Diagnostic nodes:** Monitor with Nodes 67-70 (gauge, temperature, chirality, confinement).

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Definition {prf:ref}`def-baoab-attention-heads`, line 1616.
:::

---

## G.5 Summary Table

| Module | Section | Domain | Key Dependencies | Purpose |
|:-------|:--------|:-------|:-----------------|:--------|
| `DisentangledConfig` | 3.2 | VAE | — | Configuration for split-latent agent |
| `Encoder` | 3.2 | VAE | — | CNN backbone for observations |
| `VectorQuantizer` | 3.2 | VAE | — | Discrete macro symbol via straight-through |
| `Decoder` | 3.2 | VAE | — | Reconstruction from $(K, Z_n, Z_{\text{tex}})$ |
| `MacroDynamicsModel` | 3.2 | VAE | — | Micro-blind world model |
| `DisentangledAgent` | 3.2 | VAE | Encoder, VQ, Decoder, MacroDynamics | Full split-latent VQ-VAE |
| `HierarchicalDisentangled` | 3.2 | VAE | DisentangledAgent | Multi-scale temporal hierarchy |
| `SupervisedTopologyLoss` | 25 | Topology | — | Chart purity, balance, separation losses |
| `class_modulated_jump_rate` | 25 | Topology | — | Class-consistent transition rates |
| `LorentzianConfig` | 33 | Memory | — | Configuration for causal memory |
| `LorentzianMetric` | 33 | Memory | — | Lorentzian spacetime metric |
| `CausalMask` | 33 | Memory | LorentzianMetric | Light cone causality |
| `TemporalChristoffelQuery` | 33 | Memory | — | Temporal geodesic Query |
| `LorentzianMemoryAttention` | 33 | Memory | All above | Full causal memory attention |
| `GeodesicConfig` | 35 | Gauge | — | Configuration for covariant attention |
| `WilsonLineApprox` | 35 | Gauge | — | Linearized parallel transport |
| `ConformalMetric` | 35 | Gauge | — | Poincaré disk metric |
| `ChristoffelQuery` | 35 | Gauge | — | Geodesic Query with Christoffel |
| `ChiralProjector` | 35 | Gauge | — | $SU(2)_L$ chiral projection |
| `AreaLawScreening` | 35 | Gauge | ConformalMetric | $SU(N_f)_C$ texture firewall |
| `CovariantAttention` | 35 | Gauge | Wilson, Metric, Query, Chiral, Screening | Single gauge-covariant head |
| `GeodesicCrossAttention` | 35 | Gauge | CovariantAttention | Full BAOAB integrator |

---

## G.6 Implementation Dependencies

The architecture modules have the following dependency structure:

```
DisentangledAgent
├── Encoder
├── VectorQuantizer
├── Decoder
└── MacroDynamicsModel

GeodesicCrossAttention
├── CovariantAttention (×4-5 heads)
│   ├── ChristoffelQuery
│   ├── WilsonLineApprox
│   ├── ConformalMetric
│   ├── ChiralProjector (optional)
│   └── AreaLawScreening (optional)
└── ConformalMetric

LorentzianMemoryAttention
├── LorentzianMetric
├── CausalMask
│   └── LorentzianMetric
└── TemporalChristoffelQuery

SupervisedTopologyLoss
└── class_modulated_jump_rate
```

**Cross-references:**
- Loss functions: {ref}`Appendix F <sec-appendix-f-loss-terms-reference>` (`06_losses.md`)
- Sieve diagnostic nodes: {ref}`Section 7 <sec-diagnostics-stability-checks>`
- Full gauge theory derivation: {ref}`Section 34 <sec-standard-model-cognition>`
