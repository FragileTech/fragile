(sec-appendix-g-architecture-modules-reference)=
# Appendix G: Architecture Modules Reference

## TLDR

- Centralizes **~45 neural network modules** defined throughout Volume 1
- Organized by domain:
  - **Disentangled VAE** ({ref}`Part III <sec-architecture-the-disentangled-vq-vae-rnn>`)
  - **Supervised Topology** ({ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`)
  - **Lorentzian Memory Attention** ({ref}`Part VII <sec-covariant-memory-attention-architecture>`)
  - **Gauge-Covariant Attention** ({ref}`Section 05 <sec-covariant-cross-attention-architecture>`)
  - **Gauge-Covariant Primitives** ({ref}`Section 04 <sec-dnn-blocks>`)
  - **Universal Geometric Network** ({ref}`Section 06 <sec-universal-geometric-network>`)
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

These modules implement the gauge-covariant world model from {ref}`Section 05 <sec-covariant-cross-attention-architecture>`, enforcing $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ symmetry.

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
$$\hat{n}(z) = \frac{P \nabla_A V}{\|P \nabla_A V\|}, \quad \Pi_{\text{chirality}} = \frac{1}{2}(I_2 + \hat{n} \cdot \vec{\tau})$$

where $\vec{\tau} = (\tau_1, \tau_2, \tau_3)$ are Pauli matrices.

**Key insight:** The projection extracts the component of the doublet aligned with the value gradient—the direction of improvement. When $\nabla_A V \approx 0$ (flat landscape), the projector is degenerate, encoding decision ambiguity.

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

## G.5 Gauge-Covariant Primitives (Section 04)

These modules implement the fundamental gauge-covariant building blocks from {ref}`Section 04 <sec-dnn-blocks>`, ensuring spectral normalization, rotational equivariance, and light cone preservation.

### G.5.1 SpectralLinear

:::{prf:definition} G.5.1 (SpectralLinear)
:label: def-g-spectral-linear

**Class signature:**
```python
class SpectralLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        ...
```

**Input/Output:**
- Input: `x` shape `[B, in_features]` – Feature vectors
- Output: `y` shape `[B, out_features]` – Transformed features

**Purpose:** Linear layer with spectral normalization $\sigma_{\max}(W) \leq 1$. Ensures capacity bound and light cone preservation for causal structure.

**Key parameters:**
- `in_features` – Input dimension [nat]
- `out_features` – Output dimension [nat]
- `bias` – Typically `False` for gauge invariance (breaks tangent bundle structure)

**Mathematical operation:**
$$y = W_{\text{normalized}} \cdot x \quad \text{where} \quad \sigma_{\max}(W_{\text{normalized}}) \leq 1$$

**Key properties:**
- Contraction: $\|y\| \leq \|x\|$ (no unbounded amplification)
- Light cone preservation: $d(Wz_1, Wz_2) \leq c_{\text{info}} \Delta t$ whenever inputs are causally connected
- No bias term (gauge invariance requirement)

**Diagnostic node:** Node 62 (CausalityViolationCheck) verifies $\sigma_{\max}(W) \leq 1 + \epsilon$ during training.

**Source:** {ref}`Section 04 <sec-dnn-blocks>`, Definition {prf:ref}`def-spectral-linear`, line 569.
:::

### G.5.2 NormGatedActivation

:::{prf:definition} G.5.2 (NormGatedActivation)
:label: def-g-norm-gated-activation

**Function signature:**
```python
def norm_gated_activation(v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Args:
        v: Bundle vectors [B, n_bundles, bundle_dim]
        b: Bias scalars [n_bundles]
    Returns:
        Gated vectors [B, n_bundles, bundle_dim]
    """
    norms = torch.norm(v, dim=-1, keepdim=True)  # [B, n_bundles, 1]
    gates = F.gelu(norms.squeeze(-1) + b)        # [B, n_bundles]
    return v * gates.unsqueeze(-1) / (norms + 1e-8)
```

**Input/Output:**
- Input: `v` shape `[B, n_bundles, d_b]` – Bundle vectors
- Output: Gated vectors shape `[B, n_bundles, d_b]` – Energy-filtered output

**Purpose:** $SO(d_b)$-equivariant activation using radial symmetry. Gates signal based on energy $\|v\|$ exceeding threshold $-b$.

**Mathematical operation:**
$$f(v_i) = v_i \cdot g(\|v_i\| + b_i)$$

where:
- $\|v_i\| = \sqrt{v_i^T v_i}$ is the Euclidean norm (rotation-invariant)
- $g: \mathbb{R} \to \mathbb{R}$ is GELU or another smooth scalar function
- $b_i$ is the learnable activation potential (energy barrier)

**Key properties:**
- **$SO(d_b)$ equivariance:** $f(Rv) = R f(v)$ for all $R \in SO(d_b)$
- **Physical interpretation:** Energy barrier—gate opens when $\|v\| > -b$
- **Direction independence:** Gate decision depends only on magnitude, not orientation

**GELU rationale:**
- $C^\infty$ smoothness (compatible with WFR metric)
- Linear growth at large arguments: $g(x) \approx x$ for $x \gg 1$
- Controlled Lipschitz constant $L_g \approx 1.129$
- Empirically effective (validated in transformers)

**Alternative activations:** Softplus ($C^\infty$, always positive), Sigmoid/Tanh (saturate, reduced dynamic range).

**Source:** {ref}`Section 04 <sec-dnn-blocks>`, Definition {prf:ref}`def-norm-gated-activation`, line 714.
:::

### G.5.3 IsotropicBlock

:::{prf:definition} G.5.3 (IsotropicBlock)
:label: def-g-isotropic-block

**Class signature:**
```python
class IsotropicBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bundle_size: int = 16,
        exact: bool = False
    ):
        ...
```

**Input/Output:**
- Input: `z` shape `[B, in_dim]` – Input features
- Output: `z_out` shape `[B, out_dim]` – Transformed features

**Purpose:** Atomic gauge-covariant building block combining SpectralLinear, Reshape, and NormGate in sequence.

**Architecture:**
$$\text{IsotropicBlock}(z) = \text{NormGate}(\text{Reshape}(\text{SpectralLinear}(z)))$$

**Key parameters:**
- `in_dim` – Input dimension [nat]
- `out_dim` – Output dimension (must be divisible by `bundle_size`) [nat]
- `bundle_size` – Dimension of each bundle $d_b$ [nat]
- `exact` – If `True`, uses scalar blocks $W_i = \lambda_i I_{d_b}$ for exact equivariance; if `False` (default), uses block-diagonal for approximate equivariance

**Equivariance modes:**
- **Exact mode** (`exact=True`): Strictly $\prod_{i=1}^{n_b} SO(d_b)$ equivariant via scalar blocks
  - Weight matrix: $W = \text{diag}(\lambda_1 I, \ldots, \lambda_{n_b} I)$
  - Limited expressiveness (can only scale bundles)
  - Zero equivariance violation

- **Approximate mode** (`exact=False`): Bounded equivariance violation, greater expressiveness
  - Weight matrix: Block-diagonal with general $d_b \times d_b$ blocks
  - Each block spectrally normalized: $\sigma_{\max}(W_i) \leq 1$
  - Can learn within-bundle transformations

**Mathematical constraint (exact mode):**
By Schur's lemma, any linear map commuting with all $g \in SO(d_b)$ must be a scalar multiple of identity:
$$W_i \cdot g_i = g_i \cdot W_i \quad \forall g_i \in SO(d_b) \quad \Rightarrow \quad W_i = \lambda_i I_{d_b}$$

**Diagnostic nodes:** Node 67 (GaugeInvarianceCheck), Node 62 (CausalityViolationCheck), Node 40 (PurityCheck).

**Source:** {ref}`Section 04 <sec-dnn-blocks>`, Definition {prf:ref}`def-isotropic-block`, line 803.
:::

### G.5.4 GaugeInvarianceCheck

:::{prf:definition} G.5.4 (GaugeInvarianceCheck)
:label: def-g-gauge-invariance-check

**Class signature:**
```python
class GaugeInvarianceCheck(DiagnosticNode):
    def __init__(self, layer: nn.Module, group: str = "SO(d)"):
        ...

    def check(self, z: torch.Tensor) -> Dict[str, float]:
        ...
```

**Input/Output:**
- Input: `z` shape `[B, d]` – Latent state
- Output: Dictionary with `gauge_violation`, `threshold`, `passed` keys

**Purpose:** Diagnostic node (Node 67) that verifies $G$-equivariance by sampling random group transformations and measuring violation.

**Mathematical test:**
$$\delta_{\text{gauge}} = \|f(g \cdot z) - g \cdot f(z)\| < \epsilon_{\text{gauge}}$$

where $g$ is a randomly sampled group element (e.g., rotation matrix for $SO(d)$).

**Key parameters:**
- `layer` – The module to test
- `group` – Symmetry group ("SO(d)" for rotations)
- Threshold: $\epsilon_{\text{gauge}} = 10^{-4}$ (exact equivariance) or $\epsilon_{\text{gauge}} \approx 0.1$ (soft equivariance)

**Failure modes:**
- Large violation ($\delta > 0.1$): Symmetry breaking without L1 regularization
- Asymmetric violation: Equivariant under some $g$ but not others (indicates partial symmetry)

**Source:** {ref}`Section 04 <sec-dnn-blocks>`, line 2908.
:::

### G.5.5 CovariantRetina

:::{prf:definition} G.5.5 (CovariantRetina)
:label: def-g-covariant-retina

**Class signature:**
```python
class CovariantRetina(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 512,
        num_rotations: int = 8,
        kernel_size: int = 5
    ):
        ...
```

**Input/Output:**
- Input: `x` shape `[B, C, H, W]` – RGB images
- Output: `z` shape `[B, out_dim]` – Latent features

**Purpose:** $SO(2)$-equivariant vision encoder using steerable convolutions (via E2CNN library). Ensures rotation equivariance for visual inputs.

**Architecture:**
1. **Lifting layer:** Maps trivial representation (standard image) to regular representation on $SE(2)$
2. **Steerable convolutions:** 3 layers with expanding channels (32 → 64 → 64)
3. **Group pooling:** Max over rotation group to extract rotation-invariant features
4. **Spatial pooling:** Adaptive average pooling to fixed size
5. **Linear projection:** Spectral-normalized fully connected layer to latent dimension

**Key parameters:**
- `in_channels` – Input channels (3 for RGB)
- `out_dim` – Output latent dimension [nat]
- `num_rotations` – Discretization of $SO(2)$ (typically 8 or 16)
- `kernel_size` – Convolutional kernel size [pixels]

**Equivariance guarantee:**
$$\text{Conv}(R_\theta \cdot I) = D^{(\ell)}(\theta) \cdot \text{Conv}(I)$$

where $R_\theta$ is a rotation by angle $\theta$ and $D^{(\ell)}$ is the representation matrix.

**Diagnostic node:** Node 68 (RotationEquivarianceCheck) verifies $\|f(R \cdot I) - R \cdot f(I)\| < \epsilon$ for random rotations.

**Source:** {ref}`Section 04 <sec-dnn-blocks>`, line 1464.
:::

---

## G.6 Universal Geometric Network (Section 06)

These modules implement the Universal Geometric Network from {ref}`Section 06 <sec-universal-geometric-network>`, achieving universal approximation while maintaining geometric consistency through soft equivariance.

### G.6.1 UGNConfig / BundleConfig

:::{prf:definition} G.6.1 (UGNConfig / BundleConfig)
:label: def-g-ugn-config

**Class signatures:**
```python
@dataclass
class BundleConfig:
    name: str              # Semantic label (e.g., "charge", "lepton")
    dim: int               # Bundle dimension d_b [dimensionless]
    semantic_role: str = ""  # Physical interpretation

@dataclass
class UGNConfig:
    input_dim: int         # Input dimension [dimensionless]
    output_dim: int        # Output dimension [dimensionless]
    bundles: List[BundleConfig]  # Bundle specifications
    n_latent_layers: int = 4     # Number of soft equivariant layers
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    lambda_l1: float = 0.01      # L1 regularization strength
    lambda_equiv: float = 0.0    # Equivariance penalty
    use_spectral_norm: bool = True
```

**Purpose:** Configuration dataclasses for the three-stage Universal Geometric Network architecture.

**Key properties:**
- `n_bundles` – Number of gauge bundles (computed from `bundles` list)
- `total_latent_dim` – $\sum_{i=1}^{n_b} d_i$
- `bundle_dims` – List of bundle dimensions $[d_1, \ldots, d_{n_b}]$

**Typical bundle structure:**
```python
bundles = [
    BundleConfig(name="color", dim=64, semantic_role="Binding/texture confinement"),
    BundleConfig(name="isospin", dim=8, semantic_role="Error field/chirality"),
    BundleConfig(name="hypercharge", dim=4, semantic_role="Opportunity field/capacity"),
]
```

**Units:** All dimensions [nat] or [dimensionless], loss weights [dimensionless].

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, lines 1180, 1873.
:::

### G.6.2 SoftEquivariantLayer

:::{prf:definition} G.6.2 (SoftEquivariantLayer)
:label: def-g-soft-equivariant-layer

**Class signature:**
```python
class SoftEquivariantLayer(nn.Module):
    def __init__(
        self,
        bundle_dims: List[int],
        hidden_dim: int = 64,
        use_spectral_norm: bool = True
    ):
        ...
```

**Input/Output:**
- Input: `z` shape `[B, sum(bundle_dims)]` – Latent state
- Output: `z_out` shape `[B, sum(bundle_dims)]` – Updated latent state

**Purpose:** Core latent dynamics layer combining equivariant and mixing pathways with L1 regularization for emergent structure discovery.

**Architecture:**
$$z_{\text{out}} = z + f_{\text{equiv}}(z) + g \cdot f_{\text{mix}}(z)$$

where:
- **Equivariant pathway:** $f_{\text{equiv}}(z) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)$
  - Uses only bundle norms → strictly $\prod_i SO(d_i)$ equivariant
  - Implemented via norm MLP: $\mathbb{R}^{n_b} \to \mathbb{R}^{n_b}$

- **Mixing pathway:** $f_{\text{mix}}(z) = \sum_{i,j} W_{ij} v_j$
  - Cross-bundle interactions with learnable weights
  - L1 penalized: $\mathcal{L}_{\text{L1}} = \sum_{i,j} \|W_{ij}\|_1$
  - Encouraged to be sparse (emergent texture zeros)

**Key parameters:**
- `bundle_dims` – List $[d_1, \ldots, d_{n_b}]$ of bundle dimensions
- `hidden_dim` – Hidden dimension for norm MLP
- `use_spectral_norm` – Apply spectral normalization to all linear layers

**Learnable parameters:**
- Norm MLP weights: $O(n_b \cdot h + h^2)$ parameters
- Mixing weights $W_{ij}$: $O(n_b^2 d_{\max}^2)$ parameters (largest memory consumer)
- Gate biases: $n_b$ scalars

**L1 loss:**
```python
def l1_loss(self) -> torch.Tensor:
    return sum(
        torch.sum(torch.abs(self.mixing_weights[i][j]))
        for i in range(n_b) for j in range(n_b)
    )
```

**Diagnostic methods:**
- `mixing_strength()` – Total Frobenius norm of mixing weights (measures symmetry breaking)

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, lines 1219 (simplified), 1970 (production).
:::

### G.6.3 UniversalGeometricNetwork

:::{prf:definition} G.6.3 (UniversalGeometricNetwork)
:label: def-g-universal-geometric-network

**Class signature:**
```python
class UniversalGeometricNetwork(nn.Module):
    def __init__(self, config: UGNConfig):
        ...
```

**Input/Output:**
- Input: `x` shape `[B, input_dim]` – Raw observations
- Output: `y` shape `[B, output_dim]` – Predictions/actions

**Purpose:** Three-stage architecture achieving both universal approximation and geometric consistency.

**Architecture:**
1. **Encoder** (unconstrained, universal):
   - $E: \mathbb{R}^{d_{\text{in}}} \to \bigoplus_i V_i$
   - 2-3 spectral-normalized linear layers with GELU
   - **Chooses gauge** for latent representation

2. **Latent Dynamics** (soft equivariant):
   - $D_1, \ldots, D_L: \bigoplus_i V_i \to \bigoplus_i V_i$
   - Stack of `SoftEquivariantLayer` modules
   - **Respects bundle structure** via equivariant pathway + L1-regularized mixing

3. **Decoder** (unconstrained, universal):
   - $P: \bigoplus_i V_i \to \mathbb{R}^{d_{\text{out}}}$
   - 2-3 spectral-normalized linear layers with GELU
   - **Interprets gauge** to extract observables

**Key methods:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    z = self.encode(x)      # Encoder
    z = self.dynamics(z)    # Latent layers
    y = self.decode(z)      # Decoder
    return y

def regularization_loss(self) -> torch.Tensor:
    # L1 penalty on all mixing weights
    return sum(layer.l1_loss() for layer in self.latent_layers)

def equivariance_violation(self, z=None, n_samples=16) -> torch.Tensor:
    # Measure ||D(Rz) - RD(z)||² for random rotations
    ...
```

**Total loss:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{L1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{equiv}} \mathcal{L}_{\text{equiv}}$$

**Key theorems:**
- Universal approximation (encoder/decoder handle arbitrary functions)
- Geometric consistency (latent dynamics respect bundle structure)
- Emergent gauge structure (L1 discovers texture zeros)

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, lines 1335 (simplified), 2154 (production).
:::

### G.6.4 FactoredTensorLayer

:::{prf:definition} G.6.4 (FactoredTensorLayer)
:label: def-g-factored-tensor-layer

**Class signature:**
```python
class FactoredTensorLayer(nn.Module):
    def __init__(
        self,
        d_C: int,
        d_L: int,
        d_Y: int,
        rank: int,
        d_out: int
    ):
        ...
```

**Input/Output:**
- Input: `(z_C, z_L, z_Y)` shapes `[B, d_C]`, `[B, d_L]`, `[B, d_Y]`
- Output: `y` shape `[B, d_out]`

**Purpose:** Low-rank factorization of tensor product interaction for cross-gauge coupling.

**Mathematical operation:**
$$W = \sum_{k=1}^r U_C^{(k)} \otimes U_L^{(k)} \otimes U_Y^{(k)}$$

instead of full tensor $W \in \mathbb{R}^{(d_C d_L d_Y) \times d_{\text{out}}}$.

**Parameter count:**
- Factored: $r(d_C + d_L + d_Y + d_{\text{out}})$
- Full tensor: $(d_C \times d_L \times d_Y) \times d_{\text{out}}$

**Example reduction:**
For $d_C=64, d_L=8, d_Y=4, d_{\text{out}}=64, r=16$:
- Factored: 2,240 parameters
- Full: 131,072 parameters
- **58.5× reduction**

**Use case:** Specific cross-gauge interactions when low-rank structure is empirically justified. Not used in default UGN (uses direct sum instead).

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, line 381.
:::

### G.6.5 NormInteractionLayer

:::{prf:definition} G.6.5 (NormInteractionLayer)
:label: def-g-norm-interaction-layer

**Class signature:**
```python
class NormInteractionLayer(nn.Module):
    def __init__(self, n_bundles: int, hidden_dim: int = 64):
        ...
```

**Input/Output:**
- Input: `z` shape `[B, n_bundles, bundle_dim]` – Bundle representation
- Output: `z_out` shape `[B, n_bundles, bundle_dim]` – Scaled bundles

**Purpose:** Level 1 cross-bundle interaction using only bundle norms (strictly equivariant).

**Mathematical operation:**
$$f_i(v_1, \ldots, v_{n_b}) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)$$

where $\phi: \mathbb{R}^{n_b} \to \mathbb{R}_+$ is an MLP with Softplus output.

**Equivariance:** Strictly $\prod_{i=1}^{n_b} SO(d_b)_i$ equivariant (per-bundle rotations).

**Expressiveness:** Limited—can only scale bundles based on energy, cannot represent direction-dependent interactions.

**Computational cost:** $O(n_b d_b + h^2)$ where $h$ is MLP hidden dimension.

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, line 446.
:::

### G.6.6 GramInteractionLayer

:::{prf:definition} G.6.6 (GramInteractionLayer)
:label: def-g-gram-interaction-layer

**Class signature:**
```python
class GramInteractionLayer(nn.Module):
    def __init__(self, n_bundles: int, hidden_dim: int = 64):
        ...
```

**Input/Output:**
- Input: `z` shape `[B, n_bundles, bundle_dim]` – Bundle representation
- Output: `z_out` shape `[B, n_bundles, bundle_dim]` – Scaled bundles

**Purpose:** Level 2 cross-bundle interaction using Gram matrix $G_{ij} = \langle v_i, v_j \rangle$ (encodes relative orientations).

**Mathematical operation:**
$$G = z \cdot z^T \quad \text{(Gram matrix)}$$
$$\text{scales} = \phi(G_{\text{flat}}) \quad \text{(MLP)}$$
$$z_{\text{out}} = z \cdot \text{scales}$$

**Equivariance:** Equivariant under **global** $SO(d_b)$ (same rotation applied to all bundles), **not** under per-bundle rotations.

**Expressiveness:** High—can encode relative orientations between bundles.

**Computational cost:** $O(n_b^2 d_b + h^2)$.

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, line 494.
:::

### G.6.7 L1Scheduler / AdaptiveL1Scheduler

:::{prf:definition} G.6.7 (L1Scheduler / AdaptiveL1Scheduler)
:label: def-g-l1-scheduler

**Class signature:**
```python
class AdaptiveL1Scheduler:
    def __init__(
        self,
        initial_lambda: float = 0.01,
        target_violation: float = 0.22,
        learning_rate: float = 0.05,
        min_lambda: float = 1e-4,
        max_lambda: float = 1.0
    ):
        ...

    def step(self, current_violation: float) -> float:
        ...
```

**Purpose:** Adaptive scheduler for L1 regularization strength $\lambda_{\text{L1}}$ that targets a specific equivariance violation level.

**Update rule:**
$$\lambda_{\text{L1}}(t+1) = \lambda_{\text{L1}}(t) \cdot \left(1 + \alpha \cdot (\epsilon(t) - \epsilon_{\text{target}})\right)$$

where:
- $\epsilon(t) = \mathcal{L}_{\text{equiv}}(t)$ is current equivariance violation
- $\epsilon_{\text{target}} \approx 0.22$ nat/step (proposed target)
- $\alpha$ is adaptation rate (typically 0.01-0.1)

**Strategy:**
- If $\epsilon(t) > \epsilon_{\text{target}}$: Increase $\lambda_{\text{L1}}$ (more sparsity, less mixing)
- If $\epsilon(t) < \epsilon_{\text{target}}$: Decrease $\lambda_{\text{L1}}$ (more expressiveness, more mixing)

**Key parameters:**
- `initial_lambda` – Starting $\lambda_{\text{L1}}$ value
- `target_violation` – Desired equivariance violation $\epsilon_{\text{target}}$ [nat/step]
- `learning_rate` – Adaptation rate $\alpha$
- `min_lambda` / `max_lambda` – Clamping bounds to prevent collapse or over-sparsity

**Training protocol:**
1. **Warmup (epochs 1-10):** Low $\lambda_{\text{L1}} = 0.001$, let network explore
2. **Ramp up (epochs 10-50):** Gradually increase $\lambda_{\text{L1}}$
3. **Adaptive (epochs 50+):** Use `AdaptiveL1Scheduler` to maintain target violation
4. **Fine-tune:** Fix $\lambda_{\text{L1}}$, early stopping on validation

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, lines 955, 2577.
:::

### G.6.8 CovariantAttentionLayer

:::{prf:definition} G.6.8 (CovariantAttentionLayer)
:label: def-g-covariant-attention-layer

**Class signature:**
```python
class CovariantAttentionLayer(nn.Module):
    def __init__(
        self,
        bundle_dims: List[int],
        n_heads: int = 4,
        use_wilson_lines: bool = True
    ):
        ...
```

**Input/Output:**
- Input: `z` shape `[B, sum(bundle_dims)]`, optional `context` shape `[B, T, sum(bundle_dims)]`
- Output: `z_out` shape `[B, sum(bundle_dims)]`

**Purpose:** Covariant cross-attention for explicit world modeling and trajectory prediction. Alternative to `SoftEquivariantLayer` when planning is required.

**Architecture:**
- Multi-head attention per bundle
- Wilson lines for gauge-covariant Q/K/V projections
- Position-dependent temperature $\tau(z) = \sqrt{d_k}/\lambda(z)$
- Geometric Query terms with Christoffel symbols

**Use cases:**
- **SoftEquivariantLayer:** Default latent dynamics, implicit world model
- **CovariantAttentionLayer:** Explicit trajectory prediction, planning, memory retrieval

**Multi-stage pipeline example:**
1. Encoder → latent $Z$
2. SoftEquivariantLayer (×2) for geometric regularization
3. CovariantAttentionLayer for trajectory rollout
4. SoftEquivariantLayer (×2) for policy extraction
5. Decoder → action $Y$

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, line 2445. See also {ref}`Section 05 <sec-covariant-cross-attention-architecture>` for full derivation.
:::

---

## G.7 Summary Table

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
| `GeodesicConfig` | 05 | Gauge | — | Configuration for covariant attention |
| `WilsonLineApprox` | 05 | Gauge | — | Linearized parallel transport |
| `ConformalMetric` | 05 | Gauge | — | Poincaré disk metric |
| `ChristoffelQuery` | 05 | Gauge | — | Geodesic Query with Christoffel |
| `ChiralProjector` | 05 | Gauge | — | $SU(2)_L$ chiral projection |
| `AreaLawScreening` | 05 | Gauge | ConformalMetric | $SU(N_f)_C$ texture firewall |
| `CovariantAttention` | 05 | Gauge | Wilson, Metric, Query, Chiral, Screening | Single gauge-covariant head |
| `GeodesicCrossAttention` | 05 | Gauge | CovariantAttention | Full BAOAB integrator |
| `SpectralLinear` | 04 | Primitives | — | Spectrally normalized linear layer |
| `NormGatedActivation` | 04 | Primitives | — | $SO(d_b)$-equivariant activation |
| `IsotropicBlock` | 04 | Primitives | SpectralLinear, NormGate | Atomic gauge-covariant block |
| `GaugeInvarianceCheck` | 04 | Primitives | — | Diagnostic for equivariance testing |
| `CovariantRetina` | 04 | Primitives | — | $SO(2)$-equivariant vision encoder |
| `BundleConfig` | 06 | UGN | — | Bundle specification for UGN |
| `UGNConfig` | 06 | UGN | BundleConfig | Configuration for Universal Geometric Network |
| `SoftEquivariantLayer` | 06 | UGN | SpectralLinear | Soft equivariant latent dynamics |
| `UniversalGeometricNetwork` | 06 | UGN | SoftEquivariantLayer | Three-stage universal approximator |
| `FactoredTensorLayer` | 06 | UGN | — | Low-rank tensor product interaction |
| `NormInteractionLayer` | 06 | UGN | — | Level 1 norms-only interaction |
| `GramInteractionLayer` | 06 | UGN | — | Level 2 Gram matrix interaction |
| `AdaptiveL1Scheduler` | 06 | UGN | — | Adaptive L1 regularization schedule |
| `CovariantAttentionLayer` | 06 | UGN | CovariantAttention | Alternative to soft equivariance for planning |

---

## G.8 Implementation Dependencies

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

IsotropicBlock (Section 04)
├── SpectralLinear
├── Reshape (bundle partition)
└── NormGatedActivation

CovariantRetina (Section 04)
├── E2CNN library (steerable convolutions)
├── SpectralLinear (final projection)
└── Group pooling (SO(2) → R²)

UniversalGeometricNetwork (Section 06)
├── Encoder (unconstrained)
│   └── SpectralLinear (×2-3 layers)
├── Latent Dynamics (soft equivariant)
│   └── SoftEquivariantLayer (×L layers)
│       ├── Norm MLP (equivariant pathway)
│       └── Mixing weights W_ij (L1 regularized)
└── Decoder (unconstrained)
    └── SpectralLinear (×2-3 layers)

SoftEquivariantLayer (Section 06)
├── Equivariant pathway
│   └── Norm MLP: R^{n_b} → R^{n_b}
├── Mixing pathway
│   └── W_{ij}: V_j → V_i (L1 penalized)
└── Gate biases (per bundle)

CovariantAttentionLayer (Section 06)
├── Per-bundle attention heads
│   ├── WilsonLineApprox (from Section 05)
│   ├── ConformalMetric (from Section 05)
│   └── ChristoffelQuery (from Section 05)
└── Bundle splitting/concatenation utilities
```

**Cross-references:**
- Loss functions: {ref}`Appendix F <sec-appendix-f-loss-terms-reference>` (`06_losses.md`)
- Sieve diagnostic nodes: {ref}`Section 7 <sec-diagnostics-stability-checks>`
- Full gauge theory derivation: {ref}`Section 34 <sec-standard-model-cognition>`
