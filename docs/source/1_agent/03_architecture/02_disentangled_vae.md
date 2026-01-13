(sec-the-disentangled-variational-architecture-hierarchical-latent-separation)=
# The Disentangled Variational Architecture: Hierarchical Latent Separation

## TLDR

- Enforce a **typed latent split**: discrete macro symbols $K$ (control-relevant), structured nuisance $z_n$
  (pose/disturbance), and reconstruction-only texture $z_{\mathrm{tex}}$.
- Use the split to prevent “texture Trojan horses”: policies and world models must be **blind to texture** while still
  allowing high-fidelity reconstruction.
- Measure success with **closure/grounding diagnostics** (macro predictability, synchronization, mixing windows) rather
  than only reconstruction loss.
- Treat disentanglement as a stability/safety tool: it targets information overload (BarrierEpi) and model mismatch
  (BarrierOmin) by making dependence paths auditable.
- Provide an implementation-oriented training recipe: what losses to add, what to ablate first when unstable, and what
  diagnostic nodes should light up when things go wrong.

## Roadmap

1. The split-brain concept and the three-channel decomposition.
2. Losses/constraints that enforce closure and texture blindness.
3. Architectural choices and training/checklist guidance.
4. Diagnostics: what to monitor to confirm disentanglement is real.

## Training Checklist (Practical)

1. **Stage the objectives.** Start with shutter + decoder reconstruction and codebook stabilization before coupling in
   long-horizon rollouts or heavy control objectives.
2. **Enforce macro closure early.** Add shutter↔world-model synchronization (closure cross-entropy) as soon as the macro
   channel exists; otherwise $K$ drifts into a decorative code.
3. **Make texture blindness explicit.** Ensure the policy and the world model do not have access paths that can exploit
   $z_{\mathrm{tex}}$; if they must consume a shared embedding, use stop-grad or explicit penalties to prevent leakage.
4. **Monitor the right diagnostics.** Track closure, grounding windows, mixing/compactness, and any
   “texture firewall” checks; reconstruction loss alone is not evidence of disentanglement.
5. **Ablate when unstable.** If training diverges: reduce codebook size, increase commitment weight slowly, shorten
   rollout horizons, and verify that nuisance $z_n$ is not collapsing into texture.

:::{div} feynman-prose
Let me start with a question that puzzled me for a long time: Why do most neural networks for control fail so badly when you change the lighting, or add some irrelevant texture to the background, or do anything that a human wouldn't even notice?

The answer, once you see it, is almost embarrassingly obvious. These networks are trying to remember *everything*. Every pixel, every shadow, every speck of dust. And they're cramming it all into one big latent vector, mixing together the stuff that matters for making decisions with the stuff that's just... decoration.

That's insane! When you're driving a car, you don't need to remember the exact pattern of rust on a stop sign to know it means "stop." You don't need to track every blade of grass to navigate a highway. There's information that's *predictive* (knowing there's a stop sign means you'll need to stop) and information that's merely *reconstructive* (the rust pattern helps you draw a realistic picture, but that's it).

The architecture we're about to discuss takes this distinction seriously. We're going to split the agent's brain into three parts, each with a different job. And the key insight is that by *enforcing* this separation---not hoping the network learns it---we get agents that are more robust, more interpretable, and more efficient.
:::

(rb-world-models)=
:::{admonition} Researcher Bridge: World Models with Typed Latents
:class: info
Extends world-model architectures (Dreamer, MuZero) with explicit separation between control-relevant symbols, structured nuisance, and texture. The objective is to prevent policies from depending on high-frequency detail while preserving reconstruction fidelity.
:::

:::{div} feynman-prose
This section provides a practical guide to implementing a **split-latent** architecture that separates a *predictive* macro register from two distinct residual channels: **structured nuisance** $z_n$ (pose/basis/disturbance coordinates that can be modeled and audited) and **texture** $z_{\mathrm{tex}}$ (reconstruction-only detail). This targets **BarrierEpi** (information overload) and **BarrierOmin** (model mismatch) by preventing the World Model and policy from silently depending on texture while still representing nuisance explicitly.
:::

(sec-the-core-concept-split-brain-architecture)=
## The Core Concept: Split-Brain Architecture

:::{div} feynman-prose
Alright, let's get concrete. What's the difference between a standard agent and what we're proposing?

A **standard agent** looks at an observation and squishes it down into a single vector $z$. That vector has to do *everything*: predict what happens next, reconstruct what was seen, guide what action to take. It's like asking one employee to be the CEO, the janitor, and the receptionist all at once.

A **disentangled agent** is smarter about this. It looks at the observation and says: "Wait, let me separate out three different kinds of information here."
:::

**Standard Agent:** Encodes the state into a single vector $z$.

**Disentangled Agent:** Encodes the state into a **discrete macro-symbol** plus two distinct continuous residual channels ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`):

:::{prf:definition} The Three-Channel Latent Decomposition
:label: def-three-channel-latent

The disentangled agent's internal state at time $t$ decomposes as:

$$
Z_t = (K_t, z_{n,t}, z_{\mathrm{tex},t})

$$

where each component serves a distinct representational role:

1. **$K_t \in \mathcal{K}$ (The Macro Symbol / Law Register):** A discrete code index from a finite codebook. This is the low-frequency, causal, predictable core of the state. For downstream continuous networks, we use its embedding $z_{\text{macro}}:=e_{K_t}\in\mathbb{R}^{d_m}$, but the *information-carrying* object is the discrete symbol $K_t$.

2. **$z_{n,t} \in \mathbb{R}^{d_n}$ (Structured Nuisance / Gauge Residual):** A continuous latent for pose/basis/disturbance coordinates. This is *not* "noise": it is structured variation that may be needed for actuation and for explaining boundary-driven deviations, but it must remain disentangled from macro identity.

3. **$z_{\mathrm{tex},t} \in \mathbb{R}^{d_{\mathrm{tex}}}$ (Texture Residual):** A high-rate continuous latent for reconstruction detail. Texture is treated as an **emission residual**: it may be needed to reconstruct $x_t$ but must not be required for macro closure or for control.
:::

:::{div} feynman-prose
Let me give you a concrete mental picture. Imagine you're looking at a photograph of a chess game.

The **macro symbol** $K_t$ is like asking: "What's the position on the board?" It tells you which squares have which pieces. This is what matters for playing chess. It's discrete (there are finitely many legal positions), and it's predictive (from the position plus the next move, you can predict the new position perfectly).

The **nuisance** $z_n$ is like asking: "From what angle was this photo taken? What's the perspective distortion?" This is structured information---it follows geometric rules---and you might need it for some tasks (like reaching out to move a piece). But it doesn't change *which position* you're looking at.

The **texture** $z_{\text{tex}}$ is everything else: the grain of the wood, the reflections on the pieces, the shadows, the dust. You need it to reconstruct a realistic image, but it's completely irrelevant for deciding what move to make.

The critical insight is that these three channels have different *dynamics*. The position changes slowly and predictably (one move at a time). The viewing angle changes somewhat predictably (continuous camera motion). But the texture is essentially random---it depends on lighting, camera noise, and countless other factors. By separating them, we can model each with appropriate tools.
:::

:::{admonition} Why Discrete Macro Symbols?
:class: feynman-added tip

You might wonder: why bother making $K_t$ discrete? Why not just use a continuous vector?

Here's the thing: discreteness is a *feature*, not a bug. When you have a discrete macro symbol, you can ask precise questions like "Is the agent in state 47 or state 48?" You can count how often each state is visited. You can verify whether transitions are deterministic. You can audit the entire state machine.

With continuous latents, everything is fuzzy. "Close to state 47" is meaningless. Statistical tests become subtle. Interpretability evaporates.

The cost of discreteness is the quantization error---the fact that you can't represent every possible situation with perfect fidelity. But that's exactly what the residual channels are for! The nuisance and texture channels soak up the variation that doesn't fit into the discrete grid.

Think of it like a filing system. The discrete macro symbol is which drawer you're in. The nuisance is where in the drawer. The texture is the dust on the folder. You need all three to fully specify where a document is, but for most purposes, knowing the drawer is enough.
:::

:::{prf:definition} The Golden Rule of Causal Enclosure
:label: def-causal-enclosure

The macro symbol must satisfy the **causal enclosure** property:

$$
P(K_{t+1}\mid K_t,a_t)\ \text{is sharply concentrated (ideally deterministic)}

$$

and the **texture independence** property:

$$
I(K_{t+1};Z_{\mathrm{tex},t}\mid K_t,a_t)=0.

$$

Optionally, in the strongest form, nuisance independence also holds:

$$
I(K_{t+1};Z_{n,t}\mid K_t,a_t)=0.

$$

That is: nuisance should not be needed to predict the next macro symbol once action is accounted for.
:::

:::{div} feynman-prose
This is the heart of the whole architecture, so let me say it plainly: **the macro symbol must be predictable from its own history alone**.

If you need to peek at the texture to figure out what macro state comes next, something has gone wrong. Either your macro symbols are too coarse-grained (they're lumping together situations that evolve differently), or information is leaking where it shouldn't.

This property---that $K_t$ plus $a_t$ suffices to predict $K_{t+1}$---is what makes $K$ a *sufficient statistic* for control. You don't need to know anything else. The residuals are, by design, irrelevant for the dynamics of the thing you're actually trying to control.

If this condition fails, the World Model is forced to implicitly learn micro-dynamics, and your policy will silently depend on details it shouldn't. That's a recipe for brittleness.
:::

The macro symbol must be predictable **solely from its own history** (plus action). If the World Model needs micro-residuals to predict the next macro symbol, then $K_t$ is not a sufficient macro statistic ({ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>`).

(sec-architecture-the-disentangled-vq-vae-rnn)=
## Architecture: The Disentangled VQ-VAE-RNN

:::{div} feynman-prose
Now let's look at how this actually gets implemented. The architecture has four main components:

1. **A shared encoder** that processes raw observations into features
2. **Three "heads"** that split those features into our three channels
3. **A vector quantizer** that discretizes the macro channel
4. **A macro dynamics model** that predicts $K_{t+1}$ from $(K_t, a_t)$ alone

The key architectural discipline is the **micro-blindness** of the dynamics model. It literally cannot see the nuisance or texture channels. This isn't a soft constraint we hope the network learns---it's a hard architectural fact. The prediction head is only wired to the macro embedding.
:::

:::{admonition} Configuration Parameters
:class: feynman-added note

The configuration below sets up a typical split-latent agent. Note the relative dimensions: the macro embedding is small (32D), nuisance is similar (32D), but texture is larger (96D). This reflects the intuition that reconstruction requires more capacity than control.

The loss weights deserve attention: `lambda_closure` is set to 1.0 (this is the critical constraint), while the KL weights on nuisance and texture are much smaller (0.01 and 0.05). We're regularizing the residuals toward simple priors, but not too aggressively---they need to be able to capture real structure.
:::

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DisentangledConfig:
    """Configuration for split-latent (macro + nuisance + texture) agent."""
    obs_dim: int = 64 * 64 * 3      # Observation dimension
    hidden_dim: int = 256            # Encoder hidden dimension
    macro_embed_dim: int = 32        # Macro embedding dim (code vectors e_k)
    codebook_size: int = 512         # Number of discrete macrostates |K|
    nuisance_dim: int = 32           # Structured nuisance latent dimension
    tex_dim: int = 96                # Texture latent dimension (reconstruction-only)
    action_dim: int = 4              # Action dimension
    rnn_hidden_dim: int = 256        # Dynamics model RNN hidden

    # Loss weights
    lambda_closure: float = 1.0      # Causal enclosure weight
    lambda_slowness: float = 0.1     # Temporal smoothness weight
    lambda_nuis_kl: float = 0.01     # Nuisance KL weight (regularize, not "trash")
    lambda_tex_kl: float = 0.05      # Texture KL weight (reconstruction residual)
    lambda_vq: float = 1.0           # VQ codebook + commitment weight
    lambda_recon: float = 1.0        # Reconstruction weight

    # Training
    tex_dropout_prob: float = 0.5    # Probability of dropping texture (forces macro+nuisance decoding)
    warmup_steps: int = 1000         # Warmup for closure loss
```

:::{div} feynman-prose
The encoder is straightforward---a standard convolutional network that turns images into feature vectors. Nothing fancy here.
:::

```python
class Encoder(nn.Module):
    """Shared encoder backbone."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        # For image observations, use CNN
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        # Compute flattened size (for 64x64 input: 256 * 4 * 4 = 4096)
        self.fc = nn.Linear(4096, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] or [B, T, C, H, W]
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.view(B * T, *x.shape[2:])
            h = F.gelu(self.fc(self.conv(x)))
            return h.view(B, T, -1)
        return F.gelu(self.fc(self.conv(x)))
```

:::{div} feynman-prose
Now here's where the magic happens. The Vector Quantizer takes a continuous vector and snaps it to the nearest point in a learned codebook. This is the discretization step that produces our macro symbol $K_t$.

The tricky part is: how do you backpropagate through a discrete choice? You can't take the gradient of "which codebook entry is closest." The answer is the **straight-through estimator**: during the forward pass, we use the quantized vector; during the backward pass, we pretend the quantization never happened and pass gradients straight through.
:::

:::{admonition} The Straight-Through Trick
:class: feynman-added tip

The straight-through estimator is one of those beautiful hacks that really works. Here's the mental model:

In the forward pass: $z_q = e_K$ (the quantized code).

In the backward pass: $\partial L/\partial z_e = \partial L/\partial z_q$ (pretend $z_q = z_e$).

The line `z_q_st = z_e + (z_q - z_e).detach()` implements this cleverly. Since `.detach()` cuts the gradient, the backward pass sees only the $z_e$ term, but the forward pass sees the full expression which equals $z_q$.

This means the encoder learns to produce vectors that, when quantized, lead to good outcomes---even though the quantization itself is non-differentiable.
:::

```python
class VectorQuantizer(nn.Module):
    """
    VQ layer: maps a continuous encoder output z_e to a discrete code K and
    its corresponding embedding e_K using a straight-through estimator.
    """

    def __init__(self, codebook_size: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        self.beta = beta
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B, D]
        # Compute squared L2 distances to codebook vectors
        codebook = self.codebook.weight  # [|K|, D]
        distances = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z_e @ codebook.t()
            + codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )
        K = torch.argmin(distances, dim=-1)          # [B]
        z_q = self.codebook(K)                       # [B, D]

        # VQ-VAE losses
        codebook_loss = (z_q - z_e.detach()).pow(2).mean()
        commit_loss = self.beta * (z_e - z_q.detach()).pow(2).mean()
        vq_loss = codebook_loss + commit_loss

        # Straight-through estimator: pass gradients to encoder as if identity
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, K, vq_loss

    def embed(self, K: torch.Tensor) -> torch.Tensor:
        return self.codebook(K)
```

:::{div} feynman-prose
The decoder reverses the encoding: it takes all three channels and reconstructs the original observation. Notice that it concatenates macro, nuisance, and texture together. All three contribute to reconstruction, but only the macro channel is used for prediction and control.
:::

```python
class Decoder(nn.Module):
    """Decoder using macro + nuisance + texture latents."""

    def __init__(self, macro_dim: int, nuisance_dim: int, tex_dim: int, obs_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(macro_dim + nuisance_dim + tex_dim, 4096)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, obs_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def forward(self, z_macro: torch.Tensor, z_nuis: torch.Tensor, z_tex: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_macro, z_nuis, z_tex], dim=-1)
        h = F.gelu(self.fc(z))
        h = h.view(-1, 256, 4, 4)
        return self.deconv(h)
```

:::{div} feynman-prose
Now for the crucial piece: the **Macro Dynamics Model**. Look carefully at what it receives as input: only `z_macro` and `action`. It is *architecturally blind* to `z_nuis` and `z_tex`. This isn't a suggestion or a regularization---it's a hard constraint built into the wiring.

Why does this matter? Because it forces all the predictively-relevant information into the macro channel. If texture contained information needed to predict the future, the dynamics model couldn't use it anyway, so the encoder would be forced to put that information into the macro channel instead.

This is the architectural enforcement of causal enclosure.
:::

:::{admonition} Micro-Blindness is Architectural, Not Learned
:class: feynman-added warning

A critical point: the dynamics model's blindness to micro-channels is not a soft constraint. It's not a loss term we hope drives the network toward the right behavior. It's a *wiring decision*.

The `forward` method takes `z_macro` and `action`. Period. There's no pathway by which `z_nuis` or `z_tex` could influence the prediction even if the network "wanted" to use them.

This is the key difference from approaches that try to encourage disentanglement through clever loss functions. Those approaches work sometimes, but they're fragile. Architecture is robust.
:::

```python
class MacroDynamicsModel(nn.Module):
    """
    Macro dynamics model (micro-blind).

    Important: this module only sees z_macro (it is blind to z_nuis and z_tex). This
    forces causally/predictively relevant information into the macro channel.
    """

    def __init__(self, macro_embed_dim: int, action_dim: int, hidden_dim: int, codebook_size: int):
        super().__init__()
        self.macro_embed_dim = macro_embed_dim
        self.codebook_size = codebook_size

        # GRU for temporal dynamics
        self.gru = nn.GRUCell(macro_embed_dim + action_dim, hidden_dim)

        # Project hidden state to a distribution over next macro symbol K_{t+1}
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, codebook_size),  # logits over K
        )

        # Uncertainty estimation (optional but recommended)
        self.uncertainty = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_macro: torch.Tensor,       # [B, macro_embed_dim] (= code embedding e_K)
        action: torch.Tensor,        # [B, action_dim]
        h_prev: torch.Tensor,        # [B, hidden_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the next macro symbol distribution from the current macro embedding only.

        Returns:
            logits_next: Logits for K_{t+1} over K
            h_next: Updated hidden state
            uncertainty: Scalar uncertainty proxy
        """
        # Concatenate macro state and action (NO micro!)
        x = torch.cat([z_macro, action], dim=-1)

        # Update hidden state
        h_next = self.gru(x, h_prev)

        # Predict next macro symbol (logits over K)
        logits_next = self.predictor(h_next)
        uncertainty = self.uncertainty(h_next)

        return logits_next, h_next, uncertainty
```

:::{div} feynman-prose
The full agent ties everything together. The `encode` method produces all three channels. The `forward` method runs one step of the dynamics. Notice the **texture dropout**: with probability 0.5, we zero out the texture channel during training.

Why dropout texture? To force the macro and nuisance channels to carry structural information. If the decoder could always rely on texture, it might dump important information there. By sometimes removing texture, we ensure the decoder can reconstruct reasonable outputs from macro + nuisance alone.
:::

```python
class DisentangledAgent(nn.Module):
    """
    Split-latent VQ-VAE + macro dynamics model.

    Separates:
    - K_t (macro): a discrete symbol in K (predictive/controllable state)
    - z_n (nuisance): a structured residual (pose/basis/disturbance coordinates)
    - z_tex (texture): a reconstruction residual (detail), excluded from macro closure/control
    """

    def __init__(self, config: DisentangledConfig):
        super().__init__()
        self.config = config

        # Shared encoder
        self.encoder = Encoder(config.obs_dim, config.hidden_dim)

        # Macro: continuous pre-quantization -> discrete code K via VQ
        self.head_macro = nn.Linear(config.hidden_dim, config.macro_embed_dim)
        self.vq = VectorQuantizer(config.codebook_size, config.macro_embed_dim)

        # Nuisance: structured Gaussian residual
        self.head_nuis_mean = nn.Linear(config.hidden_dim, config.nuisance_dim)
        self.head_nuis_logvar = nn.Linear(config.hidden_dim, config.nuisance_dim)

        # Texture: reconstruction-only Gaussian residual
        self.head_tex_mean = nn.Linear(config.hidden_dim, config.tex_dim)
        self.head_tex_logvar = nn.Linear(config.hidden_dim, config.tex_dim)

        # Macro dynamics model (blind to micro)
        self.macro_dynamics = MacroDynamicsModel(
            config.macro_embed_dim,
            config.action_dim,
            config.rnn_hidden_dim,
            config.codebook_size,
        )

        # Decoder (uses all three channels)
        self.decoder = Decoder(config.macro_embed_dim, config.nuisance_dim, config.tex_dim)

    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Encode observation into a discrete macro symbol plus nuisance + texture residuals.

        Returns:
            K: Macro code index in K
            z_macro: Quantized macro embedding e_K (straight-through)
            z_nuis: Nuisance latent
            z_tex: Texture latent
            nuis_dist: (mean, logvar) for nuisance
            tex_dist: (mean, logvar) for texture
            vq_loss: codebook + commitment loss
        """
        features = self.encoder(x)

        # Macro: quantize into a discrete symbol K and embedding z_macro := e_K
        z_e = self.head_macro(features)
        z_macro, K, vq_loss = self.vq(z_e)

        # Nuisance: structured residual
        nuis_mean = self.head_nuis_mean(features)
        nuis_logvar = self.head_nuis_logvar(features)
        nuis_logvar = torch.clamp(nuis_logvar, min=-7, max=2)
        z_nuis = self._reparameterize(nuis_mean, nuis_logvar)

        # Texture: reconstruction-only residual
        tex_mean = self.head_tex_mean(features)
        tex_logvar = self.head_tex_logvar(features)
        tex_logvar = torch.clamp(tex_logvar, min=-7, max=2)
        z_tex = self._reparameterize(tex_mean, tex_logvar)

        return K, z_macro, z_nuis, z_tex, (nuis_mean, nuis_logvar), (tex_mean, tex_logvar), vq_loss

    def _reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x_t: torch.Tensor,           # [B, C, H, W] current observation
        action: torch.Tensor,         # [B, action_dim]
        h_prev: torch.Tensor,         # [B, rnn_hidden_dim]
        training: bool = True,
    ) -> dict:
        """
        Full forward pass with information dropout.

        Returns dict with all intermediate values for loss computation.
        """
        # 1. Encode current observation
        K_t, z_macro, z_nuis, z_tex, nuis_dist, tex_dist, vq_loss = self.encode(x_t)

        # 2. Macro dynamics step (blind to micro)
        logits_next, h_next, uncertainty = self.macro_dynamics(z_macro, action, h_prev)
        K_pred = torch.argmax(logits_next, dim=-1)
        z_macro_pred = self.vq.embed(K_pred)

        # 3. Texture dropout for reconstruction (forces macro+nuisance to carry structure)
        if training and torch.rand(1).item() < self.config.tex_dropout_prob:
            z_tex_for_decode = torch.zeros_like(z_tex)
        else:
            z_tex_for_decode = z_tex

        # 4. Reconstruct
        x_recon = self.decoder(z_macro, z_nuis, z_tex_for_decode)

        return {
            'K_t': K_t,
            'z_macro': z_macro,
            'z_nuis': z_nuis,
            'z_tex': z_tex,
            'z_macro_logits': logits_next,
            'K_pred': K_pred,
            'z_macro_pred': z_macro_pred,
            'nuis_dist': nuis_dist,
            'tex_dist': tex_dist,
            'vq_loss': vq_loss,
            'h_next': h_next,
            'uncertainty': uncertainty,
            'x_recon': x_recon,
            'tex_dropped': z_tex_for_decode.sum() == 0,
        }

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.rnn_hidden_dim, device=device)


```

(sec-loss-function-enforcing-macro-micro-separation)=
## Loss Function: Enforcing Macro/Micro Separation

:::{div} feynman-prose
Now we come to the loss function, and this is where the rubber meets the road. The architecture gives us the *capacity* to separate macro from micro, but the loss function is what actually *enforces* the separation.

A naive approach would be: just minimize reconstruction error and hope for the best. But that's exactly what leads to entangled representations! The network will use whatever channel is convenient, mixing information freely.

We need a more sophisticated objective---one that explicitly rewards causal enclosure, penalizes symbol churn, and keeps the residuals well-behaved. The loss has six terms, and each one does something important.
:::

The Disentangled Agent cannot be trained with a reconstruction loss alone. It requires a compound loss that enforces the **learnability threshold**: the macro channel must be predictive/closed, and the micro channel must be treated as nuisance rather than state.

:::{prf:definition} The Total Disentangled Loss
:label: def-total-disentangled-loss

The compound loss for training the split-latent agent is:

$$
\mathcal{L}_{\text{total}}
=
\lambda_{\text{recon}}\mathcal{L}_{\text{recon}}
\;+\;\lambda_{\text{vq}}\mathcal{L}_{\text{vq}}
\;+\;\lambda_{\text{closure}}\mathcal{L}_{\text{closure}}
\;+\;\lambda_{\text{slowness}}\mathcal{L}_{\text{slowness}}
\;+\;\lambda_{\text{nuis}}\mathcal{L}_{\text{nuis-KL}}
\;+\;\lambda_{\text{tex}}\mathcal{L}_{\text{tex-KL}}

$$

where:
- $\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2$ is the reconstruction loss
- $\mathcal{L}_{\text{vq}}$ is the codebook + commitment loss from vector quantization
- $\mathcal{L}_{\text{closure}} = -\log p_\psi(K_{t+1}\mid K_t,a_t)$ is the cross-entropy estimating $H(K_{t+1}\mid K_t,a_t)$
- $\mathcal{L}_{\text{slowness}} = \|e_{K_t} - e_{K_{t-1}}\|^2$ penalizes rapid symbol changes
- $\mathcal{L}_{\text{nuis-KL}} = D_{\mathrm{KL}}(q(z_n|x) \| \mathcal{N}(0,I))$ regularizes nuisance
- $\mathcal{L}_{\text{tex-KL}} = D_{\mathrm{KL}}(q(z_{\text{tex}}|x) \| \mathcal{N}(0,I))$ regularizes texture
:::

:::{div} feynman-prose
Let me walk through each term so you understand what it's doing.

**Reconstruction loss** ($\mathcal{L}_{\text{recon}}$): The standard "can you reconstruct what you saw?" This ensures the latents collectively retain enough information. Without it, the network might throw away everything.

**VQ loss** ($\mathcal{L}_{\text{vq}}$): Keeps the codebook healthy. Part of it moves the codebook vectors toward the encoder outputs; part of it (the commitment loss) encourages the encoder to commit to nearby codebook vectors rather than wandering off.

**Closure loss** ($\mathcal{L}_{\text{closure}}$): *This is the key term.* It's the cross-entropy of predicting the next macro symbol. Low closure loss means $K_{t+1}$ is predictable from $(K_t, a_t)$. Remember: the dynamics model is micro-blind, so if this loss is low, causal enclosure is satisfied.

**Slowness loss** ($\mathcal{L}_{\text{slowness}}$): Penalizes rapid changes in the macro embedding. This prevents "symbol churn"---the pathology where the macro symbol flickers rapidly between states even when nothing meaningful is changing.

**KL losses** ($\mathcal{L}_{\text{nuis-KL}}$, $\mathcal{L}_{\text{tex-KL}}$): Regularize the residual channels toward simple Gaussian priors. This implements Occam's razor: use these channels only when necessary, and prefer simple explanations.
:::

:::{admonition} The Closure Loss is an Estimator
:class: feynman-added note

The closure loss $\mathcal{L}_{\text{closure}} = -\log p_\psi(K_{t+1}\mid K_t,a_t)$ is worth understanding precisely.

When averaged over the training distribution, this quantity estimates the conditional entropy $H(K_{t+1}\mid K_t,a_t)$. Low conditional entropy means high predictability---exactly what we want for a good macro state.

The beautiful thing is that this works even though the "true" dynamics are unknown. We're training a predictor $p_\psi$ to forecast the next macro symbol, and if it succeeds, we know the macro channel is capturing predictive structure. If it fails despite our best efforts, the macro channel isn't good enough.
:::

```python
class DisentangledLoss(nn.Module):
    """
    Compound loss for training the split-latent agent.

    Implements the five key constraints:
    1. Closure: Macro must predict itself (causal enclosure)
    2. Slowness: Macro should change slowly (prevents symbol churn)
    3. Nuisance prior: nuisance should be regularized (but is not "trash")
    4. Texture prior: texture should stay close to a simple prior (reconstruction residual)
    4. VQ: Macro must remain quantized (symbolic)
    5. Reconstruction: Both channels needed for full reconstruction
    """

    def __init__(self, config: DisentangledConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: dict,              # From DisentangledAgent.forward()
        x_target: torch.Tensor,     # Target observation (x_{t+1} for recon)
        K_next: torch.Tensor,       # Target macro code at t+1 (LongTensor)
        z_macro_prev: Optional[torch.Tensor] = None,  # e_{K_{t-1}} (for slowness)
        step: int = 0,              # Training step for warmup
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute all loss components.

        Returns:
            total_loss: Scalar loss for backprop
            loss_dict: Individual losses for logging
        """
        losses = {}

        # === A. RECONSTRUCTION LOSS ===
        # Standard pixel-wise reconstruction
        losses['recon'] = F.mse_loss(outputs['x_recon'], x_target)

        # === B. CAUSAL ENCLOSURE LOSS (SYMBOLIC) ===
        # Predict the next macro symbol K_{t+1} from the current macro only.
        losses['closure'] = F.cross_entropy(outputs['z_macro_logits'], K_next)

        # Warmup: gradually increase closure weight
        closure_weight = min(1.0, step / self.config.warmup_steps)

        # === C. SLOWNESS LOSS ===
        # Penalize rapid changes in the macro embedding e_K (proxy for symbol churn)
        if z_macro_prev is not None:
            losses['slowness'] = (outputs['z_macro'] - z_macro_prev).pow(2).mean()
        else:
            losses['slowness'] = torch.tensor(0.0, device=outputs['z_macro'].device)

        # === D. RESIDUAL PRIORS ===
        # Regularize nuisance and texture toward simple priors (macro closure remains micro-blind).
        nuis_mean, nuis_logvar = outputs['nuis_dist']
        tex_mean, tex_logvar = outputs['tex_dist']
        losses['nuis_kl'] = self._kl_divergence(nuis_mean, nuis_logvar)
        losses['tex_kl'] = self._kl_divergence(tex_mean, tex_logvar)

        # === E. VQ LOSS (CODEBOOK + COMMITMENT) ===
        losses['vq'] = outputs['vq_loss']

        # === TOTAL LOSS ===
        total = (
            self.config.lambda_recon * losses['recon']
            + self.config.lambda_vq * losses['vq']
            + self.config.lambda_closure * closure_weight * losses['closure']
            + self.config.lambda_slowness * losses['slowness']
            + self.config.lambda_nuis_kl * losses['nuis_kl']
            + self.config.lambda_tex_kl * losses['tex_kl']
        )

        losses['total'] = total
        losses['closure_weight'] = closure_weight

        return total, losses

    def _kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(mean, var) || N(0, I))"""
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
```

(sec-the-complete-training-loop)=
## The Complete Training Loop

:::{div} feynman-prose
Training a disentangled agent requires some care. We're processing temporal sequences (the agent sees a stream of observations over time), and we need to track hidden states, previous macro symbols, and targets at the next timestep.

The training loop processes sequences step by step, accumulating losses. A few things to note:

1. We initialize the RNN hidden state at the start of each sequence.
2. At each step, we encode both the current and next observations---the next observation gives us the target $K_{t+1}$.
3. We keep track of the previous macro embedding for the slowness loss.
4. Gradient clipping is essential: the dynamics across time can cause gradients to explode.
:::

```python
class DisentangledTrainer:
    """
    Complete training loop for the split-latent agent.

    Handles:
    - Temporal sequence processing
    - Gradient isolation between macro/micro paths
    - Warmup schedules
    - Diagnostic monitoring
    """

    def __init__(
        self,
        agent: DisentangledAgent,
        learning_rate: float = 3e-4,
        device: str = 'cuda',
    ):
        self.agent = agent.to(device)
        self.device = device
        self.loss_fn = DisentangledLoss(agent.config)

        # Separate optimizers for different components (optional)
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

        # Diagnostics
        self.closure_history = []
        self.dispersion_history = []
        self.step = 0

    def train_step(
        self,
        observations: torch.Tensor,  # [B, T, C, H, W] sequence
        actions: torch.Tensor,        # [B, T-1, action_dim]
    ) -> dict:
        """
        Train on a sequence of observations.

        Args:
            observations: Temporal sequence of observations
            actions: Actions taken between observations

        Returns:
            Dictionary of losses and diagnostics
        """
        B, T = observations.shape[:2]
        observations = observations.to(self.device)
        actions = actions.to(self.device)

        self.optimizer.zero_grad()

        # Initialize hidden state
        h = self.agent.init_hidden(B, self.device)

        total_loss = 0.0
        all_losses = {k: 0.0 for k in ['recon', 'vq', 'closure', 'slowness', 'dispersion']}

        z_macro_prev = None
        K_nexts = []          # Targets K_{t+1}
        logits_nexts = []     # Predicted logits for K_{t+1}

        for t in range(T - 1):
            x_t = observations[:, t]
            x_t1 = observations[:, t + 1]
            a_t = actions[:, t]

            # Forward pass at time t
            outputs_t = self.agent(x_t, a_t, h, training=True)
            h = outputs_t['h_next']

            # Encode t+1 for closure target
            K_t1, _, _, _, _ = self.agent.encode(x_t1)

            # Compute loss
            loss, losses = self.loss_fn(
                outputs_t,
                x_t1,
                K_t1,
                z_macro_prev,
                self.step,
            )

            total_loss = total_loss + loss
            for k in all_losses:
                if k in losses:
                    all_losses[k] = all_losses[k] + losses[k].item()

            z_macro_prev = outputs_t['z_macro'].detach()
            K_nexts.append(K_t1.detach())
            logits_nexts.append(outputs_t['z_macro_logits'].detach())

        # Backprop
        total_loss = total_loss / (T - 1)
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.step += 1

        # Compute diagnostics
        avg_losses = {k: v / (T - 1) for k, v in all_losses.items()}
        avg_losses['total'] = total_loss.item()

        # Closure ratio (see 9.5): symbolic cross-entropy gain vs baseline
        closure_ratio = self._compute_closure_ratio(K_nexts, logits_nexts)
        avg_losses['closure_ratio'] = closure_ratio

        self.closure_history.append(closure_ratio)

        return avg_losses

    def _compute_closure_ratio(
        self,
        K_nexts: List[torch.Tensor],
        logits_nexts: List[torch.Tensor],
    ) -> float:
        """
        Closure ratio diagnostic (discrete macro).

        Ratio < 1  : predictive macro model beats a baseline (law-like)
        Ratio ~ 1  : macro is no more predictable than baseline (no learned law)
        Ratio > 1  : predictor is worse than baseline (bug/degenerate training)
        """
        if not K_nexts:
            return float('nan')

        logits = torch.cat(logits_nexts, dim=0)  # [(T-1)B, |K|]
        K_next = torch.cat(K_nexts, dim=0)       # [(T-1)B]

        # Model conditional cross entropy ~ H(K_{t+1}|K_t,a_t)
        ce_model = F.cross_entropy(logits, K_next, reduction='mean')

        # Baseline: marginal code model (no conditioning)
        K_size = self.agent.config.codebook_size
        hist = torch.bincount(K_next, minlength=K_size).float()
        p = (hist + 1.0) / (hist.sum() + K_size)  # Laplace smoothing
        ce_baseline = (-torch.log(p[K_next])).mean()

        return (ce_model / (ce_baseline + 1e-8)).item()
```

(sec-runtime-diagnostics-the-closure-ratio)=
## Runtime Diagnostics: The Closure Ratio

:::{div} feynman-prose
How do you know if your split-latent agent is actually working? You can't just look at reconstruction loss---a tangled representation can reconstruct just fine.

The key diagnostic is the **Closure Ratio**. This measures how much better your macro dynamics model is at predicting $K_{t+1}$ compared to a baseline that just predicts the marginal distribution of $K$.
:::

:::{prf:definition} Closure Ratio
:label: def-closure-ratio

The **Closure Ratio** is defined as:

$$
\text{Closure Ratio}
=
\frac{\mathbb{E}\big[-\log p_\psi(K_{t+1}\mid K_t,a_t)\big]}{\mathbb{E}\big[-\log p_{\text{base}}(K_{t+1})\big]}.

$$

With $p_{\text{base}}$ chosen as the marginal symbol model, the numerator estimates $H(K_{t+1}\mid K_t,a_t)$ and the denominator estimates $H(K_{t+1})$, so the *gap* is a direct estimate of predictive information $I(K_{t+1};K_t,a_t)$.
:::

:::{div} feynman-prose
Let me explain what this ratio tells you.

If the ratio is close to 1, your dynamics model is no better than random guessing (conditioned on the marginal). That means the macro symbol carries no predictive information---either the encoding is garbage, or the dynamics are genuinely unpredictable.

If the ratio is much less than 1 (say, 0.2 or 0.3), your dynamics model is *much* better than random. The macro symbol contains meaningful structure, and the next symbol is highly predictable from the current one plus the action. This is what we want.

If the ratio is greater than 1, something is badly wrong. Your model is *worse* than random guessing, which shouldn't happen unless there's a bug or severe training instability.
:::

| Closure Ratio | Interpretation                                                 | Action                                                                            |
|---------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------|
| $\approx 1$   | **No predictive law** --- macro dynamics no better than marginal | Increase model capacity or improve macro encoder; check that actions are provided |
| $\ll 1$       | **Success** --- conditional entropy collapses vs marginal        | Sufficient macro statistic learned                                                |
| $> 1$         | **Worse than baseline**                                        | Bug/degeneracy (labels, codebook collapse, optimizer instability)                 |

:::{admonition} What the Closure Ratio Really Measures
:class: feynman-added tip

Here's another way to think about it. The closure ratio is essentially:

$$
\text{Closure Ratio} = \frac{H(K_{t+1} \mid K_t, a_t)}{H(K_{t+1})}

$$

Now, by the chain rule of entropy, $H(K_{t+1}) = H(K_{t+1} \mid K_t, a_t) + I(K_{t+1}; K_t, a_t)$.

So the closure ratio is:

$$
\text{Closure Ratio} = \frac{H(K_{t+1} \mid K_t, a_t)}{H(K_{t+1} \mid K_t, a_t) + I(K_{t+1}; K_t, a_t)}

$$

When the ratio approaches 0, almost all of $H(K_{t+1})$ is "explained" by the predictive information---the macro symbol is nearly deterministic given the past. When the ratio approaches 1, there's no predictive information at all.

This connects directly to the notion of causal emergence: a good macro state is one where the macro-level dynamics are more predictive (less noisy) than the micro-level dynamics.
:::

```python
class ClosureMonitor:
    """
    Monitor for split-latent training.

    Integrates with Sieve nodes to detect failure modes.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.closure_ratios = []
        self.macro_predictability = []
        self.micro_entropy = []

    def update(
        self,
        logits_next: torch.Tensor,   # [B, |K|]
        K_next: torch.Tensor,        # [B]
        micro_logvar: torch.Tensor,  # [B, d_mu]
    ):
        """Update diagnostics with new batch."""
        # Macro predictability: cross entropy (nats)
        ce_model = F.cross_entropy(logits_next, K_next, reduction='mean').item()

        # Baseline: marginal code model (per-batch)
        K_size = logits_next.shape[-1]
        hist = torch.bincount(K_next, minlength=K_size).float()
        p = (hist + 1.0) / (hist.sum() + K_size)  # Laplace smoothing
        ce_base = (-torch.log(p[K_next])).mean().item()

        # Closure ratio
        ratio = ce_model / (ce_base + 1e-8)
        self.closure_ratios.append(ratio)

        # Micro entropy proxy (Gaussian differential entropy)
        import math
        micro_ent = (0.5 * (micro_logvar + math.log(2 * math.pi * math.e))).sum(dim=-1).mean().item()

        # Individual metrics
        self.macro_predictability.append(ce_model)
        self.micro_entropy.append(micro_ent)

        # Keep window
        if len(self.closure_ratios) > self.window_size:
            self.closure_ratios.pop(0)
            self.macro_predictability.pop(0)
            self.micro_entropy.pop(0)

    def get_diagnostics(self) -> dict:
        """Get current diagnostic summary."""
        import numpy as np

        if not self.closure_ratios:
            return {}

        ratios = np.array(self.closure_ratios)
        macro_pred = np.array(self.macro_predictability)
        micro_ent = np.array(self.micro_entropy)

        return {
            'closure_ratio_mean': ratios.mean(),
            'closure_ratio_std': ratios.std(),
            'macro_predictability': macro_pred.mean(),
            'micro_entropy': micro_ent.mean(),
            'macro_model_learned': ratios.mean() < 0.5,  # Success threshold
            'recommendation': self._get_recommendation(ratios.mean()),
        }

    def _get_recommendation(self, ratio: float) -> str:
        if ratio < 0.3:
            return "Excellent: Clear separation of predictive macro and nuisance"
        elif ratio < 0.7:
            return "Good: Macro closure partially learned, consider increasing closure weight"
        elif ratio < 1.0:
            return "Warning: Macro/micro entanglement detected, increase lambda_closure"
        else:
            return "Error: Micro residual more predictive than macro, check architecture"

    def check_sieve_nodes(self) -> dict:
        """
        Map diagnostics to Sieve node checks.

        Returns status for relevant nodes.
        """
        diag = self.get_diagnostics()
        if not diag:
            return {}

        return {
            # TameCheck: Is the world model interpretable?
            'TameCheck': 'PASS' if diag['macro_predictability'] < 1.0 else 'WARN',

            # ComplexCheck: Is model capacity appropriate?
            'ComplexCheck': 'PASS' if diag['micro_entropy'] > -2.0 else 'WARN',

            # GeomCheck: Are latent spaces well-separated?
            'GeomCheck': 'PASS' if diag['closure_ratio_mean'] < 0.5 else 'WARN',

            # ParamCheck: Is the macro dynamics stable?
            'ParamCheck': 'PASS' if diag['closure_ratio_std'] < 0.5 else 'WARN',
        }
```

(sec-advanced-hierarchical-multi-scale-latents)=
## Advanced: Hierarchical Multi-Scale Latents

:::{div} feynman-prose
So far we've split the latent space into macro and micro. But what if one level of "macro" isn't enough?

Consider a robot navigating a building. There's the question of "which room am I in?" (very slow, changes rarely), "where in the room am I?" (medium speed, changes every few seconds), and "exactly what's my pose?" (fast, changes continuously). These are all "macro" in different senses---they're all about structure rather than texture---but they operate at different timescales.

A single macro register can't capture this hierarchy. We need multiple levels of discretization, each operating at its own clock speed.
:::

For complex environments, a single macro/micro split may be insufficient. A macro hierarchy extends the split-latent idea to multiple discrete scales:

:::{prf:definition} Hierarchical Multi-Scale Latent Decomposition
:label: def-hierarchical-latent

A hierarchical split-latent state has the form:

$$
Z_t = (K_t^{(1)}, K_t^{(2)}, \ldots, K_t^{(L)}, z_{\mu,t}),
\qquad
z_{\text{macro}}^{(i)} := e^{(i)}_{K_t^{(i)}}\in\mathbb{R}^{d_i}.

$$

Where $K^{(1)}$ is the slowest (most abstract) level and $K^{(L)}$ is the fastest (most detailed) macro symbol. The micro residual $z_{\mu,t}$ handles reconstruction detail below the finest macro scale.
:::

:::{div} feynman-prose
The key insight is **clockwork updating**. The slowest level (level 1) only updates every, say, 8 steps. The medium level (level 2) updates every 4 steps. The fastest macro level (level 3) updates every step. This reflects the intuition that abstract state changes slowly while detailed state changes quickly.

There's also **top-down modulation**: the slower levels influence the faster levels. If you know which room you're in, that constrains where in the room you could be. If you know where in the room you are, that constrains your exact pose.

This hierarchical structure is inspired by several ideas in the literature: Clockwork RNNs (where different RNN layers operate at different speeds), hierarchical world models, and the general principle that good representations separate timescales.
:::

```python
class HierarchicalDisentangled(nn.Module):
    """
    Multi-scale split-latent architecture.

    Each level operates at a different timescale:
    - Level 1: Slowest (global game state, long-term goals)
    - Level 2: Medium (object positions, velocities)
    - Level 3: Fast (fine motor control, reactions)
    - Micro: Noise (textures, particles)

    Inspired by Clockwork VAE (Saxena et al.) and
    Hierarchical World Models (Hafner et al.)
    """

    def __init__(
        self,
        config: DisentangledConfig,
        n_levels: int = 3,
        level_dims: List[int] = [8, 16, 32],
        level_codebook_sizes: List[int] = [64, 128, 256],
        level_update_freqs: List[int] = [8, 4, 1],  # Update every N steps
    ):
        super().__init__()
        self.n_levels = n_levels
        self.level_dims = level_dims
        self.level_codebook_sizes = level_codebook_sizes
        self.update_freqs = level_update_freqs

        self.encoder = Encoder(config.obs_dim, config.hidden_dim)

        # Separate heads for each macro level (pre-quantization)
        self.macro_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, dim) for dim in level_dims
        ])

        # VQ quantizers and macro dynamics models at each level
        self.vq_levels = nn.ModuleList([
            VectorQuantizer(level_codebook_sizes[i], level_dims[i])
            for i in range(n_levels)
        ])
        self.macro_dynamics_models = nn.ModuleList([
            MacroDynamicsModel(
                level_dims[i],
                config.action_dim,
                config.rnn_hidden_dim // n_levels,
                level_codebook_sizes[i],
            )
            for i in range(n_levels)
        ])

        # Cross-level connections (top-down modulation)
        self.cross_level = nn.ModuleList([
            nn.Linear(level_dims[i], level_dims[i + 1])
            for i in range(n_levels - 1)
        ])

        # Texture head (reconstruction residual)
        self.tex_head = nn.Linear(config.hidden_dim, config.tex_dim)

        # Decoder uses macro levels + texture (no nuisance channel in this sketch)
        macro_total_dim = sum(level_dims)
        self.decoder = Decoder(macro_total_dim, nuisance_dim=0, tex_dim=config.tex_dim)

        self.step_counter = 0

    def forward(
        self,
        x_t: torch.Tensor,
        action: torch.Tensor,
        h_prevs: List[torch.Tensor],  # Hidden state for each level
        training: bool = True,
    ) -> dict:
        """
        Hierarchical forward pass with clockwork updates.
        """
        features = self.encoder(x_t)

        z_macros = []
        z_macro_preds = []
        h_nexts = []

        top_down_context = None

        for i in range(self.n_levels):
            # Encode this level (pre-quantization)
            z_e_i = self.macro_heads[i](features)

            # Add top-down modulation from slower levels
            if top_down_context is not None:
                z_e_i = z_e_i + self.cross_level[i - 1](top_down_context)

            # Quantize into a discrete macro symbol K^{(i)} and embedding z_macro^{(i)} := e_{K^{(i)}}
            z_macro_i, K_i, _ = self.vq_levels[i](z_e_i)

            # Update macro dynamics only at appropriate frequency
            if self.step_counter % self.update_freqs[i] == 0:
                logits_i, h_next_i, _ = self.macro_dynamics_models[i](
                    z_macro_i, action, h_prevs[i]
                )
                K_pred_i = torch.argmax(logits_i, dim=-1)
                z_pred_i = self.vq_levels[i].embed(K_pred_i)
            else:
                # Hold state
                z_pred_i = z_macro_i
                h_next_i = h_prevs[i]

            z_macros.append(z_macro_i)
            z_macro_preds.append(z_pred_i)
            h_nexts.append(h_next_i)

            top_down_context = z_macro_i

        # Texture (always updates)
        z_tex = self.tex_head(features)

        # Texture dropout: force the macro stack to carry structure
        if training and torch.rand(1).item() < self.config.tex_dropout_prob:
            z_tex = torch.zeros_like(z_tex)

        # Decode from macro levels + texture
        z_macro_all = torch.cat(z_macros, dim=-1)
        z_nuis_empty = torch.zeros(z_tex.shape[0], 0, device=z_tex.device)
        x_recon = self.decoder(z_macro_all, z_nuis_empty, z_tex)

        self.step_counter += 1

        return {
            'z_macros': z_macros,
            'z_macro_preds': z_macro_preds,
            'z_tex': z_tex,
            'h_nexts': h_nexts,
            'x_recon': x_recon,
        }
```

:::{admonition} When to Use Hierarchical vs. Flat
:class: feynman-added note

The hierarchical architecture adds complexity. When is it worth it?

**Use hierarchical when:**
- Your environment has multiple obvious timescales (rooms vs. positions vs. poses)
- Long-horizon planning is important (slow levels provide stable targets)
- You're seeing symbol churn at a single macro level

**Stay with flat when:**
- Your dynamics are already fairly uniform in timescale
- You're compute-constrained (multiple VQ codebooks are expensive)
- Interpretability at a single level is more important than capturing temporal hierarchy

The flat architecture is a good default. Upgrade to hierarchical when you have evidence that a single macro level isn't capturing the structure you need.
:::

(sec-literature-connections)=
## Literature Connections (Mapping + Differences)

:::{div} feynman-prose
Let me be honest: very little of what we're presenting here is truly new. The ingredients---vector quantization, world models, information bottlenecks, disentangled representations---all exist in the literature. What's different is how we're combining them and what we're insisting on.

The key differences are:
1. We use a **discrete** macro state, not just a low-dimensional continuous one
2. We enforce causal enclosure **architecturally**, not just through loss functions
3. We have explicit **diagnostics** that tell you whether the separation is working

The table below maps our constructs to the closest literature, along with what's different here.
:::

This framework is largely a **recomposition** of known ideas. What differs is the insistence on (i) a *discrete* macro state used for prediction/control, and (ii) explicit, online-auditable contracts tying representation, dynamics, and safety.

:::{div} feynman-added
| Construct in this document                         | Closest literature                                                                  | What is different here                                                                                                                                                                          | Representative references                                                        |
|----------------------------------------------------|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Discrete macro register $K_t$**                  | discrete latents / vector quantization                                              | $K_t$ is treated as the *control-relevant* state so closure/capacity checks are well-typed, an enabler for audit-friendly information constraints rather than merely a compression mechanism                                                                        | {cite}`oord2017vqvae`                                                            |
| **Causal enclosure / closure loss**                | predictive state representations, state abstraction, bisimulation-style sufficiency | closure is used as an explicit defect functional to certify "macro sufficiency" for predicting macro dynamics                                                                                   | {cite}`littman2001predictive,singh2004predictive,li2006towards,ferns2004metrics` |
| **Typed residual split $(z_n, z_{\mathrm{tex}})$** | rate-distortion / information bottleneck views of representation learning           | nuisance $z_n$ is modeled and auditable (and may be control-relevant), while texture $z_{\mathrm{tex}}$ is explicitly reconstruction-only and prohibited from influencing macro closure/control | {cite}`tishby2015deep`                                                           |
| **Micro-blind macro dynamics $\bar P$**            | latent world models / predictive models                                             | macro dynamics is constrained to depend on $K$ (and action) only; violation is diagnosed as enclosure failure                                                                                   | {cite}`hafner2019dreamer,ha2018worldmodels,lecun2022path`                        |
| **Gate Nodes + Barriers**                          | CMDPs and safe RL constraints                                                       | constraints include *representation* and *interface* diagnostics (grounding, mixing, saturation, switching), not only expected cost                                                             | {cite}`altman1999constrained,achiam2017constrained,chow2018lyapunov`             |
| **MaxEnt/KL control + path-entropy exploration**   | entropy-regularized RL, KL-control, linearly-solvable control                       | exploration is defined on the discrete macro register and tied to capacity/grounding diagnostics                                                                                                | {cite}`haarnoja2018soft,todorov2009efficient,kappen2005path`                     |
| **State-space sensitivity metric $G$**             | information geometry / natural gradient                                             | emphasizes **state-space** sensitivity as a runtime regulator (in addition to parameter-space natural gradients)                                                                                | {cite}`amari1998natural,schulman2015trpo,martens2015kfac`                        |
| **Belief evolution as filter + projection**        | Bayesian filtering + constrained inference                                          | maps Sieve events to explicit belief-space projections/reweightings (predict -> update -> project)                                                                                                | {cite}`rabiner1989tutorial`                                                      |
| **Entropic OT bridge (optional)**                  | entropic optimal transport / Schrodinger bridge                                     | used only as a unifying *view* of KL-regularized path measures, not as a required ontology                                                                                                      | {cite}`cuturi2013sinkhorn,leonard2014schrodinger`                                |
:::

:::{div} feynman-prose
**Key pointers for further reading:**
- Causal emergence and macro closure as a modeling advantage: {cite}`hoel2017map,rosas2020reconciling`
- Information bottleneck perspective on representation learning: {cite}`tishby2015deep`
:::

(sec-computational-costs)=
## Computational Costs

:::{div} feynman-prose
Let's be practical. What does all this cost in terms of compute?

The good news is that most of the cost comes from things you'd be doing anyway: encoding observations, decoding reconstructions. The disentanglement-specific costs---the VQ lookup, the closure cross-entropy---are relatively cheap.

The main overhead is the codebook lookup in the VQ layer, which scales with the codebook size $|\mathcal{K}|$. If you're using 512 codes, that's 512 distance computations per sample. For most applications this is negligible compared to the CNN encoder.
:::

| Loss Component | Formula | Time Complexity | Notes |
|----------------|---------|-----------------|-------|
| $\mathcal{L}_{\text{recon}}$ | $\Vert x - \hat{x} \Vert^2$ | $O(BD)$ | Baseline reconstruction term |
| $\mathcal{L}_{\text{vq}}$ | $\lVert \operatorname{sg}[z_e]-e_{K}\rVert^2 + \beta\lVert z_e-\operatorname{sg}[e_K]\rVert^2$ | $O(B\lvert\mathcal{K}\rvert)$ | Codebook lookup/update |
| $\mathcal{L}_{\text{closure}}$ | $-\log p_\psi(K_{t+1}\mid K_t,a_t)$ | $O(B\lvert\mathcal{K}\rvert)$ | Macro-prediction head + cross-entropy |
| $\mathcal{L}_{\text{slowness}}$ | $\Vert e_{K_t} - e_{K_{t-1}} \Vert^2$ | $O(Bd_m)$ | Embedding drift penalty |
| $\mathcal{L}_{\text{dispersion}}$ | $D_{\mathrm{KL}}(q_{\text{micro}} \Vert \mathcal{N}(0,I))$ | $O(BZ_\mu)$ | Micro KL / nuisance regularizer |

Total cost depends on $|\mathcal{K}|$, whether closure is computed online or intermittently, and whether heavy diagnostics are amortized across steps.

:::{div} feynman-prose
**When should you use a split-latent architecture vs. something simpler?** Here's a rough guide:
:::

**When to Use Split-Latent vs Standard:**

| Scenario                              | Recommendation                                                                  |
|---------------------------------------|---------------------------------------------------------------------------------|
| High-frequency texture (games, video) | Use split-latent --- separate predictive state from nuisance texture              |
| Low-noise simulation (MuJoCo)         | Standard may suffice --- dynamics are already clean                               |
| Real-world robotics                   | Use split-latent --- sensor noise is significant                                  |
| Long-horizon planning                 | Use hierarchical split-latent --- multiple timescales                             |
| Compute-constrained                   | Standard may be preferable if codebook + closure diagnostics are not affordable |

(sec-control-theory-translation-dictionary)=
## Control Theory Translation: Dictionary

:::{div} feynman-prose
If you come from a control theory background, you might be wondering how all this relates to concepts you already know. Here's a translation dictionary.

The key insight is that the critic is essentially a Lyapunov function---it defines "how far from the goal" you are, and good policies decrease it. The world model is the system dynamics. The Fragile Index measures how sensitive the system is to perturbations.
:::

To ensure rigorous connections to the established literature, we explicitly map Hypostructure components to their Control Theory and optimization equivalents.

:::{div} feynman-added
| Hypostructure Component                                    | Control / Optimization Term                       | Role                                                          |
|:-----------------------------------------------------------|:--------------------------------------------------|:--------------------------------------------------------------|
| **Critic**                                                 | **Lyapunov / value function** ($V$)               | Defines stability regions and cost contours.                  |
| **Policy**                                                 | **Lie Derivative Controller** ($\mathcal{L}_f V$) | Actuator that maximizes negative definiteness of $\dot{V}$.   |
| **World Model**                                            | **System Dynamics** ($f(x, u)$)                   | The vector field governing the flow.                          |
| **Fragile Index**                                          | **State-space sensitivity metric** ($G_{ij}$)     | Local conditioning from Hessian/Fisher structure.             |
| **StiffnessCheck**                                         | **LaSalle's Invariance Principle**                | Guarantee that the system does not get stuck in limit cycles. |
| **BarrierAction**                                          | **Controllability Gramian**                       | Measure of whether the actuator can affect the state.         |
| **Scaling coefficients** ($\alpha, \beta, \gamma, \delta$) | **Learning-dynamics coefficients**                | Relative update/volatility scales per component.              |
| **BarrierTypeII**                                          | **Scaling Hierarchy**                             | Ensures faster components don't outrun slower ones.           |
:::

**Related Work:**
- {cite}`chang2019neural` --- Neural Lyapunov Control
- {cite}`berkenkamp2017safe` --- Safe Model-Based RL with Stability Guarantees
- {cite}`lasalle1960extent` --- The Extent of Asymptotic Stability

(sec-differential-geometry-view-curvature-as-conditioning)=
## Differential-Geometry View (No Physics): Curvature as Conditioning

:::{div} feynman-prose
There's one more perspective I want to share: the geometric view of learning. This isn't about physics---it's about understanding the "shape" of the optimization problem.

When you're training a neural network, you're navigating a high-dimensional parameter space. But not all directions are equal. Some directions are "steep" (small changes cause big effects), others are "flat" (big changes cause small effects). The metric tensor $G$ captures this local geometry.

The natural gradient uses this geometry to take steps that are sensible in "intrinsic" terms, not just in terms of raw parameter coordinates. It's like the difference between walking 100 meters on flat ground vs. 100 meters up a steep hill---same coordinate distance, very different effort.
:::

The Fragile Agent uses differential geometry as a **regulation tool**: the metric $G$ (from Hessian curvature and/or Fisher information) defines a local notion of distance/conditioning in latent space, and therefore how aggressively the controller should update.

**Core relationship.** Objective curvature defines $G$; $G$ defines how updates are scaled:

$$
\theta_{t+1} = \theta_t + \eta\,G^{-1}\nabla_\theta \mathcal{L}.

$$

**Dictionary (geometry to optimization):**

:::{div} feynman-added
| Differential geometry / optimization | Fragile Agent            | Interpretation                              |
|:-------------------------------------|:-------------------------|:--------------------------------------------|
| Metric tensor                        | $G$                      | Local sensitivity / conditioning            |
| Metric distance                      | $d_G(\cdot,\cdot)$       | Trust-region measure in latent space        |
| Natural gradient                     | $G^{-1}\nabla$           | Preconditioned update direction             |
| Ill-conditioning                     | $\lambda_{\max}(G)\gg 1$ | Updates should shrink to maintain stability |
:::

**Adaptive conditioning ("freeze-out").** When $G$ becomes extremely ill-conditioned, geometry-aware updates shrink ($G^{-1}\to 0$ along stiff directions), preventing destabilizing steps and signaling that the current regime is hard to model/control with available capacity.

:::{div} feynman-prose
**Related Work:**
- {cite}`bronstein2021geometric` --- Geometric Deep Learning (geometry for inductive bias)
- {cite}`amari1998natural` --- Natural Gradient (information geometry)

**Key Distinction from Geometric Deep Learning:** Geometric Deep Learning uses geometry to design **architectures** (equivariant neural networks). The Fragile Agent uses geometry for **runtime regulation**: curvature is estimated from the critic/policy sensitivity and used to adapt update magnitudes online.
:::

(sec-the-entropy-regularized-objective-functional)=
## The Entropy-Regularized Objective Functional

:::{div} feynman-prose
Finally, let's put together the full objective that the agent is optimizing. This brings together all the pieces: task cost, representation complexity, and control effort.

The key idea is that we're not just minimizing cost---we're trading off multiple objectives, each measured in consistent units (nats). This is the natural structure behind information bottlenecks (for representation) and MaxEnt RL (for policy).
:::

We use a regularized objective (in nats) that trades off task cost, representation complexity, and control effort.

:::{prf:definition} Instantaneous Regularized Objective
:label: def-instantaneous-objective

Define an instantaneous regularized objective:

$$
F_t
:=
V(Z_t)
+ \beta_K\big(-\log p_\psi(K_t)\big)
+ \beta_n D_{\mathrm{KL}}\!\left(q(z_{n,t}\mid x_t)\ \Vert\ p(z_n)\right)
+ \beta_{\mathrm{tex}} D_{\mathrm{KL}}\!\left(q(z_{\mathrm{tex},t}\mid x_t)\ \Vert\ p(z_{\mathrm{tex}})\right)
+ T_c D_{\mathrm{KL}}\!\left(\pi(\cdot\mid K_t)\ \Vert\ \pi_0(\cdot\mid K_t)\right),

$$

where {math}`Z_t=(K_t,z_{n,t},z_{\mathrm{tex},t})` and all terms are measured in nats.
:::

A practical monotonicity surrogate is:

$$
\mathcal{L}_{\downarrow F}
:=
\mathbb{E}\!\left[\mathrm{ReLU}\!\left(F_{t+1}-F_t\right)^2\right].

$$

:::{div} feynman-prose
Let me unpack each term:

- **$V(Z_t)$**: The task-aligned cost-to-go. This is what the critic estimates. Lower is better.

- **$\beta_K(-\log p_\psi(K_t))$**: The macro codelength penalty. Using rare symbols costs more. This implements Occam's razor for the discrete representation---prefer common, predictable states.

- **$\beta_n D_{\mathrm{KL}}(q(z_{n,t}\mid x_t)\Vert p(z_n))$**: The nuisance regularizer. Don't let the nuisance channel become a junk drawer. Keep it close to a simple prior unless there's good reason to deviate.

- **$\beta_{\mathrm{tex}} D_{\mathrm{KL}}(q(z_{\mathrm{tex},t}\mid x_t)\Vert p(z_{\mathrm{tex}}))$**: The texture regularizer. Same idea: texture is for reconstruction, so don't let it explode unnecessarily.

- **$T_c D_{\mathrm{KL}}(\pi\Vert\pi_0)$**: The KL-regularized control effort. Don't deviate from the prior policy unless it helps. This is the MaxEnt / soft RL term that encourages exploration and robustness.

The monotonicity loss $\mathcal{L}_{\downarrow F}$ is a clever trick: instead of just minimizing $F$, we penalize increases in $F$ from one step to the next. This encourages trajectories that smoothly descend the objective landscape, which tends to produce more stable behavior.
:::

(sec-atlas-manifold-dictionary-from-topology-to-neural-networks)=
## Atlas-Manifold Dictionary: From Topology to Neural Networks

:::{div} feynman-prose
Let me end with a beautiful connection: the relationship between manifold theory and the multi-chart architectures we might use for complex representation learning.

A manifold is a space that locally looks like Euclidean space, but globally might have interesting topology (think of the surface of a sphere---locally flat, globally curved). To describe a manifold, you need multiple "charts"---local coordinate systems that together cover the whole space.

The neural network analog is a mixture of experts: each "expert" handles a local region, and a gating network decides which expert to use. This is exactly an atlas structure!
:::

This section provides a translation dictionary connecting **manifold theory** to the **neural network implementations** described in Sections 7.7-7.8 (Attentive Atlas routing assumed unless noted).

(sec-core-correspondences)=
### Core Correspondences

:::{div} feynman-added
| Manifold Theory                     | Neural Implementation                                          | Role                     | Section Reference |
|-------------------------------------|----------------------------------------------------------------|--------------------------|-------------------|
| **Manifold $M$**                    | Input data distribution                                        | The space to be embedded | ---                 |
| **Chart $(U_i, \phi_i)$**           | Local VQ codebook $i$                                          | Local embedding function | 7.8.3             |
| **Atlas $\mathcal{A} = \{U_i\}$**   | Attentive router + chart codebooks                             | Global coverage          | 7.8.3             |
| **Transition function $\tau_{ij}$** | Attention-weighted blending                                    | Chart overlap handling   | 7.8.3             |
| **Riemannian metric $g$**           | Orthogonality regularizer $\lVert W^TW - I\rVert^2$ (optional) | Distance preservation    | 7.7.2             |
| **Geodesic $\gamma(t)$**            | Latent space trajectory                                        | Optimal path             | 2.4               |
| **Curvature $R$**                   | Hessian of loss landscape                                      | Local complexity         | 2.5               |
| **Chart separation**                | Separation loss                                                | Chart partitioning       | 7.7.4             |
:::

(sec-self-supervised-learning-correspondences)=
### Self-Supervised Learning Correspondences

:::{div} feynman-added
| SSL Concept                 | VICReg Term                | Geometric Interpretation | Failure Mode Prevented    |
|-----------------------------|----------------------------|--------------------------|---------------------------|
| **Augmentation invariance** | $\mathcal{L}_{\text{inv}}$ | Metric tensor stability  | Sensitivity to noise      |
| **Non-collapse**            | $\mathcal{L}_{\text{var}}$ | Non-degenerate metric    | Trivial constant solution |
| **Decorrelation**           | $\mathcal{L}_{\text{cov}}$ | Coordinate independence  | Redundant dimensions      |
| **Negative sampling**       | (Not needed in VICReg)     | Contrastive boundary     | ---                         |
:::

(sec-mixture-of-experts-correspondences)=
### Mixture of Experts Correspondences

:::{div} feynman-added
| MoE Concept {cite}`jacobs1991adaptive` | Atlas Concept         | Implementation                                  |
|----------------------------------------|-----------------------|-------------------------------------------------|
| **Gating network**                     | Chart selector        | Cross-attention over chart queries              |
| **Expert networks**                    | Local charts $\phi_i$ | Chart-specific VQ codebooks                     |
| **Expert specialization**              | Chart coverage $U_i$  | Learned via separation loss                     |
| **Load balancing**                     | Atlas completeness    | Balance loss $\lVert\text{usage} - 1/K\rVert^2$ |
| **Expert capacity**                    | Chart dimension       | Latent dimension $d$                            |
:::

(sec-loss-function-decomposition)=
### Loss Function Decomposition

:::{div} feynman-prose
The Universal Loss from Section 7.7.4 can be understood geometrically. Each term enforces a specific property of the manifold embedding.
:::

The **Universal Loss** ({ref}`Section 7.7.4 <sec-the-universal-loss-functional>`) decomposes into geometric objectives, using attentive router weights $w_i(x)$ from {ref}`Section 7.8.1 <sec-theoretical-motivation-charts-as-query-vectors>`:

:::{div} feynman-added
| Loss Component                 | Geometric Objective | Manifold Property Enforced         |
|--------------------------------|---------------------|------------------------------------|
| $\mathcal{L}_{\text{inv}}$     | Metric stability    | Local isometry                     |
| $\mathcal{L}_{\text{var}}$     | Non-degeneracy      | Full rank Jacobian                 |
| $\mathcal{L}_{\text{cov}}$     | Orthonormality      | Riemannian normal coordinates      |
| $\mathcal{L}_{\text{entropy}}$ | Sharp boundaries    | Distinct chart domains             |
| $\mathcal{L}_{\text{balance}}$ | Complete coverage   | Atlas covers all of $M$            |
| $\mathcal{L}_{\text{sep}}$     | Disjoint interiors  | $U_i \cap U_j$ minimal             |
| $\mathcal{L}_{\text{orth}}$    | Isometric embedding | $\lVert Wx\rVert = \lVert x\rVert$ |
:::

(sec-when-to-use-atlas-architecture)=
### When to Use Atlas Architecture

:::{div} feynman-prose
When do you need multiple charts? It depends on the topology of your data.

If your data lives on something topologically simple (like a blob in Euclidean space), a single chart suffices. But if your data has holes, wraps around, or has disconnected components, you'll need multiple charts to cover it properly.

The Hairy Ball Theorem tells us you can't comb a sphere with a single continuous vector field---there's always a cowlick somewhere. Similarly, you can't embed a sphere into a plane with a single chart. You need at least two.
:::

:::{div} feynman-added
| Data Topology            | Single Chart | Atlas Required | Why                   |
|--------------------------|--------------|----------------|-----------------------|
| Euclidean $\mathbb{R}^n$ | Yes            | ---              | Trivially covered     |
| Sphere $S^2$             | No            | 2 or more charts      | Hairy Ball Theorem    |
| Torus $T^2$              | No            | 4 or more charts      | Non-trivial $H_1$     |
| Swiss Roll               | Yes*           | ---              | Topologically trivial |
| Disconnected components  | No            | k or more charts      | k components          |
| Mixed topology           | No            | Adaptive       | Data-dependent        |
:::

*Swiss Roll is topologically trivial but may benefit from multiple charts for geometric reasons (unrolling).

(sec-key-citations)=
### Key Citations

:::{div} feynman-added
| Concept                  | Citation                          | Contribution                                 |
|--------------------------|-----------------------------------|----------------------------------------------|
| **Manifold Atlas**       | {cite}`lee2012smooth`             | *Smooth Manifolds* textbook                  |
| **Embedding Theorem**    | {cite}`whitney1936differentiable` | Any $n$-manifold embeds in $\mathbb{R}^{2n}$ |
| **Mixture of Experts**   | {cite}`jacobs1991adaptive`        | Gated expert networks                        |
| **VICReg**               | {cite}`bardes2022vicreg`          | Collapse prevention without negatives        |
| **Barlow Twins**         | {cite}`zbontar2021barlow`         | Redundancy reduction                         |
| **InfoNCE**              | {cite}`oord2018cpc`               | Contrastive predictive coding                |
| **Information Geometry** | {cite}`saxe2019information`       | Fisher information in NNs                    |
:::
