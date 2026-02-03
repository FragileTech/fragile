import math

import torch
from torch import nn
import torch.nn.functional as F


def _poincare_temperature(
    z: torch.Tensor,
    key_dim: int,
    tau_min: float,
    tau_denom_min: float,
) -> torch.Tensor:
    """Compute position-dependent temperature for Poincare ball."""
    r2 = (z**2).sum(dim=-1)
    denom = (1.0 - r2).clamp(min=tau_denom_min)
    tau = math.sqrt(key_dim) * denom / 2.0
    return tau.clamp(min=tau_min)


def _poincare_hyperbolic_score(
    z: torch.Tensor,
    centers: torch.Tensor,
    key_dim: int,
    tau_min: float,
    tau_denom_min: float,
    eps: float,
) -> torch.Tensor:
    """Compute hyperbolic distance-based scores with metric temperature."""
    z_exp = z.unsqueeze(1)  # [B, 1, D]
    c_exp = centers.unsqueeze(0)  # [1, N_c, D]
    diff = z_exp - c_exp
    dist_sq = (diff**2).sum(dim=-1)  # [B, N_c]
    z_sq = (z**2).sum(dim=-1, keepdim=True)  # [B, 1]
    c_sq = (centers**2).sum(dim=-1).unsqueeze(0)  # [1, N_c]
    denom = (1.0 - z_sq) * (1.0 - c_sq)
    arg = 1.0 + 2.0 * dist_sq / (denom + eps)
    dist = torch.acosh(arg.clamp(min=1.0 + eps))  # [B, N_c]
    tau = _poincare_temperature(z, key_dim, tau_min, tau_denom_min)
    return -dist / tau.unsqueeze(1)


class TokenSelfAttentionBlock(nn.Module):
    """Tokenized self-attention block for MLP baselines."""

    def __init__(
        self,
        hidden_dim: int,
        num_tokens: int,
        attn_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_tokens <= 0:
            msg = "num_tokens must be positive."
            raise ValueError(msg)
        if attn_dim <= 0:
            msg = "attn_dim must be positive."
            raise ValueError(msg)
        if num_heads <= 0:
            msg = "num_heads must be positive."
            raise ValueError(msg)
        if attn_dim % num_heads != 0:
            msg = "attn_dim must be divisible by num_heads."
            raise ValueError(msg)
        self.num_tokens = num_tokens
        self.attn_dim = attn_dim
        self.to_tokens = nn.Linear(hidden_dim, num_tokens * attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(attn_dim)
        self.out_proj = nn.Linear(num_tokens * attn_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project into token space, then self-attend for global interactions.
        tokens = self.to_tokens(x).reshape(x.shape[0], self.num_tokens, self.attn_dim)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attn_out = self.norm(attn_out + tokens)
        flat = attn_out.reshape(x.shape[0], self.num_tokens * self.attn_dim)
        return self.out_proj(flat) + x


class StandardVQ(nn.Module):
    """Standard Vector-Quantized VAE baseline.

    Uses a single global codebook with Euclidean distance quantization.
    This represents the typical VQ-VAE without topological awareness.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_codes: int = 64,
        use_attention: bool = False,
        attn_tokens: int = 4,
        attn_dim: int = 32,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        # Encoder: x → z_e
        encoder_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        if use_attention:
            encoder_layers.append(
                TokenSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_tokens=attn_tokens,
                    attn_dim=attn_dim,
                    num_heads=attn_heads,
                    dropout=attn_dropout,
                )
            )
        encoder_layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        ])
        self.encoder = nn.Sequential(*encoder_layers)

        # Codebook: learnable embeddings
        self.embeddings = nn.Embedding(num_codes, latent_dim)
        # Initialize uniformly
        self.embeddings.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

        # Decoder: z_q → x_recon
        decoder_layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        ]
        if use_attention:
            decoder_layers.append(
                TokenSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_tokens=attn_tokens,
                    attn_dim=attn_dim,
                    num_heads=attn_heads,
                    dropout=attn_dropout,
                )
            )
        decoder_layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with VQ.

        Returns:
            x_recon: Reconstructed input
            vq_loss: Commitment + codebook loss
            indices: Quantized code indices
        """
        # Encode
        z_e = self.encoder(x)  # [B, latent_dim]

        # Vector quantization (nearest codebook entry).
        dists = torch.cdist(z_e, self.embeddings.weight)  # [B, num_codes]
        indices = torch.argmin(dists, dim=1)  # [B]
        z_q = self.embeddings(indices)  # [B, latent_dim]

        # VQ losses
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + 0.25 * commitment_loss

        # Straight-through estimator keeps encoder gradients.
        z_out = z_e + (z_q - z_e).detach()

        # Decode
        x_recon = self.decoder(z_out)

        return x_recon, vq_loss, indices

    def compute_perplexity(self, indices: torch.Tensor) -> float:
        """Compute codebook usage perplexity."""
        num_codes = self.embeddings.num_embeddings
        counts = torch.bincount(indices, minlength=num_codes).float()
        probs = counts / counts.sum()
        # Filter zeros for log
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum()
        return math.exp(entropy.item())


class VanillaAE(nn.Module):
    """Continuous Autoencoder baseline (reconstruction upper bound).

    No discrete bottleneck - should reconstruct perfectly but
    fails to capture topology/clustering explicitly. Serves as
    the "gold standard" for reconstruction quality.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        use_attention: bool = False,
        attn_tokens: int = 4,
        attn_dim: int = 32,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        encoder_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        if use_attention:
            encoder_layers.append(
                TokenSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_tokens=attn_tokens,
                    attn_dim=attn_dim,
                    num_heads=attn_heads,
                    dropout=attn_dropout,
                )
            )
        encoder_layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        ])
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        ]
        if use_attention:
            decoder_layers.append(
                TokenSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_tokens=attn_tokens,
                    attn_dim=attn_dim,
                    num_heads=attn_heads,
                    dropout=attn_dropout,
                )
            )
        decoder_layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            x_recon: Reconstructed input
            z: Latent representation
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class AttentiveAtlasEncoder(nn.Module):
    """Attentive Atlas encoder with cross-attention routing.

    Architecture (from mermaid diagram):
    - Feature extractor: x → features [B, D]
    - Key/Value projections: features → k(x), v(x)
    - Chart Query Bank: learnable q_i [N_c, D]
    - Cross-attention Router: softmax(k @ q.T / sqrt(d)) → w_i(x)
    - Local VQ codebooks: per-chart quantization
    - Recursive decomposition: delta → (z_n, z_tex)

    Output: (K_chart, K_code, z_n, z_tex, router_weights, z_geo)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart

        # --- Shared Backbone (Feature Extractor) ---
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),  # Added - was missing
        )

        # --- Routing (Topology) ---
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        # Chart Query Bank: learnable prototypes for each manifold
        self.chart_queries = nn.Parameter(torch.randn(num_charts, hidden_dim))
        # Scale for attention
        self.scale = math.sqrt(hidden_dim)

        # --- Value (Geometry) ---
        self.val_proj = nn.Linear(hidden_dim, latent_dim)

        # --- Local VQ Codebooks (one per chart) ---
        self.codebooks = nn.ModuleList([
            nn.Embedding(codes_per_chart, latent_dim) for _ in range(num_charts)
        ])
        # Initialize codebooks
        for cb in self.codebooks:
            if hasattr(cb, "weight"):
                cb.weight.data.uniform_(-1.0 / codes_per_chart, 1.0 / codes_per_chart)

        # --- Recursive Decomposition ---
        # Structure filter: extracts structured nuisance z_n from residual
        self.structure_filter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2 if latent_dim > 2 else latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim // 2 if latent_dim > 2 else latent_dim, latent_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,  # K_chart [B]
        torch.Tensor,  # K_code [B]
        torch.Tensor,  # z_n [B, D]
        torch.Tensor,  # z_tex [B, D]
        torch.Tensor,  # router_weights [B, N_c]
        torch.Tensor,  # z_geo [B, D] (for decoder)
        torch.Tensor,  # vq_loss
        torch.Tensor,  # indices_stack [B, N_c] (for code entropy loss)
        torch.Tensor,  # z_n_all_charts [B, N_c, D] (for jump operator)
    ]:
        """Forward pass through the Attentive Atlas.

        Returns:
            K_chart: Hard chart assignment (argmax of router)
            K_code: VQ code index within selected chart
            z_n: Structured nuisance (blended from all charts)
            z_tex: Texture residual (reconstruction-only)
            router_weights: Soft routing weights [B, N_c]
            z_geo: Geometric latent (e_K + z_n) for decoder
            vq_loss: Combined VQ loss
            indices_stack: Code indices per chart [B, N_c] (for entropy loss)
            z_n_all_charts: Nuisance per chart [B, N_c, D] (for jump operator)
        """
        B = x.shape[0]
        device = x.device

        # 1. Feature extraction
        features = self.feature_extractor(x)  # [B, hidden_dim]

        # 2. Cross-attention routing
        k = self.key_proj(features)  # [B, hidden_dim]
        # Attention: k @ q.T / sqrt(d)
        scores = torch.matmul(k, self.chart_queries.T) / self.scale  # [B, N_c]
        # Router weights define atlas chart responsibilities.
        router_weights = F.softmax(scores, dim=-1)  # [B, N_c]

        # Hard chart assignment
        K_chart = torch.argmax(router_weights, dim=1)  # [B]

        # 3. Value projection
        v = self.val_proj(features)  # [B, latent_dim]

        # 4. Local VQ per chart (chart-specific macro codes).
        codebook_weights = torch.stack(
            [cb.weight for cb in self.codebooks], dim=0
        )  # [N_c, codes_per_chart, latent_dim]
        v_exp = v.unsqueeze(0).expand(self.num_charts, -1, -1)  # [N_c, B, D]
        dists = torch.cdist(v_exp, codebook_weights)  # [N_c, B, codes_per_chart]
        indices = torch.argmin(dists, dim=-1)  # [N_c, B]
        indices_stack = indices.transpose(0, 1)  # [B, N_c]

        z_q_all = torch.gather(
            codebook_weights,
            1,
            indices.unsqueeze(-1).expand(-1, -1, self.latent_dim),
        ).transpose(0, 1)  # [B, N_c, D]

        w = router_weights.unsqueeze(-1).detach()  # [B, N_c, 1]
        v_bc = v.unsqueeze(1)  # [B, 1, D]
        commitment = ((v_bc - z_q_all.detach()) ** 2 * w).mean(dim=(0, 2)).sum()
        codebook = ((z_q_all - v_bc.detach()) ** 2 * w).mean(dim=(0, 2)).sum()
        vq_loss = codebook + 0.25 * commitment

        # 5. Soft blending for differentiability across charts.
        z_q_blended = (z_q_all * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # Get hard K_code from selected chart
        K_code = indices_stack[torch.arange(B, device=device), K_chart]  # [B]

        # 6. Recursive Decomposition - compute z_n per chart for jump operator
        # This allows learning chart transitions on the nuisance coordinates
        delta = v_bc - z_q_all.detach()
        z_n_all_charts = self.structure_filter(
            delta.reshape(B * self.num_charts, self.latent_dim)
        ).view(B, self.num_charts, self.latent_dim)

        # Blend z_n weighted by router (gauge residual).
        z_n = (z_n_all_charts * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, latent_dim]

        # Texture residual: z_tex = delta_blended - z_n
        delta_blended = v - z_q_blended.detach()
        z_tex = delta_blended - z_n  # [B, latent_dim]

        # 7. Geometric latent for decoder: z_geo = e_K + z_n
        # Use straight-through for z_q
        z_q_st = v + (z_q_blended - v).detach()
        z_geo = z_q_st + z_n  # [B, latent_dim]

        return (
            K_chart,
            K_code,
            z_n,
            z_tex,
            router_weights,
            z_geo,
            vq_loss,
            indices_stack,
            z_n_all_charts,
        )


class FactorizedJumpOperator(nn.Module):
    """Learns transition functions between atlas charts.

    Implements L_{i->j}(z) = A_j(B_i z + c_i) + d_j
    via low-rank bottleneck (global tangent space).

    This enforces transitive consistency by construction:
    τ_ik = τ_jk ∘ τ_ij

    The factorization uses the Whitney Embedding hypothesis:
    - E_i: Maps Chart i local coords → Global Canonical coords
    - D_j: Maps Global Canonical coords → Chart j local coords

    Reference: Section 7.11 (Topological Gluing)
    """

    def __init__(self, num_charts: int, latent_dim: int, global_rank: int = 0):
        """Initialize the Jump Operator.

        Args:
            num_charts: Number of atlas charts
            latent_dim: Dimension of local coordinates (z_n)
            global_rank: Rank of global tangent space (0 = use latent_dim)
        """
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.rank = global_rank if global_rank > 0 else latent_dim

        # Encoder: Chart_i -> Global
        # B_i [N_c, rank, latent_dim]
        self.B = nn.Parameter(torch.randn(num_charts, self.rank, latent_dim))
        # c_i [N_c, rank]
        self.c = nn.Parameter(torch.zeros(num_charts, self.rank))

        # Decoder: Global -> Chart_j
        # A_j [N_c, latent_dim, rank]
        self.A = nn.Parameter(torch.randn(num_charts, latent_dim, self.rank))
        # d_j [N_c, latent_dim]
        self.d = nn.Parameter(torch.zeros(num_charts, latent_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights near identity for stability."""
        with torch.no_grad():
            # Init B matrices to approximate identity
            for i in range(self.num_charts):
                if self.rank <= self.latent_dim:
                    self.B.data[i] = torch.eye(self.rank, self.latent_dim)
                else:
                    self.B.data[i, : self.latent_dim, :] = torch.eye(self.latent_dim)

            # Init A matrices to approximate identity
            for i in range(self.num_charts):
                if self.latent_dim <= self.rank:
                    self.A.data[i] = torch.eye(self.latent_dim, self.rank)
                else:
                    self.A.data[i, :, : self.latent_dim] = torch.eye(self.latent_dim)

            # Add small noise for symmetry breaking
            self.B.data += torch.randn_like(self.B) * 0.01
            self.A.data += torch.randn_like(self.A) * 0.01

    def forward(
        self,
        z_n: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Jump Operator: z_target = L_{source->target}(z_source).

        Args:
            z_n: [B, D] nuisance coordinates in source chart
            source_idx: [B] index of source chart
            target_idx: [B] index of target chart

        Returns:
            z_out: [B, D] nuisance coordinates in target chart
        """
        # 1. Lift Source -> Global: z_global = B_src @ z + c_src
        B_src = self.B[source_idx]  # [B, rank, D]
        c_src = self.c[source_idx]  # [B, rank]
        z_global = torch.bmm(B_src, z_n.unsqueeze(-1)).squeeze(-1) + c_src  # [B, rank]

        # 2. Project Global -> Target: z_out = A_tgt @ z_global + d_tgt
        A_tgt = self.A[target_idx]  # [B, D, rank]
        d_tgt = self.d[target_idx]  # [B, D]
        return torch.bmm(A_tgt, z_global.unsqueeze(-1)).squeeze(-1) + d_tgt  # [B, D]

    def get_transition_matrix(self, source: int, target: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the full affine transition matrix from source to target.

        Returns (M, b) where z_target = M @ z_source + b
        """
        # M = A_target @ B_source
        M = self.A[target] @ self.B[source]  # [D, D]
        # b = A_target @ c_source + d_target
        b = self.A[target] @ self.c[source] + self.d[target]  # [D]
        return M, b


class TopologicalDecoder(nn.Module):
    """Topological Decoder (Inverse Atlas) from Section 7.10.

    The inverse atlas - decodes chart-local geometry back to observation space.
    **Autonomous**: Can infer routing from geometry alone during dreaming,
    or accept a discrete chart index during planning.

    Architecture:
    - Chart projectors: z_geo → h_i (one per chart)
    - Inverse router: z_geo → w_soft (infers routing from geometry)
    - One-hot: K_chart → w_hard (optional hard routing)
    - Chart blend: h_global = sum(w_i * h_i)
    - Texture projector: z_tex → h_tex
    - Add: h_total = h_global + h_tex
    - Renderer: h_total → x_hat

    Routing modes:
    - Discrete planning: provide chart_index, use one-hot hard routing
    - Continuous generation: omit chart_index, infer weights from z_geo
    """

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        num_charts: int = 3,
        output_dim: int = 2,
        tau_min: float = 1e-2,
        tau_denom_min: float = 1e-3,
        transport_eps: float = 1e-3,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.router_tau_min = tau_min
        self.router_tau_denom_min = tau_denom_min
        self.router_transport_eps = transport_eps

        # Chart-specific projectors (one per chart)
        self.chart_projectors = nn.ModuleList([
            nn.Linear(latent_dim, hidden_dim) for _ in range(num_charts)
        ])

        # Inverse router correction (dreaming mode)
        self.latent_router = nn.Linear(latent_dim, num_charts)

        # Chart centers for hyperbolic routing
        self.chart_centers = nn.Parameter(torch.randn(num_charts, latent_dim) * 0.02)

        # Texture projector (global)
        self.tex_projector = nn.Linear(latent_dim, hidden_dim)

        # Shared renderer
        self.renderer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        z_geo: torch.Tensor,
        z_tex: torch.Tensor,
        chart_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode from latent components.

        Args:
            z_geo: [B, D] geometric content (e_K + z_n)
            z_tex: [B, D] texture residual
            chart_index: [B] optional chart IDs for hard routing.
                         If None, infers routing from z_geo (dreaming mode).

        Returns:
            x_hat: [B, output_dim] reconstructed observation
            router_weights: [B, N_c] the routing weights used (for consistency loss)
        """
        # Determine routing weights
        if chart_index is not None:
            # Discrete planning mode: hard one-hot routing
            router_weights = F.one_hot(chart_index, num_classes=self.num_charts).float()
        else:
            # Continuous generation / dreaming mode: hyperbolic routing from geometry
            scores = _poincare_hyperbolic_score(
                z_geo,
                self.chart_centers,
                key_dim=self.latent_dim,
                tau_min=self.router_tau_min,
                tau_denom_min=self.router_tau_denom_min,
                eps=self.router_transport_eps,
            )
            tau = _poincare_temperature(
                z_geo,
                key_dim=self.latent_dim,
                tau_min=self.router_tau_min,
                tau_denom_min=self.router_tau_denom_min,
            )
            scores = scores + 0.1 * self.latent_router(z_geo) / tau.unsqueeze(1)
            router_weights = F.softmax(scores, dim=-1)

        # Project through each chart using einsum for proper broadcasting
        # weights[c] is [hidden_dim, latent_dim] for chart c
        # We compute z_geo @ weights[c].T + biases[c] for each chart
        weights = torch.stack([proj.weight for proj in self.chart_projectors], dim=0)
        biases = torch.stack([proj.bias for proj in self.chart_projectors], dim=0)
        # einsum: 'bl,clh->bch' means z_geo[b,l] @ weights[c,l,h] -> h_stack[b,c,h]
        # But weights is [C, hidden, latent], so we need to transpose
        h_stack = torch.einsum("bl,chl->bch", z_geo, weights)
        h_stack += biases.unsqueeze(0)  # [B, N_c, hidden_dim]

        # Blend using router weights
        h_global = (h_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden]

        # Add texture
        h_tex = self.tex_projector(z_tex)
        h_total = h_global + h_tex

        # Render to output
        x_hat = self.renderer(h_total)

        return x_hat, router_weights


class TopoEncoder(nn.Module):
    """Complete TopoEncoder: Encoder + Autonomous Decoder.

    Combines AttentiveAtlasEncoder and TopologicalDecoder.

    The decoder is autonomous: it can infer routing from geometry alone
    (dreaming mode) or use explicit chart indices (planning mode).

    Training modes:
    - use_hard_routing=True: Decoder uses K_chart from encoder (planning)
    - use_hard_routing=False: Decoder infers routing from z_geo (dreaming)

    Consistency loss aligns encoder and decoder routing distributions.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
    ):
        super().__init__()
        self.num_charts = num_charts

        self.encoder = AttentiveAtlasEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_charts=num_charts,
            codes_per_chart=codes_per_chart,
        )

        self.decoder = TopologicalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_charts=num_charts,
            output_dim=input_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_hard_routing: bool = False,
    ) -> tuple[
        torch.Tensor,  # x_recon
        torch.Tensor,  # vq_loss
        torch.Tensor,  # enc_router_weights (from encoder)
        torch.Tensor,  # dec_router_weights (from decoder)
        torch.Tensor,  # K_chart
        torch.Tensor,  # z_geo
        torch.Tensor,  # z_n
        torch.Tensor,  # c_bar
    ]:
        """Full forward pass.

        Args:
            x: Input tensor [B, D_in]
            use_hard_routing: If True, decoder uses K_chart (planning mode).
                              If False, decoder infers routing from z_geo (dreaming).

        Returns:
            x_recon: Reconstructed input
            vq_loss: VQ commitment + codebook loss
            enc_router_weights: Encoder routing weights (for entropy loss)
            dec_router_weights: Decoder routing weights (for consistency loss)
            K_chart: Hard chart assignments from encoder
            z_geo: Geometric latent (macro + gauge residual)
            z_n: Nuisance latent (continuous gauge vector)
            c_bar: Chart center mixture
        """
        (
            K_chart,
            _K_code,
            z_n,
            z_tex,
            enc_router_weights,
            z_geo,
            vq_loss,
            _indices,
            _z_n_all,
            c_bar,
        ) = self.encoder(x)

        # Decoder can use hard routing (planning) or infer from geometry (dreaming)
        chart_index = K_chart if use_hard_routing else None
        x_recon, dec_router_weights = self.decoder(z_geo, z_tex, chart_index)

        return x_recon, vq_loss, enc_router_weights, dec_router_weights, K_chart, z_geo, z_n, c_bar

    def compute_consistency_loss(
        self,
        enc_weights: torch.Tensor,
        dec_weights: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute KL divergence between encoder and decoder routing.

        Keeps the inverse router aligned with the encoder routing.
        L_consistency = KL(w_enc || w_dec)
        """
        # KL(P || Q) = sum(P * log(P/Q))
        kl = (enc_weights * torch.log((enc_weights + eps) / (dec_weights + eps))).sum(dim=-1)
        return kl.mean()

    def compute_perplexity(self, K_chart: torch.Tensor) -> float:
        """Compute chart usage perplexity."""
        counts = torch.bincount(K_chart, minlength=self.num_charts).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum()
        return math.exp(entropy.item())
