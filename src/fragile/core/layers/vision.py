from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from fragile.core.layers.primitives import SpectralLinear
from fragile.core.layers.ugn import SoftEquivariantLayer


try:  # Optional dependency
    from e2cnn import gspaces, nn as e2nn

    _E2CNN_AVAILABLE = True
except Exception:  # pragma: no cover - optional import
    gspaces = None
    e2nn = None
    _E2CNN_AVAILABLE = False


class CovariantRetina(nn.Module):
    """SO(2)-equivariant vision encoder using steerable convolutions (E2CNN).

    Args:
        in_channels: Input channel count.
        out_dim: Output feature dimension.
        num_rotations: Discretization of SO(2).
        kernel_size: Convolution kernel size.
        use_reflections: Include reflections (O(2)) if True; otherwise SO(2) only.
        norm_nonlinearity: Norm nonlinearity for steerable layers. Use "n_sigmoid" for
            smooth gating, or "n_relu" for the non-smooth default. "squash" is also supported.
        norm_bias: Whether to include a learned bias in the norm nonlinearity. Must be False
            when using "squash".
        spectral_n_power_iterations: Power-iteration steps for spectral normalization.
        spectral_eps: Numerical epsilon for spectral normalization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 512,
        num_rotations: int = 8,
        kernel_size: int = 5,
        use_reflections: bool = False,
        norm_nonlinearity: str = "n_sigmoid",
        norm_bias: bool = True,
        spectral_n_power_iterations: int = 3,
        spectral_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if not _E2CNN_AVAILABLE:
            msg = (
                "CovariantRetina requires the optional dependency 'e2cnn'. "
                "Install it to use this layer."
            )
            raise ImportError(msg)

        self._e2nn: Any = e2nn
        self._gspaces: Any = gspaces
        self.out_dim = out_dim

        if use_reflections:
            self.r2_act = gspaces.FlipRot2dOnR2(N=num_rotations)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=num_rotations)

        in_type = e2nn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])

        self.feature_type_32 = e2nn.FieldType(
            self.r2_act,
            32 * [self.r2_act.regular_repr],
        )
        self.feature_type_64 = e2nn.FieldType(
            self.r2_act,
            64 * [self.r2_act.regular_repr],
        )

        self.lift = e2nn.R2Conv(
            in_type,
            self.feature_type_32,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.conv1 = e2nn.R2Conv(
            self.feature_type_32,
            self.feature_type_64,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.conv2 = e2nn.R2Conv(
            self.feature_type_64,
            self.feature_type_64,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        if norm_nonlinearity == "squash" and norm_bias:
            msg = 'norm_bias must be False when norm_nonlinearity="squash".'
            raise ValueError(msg)

        self.relu1 = e2nn.NormNonLinearity(
            self.feature_type_32,
            function=norm_nonlinearity,
            bias=norm_bias,
        )
        self.relu2 = e2nn.NormNonLinearity(
            self.feature_type_64,
            function=norm_nonlinearity,
            bias=norm_bias,
        )
        self.relu3 = e2nn.NormNonLinearity(
            self.feature_type_64,
            function=norm_nonlinearity,
            bias=norm_bias,
        )
        self.group_pool = e2nn.GroupPooling(self.feature_type_64)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = SpectralLinear(
            64 * 4 * 4,
            out_dim,
            bias=False,
            n_power_iterations=spectral_n_power_iterations,
            eps=spectral_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e2nn = self._e2nn
        # Lift input into a steerable (equivariant) field.
        x = e2nn.GeometricTensor(
            x,
            e2nn.FieldType(self.r2_act, x.shape[1] * [self.r2_act.trivial_repr]),
        )

        # Equivariant feature extraction in the SO(2)/O(2) field space.
        x = self.lift(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.relu3(x)

        # Pool over the group to obtain rotation-invariant features.
        x = self.group_pool(x)
        x = x.tensor
        x = self.spatial_pool(x)
        x = x.flatten(1)
        return self.fc(x)


class CovariantRetinaDecoder(nn.Module):
    """SO(2)-equivariant vision decoder using steerable convolutions (E2CNN)."""

    def __init__(
        self,
        in_dim: int,
        out_channels: int,
        out_height: int,
        out_width: int,
        num_rotations: int = 8,
        kernel_size: int = 5,
        use_reflections: bool = False,
        norm_nonlinearity: str = "n_sigmoid",
        norm_bias: bool = True,
        base_size: int = 4,
    ) -> None:
        super().__init__()
        if not _E2CNN_AVAILABLE:
            msg = (
                "CovariantRetinaDecoder requires the optional dependency 'e2cnn'. "
                "Install it to use this layer."
            )
            raise ImportError(msg)
        if in_dim <= 0:
            msg = "in_dim must be positive."
            raise ValueError(msg)
        if out_channels <= 0 or out_height <= 0 or out_width <= 0:
            msg = "out_channels and spatial dimensions must be positive."
            raise ValueError(msg)
        if base_size <= 0:
            msg = "base_size must be positive."
            raise ValueError(msg)
        if norm_nonlinearity == "squash" and norm_bias:
            msg = 'norm_bias must be False when norm_nonlinearity="squash".'
            raise ValueError(msg)

        self._e2nn: Any = e2nn
        self._gspaces: Any = gspaces
        self.out_channels = out_channels
        self.out_height = out_height
        self.out_width = out_width
        self.base_size = base_size

        if use_reflections:
            self.r2_act = gspaces.FlipRot2dOnR2(N=num_rotations)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=num_rotations)

        self.feature_type_32 = e2nn.FieldType(
            self.r2_act,
            32 * [self.r2_act.regular_repr],
        )
        self.feature_type_64 = e2nn.FieldType(
            self.r2_act,
            64 * [self.r2_act.regular_repr],
        )
        self.out_type = e2nn.FieldType(
            self.r2_act,
            out_channels * [self.r2_act.trivial_repr],
        )

        self.fc = SpectralLinear(
            in_dim,
            self.feature_type_64.size * base_size * base_size,
            bias=False,
        )

        self.conv1 = e2nn.R2Conv(
            self.feature_type_64,
            self.feature_type_64,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.relu1 = e2nn.NormNonLinearity(
            self.feature_type_64,
            function=norm_nonlinearity,
            bias=norm_bias,
        )
        self.up1 = e2nn.R2Upsampling(self.feature_type_64, scale_factor=2, mode="bilinear")

        self.conv2 = e2nn.R2Conv(
            self.feature_type_64,
            self.feature_type_32,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.relu2 = e2nn.NormNonLinearity(
            self.feature_type_32,
            function=norm_nonlinearity,
            bias=norm_bias,
        )
        self.up2 = e2nn.R2Upsampling(self.feature_type_32, scale_factor=2, mode="bilinear")

        self.conv3 = e2nn.R2Conv(
            self.feature_type_32,
            self.feature_type_32,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.relu3 = e2nn.NormNonLinearity(
            self.feature_type_32,
            function=norm_nonlinearity,
            bias=norm_bias,
        )
        self.up3 = e2nn.R2Upsampling(
            self.feature_type_32,
            size=(out_height, out_width),
            mode="bilinear",
        )

        self.out_conv = e2nn.R2Conv(
            self.feature_type_32,
            self.out_type,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        e2nn = self._e2nn
        # Expand latent code into an equivariant feature field.
        h = self.fc(z)
        h = h.view(z.shape[0], self.feature_type_64.size, self.base_size, self.base_size)
        x = e2nn.GeometricTensor(h, self.feature_type_64)

        # Progressive equivariant upsampling to target resolution.
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.up3(x)

        return self.out_conv(x).tensor


# =============================================================================
# Gauge-Covariant CIFAR Backbone (Theory-Aligned CNN Alternative)
# =============================================================================


class SpectralConv2d(nn.Module):
    """Conv2d with spectral normalization (Lipschitz ≤ 1).

    Mirrors SpectralLinear but for spatial convolutions. Ensures the operator
    is non-expansive, preserving light-cone causality in the feature hierarchy.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        bias: Whether to include bias.
        n_power_iterations: Power iterations for spectral norm estimation.
        eps: Numerical epsilon.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        n_power_iterations: int = 3,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # For spectral norm: reshape weight to [out, in*k*k] and track u vector
        fan_out = out_channels
        self.register_buffer("_u", F.normalize(torch.randn(fan_out), dim=0))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def _spectral_normalized_weight(self, update_u: bool = True) -> torch.Tensor:
        # Reshape to [out_channels, in_channels * k * k]
        weight_mat = self.weight.reshape(self.out_channels, -1)
        u = self._u

        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(weight_mat.t() @ u, dim=0, eps=self.eps)
                u = F.normalize(weight_mat @ v, dim=0, eps=self.eps)
            if update_u:
                self._u.copy_(u)

        sigma = (u @ weight_mat @ v).abs()
        # Only normalize if σ > 1 (non-expansive, not contractive)
        return self.weight / sigma.clamp(min=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._spectral_normalized_weight(update_u=self.training)
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)


class NormGatedConv2d(nn.Module):
    """Gauge-covariant conv block: SpectralConv2d → NormGatedGELU.

    Replaces standard Conv2d + BatchNorm + ReLU with:
    - SpectralConv2d: Lipschitz-bounded convolution (σ_max ≤ 1)
    - NormGatedGELU: SO(d_b)-equivariant activation on channel bundles

    The NormGate treats each spatial position independently, with channels
    grouped into bundles. The gate decision is based on bundle energy (norm),
    preserving rotational equivariance in feature space.

    Theory alignment:
    - SpectralConv2d enforces Lipschitz bound (Def. def-spectral-linear)
    - NormGate is SO(d_b)-equivariant (Def. def-norm-gated-activation)
    - No batch statistics → gauge-covariant (unlike BatchNorm)

    Args:
        in_channels: Input channels.
        out_channels: Output channels (must be divisible by bundle_size).
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        bundle_size: Size of each gauge bundle for NormGate.
        smooth_norm_eps: Epsilon for smooth norm computation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bundle_size: int = 4,
        smooth_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if out_channels % bundle_size != 0:
            msg = (
                f"out_channels ({out_channels}) must be divisible by bundle_size ({bundle_size})."
            )
            raise ValueError(msg)

        self.conv = SpectralConv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.n_bundles = out_channels // bundle_size
        self.bundle_size = bundle_size
        self.smooth_norm_eps = smooth_norm_eps

        # Learnable bias for norm gating (one per bundle)
        self.norm_bias = nn.Parameter(torch.zeros(1, self.n_bundles, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        x = self.conv(x)  # [B, C_out, H, W]

        B, C, H, W = x.shape
        # Reshape to [B, n_bundles, bundle_size, H, W]
        x = x.reshape(B, self.n_bundles, self.bundle_size, H, W)

        # Compute bundle energy (norm over bundle_size dimension)
        energy = torch.sqrt((x**2).sum(dim=2, keepdim=True) + self.smooth_norm_eps**2)

        # Apply GELU gate based on energy
        gate = F.gelu(energy + self.norm_bias)
        x = x * gate

        # Reshape back to [B, C_out, H, W]
        return x.reshape(B, C, H, W)


def _init_soft_equiv_layer(layer: SoftEquivariantLayer) -> None:
    """Initialize soft-equivariant layer to minimize cross-bundle mixing."""
    with torch.no_grad():
        if isinstance(layer.mixing_weights, torch.Tensor):
            layer.mixing_weights.zero_()
        else:
            for row in layer.mixing_weights:
                for weight in row:
                    weight.zero_()


class SoftEquivariantConvMix(nn.Module):
    """Apply soft-equivariant bundle mixing to feature maps."""

    def __init__(
        self,
        channels: int,
        bundle_size: int,
        hidden_dim: int = 64,
        use_spectral_norm: bool = True,
        zero_self_mixing: bool = False,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if channels <= 0:
            msg = "channels must be positive."
            raise ValueError(msg)
        if bundle_size <= 0:
            msg = "bundle_size must be positive."
            raise ValueError(msg)
        if channels % bundle_size != 0:
            msg = "channels must be divisible by bundle_size."
            raise ValueError(msg)
        if alpha < 0.0 or alpha > 1.0:
            msg = "alpha must be in [0, 1]."
            raise ValueError(msg)

        n_bundles = channels // bundle_size
        self.layer = SoftEquivariantLayer(
            n_bundles=n_bundles,
            bundle_dim=bundle_size,
            hidden_dim=hidden_dim,
            use_spectral_norm=use_spectral_norm,
            zero_self_mixing=zero_self_mixing,
        )
        _init_soft_equiv_layer(self.layer)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            msg = "Expected input shape [B, C, H, W]."
            raise ValueError(msg)
        b, c, h, w = x.shape
        z = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        z_mixed = self.layer(z)
        z = (1.0 - self.alpha) * z + self.alpha * z_mixed
        return z.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class StandardConvBlock(nn.Module):
    """Non-covariant CNN block for symmetry breaking."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            msg = "channels must be positive."
            raise ValueError(msg)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CovariantCIFARBackbone(nn.Module):
    """Gauge-covariant vision backbone for CIFAR-10 benchmarking.

    Mirrors standard "tiny CNN" architectures but replaces:
    - Conv2d + BatchNorm + ReLU → NormGatedConv2d (SpectralConv + NormGate)

    Design rationale (matching mainstream CNNs):
    - Channels increase as spatial resolution decreases: base → 2x → 4x
    - Two conv blocks per stage, MaxPool between stages
    - Global average pooling before final linear

    Theory alignment vs standard CNN:

    | Standard CNN          | CovariantCIFARBackbone      | Theoretical Basis            |
    |-----------------------|-----------------------------|------------------------------|
    | Conv2d                | SpectralConv2d              | Lipschitz bound (σ ≤ 1)      |
    | BatchNorm             | (none)                      | Gauge covariance preserved   |
    | ReLU                  | NormGatedGELU               | SO(d_b) equivariance         |
    | Free parameters       | Spectral-constrained        | Light-cone causality         |

    Architecture variants (parameter counts match standard tiny CNNs):
    - Tiny (base=16):  16→32→64,   ~36k params  → targets ~60% accuracy
    - Small (base=32): 32→64→128,  ~142k params → comfortable 60%+
    - Medium (base=64): 64→128→256, ~565k params → 70%+ territory

    Args:
        in_channels: Input channels (3 for CIFAR RGB).
        num_classes: Output classes (10 for CIFAR-10).
        base_channels: Starting channel width (16, 32, or 64 recommended).
        bundle_size: Bundle size for NormGated activations.
        use_spectral_fc: Use SpectralLinear for final classifier.

    Example:
        >>> model = CovariantCIFARBackbone(3, 10, base_channels=16)  # ~36k params
        >>> x = torch.randn(2, 3, 32, 32)
        >>> logits = model(x)  # [2, 10]
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 16,
        bundle_size: int = 4,
        use_spectral_fc: bool = True,
        soft_equiv_per_block: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_alpha: float = 0.1,
        standard_head: bool = False,
        standard_head_blocks: int = 2,
    ) -> None:
        super().__init__()
        if base_channels % bundle_size != 0:
            msg = f"base_channels ({base_channels}) must be divisible by bundle_size ({bundle_size})."
            raise ValueError(msg)

        c1 = base_channels  # Stage 1: 32x32 → 16x16
        c2 = base_channels * 2  # Stage 2: 16x16 → 8x8
        c3 = base_channels * 4  # Stage 3: 8x8 → 4x4

        # Stage 1: [B, 3, 32, 32] → [B, c1, 16, 16]
        self.stage1 = nn.Sequential(
            NormGatedConv2d(in_channels, c1, kernel_size=3, padding=1, bundle_size=bundle_size),
            NormGatedConv2d(c1, c1, kernel_size=3, padding=1, bundle_size=bundle_size),
            nn.MaxPool2d(2, 2),
        )

        # Stage 2: [B, c1, 16, 16] → [B, c2, 8, 8]
        self.stage2 = nn.Sequential(
            NormGatedConv2d(c1, c2, kernel_size=3, padding=1, bundle_size=bundle_size),
            NormGatedConv2d(c2, c2, kernel_size=3, padding=1, bundle_size=bundle_size),
            nn.MaxPool2d(2, 2),
        )

        # Stage 3: [B, c2, 8, 8] → [B, c3, 4, 4]
        self.stage3 = nn.Sequential(
            NormGatedConv2d(c2, c3, kernel_size=3, padding=1, bundle_size=bundle_size),
            NormGatedConv2d(c3, c3, kernel_size=3, padding=1, bundle_size=bundle_size),
            nn.MaxPool2d(2, 2),
        )

        # Global average pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if use_spectral_fc:
            self.fc = SpectralLinear(c3, num_classes, bias=True)
        else:
            self.fc = nn.Linear(c3, num_classes)

        self._num_features = c3
        self.soft_equiv1 = None
        self.soft_equiv2 = None
        self.soft_equiv3 = None
        self.standard_head = None
        if soft_equiv_per_block:
            mix_bundle_size = soft_equiv_bundle_size or bundle_size
            self.soft_equiv1 = SoftEquivariantConvMix(
                channels=c1,
                bundle_size=mix_bundle_size,
                hidden_dim=soft_equiv_hidden_dim,
                use_spectral_norm=soft_equiv_use_spectral_norm,
                zero_self_mixing=soft_equiv_zero_self_mixing,
                alpha=soft_equiv_alpha,
            )
            self.soft_equiv2 = SoftEquivariantConvMix(
                channels=c2,
                bundle_size=mix_bundle_size,
                hidden_dim=soft_equiv_hidden_dim,
                use_spectral_norm=soft_equiv_use_spectral_norm,
                zero_self_mixing=soft_equiv_zero_self_mixing,
                alpha=soft_equiv_alpha,
            )
            self.soft_equiv3 = SoftEquivariantConvMix(
                channels=c3,
                bundle_size=mix_bundle_size,
                hidden_dim=soft_equiv_hidden_dim,
                use_spectral_norm=soft_equiv_use_spectral_norm,
                zero_self_mixing=soft_equiv_zero_self_mixing,
                alpha=soft_equiv_alpha,
            )
        if standard_head:
            if standard_head_blocks <= 0:
                msg = "standard_head_blocks must be positive when standard_head is enabled."
                raise ValueError(msg)
            self.standard_head = nn.Sequential(*[
                StandardConvBlock(c3) for _ in range(int(standard_head_blocks))
            ])

    @property
    def num_features(self) -> int:
        """Feature dimension before classifier."""
        return self._num_features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        x = self.stage1(x)
        if self.soft_equiv1 is not None:
            x = self.soft_equiv1(x)
        x = self.stage2(x)
        if self.soft_equiv2 is not None:
            x = self.soft_equiv2(x)
        x = self.stage3(x)
        if self.soft_equiv3 is not None:
            x = self.soft_equiv3(x)
        if self.standard_head is not None:
            x = self.standard_head(x)
        x = self.pool(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.fc(features)


class StandardCIFARBackbone(nn.Module):
    """Standard tiny CNN baseline for fair comparison with CovariantCIFARBackbone.

    Uses conventional Conv2d + BatchNorm + ReLU with the same architecture
    (channels, stages, pooling) as CovariantCIFARBackbone.

    This is the "mainstream ML" reference point for benchmarking.

    Args:
        in_channels: Input channels (3 for CIFAR RGB).
        num_classes: Output classes (10 for CIFAR-10).
        base_channels: Starting channel width (16, 32, or 64 recommended).

    Example:
        >>> model = StandardCIFARBackbone(3, 10, base_channels=16)  # ~36k params
        >>> x = torch.randn(2, 3, 32, 32)
        >>> logits = model(x)  # [2, 10]
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Stage 1: [B, 3, 32, 32] → [B, c1, 16, 16]
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Stage 2: [B, c1, 16, 16] → [B, c2, 8, 8]
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Stage 3: [B, c2, 8, 8] → [B, c3, 4, 4]
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, num_classes)
        self._num_features = c3

    @property
    def num_features(self) -> int:
        """Feature dimension before classifier."""
        return self._num_features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.fc(features)


class _BasicResBlock(nn.Module):
    """Minimal residual block for small-image backbones."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.act(out)


class StandardResNetBackbone(nn.Module):
    """Standard ResNet-style backbone for baseline feature extraction."""

    def __init__(self, in_channels: int, out_dim: int, base_channels: int = 32) -> None:
        super().__init__()
        if in_channels <= 0:
            msg = "in_channels must be positive."
            raise ValueError(msg)
        if out_dim <= 0:
            msg = "out_dim must be positive."
            raise ValueError(msg)
        if base_channels <= 0:
            msg = "base_channels must be positive."
            raise ValueError(msg)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, out_dim)

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [_BasicResBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(_BasicResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class StandardResNetDecoder(nn.Module):
    """Standard decoder for baseline vision reconstruction."""

    def __init__(
        self,
        in_dim: int,
        out_channels: int,
        out_height: int,
        out_width: int,
        base_channels: int = 32,
        base_size: int = 4,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            msg = "in_dim must be positive."
            raise ValueError(msg)
        if out_channels <= 0 or out_height <= 0 or out_width <= 0:
            msg = "out_channels and spatial dimensions must be positive."
            raise ValueError(msg)
        if base_channels <= 0:
            msg = "base_channels must be positive."
            raise ValueError(msg)
        if base_size <= 0:
            msg = "base_size must be positive."
            raise ValueError(msg)

        self.out_height = out_height
        self.out_width = out_width
        self.out_channels = out_channels
        self.base_size = base_size

        hidden_channels = base_channels * 4
        self.hidden_channels = hidden_channels
        self.fc = nn.Linear(in_dim, hidden_channels * base_size * base_size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels, base_channels * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.shape[0], self.hidden_channels, self.base_size, self.base_size)
        x = self.deconv(h)
        if x.shape[2] != self.out_height or x.shape[3] != self.out_width:
            x = F.interpolate(
                x,
                size=(self.out_height, self.out_width),
                mode="bilinear",
                align_corners=False,
            )
        return x
