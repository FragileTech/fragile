from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from fragile.core.layers.primitives import SpectralLinear

try:  # Optional dependency
    from e2cnn import gspaces
    from e2cnn import nn as e2nn

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
            raise ImportError(
                "CovariantRetina requires the optional dependency 'e2cnn'. "
                "Install it to use this layer."
            )

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
            raise ValueError('norm_bias must be False when norm_nonlinearity="squash".')

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
        x = e2nn.GeometricTensor(
            x,
            e2nn.FieldType(self.r2_act, x.shape[1] * [self.r2_act.trivial_repr]),
        )

        x = self.lift(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.relu3(x)

        x = self.group_pool(x)
        x = x.tensor
        x = self.spatial_pool(x)
        x = x.flatten(1)
        return self.fc(x)


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
        out = out + identity
        return self.act(out)


class StandardResNetBackbone(nn.Module):
    """Standard ResNet-style backbone for baseline feature extraction."""

    def __init__(self, in_channels: int, out_dim: int, base_channels: int = 32) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if out_dim <= 0:
            raise ValueError("out_dim must be positive.")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(
            base_channels, base_channels * 2, blocks=2, stride=2
        )
        self.layer3 = self._make_layer(
            base_channels * 2, base_channels * 4, blocks=2, stride=2
        )
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
