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
