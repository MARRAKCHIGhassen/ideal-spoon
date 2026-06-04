# -*- coding: utf-8 -*-
"""
Phase-Aware Upsample Block.

This module implements the core contribution of the research: the Phase-Gated
decoder block. It leverages the complex phase information preserved in the 
scattering coefficients to recover spatial structure lost during global pooling.
"""

# Importing Global Libraries
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, Final, Callable, Optional

# Import custom libraries
from .no_second_relu_double_conv_2d_layer import NoSecondReLuDoubleConv2DLayer
from .no_second_relu_double_conv_2d_layer_sigmoid import NoSecondReLuDoubleConv2DLayer_SIGMOID
from .no_second_relu_double_conv_2d_layer_leaky import NoSecondReLuDoubleConv2DLayer_LEAKY
from .double_conv_2d_layer import DoubleConv2DLayer

from ....constants import INTERPOLATION_BILINIAR
from ....constants import SKIP_STANDARD, SKIP_MODULUS, SKIP_COMPLEX_RAW, SKIP_COMPLEX_MAG_PHASE

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# LOGGERS
# ---------------------------------------------------------------------

debug_logger = logging.getLogger(f"{__name__.split('.')[0]}.debug")
semantic_logger = logging.getLogger(f"{__name__.split('.')[0]}.semantic")


# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# IMPLEMENTATION
# ---------------------------------------------------------------------

# == PhaseAwareUpsampleBlock ==
class PhaseAwareUpsampleBlock(nn.Module):
    r"""
    Phase-Aware Gated Upsampling Block.

    This block fuses coarse semantic features (from the bottleneck) with fine-grained
    structural features (from scattering skips) using a learnable gating mechanism
    driven by phase information.

    Mechanism:
    1. **Input**: Coarse feature map ($U_{up}$) and Complex scattering coeffs ($W_j$).
    2. **Extraction**: Decompose $W_j$ into Magnitude ($A$) and Phase ($\Phi$).
    3. **Gating**: Compute a spatial gate $G = \sigma(Conv(A, \Phi))$.
    4. **Modulation**: Apply gate to coarse features: $U_{gated} = U_{up} \odot G$.
    5. **Fusion**: Concatenate and refine: $Conv([U_{gated}, A, \Phi])$.

    This allows the network to use phase discontinuities to "guide" the placement
    of semantic features during upsampling.
    """
    
    # ===========

    # -- __init__
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        L: int,
        skip_type: str = SKIP_COMPLEX_RAW,
        gate_hidden_channels: int = 32,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        inplace: bool = True,
        eps: float = 1e-6,
        interpolation: str = INTERPOLATION_BILINIAR, # "bilinear" or "nearest"
        align_corners: bool = False,
        activation: Optional[str] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.gate_hidden_channels = gate_hidden_channels
        self.base_channels = base_channels
        
        self.L = L
        self.wave_channels = self.in_channels * self.L

        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.inplace = inplace

        self.eps = eps
        self.interpolation = interpolation
        self.align_corners = align_corners

        
        # Setting up [A_mag, Phi] management
        if skip_type == SKIP_STANDARD or skip_type == SKIP_MODULUS:
            semantic_logger.error(
                f"PhaseAwareUpsampleBlock is dedicated to complex skips. use either {SKIP_COMPLEX_RAW} or {SKIP_COMPLEX_MAG_PHASE}"
            )
            debug_logger.error(
                f"PhaseAwareUpsampleBlock is dedicated to complex skips. use either {SKIP_COMPLEX_RAW} or {SKIP_COMPLEX_MAG_PHASE}"
            )
            raise ValueError(
                f"PhaseAwareUpsampleBlock is dedicated to complex skips. use either {SKIP_COMPLEX_RAW} or {SKIP_COMPLEX_MAG_PHASE}"
            )
        if skip_type == SKIP_COMPLEX_RAW:
            semantic_logger.error(
                "PhaseAwareUpsampleBlock transform to mag and phase couple set up."
            )
            debug_logger.error(
                "PhaseAwareUpsampleBlock transform to mag and phase couple set up."
            )
            self.to_mag_phase_fn = self.transform_to_mag_phase_couple
        elif skip_type == SKIP_COMPLEX_MAG_PHASE:
            self.to_mag_phase_fn = self.pass_through_mag_phase
        else:
            semantic_logger.error(
                f"Unknown skip_type: {skip_type}"
            )
            debug_logger.error(
                f"Unknown skip_type: {skip_type}"
            )
            raise ValueError(f"Unknown skip_type: {skip_type}")
        
        # Gating network: takes [A_mag, Phi] -> gate G with base_channels channels
        # The activation param controls the INTERNAL activation of the double conv,
        # not the final sigmoid (which is applied in forward).
        if activation == 'relu':
            self.gate_net = NoSecondReLuDoubleConv2DLayer(
                in_channels=2 * self.wave_channels,
                hidden_channels=self.gate_hidden_channels,
                out_channels=self.base_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                inplace=inplace,
            )
        elif activation == 'sigmoid':
            self.gate_net = NoSecondReLuDoubleConv2DLayer_SIGMOID(
                in_channels=2 * self.wave_channels,
                hidden_channels=self.gate_hidden_channels,
                out_channels=self.base_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                inplace=inplace,
            )
        elif activation == 'leaky':
            self.gate_net = NoSecondReLuDoubleConv2DLayer_LEAKY(
                in_channels=2 * self.wave_channels,
                hidden_channels=self.gate_hidden_channels,
                out_channels=self.base_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                inplace=inplace,
            )
        else:
            self.gate_net = NoSecondReLuDoubleConv2DLayer(
                in_channels=2 * self.wave_channels,
                hidden_channels=self.gate_hidden_channels,
                out_channels=self.base_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                inplace=inplace,
            )

        # Fusion conv: combines [U_gated, A_mag, Phi] -> U_j
        fuse_in_channels = self.base_channels + 2 * self.wave_channels
        self.fuse_conv = DoubleConv2DLayer(
            in_channels=fuse_in_channels,
            hidden_channels=self.base_channels,
            out_channels=self.base_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            inplace=inplace,
        )

    # -- forward
    def forward(
        self,
        bottleneck_in: torch.Tensor,  # (B, C_in, Hc, Wc)
        W_j: torch.Tensor,   # (B, C_lp, H_l, W_l)
    ) -> torch.Tensor:
        """
        Executes the Phase-Aware Fusion.

        Args:
            bottleneck_in: Features from coarser scale.
            W_j: Complex scattering coefficients (Real, Imag) or (Mag, Phase).

        Returns:
            Gated and refined feature map.
        """

        # 1. Upsample to match scattering spatial size
        B, C_in, L, Hj, Wj, two = W_j.shape
        interpolation_kwargs: Dict[str, Any] = {"mode": self.interpolation}
        if self.interpolation in ["bilinear", "bicubic", "linear", "trilinear"]:
            interpolation_kwargs["align_corners"] = self.align_corners
        bottleneck_up = F.interpolate(bottleneck_in, size=(Hj, Wj), **interpolation_kwargs)

        # 2. Split real/imag or (A, Phi) from last dim
        w_magnitude = W_j[..., 0]  # (B, C_in, L, Hj, Wj)
        w_phase = W_j[..., 1]  # (B, C_in, L, Hj, Wj)

        # Reshape to flatten orientations into channels
        w_magnitude_flat = w_magnitude.reshape(B, C_in * L, Hj, Wj)
        w_phase_flat = w_phase.reshape(B, C_in * L, Hj, Wj)

        # 3. Magnitude and phase (works whether a,b are true real/imag or already (A,Phi))
        A_mag, Phi = self.to_mag_phase_fn(w_magnitude_flat, w_phase_flat)

        # 4. Phase‑aware gating
        gate_in = torch.cat([A_mag, Phi], dim=1)  # (B, 2*C_in*L, Hj, Wj)
        G = torch.sigmoid(self.gate_net(gate_in)) # (B, base_channels, Hj, Wj)
        bottlneck_gated = bottleneck_up * G

        # 5. Fusion with magnitude and phase
        # Explicitly concatenate original structural info for refinement
        fusion_in = torch.cat([bottlneck_gated, A_mag, Phi], dim=1)
        block_out = self.fuse_conv(fusion_in)
        return block_out

    # -- transform_to_mag_phase_couple
    def transform_to_mag_phase_couple(self, a_flat: torch.Tensor, b_flat: torch.Tensor):
        """Converts Real/Imag inputs to Magnitude/Phase."""
        A_mag = torch.sqrt(a_flat * a_flat + b_flat * b_flat + self.eps)
        Phi   = torch.atan2(b_flat, a_flat)
        return A_mag, Phi
    
    # -- pass_through_mag_phase
    def pass_through_mag_phase(self, a_flat: torch.Tensor, b_flat: torch.Tensor):
        """Pass-through for inputs already in Polar form."""
        return a_flat, b_flat
