# -*- coding: utf-8 -*-
"""
Modulus Upsample Block.

This module implements the standard fusion block for scattering-based U-Nets.
It aligns with the baseline approach where only the magnitude (Modulus) of the 
scattering coefficients is used, ensuring translation invariance but discarding 
phase information.
"""

# Importing Global Libraries
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, Final, Callable, Optional

# Import custom libraries
from .double_conv_2d_layer import DoubleConv2DLayer

from ....constants import INTERPOLATION_BILINIAR

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

# == ModulusUpsampleBlock ==
class ModulusUpsampleBlock(nn.Module):
    """
    Decoder block for fusing Modulus Scattering coefficients.

    This block implements the 'Standard' path of the ablation study.
    It treats scattering coefficients as standard feature channels, 
    concatenating them with the upsampled bottleneck features.

    Inputs:
      - bottleneck_in: Features from the coarser level (B, C, H_c, W_c).
      - U_j: Scattering coefficients at current scale j (B, C, L, H_j, W_j).
    """
    
    # ===========

    # -- __init__
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        L: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        inplace: bool = True,
        interpolation: str = INTERPOLATION_BILINIAR, # "bilinear" or "nearest"
        align_corners: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        self.L = L
        self.skip_channels = self.in_channels * self.L

        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.inplace = inplace

        self.interpolation = interpolation
        self.align_corners = align_corners
        if self.interpolation == 'nearest':
            self.align_corners = None

        # Fusion conv: combines [U_gated, A_mag, Phi] -> U_j
        fuse_in_channels = self.base_channels + self.skip_channels
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
        U_j: torch.Tensor,   # (B, C_lp, H_l, W_l)
    ) -> torch.Tensor:
        """
        Executes the fusion pass.

        1. Upsamples `bottleneck_in` to match `U_j` spatial resolution.
        2. Flattens `U_j` (scattering) orientations into the channel dimension.
        3. Concatenates and refines via DoubleConv.
        """

        # 1. Upsample to match post-modulues spatial size
        # Z_j: (B, C_in, L, Hj, Wj)
        B, C_in, L, Hj, Wj = U_j.shape
        interpolation_kwargs: Dict[str, Any] = {"mode": self.interpolation}
        if self.interpolation in ["bilinear", "bicubic", "linear", "trilinear"]:
            interpolation_kwargs["align_corners"] = self.align_corners
        bottleneck_up = F.interpolate(bottleneck_in, size=(Hj, Wj), **interpolation_kwargs)

        # 2. Reshape to flatten orientations into channels
        # Input U_j is (B, C, L, H, W)
        U_j_flat = U_j.reshape(B, C_in * self.L, Hj, Wj)

        # 3. Fusion
        fusion_in = torch.cat([bottleneck_up, U_j_flat], dim=1)
        block_out = self.fuse_conv(fusion_in)
        return block_out
