# -*- coding: utf-8 -*-
"""
Scattering-Specific U-Net Decoder.

This module implements the specialized decoder used in the Phase-Aware U-Net.
Unlike the standard decoder, it is designed to handle:
1.  **Multi-Scale Scattering Inputs**: It ingests skip connections indexed by scale (j).
2.  **Phase Gating**: It supports 'PhaseAwareUpsampleBlock' to utilize complex structural info.
3.  **Modulus Fusion**: It supports 'ModulusUpsampleBlock' for invariant baselines.
"""

# Importing Global Libraries
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from typing import Final, Callable, Dict, Optional

# Import custom libraries
from ..components.modulus_upsample_block import ModulusUpsampleBlock
from ..components.phase_aware_upsample_block import PhaseAwareUpsampleBlock

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

# == ScatteringUNetDecoder ==
class ScatteringUNetDecoder(nn.Module):
    """
    Coarse-to-Fine Scattering Decoder.
    
    This decoder reconstructs the image by iterating from the coarsest scattering 
    scale (J-1) up to the finest scale (0). At each level, it fuses the current
    feature map with the corresponding scattering coefficients (skip connection).

    Architecture:
        * **Dynamic Block Selection**: Instantiates either Modulus or Phase-Aware 
            blocks based on `skip_type`.
        * **Scale Iteration**: Explicitly handles the J scales defined by the encoder.

    Attributes:
        blocks (nn.ModuleList): The sequence of upsampling blocks (Deep -> Shallow).
        final (nn.Conv2d): Final output projection.
    """
    
    def __init__(
        self,
        J: int,
        L: int,
        in_channels: int,
        base_channels: int,
        skip_type: str = SKIP_MODULUS, # "modulus", "raw", "polar"
        out_channels: int = 6,
        gate_hidden_channels: int = 32,
        eps: float = 1e-6,
        upsample_kernel_size: int = 3,
        upsample_padding: int = 1,
        upsample_bias: bool = False,
        upsample_inplace: bool = True,
        interpolation: str = INTERPOLATION_BILINIAR, # "bilinear" or "nearest"
        align_corners: bool = False,
        fuse_kernel_size: int = 1,
        fuse_padding: int = 0,
        fuse_bias: bool = False,
        activation: Optional[str] = None,
        **unused
    ):
        """
        Initializes the Scattering Decoder.

        Args:
            J (int): Number of scattering scales (depth).
            L (int): Number of orientations (used to calculate skip width).
            in_channels (int): Channels entering from bottleneck.
            skip_type (str): Determines the block type (Modulus vs Phase-Aware).
            gate_hidden_channels (int): Hidden dim for the phase gate MLP.
        """
        super().__init__()
        self.J = J
        self.L = L

        self.in_channels = in_channels
        self.base_channels = base_channels

        self.skip_type = skip_type

        self.gate_hidden_channels = gate_hidden_channels
        self.eps = eps
        
        self.upsample_kernel_size = upsample_kernel_size
        self.upsample_padding = upsample_padding
        self.upsample_bias = upsample_bias
        self.upsample_inplace = upsample_inplace

        self.interpolation = interpolation
        self.align_corners = align_corners

        self.fuse_kernel_size = fuse_kernel_size
        self.fuse_padding = fuse_padding
        self.fuse_bias = fuse_bias

        # Calculate Skip Width: 
        # Scattering Encoder flattens (B, C, L, H, W) -> (B, C*L, H, W)
        self.blocks = nn.ModuleList()
        
        # Setting upsample block
        if self.skip_type == SKIP_STANDARD:
            semantic_logger.error(
                f"{SKIP_STANDARD} is dedicated to standard u-net. use either {SKIP_MODULUS} instead"
            )
            debug_logger.error(
                f"{SKIP_STANDARD} is dedicated to standard u-net. use either {SKIP_MODULUS} instead"
            )
            raise ValueError(
                f"{SKIP_STANDARD} is dedicated to standard u-net. use either {SKIP_MODULUS} instead"
            )
        
        # Build Levels from Coarse (Deep) to Fine (Shallow)
        for j in range(self.J):
            # Scale Index: i=0 is Deepest (Level J-1), i=J-1 is Finest (Level 0)
            
            # Constant width strategy (matching typical Scattering networks)
            if self.skip_type == SKIP_COMPLEX_RAW or self.skip_type == SKIP_COMPLEX_MAG_PHASE:
                # M6: Use Phase-Aware Block
                # Argument name: 'wav_channels'
                self.blocks.append(
                    PhaseAwareUpsampleBlock(
                        in_channels=self.in_channels,
                        base_channels=self.base_channels,
                        L=self.L,
                        skip_type=self.skip_type,
                        gate_hidden_channels=self.gate_hidden_channels,
                        kernel_size=upsample_kernel_size,
                        padding=upsample_padding,
                        bias=upsample_bias,
                        inplace=upsample_inplace,
                        eps=self.eps,
                        interpolation=self.interpolation,
                        align_corners=self.align_corners,
                        activation=activation,
                    )
                )
            elif self.skip_type == SKIP_MODULUS:
                # M4/M5: Use Standard Block
                # Argument name: 'skip_channels'
                
                # Handle M9 (none): Skips are effectively 0 or handled by block logic
                # If 'none', we might pass 0, but the block might expect a tensor.
                # Usually we pass 'skip_width' and feed zeros if 'none'.
                self.blocks.append(
                    ModulusUpsampleBlock(
                        in_channels=self.in_channels,
                        base_channels=self.base_channels,
                        L=self.L,
                        kernel_size=upsample_kernel_size,
                        padding=self.upsample_padding,
                        bias=self.upsample_bias,
                        inplace=self.upsample_inplace,
                        interpolation=self.interpolation,
                        align_corners=self.align_corners,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported skip_type: {self.skip_type}"
                )
        
        self.final = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=fuse_kernel_size,
            padding=fuse_padding,
            bias=fuse_bias
        )

    # -- forward
    def forward(
        self,
        bottleneck_feat: torch.Tensor,
        skips_list: Dict[int, torch.Tensor],
    ):
        """
        Executes the coarse-to-fine decoding pass.

        Args:
            bottleneck_feat (torch.Tensor): Features projected from the bottleneck.
            skips_list (Dict[int, torch.Tensor]): Map of {scale_j: features}.
                Note: The key 'j' corresponds to resolution, where j=0 is fine 
                and j=J-1 is coarse.

        Returns:
            torch.Tensor: The final reconstructed image.
        """
        x = bottleneck_feat
        
        # We iterate through blocks. 
        # Block[0] is the Coarsest (Deepest) -> Handles scale J-1
        for level in reversed(range(self.J)):
            skip_feat = skips_list[level]
            # Mapping: level J-1 (coarse) -> block index 0
            block_idx = self.J - 1 - level
            x = self.blocks[block_idx](x, skip_feat)
            
        return self.final(x)
