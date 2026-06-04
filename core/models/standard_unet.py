# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Standard U-Net
==============================================================

This module implements the classic U-Net architecture (Ronneberger et al., 2015).
It serves as the primary baseline for evaluating the stability-expressiveness 
trade-offs in dense prediction tasks.
"""

# Importing Global Libraries
from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Import custom libraries
from .components.double_conv_2d_layer import DoubleConv2DLayer

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

# == UNet ==
class UNet(nn.Module):
    """
    Standard U-Net architecture for image-to-image translation tasks.

    This implementation follows the contract:
    1. Encoder: Successive application of pooling and Double Convolution blocks.
    2. Bottleneck: The deepest feature representation.
    3. Decoder: Upsampling followed by concatenation with skip connections 
       and Double Convolution blocks.

    Attributes
    ----------
    depth : int
        The number of downsampling/upsampling levels in the network.
    encoder_convs : nn.ModuleList
        List of DoubleConv2D layers for the contracting path.
    pool : nn.MaxPool2d
        Standard 2x2 max-pooling operation.
    decoder_ups : nn.ModuleList
        List of ConvTranspose2d layers for increasing spatial resolution.
    decoder_convs : nn.ModuleList
        List of DoubleConv2D layers for processing concatenated features.
    outc : nn.Conv2d
        Final 1x1 convolution layer to project features to output channels.
    """

    # ===========

    # -- __init__
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        **unused
    ) -> None:
        """
        Initializes the U-Net architecture with configurable depth and width.

        Parameters
        ----------
        in_channels : int, optional
            Number of input image channels, by default 1.
        out_channels : int, optional
            Number of target output channels, by default 1.
        base_channels : int, optional
            Number of filters in the first layer, doubling at each level, by default 64.
        depth : int, optional
            Number of downsampling operations, by default 4.
        **unused : Any
            Additional keyword arguments ignored during initialization.
        """
        
        # ------------------------------------------------

        super().__init__()
        
        self.depth = depth
        semantic_logger.info(f"Building Standard U-Net (depth={depth}, base_channels={base_channels})")
        debug_logger.debug(f"Initializing U-Net structure: in={in_channels}, out={out_channels}")

        # 1. Base Components
        self.pool = nn.MaxPool2d(2)
        self.encoder_convs = nn.ModuleList()
        
        # 2. Encoder (Down Path) Construction
        debug_logger.debug("Constructing Encoder Path...")
        # Input layer (Level 0)
        self.encoder_convs.append(DoubleConv2DLayer(in_channels, base_channels, base_channels))
        
        feat_ch = base_channels
        for i in range(depth):
            # Successive Downsampling blocks
            debug_logger.debug(f"Encoder Level {i+1}: channels {feat_ch} -> {feat_ch * 2}")
            self.encoder_convs.append(DoubleConv2DLayer(feat_ch, feat_ch * 2, feat_ch * 2))
            feat_ch *= 2
            
        # 3. Decoder (Up Path) Construction
        debug_logger.debug("Constructing Decoder Path...")
        self.decoder_ups = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        
        for j in range(depth):
            debug_logger.debug(f"Decoder Level {depth - j}: channels {feat_ch} -> {feat_ch // 2}")
            # Upsampling operation (reduces channel depth)
            self.decoder_ups.append(nn.ConvTranspose2d(feat_ch, feat_ch // 2, kernel_size=2, stride=2))
            
            # Post-concatenation convolution
            # Concatenation brings: (feat_ch // 2 from up) + (feat_ch // 2 from skip) = feat_ch
            self.decoder_convs.append(DoubleConv2DLayer(feat_ch, feat_ch // 2, feat_ch // 2))
            feat_ch //= 2
            
        # 4. Final Output Projection
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        debug_logger.debug("U-Net initialization complete.")

    # -- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the U-Net forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (B, C_in, H, W).

        Returns
        -------
        torch.Tensor
            Output prediction tensor of shape (B, C_out, H, W).
        """
        # ------------------------------------------------

        # -- Encoding Path --
        debug_logger.debug(f"Forward pass started. Input shape: {x.shape}")
        skips: List[torch.Tensor] = []
        
        # First block: extract features before pooling
        x = self.encoder_convs[0](x)
        skips.append(x)
        debug_logger.debug(f"L0 Encoder features extracted. Shape: {x.shape}")
        
        # Iterative pooling and encoding
        for i in range(1, self.depth + 1):
            x = self.pool(x)
            x = self.encoder_convs[i](x)
            # Store features for skip connections (except the deepest bottleneck)
            if i < self.depth:
                skips.append(x)
                debug_logger.debug(f"L{i} skip connection stored. Shape: {x.shape}")

        # x is now at the bottleneck
        debug_logger.debug(f"Bottleneck reached. Shape: {x.shape}")
                
        # -- Decoding Path --
        for i in range(self.depth):
            # 1. Upsample
            x = self.decoder_ups[i](x)
            
            # 2. Retrieve skip features using LIFO (Last-In, First-Out)
            skip = skips[-(i+1)]
            
            # 3. Dynamic Resizing for spatial consistency
            if x.shape != skip.shape:
                debug_logger.debug(f"Interpolating decoder x from {x.shape[2:]} to {skip.shape[2:]}")
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
            # 4. Feature Concatenation along channel dimension
            x = torch.cat([skip, x], dim=1)
            
            # 5. Feature Refinement via Double Convolution
            x = self.decoder_convs[i](x)
            debug_logger.debug(f"Decoder level {i} completed. Shape: {x.shape}")
            
        # 6. Final mapping to out_channels
        return self.outc(x)
