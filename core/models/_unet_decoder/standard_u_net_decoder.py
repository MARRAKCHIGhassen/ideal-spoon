# -*- coding: utf-8 -*-
"""
Standard U-Net Decoder.

This module implements the decoding path of the classic U-Net (Ronneberger et al., 2015).
It performs iterative upsampling, feature concatenation (skip connections), and 
feature refinement using Double Convolution blocks.
"""

# Importing Global Libraries
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Final, Callable

# Import custom libraries
from ..components.double_conv_2d_layer import DoubleConv2DLayer

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

# == StandardUNetDecoder ==
class StandardUNetDecoder(nn.Module):
    """
    Standard Symmetric U-Net Decoder.
    
    This component reconstructs the spatial resolution from the bottleneck features.
    It adheres to the classic design pattern:
    1. Upsample (ConvTranspose2d).
    2. Concatenate with corresponding encoder skip connection (LIFO).
    3. Refine features (DoubleConv2D).

    Attributes:
        decoder_ups (nn.ModuleList): Layers for increasing spatial resolution.
        decoder_convs (nn.ModuleList): Layers for processing concatenated features.
        final (nn.Conv2d): The final projection layer to the target number of classes/channels.
    """
    
    def __init__(
        self,
        in_channels: int = 1, # Grayscale default
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        upsample_stride: int = 2,
        upsample_padding: int = 1,
        upsample_bias: bool = False,
        conv_padding: int = 2,
        conv_bias: bool = False,
        conv_inplace: bool = True,
        interpolation: str = INTERPOLATION_BILINIAR, # "bilinear" or "nearest"
        align_corners: bool = False,
        fuse_kernel_size: int = 1,
        fuse_padding: int = 0,
        fuse_bias: bool = False,
        **unused
    ) -> None:
        """
        Initializes the Standard Decoder.

        Args:
            in_channels (int): Channels entering from the bottleneck.
            out_channels (int): Final output channels (e.g., num_classes).
            base_channels (int): Channel width at the shallowest level.
            depth (int): Number of upsampling steps (must match encoder depth).
            upsample_stride (int): Stride for ConvTranspose2d (usually 2).
        """
        super().__init__()
        self.depth = depth

        self.upsample_stride = upsample_stride
        self.upsample_padding = upsample_padding
        self.upsample_bias = upsample_bias

        self.conv_padding = conv_padding
        self.conv_bias = conv_bias
        self.conv_inplace = conv_inplace

        self.interpolation = interpolation
        self.align_corners = align_corners

        self.fuse_kernel_size = fuse_kernel_size
        self.fuse_padding = fuse_padding
        self.fuse_bias = fuse_bias

        self.decoder_ups = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        
        feat_ch = in_channels
        for _ in range(depth):
            # Up-conv: reduces channels by 2
            self.decoder_ups.append(
                nn.ConvTranspose2d(
                    feat_ch,
                    feat_ch // 2,
                    kernel_size=2,
                    stride=upsample_stride,
                    padding=upsample_padding,
                    bias=upsample_bias,
                )
            )
            
            # Double Conv: takes cat(skip, up) -> feat_ch // 2
            # Skip has (feat_ch // 2) channels
            # Up has (feat_ch // 2) channels
            # Concat = feat_ch
            # Output = feat_ch // 2
            self.decoder_convs.append(
                DoubleConv2DLayer(
                    feat_ch,
                    feat_ch // 2,
                    feat_ch // 2,
                    padding=conv_padding,
                    bias=conv_bias,
                    inplace=conv_inplace,
                )
            )
            feat_ch //= 2

        self.final = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=fuse_kernel_size,
            padding=fuse_padding,
            bias=fuse_bias
        )

    # -- forward
    def forward(self, x, skips):
        """
        Forward pass for the standard decoder.

        Args:
            x (torch.Tensor): Bottleneck features.
            skips (list[torch.Tensor]): List of skip connections collected 
                from the encoder (order: shallow -> deep).

        Returns:
            torch.Tensor: The reconstructed output.
        """
        # We iterate backwards from bottlenecks
        for i in range(self.depth):
            # 1. Upsample
            x = self.decoder_ups[i](x)
            
            # 2. Get corresponding skip (LIFO)
            skip = skips[-(i+1)]
            
            # 3. Resize if needed (for odd dimensions)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
            # 4. Concat
            x = torch.cat([skip, x], dim=1)
            
            # 5. Conv
            x = self.decoder_convs[i](x)
            
        return self.final(x)
