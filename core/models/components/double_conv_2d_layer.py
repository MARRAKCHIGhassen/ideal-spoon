# -*- coding: utf-8 -*-
"""
Standard Double Convolution Block.

This module implements the fundamental building block of the U-Net architecture.
It consists of two repeated sequences of Convolution, Batch Normalization, 
and ReLU activation.
"""

# Importing Global Libraries
from __future__ import annotations

import logging

from torch import nn

from typing import Final, Callable

# Import custom libraries

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

# == DoubleConv2DLayer ==
class DoubleConv2DLayer(nn.Module):
    """
    Two consecutive Convolutional steps with Batch Normalization and ReLU.

    This block increases the non-linearity of the network and expands the 
    effective receptive field. It is used in both the Encoder (feature extraction)
    and the Decoder (feature refinement).

    Structure:
        (Conv3x3 -> BN -> ReLU) -> (Conv3x3 -> BN -> ReLU)
    """
    
    # ===========

    # -- __init__
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            bias: bool = False,
            inplace: bool = True,
    ):
        """
        Initializes the DoubleConv2DLayer.

        Args:
            in_channels (int): Input channel depth.
            hidden_channels (int): Intermediate channel depth (usually equals out_channels).
            out_channels (int): Output channel depth.
            kernel_size (int): Convolution kernel size (default: 3).
            padding (int): Padding to maintain spatial dimensions (default: 1).
            bias (bool): Enable learnable bias (usually False if BN is used).
            inplace (bool): Perform ReLU in-place to save memory.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.inplace = inplace
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=inplace),
        )

    # -- forward
    def forward(self, x):
        return self.double_conv(x)
