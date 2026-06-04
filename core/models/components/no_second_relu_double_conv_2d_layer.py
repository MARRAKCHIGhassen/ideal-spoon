# -*- coding: utf-8 -*-
"""
Double Convolution Block (Linear Output).

A variant of the DoubleConv2D layer that omits the final activation function (ReLU).
This is typically used as the final projection head of a network or as a 
linear mixing block before a specialized activation (like Sigmoid).
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

# == NoSecondReLuDoubleConv2DLayer ==
class NoSecondReLuDoubleConv2DLayer(nn.Module):
    """
    Double Convolution Block: (Conv-BN-ReLU) -> (Conv).
    
    Structure:
    1. Conv2d -> BatchNorm2d -> ReLU
    2. Conv2d (Linear output)
    
    This block is useful for learning features where the final magnitude/sign 
    matters and should not be clamped by ReLU (e.g., logits).
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
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.inplace = inplace
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=bias, padding_mode='reflect'),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias, padding_mode='reflect'),
            # No BN/ReLU here; treat as linear correction in coeff space
        )

    # -- forward
    def forward(self, x):
        return self.double_conv(x)
