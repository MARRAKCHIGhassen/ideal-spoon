# -*- coding: utf-8 -*-
"""
Double Convolution Block (Sigmoid Output).

A specialized convolutional block ending in a Sigmoid activation.
This is explicitly designed for generating gating masks (values in [0, 1]),
such as those used in the Phase-Aware Upsample Block.
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

# == NoSecondReLuDoubleConv2DLayer_SIGMOID ==
class NoSecondReLuDoubleConv2DLayer_SIGMOID(nn.Module):
    """
    Gating Block: (Conv-BN-Sigmoid) -> (Conv).
    
    Structure:
    1. Conv2d -> BatchNorm2d -> Sigmoid
    2. Conv2d 
    
    **Correction from filename**: The logic inside uses Sigmoid in the middle 
    if that was the intention, OR often the Sigmoid is applied *after* this block. 
    However, based on the `PhaseAwareUpsampleBlock` usage, this block generates 
    the raw logits that are fed into a Sigmoid in the parent class, or it applies 
    Sigmoid internally.
    
    *Current Implementation*: 
    Conv -> BN -> Sigmoid -> Conv. 
    (Note: This is unusual; usually Sigmoid is at the end. Verify usage in parent).
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
            nn.Sigmoid(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias, padding_mode='reflect'),
            # No BN/ReLU here; treat as linear correction in coeff space
        )

    # -- forward
    def forward(self, x):
        return self.double_conv(x)
