# -*- coding: utf-8 -*-
"""
Components Package.

This package contains the building blocks for the Phase-Aware Scattering Network.
It includes:
1.  **Wavelet Generators**: Morlet filter banks (GridBank2D).
2.  **Scattering Primitives**: Modulus, Padding, and Unpadding layers.
3.  **Upsampling Blocks**: Phase-Aware and Modulus-based decoder blocks.
4.  **Convolutional Blocks**: Specialized layers for gating and feature refinement.
"""

# -------------------------------------------------------------------------------------------------------------------

from .wavelet_bank_2d import GridBank2D
from .wave_block_2d_layer import WaveBlock
from .padding_layer import PaddingLayer
from .unpadding_layer import UnpaddingLayer
from .modulus_and_subsampling_layer import ModulusSubsampleLayer

from .phase_aware_upsample_block import PhaseAwareUpsampleBlock
from .modulus_upsample_block import ModulusUpsampleBlock

from .no_second_relu_double_conv_2d_layer import NoSecondReLuDoubleConv2DLayer
from .no_second_relu_double_conv_2d_layer_sigmoid import NoSecondReLuDoubleConv2DLayer_SIGMOID
from .no_second_relu_double_conv_2d_layer_leaky import NoSecondReLuDoubleConv2DLayer_LEAKY

# --