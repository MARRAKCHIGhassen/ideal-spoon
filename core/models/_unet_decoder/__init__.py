# -*- coding: utf-8 -*-
"""
U-Net Decoder Package.

This package provides the decoding mechanisms for dense prediction tasks.
It includes both the standard U-Net decoder (for baselines) and the
specialized Phase-Aware decoder that integrates scattering features.

Exports:
    * StandardUNetDecoder: Classic U-Net upsampling path (UpConv -> Concat -> Conv).
    * ScatteringUNetDecoder: Specialized decoder that handles Phase-Aware
      Upsampling blocks and orchestrates coarse-to-fine reconstruction
      using scattering coefficients.
"""

# -------------------------------------------------------------------------------------------------------------------

from .standard_u_net_decoder import StandardUNetDecoder
from .scattering_u_net_decoder import ScatteringUNetDecoder

# --