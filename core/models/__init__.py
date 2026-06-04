# -*- coding: utf-8 -*-
"""
Models Package.

This package contains the deep learning architectures used for the
Phase-Aware Scattering Encoder-Decoder project.

Exports:
    * UNet: A standard U-Net baseline (Ronneberger et al., 2015).
    * PhaseAwareUNet: The proposed hybrid architecture integrating 
      Scattering Transforms with Phase-Aware Skip Connections.
"""

# -------------------------------------------------------------------------------------------------------------------

from .standard_unet import UNet
from .phase_aware_unet import PhaseAwareUNet

# --