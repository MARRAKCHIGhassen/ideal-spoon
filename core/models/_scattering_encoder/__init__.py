# -*- coding: utf-8 -*-
"""
Scattering Encoder Package.

This package provides the core Wavelet Scattering Transform implementations
used as the feature extractor in the Phase-Aware U-Net.

Exports:
    * StandardVectorizedScattering2DEncoder: High-performance, high-memory
      implementation using batched tensor operations. Best for training on
      standard GPUs with fixed resolution.
    * StandardMonolithicScattering2DEncoder: Sequential loop-based
      implementation. Low memory footprint, supports internal subsampling
      and variable scales. Best for high-resolution images or ablation studies
      requiring specific subsampling factors.
"""

# -------------------------------------------------------------------------------------------------------------------

from .standard_vectorized_2d_encoder import StandardVectorizedScattering2DEncoder
from .standard_monolithic_2d_encoder import StandardMonolithicScattering2DEncoder

# --