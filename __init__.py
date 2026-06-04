# -*- coding: utf-8 -*-
"""
Phase-Aware Scattering Encoder-Decoder Package.

This package implements a hybrid deep learning architecture that integrates
Wavelet Scattering Transforms with an Encoder-Decoder framework.

Key Features:
    * **Phase-Aware Skip Connections**: Restores spatial information lost in
        global averaging by explicitly preserving phase in skip connections.
    * **Scattering Integration**: Leverages the Lipschitz stability and translation
        invariance of scattering transforms.
    * **Dense Prediction**: Tailored for tasks like Image Denoising and Segmentation.
    * **Ablation Utilities**: Includes tools for spatial shuffling to validate
        the role of phase in encoding location-dependent structure.

Research Context:
    This work addresses the trade-off between stability (Scattering) and
    spatial precision (Dense Prediction), demonstrating significant improvements
    in PSNR and segmentation accuracy by breaking strict translation invariance
    via phase preservation.
"""

# -------------------------------------------------------------------------------------------------------------------

# --