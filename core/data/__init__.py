# -*- coding: utf-8 -*-
"""
Data Loading and Augmentation Package.

This package manages the ingestion, preprocessing, and augmentation of datasets
used for training and evaluation. It implements specific wrappers for:
1.  **Image Denoising (BSD)**: Handles on-the-fly Gaussian noise injection.
2.  **Semantic Segmentation (ISIC)**: Handles synchronized image-mask transforms.

Exports:
    * BSDDataManager: Factory for the Berkeley Segmentation Dataset (Denoising).
    * ISICDataManager: Factory for the Skin Lesion Segmentation Dataset.
"""

# -------------------------------------------------------------------------------------------------------------------

from .bsd import BSDDataManager
from .isic import ISICDataManager

# --