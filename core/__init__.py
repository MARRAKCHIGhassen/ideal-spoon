# -*- coding: utf-8 -*-
"""
Core Components Package.

This package organizes the primary functional blocks of the deep learning pipeline.
It encapsulates the logic for constructing, training, and evaluating the
Phase-Aware Scattering Encoder-Decoder.

Subpackages:
    * **models**: Implementations of the neural network architectures, including
      the hybrid Scattering-UNet and the Phase-Aware skip connection mechanisms.
    * **losses**: Custom and standard loss functions for dense prediction tasks
      (e.g., reconstruction loss, segmentation loss).
    * **metrics**: Evaluation metrics (PSNR, SSIM, IoU) used to quantify
      performance on standard benchmarks (BSD68) and medical datasets (ISIC).
    * **optimizers**: Wrappers and configurations for optimization algorithms
      and learning rate schedulers.
    * **data**: Data loading strategies, augmentation pipelines (including the
      rotation strategies defined in constants), and dataset wrappers.
"""

# -------------------------------------------------------------------------------------------------------------------

# --