# -*- coding: utf-8 -*-
"""
Metrics Package.

This package provides evaluation suites for various tasks within the project.
It leverages `torchmetrics` for reliable, distributed-aware calculation and 
aggregation.

Exports:
    * MAEMetric: Simple Mean Absolute Error for regression.
    * RestorationMetrics: Signal fidelity suite (PSNR, SSIM, LPIPS).
    * SegmentationMetrics: Dense prediction suite (Dice, IoU, Hausdorff).
    * StabilityMetrics: Feature invariance suite (Relative Error, Cosine Sim).
"""

# -------------------------------------------------------------------------------------------------------------------

from .mae import MAEMetric
from .restoration_metrics import RestorationMetrics
from .segmentation_metrics import SegmentationMetrics
from .stability_metrics import StabilityMetrics

# --