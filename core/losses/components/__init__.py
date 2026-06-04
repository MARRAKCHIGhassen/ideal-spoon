# -*- coding: utf-8 -*-
"""
Loss Components Package.

This package provides the atomic implementations of specific objective functions.
These modules are designed to be aggregated by the high-level builders in the
parent `losses` package.

Exports:
    * DiceLoss: differentiable Dice coefficient for segmentation.
    * PhaseAlignLoss: Cross-scale phase consistency regularization.
    * PhaseTVLoss: Intra-scale phase smoothness regularization (Total Variation).
"""

# -------------------------------------------------------------------------------------------------------------------

from .dice_loss import DiceLoss
from .phase_align_loss import PhaseAlignLoss
from .phase_tv_loss import PhaseTVLoss

# --