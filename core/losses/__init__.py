# -*- coding: utf-8 -*-
"""
Loss Functions Package.

This package contains the objective functions used to train the network.
It provides:
1.  **Standard Losses**: Wrappers for MSE, L1, CrossEntropy, and Dice.
2.  **Hybrid Losses**: Weighted combinations (e.g., Dice + CE) for complex tasks.
3.  **Phase-Aware Losses**: Specialized objectives that enforce phase alignment 
    and structural smoothness in addition to pixel-wise accuracy.

Usage:
    The package exposes 'builder' functions (e.g., l1_phase_standard_builder) 
    that instantiate and configure the composite loss modules.
"""

# -------------------------------------------------------------------------------------------------------------------

from ._baseline import BaseLoss, WeightedSumLoss

from .mse import mse_standard_builder
from .l1 import l1_standard_builder
from .l1_phase import l1_phase_standard_builder
from .dice_ce import dice_ce_standard_builder
from .dice_ce_phase import dice_ce_phase_builder

# --