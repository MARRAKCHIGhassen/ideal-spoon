# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Dice Loss
==========================================================

This module implements the Dice Loss objective function, specifically designed 
for semantic segmentation tasks. It measures the overlap between predicted 
probability maps and ground truth masks, providing robustness against 
class imbalance.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Final, Callable, Any

# Import custom libraries

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# LOGGERS
# ---------------------------------------------------------------------

debug_logger = logging.getLogger(f"{__name__.split('.')[0]}.debug")
semantic_logger = logging.getLogger(f"{__name__.split('.')[0]}.semantic")

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# IMPLEMENTATION
# ---------------------------------------------------------------------

# == DiceLoss ==
class DiceLoss(nn.Module):
    """
    Dice Loss implementation for semantic segmentation.

    This loss optimizes the Dice coefficient (F1-score) directly. It handles 
    both integer class masks and one-hot encoded targets.

    Parameters
    ----------
    weight : float, default=1.0
        The importance weight of this loss in the total objective sum.
    smooth : float, default=1e-6
        Smoothing factor to prevent division by zero and stabilize training.
    **unused : Any
        Captured arguments for compatibility with automated builder dispatchers.
    """

    # ===========

    # -- __init__
    def __init__(self, smooth: float = 1e-6, **unused: Any):
        """
        Initializes the Dice Loss module.
        """
        # ------------------------------------------------

        super().__init__()
        self.smooth = smooth
        
        debug_logger.debug(f"DiceLoss initialized with smooth={smooth}")

    # -- forward
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Computes the Dice loss between predictions and targets.

        Parameters
        ----------
        prediction : torch.Tensor
            Raw logits from the model of shape (B, C, H, W).
        target : torch.Tensor
            Ground truth masks of shape (B, H, W) or (B, C, H, W).
        **kwargs : Any
            Additional metadata passed during the training step.

        Returns
        -------
        torch.Tensor
            Weighted scalar Dice loss.
        """
        # ------------------------------------------------

        debug_logger.debug("Computing Dice Loss for segmentation overlap.")

        # 1. Prediction Softmax
        debug_logger.debug(f"Applying Softmax to prediction of shape: {prediction.shape}")
        probs = F.softmax(prediction, dim=1)

        # 2. Target Pre-processing
        if target.dim() == 3:
            debug_logger.debug("Converting integer class target to one-hot encoding.")
            target = F.one_hot(
                target.long(), 
                num_classes=prediction.shape[1]
            ).permute(0, 3, 1, 2).float()
            
        # 3. Calculation of Overlap
        debug_logger.debug("Calculating intersection and union for Dice coefficient.")
        intersection = (probs * target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        # 4. Score Computation
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice_score.mean()

        debug_logger.debug(f"Raw Dice Loss (mean): {loss.item():.6f}")

        return loss

    # -- __repr__
    def __repr__(self) -> str:
        """
        Returns a string representation of the module state.

        Returns
        -------
        str
            State summary.
        """
        # ------------------------------------------------
        return (
            f"{self.__class__.__name__}("
            f"smooth={self.smooth})"
        )
