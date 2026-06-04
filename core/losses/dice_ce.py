# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Dice‑CE Hybrid Loss
====================================================================

This module implements a hybrid loss function combining Dice Loss and 
Cross-Entropy (CE). This combination is a standard objective for semantic 
segmentation tasks, leveraging the spatial overlap optimization of Dice and 
the pixel-wise distribution alignment of Cross-Entropy.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from typing import Final, Optional, Callable, Any, Dict

# Import custom libraries
from ._baseline import WeightedSumLoss
from .components.dice_loss import DiceLoss
from ...constants import KIND_LOSS

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

# Implementation leverages atomic DiceLoss and standard nn.CrossEntropyLoss
# aggregated via the WeightedSumLoss baseline.

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# BUILDERS
# ---------------------------------------------------------------------

# -- dice_ce_standard_builder
def dice_ce_standard_builder(**kwargs) -> WeightedSumLoss:
    """
    Constructs a weighted combination of Dice and Cross-Entropy losses.

    The resulting objective is: L = lambda_ce * CE + lambda_dice * Dice.
    This hybrid approach is particularly effective for handling class 
    imbalance in medical image segmentation.

    Parameters
    ----------
    lambda_ce : float, default=0.5
        Weighting factor for the Cross-Entropy component.
    ce_class_rescale : torch.Tensor, default=torch.tensor(1.0)
        Optional rescaling weights for CE classes.
    ce_reduction : str, default='mean'
        The reduction strategy for the CE component.
    lambda_dice : float, default=0.5
        Weighting factor for the Dice component.
    dice_class_rescale : torch.Tensor, default=torch.tensor(1.0)
        Optional weights for Dice computation (reserved for class-wise Dice).
    **kwargs : Any
        Additional parameters passed to the underlying CrossEntropyLoss.

    Returns
    -------
    WeightedSumLoss
        An aggregated loss module comprising Dice and CE components.
    """
    
    # ------------------------------------------------

    semantic_logger.info("Builder: Configuring Hybrid Dice-CrossEntropy Loss.")
    
    # 1. Resolve Cross-Entropy Parameters
    lambda_ce = kwargs.pop('lambda_ce', None)
    if lambda_ce is None:
        debug_logger.debug("CE weight 'lambda_ce' not provided. Defaulting to 0.5.")
        lambda_ce = 0.5

    ce_class_rescale = kwargs.pop('ce_class_rescale', None)
    if ce_class_rescale is None:
        debug_logger.debug("Using uniform class weights for Cross-Entropy. Defaulting to None for all.")

    ce_reduction = kwargs.pop('ce_reduction', None)
    if ce_class_rescale is None:
        debug_logger.debug(f"CE weight 'ce_reduction' not provided. Defaulting {ce_reduction}")
        ce_reduction = 'mean'

    # 2. Resolve Dice Parameters
    lambda_dice = kwargs.pop('lambda_dice', 0.5)
    if lambda_dice is None:
        debug_logger.debug("Dice weight 'lambda_dice' not provided. Defaulting to 0.5.")
        lambda_dice = 0.5
    
    # 3. Component Instantiation
    debug_logger.debug("Instantiating atomic loss components (CE and Dice).")
    ce = nn.CrossEntropyLoss(
        weight=ce_class_rescale,
        reduction=ce_reduction,
        **kwargs
    )
    dice = DiceLoss(smooth=1e-5)
    
    # 4. Aggregation
    semantic_logger.info(f"Loss Balance: {lambda_ce}*CE + {lambda_dice}*Dice")
    return WeightedSumLoss(
        losses=[ce, dice], 
        weights=[lambda_ce, lambda_dice],
    )
