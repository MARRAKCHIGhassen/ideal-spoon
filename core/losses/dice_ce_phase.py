# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Dice‑CE‑Phase Hybrid Loss
==========================================================================

This module implements a comprehensive hybrid loss function for phase-aware 
segmentation. It combines spatial overlap optimization (Dice), pixel-wise 
distribution alignment (Cross-Entropy), and phase-specific regularization 
(Alignment and Total Variation) into a single objective.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from .components.dice_loss import DiceLoss
from .components.phase_align_loss import PhaseAlignLoss
from .components.phase_tv_loss import PhaseTVLoss

from typing import Final, Optional, Callable, Any, Dict

# Import custom libraries
from ._baseline import WeightedSumLoss
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

# Implementation leverages atomic components (CE, Dice, PhaseAlign, PhaseTV) 
# aggregated via the WeightedSumLoss baseline container.

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# BUILDERS
# ---------------------------------------------------------------------

# -- dice_ce_phase_builder
def dice_ce_phase_builder(**kwargs) -> WeightedSumLoss:
    """
    Constructs the compound Dice-CE-Phase objective.

    The total loss is a weighted sum of four components:
    L = w_ce*CE + w_dice*Dice + w_align*PhaseAlign + w_tv*PhaseTV.

    Parameters
    ----------
    **kwargs : Any
        Dynamic configuration keys including:
        - lambda_ce : float (default 0.5)
        - ce_class_rescale : torch.Tensor (default None)
        - ce_reduction : str (default 'mean')
        - lambda_dice : float (default 0.5)
        - dice_class_rescale : torch.Tensor (default 1.0)
        - lambda_phase_align : float (default 0.5)
        - lambda_phase_tv : float (default 0.5)

    Returns
    -------
    WeightedSumLoss
        The multi-component aggregated loss module.
    """
    
    # ------------------------------------------------

    semantic_logger.info("Builder: Configuring Hybrid Dice-CE-Phase Segmentation Loss.")
    
    # 1. Resolve Cross-Entropy Parameters
    lambda_ce = kwargs.pop('lambda_ce', None)
    if lambda_ce is None:
        debug_logger.warning("No 'lambda_ce' provided. Defaulting to 0.5.")
        lambda_ce = 0.5

    ce_class_rescale = kwargs.pop('ce_class_rescale', None)
    if ce_class_rescale is None:
        debug_logger.debug("Using uniform class weights for Cross-Entropy.")

    ce_reduction = kwargs.pop('ce_reduction', None)
    if ce_reduction is None:
        debug_logger.warning("No 'ce_reduction' provided. Defaulting to 'mean' reduction for CE.")
        ce_reduction = 'mean'

    # 2. Resolve Dice Parameters
    lambda_dice = kwargs.pop('lambda_dice', None)
    if lambda_dice is None:
        debug_logger.warning("No 'lambda_dice' provided. Defaulting to 0.5.")
        lambda_dice = 0.5

    dice_class_rescale = kwargs.pop('dice_class_rescale', None)
    if dice_class_rescale is None:
        debug_logger.warning("No 'dice_class_rescale' provided. Defaulting to 1.0.")
        dice_class_rescale = torch.tensor(1.0)
    
    # 3. Resolve Phase Regularization Parameters
    lambda_phase_align = kwargs.pop('lambda_phase_align', None)
    if lambda_phase_align is None:
        debug_logger.warning("No 'lambda_phase_align' provided. Defaulting to 0.5.")
        lambda_phase_align = 0.5

    lambda_phase_tv = kwargs.pop('lambda_phase_tv', None)
    if lambda_phase_tv is None:
        debug_logger.warning("No 'lambda_phase_tv' provided. Defaulting to 0.5.")
        lambda_phase_tv = 0.5
    
    # 4. Component Instantiation
    debug_logger.debug("Instantiating atomic loss components: CE, Dice, PhaseAlign, PhaseTV.")
    ce = nn.CrossEntropyLoss(
        weight=ce_class_rescale,
        reduction=ce_reduction,
        **kwargs
    )
    dice = DiceLoss(smooth=1e-5)
    phase_align = PhaseAlignLoss(**kwargs)
    phase_tv = PhaseTVLoss(**kwargs)
    
    # 5. Final Aggregation
    semantic_logger.info(f"Loss Balance: CE({lambda_ce}) + Dice({lambda_dice}) + Align({lambda_phase_align}) + TV({lambda_phase_tv})")
    return WeightedSumLoss(
        losses=[ce, dice, phase_align, phase_tv], 
        weights=[lambda_ce, lambda_dice, lambda_phase_align, lambda_phase_tv],
    )
