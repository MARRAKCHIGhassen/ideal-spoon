# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: L1 Phase Loss
==============================================================

This module implements a compound loss function combining the L1 (MAE) distance 
with phase-specific regularization terms: Phase Alignment (cross-scale) and 
Phase Total Variation (intra-scale smoothness). It is specifically designed 
for phase-aware signal reconstruction tasks.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging
import torch.nn as nn

from typing import Final, Optional, Callable, Any, Dict

# Import custom libraries
from ._baseline import WeightedSumLoss
from .components.phase_align_loss import PhaseAlignLoss
from .components.phase_tv_loss import PhaseTVLoss
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

# Implementation relies on a weighted sum of L1Loss, PhaseAlignLoss, and PhaseTVLoss.

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# BUILDERS
# ---------------------------------------------------------------------

# -- l1_phase_standard_builder
def l1_phase_standard_builder(**kwargs) -> WeightedSumLoss:
    """
    Constructs a compound objective: L1 + lambda_align * Align + lambda_tv * TV.

    This builder orchestrates the creation of the primary reconstruction loss 
    alongside phase-specific regularizers to ensure both spatial accuracy 
    and phase coherence.

    Parameters
    ----------
    **kwargs : Any
        Dynamic configuration keys including:
        - l1_reduction : str (default 'mean')
        - lambda_phase_align : float (default 0.5)
        - lambda_phase_tv : float (default 0.5)

    Returns
    -------
    WeightedSumLoss
        The aggregated loss module.
    """
    
    # ------------------------------------------------

    semantic_logger.info("Builder: Configuring L1 Reconstruction Loss with Phase Regularization.")
    
    # Extract reduction
    l1_reduction = kwargs.pop('l1_reduction', None)
    if l1_reduction is None:
        debug_logger.warning("No 'l1_reduction' provided. Defaulting to 'mean' reduction for L1.")
        l1_reduction = 'mean'
    
    lambda_phase_align = kwargs.pop('lambda_phase_align', None)
    if lambda_phase_align is None:
        debug_logger.warning("No 'lambda_phase_align' provided. Defaulting to 0.5.")
        lambda_phase_align = 0.5

    lambda_phase_tv = kwargs.pop('lambda_phase_tv', None)
    if lambda_phase_tv is None:
        debug_logger.warning("No 'lambda_phase_tv' provided. Defaulting to 0.5.")
        lambda_phase_tv = 0.5
    
    # Component Initialization
    debug_logger.debug(f"Initializing atomic components: L1(reduction={l1_reduction}), Align, and TV.")
    l1 = nn.L1Loss(
        reduction=l1_reduction,
        **kwargs
    )
    phase_align = PhaseAlignLoss(**kwargs)
    phase_tv = PhaseTVLoss(**kwargs)
    
    # Aggregation
    semantic_logger.info(f"Loss weights established: L1=1.0, Align={lambda_phase_align}, TV={lambda_phase_tv}")
    return WeightedSumLoss(
        losses=[l1, phase_align, phase_tv], 
        weights=[1.0, lambda_phase_align, lambda_phase_tv],
    )

