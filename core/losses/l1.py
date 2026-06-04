# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: L1 Loss
=========================================================

This module implements the Mean Absolute Error (L1) loss element. It calculates 
the average absolute difference between predicted and target tensors. It is 
widely used in image restoration tasks where robustness to outliers is 
preferred over the squared penalty of MSE.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging
import torch.nn as nn
from typing import Final, Optional, Callable, Any, Dict

# Import custom libraries
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

# Implementation relies on standard PyTorch nn.L1Loss

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# BUILDERS
# ---------------------------------------------------------------------

# -- l1_standard_builder
def l1_standard_builder(**kwargs) -> nn.L1Loss:
    """
    Builder function that instantiates a standard PyTorch L1 (MAE) loss.

    Parameters
    ----------
    **kwargs : Any
        Dynamic keyword arguments passed during the build process, 
        including 'reduction' strategies.

    Returns
    -------
    nn.L1Loss
        An instance of the Mean Absolute Error loss module.
    """
    
    # ------------------------------------------------

    semantic_logger.info("Builder: Initializing standard L1 (MAE) Loss.")
    
    # Extract reduction
    reduction = kwargs.pop('reduction', None)
    if reduction is None:
        debug_logger.warning("No 'reduction' provided. Defaulting to 'mean' reduction.")
        reduction = 'mean'

    debug_logger.debug(f"Configuring L1 loss with reduction: {reduction}")

    return nn.L1Loss(
        reduction=reduction,
        **kwargs
    )

