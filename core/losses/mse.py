# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: MSE Loss
=========================================================

This module implements the Mean Squared Error (MSE) loss element. It serves as 
a standard regression objective, calculating the average squared difference 
between predicted and target tensors, commonly used for signal reconstruction 
and denoising tasks.
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

# Implementation relies on standard PyTorch nn.MSELoss

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# BUILDERS
# ---------------------------------------------------------------------

# -- mse_standard_builder
def mse_standard_builder(**kwargs) -> nn.MSELoss:
    """
    Builder function that instantiates a standard PyTorch MSE loss.

    Parameters
    ----------
    **kwargs : Any
        Dynamic keyword arguments passed during the build process, 
        including 'reduction' strategies.

    Returns
    -------
    nn.MSELoss
        An instance of the Mean Squared Error loss module.
    """
    # ------------------------------------------------

    semantic_logger.info("Builder: Initializing standard MSE Loss.")
    
    # Extract reduction
    reduction = kwargs.pop('reduction', None)
    if reduction is None:
        debug_logger.warning("No 'reduction' provided. Defaulting to 'mean' reduction.")
        reduction = 'mean'

    return nn.MSELoss(
        reduction=reduction,
        **kwargs
    )

