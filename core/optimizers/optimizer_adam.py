# -*- coding: utf-8 -*-
"""
Adam Optimizer Module.
======================

This module provides a wrapper for PyTorch's Adam and AdamW optimizers. 
It facilitates the construction of optimization algorithms with standard 
configurations or specialized L2 weight decay (AdamW) variants, integrated 
into the project's element specification system.
"""

# ---------------------------------------------------------------------
# GLOBAL LIBRARIES
# ---------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import Any, Tuple

import torch.optim as optim

# ---------------------------------------------------------------------
# CUSTOM LIBRARIES
# ---------------------------------------------------------------------

from ...constants import KIND_OPTIM
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

# == AdamOptimizer ==
class AdamOptimizer:
    """
    Adam optimizer wrapper factory.

    This class serves as a configuration container and factory for creating 
    PyTorch Adam or AdamW optimizer instances. The actual optimizer is 
    instantiated and bound to model parameters during the .build() phase.

    Attributes
    ----------
    lr : float
        The learning rate for the optimizer.
    betas : Tuple[float, float]
        Coefficients used for computing running averages of gradient and its square.
    weight_decay : float
        The weight decay (L2 penalty) coefficient.
    """
    
    # ===========

    # -- __init__
    def __init__(
        self,
        adam_lr: float = 0.001,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_weight_decay: float = 0.0,
        **unused: object,
    ) -> None:
        """
        Initializes the Adam configuration with the specified hyperparameters.

        Parameters
        ----------
        adam_lr : float, optional
            The learning rate. Default is 0.001.
        adam_betas : float, optional
            The first beta coefficient (beta1). Beta2 is fixed at 0.999. Default is 0.9.
        adam_weight_decay : float, optional
            The L2 penalty coefficient. Default is 0.0.
        **unused : object
            Additional arguments captured but not used by the optimizer.
        """
        # ------------------------------------------------

        debug_logger.debug(f"Initializing AdamOptimizer config: lr={adam_lr}, beta1={adam_betas}, weight_decay={adam_weight_decay}")
        
        self.lr = adam_lr
        self.betas = adam_betas  # (beta1, beta2)
        self.weight_decay = adam_weight_decay

    # -- build
    def build(self, model_params: Any) -> Any:
        """
        Instantiates the PyTorch optimizer bound to the provided model parameters.

        This method selects between Adam and AdamW based on the weight decay value.

        Parameters
        ----------
        model_params : Any
            The model parameters (typically model.parameters() or a list of param groups).

        Returns
        -------
        Any
            An instantiated instance of optim.Adam or optim.AdamW.
        """
        # ------------------------------------------------

        # 1. Selection Logic based on Weight Decay
        if self.weight_decay > 0:
            semantic_logger.info(f"Building AdamW optimizer (Weight Decay: {self.weight_decay})")
            debug_logger.debug(f"Instantiating optim.AdamW with lr={self.lr}, betas={self.betas}")
            return optim.AdamW(
                model_params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        else:
            semantic_logger.info("Building standard Adam optimizer.")
            debug_logger.debug(f"Instantiating optim.Adam with lr={self.lr}, betas={self.betas}")
            return optim.Adam(
                model_params,
                lr=self.lr,
                betas=self.betas,
            )
