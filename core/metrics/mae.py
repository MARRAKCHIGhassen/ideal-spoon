# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: MAE Metric
===========================================================

This module implements the Mean Absolute Error (MAE) metric. It is used to 
evaluate the performance of regression models by calculating the average 
magnitude of errors between predictions and targets, without considering 
their direction.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Final, Optional, Callable, Any, Dict

from torchmetrics import Metric

# Import custom libraries
from ...constants import KIND_METRIC

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

# == MAEMetric ==
class MAEMetric(Metric):
    """
    Mean Absolute Error (MAE) metric implementation using TorchMetrics.

    Tracks the sum of absolute errors and the total number of elements 
    across batches to compute the global average error.

    Attributes
    ----------
    sum_abs_error : torch.Tensor
        Accumulated sum of absolute differences.
    total : torch.Tensor
        Total count of elements processed.
    """
    
    # ===========

    # -- __init__
    def __init__(self):
        """
        Initializes the metric state variables.
        """
        # ------------------------------------------------

        super().__init__()
        
        # 1. Register state variables (TorchMetrics handles device syncing automatically)
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
        debug_logger.debug("MAEMetric state initialized.")


    # -- update
    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        Updates the metric state with new predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        target : torch.Tensor
            Ground truth reference values.
        """
        # ------------------------------------------------

        # 2. Accumulate state
        debug_logger.debug(f"Updating MAE state with batch of shape: {preds.shape}")
        
        abs_diff = torch.abs(preds - target)
        self.sum_abs_error += abs_diff.sum()
        self.total += abs_diff.numel()


    # -- compute
    def compute(self) -> torch.Tensor:
        """
        Computes the final Mean Absolute Error.

        Returns
        -------
        torch.Tensor
            The calculated MAE scalar.
        """
        # ------------------------------------------------

        semantic_logger.info("Computing final Mean Absolute Error (MAE).")
        
        # 3. Compute final metric
        result = self.sum_abs_error / self.total # type: ignore
        
        debug_logger.debug(f"Computed MAE: {result.item():.6f}")
        
        return result # type: ignore
