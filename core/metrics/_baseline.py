# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Metric Base
============================================================

This module provides the abstract base class for metric collections. It enforces 
a uniform interface for updating, computing, and extracting tensors from complex 
model outputs, ensuring that results are returned as clean dictionaries of 
standard Python scalars for downstream logging and analysis.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging
import torch

from torchmetrics import Metric
from torchmetrics import MetricCollection

from abc import ABC
from typing import Dict, Optional, Union, Tuple

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

# == BaseMetric ==
class BaseMetric(Metric, ABC):
    """
    Abstract base class for grouped metric implementations.

    This class provides a wrapper around `torchmetrics.MetricCollection` to 
    standardize how multiple metrics are updated and computed. It specifically 
    handles the conversion of GPU tensors to CPU scalars for clean reporting 
    and provides robust tensor extraction from dictionary-based model outputs.

    Attributes
    ----------
    collection : Optional[MetricCollection]
        The internal collection of metrics to be tracked and calculated.
    """
    
    # ===========

    # -- __init__
    def __init__(self):
        """
        Initializes the base metric object.
        """
        # ------------------------------------------------

        super().__init__()
        self.collection: Optional[MetricCollection] = None  # To be populated by children (MetricCollection)
        
        debug_logger.debug("BaseMetric base initialized. Collection placeholder established.")


    # -- update
    def update(self, *args, **kwargs):
        """
        Routes data updates to the internal metric collection.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to the collection update.
        **kwargs : Any
            Keyword arguments passed to the collection update.
        """
        # ------------------------------------------------

        # Pass data to the internal collection if it exists
        if self.collection:
            debug_logger.debug(f"Updating internal MetricCollection with args count: {len(args)}")
            self.collection.update(*args, **kwargs)
        else:
            debug_logger.warning("Update called on BaseMetric with no initialized collection.")


    # -- compute
    def compute(self) -> Dict[str, float]:
        """
        Calculates final metrics and resets internal states.

        Returns
        -------
        Dict[str, float]
            A dictionary where keys are metric names and values are Python floats.
        """
        # ------------------------------------------------

        debug_logger.info("Computing final metrics from current state.")

        # Check for initialized collection
        if not self.collection:
            debug_logger.debug("Compute called on empty collection; returning empty dictionary.")
            return {}
        
        # 1. Compute raw results (Tensors)
        results = self.collection.compute()
        
        # 2. Reset state for the next epoch/batch
        self.collection.reset()
        debug_logger.debug("Metric collection results computed and state reset.")
        
        # 3. Convert to clean floats for JSON/Tensorboard/Reporting
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                clean_results[k] = v.item()
            else:
                clean_results[k] = v
                
        debug_logger.debug(f"Cleaned results: {list(clean_results.keys())}")
        return clean_results
    
    
    # -- get_tensors
    def get_tensors(self, preds: Union[Dict[str, torch.Tensor], torch.Tensor], target: Union[dict, torch.Tensor], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resolves predictions and targets from raw inputs or output dictionaries.

        Parameters
        ----------
        preds : Union[Dict[str, torch.Tensor], torch.Tensor]
            The model predictions, potentially wrapped in a dictionary under 'preds'.
        target : Union[dict, torch.Tensor]
            The ground truth data, potentially wrapped in a dictionary under 'target'.
        **kwargs : Any
            Additional metadata for tensor extraction.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A pair containing (extracted_predictions, extracted_targets).

        Raises
        ------
        ValueError
            If expected keys are missing from dictionary inputs.
        """
        # ------------------------------------------------

        debug_logger.info("Extracting tensors for metric calculation.")

        # 1. Prediction Extraction Logic
        if isinstance(preds, dict):
            debug_logger.debug("Predictions provided as dictionary; searching for 'preds' key.")
            extracted_preds = preds.get('preds', None)
            if extracted_preds is None: 
                debug_logger.warning("No 'preds' key found in output_dict for RestorationMetrics. Skipping update.")
                raise ValueError("No 'preds' key found in output_dict for RestorationMetrics.")
        else:
            extracted_preds = preds

        # 2. Target Extraction Logic
        if isinstance(target, dict):
            debug_logger.debug("Targets provided as dictionary; searching for 'target' key.")
            extracted_target = target.get('target', None)
            if extracted_target is None: 
                debug_logger.warning("No 'target' key found in output_dict for RestorationMetrics. Skipping update.")
                raise ValueError("No 'target' key found in output_dict for RestorationMetrics.")
        else:
            extracted_target = target
        
        debug_logger.debug(f"Tensors successfully extracted. Preds shape: {extracted_preds.shape}")
        return extracted_preds, extracted_target
    