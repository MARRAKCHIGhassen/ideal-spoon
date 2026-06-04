# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Restoration Metrics
====================================================================

This module provides metrics for evaluating image restoration performance. 
It encapsulates standard signal fidelity measures like PSNR and SSIM, with 
optional support for heavy perceptual metrics like LPIPS, ensuring 
reproducible benchmarking for denoising and super-resolution tasks.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.image import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure, 
    LearnedPerceptualImagePatchSimilarity
)
from typing import Dict, Final, Optional, Tuple, Union, Callable, cast

# Import custom libraries
from ._baseline import BaseMetric

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

# == RestorationMetrics ==
class RestorationMetrics(BaseMetric):
    """
    Evaluation suite for image restoration and signal recovery.

    This class aggregates multiple metrics into a single collection, providing 
    automated tensor extraction and range-safe processing for Peak 
    Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), and 
    Learned Perceptual Image Patch Similarity (LPIPS).

    Attributes
    ----------
    data_range : float
        The dynamic range of the input data (usually 1.0 for normalized tensors).
    collection : MetricCollection
        The internal container for the tracked metrics.
    """
    
    # ===========

    # -- __init__
    def __init__(
        self,
        data_range: float = 1.0,
        include_lpips: bool = False,
        **unused
    ) -> None:
        """
        Initializes the restoration metrics collection.

        Parameters
        ----------
        data_range : float, default=1.0
            The expected maximum value of the signal (e.g., 1.0 or 255.0).
        include_lpips : bool, default=False
            If True, initializes Learned Perceptual Image Patch Similarity. 
            Note that this requires downloading external weights (AlexNet).
        **unused : Any
            Captured arguments for compatibility with builder dispatchers.
        """
        # ------------------------------------------------

        super().__init__()

        semantic_logger.info("Initializing restoration metric collection.")
        debug_logger.debug(f"RestorationMetrics params -> data_range: {data_range}, include_lpips: {include_lpips}")

        # 1. State Configuration
        _raw_metrics = {}
        self.data_range = data_range
        
        # 2. Standard Metrics Registration
        debug_logger.debug("Registering standard signal fidelity metrics: PSNR, SSIM.")
        _raw_metrics = {
            "psnr": PeakSignalNoiseRatio(data_range=data_range),
            "ssim": StructuralSimilarityIndexMeasure(data_range=data_range)
        }
        
        # 3. Optional Perceptual Metrics
        # LPIPS is heavy (downloads AlexNet/VGG), so we make it optional
        if include_lpips:
            semantic_logger.info("Enabling perceptual LPIPS metric (AlexNet backbone).")
            # normalize=True assumes inputs are in [0,1] and scales them to [-1,1] for the model
            _raw_metrics["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)
        
        self.collection = MetricCollection(cast(Dict[str, Union[Metric, MetricCollection]], _raw_metrics))
        debug_logger.debug("MetricCollection successfully instantiated.")

    
    # -- update
    def update(self, preds: Union[Dict[str, torch.Tensor], torch.Tensor], target: Union[dict, torch.Tensor], **kwargs) -> None:
        """
        Processes a batch of predictions and targets to update metric states.

        Parameters
        ----------
        preds : Union[Dict[str, torch.Tensor], torch.Tensor]
            Predicted tensors or a dictionary containing the 'preds' key.
        target : Union[dict, torch.Tensor]
            Ground truth tensors or a dictionary containing the 'target' key.
        **kwargs : Any
            Additional metadata passed to the extraction logic.
        """
        # ------------------------------------------------

        # 1. Extract Tensors
        extracted_preds, extracted_target = self.get_tensors(preds, target)
        debug_logger.debug(f"Updating restoration metrics for batch shape: {extracted_preds.shape}")

        # 2. Dynamic Clamping
        # Clamp to valid range to prevent NaN in PSNR
        debug_logger.debug(f"Clamping tensors to data_range: [0.0, {self.data_range}]")
        extracted_preds = torch.clamp(extracted_preds, 0.0, self.data_range)
        extracted_target = torch.clamp(extracted_target, 0.0, self.data_range)
        
        # 3. State Accumulation
        if self.collection:
            self.collection.update(extracted_preds, extracted_target)
