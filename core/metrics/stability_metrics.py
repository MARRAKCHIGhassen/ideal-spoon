# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Stability Metrics
==================================================================

This module implements metrics for quantifying feature stability and invariance. 
It measures the consistency of representations between reference and perturbed 
inputs using Relative Euclidean Distance and Cosine Similarity, which is 
critical for evaluating the robustness of scattering-inspired architectures.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging

import torch
from torchmetrics import Metric, MetricCollection, MeanMetric
from typing import Dict, Final, Optional, Union, Callable, cast, Any

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

# == StabilityMetrics ==
class StabilityMetrics(BaseMetric):
    """
    Evaluator for feature-level stability and transformation invariance.

    This suite computes the geometric distance and angular similarity between 
    feature vectors extracted from original images and their perturbed 
    counterparts. 

    Note: This metric expects model outputs containing 'features_ref' and 
    'features_pert' tensors rather than standard pixel-wise predictions.

    Attributes
    ----------
    collection : MetricCollection
        Internal container for 'relative_error' and 'cosine_similarity'.
    """
    
    # ===========

    # -- __init__
    def __init__(self, **unused: Any) -> None:
        """
        Initializes the stability metric collection.

        Parameters
        ----------
        **unused : Any
            Captured arguments for compatibility with builder dispatchers.
        """
        # ------------------------------------------------

        super().__init__()
        
        semantic_logger.info("Initializing Stability and Invariance metrics suite.")

        # 1. Metric Registration
        debug_logger.debug("Registering MeanMetric components for relative error and cosine similarity.")
        self.collection: MetricCollection = MetricCollection({
            "relative_error": MeanMetric(),
            "cosine_similarity": MeanMetric(),
        })
        
        debug_logger.debug("Stability MetricCollection established.")


    # -- update
    def update(self, preds: Union[Dict[str, torch.Tensor], torch.Tensor], target: Union[dict, torch.Tensor], **kwargs: Any) -> None:
        """
        Updates stability statistics using reference and perturbed features.

        Parameters
        ----------
        preds : Union[Dict[str, torch.Tensor], torch.Tensor]
            Model output containing the 'preds' (perturbed features) key.
        target : Union[dict, torch.Tensor]
            Reference output containing the 'target' (clean features) key.
        **kwargs : Any
            Additional metadata for tensor extraction.
        """
        # ------------------------------------------------

        # 1. Extract Feature Tensors
        # In Stability Protocols, preds/target are usually feature dictionaries
        extracted_preds, extracted_target = self.get_tensors(preds, target)

        semantic_logger.info("Computing stability metrics for the current feature batch.")
        debug_logger.debug(f"Feature extraction successful. Shape: {extracted_preds.shape}")

        # 2. Relative Error (L2 Norm)
        # Measures the normalized magnitude of the displacement vector
        debug_logger.debug("Calculating normalized L2 distance (Relative Error).")
        diff_norm = torch.norm(extracted_target - extracted_preds, p=2, dim=1)
        ref_norm = torch.norm(extracted_target, p=2, dim=1)
        rel_error = diff_norm / (ref_norm + 1e-8)  # Avoid division by zero
        
        # 3. Cosine Similarity
        # Measures the alignment of the feature vectors in the latent space
        debug_logger.debug("Calculating Latent Cosine Similarity.")
        cos_sim = torch.nn.functional.cosine_similarity(extracted_target, extracted_preds, dim=1)

        # 4. State Update
        self.collection["relative_error"].update(rel_error)
        self.collection["cosine_similarity"].update(cos_sim)
        
        debug_logger.debug(f"Batch metrics updated: mean_rel_err={rel_error.mean().item():.4f}")
