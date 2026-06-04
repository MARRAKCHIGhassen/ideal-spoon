# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Segmentation Metrics
=====================================================================

This module implements a flexible suite of metrics for semantic segmentation. 
It supports binary, multiclass, and multilabel tasks, providing configurable 
averaging strategies and overlap-based measures such as Dice and IoU, as well 
as distance-based measures like Hausdorff Distance.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging

import torch

from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import JaccardIndex, Precision, Recall
from torchmetrics.segmentation import HausdorffDistance
from typing import Dict, Final, List, Literal, Optional, Union, Callable, cast

# Import custom libraries
from ._baseline import BaseMetric

from .components.dice import Dice

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

# == SegmentationMetrics ==
class SegmentationMetrics(BaseMetric):
    """
    Highly flexible evaluator for semantic segmentation performance.

    This class encapsulates multiple TorchMetrics, allowing for dynamic 
    configuration of tasks (binary, multiclass, multilabel), averaging 
    methods, and specific metrics like Dice Score, IoU (Jaccard Index), 
    Precision, and Recall.

    Attributes
    ----------
    collection : MetricCollection
        The internal container for the registered metrics.
    """
    
    # ===========

    # -- __init__
    def __init__(
        self,
        task: str,
        num_classes: int,
        metrics_list: List[str],
        averages_list: List[str],
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        data_range: float = 1.0,
        **unused
    ) -> None:
        """
        Initializes the segmentation metrics collection.

        Parameters
        ----------
        task : str
            The type of segmentation task: 'binary', 'multiclass', or 'multilabel'.
        num_classes : int
            Number of classes (including background).
        metrics_list : List[str]
            List of metric names to include (e.g., ['dice', 'iou']).
        averages_list : List[str]
            List of averaging strategies for multiclass/multilabel tasks.
        threshold : float, default=0.5
            Probability threshold for binarization.
        ignore_index : Optional[int], default=None
            Class index to ignore in metric calculations.
        **unused : Any
            Additional arguments for compatibility.
        """
        # ------------------------------------------------

        super().__init__()
        
        semantic_logger.info(f"Initializing segmentation metrics for {task} task.")
        debug_logger.debug(f"Params -> classes: {num_classes}, metrics: {metrics_list}, averages: {averages_list}")
        self.num_classes = num_classes
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.data_range = data_range
        _raw_metrics = {}
        task_lit = cast(Literal["binary", "multiclass", "multilabel"], task)

        # -- _get_metric_instance
        def _get_metric_instance(name: str, avg: Optional[str] = None, **kwargs) -> Metric:
            """Internal helper to instantiate specific metric classes."""
            # ------------------------------------------------
            # For binary, average is typically ignored/handled internally by torchmetrics (uses threshold)
            # For multiclass, average is mandatory
            avg_arg = cast(Literal["macro", "micro", "weighted", "none"], avg) if task != "binary" else None 
            
            if name == "dice":
                return Dice(
                    task=task_lit,
                    num_classes=num_classes,
                    threshold=threshold,
                    ignore_index=ignore_index,
                    average=avg_arg,
                )
            elif name == "iou":
                return JaccardIndex(
                    task=task_lit,
                    num_classes=num_classes,
                    threshold=threshold,
                    average=avg_arg,
                    ignore_index=ignore_index,
                )
            elif name == "precision":
                return Precision(
                    task=task_lit,
                    num_classes=num_classes,
                    threshold=threshold,
                    average=avg_arg,
                    ignore_index=ignore_index,
                )
            elif name == "recall":
                return Recall(
                    task=task_lit,
                    num_classes=num_classes,
                    threshold=threshold,
                    average=avg_arg,
                    ignore_index=ignore_index,
                )
            elif name == "hausdorff":
                # TorchMetrics HausdorffDistance uses segmentation API.
                # For semantic segmentation, `input_format="index"` matches your logits->argmax usage.
                return HausdorffDistance(
                    num_classes=num_classes,
                    include_background=(ignore_index is None),
                    input_format="index",
                )
            else:
                raise ValueError(f"Unknown metric: {name}")
        
        # 1. Build Collection
        for m_name in metrics_list:
            debug_logger.debug(f"Registering metric: {m_name}")
            if m_name == "hausdorff":
                # Hausdorff usually doesn't have "micro/macro" in the same way, 
                # we add it once as 'hausdorff'.
                _raw_metrics["hausdorff"] = _get_metric_instance("hausdorff", None)
                continue

            if task == "binary":
                # Binary doesn't use the averaging list (it produces one scalar)
                _raw_metrics[m_name] = _get_metric_instance(m_name, None)
            else:
                # Multiclass/Multilabel iterate over requested averages
                for avg in averages_list:
                    suffix = f"_{avg}" if avg != "none" else ""
                    key = f"{m_name}{suffix}"
                    _raw_metrics[key] = _get_metric_instance(m_name, avg)
        
        self.collection = MetricCollection(cast(Dict[str, Union[Metric, MetricCollection]], _raw_metrics))
        debug_logger.debug("Segmentation MetricCollection established.")

    def update(self, preds: torch.Tensor, target: torch.Tensor, **kwargs) -> None:
        # 1) Extract tensors if dicts (BaseMetric already has get_tensors if you need it)
        preds, target = self.get_tensors(preds, target)

        # 2) Shape normalization for preds
        if preds.dim() == 4:
            # Case: logits [B, C, H, W]
            if preds.size(1) == self.num_classes:
                # multiclass / binary with 2 channels -> indices [B, H, W]
                preds = preds.argmax(dim=1)
            elif preds.size(1) == 1:
                # single-channel logits -> threshold to binary indices [B, H, W]
                if preds.is_floating_point():
                    preds = (torch.sigmoid(preds) >= 0.5).long()
                preds = preds.squeeze(1)

        # logits [B, C, H, W] -> indices [B, H, W]
        if preds.dim() == 4 and preds.size(1) == self.collection["iou"].num_classes: # type: ignore
            preds = preds.argmax(dim=1)

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        if preds.is_floating_point():
            preds = torch.clamp(preds, 0.0, self.data_range)
        if target.is_floating_point():
            target = torch.clamp(target, 0.0, self.data_range)
        
        if self.collection:
            self.collection.update(preds, target)
