# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Dice Metric Component
======================================================================

This module implements the Dice Coefficient (also known as the F1 Score) 
as a cumulative metric. It is designed to handle both binary and multiclass 
segmentation masks, accumulating intersection and union statistics over 
an entire epoch to compute a global score.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging

import torch
from torchmetrics import Metric
from typing import Final, Callable, Literal, Optional

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

# == Dice ==
class Dice(Metric):
    """
    Computes the Dice Coefficient (F1 Score) for semantic segmentation.

    This metric measures the overlap between the predicted segmentation mask 
    and the ground truth. It supports:
    1. **Binary Mode**: Applies a sigmoid threshold to logits.
    2. **Multiclass Mode**: Applies argmax to logits.

    Attributes
    ----------
    dice_sum : torch.Tensor
        Accumulated sum of dice scores for processed batches.
    num_batches : torch.Tensor
        Count of batches processed (for averaging).
    """
    
    full_state_update: bool = False
    
    # ===========

    # -- __init__
    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: int,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        average: Optional[Literal["macro", "micro", "weighted", "none"]] = None,
    ) -> None:
        """
        Initializes the Dice metric state.

        Parameters
        ----------
        num_classes : int
            Number of classes in the segmentation task.
        threshold : float, default=0.5
            Probability threshold for binary classification.
        task : str, default='binary'
            Task type determining the reduction strategy.
        average : str, optional
            Averaging method for multiclass scores (default: 'macro').
        ignore_index : int, optional
            Class index to exclude from computation (e.g., background).
        """
        # ------------------------------------------------

        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.average = average
        
        # State variables for accumulation
        self.add_state("dice_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def _binary_dice(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes Dice score for binary masks (Class 1 vs Background)."""
        # preds, target: [B, H, W] with {0,1}
        preds_fg = preds == 1
        target_fg = target == 1

        preds_flat = preds_fg.view(preds_fg.size(0), -1)
        target_flat = target_fg.view(target_fg.size(0), -1)

        intersection = (preds_flat & target_flat).sum(dim=1).float()
        union = preds_flat.sum(dim=1).float() + target_flat.sum(dim=1).float()

        dice_per_sample = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return dice_per_sample.mean()
    
    def _multiclass_dice(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes Dice score per class and averages them (Macro average).
        
        Args:
            preds: Class indices (B, H, W).
            target: Class indices (B, H, W).
        """
        # preds, target: [B, H, W] with class indices 0..C-1
        C = self.num_classes
        ignore = self.ignore_index

        dices = []
        for c in range(C):
            if ignore is not None and c == ignore:
                continue

            preds_c = preds == c
            target_c = target == c

            preds_flat = preds_c.view(preds_c.size(0), -1)
            target_flat = target_c.view(target_c.size(0), -1)

            intersection = (preds_flat & target_flat).sum(dim=1).float()
            union = preds_flat.sum(dim=1).float() + target_flat.sum(dim=1).float()

            dice_c = (2.0 * intersection + 1e-6) / (union + 1e-6)  # [B]
            dices.append(dice_c.mean())  # scalar per class

        if not dices:
            return torch.tensor(0.0, device=preds.device)

        dices = torch.stack(dices)  # [C_eff]

        if self.average == "macro" or self.average is None:
            return dices.mean()
        elif self.average == "none":
            # Return mean over batch but keep per-class; here we just aggregate as macro.
            return dices
        else:
            # For brevity, micro/weighted are not expanded;
            # if needed you can weight by support.
            return dices.mean()
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        """
        Updates the accumulated statistics with a new batch.

        Handles shape normalization:
        - Binary: [B, 1, H, W] -> Squeeze -> Threshold.
        - Multiclass: [B, C, H, W] -> Argmax -> [B, H, W].
        """
        if self.task == "binary":
            if preds.dim() == 4 and preds.size(1) == 1:
                preds = preds.squeeze(1)
            if preds.dtype.is_floating_point:
                preds = (torch.sigmoid(preds) >= self.threshold).long()
            if target.dim() == 4 and target.size(1) == 1:
                target = target.squeeze(1)
            target = target.long()
            dice_val = self._binary_dice(preds, target)
        elif self.task == "multiclass":
            if preds.dim() == 4:
                preds = preds.argmax(dim=1)
            if target.dim() == 4 and target.size(1) == 1:
                target = target.squeeze(1)
            preds = preds.long()
            target = target.long()
            dice_val = self._multiclass_dice(preds, target)
        else:
            raise NotImplementedError("Multilabel Dice not implemented in this snippet")

        self.dice_sum = self.dice_sum + dice_val.detach()
        self.num_batches = self.num_batches + 1

    def compute(self) -> torch.Tensor:  # type: ignore[override]
        """
        Calculates the final averaged Dice score over all processed batches.
        """
        if self.num_batches == 0:
            return torch.tensor(0.0, device=self.dice_sum.device)
        return self.dice_sum / self.num_batches
    