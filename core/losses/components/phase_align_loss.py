# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Phase Align Loss
==================================================================

This module implements cross-scale phase alignment regularization. It ensures
consistency between phase maps at different scales by downsampling fine-scale
phases and calculating the angular distance to coarser-scale phase maps.
This enforces the physical property that phase evolution should be coherent
across the scattering transform's multiresolution hierarchy.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Final, Callable, Any

# Import custom libraries

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# LOGGERS
# ---------------------------------------------------------------------

debug_logger = logging.getLogger(f"{__name__.split('.')[0]}.debug")
semantic_logger = logging.getLogger(f"{__name__.split('.')[0]}.semantic")

# -------------------------------------------------------------------------------------------------------------------

# -- complex_pool_phase
def complex_pool_phase(phase: torch.Tensor, kernel_size=2, stride=2) -> torch.Tensor:
    """
    Downsamples a phase map correctly by operating in the complex domain.
    
    1. Projects phase to unit complex vectors: e^(i*phi).
    2. Averages the complex vectors (AvgPool).
    3. Recovers the phase of the average vector.
    
    This avoids invalid arithmetic averaging of angles (e.g., avg(0, 2pi) != pi).
    """
    z = torch.polar(torch.ones_like(phase), phase)
    pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    z_down = torch.complex(pool(z.real), pool(z.imag))
    return torch.angle(z_down)

# -- angular_distance
def angular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared Euclidean distance between unit vectors defined by angles a and b.
    Equivalent to 2(1 - cos(a-b)).
    """
    return torch.abs(torch.polar(torch.ones_like(a), a) - torch.polar(torch.ones_like(b), b))**2

# ---------------------------------------------------------------------
# IMPLEMENTATION
# ---------------------------------------------------------------------

# == PhaseAlignLoss ==
class PhaseAlignLoss(nn.Module):
    """
    Regularization layer for cross-scale phase alignment.

    This loss penalizes phase inconsistencies between adjacent scales of a
    multiresolution phase representation. It utilizes complex pooling to
    preserve phase integrity during downsampling.
    
    Mechanism:
        For each scale pair (j, j+1):
        1. Downsample phi_j (fine) to match phi_{j+1} (coarse).
        2. Calculate angular distance between phi_j_down and phi_{j+1}.
    """

    # ===========

    # -- __init__
    def __init__(self, **unused: object):
        """
        Initializes the phase alignment loss module.
        """
        # ------------------------------------------------

        super().__init__()
        debug_logger.debug(f"PhaseAlignLoss initialized")

    # -- forward
    def forward(self, prediction: Any, target: Any, **kwargs: Any) -> torch.Tensor:
        """
        Calculates the cross-scale phase alignment objective.

        Parameters
        ----------
        prediction : Any
            The model output, expected to be a dictionary containing 'phase_maps'
            (a list of phase tensors ordered from fine to coarse).
        target : Any
            The ground truth (unused in this regularization objective).

        Returns
        -------
        torch.Tensor
            Scalar loss representing the mean angular distance across scales.
        """
        # ------------------------------------------------

        # 1. Structural Validation
        if not isinstance(prediction, dict):
            debug_logger.debug("Prediction is not a dictionary; skipping alignment loss.")
            return torch.tensor(0.0, device=prediction.device)

        phase_maps = prediction.get('phase_maps')

        # 2. Logic Check
        if not phase_maps or len(phase_maps) < 2:
            debug_logger.debug("Insufficient phase maps for cross-scale alignment.")
            return torch.tensor(0.0, device=next(iter(prediction.values())).device)

        debug_logger.debug(f"Calculating phase alignment loss across {len(phase_maps)} scales.")
        loss_align = torch.tensor(0.0, device=phase_maps[0].device)

        # 3. Scale Interaction Loop
        for i in range(len(phase_maps) - 1):
            phi_fine = phase_maps[i]
            phi_coarse = phase_maps[i+1]

            debug_logger.debug(f"Aligning scale {i} (shape {phi_fine.shape}) to scale {i+1} (shape {phi_coarse.shape})")

            # Downsample fine phase in complex domain
            phi_fine_down = complex_pool_phase(phi_fine)

            # Spatial alignment for boundary handling
            h = min(phi_fine_down.shape[-2], phi_coarse.shape[-2])
            w = min(phi_fine_down.shape[-1], phi_coarse.shape[-1])

            # Accumulate angular distance
            dist = angular_distance(phi_fine_down[..., :h, :w], phi_coarse[..., :h, :w]).mean()
            loss_align += dist

        debug_logger.debug(f"Total alignment loss: {loss_align.item():.6f}")
        return loss_align

    # -- __repr__
    def __repr__(self) -> str:
        """
        Returns a string representation of the module state.

        Returns
        -------
        str
            State summary.
        """
        # ------------------------------------------------
        return (
            f"{self.__class__.__name__}("
        )
