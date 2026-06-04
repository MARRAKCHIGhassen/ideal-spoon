# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Phase TV Loss
==============================================================

This module implements Total Variation (TV) regularization for phase maps.
It enforces intra-scale spatial smoothness by penalizing angular differences
between adjacent pixels in both horizontal and vertical directions.
This helps reduce high-frequency noise in the predicted phase information.
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

# -- angular_distance
def angular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calculates squared angular distance, robust to phase wrap-around."""
    return torch.abs(torch.polar(torch.ones_like(a), a) - torch.polar(torch.ones_like(b), b))**2

# ---------------------------------------------------------------------
# IMPLEMENTATION
# ---------------------------------------------------------------------

# == PhaseTVLoss ==
class PhaseTVLoss(nn.Module):
    r"""
    Intra-scale phase smoothness regularization.

    This objective applies Total Variation (TV) to the phase domain, utilizing
    angular distance to respect the circular nature of phase values.
    
    Objective:
        $$ L_{TV} = \sum_{x,y} |e^{i\phi(x+1,y)} - e^{i\phi(x,y)}|^2 + |e^{i\phi(x,y+1)} - e^{i\phi(x,y)}|^2 $$
    """

    # ===========

    # -- __init__
    def __init__(self, **unused: object):
        """
        Initializes the Phase TV loss module.
        """
        # ------------------------------------------------

        super().__init__()
        debug_logger.debug(f"PhaseTVLoss initialized")

    # -- forward
    def forward(self, prediction: Any, target: Any, **kwargs: Any) -> torch.Tensor:
        """
        Calculates the phase Total Variation objective.

        Parameters
        ----------
        prediction : Any
            The model output, expected to be a dictionary containing 'phase_maps'.
        target : Any
            The ground truth (unused in this regularization objective).

        Returns
        -------
        torch.Tensor
            Scalar loss representing the accumulated spatial roughness.
        """
        # ------------------------------------------------

        # 1. Structural Validation
        if not isinstance(prediction, dict):
            debug_logger.debug("Prediction is not a dictionary; skipping TV loss.")
            return torch.tensor(0.0, device=prediction.device)

        phase_maps = prediction.get('phase_maps')

        # 2. Logic Check
        if not phase_maps:
            debug_logger.debug("No phase maps found in prediction.")
            return torch.tensor(0.0, device=next(iter(prediction.values())).device)

        debug_logger.debug(f"Applying phase Total Variation to {len(phase_maps)} maps.")
        loss_tv = torch.tensor(0.0, device=phase_maps[0].device)

        # 3. Spatial Gradient Loop
        for phi in phase_maps:
            debug_logger.debug(f"Processing TV for phase map of shape {phi.shape}")

            # Calculate angular gradients (Horizontal and Vertical)
            # Slice [:, 1:] and [:, :-1] to compute neighbor differences
            dx = angular_distance(phi[..., :, 1:], phi[..., :, :-1])
            dy = angular_distance(phi[..., 1:, :], phi[..., :-1, :])

            # Accumulate mean roughness
            loss_tv += (dx.mean() + dy.mean())

        debug_logger.debug(f"Total Phase TV loss: {loss_tv.item():.6f}")
        return loss_tv

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
