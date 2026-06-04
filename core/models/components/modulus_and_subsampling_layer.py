# -*- coding: utf-8 -*-
"""
Modulus and Subsampling Layer.

This module implements the non-linear operator of the Scattering Transform.
It computes the complex modulus (magnitude) of filtered coefficients and 
optionally performs spatial downsampling (stride).
"""

# Importing Global Libraries
from __future__ import annotations
import logging
from typing import Final, Callable

import torch
from torch import nn, Tensor

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

# == ModulusSubsampleLayer ==
class ModulusSubsampleLayer(nn.Module):
    r"""
    Computes the Complex Modulus and Subsamples the result.

    Operation: $|x| \downarrow k$
    
    Key Features:
    * **Numerical Stability**: Adds a small epsilon inside the square root 
        to ensure gradients do not explode at zero.
    * **Fused Operation**: Handles both magnitude computation and striding 
        efficiently.
    """
    
    # ===========

    # -- __init__
    def __init__(
        self,
    ) -> None:
        super().__init__()

    # -- forward
    def forward(self, x: Tensor, k: int = 1) -> Tensor:
        """
        Executes L2 Norm + Subsampling.

        Args:
            x (Tensor): Input tensor. Can be complex (..., H, W) or 
                        real with last dim 2 (..., H, W, 2).
            k (int): Subsampling stride (default: 1).

        Returns:
            Tensor: Real-valued magnitude tensor, optionally downsampled.
        """

        # 1. Modulus (Magnitude)
        # Using a small epsilon for gradient stability
        if torch.is_complex(x):
            # Complex Tensor: manually compute magnitude to add epsilon
            mod = torch.sqrt(x.real.pow(2) + x.imag.pow(2) + 1e-8)
        else:
            # Fallback for old (..., 2) format
            mod = torch.sqrt(x[..., 0].pow(2) + x[..., 1].pow(2) + 1e-8)

        # 2. Subsample
        # If k=1, this is a no-op slice which Inductor optimizes away
        return mod[..., ::k, ::k] if k > 1 else mod