# -*- coding: utf-8 -*-
"""
Reflection Padding Layer.

This layer wraps PyTorch's padding functionality to ensure consistent boundary 
handling across the scattering transform. Reflection padding is critical for 
minimizing edge artifacts in Fourier-based convolutions.
"""

# Importing Global Libraries
from __future__ import annotations
import logging
from typing import Final, Union, Tuple, Callable

from torch import nn, Tensor
import torchvision.transforms.functional as TF

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

# == PaddingLayer ==
class PaddingLayer(nn.Module):
    """
    Applies padding to tensors to prepare for convolution or scattering.

    Defaults to Reflection Padding to maintain signal continuity at boundaries,
    which drastically reduces high-frequency artifacts in the Fourier domain
    compared to zero-padding.

    Attributes:
        pads (list[int]): Normalized padding [left, top, right, bottom].
        mode (str): Padding mode (default: "reflect").
    """
    
    # ===========

    # -- __init__
    def __init__(
        self,
        padding: int | list[int] = 4,
        mode: str = "reflect",
    ) -> None:
        """
        Initializes the PaddingLayer.

        Args:
            padding (int | list[int]): Padding size. 
                - int: symmetric padding on all sides.
                - list[2]: [pad_w, pad_h].
                - list[4]: [left, top, right, bottom].
            mode (str): Padding strategy ('reflect', 'replicate', 'constant').
        """
        
        # ------------------------------------------------

        #
        super().__init__()

        self.mode = mode
        
        # Normalize to [left, top, right, bottom]
        if isinstance(padding, int):
            self.pads = [padding] * 4
        elif len(padding) == 2:
            self.pads = [padding[1], padding[0], padding[1], padding[0]]
        elif len(padding) == 4:
            self.pads = padding
        else:
            raise ValueError(f"Unexpected padding values. Either 1, 2, or 4 values are accepted")
        

    # -- forward
    def forward(
            self,
            *tensors: Tensor,
            **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Apply padding to input tensors.
        
        Args:
            *tensors: Variable number of tensors (C, H, W) to pad.
            
        Returns:
            Tensor | Tuple[Tensor]: Padded tensor(s) of shape (C, H+2*pad, W+2*pad).
        """
        results = []
        for _, x in enumerate(tensors):
            results.append(TF.pad(x, self.pads, padding_mode=self.mode))
        return results[0] if len(results) == 1 else tuple(results)

    # -- __repr__
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"size (left, top, right, bottom) ={self.pads}, "
            f"mode='{self.mode}')"
        )