# -*- coding: utf-8 -*-
"""
Unpadding (Cropping) Layer.

This layer restores the original spatial dimensions of a tensor by cropping 
the padded borders. It accounts for any total subsampling factor (stride) 
accumulated during the network pass.
"""

# Importing Global Libraries
from __future__ import annotations
import logging
from typing import Final, Union, Tuple, Callable

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

# == UnpaddingLayer ==
class UnpaddingLayer(nn.Module):
    """
    Removes padding from a tensor by slicing the central region.
    
    This is the inverse operation of PaddingLayer. It dynamically adjusts
    the crop size based on the `total_subsampling` factor (e.g., if the signal
    was downsampled by 2, we crop half the original padding pixels).
    """
    
    # ===========

    # -- __init__
    def __init__(
        self,
        padding: int | list[int] = 4,
        total_subsampling: int = 1,
    ) -> None:
        """
        Initializes the UnpaddingLayer.

        Args:
            padding (int | list[int]): The original padding size applied at input.
            total_subsampling (int): The stride/downsampling factor at this point in the network.
        """
        
        # ------------------------------------------------

        #
        super().__init__()

        self.total_subsampling = total_subsampling
        
        # Normalize to [left, top, right, bottom]
        if isinstance(padding, int):
            self.pads = [padding] * 4
        elif len(padding) == 2:
            # Assumed (H, W) input -> (left, right, top, bottom)
            # Correction: TF.pad usually takes (left, top, right, bottom). 
            # If input is [pad_h, pad_w], map to [w, h, w, h]
            self.pads = [padding[1], padding[0], padding[1], padding[0]]
        elif len(padding) == 4:
            self.pads = padding
        else:
            raise ValueError(f"Unexpected padding values. Either 1, 2, or 4 values are accepted")
        

    # -- forward
    def forward(self, x: Tensor) -> Tensor:
        """
        Slices the tensor to remove padding.
        
        Args:
            x: Tensor of shape (..., H_padded, W_padded).
            
        Returns:
            Tensor: Cropped tensor of shape (..., H, W).
        """
        # Calculate current padding at this scale
        current_pads = [p // self.total_subsampling for p in self.pads]
        
        # Unpack: (left, top, right, bottom)
        l, t, r, b = current_pads
        
        # Handle H dimension (Top/Bottom)
        h_start = t
        h_end = -b if b > 0 else None # Python slice x[:-0] is empty, x[:None] is full
        
        # Handle W dimension (Left/Right)
        w_start = l
        w_end = -r if r > 0 else None
        
        return x[..., h_start:h_end, w_start:w_end]