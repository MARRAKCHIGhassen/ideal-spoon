# -*- coding: utf-8 -*-
"""
Core Utilities and Data Structures.

This module provides foundational data structures and validation logic used
across the Phase-Aware Scattering Encoder-Decoder project. It primarily
defines the specifications for Wavelet Filter Banks.
"""

# Importing Global Libraries
from __future__ import annotations

import logging

import torch

from typing import NamedTuple, Optional

# Import custom libraries

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# LOGGERS
# ---------------------------------------------------------------------
# Configured loggers for distinct logging levels/purposes.
debug_logger = logging.getLogger(f"{__name__.split('.')[0]}.debug")
semantic_logger = logging.getLogger(f"{__name__.split('.')[0]}.semantic")

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# SPECIFICATIONS
# ---------------------------------------------------------------------

class GridBankOutput(NamedTuple):
    """
    Container for the discretized Wavelet Filter Bank tensors.
    
    This structure holds the actual numerical filters used to compute the
    scattering transform. It distinguishes between band-pass wavelets (psi)
    and low-pass scaling functions (phi).

    Attributes:
        psi (torch.Tensor | None): The band-pass wavelet filters.
            Shape: (Total_Filters, M, N) for 2D, or (Total_Filters, M, N, D) for 3D.
            These are complex-valued filters used to capture high-frequency details.
        
        phi_global (torch.Tensor | None): The global low-pass filter (scaling function).
            Shape: (1, M, N) [or ...D].
            Used for the final averaging (S0) or standard scattering smoothing.
            
        phi_local (torch.Tensor | None): Scale-dependent low-pass filters.
            Shape: (J, M, N) [or ...D].
            Used when `smoothing='scaled'` to prevent aliasing at intermediate scales
            without excessive blurring.
            
        indices_map (torch.Tensor | None): Mapping of filter indices to scales.
            Shape: (J, 2).
            Row `j` contains [start_index, end_index] for filters at scale `j` in `psi`.
            
        j_effective (torch.Tensor | None): The physical scales associated with the bank.
            Shape: (J,).
            Useful for debugging and verifying that the filter bank covers the
            expected frequency ranges.
    """
    
    # ===========

    #
    psi: Optional[torch.Tensor]         # (Total_Filters, M, N) - None if psi_type is None | (Total_Filters, M, N, D) for 3D (solid harmonics)
    phi_global: Optional[torch.Tensor]  # (1, M, N) - None if phi_type is None | (1, M, N, D) for 3D (solid harmonics)
    phi_local: Optional[torch.Tensor]   # (J, M, N) - None if smoothing is STANDARD | (J, M, N, D) for 3D (solid harmonics)
    indices_map: Optional[torch.Tensor] # (J, 2) - Map of start/end indices per scale
    j_effective: Optional[torch.Tensor] # (J,) - The actual physical scales used (for debugging/orchestration)

# -- validate_bank_output
def validate_bank_output(output: GridBankOutput) -> None:
    """
    Validates the type consistency of a GridBankOutput instance.

    Ensures that all populated fields in the GridBankOutput are valid PyTorch
    tensors. Raises a TypeError if validation fails.

    Args:
        output (GridBankOutput): The filter bank object to validate.

    Raises:
        TypeError: If any non-None field is not a torch.Tensor.
    """
    
    # ------------------------------------------------

    # psi
    if output.psi is not None and not isinstance(output.psi, torch.Tensor):
        raise TypeError(
            f"BankOutput.psi must be a torch.Tensor, "
            f"got {type(output.psi)!r}"
        )

    # phi_global
    if output.phi_global is not None and not isinstance(output.phi_global, torch.Tensor):
        raise TypeError(
            f"BankOutput.phi_global must be a torch.Tensor, "
            f"got {type(output.phi_global)!r}"
        )

    # phi_local
    if output.phi_local is not None and not isinstance(output.phi_local, torch.Tensor):
        raise TypeError(
            f"BankOutput.phi_local must be a torch.Tensor, "
            f"got {type(output.phi_local)!r}"
        )

    # indices_map
    if output.indices_map is not None and not isinstance(output.indices_map, torch.Tensor):
        raise TypeError(
            f"BankOutput.indices_map must be a torch.Tensor, "
            f"got {type(output.indices_map)!r}"
        )

    # j_effective
    if output.j_effective is not None and not isinstance(output.j_effective, torch.Tensor):
        raise TypeError(
            f"BankOutput.j_effective must be a torch.Tensor, "
            f"got {type(output.j_effective)!r}"
        )