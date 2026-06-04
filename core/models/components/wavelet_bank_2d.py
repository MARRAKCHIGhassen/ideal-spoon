# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Wavelet Filter Banks
====================================================================

This module provides a high-performance implementation of 2D Morlet filter 
banks. It leverages vectorized broadcasting to generate complex wavelet 
kernels in the Fourier domain, supporting multiresolution analysis for 
scattering transforms and directional feature extraction.
"""

# Importing Global Libraries
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from typing import Callable, Tuple, Final

# Import custom libraries
from ....utils.utils import GridBankOutput

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

# == GridBank2D ==
class GridBank2D(nn.Module):
    """
    Vectorized Morlet Filter Grid Bank for 2D Multiresolution Analysis.

    This class generates a complete stack of directional Morlet wavelets and 
    scaling functions. By using broadcasting operations, it avoids expensive 
    Python loops during initialization and produces L2-normalized kernels 
    directly in the Fourier domain for efficient convolution.

    Parameters
    ----------
    M : int
        The height (number of rows) of the filter kernels.
    N : int
        The width (number of columns) of the filter kernels.
    J : int
        The number of scales (octaves) in the multiresolution decomposition.
    L : int
        The number of orientations per scale.
    slant : float, default=0.5
        The ellipticity of the Morlet wavelet. A value of 1.0 results in an 
        isotropic kernel.
    smoothing_mode : str, default='standard'
        Determines the scaling function (low-pass) strategy:
        - 'standard': Uses a single global scaling function at the largest scale.
        - 'scale': Generates a unique scaling function for every scale $j$.
    """
    
    # ===========

    # -- __init__
    def __init__(
        self,
        M: int, N: int,
        J: int, L: int,
        slant: float = 0.5,
        smoothing_mode: str = 'standard', # 'standard' (Global S0), 'scale' (Local S0 per j)
    ):
        """
        Initializes the filter bank by pre-computing kernels and metadata.
        """
        # ------------------------------------------------
        super().__init__()
        
        # 1. Internal State Assignment
        semantic_logger.info(f"Initializing GridBank2D: Scales={J}, Orientations={L}, Resolution={M}x{N}")
        debug_logger.debug(f"Parameters: slant={slant}, smoothing_mode={smoothing_mode}")

        self.M, self.N = M, N
        self.J, self.L = J, L
        self.slant = slant
        self.smoothing_mode = smoothing_mode.lower()
        
        # 2. Build Physical Parameter Grids
        # We need parameter vectors of shape (K,) where K = J * Slants * L
        # -------------------------------------------------------------------
        
        # A. Scales (J)
        # ----------------------------
        # Standard Dyadic Grid: 0, 1, ..., J-1
        j_values = torch.arange(J, dtype=torch.float32)
        self.register_buffer("j_effective", j_values)

        # B. Geometry (Theta, Slant)
        # ----------------------------
        # Grid dimensions: (J, L)
        
        # 1. Scale Grid: (J, 1) -> (J, L)
        grid_j = j_values.view(J, 1).expand(J, self.L)
        
        # 2. Theta Grid: (1, L) -> (J, L)
        # Linspace [0, pi) excluding endpoint
        thetas = torch.linspace(0, math.pi, L + 1)[:-1]
        grid_theta = thetas.view(1, L).expand(J, self.L)

        # C. Flattening for Vectorized Kernels
        # ----------------------------
        # Physics:
        # Sigma = 0.8 * 2^j 
        # Xi    = (3*pi/4) / 2^j 
        
        flat_sigmas = (0.8 * (2 ** grid_j)).reshape(-1)
        flat_xis = ((3 * math.pi / 4) / (2 ** grid_j)).reshape(-1)
        flat_thetas = grid_theta.reshape(-1)

        # Slant is constant for all filters in basic scattering
        # We create a vector (K,) filled with self.slant
        flat_slants = torch.full_like(flat_sigmas, self.slant)
        
        # D. Indices Map
        # ----------------------------
        # Maps scale 'j' to the slice indices [start, end)
        # Since we just have L orientations per scale:
        k_per_scale = self.L
        starts = torch.arange(0, J * k_per_scale, k_per_scale)
        ends = starts + k_per_scale
        self.register_buffer("indices_map", torch.stack([starts, ends], dim=1))


        # 3. Generate Filters (Internal Morlet Calls)
        # -------------------------------------------------------------------
        
        # A. Psi (High-Pass)
        # ------------------
        debug_logger.debug("Generating High-Pass Morlet Psi Kernels...")
        psi_native = self._morlet_psi_kernel(
            M, N,
            sigma=flat_sigmas,
            theta=flat_thetas,
            slant=flat_slants,
            xi=flat_xis
        )
        psi_f = self._to_fourier_and_normalize(psi_native)
        self.register_buffer("psi", psi_f)

        # B. Phi (Low-Pass)
        # -----------------
        # 1. Global Phi: Smoothing at the largest scale (usually J-1)
        sigma_global = 0.8 * (2 ** (J - 1))
        
        phi_g_native = self._morlet_phi_kernel(
            M, N,
            sigma=torch.tensor([sigma_global]),
            theta=torch.tensor([0.0]),
            slant=torch.tensor([1.0]), # Isotropic
            xi=torch.tensor([0.0])
        )
        phi_global_f = self._to_fourier_and_normalize(phi_g_native)
        self.register_buffer("phi_global", phi_global_f)

        # 2. Local Phi: Scale-dependent smoothing (One per scale j)
        if self.smoothing_mode != 'standard':
            debug_logger.debug("Generating per-scale Local Phi kernels...")
            local_sigmas = 0.8 * (2 ** j_values)
            phi_l_native = self._morlet_phi_kernel(
                M, N,
                sigma=local_sigmas,
                theta=torch.zeros(J),
                slant=torch.ones(J),
                xi=torch.zeros(J)
            )
            phi_local_f = self._to_fourier_and_normalize(phi_l_native)
            self.register_buffer("phi_local", phi_local_f)
        else:
            self.register_buffer("phi_local", None)

        # 4. Precompute Cropping
        # ----------------------
        # Stores static crop indices for each scale j for spatial subsampling
        for j in range(J):
            factor = 2**(j+1)
            # Center crop logic
            m_s, m_e = M // 2 - M // (2 * factor), M // 2 + M // (2 * factor)
            n_s, n_e = N // 2 - N // (2 * factor), N // 2 + N // (2 * factor)
            self.register_buffer(f"crop_idx_{j}", torch.tensor([m_s, m_e, n_s, n_e]))
        
        semantic_logger.info("GridBank2D filters successfully pre-computed.")

    # -- forward
    def forward(self) -> GridBankOutput:
        """
        Retrieves the pre-computed filter bank.

        Returns
        -------
        GridBankOutput
            A data container holding references to the Fourier-domain 
            wavelets, scaling functions, and index maps.
        """
        # ------------------------------------------------
        return GridBankOutput(
            self.psi,
            self.phi_global,
            self.phi_local,
            self.indices_map,
            self.j_effective
        )
    
    # -- get_crop
    def get_crop(self, j: int) -> torch.Tensor:
        """Retrieves the center-crop indices for a specific scale."""
        # ------------------------------------------------
        return getattr(self, f"crop_idx_{j}")

    # -- __getitem__
    def __getitem__(self, j: int) -> Tuple[int, int]:
        """Returns the filter index range (start, end) for a given scale j."""
        # ------------------------------------------------
        return self.indices_map[j][0].item(), self.indices_map[j][1].item()
    
    # ----------------------------------------------
    # Internal Functional Kernels (Morlet Physics)
    # ----------------------------------------------

    # -- _morlet_phi_kernel
    def _morlet_phi_kernel(
        self,
        M: int, N: int,
        sigma: torch.Tensor,
        theta: torch.Tensor,
        slant: torch.Tensor,
        xi: torch.Tensor # Unused for Phi but kept for signature consistency
    ) -> torch.Tensor:
        r"""
        Generates the Gaussian envelope for the Morlet wavelet.

        Math:
        $$ \Phi(x, y) = \exp\left( -\frac{x_\theta^2 + \text{slant}^2 y_\theta^2}{2\sigma^2} \right) $$

        Parameters
        ----------
        M, N : int
            Grid dimensions.
        sigma, theta, slant, xi : torch.Tensor
            Physical wavelet parameters.

        Returns
        -------
        torch.Tensor
            The spatial Gaussian envelope.
        """
        # ------------------------------------------------
        # 1. Grid
        # We generate on CPU initially to avoid massive VRAM spikes during init
        device = torch.device('cpu') 
        (y, x) = self._prepare_grid(M, N, device)
        
        # 2. Reshape Parameters for Broadcasting
        # Input: (K,) -> Output: (K, 1, 1) to broadcast against (1, M, N)
        sigma = sigma.view(-1, 1, 1)
        theta = theta.view(-1, 1, 1)
        slant = slant.view(-1, 1, 1)

        # 3. Rotation
        # x corresponds to N (columns), y to M (rows)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        x_theta = x * cos_t + y * sin_t
        y_theta = -x * sin_t + y * cos_t

        # 4. Gaussian Envelope
        gaussian = torch.exp(-(x_theta**2 + (slant**2) * y_theta**2) / (2 * sigma**2))
        return gaussian
    
    # -- _morlet_psi_kernel
    def _morlet_psi_kernel(
        self,
        M: int, N: int,
        sigma: torch.Tensor,
        theta: torch.Tensor,
        slant: torch.Tensor,
        xi: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Generates complex high-pass Morlet wavelets.

        Math:
        $$ \Psi(x, y) = \Phi(x, y) \cdot \left( \exp(i \xi x_\theta) - \text{correction} \right) $$

        Parameters
        ----------
        M, N : int
            Grid dimensions.
        sigma, theta, slant, xi : torch.Tensor
            Physical wavelet parameters.

        Returns
        -------
        torch.Tensor
            The complex spatial wavelet kernel.
        """
        # ------------------------------------------------
        # 1. Reuse Envelope Logic
        gaussian = self._morlet_phi_kernel(M, N, sigma, theta, slant, xi)
        
        # 2. Grid & Params
        device = torch.device('cpu')
        (y, x) = self._prepare_grid(M, N, device)
        
        theta = theta.view(-1, 1, 1)
        xi = xi.view(-1, 1, 1)
        
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        x_theta = x * cos_t + y * sin_t

        # 3. Complex Oscillation
        # exp(i * xi * x)
        oscillation = torch.complex(torch.cos(xi * x_theta), torch.sin(xi * x_theta))
        
        # 4. Combine
        psi = gaussian * oscillation
        
        # 5. Admissibility Condition (Zero Mean)
        # We subtract the mean to ensure integral is zero (band-pass property)
        # Operations done per-filter (dim -2, -1)
        psi = psi - psi.mean(dim=(-2, -1), keepdim=True)
        
        return psi
    
    # -- _prepare_grid
    def _prepare_grid(self, M: int, N: int, device: torch.device):
        """
        Constructs centered spatial coordinate grids.
        """
        # ------------------------------------------------
        y = torch.arange(M, device=device) - M // 2
        x = torch.arange(N, device=device) - N // 2
        y, x = torch.meshgrid(y, x, indexing='ij')
        # Unsqueeze for broadcasting against K filters
        return y.unsqueeze(0), x.unsqueeze(0)
    
    # -- _to_fourier_and_normalize
    def _to_fourier_and_normalize(self, w_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts kernels to the Fourier domain with L2 normalization.
        """
        # ------------------------------------------------

        w_f = torch.fft.fft2(torch.fft.ifftshift(w_tensor, dim=(-2, -1)))
        max_val = torch.amax(w_f.abs(), dim=(-2, -1), keepdim=True)
        w_f_norm = w_f / (max_val + 1e-8)
        # scale_factor = math.sqrt(self.L / 2.0)
        # w_f_final = w_f_norm / scale_factor
        # return w_f_final
        return w_f_norm
