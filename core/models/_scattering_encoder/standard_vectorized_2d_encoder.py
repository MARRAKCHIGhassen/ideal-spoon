# -*- coding: utf-8 -*-
"""
Scattering Orchestrator (Vectorized).

This module implements the "Fast Path" scattering transform. By stacking filters
into 4D/5D tensors, it computes convolutions for all orientations (L) simultaneously.
It is ideal for training the Phase-Aware U-Net where throughput is critical.
"""

# Importing Global Libraries
from __future__ import annotations

import logging
import math
import torch
import torch.nn as nn

from typing import Any, Dict, Tuple, Union

# Import custom libraries
from ..components.wavelet_bank_2d import GridBank2D
from ..components.modulus_and_subsampling_layer import ModulusSubsampleLayer
from ..components.padding_layer import PaddingLayer
from ..components.unpadding_layer import UnpaddingLayer

from ....constants import SCATTERING_SMOOTHING_STANDARD, SCATTERING_SMOOTHING_SCALED
from ....constants import SCATTERING_OUTPUT_TYPE_ARRAY, SCATTERING_OUTPUT_TYPE_DICT
from ....constants import SCATTERING_OUTPUT_STRUCTURE_ORDER, SCATTERING_OUTPUT_STRUCTURE_SCALE
from ....constants import SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR, SCATTERING_OUTPUT_COMPLEX_STRUCTURE_RAW

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# LOGGERS
# ---------------------------------------------------------------------

debug_logger = logging.getLogger(f"{__name__.split('.')[0]}.debug")
semantic_logger = logging.getLogger(f"{__name__.split('.')[0]}.semantic")

# -------------------------------------------------------------------------------------------------------------------

@torch.compiler.disable
def fft2_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(x)

# -- ifft2_safe
@torch.compiler.disable
def ifft2_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(x)

# -- get_stateless_padding_size_for_morlet
def get_stateless_padding_size_for_morlet(
    J: int, 
    sigma0: float = 0.8
) -> int:
    """
    Stateless calculation of the required padding size.
    Based on the maximum spatial spread of the wavelet at scale J.
    """
        
    # ------------------------------------------------

    # 
    # 1. Calculate max sigma at the largest scale (J-1)
    # sigma = sigma0 * 2^j
    sigma_max = sigma0 * (2 ** (J - 1))

    # 2. Captures ~99.9% of the spatial envelope
    multiplier = 4.5

    # 3. Final Padding Size
    # Rule of thumb: pad by the radius of the largest wavelet
    pad = int(math.ceil(sigma_max * multiplier))
    
    # Return as a safe integer
    return pad

# ---------------------------------------------------------------------
# IMPLEMENTATION
# ---------------------------------------------------------------------

# == StandardVectorizedScattering2DEncoder ==
class StandardVectorizedScattering2DEncoder(nn.Module):
    """
    Vectorized Scattering Transform.
    
    Implements the standard scattering cascade using high-performance 
    tensor broadcasting instead of loops.

    Architecture:
    1. Input -> Pad -> FFT
    2. S0: Low-pass filtering (Invariant).
    3. Order 1: Vectorized Convolution with Psi_1 -> Modulus -> Smoothing (S1).
    4. Order 2: Vectorized Convolution with Psi_2 -> Modulus -> Smoothing (S2).
    5. Finalize: Unpad and Concatenate.

    Limitations:
        * **No internal subsampling**: To keep tensors rectangular (batchable),
          intermediate downsampling is disabled.
        * **High VRAM usage**: Materializes full sets of feature maps simultaneously.
    """
    
    def __init__(
        self,
        M: int, N: int,
        J: int, L: int,
        slant: float = 0.5,
        smoothing_mode: str = SCATTERING_SMOOTHING_STANDARD,
        pre_pad: bool = False,
        outputs: Tuple[str, ...] = ("S",), # Options: "S", "U", "W"
        out_type: str = SCATTERING_OUTPUT_TYPE_ARRAY, # 'array' or 'dict'
        out_structure: str = SCATTERING_OUTPUT_STRUCTURE_ORDER, # 'scale' or 'order'
        out_complex_structure: str = SCATTERING_OUTPUT_COMPLEX_STRUCTURE_RAW, # 'polar' or 'raw'
        modulus_subsampling: bool = False,
        subsample_output_scale_dependent: bool = False,
        **unused
    ):
        super().__init__()
        self.M, self.N = M, N
        self.J, self.L = J, L
        self.smoothing_mode = smoothing_mode
        self.pre_pad = pre_pad
        self.outputs = outputs
        self.out_type = out_type
        self.out_structure = out_structure
        self.out_complex_structure = out_complex_structure
        if modulus_subsampling:
            debug_logger.error(
                "Vectorized implementation does not support post-modulus subsampling "
                "due to shared tensor layout constraints. Use monolithic encoder instead."
            )
            semantic_logger.error(
                "Vectorized implementation does not support post-modulus subsampling "
                "due to shared tensor layout constraints. Use monolithic encoder instead."
            )
            raise ValueError(
                "Vectorized implementation does not support post-modulus subsampling "
                "due to shared tensor layout constraints. Use monolithic encoder instead."
            )
        if subsample_output_scale_dependent:
            debug_logger.error(
                "Vectorized implementation does not support post-modulus subsampling "
                "due to shared tensor layout constraints. Use monolithic encoder instead."
            )
            semantic_logger.error(
                "Vectorized implementation does not support post-modulus subsampling "
                "due to shared tensor layout constraints. Use monolithic encoder instead."
            )
            raise ValueError(
                "Vectorized implementation does not support post-modulus subsampling "
                "due to shared tensor layout constraints. Use monolithic encoder instead."
            )

        # --- 1. Dimensions & Padding ---
        # We determine padding based on the largest scale (J) to avoid boundary artifacts.
        self.pad_size = get_stateless_padding_size_for_morlet(J)

        # --- HARDENING: Validate Input Dimensions ---
        # Reflection padding fails if input < pad. We raise a clear error here.
        if M < self.pad_size or N < self.pad_size:
            raise ValueError(
                f"Input dimensions ({M}x{N}) are too small for the required scattering padding "
                f"({self.pad_size} px for J={J}). Increase input size or reduce J."
            )
        
        self.M_pad = M + 2 * self.pad_size
        self.N_pad = N + 2 * self.pad_size

        # --- 2. Components ---
        self.padding_layer = PaddingLayer(padding=self.pad_size)
        self.unpadding_layer = UnpaddingLayer(padding=self.pad_size)
        
        # Modulus: We use k=1 because subsampling is handled at the end 
        #   we keep full resolution internally (Vectorized requirement)
        self.modulus = ModulusSubsampleLayer()

        # --- 3. Filter Bank ---
        # The bank is initialized with the PADDED dimensions.
        # It generates all filters (J*L) in a single vectorized pass during init.
        self.bank = GridBank2D(
            self.M_pad, self.N_pad, 
            J, L, 
            slant=slant,
            smoothing_mode=smoothing_mode
        )

        # --- 4. Guardrails ---
        if self.out_structure != SCATTERING_OUTPUT_STRUCTURE_ORDER and self.out_structure != SCATTERING_OUTPUT_STRUCTURE_SCALE:
            debug_logger.warning(f"Unknown output structure. Defaulting to {SCATTERING_OUTPUT_STRUCTURE_ORDER}.")
            semantic_logger.warning(f"Unknown output structure. Defaulting to {SCATTERING_OUTPUT_STRUCTURE_ORDER}.")
            self.out_structure = SCATTERING_OUTPUT_STRUCTURE_ORDER
        if self.outputs in [("S", "W"), ("U", "W")] and \
            (self.out_complex_structure != SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR and self.out_complex_structure != SCATTERING_OUTPUT_COMPLEX_STRUCTURE_RAW):
            raise ValueError("Unsupported output complex structure")
        if "S" not in self.outputs:
            debug_logger.warning(f"Scattering coefficients are always ouptut.")
            semantic_logger.warning(f"Scattering coefficients are always ouptut.")

        # --- 4. Wiring Methods ---
        if smoothing_mode == SCATTERING_SMOOTHING_STANDARD:
            self.compute_S_order_0 = self._compute_standard_S_order_0
            self.compute_S_order_1 = self._compute_standard_S_order_1
            self.compute_S_order_2 = self._compute_standard_S_order_2
        elif smoothing_mode == SCATTERING_SMOOTHING_SCALED:
            self.compute_S_order_0 = self._compute_scaled_S_order_0
            self.compute_S_order_1 = self._compute_scaled_S_order_1
            self.compute_S_order_2 = self._compute_scaled_S_order_2
        else:
            raise ValueError(f"Unknown smoothing mode: {smoothing_mode}")

        if self.out_type == SCATTERING_OUTPUT_TYPE_ARRAY:
            if any(opt in self.outputs for opt in ["U", "W"]):
                debug_logger.warning("Array mode only supports 'S' outputs. Array mode is ignored.")
                semantic_logger.warning("Array mode only supports 'S' outputs. Array mode is ignored.")
                self.finalize_fn = self._finalize_dict_tensor_based
            else:
                self.finalize_fn = self._finalize_array
        elif self.out_type == SCATTERING_OUTPUT_TYPE_DICT:
            self.finalize_fn = self._finalize_dict_tensor_based
        else:
            raise ValueError(f"Unknown out_type: {self.out_type}")
            
    # -- forward
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # --- HARDENING: Auto-Device Sync ---
        # If input is on GPU but filters (buffers) are on CPU, move filters automatically.
        if x.device != self.bank.psi.device:
            self.to(x.device)

        # 1. Pad (Reflection)
        x_pad = self.padding_layer(x) if not self.pre_pad else x
        
        # 2. FFT
        x_f = fft2_safe(x_pad.to(torch.complex64))
        
        # 3. Get Filters
        bank_out = self.bank()
        # bank_out.psi: (K, M_pad, N_pad) where K = J*L
        
        # --- ORDER 0 (S0) ---
        S0 = self.compute_S_order_0(x_f, bank_out)
        
        # --- ORDER 1 (W1, U1, S1) ---
        # 1. W1: Convolution (Broadcasted)
        psi_stack = bank_out.psi
        w1_f = x_f.unsqueeze(2) * psi_stack.view(1, 1, -1, self.M_pad, self.N_pad)
        
        # 2. U1: Spatial Modulus
        w1_spatial = ifft2_safe(w1_f)
        u1_spatial = self.modulus(torch.view_as_real(w1_spatial))
        
        # 3. S1: Smoothing
        S1 = self.compute_S_order_1(u1_spatial, bank_out)
        
        # --- ORDER 2 (W2, U2, S2) ---
        # 1. W2: Cross-Convolution
        # U1: (B, C, K1, M, N) * Psi: (K2, M, N) -> (B, C, K1, K2, M, N)
        u1_f = fft2_safe(u1_spatial.to(torch.complex64))
        w2_f = u1_f.unsqueeze(3) * psi_stack.view(1, 1, 1, -1, self.M_pad, self.N_pad)
        
        # 2. U2: Spatial Modulus
        w2_spatial = ifft2_safe(w2_f)
        u2_spatial = self.modulus(torch.view_as_real(w2_spatial))
        
        # 3. S2: Smoothing
        S2 = self.compute_S_order_2(u2_spatial, bank_out)
        
        # --- FINALIZATION ---
        return self.finalize_fn(S0, S1, S2, u1_spatial, u2_spatial, w1_spatial, w2_spatial)
    

    # ---------------------------------------------------------------------
    # Smoothing Implementations
    # ---------------------------------------------------------------------
    def _compute_standard_S_order_0(self, x_f, bank_out):
        phi = bank_out.phi_global
        s_f = x_f * phi
        return ifft2_safe(s_f).real

    def _compute_standard_S_order_1(self, u1_spatial, bank_out):
        u1_f = fft2_safe(u1_spatial.to(torch.complex64))
        phi = bank_out.phi_global
        s_f = u1_f * phi.view(1, 1, 1, self.M_pad, self.N_pad)
        return ifft2_safe(s_f).real

    def _compute_standard_S_order_2(self, u2_spatial, bank_out):
        u2_f = fft2_safe(u2_spatial.to(torch.complex64))
        phi = bank_out.phi_global
        s_f = u2_f * phi.view(1, 1, 1, 1, self.M_pad, self.N_pad)
        return ifft2_safe(s_f).real

    def _compute_scaled_S_order_0(self, x_f, bank_out):
        return self._compute_standard_S_order_0(x_f, bank_out)

    def _compute_scaled_S_order_1(self, u1_spatial, bank_out):
        u1_f = fft2_safe(u1_spatial.to(torch.complex64))
        phi = bank_out.phi_local # (J, M, N)
        phi_expanded = phi.repeat_interleave(self.L, dim=0) 
        s_f = u1_f * phi_expanded.view(1, 1, -1, self.M_pad, self.N_pad)
        return ifft2_safe(s_f).real

    def _compute_scaled_S_order_2(self, u2_spatial, bank_out):
        u2_f = fft2_safe(u2_spatial.to(torch.complex64))
        phi = bank_out.phi_local
        phi_expanded = phi.repeat_interleave(self.L, dim=0)
        s_f = u2_f * phi_expanded.view(1, 1, 1, -1, self.M_pad, self.N_pad)
        return ifft2_safe(s_f).real
    
    # -- __repr__
    def __repr__(self):
        output_str = f"{self.__class__.__name__}("
        output_str += f"M={self.M}, N={self.N}, "
        output_str += f"J={self.J}, L={self.L}, "
        output_str += f"smoothing_mode={self.smoothing_mode}, "
        output_str += f"outputs={self.outputs}, "
        output_str += f"out_type={self.out_type}, "
        output_str += f"out_structure={self.out_structure}, "
        output_str += f"out_complex_structure={self.out_complex_structure}, "

        return output_str  

    # ---------------------------------------------------------------------
    # Finalization
    # ---------------------------------------------------------------------
    def _finalize_array(
        self,
        s0: torch.Tensor,   # (B, C, H_pad, W_pad)
        s1: torch.Tensor,   # (B, C, K1, H_pad, W_pad)
        s2: torch.Tensor,   # (B, C, K1, K2, H_pad, W_pad)
        u1: torch.Tensor,   # (B, C, K1, H_pad, W_pad)
        u2: torch.Tensor,   # (B, C, K1, K2, H_pad, W_pad)
        w1: torch.Tensor,   # (B, C, K1, H_pad, W_pad) - Complex64
        w2: torch.Tensor,   # (B, C, K1, K2, H_pad, W_pad) - Complex64
    ) -> torch.Tensor:
        """
        Unified finalization that can return S, U, W depending on self.outputs.

        self.outputs: tuple of any subset of {"S", "U", "W"}
        - "S": scattering coefficients (S0, S1, S2)
        - "U": modulus (U1, U2)
        - "W": complex wavelets (W1, W2) (real/imag stacked in last dim)

        out_type:
        - SCATTERING_OUTPUT_TYPE_ARRAY: flatten to (B, C_total, H, W)
        - SCATTERING_OUTPUT_TYPE_DICT: nested dict with structure = self.out_structure
        """
        
        # --- 1. SPATIAL UNPADDING ---
        # We unpad everything to match input size (M, N). 
        # Note: _w1/_w2 remain complex tensors at this stage.
        _s0 = self.unpadding_layer(s0)
        _s1 = self.unpadding_layer(s1)
        _s2 = self.unpadding_layer(s2)

        # In array mode, we typically concatenate S0, S1, and flattened S2
        # This follows the native resolution H_S = H / 2^J
        stride = 2**self.J if self.smoothing_mode == SCATTERING_SMOOTHING_STANDARD else 1
        _s0_down = _s0[..., ::stride, ::stride]
        _s0_ex = _s0_down.unsqueeze(2)  # (B, C, 1, H, W)
        _s2_flat = _s2.flatten(2, 3) # (B, C, K1*K2, H, W)
        
        # Concatenate along the path dimension (dim 2)
        s_out = torch.cat([_s0_ex, _s1, _s2_flat], dim=2)
        # Flatten Batch and Path channels: (B, C, Paths, H, W) -> (B, C * Paths, H, W)
        flattened_s_out = s_out.flatten(1, 2)
        return flattened_s_out
        
    def _finalize_dict_tensor_based(
        self,
        s0: torch.Tensor,   # (B, C, H_pad, W_pad)
        s1: torch.Tensor,   # (B, C, K1, H_pad, W_pad)
        s2: torch.Tensor,   # (B, C, K1, K2, H_pad, W_pad)
        u1: torch.Tensor,   # (B, C, K1, H_pad, W_pad)
        u2: torch.Tensor,   # (B, C, K1, K2, H_pad, W_pad)
        w1: torch.Tensor,   # (B, C, K1, H_pad, W_pad) - Complex64
        w2: torch.Tensor,   # (B, C, K1, K2, H_pad, W_pad) - Complex64
    ) -> Dict[str, Any]:
        """
        Unified finalization that can return S, U, W depending on self.outputs.

        self.outputs: tuple of any subset of {"S", "U", "W"}
        - "S": scattering coefficients (S0, S1, S2)
        - "U": modulus (U1, U2)
        - "W": complex wavelets (W1, W2) (real/imag stacked in last dim)

        out_type:
        - SCATTERING_OUTPUT_TYPE_ARRAY: flatten to (B, C_total, H, W)
        - SCATTERING_OUTPUT_TYPE_DICT: nested dict with structure = self.out_structure
        """

        # Helper
        def get_complex_format(x):
            if x is None: return None
            if self.out_complex_structure == SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR:
                return (x.abs(), x.angle())
            return torch.view_as_real(x)
        
        # --- 0. PRE-ALIGNMENT ---
        # Vectorized output: (B, C, K1, K2, H, W) -> Layout: j1, l1, j2, l2
        # Target logic expects: j1, j2, l1, l2 (to slice L*L blocks
        B, C, _, _, H, W = s2.shape
        # 1. Unfold K dimensions back to (J, L)
        s2 = s2.view(B, C, self.J, self.L, self.J, self.L, H, W)
        # 2. Permute: Swap l1 (dim 3) and j2 (dim 4) -> (j1, j2, l1, l2)
        s2 = s2.permute(0, 1, 2, 4, 3, 5, 6, 7)
        # 3. Flatten to match the monolithic slicing logic: (B, C, J*J*L*L, H, W)
        s2 = s2.reshape(B, C, -1, H, W)
        # Apply same fix to U2 and W2
        if u2 is not None:
            u2 = u2.view(B, C, self.J, self.L, self.J, self.L, H, W) \
                   .permute(0, 1, 2, 4, 3, 5, 6, 7) \
                   .reshape(B, C, -1, H, W)
        if w2 is not None:
            w2 = w2.view(B, C, self.J, self.L, self.J, self.L, H, W) \
                   .permute(0, 1, 2, 4, 3, 5, 6, 7) \
                   .reshape(B, C, -1, H, W)

        # --- 1. SPATIAL UNPADDING ---
        # We unpad everything to match input size (M, N). 
        # Note: _w1/_w2 remain complex tensors at this stage.
        _s0 = self.unpadding_layer(s0)
        _s1 = self.unpadding_layer(s1)
        _s2 = self.unpadding_layer(s2)

        # --- U Stream (Optional) ---
        if "U" in self.outputs:
            _u1 = self.unpadding_layer(u1)
            _u2 = self.unpadding_layer(u2)

        # --- W Stream (Optional) ---
        if "W" in self.outputs:
            _w1 = self.unpadding_layer(w1)
            _w2 = self.unpadding_layer(w2)

        # --- 2. VIEW GENERATION ---
        B, C, _, H, W = _s1.shape
        stride_max = 2**self.J
        stride = stride_max if self.smoothing_mode == SCATTERING_SMOOTHING_STANDARD else 1
        L2_block = self.L * self.L

        # Structured Views
        s0_out = _s0[..., ::stride, ::stride]
        results = {}

        # --- CASE A: ORDER STRUCTURE ---
        if self.out_structure == SCATTERING_OUTPUT_STRUCTURE_ORDER:
            s_dict = {"S0": s0_out, "S1": [], "S2": []}
            u_dict = {"U1": [], "U2": []} if "U" in self.outputs else None

            # Initialize W dict with correct keys
            w_dict = None
            if "W" in self.outputs:
                if self.out_complex_structure == SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR:
                    w_dict = {"A1": [], "Phi1": [], "A2": [], "Phi2": []}
                else:
                    w_dict = {"W1": [], "W2": []}

            for j1 in range(self.J):
                # --- Order 1 (Slice: j1*L -> (j1+1)*L) ---
                start1 = j1 * self.L
                end1 = start1 + self.L
                
                # S1
                s1_slice = _s1[:, :, start1:end1, ...]
                s_dict["S1"].append(s1_slice[..., ::stride, ::stride])
                
                # U1
                if u_dict: u_dict["U1"].append(_u1[:, :, start1:end1, ...])

                # W1 (Split A1/Phi1)
                if w_dict:
                    formatted = get_complex_format(_w1[:, :, start1:end1, ...])
                    if isinstance(formatted, tuple):
                        w_dict["A1"].append(formatted[0])
                        w_dict["Phi1"].append(formatted[1])
                    else:
                        w_dict["W1"].append(formatted)

                # --- Order 2 (Nested Loop) ---
                for j2 in range(self.J):
                    # Linear Index into the J^2 stack
                    idx = j1 * self.J + j2
                    start2 = idx * L2_block
                    end2 = start2 + L2_block
                    
                    # S2: Slice -> Reshape(L^2 -> L,L) -> Stride
                    s2_flat = _s2[:, :, start2:end2, ...]
                    s2_reshaped = s2_flat.reshape(B, C, self.L, self.L, H, W)
                    s_dict["S2"].append(s2_reshaped[..., ::stride, ::stride])

                    # U2
                    if u_dict:
                        u2_flat = _u2[:, :, start2:end2, ...]
                        u_dict["U2"].append(u2_flat.reshape(B, C, self.L, self.L, H, W))

                    # W2 (Split A2/Phi2)
                    if w_dict:
                        w2_flat = _w2[:, :, start2:end2, ...]
                        w2_reshaped = w2_flat.reshape(B, C, self.L, self.L, H, W)
                        formatted = get_complex_format(w2_reshaped)
                        
                        if isinstance(formatted, tuple):
                            w_dict["A2"].append(formatted[0])
                            w_dict["Phi2"].append(formatted[1])
                        else:
                            w_dict["W2"].append(formatted)
                    
            results["S"] = s_dict
            if u_dict: results["U"] = u_dict
            if w_dict: results["W"] = w_dict

        else:
            results["S0"] = s0_out

            for j1 in range(self.J):
                scale_obj = {}
                
                # --- Order 1 Slices ---
                start1 = j1 * self.L
                end1 = start1 + self.L
                
                scale_obj["S1"] = _s1[:, :, start1:end1, ...][..., ::stride, ::stride]

                if "U" in self.outputs:
                    scale_obj["U1"] = _u1[:, :, start1:end1, ...]
                
                if "W" in self.outputs:
                    w1_slice = _w1[:, :, start1:end1, ...]
                    formatted = get_complex_format(w1_slice)
                    if isinstance(formatted, tuple):
                        scale_obj["A1"], scale_obj["Phi1"] = formatted
                    else:
                        scale_obj["W1"] = formatted
                
                # --- Order 2 Children (Lists) ---
                s2_sibs, u2_sibs = [], []
                # W2 Siblings must be lists
                w2_real = [] 
                w2_abs, w2_phi = [], []
                
                for j2 in range(self.J):
                    idx = j1 * self.J + j2
                    start2 = idx * L2_block
                    end2 = start2 + L2_block
                    
                    # S2
                    s2_flat = _s2[:, :, start2:end2, ...]
                    s2_reshaped = s2_flat.reshape(B, C, self.L, self.L, H, W)
                    s2_sibs.append(s2_reshaped[..., ::stride, ::stride])

                    # U2
                    if "U" in self.outputs:
                        u2_flat = _u2[:, :, start2:end2, ...]
                        u2_sibs.append(u2_flat.reshape(B, C, self.L, self.L, H, W))
                    
                    # W2
                    if "W" in self.outputs:
                        w2_flat = _w2[:, :, start2:end2, ...]
                        w2_reshaped = w2_flat.reshape(B, C, self.L, self.L, H, W)
                        formatted = get_complex_format(w2_reshaped)
                        
                        if isinstance(formatted, tuple):
                            w2_abs.append(formatted[0])
                            w2_phi.append(formatted[1])
                        else:
                            w2_real.append(formatted)
                
                scale_obj["S2"] = s2_sibs
                if u2_sibs: scale_obj["U2"] = u2_sibs
                
                if "W" in self.outputs:
                    if self.out_complex_structure == SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR:
                        scale_obj["A2"] = w2_abs
                        scale_obj["Phi2"] = w2_phi
                    else:
                        scale_obj["W2"] = w2_real
                
                results[f"scale_{j1}"] = scale_obj
    
        return results
    
    # ---------------------------------------------------------------------
    # Meta-Data Helper
    # ---------------------------------------------------------------------
    def get_channel_map(self) -> Dict[str, slice]:
        """
        Returns a dictionary mapping logical groups to channel slices 
        in the dense output array.
        """
        K1 = self.J * self.L
        K2 = self.J * self.L # Vectorized bank computes all J*L
        
        mapping = {}
        
        # S0
        curr = 0
        mapping["S0"] = slice(curr, curr+1)
        curr += 1
        
        # S1
        mapping["S1"] = slice(curr, curr+K1)
        
        # S1 per scale j1
        for j1 in range(self.J):
            start = curr + j1 * self.L
            end = start + self.L
            mapping[f"S1_j{j1}"] = slice(start, end)
        
        curr += K1
        
        # S2
        mapping["S2"] = slice(curr, curr + K1*K2)
        
        # S2 per parent scale j1
        # S2 is flattened (K1, K2). K1 corresponds to j1.
        for j1 in range(self.J):
            # The K1 dimension is split into J blocks of size L
            # Each of those L paths has K2 children.
            block_size = self.L * K2 
            start = curr + j1 * block_size
            end = start + block_size
            mapping[f"S2_j{j1}"] = slice(start, end)
            
        return mapping
