# -*- coding: utf-8 -*-
"""
Scattering Orchestrator (Monolithic).

This module implements the "Flexible Path" scattering transform using sequential 
loops over scales. While slower than the vectorized implementation, it supports:
1. **Memory Efficiency**: Processes one scale at a time, avoiding large 5D tensors.
2. **Subsampling**: Supports inter-layer downsampling (k=2^j) to reduce 
   computational load for high-order coefficients.
3. **Variable Scales**: Adaptable for ablation studies requiring non-standard 
   filter bank configurations.
"""

# Importing Global Libraries
from __future__ import annotations

import logging
import math
import torch
import torch.nn as nn

from typing import Dict, List, Tuple, Final, Callable, Union, Any

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

# == StandardMonolithicScattering2DEncoder ==
class StandardMonolithicScattering2DEncoder(nn.Module):
    """
    Monolithic (Loop-Based) Scattering Transform.
    
    Processing:
        Iterates over scales (j1, j2) sequentially.
        
    Pros:
        * **Lower peak memory usage**: Never materializes full 6D tensors.
        * **Subsampling Support**: Can downsample features after modulus (U),
          drastically reducing the size of Order 2 coefficients.
          
    Cons:
        * **Slower wall-clock time**: Lacks kernel fusion/parallelism of the
          vectorized approach.
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
        self.outputs = outputs
        self.out_type = out_type
        self.pre_pad = pre_pad
        self.out_structure = out_structure
        self.out_complex_structure = out_complex_structure
        self.modulus_subsampling = modulus_subsampling
        self.subsample_output_scale_dependent = subsample_output_scale_dependent

        # --- 1. Dimensions & Padding ---
        # We determine padding based on the largest scale (J) to avoid boundary artifacts.
        self.pad_size = get_stateless_padding_size_for_morlet(J)
        # HARDENING: Validate Input Dimensions
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
            self.J, self.L, 
            slant=slant,
            smoothing_mode=self.smoothing_mode
        )

        # --- 4. Guardrails ---
        if self.smoothing_mode != SCATTERING_SMOOTHING_SCALED and self.subsample_output_scale_dependent:
            debug_logger.warning("Ignoring scale dependent subsampling of the output. If forcing required, it should be done outside")
            semantic_logger.warning("Ignoring scale dependent subsampling of the output. If forcing required, it should be done outside")
            self.subsample_output_scale_dependent = False
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
        
        # --- 5. Wiring Methods ---
        if self.smoothing_mode == SCATTERING_SMOOTHING_STANDARD:
            self.compute_S_order_0 = self._compute_standard_S_order_0
            self.compute_S_order_1 = self._compute_standard_S_order_1
            self.compute_S_order_2 = self._compute_standard_S_order_2
        elif self.smoothing_mode == SCATTERING_SMOOTHING_SCALED:
            self.compute_S_order_0 = self._compute_scaled_S_order_0
            self.compute_S_order_1 = self._compute_scaled_S_order_1
            self.compute_S_order_2 = self._compute_scaled_S_order_2
        else:
            raise ValueError(f"Unknown smoothing mode: {self.smoothing_mode}")
        
        if "W" in self.outputs:
            self.append_W_outputs_fn = self._append_to_list
        else:
            self.append_W_outputs_fn = self._noop
        if "U" in self.outputs:
            self.append_U_outputs_fn = self._append_to_list
        else:
            self.append_U_outputs_fn = self._noop

        if self.modulus_subsampling:
            debug_logger.warning("Between-layer post-modulus subsampling is performed. Ignoring array and tensor-based dict output type")
            semantic_logger.warning("Between-layer post-modulus subsampling is performed. Ignoring array and tensor-based dict output type")
            self.finalize_fn = self._finalize_dict
        elif self.subsample_output_scale_dependent:
            debug_logger.warning("Scale dependent subsampling of the output. Ignoring array and tensor-based dict output type")
            semantic_logger.warning("Scale dependent subsampling of the output. Ignoring array and tensor-based dict output type")
            self.finalize_fn = self._finalize_dict
        else:
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
        
        # 1. Pad
        x_pad = self.padding_layer(x) if not self.pre_pad else x
        
        # 2. FFT
        x_f = fft2_safe(x_pad.to(torch.complex64))
        
        # 3. Containers
        # We store results in lists to mimic the structure we will finalize later
        s0 = None
        s1_list = [] # Will hold tensors of shape (B, C, L, H, W) per j1
        s2_list = [] # Will hold tensors of shape (B, C, L1, L2, H, W) per (j1, j2)
        
        w1_list = [] # For optional W outputs of order 1
        w2_list = [] # For optional W outputs of order 2

        u1_list = [] # For optional U outputs of order 1
        u2_list = [] # For optional U outputs of order 2

        # --- ORDER 0 ---
        s0 = self.compute_S_order_0(x_f)

        # --- ORDER 1 LOOP ---
        for j1 in range(self.J):
            # A. Get Filters for j1 (Batch of L filters)
            # bank.psi is (J*L, M, N). Slice out [j1*L : (j1+1)*L]
            start_idx = j1 * self.L
            end_idx = start_idx + self.L
            
            # Shape: (L, M, N)
            psi_j1 = self.bank.psi[start_idx:end_idx, ...]
            
            # B. Convolve (Broadcast over L)
            # x_f: (B, C, M, N) -> (B, C, 1, M, N)
            # psi: (L, M, N) -> (1, 1, L, M, N)
            w1_f = x_f.unsqueeze(2) * psi_j1.unsqueeze(0).unsqueeze(0)
            
            # C. Modulus
            w1_spatial = ifft2_safe(w1_f)
            self.append_W_outputs_fn(w1_spatial, w1_list)
            u1_spatial = self.modulus(torch.view_as_real(w1_spatial), k=j1) if self.modulus_subsampling else self.modulus(torch.view_as_real(w1_spatial), k=1)
            self.append_U_outputs_fn(u1_spatial, u1_list)
            
            # D. Smoothing S1
            # S1 depends on u1. In 'Standard', smoothed by Phi_J. In 'Scaled', by Phi_{j1+1}.
            # The specific compute function handles the choice of phi.
            s1 = self.compute_S_order_1(u1_spatial, j1)
            s1_list.append(s1)
            
            # --- ORDER 2 LOOP ---
            # We iterate all j2 (vectorized does this). 
            # Usually only j2 > j1 is useful, but we match Vectorized behavior (full grid).
            
            # Pre-compute U1_f for the inner loop
            u1_f = fft2_safe(u1_spatial.to(torch.complex64))
            
            # Container for this j1's children (to keep structure for S2 list)
            # We will flatten later if needed, but keeping (J, J) structure is cleaner.
            
            for j2 in range(self.J):
                if j2 <= j1:
                    # 1. SKIP CONVOLUTION (Heavy)
                    # Create zeros with the shape that W2 *would* have had.
                    # u1_spatial: (B, C, L1, H, W) -> w2: (B, C, L1, 1, H, W)
                    # For safety, we match the w2_f shape logic:
                    # w2_f = u1_f.unsqueeze(3) * psi... 
                    # so w2_spatial is effectively u1 expanded
                    B_sz, C_sz, L_sz, H_sz, W_sz = u1_spatial.shape
                    w2_ghost = torch.zeros(
                        (B_sz, C_sz, L_sz, self.L, H_sz, W_sz),
                        dtype=x_f.dtype, # Must be complex
                        device=x_f.device
                    )

                    # 2. RUN MODULUS (Structural - Handles Subsampling)
                    # We run the layer since it is fast on zeros and guarantees u2 has correct dims.
                    self.append_W_outputs_fn(w2_ghost, w2_list)
                    u2_ghost = self.modulus(torch.view_as_real(w2_ghost), k=j2) if self.modulus_subsampling else self.modulus(torch.view_as_real(w2_ghost), k=1)
                    self.append_U_outputs_fn(u2_ghost, u2_list)

                    # 3. SKIP SMOOTHING FFTs (Heavy)
                    # In 'standard_monolithic', compute_S_order_2 is purely:
                    # FFT -> Mult -> IFFT.
                    # It DOES NOT change dimensions (spatial size or channel count).
                    # Therefore, s2 has the exact same shape as u2_ghost.
                    s2_ghost = torch.zeros_like(u2_ghost)
                    s2_list.append(s2_ghost)
                    continue
                
                # E. Get Filters for j2
                start_idx_2 = j2 * self.L
                end_idx_2 = start_idx_2 + self.L
                psi_j2 = self.bank.psi[start_idx_2:end_idx_2, ...]
                
                # F. Convolve
                # u1_f: (B, C, L1, M, N) -> (B, C, L1, 1, M, N)
                # psi2: (L2, M, N) -> (1, 1, 1, L2, M, N)
                # Result W2: (B, C, L1, L2, M, N)
                w2_f = u1_f.unsqueeze(3) * psi_j2.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                
                # G. Modulus
                w2_spatial = ifft2_safe(w2_f)
                self.append_W_outputs_fn(w2_spatial, w2_list)
                u2_spatial = self.modulus(torch.view_as_real(w2_spatial), k=j2) if self.modulus_subsampling else self.modulus(torch.view_as_real(w2_spatial), k=1)
                self.append_U_outputs_fn(u2_spatial, u2_list)

                # H. Smoothing S2
                s2 = self.compute_S_order_2(u2_spatial, j2)
                s2_list.append(s2)

        # --- FINALIZATION ---
        return self.finalize_fn(s0, s1_list, s2_list, u1_list, u2_list, w1_list, w2_list)
    
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
        output_str += f"modulus_subsampling={self.modulus_subsampling}, "
        output_str += f"subsample_output_scale_dependent={self.subsample_output_scale_dependent})"
        return output_str  
    

    # ---------------------------------------------------------------------
    # Smoothing Implementations
    # ---------------------------------------------------------------------
    def _compute_standard_S_order_0(self, x_f):
        phi = self.bank.phi_global
        s_f = x_f * phi
        return ifft2_safe(s_f).real

    def _compute_standard_S_order_1(self, u1_spatial, j1: int):
        u1_f = fft2_safe(u1_spatial.to(torch.complex64))
        phi = self.bank.phi_global
        # u1_f: (B, C, L, M, N)
        # phi: (M, N)
        s_f = u1_f * phi.view(1, 1, 1, self.M_pad, self.N_pad)
        return ifft2_safe(s_f).real

    def _compute_standard_S_order_2(self, u2_spatial, j2: int):
        u2_f = fft2_safe(u2_spatial.to(torch.complex64))
        phi = self.bank.phi_global
        # u2_f: (B, C, L1, L2, M, N)
        s_f = u2_f * phi.view(1, 1, 1, 1, self.M_pad, self.N_pad)
        return ifft2_safe(s_f).real

    def _compute_scaled_S_order_0(self, x_f):
        # S0 is always low-passed by the global window (approx) or J.
        return self._compute_standard_S_order_0(x_f)

    def _compute_scaled_S_order_1(self, u1_spatial, j1: int):
        u1_f = fft2_safe(u1_spatial.to(torch.complex64))
        # Scaled: Use phi at scale j1 (actually j1+1 usually to cover bandwidth)
        # GridBank stores locals in .phi_local of shape (J, M, N)
        # We pick index j1.
        phi = self.bank.phi_local[j1] 
        s_f = u1_f * phi.view(1, 1, 1, self.M_pad, self.N_pad)
        return ifft2_safe(s_f).real

    def _compute_scaled_S_order_2(self, u2_spatial, j2: int):
        u2_f = fft2_safe(u2_spatial.to(torch.complex64))
        # Scaled: S2 depends on j2 scale
        phi = self.bank.phi_local[j2]
        s_f = u2_f * phi.view(1, 1, 1, 1, self.M_pad, self.N_pad)
        return ifft2_safe(s_f).real
    

    # ---------------------------------------------------------------------
    # Appending Implementations
    # ---------------------------------------------------------------------
    def _append_to_list(
        self,
        x: torch.Tensor,
        list_to_append: List[torch.Tensor],
    ) -> None:
        list_to_append.append(x)

    def _noop(
        self,
        x: torch.Tensor,
        list_to_append: List[torch.Tensor],
    ) -> None:
        pass


    # ---------------------------------------------------------------------
    # Finalization (Aligned with Vectorized)
    # ---------------------------------------------------------------------
    def _finalize_array(
        self,
        s0: torch.Tensor,
        s1_list: list[torch.Tensor],
        s2_list: list[torch.Tensor],
        u1_list: list[torch.Tensor],
        u2_list: list[torch.Tensor],
        w1_list: list[torch.Tensor],
        w2_list: list[torch.Tensor],
    ) -> torch.Tensor:

        # --- 1. PRE-ALIGNMENT: Convert Lists to Stacked Tensors ---
        # Monolithic loops produce lists; we stack them to reuse vectorized logic
        # S1: [J tensors of (B,C,L,H,W)] -> (B,C,J*L,H,W)
        s1 = torch.cat(s1_list, dim=2)
        # S2: [J*J tensors of (B,C,L,L,H,W)] -> (B,C,J*L, J*L, H,W)
        # We flatten the internal L1,L2 to match the vectorized K1,K2 expectation
        s2 = torch.cat([s.flatten(2, 3) for s in s2_list], dim=2).view(
            s0.shape[0], s0.shape[1], self.J * self.L, self.J * self.L, s0.shape[-2], s0.shape[-1]
        )
        
        # --- 2. SPATIAL UNPADDING ---
        # Uniform unpadding across all requested streams
        _s0, _s1, _s2 = self.unpadding_layer(s0), self.unpadding_layer(s1), self.unpadding_layer(s2)
        
        # --- 3. CONCATENATING ---
        # Dimensions:
        # S0: (B, C, H, W) -> (B, C, 1, H, W)
        # S1: (B, C, K1, H, W)
        # S2: (B, C, K1, K2, H, W) -> (B, C, K1*K2, H, W)
        # In array mode, we typically concatenate S0, S1, and flattened S2
        # This follows the native resolution H_S = H / 2^J
        stride = 2**self.J if self.smoothing_mode == SCATTERING_SMOOTHING_STANDARD else 1
        _s0_down = _s0[..., ::stride, ::stride]
        _s0_ex = _s0_down.unsqueeze(2)  # (B, C, 1, H, W)
        _s2_flat = _s2.flatten(2, 3) # (B, C, K1*K2, H, W)
        # Concatenate along 'Path' dimension (dim 2)
        s_out = torch.cat([_s0_ex, _s1, _s2_flat], dim=2)
        # Flatten Batch and Path channels: (B, C, Paths, H, W) -> (B, C * Paths, H, W)
        flattened_s_out = s_out.flatten(1, 2)
        
        return flattened_s_out

    def _finalize_dict_tensor_based(
        self,
        s0: torch.Tensor,
        s1_list: list[torch.Tensor],
        s2_list: list[torch.Tensor],
        u1_list: list[torch.Tensor],
        u2_list: list[torch.Tensor],
        w1_list: list[torch.Tensor],
        w2_list: list[torch.Tensor],
    ) -> Dict[str, Any]:

        # Helper
        def get_complex_format(x):
            if x is None: return None
            if self.out_complex_structure == SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR:
                return (x.abs(), x.angle())
            return torch.view_as_real(x)
        
        # --- S Stream ---
        s1 = torch.cat(s1_list, dim=2)
        s2 = torch.cat([s.flatten(2, 3) for s in s2_list], dim=2)
        
        # Apply Unpadding (Returns non-contiguous slices)
        _s0 = self.unpadding_layer(s0)
        _s1 = self.unpadding_layer(s1)
        _s2 = self.unpadding_layer(s2)

        # --- U Stream (Optional) ---
        if "U" in self.outputs:
            u1_stack = torch.cat(u1_list, dim=2)
            u2_stack = torch.cat([u.flatten(2, 3) for u in u2_list], dim=2)
            _u1 = self.unpadding_layer(u1_stack)
            _u2 = self.unpadding_layer(u2_stack)

        # --- W Stream (Optional) ---
        if "W" in self.outputs:
            w1_stack = torch.cat(w1_list, dim=2)
            w2_stack = torch.cat([w.flatten(2, 3) for w in w2_list], dim=2)
            _w1 = self.unpadding_layer(w1_stack)
            _w2 = self.unpadding_layer(w2_stack)
        
        # --- 3. VIEW GENERATION ---
        B, C, _, H, W = _s1.shape
        stride_max = 2**self.J
        stride = stride_max if self.smoothing_mode == SCATTERING_SMOOTHING_STANDARD else 1
        L2_block = self.L * self.L

        # Structured Views
        s0_out = _s0[..., ::stride_max, ::stride_max]
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
    
    def _finalize_dict(
        self,
        s0: torch.Tensor,
        s1_list: list[torch.Tensor],
        s2_list: list[torch.Tensor],
        u1_list: list[torch.Tensor],
        u2_list: list[torch.Tensor],
        w1_list: list[torch.Tensor],
        w2_list: list[torch.Tensor],
    ) -> Dict[str, Any]:

        # --- 1. HELPERS ---
        stride_max = 2**self.J
        s0_stride = stride_max if self.smoothing_mode == SCATTERING_SMOOTHING_STANDARD else 1

        def get_target_stride(j):
            """Returns the absolute stride required for the final output."""
            if self.smoothing_mode == SCATTERING_SMOOTHING_STANDARD:
                return stride_max
            if self.subsample_output_scale_dependent:
                return 2**(j+1)
            # SCATTERING_SMOOTHING_SCALED, main variant
            return 1 

        def get_complex_format(x):
            if self.out_complex_structure == SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR:
                return (x.abs(), x.angle())
            return torch.view_as_real(x)

        def safe_unpad(tensor, k_factor):
            """
            Dynamically calculates unpadding based on the subsampling factor k.
            If k=1 (No subsampling), removes self.pad_size.
            If k=2 (Subsampled), removes self.pad_size // 2.
            """
            current_pad = self.pad_size // k_factor
            # We instantiate a temporary layer for safety, or you could use functional slicing
            # Assuming UnpaddingLayer accepts padding in init
            return UnpaddingLayer(padding=current_pad)(tensor)

        # --- 2. S0 PROCESSING ---
        # S0 is always Global Invariant (effectively stride max)
        # S0 has no subsampling in forward, so factor=1
        _s0 = self.unpadding_layer(s0)[..., ::s0_stride, ::s0_stride]
        
        results = {}
        if self.out_structure == SCATTERING_OUTPUT_STRUCTURE_ORDER:
            s_dict = {"S0": _s0, "S1": [], "S2": []}
            u_dict = {"U1": [], "U2": []} if "U" in self.outputs else None
            if "W" in self.outputs:
                w_dict = {"A1": [], "Phi1": [], "A2": [], "Phi2": []} if self.out_complex_structure == SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR else {"W1": [], "W2": []}
            else:
                w_dict = None
            # --- ORDER 1 LOOP ---
            for j1 in range(self.J):
                # A. Determine Subsampling Factor of the input Tensor
                # If modulus_subsampling is True, the tensor is already downsampled by 2^j1
                k1 = 2**j1 if self.modulus_subsampling else 1

                # B. Unpad Dynamically
                s1_t = safe_unpad(s1_list[j1], k1)

                # C. Slice with Relative Stride
                # We need target_stride total. We already have k1.
                # slice_step = target / k1
                target_s1 = get_target_stride(j1)
                slice_s1 = max(1, target_s1 // k1)
                s_dict["S1"].append(s1_t[..., ::slice_s1, ::slice_s1])

                # Optional Outputs (U/W)
                if u_dict is not None:
                    u_dict["U1"].append(safe_unpad(u1_list[j1], k1)[..., ::slice_s1, ::slice_s1])
                
                if w_dict is not None:
                    # W is usually not subsampled in forward, check your forward logic.
                    # Assuming W matches U resolution for consistency here:
                    w1_t = safe_unpad(w1_list[j1], k1) # W might be k=1 if extracted before modulus
                    _ = get_complex_format(w1_t)
                    if isinstance(_, tuple):
                        w_dict["A1"].append(_[0])
                        w_dict["Phi1"].append(_[1])
                    else:
                        w_dict["W1"].append(_)
                
                # --- ORDER 2 LOOP ---
                for j2 in range(self.J):
                    idx_2 = j1 * self.J + j2
                    
                    # A. Determine Subsampling Factor
                    # If modulus_subsampling, tensor is downsampled by 2^j2
                    k2 = 2**j2 if self.modulus_subsampling else 1
                    
                    # B. Unpad Dynamically
                    s2_t = safe_unpad(s2_list[idx_2], k2)
                    
                    # C. Slice Relative
                    target_s2 = get_target_stride(j2)
                    slice_s2 = max(1, target_s2 // k2)
                    s_dict["S2"].append(s2_t[..., ::slice_s2, ::slice_s2])

                    # Optional Outputs
                    if u_dict is not None:
                        u_dict["U2"].append(safe_unpad(u2_list[idx_2], k2)[..., ::slice_s2, ::slice_s2])
                    
                    if w_dict is not None:
                        w2_t = safe_unpad(w2_list[idx_2], k2)
                        _ = get_complex_format(w2_t)
                        if isinstance(_, tuple):
                            w_dict["A2"].append(_[0])
                            w_dict["Phi2"].append(_[1])
                        else:
                            w_dict["W2"].append(_)
                
            results["S"] = s_dict
            if u_dict: results["U"] = u_dict
            if w_dict: results["W"] = w_dict

        # --- 4. STRUCTURE: SCALE ---
        elif self.out_structure == SCATTERING_OUTPUT_STRUCTURE_SCALE:
            results["S0"] = _s0

            for j1 in range(self.J):
                scale_key = f"scale_{j1}"
                scale_obj = {}
                k1 = 2**j1 if self.modulus_subsampling else 1

                # S1
                s1_t = safe_unpad(s1_list[j1], k1)
                target_s1 = get_target_stride(j1)
                slice_s1 = max(1, target_s1 // k1)
                scale_obj["S1"] = s1_t[..., ::slice_s1, ::slice_s1]

                # U1 / W1
                if "U" in self.outputs:
                    scale_obj["U1"] = safe_unpad(u1_list[j1], k1)[..., ::slice_s1, ::slice_s1]
                if "W" in self.outputs:
                    w1_t = safe_unpad(w1_list[j1], k1)
                    _ = get_complex_format(w1_t)
                    if isinstance(_, tuple):
                        scale_obj["A1"] = _[0]   
                        scale_obj["Phi1"] = _[1] 
                    else:
                        scale_obj["W1"] = _

                # --- PREPARE S2/U2/W2 LISTS (Before j2 loop!) ---
                scale_obj["S2"] = []
                if "U" in self.outputs: 
                    scale_obj["U2"] = []
                
                # Initialize W2 containers based on structure
                if "W" in self.outputs:
                    if self.out_complex_structure == SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR:
                        scale_obj["A2"] = []
                        scale_obj["Phi2"] = []
                    else:
                        scale_obj["W2"] = []

                for j2 in range(self.J):
                    idx_2 = j1 * self.J + j2
                    k2 = 2**j2 if self.modulus_subsampling else 1
                    
                    # S2
                    s2_t = safe_unpad(s2_list[idx_2], k2)
                    target_s2 = get_target_stride(j2)
                    slice_s2 = max(1, target_s2 // k2)
                    scale_obj["S2"].append(s2_t[..., ::slice_s2, ::slice_s2])
                    
                    if "U" in self.outputs:
                        scale_obj["U2"].append(safe_unpad(u2_list[idx_2], k2)[..., ::slice_s2, ::slice_s2])
                    if "W" in self.outputs:
                        w2_t = safe_unpad(w2_list[idx_2], k2)
                        _ = get_complex_format(w2_t)
                        if isinstance(_, tuple):
                            scale_obj["A2"].append(_[0])
                            scale_obj["Phi2"].append(_[1])
                        else:
                            scale_obj["W2"].append(_)
                
                results[f"scale_{j1}"] = scale_obj
    
        return results

    # ---------------------------------------------------------------------
    # NEW: Meta-Data Helper
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
