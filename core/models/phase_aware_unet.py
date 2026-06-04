# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Phase‑Aware U‑Net
===================================================================

This module implements a hybrid architecture combining a Scattering Transform 
Encoder for texture feature extraction and a Phase-Aware Decoder for structure 
reconstruction. It supports multiple skip connection types (modulus, raw complex, 
and polar magnitude-phase) and spatial shuffling for ablation studies.
"""

# Importing Global Libraries
from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List, Dict

# Import custom libraries
from ._scattering_encoder.standard_monolithic_2d_encoder import StandardMonolithicScattering2DEncoder
from ._scattering_encoder.standard_vectorized_2d_encoder import StandardVectorizedScattering2DEncoder
from ._unet_decoder.scattering_u_net_decoder import ScatteringUNetDecoder

from ...constants import SCATTERING_SMOOTHING_SCALED, SCATTERING_SMOOTHING_STANDARD
from ...constants import SCATTERING_OUTPUT_TYPE_DICT
from ...constants import SCATTERING_OUTPUT_STRUCTURE_SCALE
from ...constants import SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR, SCATTERING_OUTPUT_COMPLEX_STRUCTURE_RAW
from ...constants import SKIP_MODULUS, SKIP_COMPLEX_RAW, SKIP_COMPLEX_MAG_PHASE
from ...constants import INTERPOLATION_NEAREST
from ...constants import SHUFFLE_MODE_ALL, SHUFFLE_MODE_PHASE, SHUFFLE_MODE_AMPLITUDE

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

# == PhaseAwareUNet ==
class PhaseAwareUNet(nn.Module):
    """
    Hybrid Scattering-Inspired U-Net for supervised signal recovery.

    This model replaces traditional convolutional encoders with a fixed 
    Scattering Transform to capture multi-scale texture information. The 
    decoder is augmented with phase-gating mechanisms to utilize the 
    structural information preserved in the complex scattering coefficients.

    Attributes
    ----------
    L : int
        Number of orientations in the scattering transform.
    skip_type : str
        The mathematical representation of the skip connections.
    scattering_implementation : str
        Choice between 'monolithic' or 'vectorized' scattering backends.
    shuffle_mode : Optional[str]
        Ablation flag to shuffle spatial information in skips.
    encoder : nn.Module
        The scattering-based feature extractor.
    bottleneck : nn.Conv2d
        Linear projection of aggregated scattering coefficients.
    decoder : ScatteringUNetDecoder
        Phase-aware upsampling and feature fusion network.
    """
    
    # ===========

    # -- __init__
    def __init__(
        self,
        M: int, N: int,
        J: int, L: int,
        in_channels: int,
        base_channels: int,
        slant: float = 0.5,
        out_channels: int = 6,
        gate_hidden_channels: int = 32,
        eps: float = 1e-6,
        interpolation: str = INTERPOLATION_NEAREST, # "bilinear" or "nearest"
        # -- Variant defintion Parameters --
        smoothing_mode: str = SCATTERING_SMOOTHING_STANDARD,
        skip_type: str = SKIP_COMPLEX_MAG_PHASE, # "modulus", "raw", "polar"
        modulus_subsampling: bool = False,
        skip_subsample_scale_dependent: bool = False,
        shuffle_mode: Optional[str] = None, # "all", "phase", "amplitude"
        # -- Loading Parameters --
        scattering_implementation: str = "monolithic",
        fuse_bias: bool = False,
        upsample_bias: bool = False,
        activation: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the PhaseAwareUNet with a scattering encoder and gated decoder.

        Parameters
        ----------
        M, N : int
            Spatial dimensions of the input images.
        J : int
            Number of scattering scales (log-base-2 of the spatial invariant).
        L : int
            Number of orientations in the scattering transform.
        in_channels : int
            Number of input image channels.
        base_channels : int
            Number of filters in the internal bottleneck projection.
        slant : float, optional
            Slant of the Morlet wavelets, by default 0.5.
        skip_type : str, optional
            Representation of skips (modulus, raw, polar), by default polar.
        scattering_implementation : str, optional
            Backend engine for scattering ('monolithic', 'vectorized'), by default "monolithic".
        out_channels : int, optional
            Number of output channels for reconstruction, by default 6.
        gate_hidden_channels : int, optional
            Dimensionality of the phase-gating hidden layers, by default 32.
        eps : float, optional
            Numerical stability constant for phase computations, by default 1e-6.
        interpolation : str, optional
            Upsampling mode ('bilinear', 'nearest'), by default 'nearest'.
        modulus_subsampling : bool, optional
            Whether to subsample modulus coefficients, by default False.
        subsample_scale_dependent : bool, optional
            Flag for scale-specific subsampling logic, by default False.
        shuffle_mode : str, optional
            Ablation mode for spatial shuffling, by default None.
        **kwargs : Any
            Additional parameters for the encoder or decoder components.
        """
        # ------------------------------------------------

        super().__init__()
        
        self.M, self.N = M, N
        self.J, self.L= J, L
        self.slant = slant
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.gate_hidden_channels = gate_hidden_channels
        self.eps = eps
        self.interpolation = interpolation
        # -- Variant defintion Parameters --
        self.smoothing_mode = smoothing_mode
        self.skip_type = skip_type
        self.modulus_subsampling = modulus_subsampling
        self.skip_subsample_scale_dependent = skip_subsample_scale_dependent
        self.shuffle_mode = shuffle_mode
        # -- Loading Parameters --
        self.scattering_implementation = scattering_implementation
        
        semantic_logger.info(f"Initializing PhaseAwareUNet (Implementation: {self.scattering_implementation})")
        debug_logger.debug(f"Architecture: J={self.J}, L={self.L}, skip_type={self.skip_type}, shuffle={self.shuffle_mode}")

        # 1. Encoder Construction
        debug_logger.debug("Building Scattering Encoder...")
        enc_params = {
            "M": self.M, "N": self.N, "J": self.J, "L": self.L, "slant": self.slant,
            "smoothing_mode": self.smoothing_mode,
            "pre_pad": False,
            "outputs": ("S", "U") if self.skip_type == SKIP_MODULUS else ("S", "W"),
            "out_type": SCATTERING_OUTPUT_TYPE_DICT,
            "out_structure": SCATTERING_OUTPUT_STRUCTURE_SCALE,
            "out_complex_structure": SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR 
            if self.skip_type == SKIP_COMPLEX_MAG_PHASE 
            else SCATTERING_OUTPUT_COMPLEX_STRUCTURE_RAW,
            "modulus_subsampling": False,
            "subsample_output_scale_dependent": False,
            **kwargs
        }

        self.encoder = StandardMonolithicScattering2DEncoder(**enc_params) if self.scattering_implementation == "monolithic" else StandardVectorizedScattering2DEncoder(**enc_params)
        
        # 2. Skip Connection Logic Routing
        debug_logger.debug(f"Routing skip logic for type: {skip_type}")
        if skip_type == SKIP_MODULUS:
            self.build_skips_dict = self.build_skips_dict_modulus
            self.build_phase_maps_list = self.build_phase_maps_list_modulus
        elif skip_type == SKIP_COMPLEX_RAW:
            self.build_skips_dict = self.build_skips_dict_raw
            self.build_phase_maps_list = self.build_phase_maps_list_raw
        elif skip_type == SKIP_COMPLEX_MAG_PHASE:
            self.build_skips_dict = self.build_skips_dict_mag_phase
            self.build_phase_maps_list = self.build_phase_maps_list_phase

        # 3. Bottleneck Initialization
        # Total channels = S0 (in_ch) + S1 (in_ch * J * L) + S2 (in_ch * (J * L)^2)
        # We account for the full scattering tree (all j1, j2 pairs) produced by the encoder.
        total_scattering_channels = in_channels * (1 + (self.J * self.L) + (self.J * self.L)**2)
        debug_logger.debug(f"Bottleneck Input Channels: {total_scattering_channels}")
        
        self.bottleneck = nn.Conv2d(
            in_channels=total_scattering_channels,
            out_channels=base_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        
        # 4. Decoder Initialization
        debug_logger.debug("Building Phase-Aware Decoder...")
        self.decoder = ScatteringUNetDecoder(
            J=J, L=L,
            in_channels=in_channels,
            base_channels=base_channels,
            skip_type = skip_type,
            out_channels=out_channels,
            gate_hidden_channels=gate_hidden_channels,
            eps=eps,
            upsample_kernel_size=3,
            upsample_padding=1,
            upsample_bias=upsample_bias if upsample_bias is not None else False,
            upsample_inplace=True,
            interpolation=interpolation,
            align_corners=False,
            fuse_kernel_size=1,
            fuse_padding=0,
            fuse_bias=fuse_bias if fuse_bias is not None else False,
            activation=activation,
            **kwargs
        )
        semantic_logger.info("PhaseAwareUNet construction complete.")
    
    # -- forward
    def forward(self, x: torch.Tensor, use_skip_loss: bool = False) -> Dict[str, Any]:
        """
        Executes the hybrid Scattering-Phase forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images (B, C, M, N).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing 'pred' (reconstruction), 'scattering_coeffs' 
            (bottleneck features), and 'phase_maps' (internal structural data).
        """
        # ------------------------------------------------

        # 1. Fixed Scattering Encoding
        debug_logger.debug(f"Forward Pass: Input Tensor {x.shape}")
        raw = self.encoder(x)

        # 2. Skip and Bottleneck Feature Preparation
        debug_logger.debug("Building skip dictionary and aggregating bottleneck features.")
        skip_feat_list = self.build_skips_dict(raw)
        bottleneck_feat = self.transform_single_tensor(raw)

        # 3. Bottleneck Projection
        # Maps concatenated scattering coefficients to the decoder base channel depth
        debug_logger.debug(f"Bottleneck Input: {bottleneck_feat.shape}")
        x_latent = self.bottleneck(bottleneck_feat)
        debug_logger.debug(f"Bottleneck Output: {x_latent.shape}")

        # 4. Gated Decoding
        # Fuses latent coarse features with high-frequency skips using gated logic
        debug_logger.debug("Entering ScatteringUNetDecoder...")
        out = self.decoder(x_latent, skip_feat_list)
        
        return {
            "pred": out,                                        # The actual reconstructed image
            "scattering_coeffs": bottleneck_feat,               # The internal texture data
            "phase_maps": self.build_phase_maps_list(raw)       # The internal structure data
        }

    # -- transform_single_tensor
    def transform_single_tensor(self, raw: Dict[str, Any]) -> torch.Tensor:
        """
        Aggregates multi-scale scattering coefficients into a single texture map.
        
        This method iterates through the encoder's dictionary structure to collect
        S0, all S1 scales, and all S2 scales, flattening them into a dense 
        bottleneck tensor aligned with the coarsest resolution (S0).

        Returns
        -------
        torch.Tensor
            Concatenated tensor [S0, S1_all, S2_all] of shape (B, C_total, H_s, W_s).
        """
        # 1. Extract Global Invariant S0 (Reference Resolution)
        # Shape: (B, C, H_s, W_s)
        s0 = raw["S0"]
        B, C, H_s0, W_s0 = s0.shape
        
        # Containers for flattening
        # We start with S0
        feats = [s0]

        # 2. Iterate over all scales to collect S1 and S2
        # The encoder outputs 'scale_0' ... 'scale_{J-1}'
        for j1 in range(self.J):
            key = f"scale_{j1}"
            if key not in raw: continue
            scale_data = raw[key]

            # --- Process S1 ---
            # Shape: (B, C, L, H_j, W_j) -> Flatten to (B, C*L, H_j, W_j)
            s1 = scale_data["S1"]
            B_s, C_s, L_s, H_s, W_s = s1.shape
            s1_flat = s1.reshape(B_s, C_s * L_s, H_s, W_s)
            
            # Align resolution to S0 if needed (Handling 'scaled' mode)
            if (H_s, W_s) != (H_s0, W_s0):
                # We simply interpolate or stride. Stride is safer for aliasing.
                # Calculate stride factor
                stride_h = H_s // H_s0
                stride_w = W_s // W_s0
                if stride_h > 0 and stride_w > 0:
                     s1_flat = s1_flat[..., ::stride_h, ::stride_w]
                else:
                    # Fallback: S0 is larger than S1 (unlikely in scattering)
                     s1_flat = F.interpolate(s1_flat, size=(H_s0, W_s0), mode='nearest')
            feats.append(s1_flat)

            # --- Process S2 ---
            # S2 is a list of tensors (children), one per j2 > j1
            if "S2" in scale_data and len(scale_data["S2"]) > 0:
                s2_list = scale_data["S2"] # List of (B, C, L, H, W)
                
                # Stack children: (B, C, K_siblings, L, H, W)
                s2_stack = torch.stack(s2_list, dim=2) 
                
                # Flatten everything: (B, C * K * L, H, W)
                K_sibs = s2_stack.shape[2]
                total_s2_channels = C_s * K_sibs * L_s * L_s
                s2_flat = s2_stack.reshape(B_s, total_s2_channels, H_s, W_s)
                
                # Align resolution (Same logic as S1)
                if (H_s, W_s) != (H_s0, W_s0):
                    stride_h = H_s // H_s0
                    stride_w = W_s // W_s0
                    if stride_h > 0 and stride_w > 0:
                        s2_flat = s2_flat[..., ::stride_h, ::stride_w]
                    else:
                        s2_flat = F.interpolate(s2_flat, size=(H_s0, W_s0), mode='nearest')
                feats.append(s2_flat)

        # 3. Concatenate all features
        # [S0, S1_j0, S2_j0, S1_j1, S2_j1, ...]
        # return torch.cat(feats, dim=1)
        bottleneck_feat = torch.cat(feats, dim=1)
        # ENERGY PRESERVING NORMALIZATION (safe for ablation)
        s0_energy = torch.norm(raw["S0"])**2 + 1e-8
        total_energy = torch.norm(bottleneck_feat)**2 + 1e-8
        scale_factor = torch.sqrt(s0_energy / total_energy)  # S0 energy reference
        
        bottleneck_feat = bottleneck_feat * scale_factor  # Now ||bottleneck||² ≈ ||S0||² ≈ ||x||²

        return bottleneck_feat

    # -- build_skips_dict_modulus
    def build_skips_dict_modulus(self, x: Dict[str, Any]) -> Dict[int, torch.Tensor]:
        """
        Extracts first-order modulus coefficients (U1) as skip connections.

        Parameters
        ----------
        x : Dict[str, Any]
            The raw output dictionary from the scattering encoder.

        Returns
        -------
        Dict[int, torch.Tensor]
            Dictionary mapping scale index 'j' to tensors of shape (B, C, L, H_j, W_j).
        """
        skips = {}
        for j in range(self.J):
            key = f"scale_{j}"
            if key in x and "U1" in x[key]:
                # Shape: (B, C, L, H, W)
                u1 = x[key]["U1"]
                
                # Hardening: Check for extra singleton dims sometimes added by monalithic logic
                if u1.ndim == 6 and u1.shape[2] == 1:
                    u1 = u1.squeeze(2)
                
                # --- SUBSAMPLING LOGIC ---
                if self.skip_subsample_scale_dependent:
                    # Encoder output is dense (stride 1). We need stride 2^j.
                    stride = 2**j
                    if stride > 1:
                        u1 = u1[..., ::stride, ::stride]

                skips[j] = u1
        return skips
    
    # -- build_skips_dict_raw
    def build_skips_dict_raw(self, x: Dict[str, Any]) -> Dict[int, torch.Tensor]:
        """
        Extracts raw complex coefficients (W1) as skip connections.

        Parameters
        ----------
        x : Dict[str, Any]
            The raw output dictionary from the scattering encoder.

        Returns
        -------
        Dict[int, torch.Tensor]
            Dictionary mapping scale index 'j' to complex tensors (W1).
        """
        # ------------------------------------------------
        skips = {}
        for j in range(self.J):
            key = f"scale_{j}"
            if key in x and "W1" in x[key]:
                # Shape: (B, C, L, H, W, 2)
                skips[j] = x[key]["W1"]

                # --- SUBSAMPLING LOGIC ---
                if self.skip_subsample_scale_dependent:
                    stride = 2**j
                    if stride > 1:
                        # w1 shape is (..., H, W, 2), so we stride dims -3 and -2
                        w1 = w1[..., ::stride, ::stride, :]
        return skips
    
    # -- build_skips_dict_mag_phase
    def build_skips_dict_mag_phase(self, x: Dict[str, Any]) -> Dict[int, torch.Tensor]:
        """
        Extracts and optionally shuffles Polar (Magnitude/Phase) skip connections.

        This method supports the M7 ablation logic by allowing spatial shuffling 
        of either the Amplitude (A1), the Phase (Phi1), or both.

        Parameters
        ----------
        x : Dict[str, Any]
            The raw output dictionary from the scattering encoder.

        Returns
        -------
        Dict[int, torch.Tensor]
            Dictionary mapping scale index 'j' to stacked (A, Phi) tensors.
        """
        # ------------------------------------------------
        debug_logger.debug(f"Building Polar Skip Connections. Shuffle mode: {self.shuffle_mode}")
        skips = {}
        for j in range(self.J):
            key = f"scale_{j}"
            if key not in x: continue
            
            # 1. Extract Components
            if "A1" in x[key] and "Phi1" in x[key]:
                A = x[key]["A1"]     # (B, C, L, H, W)
                Phi = x[key]["Phi1"] # (B, C, L, H, W)
            elif "W1" in x[key]:
                 # Fallback if polar conversion didn't happen in encoder
                 w = x[key]["W1"]
                 A = torch.norm(w, dim=-1)
                 Phi = torch.atan2(w[..., 1], w[..., 0])
            else:
                continue

            # 2. --- SUBSAMPLING LOGIC (Before Shuffling) ---
            if self.skip_subsample_scale_dependent:
                stride = 2**j
                if stride > 1:
                    A = A[..., ::stride, ::stride]
                    Phi = Phi[..., ::stride, ::stride]

            # 3. M7 Ablation: Spatial Shuffling
            if self.shuffle_mode:
                if self.shuffle_mode == SHUFFLE_MODE_ALL:
                    indices = self._get_shuffle_indices(A)
                    A = self._apply_shuffle(A, indices)
                    Phi = self._apply_shuffle(Phi, indices)
                elif self.shuffle_mode == SHUFFLE_MODE_AMPLITUDE:
                    A = self._apply_shuffle(A)
                elif self.shuffle_mode == SHUFFLE_MODE_PHASE:
                    Phi = self._apply_shuffle(Phi)

            # 3. Stack for Decoder
            # Decoder expects (B, C, L, H, W, 2) where 2 is (Mag, Phase)
            skips[j] = torch.stack([A, Phi], dim=-1)
            
        return skips
    
    # -- build_phase_maps_list_modulus
    def build_phase_maps_list_modulus(self, raw: Dict[str, Any]) -> List[torch.Tensor]:
        """Returns an empty list as modulus coefficients have no phase information."""
        # ------------------------------------------------
        return []

    # -- build_phase_maps_list_raw
    def build_phase_maps_list_raw(self, raw: Dict[str, Any]) -> List[torch.Tensor]:
        """
        Computes phase maps from raw complex coefficients (W1).
        """
        # ------------------------------------------------
        maps = []
        for j in range(self.J):
            key = f"scale_{j}"
            if key in raw and "W1" in raw[key]:
                w = raw[key]["W1"]
                # Compute phase from raw complex (Im, Re)
                phi = torch.atan2(w[..., 1], w[..., 0])
                # Flatten C and L for visualization: (B, C*L, H, W)
                B, C, L, H, W = phi.shape
                maps.append(phi.reshape(B, C*L, H, W))
        return maps

    # -- build_phase_maps_list_phase
    def build_phase_maps_list_phase(self, raw: Dict[str, Any]) -> List[torch.Tensor]:
        """
        Extracts pre-computed phase maps (Phi1) from polar representation.
        """
        # ------------------------------------------------
        maps = []
        for j in range(self.J):
            key = f"scale_{j}"
            if key in raw and "Phi1" in raw[key]:
                phi = raw[key]["Phi1"]
                B, C, L, H, W = phi.shape
                maps.append(phi.reshape(B, C*L, H, W))
        return maps

    # -- _get_shuffle_indices
    def _get_shuffle_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates random permutation indices for the spatial dimensions.
        Used to ensure structural ablation across orientations/channels.
        """
        H, W = x.shape[-2:]
        return torch.randperm(H * W, device=x.device)
    
    # -- _apply_shuffle
    def _apply_shuffle(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Spatially shuffles the last two dimensions of a tensor to destroy 
        structural information while preserving channel/orientation statistics.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, ..., H, W).
            
        Returns
        -------
        torch.Tensor
            The spatially shuffled tensor with the same shape.
        """
        # ------------------------------------------------
        # 1. Capture original shape
        B, *dims, H, W = x.shape # Get height and width
        
        # 2. Flatten spatial dimensions: (..., H, W) -> (..., H*W)
        # We use start_dim=-2 to flatten only the last two dimensions
        x_flat = x.view(B, *dims, -1) # (B, ..., H*W)
        
        # 3. Generate random permutation indices
        # We use a single permutation for the operation to save compute time
        if indices is None:
            indices = self._get_shuffle_indices(x)
        # num_pixels = H * W
        # idx = torch.randperm(num_pixels, device=x.device)
        
        # 4. Apply permutation and Reshape back
        # x_flat[..., idx] reorders the last dimension based on indices
        x_shuffled = x_flat[..., indices].view(B, *dims, H, W)
        
        return x_shuffled
    
    # -- eval
    def eval(self):
        """Standardizes eval mode across all sub-components."""
        # ------------------------------------------------
        return self.train(False)

    # -- train
    def train(self, mode: bool = True):
        """
        Sets the training mode with a fixed-encoder research constraint.
        
        Note: The scattering encoder remains in eval mode even when the rest 
        of the network is training to ensure deterministic feature extraction.
        """
        # ------------------------------------------------
        super().train(mode)
        if not mode:
            self.encoder.eval()
            self.bottleneck.eval()
            self.decoder.eval()
            return self

        # Research Logic: Freeze Encoder
        self.encoder.eval() 
        self.bottleneck.train()
        self.decoder.train()
        return self
