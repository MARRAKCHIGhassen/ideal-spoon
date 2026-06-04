# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: BSD Dataset
============================================================

This module provides the implementation for the Berkeley Segmentation Dataset (BSD) 
specifically tailored for denoising benchmarks. It handles automated noise 
injection, geometric augmentations, and split management for BSD68 and BSD500.
"""

# Importing Global Libraries
from __future__ import annotations

import logging
import os
import glob
from PIL import Image
from typing import Final, Tuple, Optional, Dict, Callable, Any

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode # Required for Tensor operations

# Import custom libraries
from ...constants import KIND_DATA

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

# == BSDDataset ==
class BSDDataset(Dataset):
    r"""
    Standard Denoising Dataset Wrapper for BSD.

    This dataset produces pairs of (Noisy Input, Clean Target).
    
    Key Features:
    * **Dynamic Noise**: Adds Gaussian white noise ($\mathcal{N}(0, \sigma^2)$) on the fly.
    * **Grayscale Conversion**: Converts images to 1-channel tensors (Standard for Scattering).
    * **Geometric Augmentation**: Random crops and rotations to improve invariance learning.

    Attributes:
        files (List[str]): List of file paths.
        sigma (float): Standard deviation of the additive noise (in [0, 255] scale).
    """
    
    # ===========

    # -- __init__
    def __init__(
        self, 
        root: str, 
        sigma: float = 25.0,
        crop_size: int = 128,
        split: str = "train", 
        color: str = 'rgb',
        subset_fraction: float = 1.0, # Handled for grid search
        fixed_rotation: float = 0.0,  # NEW: Support for Stability Experiment
        **kwargs
    ):
        """
        Initializes the BSD Dataset.

        Parameters
        ----------
        root_dir : str
            Path to the dataset root containing 'train', 'test', 'val' subfolders.
        split : str
            The dataset partition to load.
        sigma : float
            Noise level to inject (standard deviation, 0-255).
        crop_size : int
            Spatial size for random crops during training.
        subset_fraction : float
            Percentage of data to load (useful for quick debugging).
        """
        # ------------------------------------------------

        super().__init__()

        self.root = root
        self.sigma = sigma
        self.split = split
        self.crop_size = crop_size
        self.grayscale = (color == 'grayscale')
        self.subset_fraction = subset_fraction
        self.fixed_rotation = fixed_rotation

        semantic_logger.info(
            f"Initializing BSDDataset \n"
            f"(Mode: {'Train' if split == 'train' else 'Test'}, \n"
            f"Color: {'Grayscale' if self.grayscale else 'RGB'}) \n"
            f"Crop size: {self.crop_size}, Subset: {subset_fraction}) \n"
            f"Fixed Rotation: {self.fixed_rotation}"
        )

        # 1. Path Resolution
        if not root:
            # Fallback for debugging, assuming 'data' exists in CWD
            if os.path.exists("data"):
                root = "data"
                debug_logger.debug("Root not provided, falling back to local 'data' directory.")
            else:
                raise ValueError("Dataset 'root' path is empty. Please specify 'root' in params or check paths config.")

        # 2. Select Sub-Dataset
        if self.split == 'train':
            # Train: Loads clean images from BSD400
            target_dir = os.path.join(root, "BSD", "BSD400")
        else:
            # Test/Val: Loads clean images from BSD68/original
            # We ignore 'noise25', 'noise50' folders and generate noise synthetically
            # to ensure consistency with the requested 'sigma' param.
            target_dir = os.path.join(root, "BSD", "BSD68", "original")

        if not os.path.exists(target_dir):
            raise ValueError(f"Target directory not found: {target_dir}")

        # Load Files (png or jpg)
        self.files = sorted(glob.glob(os.path.join(target_dir, "*.png")) + 
                            glob.glob(os.path.join(target_dir, "*.jpg")))

        if not self.files:
            raise ValueError(f"No images found in {target_dir}")
            
        # Subset Fraction
        if subset_fraction < 1.0:
            count = int(len(self.files) * subset_fraction)
            self.files = self.files[:max(1, count)]
            debug_logger.debug(f"Subset fraction applied. Reduced file count to {len(self.files)}")
            
        semantic_logger.info(f"BSDDataset ready with {len(self.files)} samples.")
    
    # -- __len__
    def __len__(self):
        return len(self.files)

    # -- __getitem__
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads an image, processes it, and adds noise.

        Returns
        -------
        noisy_t : torch.Tensor
            Input image with added Gaussian noise (1, H, W).
        clean_t : torch.Tensor
            Ground truth clean image (1, H, W).
        """
        # ------------------------------------------------

        path = self.files[idx]
        img = Image.open(path)
        debug_logger.debug(f"Loading image index {idx}: {path}")
        
        # Force Grayscale if requested (Standard for BSD68)
        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        # Spatial Normalization
        img = TF.to_tensor(img) 

        # A. Geometric Transforms
        if self.split == 'train':
            # Training: Random Patch Extraction and Flips

            # Shape is (C, H, W)
            H, W = img.shape[-2], img.shape[-1]
            
            # Ensure image is large enough for crop
            if H < self.crop_size or W < self.crop_size:
                raise ValueError(f"Image {path} too small ({H}x{W}) for crop {self.crop_size}")

            i = torch.randint(0, H - self.crop_size + 1, (1,)).item()
            j = torch.randint(0, W - self.crop_size + 1, (1,)).item()
            
            img = TF.crop(img, int(i), int(j), self.crop_size, self.crop_size)
            
            # Stochastic Flips
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
            if torch.rand(1).item() < 0.5:
                img = TF.vflip(img)

            # Random 90-degree Rotations
            rot = torch.randint(0, 4, (1,)).item()
            if rot > 0:
                img = TF.rotate(img, rot * 90, interpolation=InterpolationMode.NEAREST)
        else:
            # We must center crop to EXACTLY crop_size (128).
            # 1. Matches Scattering Filters (128x128).
            # 2. Fits U-Net because 128 is divisible by 16.
            img = TF.center_crop(img, [self.crop_size, self.crop_size])
            
            # Apply fixed rotation for Stability Experiment
            if self.fixed_rotation != 0:
                img = TF.rotate(img, self.fixed_rotation, interpolation=InterpolationMode.BILINEAR)

        # C. Add Noise
        sigma_norm = self.sigma / 255.0
        noise = torch.randn_like(img) * sigma_norm
        noisy_t = torch.clamp(img + noise, 0.0, 1.0)

        # Return: Input (Noisy), Target (Clean)
        return noisy_t, img

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# DATA MANAGER (NEW)
# ---------------------------------------------------------------------
class BSDDataManager:
    """
    Manager that spawns Phase-Specific Datasets (Train/Val/Test).
    """
    def __init__(self, **kwargs):
        self.config = kwargs
        self.subset_cfg = kwargs.get('subset_fraction', 1.0)

    def build_split(self, phase: str) -> BSDDataset:
        params = self.config.copy()
        
        if phase == 'train':
            params['split'] = 'train'
        elif phase == 'val':
            params['split'] = 'val'
        else:
            params['split'] = 'test'
        semantic_logger.info(f"Split: {params['split']}")

        if isinstance(self.subset_cfg, dict):
            params['subset_fraction'] = self.subset_cfg.get(phase, 1.0)
        else:
            params['subset_fraction'] = float(self.subset_cfg)
        semantic_logger.info(f"Used subset fraction... {params['subset_fraction']}")

        return BSDDataset(**params)
