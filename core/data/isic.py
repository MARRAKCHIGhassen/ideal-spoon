# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: ISIC Dataset.
====================

This module provides the implementation for the International Skin Imaging 
Collaboration (ISIC) dataset, specifically tailored for lesion segmentation. 
It handles synchronized image-mask augmentations, spatial normalization, 
and split management for training and validation.
"""

# Importing Global Libraries
from __future__ import annotations

import logging
import os
import glob
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from typing import Final, Tuple, Optional, Dict, Callable, Any

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize

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

# == ISICDataset ==
class ISICDataset(Dataset):
    """
    Segmentation Dataset Wrapper for ISIC (Skin Lesion).

    This dataset handles paired Image-Mask data.
    
    Key Features:
    * **Synchronized Augmentation**: Applies the same random geometric transforms 
      (crop, flip, rotate) to both image and mask to maintain alignment.
    * **Normalization**: Standardizes input intensity (Mean/Std).
    * **Mask Handling**: Converts masks to long tensors (indices) for Cross-Entropy.

    Attributes:
        image_files (List[str]): Paths to input images.
        mask_files (List[str]): Paths to corresponding ground truth masks.
        size (Tuple[int, int]): Target spatial resolution (H, W).
    """
    
    # ===========

    # -- __init__
    def __init__(
        self, 
        root: str, 
        split: str = "train", 
        size: Tuple[int, int] = (256, 256),
        color: str = 'rgb',
        subset_fraction: float = 1.0,
        **kwargs
    ):
        """
        Initializes the ISIC Dataset.
        
        Args:
            root_dir: Path to dataset root.
            split: 'train', 'val', or 'test'.
            size: Target resize dimension.
        """
        # ------------------------------------------------
        super().__init__()

        self.root = root
        self.split = split
        self.size = size if isinstance(size, tuple) else tuple(size)
        self.grayscale = (color == 'grayscale')
        self.subset_fraction = subset_fraction
        
        semantic_logger.info(f"Initializing ISICDataset \n"
            f"(Mode: {'Train' if split == 'train' else 'Test'}, \n"
            f"Color: {'Grayscale' if self.grayscale else 'RGB'}) \n"
            f"Size: {self.size}, Subset: {subset_fraction})"
        )
        
        # 1. Path Resolution
        self.img_dir = os.path.join(root, "ISIC-18", split, "data")
        self.mask_dir = os.path.join(root, "ISIC-18", split, "ground_truth")
        semantic_logger.warning(f"Fetching {split} split from {os.path.join(root, 'ISIC-18', split)}")

        # 2. File Discovery
        extensions = ["*.jpg", "*.png"]
        self.images = []
        self.masks = []
        for ext in extensions:
            self.images.extend(glob.glob(os.path.join(self.img_dir, ext)))
            self.masks.extend(glob.glob(os.path.join(self.mask_dir, ext)))
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)

        # Validation checks
        if not self.images:
            semantic_logger.warning(f"No images found in {self.img_dir}")
            raise ValueError(f"No images found in {self.img_dir}")
        if not self.masks:
            semantic_logger.warning(f"No images found in {self.mask_dir}")
            raise ValueError(f"No masks found in {self.mask_dir}")
        if len(self.images) != len(self.masks) and split != 'test':
            min_len = min(len(self.img_dir), len(self.mask_dir))
            self.images = self.images[:min_len]
            self.masks = self.masks[:min_len]
            semantic_logger.warning(f"Mismatch between images and masks count. Keeping {min_len} samples.")

        # 3. Subsetting
        if subset_fraction < 1.0:
            count = int(len(self.images) * subset_fraction)
            self.images = self.images[:count]
            self.masks = self.masks[:count]
            debug_logger.info(f"Subset active: Loaded {count} samples ({subset_fraction*100}%)")
        

        # Standard Normalization (ImageNet)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        
        semantic_logger.info(f"ISICDataset loaded successfully with {len(self.images)} samples.")

    # -- __len__
    def __len__(self):
        return len(self.images)

    # -- __getitem__
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and transforms an image-mask pair.
        """
        # ------------------------------------------------

        debug_logger.debug(f"Fetching sample index: {idx}")

        # Load Image
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(img_path).convert('L' if self.grayscale else 'RGB')
        mask = Image.open(mask_path).convert('L')

        # 1. To Tensor (Convert immediately to fix type errors)
        # This scales [0, 255] -> [0.0, 1.0]
        img = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # 2. Resize (On Tensor)
        # Use InterpolationMode enum to handle Tensor resizing correctly
        img = TF.resize(img, list(self.size), interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, list(self.size), interpolation=InterpolationMode.NEAREST)

        # 3. Synchronized Augmentation (Train Only)
        if self.split == 'train':
            # Horizontal Flip
            if torch.rand(1) < 0.5:
                img = TF.hflip(img)
                if mask is not None: mask = TF.hflip(mask)

            # Vertical Flip
            if torch.rand(1) < 0.5:
                img = TF.vflip(img)
                if mask is not None: mask = TF.vflip(mask)

            # Rotation (90 deg increments)
            rot = torch.randint(0, 4, (1,)).item()
            if rot > 0:
                angle = rot * 90
                img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
                if mask is not None: mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        # 4. Final Processing
        if self.grayscale:
            img = TF.rgb_to_grayscale(img)
        else:
            # Apply Normalization for RGB
            img = self.normalize(img)
            
        # Binarize Mask (Ensure exact 0/1)
        if mask is not None:
            # 1. Binarize
            mask = (mask > 0.5).float()
            # 2. Remove the channel dim: (1, H, W) -> (H, W)
            mask = mask.squeeze(0)
            # 3. Convert to Long for CrossEntropy compatibility
            mask = mask.long()
        else:
            mask = torch.zeros(self.size, dtype=torch.long)

        # Return: Input (Image), Target (Mask)
        return img, mask

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# DATA MANAGER (NEW)
# ---------------------------------------------------------------------
class ISICDataManager:
    """
    Manager that spawns Phase-Specific Datasets (Train/Val/Test).
    """
    def __init__(self, **kwargs):
        self.config = kwargs
        self.subset_cfg = kwargs.get('subset_fraction', 1.0)

    def build_split(self, phase: str) -> ISICDataset:
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

        return ISICDataset(**params)
