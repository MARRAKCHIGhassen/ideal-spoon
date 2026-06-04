# -*- coding: utf-8 -*-
"""
Global Constants for the Phase-Aware Scattering Encoder-Decoder Project.

This module defines immutable constants used throughout the package for:
1.  **System Configuration**: Data splits, execution phases, and component types.
2.  **Scattering Transform Parameters**: Configuration for the wavelet scattering network,
    including smoothing strategies and output structures.
3.  **Model Architecture**: Specific flags for the novel contributions of this work,
    such as Phase-Aware Skip Connections and Spatial Shuffling Ablations.

Usage:
    from constants import KIND_MODEL, SPLIT_TRAIN, SKIP_COMPLEX_MAG_PHASE
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

from typing import Final

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------
# Default undefined state (do not modify at runtime)
UNDEFINED: Final[str] = "undefined"

# -------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------
# Core Kinds
# ------------------------------------------------
# Identifiers for the primary components of the training pipeline.

KIND_MODEL: Final[str]    = "model"     # The neural network architecture
KIND_DATA: Final[str]     = "data"      # Dataloaders and dataset wrappers
KIND_OPTIM: Final[str]    = "optimizer" # Optimization algorithms (Adam, SGD, etc.)
KIND_LOSS: Final[str]     = "loss"      # Loss functions (MSE, CrossEntropy, etc.)
KIND_METRIC: Final[str]   = "metric"    # Evaluation metrics (PSNR, SSIM, IoU)
KIND_PROFILER: Final[str] = "profiler"  # Performance profiling tools
KIND_PHASE: Final[str]    = "phase"     # Execution phase context
KIND_PATTERN: Final[str]  = "pattern"   # Regex or file patterns
KIND_PLOT: Final[str]     = "plot"      # Visualization outputs
KIND_CUSTOM: Final[str]   = "custom"    # User-defined modules

KINDS: Final[list[str]] = [
    KIND_DATA,
    KIND_MODEL,
    KIND_OPTIM,
    KIND_LOSS,
    KIND_METRIC,
    KIND_PATTERN,
    KIND_PLOT,
    KIND_PROFILER,
]

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# SPLITS
# ---------------------------------------------------------------------
# Standard dataset partitions.

SPLIT_TRAIN: Final[str]     = "train"
SPLIT_VAL: Final[str]       = "val"
SPLIT_TRAIN_VAL: Final[str] = "train/val"
SPLIT_TEST: Final[str]      = "test"
SPLIT_ALL: Final[str]       = "all"

SPLITS: Final[list[str]] = [
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TRAIN_VAL,
    SPLIT_TEST,
    SPLIT_ALL,
]

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# PHASES
# ---------------------------------------------------------------------
# Execution modes for the pipeline.

PHASE_TRAIN: Final[str]     = "train"      # Training loop
PHASE_TRAIN_VAL: Final[str] = "train/val"  # Training with validation
PHASE_TEST: Final[str]      = "test"       # Inference/Evaluation only

PHASES: Final[list[str]] = [
    PHASE_TRAIN,
    PHASE_TRAIN_VAL,
    PHASE_TEST,
]

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# CUSTOM: SCATTERING
# ---------------------------------------------------------------------
# Configuration for the Wavelet Scattering Transform and Hybrid Architecture.

# SMOOTHING
# Strategies for the low-pass filtering in the scattering transform.
SCATTERING_SMOOTHING_STANDARD: Final[str] = "standard" # Standard Gaussian smoothing
SCATTERING_SMOOTHING_SCALED: Final[str]   = "scaled"   # Smoothing scaled by scale factor
SCATTERING_SMOOTHING: Final[list[str]] = [
    SCATTERING_SMOOTHING_STANDARD,
    SCATTERING_SMOOTHING_SCALED,
]

# OUTPUT TYPE
SCATTERING_OUTPUT_TYPE_ARRAY: Final[str] = "array" # Concatenated tensor output
SCATTERING_OUTPUT_TYPE_DICT: Final[str]  = "dict"  # Dictionary split by order
SCATTERING_OUTPUT_TYPE: Final[list[str]] = [
    SCATTERING_OUTPUT_TYPE_ARRAY,
    SCATTERING_OUTPUT_TYPE_DICT,
]

# OUTPUT STRUCTURE
SCATTERING_OUTPUT_STRUCTURE_ORDER: Final[str] = "order" # Group coefficients by scattering order (m=1, m=2)
SCATTERING_OUTPUT_STRUCTURE_SCALE: Final[str] = "scale" # Group coefficients by scale (j)
SCATTERING_OUTPUT_STRUCTURE: Final[list[str]] = [
    SCATTERING_OUTPUT_STRUCTURE_ORDER,
    SCATTERING_OUTPUT_STRUCTURE_SCALE,
]

# OUTPUT COMPLEX STRUCTURE
# Determines how complex coefficients are represented.
SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR: Final[str] = "polar" # Magnitude and Phase (Real, Imag or Mag, Angle)
SCATTERING_OUTPUT_COMPLEX_STRUCTURE_RAW: Final[str]   = "raw"   # Raw complex numbers (if supported)
SCATTERING_OUTPUT_COMPLEX_STRUCTURE: Final[list[str]] = [
    SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR,
    SCATTERING_OUTPUT_COMPLEX_STRUCTURE_RAW,
]

# INTERPOLATION
# Downsampling/Upsampling strategies within the network.
INTERPOLATION_BILINIAR: Final[str] = "bilinear"
INTERPOLATION_NEAREST: Final[str]  = "nearest"
INTERPOLATION: Final[list[str]] = [
    INTERPOLATION_BILINIAR,
    INTERPOLATION_NEAREST,
]

# SKIP CONNECTIONS
# Defines the information passed through skip connections in the Encoder-Decoder.
# This directly relates to the "Phase-Aware" contribution of the abstract.
SKIP_STANDARD: Final[str] = "standard"        # Standard U-Net style skips
SKIP_MODULUS: Final[str]  = "modulus"         # Pass only the magnitude (Scattering invariant)
SKIP_COMPLEX_RAW: Final[str]        = SCATTERING_OUTPUT_COMPLEX_STRUCTURE_RAW
SKIP_COMPLEX_MAG_PHASE: Final[str]  = SCATTERING_OUTPUT_COMPLEX_STRUCTURE_POLAR # Explicitly preserve phase
SKIP: Final[list[str]] = [
    SKIP_STANDARD,
    SKIP_MODULUS,
    SKIP_COMPLEX_RAW,
    SKIP_COMPLEX_MAG_PHASE,
]

# SHUFFLE MODE
# Controls the spatial shuffling ablation study mentioned in the abstract.
# Used to demonstrate that phase encodes location-dependent structure.
SHUFFLE_MODE_ALL: Final[str]       = "all"       # Shuffle all coefficients
SHUFFLE_MODE_PHASE: Final[str]     = "phase"     # Shuffle only phase information
SHUFFLE_MODE_AMPLITUDE: Final[str] = "amplitude" # Shuffle only amplitude information
SHUFFLE_MODES: Final[list[str]] = [
    SHUFFLE_MODE_ALL,
    SHUFFLE_MODE_PHASE,
    SHUFFLE_MODE_AMPLITUDE,
]

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# CUSTOM: DATA
# ---------------------------------------------------------------------
# Augmentation strategies.

# ROTATION TRANSFORM
ROTATION_STRATEGIE_UNIFORM = "uniform"   # Continuous random rotation
ROTATION_STRATEGIE_DISCRETE = "discrete" # Fixed 90-degree increments

ROTATION_STRATEGIES: Final[list[str]] = [
    ROTATION_STRATEGIE_UNIFORM,
    ROTATION_STRATEGIE_DISCRETE,
]

# -------------------------------------------------------------------------------------------------------------------