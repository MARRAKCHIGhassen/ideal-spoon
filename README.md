# Phase-Aware Scattering Encoder–Decoder: Decoupling Stability from Invariance for Dense Prediction (ICDM 2026 Submission)

This repository contains the core implementation and experimental data for the "Phase-Aware Scattering Encoder–Decoder: Decoupling Stability from Invariance for Dense Prediction" submission. The code is organized as a modular library to facilitate easy integration and reproducibility.

## Repository Structure

- `core/`: Contains the primary mathematical and architectural contributions.
    - `models/`: Implementation of the Gated Scattering Encoder-Decoder.
    - `losses/`: Phase-aware loss functions (PTV and Phase Alignment).
- `yaml/`: Detailed specifications for all ablation studies, including hyperparameters and architectural configurations used in the paper.
- `results/`: Raw and processed results organized by study, experiment, seed, and attempt, providing the evidence for the tables and figures in the main text.
- `Preliminary-Segmentation-Study/`: Specific experimental scripts and implementations for the ISIC skin lesion segmentation tasks.
- `constants.py`: Global constants and fixed parameters.

## Installation & Requirements

Ensure you have a modern Python environment (3.11+) and the following core dependencies:
- torch
- torchvision
- numpy
- pywavelets
- scipy

A `requirements.txt` file is provided for convenience:
`pip install -r requirements.txt`

## Usage & Integration

The architecture is designed to be modular. You can import the core components directly into your workflow:

```python
from core.models.scattering_gate import PhaseAwareScattering
from core.losses.phase_losses import PhaseAlignmentLoss

# Example: Initialize the scattering block
model = PhaseAwareScattering(input_dim=3, phase_preservation=True)
