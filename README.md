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
- `Plots`: Plots defining
- `ckpts`: The main checkpoint of Denoising M6 named as `M6-J_3-L_8.pth`

## Installation & Requirements

Ensure you have a modern Python environment (3.11+) and the following core dependencies:
- torch
- torchvision
- numpy
- pywavelets
- scipy
- matplotlib

A `requirements.txt` file is provided for convenience:
`pip install -r requirements.txt`

## Usage & Integration

The architecture is designed to be modular. You can import the core components directly into your workflow:

```python
import os

import matplotlib.pyplot as plt

from core.models.phase_aware_unet import PhaseAwareScattering
from core.data.bsd import BSDDataset

# ====== CUDA ======
device = "cuda" if torch.cuda.is_available() else 'cpu'

# ====== M6 ======
# # M6 Configuration
m6_CONFIG ={
    # -- Main target Parameters --
    "J": 3,
    "L": 8,
    "slant": 5e-1,
    "gate_hidden_channels": 16,
    "base_channels": 32,
    # -- Data and splits dependent Parameters --
    "M": 128,
    "N": 128,
    "in_channels": 1,
    "out_channels": 1,
    "interpolation": "nearest",
    # -- Variant defintion Parameters --
    "smoothing_mode": "scaled",
    "skip_type": "polar", 
    "skip_subsample_scale_dependent": False,
    "shuffle_mode": None,
    # -- Loading Parameters --
    "scattering_implementation": "vectorized",
    "warmup": False
}
# # M6 Instanciation
model = PhaseAwareScattering(**m6_CONFIG)
# # M6 Load checkpoint
checkpoint = torch.load(weights_path, map_location=device)
state_dict = checkpoint["models"].get("main", checkpoint["models"])
missing, unexpected = model.load_state_dict(state_dict, strict=True)
assert missing == []
assert unexpected == []
# # M6 Preparing
model.train() # Ensure we are in train mode to trigger the rebuild
with torch.no_grad():
    # Create a tiny dummy batch on the correct device
    dummy_flush = torch.randn(
        2,
        m6_CONFIG['in_channels'], m6_CONFIG['N'], m6_CONFIG['M'],
        device=device
    )
    # Push it through the entire model! 
    _ = model(dummy_flush)

# ====== BSD ======
bsd_CONFIG = {
    "root"  : os.path.join('.', 'data'),
    "sigma" : 25,
    "split" : "train", 
    "color" : 'grayscale',
}
bsd_dataset = BSDDataset(**bsd_CONFIG)

# ====== Usage ======
# # Example
outputs = model(next(iter(bsd_dataset)))
usable_predictions = outputs["pred"]
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.show()
