# phase-aware-icml-26
This repository contains the high-resolution supplementary visualizations requested during the rebuttal phase.

## Figure S1-S3: Full Architecture
End-to-end architecture: scattering encoder (stride-1, $J=3$, $L=8$) 
with phase-aware skip connections to decoder via gating and fusion.

* files:
  - Full_Architecture.pdf
  - Scattering_Encoder_Pipeline.pdf
  - Upsampling_Block.pdf

## Figure S4-S7: Phase Maps (3 scales × 4 orientations)
Phase coefficients $\Phi_j^\theta$ showing sharp coherent transitions at 
edges and boundaries, with orientation-selective edge detection.

* files:
  - Phase_Encoder_Overview_Per_Scale_General_semantic.pdf
  - Phase_Encoder_Overview_Per_Scale_Animal_Portrait_semantic.pdf
  - Phase_Encoder_Overview_Per_Scale_Landscape_semantic.pdf
  - Phase_Encoder_Overview_Per_Scale_Sharp_Edges_semantic.pdf

## Figure S8–S11: Phase Overlay on BSD68 Images
Summed phase energy $\sum_{\theta}|\Phi_0^\theta|$ overlaid on input images. Phase 
concentrates at edges and structural boundaries across image types.

* files:
  - Phase_Overlay_j0_General.pdf
  - Phase_Overlay_j0_Animal_Portrait.pdf
  - Phase_Overlay_j0_Landscape.pdf
  - Phase_Overlay_j0_Sharp_Edges.pdf


## Figure S12–S15: Magnitude Maps
Amplitude $|W|$ showing directional energy concentrations, smoother 
and less sharply localized than phase.

* files:
  - Magnitude_Encoder_Overview_Per_Scale_General_semantic.pdf
  - Magnitude_Encoder_Overview_Per_Scale_Animal_Portrait_semantic.pdf
  - Magnitude_Encoder_Overview_Per_Scale_Landscape_semantic.pdf
  - Magnitude_Encoder_Overview_Per_Scale_Sharp_Edges_semantic.pdf


## Figure S16: Denoising Comparison
Noisy $\rightarrow$ M6 (Ours) $\rightarrow$ DnCNN $\rightarrow$ DSWN $\rightarrow$ Ground Truth across four 
BSD68 image categories.

* files:
  - Noise_Effect_Overview.pdf


## Figure S17: Perturbation Robustness
PSNR drop under translations (1–8 px) and rotations (0.5° -- 10°). 
M6 is 1.55x more robust than DnCNN at 10° rotation.

* files:
  - Perturbation_Robustness.pdf


## Figure S18: Stability Comparison
Absolute PSNR and PSNR drop under rotation for M6, DnCNN, DSWN.

* files:
  - Stability_Comparison.pdf
