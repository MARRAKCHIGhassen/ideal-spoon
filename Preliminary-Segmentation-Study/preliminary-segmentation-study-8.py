import torchvision.transforms.functional as TF
from PIL import Image
import torch
import torch.nn as nn
from pprint import pformat
def tensor_to_grayscale_to_tensor(tensor: torch.Tensor):
    """
    1. Converts a Tensor (C, H, W) to a PIL Image.
    2. Converts that Image to Grayscale (L mode).
    3. Converts it back to a Tensor (1, H, W).
    """

    return TF.rgb_to_grayscale(tensor, num_output_channels=1)
    
device = "cuda" if torch.cuda.is_available() else 'cpu'
from phase_aware.core.models.phase_aware_unet import PhaseAwareUNetFactory
LOCAL_MODEL_CONFIG ={
            "J": 2,
            "L": 2,
            "slant": 5e-1,
            "gate_hidden_channels": 4,
            "base_channels": 8,
            # -- Data and splits dependent Parameters --
            "M": 256,
            "N": 256,
            "in_channels": 1,
            "out_channels": 2,
            "interpolation": "nearest",
            # -- Variant defintion Parameters --
            "smoothing_mode": "scaled",
            "skip_type": "polar",
            "skip_subsample_scale_dependent": True,
            "shuffle_mode": None,
            # -- Loading Parameters --
            "scattering_implementation": "monolithic",
            "warmup": False,
            "activation": "sigmoid",
}
model = PhaseAwareUNetFactory(variant_id="phase_aware_u_net_standard", **LOCAL_MODEL_CONFIG)
model.to(device)
print(model)
from phase_aware.core.data.isic import ISICFactory
ISIC_CONFIG = {"root":"Datasets/Data/RAW"}
isic_dataset = ISICFactory("isic_segmentation_standard", **ISIC_CONFIG)
print(isic_dataset)
from torch.utils.data import DataLoader
isic_dataloader = DataLoader(isic_dataset.build_split("train"), batch_size=16, shuffle=True)
from phase_aware.core.losses.dice_ce_phase_skip import DiceCEPhaseSkipFactory
DICE_CE_PHASE_CONFIG = {"lambda_ce":0.5, "lambda_dice":0.5,"lambda_skip":0.01,"lambda_phase_align":0.4,"lambda_phase_tv":0.0}
dice_ce_loss = DiceCEPhaseSkipFactory("dice_ce_phase_skip_standard", **DICE_CE_PHASE_CONFIG)
print(dice_ce_loss)
from phase_aware.core.metrics.segmentation_metrics import SegmentationMetricsFactory
SEG_MET_CONFIG = {
    "seg_task": "binary",
    "seg_num_classes": 2,
    "seg_metrics": "dice,iou,hausdorff",
    "seg_averages": "micro,macro",
    "seg_threshold": 0.5
}
segmentation_metric = SegmentationMetricsFactory("segmentation_metrics_standard", **SEG_MET_CONFIG)
segmentation_metric.to(device)
print(segmentation_metric)
from phase_aware.core.optimizers.optimizer_adam import AdamFactory
ADAM_CONFIG = {"params": model.parameters(), "lr":1e-3}
adam_optimizer = AdamFactory("adam_standard", **ADAM_CONFIG)
print(adam_optimizer)

loss_list =[]
metric_list = []
num_epochs = 30
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    epoch_loss = 0.0
    epoch_metrics = 0.0
    
    # --- Batch Loop ---
    for batch_idx, (inputs, targets) in enumerate(isic_dataloader):
        print_gpu_memory()
        # A. Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape)
        inputs = tensor_to_grayscale_to_tensor(inputs)
        # print(inputs.shape)
        
        # B. Zero Gradients (Clear previous step)
        adam_optimizer.zero_grad()
        
        # C. Forward Pass
        outputs = model(inputs)
        # print(outputs.keys())
        
        # D. Compute Loss
        loss = dice_ce_loss(outputs, targets)
        # segmentation_metric
        # E. Backward Pass & Optimize
        loss.backward()
        adam_optimizer.step()
        
        # Accumulate metrics
        epoch_loss += loss.item()
        loss_list.append(loss.item())

        # Metric
        segmentation_metric.update({"preds":outputs["pred"]}, targets)
        
        # Optional: Print every N batches
        print(f"  Batch {batch_idx+1}/{len(isic_dataloader)} - Loss: {loss.item():.4f}")
        # break
    
    # --- NEW: End of Epoch Report ---
    # 1. Compute global average for the epoch
    epoch_results = segmentation_metric.compute()
    metric_list.append(epoch_results)
    
    # --- End of Epoch Report ---
    avg_loss = epoch_loss / len(isic_dataloader)
    # res_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in epoch_results.items()])
    print(f"Epoch [{epoch+1}/{num_epochs}] Completed. Average Loss: {avg_loss:.6f} - Metric  {pformat(epoch_results)}")
    # break
    
print(f"LOSSES = {pformat(loss_list)}")
print(f"METRICS = {pformat(metric_list)}")
    
isic_dataloader = DataLoader(isic_dataset.build_split("test"), batch_size=16, shuffle=False)
model.eval()          # Disable dropout/batchnorm training behavior
segmentation_metric.reset()       # Clear old metric data

# 2. The Evaluation Loop
with torch.no_grad(): # Disable gradient calculation (saves memory/speed)
    for inputs, targets in isic_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = tensor_to_grayscale_to_tensor(inputs)
        # Forward pass only
        outputs = model(inputs)
        
        # Update metrics (accumulate stats)
        segmentation_metric.update({"preds":outputs["pred"]}, targets)
        # break

# 3. Compute & Print Final Results
val_results = segmentation_metric.compute()

# Format dictionary for clean printing
# res_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in val_results.items()])
print(f"Validation Complete. Results: {pformat(val_results)}")
