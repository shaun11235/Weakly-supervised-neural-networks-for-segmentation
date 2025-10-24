# We used generative AI in an assistive role for this assessment. 
# Specifically, we used ChatGPT (GPT-4), developed by OpenAI (https://chat.openai.com/), to check the comments in our Python files, 
# as well as to identify grammar and spelling issues in the instruction file. Additionally, 
# we used it to review potential errors in our Python code to reduce bugs and improve overall robustness.

import os
import csv
import numpy as np
from PIL import Image

# Compute IoU for each class
def compute_iou_per_class(pred, label, num_classes):
    mask = (label != 255)
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls) & mask
        label_inds = (label == cls) & mask
        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return ious

# List all valid .png files
def list_valid_png_files(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.endswith(".png") and not f.startswith("._")
    ])

# Compute mIoU and per-class IoUs for a folder
def compute_miou_for_folder(pred_dir, label_dir, num_classes=2):
    pred_files = list_valid_png_files(pred_dir)
    all_ious = []
    for pred_file in pred_files:
        pred_path = os.path.join(pred_dir, pred_file)
        label_name = pred_file.replace('_mask', '')
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            continue

        pred_mask = np.array(Image.open(pred_path).convert('L'))
        pred_mask = (pred_mask > 127).astype(np.uint8)

        label_img = Image.open(label_path).convert('L')
        label_img = label_img.resize(pred_mask.shape[::-1], Image.NEAREST)
        label_mask = np.array(label_img)
        label_mask[label_mask == 3] = 255  # ignore boundary
        label_mask[label_mask == 2] = 0    # background
        label_mask[label_mask == 1] = 1    # foreground

        ious = compute_iou_per_class(pred_mask, label_mask, num_classes)
        all_ious.append(ious)

    all_ious = np.array(all_ious)
    mean_ious = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(mean_ious)
    return miou, mean_ious

# ======== Configuration ========
mask_root = r'plots/binary_mask_all'
label_dir = r'oxford-iiit-pet/annotations/trimaps'
output_csv = 'miou_results_all.csv'
thresholds = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]

# ======== Run evaluation and save to CSV ========
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Threshold", "mIoU", "Class 0 IoU", "Class 1 IoU"])

    for model_idx in range(1, 21):
        for threshold in thresholds:
            pred_dir = os.path.join(mask_root, f'CAM_model_{model_idx}', f'th_{threshold:.1f}')
            if not os.path.exists(pred_dir):
                print(f"⚠️ Skipping missing folder: {pred_dir}")
                continue
            miou, ious = compute_miou_for_folder(pred_dir, label_dir)
            print(f"[Model {model_idx}][Threshold {threshold:.1f}] mIoU: {miou:.4f} | Class 0: {ious[0]:.4f}, Class 1: {ious[1]:.4f}")
            writer.writerow([model_idx, threshold, f"{miou:.4f}", f"{ious[0]:.4f}", f"{ious[1]:.4f}"])
