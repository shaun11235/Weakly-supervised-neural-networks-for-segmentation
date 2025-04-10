import os
import numpy as np
from PIL import Image


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

def list_valid_png_files(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.endswith(".png") and not f.startswith("._")
    ])


def compute_miou_for_folder(pred_dir, label_dir, num_classes=2):
    pred_files = list_valid_png_files(pred_dir)
    label_files = list_valid_png_files(label_dir)
    all_ious = []
    for pred_file in pred_files:
        pred_path = os.path.join(pred_dir, pred_file)
        label_name = pred_file.replace('_refined', '')
        #label_name = pred_file.replace('_mask', '')
        label_path = os.path.join(label_dir, label_name)
        pred_mask = np.array(Image.open(pred_path).convert('L'))
        pred_mask = (pred_mask > 127).astype(np.uint8)
        label_img = Image.open(label_path).convert('L')
        label_img = label_img.resize(pred_mask.shape[::-1], Image.NEAREST)
        label_mask = np.array(label_img)
        label_mask[label_mask == 3] = 255
        label_mask[label_mask == 2] = 0
        label_mask[label_mask == 1] = 1
        ious = compute_iou_per_class(pred_mask, label_mask, num_classes)
        all_ious.append(ious)

    all_ious = np.array(all_ious)
    mean_ious = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(mean_ious)
    return miou, mean_ious



#path
pred_dir = r"results/refined_masks"
#pred_dir = '../binary_cams'
label_dir = r'dataset/annotations/trimaps'
miou, ious = compute_miou_for_folder(pred_dir, label_dir)
print(f"MIoU: {miou:.4f}")
for i, iou in enumerate(ious):
    print(f"Class {i} IoU: {iou:.4f}")
