import os
import numpy as np
from PIL import Image

def compute_iou(pred_mask, gt_mask):
    # 使用动态阈值：前景是 > 最大值的20%
    pred_bin = pred_mask > (0.5 * pred_mask.max())
    gt_bin = gt_mask > 128  # Ground Truth 通常是二值图：255 vs 0

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return intersection / union

def compute_miou(pred_dir, gt_dir):
    ious = []
    for fname in os.listdir(pred_dir):
        if not fname.endswith(".png"):
            continue
        pred = Image.open(os.path.join(pred_dir, fname)).convert("L")
        gt = Image.open(os.path.join(gt_dir, fname)).convert("L")

        pred_mask = np.array(pred.resize(gt.size))
        gt_mask = np.array(gt)

        iou = compute_iou(pred_mask, gt_mask)
        ious.append(iou)

    miou = np.mean(ious)
    print(f"{miou * 100:.2f}%")  # 以百分比形式输出
    return miou

if __name__ == "__main__":
    pred_folder = "plots/cam_heatmaps"
    gt_folder = "masks/ground_truth_binary"
    compute_miou(pred_folder, gt_folder)
