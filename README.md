# Weakly Supervised Semantic Segmentation on Oxford-IIIT Pet Dataset

---

## 📘 1. Overview
This project explores **semantic segmentation** on the **Oxford-IIIT Pet dataset** using **weakly supervised approaches**.  
We use **image-level labels** as limited supervision signals to generate heatmaps and convert them into pseudo segmentation masks for downstream training.  
We also train a **fully supervised baseline model** for comparison and experiment with **bounding box–based weak supervision** to generate alternative pseudo labels.

---

## ⚙️ 2. Setup

### Requirements
- Python 3.x  
- PyTorch  
- torchvision  
- numpy  
- PIL (Pillow)  
- matplotlib  
- skimage  
- tarfile  
- urllib.request  
- os  
- random  
- xml.etree.ElementTree  
- scipy  

### Extra packages
- matplotlib  
- skimage  
- scipy  

---

## 📂 3. Project Files Description

### `download_pet_dataset.py`
Automates downloading and extracting the Oxford-IIIT Pet dataset.  
Creates `oxford-iiit-pet/` directory, downloads image and annotation archives, and extracts them locally.

### `cam_training_model.py`
Implements a **ResNet-18** classifier training pipeline.  
- Loads image-level labels and applies preprocessing.  
- Trains for **20 epochs** with CrossEntropyLoss and Adam optimizer.  
- Saves weights in `trained_model/epoch/`.

### `cam_heatmap.py`
Generates **Class Activation Maps (CAMs)** for all images:  
- Loads the best trained ResNet-18 model.  
- Extracts feature maps via a forward hook.  
- Generates CAMs and saves them in `cam_heatmaps/`.

### `bbox_heatmap.py`
Generates **bounding box–based heatmaps** in two stages:  
1. Parses XML annotations to extract bounding boxes → saves as `trimaps_bbox.txt`.  
2. Applies Gaussian-based heatmaps to visualize and save in `bbox_heatmaps/`.

### `generate_all_cams.py`
Automates CAM generation for 20 ResNet models.  
Saves outputs in `plots/cam_all_models/`.

### `generate_all_binary.py`
Converts CAM heatmaps into **binary masks** with thresholds (0.1–0.9).  
Stores results in `results/binary_mask_all/`.

### `evaluate_all_model.py`
Evaluates **mIoU (mean Intersection over Union)** for all binary masks.  
Outputs results to `miou_results_all.csv`.

### `cam_affinity_1.py` / `bbox_affinity_1.py`
Converts CAM or bounding box heatmaps into binary pseudo masks for AffinityNet training.  
Outputs stored in:
- `plots/binary_mask_cam/`  
- `plots/binary_mask_bbox/`

### `cam_affinity_2.py` / `bbox_affinity_2.py`
Trains **AffinityNet** to propagate pseudo labels.  
Saves models as:
- `trained_model/affinitynet_cam.pth`  
- `trained_model/affinitynet_bbox.pth`

### `cam_affinity_3.py` / `bbox_affinity_3.py`
Generates **refined masks** using trained AffinityNet.  
Outputs stored in:
- `plots/refined_masks_cam/`  
- `plots/refined_masks_bbox/`

### `cam_affinity_4.py` / `bbox_affinity_4.py`
Evaluates segmentation quality (mIoU results displayed in terminal).

### `unet.py`
Defines the **SimpleUNet** model, **FocalLoss**, and `compute_miou()` function.  
Includes full training and evaluation pipeline.

### `train_unet.py`
Implements U-Net training:  
- Custom dataset and preprocessing.  
- Focal loss + Adam optimizer.  
- Periodic evaluation with mIoU.  
- Saves `unet_model.pth`.

### `test_unet.py`
Evaluates pretrained U-Net model:  
- Loads model and test data.  
- Computes overall mIoU.  
- Visualizes predicted vs ground truth masks.  
- Saves comparison figures.

---

## 🧩 4. Model Descriptions

### **Model 1: CAM + AffinityNet**
1. Generate heatmaps using CAM.  
2. Convert heatmaps to binary masks.  
3. Refine masks via AffinityNet propagation.  
4. Evaluate pseudo labels using mIoU.

---

### **Model 0: Hyperparameter Selection**
1. Generate 20 CAM models (1–20 epochs).  
2. Convert to binary masks (thresholds 0.1–0.9).  
3. Compute 180 mIoU results and save `.csv` summary.

---

### **Model 2: Fully Supervised U-Net**
- Train U-Net using Oxford-IIIT Pet ground truth masks.  
- Focal loss + Adam optimizer for class balancing.  
- Evaluate with mIoU and save model as `unet_model.pth`.

---

### **Model 3: Bounding Box + AffinityNet**
1. Generate heatmaps using bounding boxes.  
2. Convert to binary masks.  
3. Refine using AffinityNet.  
4. Evaluate mIoU.

---

## 🗂️ 5. Directory Structure

```bash
project_root/
├── download_pet_dataset.py
├── cam_training_model.py
├── bbox_heatmap.py
├── cam_heatmap.py
├── generate_all_binary.py
├── generate_all_cams.py
├── evaluate_all_model.py
├── cam_affinity_1/
├── cam_affinity_2/
├── cam_affinity_3/
├── cam_affinity_4/
├── bbox_affinity_1/
├── bbox_affinity_2/
├── bbox_affinity_3/
├── bbox_affinity_4/
├── unet.py
├── train_unet.py
├── test_unet.py
│
├── results/
│   └── binary_mask_all/
│
├── trained_model/
│   ├── affinitynet_bbox.pth
│   ├── affinitynet_cam.pth
│   ├── resnet_pet_epoch1–20.pth
│   ├── trimaps_bbox.txt
│   └── unet_model.pth
│
├── plots/
│   ├── bbox_heatmaps/
│   ├── cam_heatmaps/
│   ├── binary_mask_cam/
│   ├── refined_masks_cam/
│   ├── binary_mask_bbox/
│   ├── refined_masks_bbox/
│   ├── cam_all_models/
│   └── UNET.png
│
└── oxford-iiit-pet/
    ├── images/
    └── annotations/
        ├── trimaps/
        └── xmls/
```

## 🚀 6. Simplified Steps to Run

🔹 Option A: Using Pretrained Models


Step 1: Download pretrained models
OneDrive link:
https://1drv.ms/f/c/e7566ddc0b2c213d/EuKFxeWPT9hDisunjivQR54BJrzcJHybY5j5r-4Bpz9rNQ?e=8pYI7o

Place the `trained_model` folder (with all model files)
parallel to the `oxford-iiit-pet` dataset folder.


----------------------------------------------------
⚙️ Model 1: CAM + AffinityNet
----------------------------------------------------
Models required:
resnet_pet_epoch12.pth
affinitynet_cam.pth

Run sequence:
python download_pet_dataset.py
python cam_heatmap.py
python cam_affinity_1.py
python cam_affinity_3.py
python cam_affinity_4.py

Results:
cam_heatmaps/
binary_mask_cam/
refined_masks_cam/
mIoU


----------------------------------------------------
⚙️ Model 0: Hyperparameter Selection
----------------------------------------------------
Checkpoints:
resnet_pet_epoch1.pth → resnet_pet_epoch20.pth

Run sequence:
python download_pet_dataset.py
python generate_all_cams.py
python generate_all_binary.py
python evaluate_all_model.py

Results:
plots/cam_all_models/
results/binary_mask_all/
mIoU


----------------------------------------------------
⚙️ Model 2: Fully Supervised U-Net
----------------------------------------------------
Model required:
unet_model.pth

Run sequence:
python download_pet_dataset.py
python unet.py
python train_unet.py
python test_unet.py

Results:
plots/UNET.png
trained_model/unet_model.pth
mIoU


----------------------------------------------------
⚙️ Model 3: Bounding Box + AffinityNet
----------------------------------------------------
Models required:
resnet_pet_epoch12.pth
affinitynet_bbox.pth

Run sequence:
python download_pet_dataset.py
python bbox_affinity_1.py
python bbox_affinity_3.py
python bbox_affinity_4.py

Results:
bbox_heatmaps/
binary_mask_bbox/
refined_masks_bbox/
mIoU


🔹 Option B: Train from Scratch


----------------------------------------------------
⚙️ Model 1: CAM + AffinityNet
----------------------------------------------------
Run sequence:
python download_pet_dataset.py
python cam_training_model.py
python cam_heatmap.py
python cam_affinity_1.py
python cam_affinity_2.py
python cam_affinity_3.py
python cam_affinity_4.py

Results:
cam_heatmaps/
binary_mask_cam/
refined_masks_cam/
mIoU


----------------------------------------------------
⚙️ Model 0: Hyperparameter Selection (Alternative)
----------------------------------------------------
Run sequence:
python download_pet_dataset.py
python generate_all_cams.py
python generate_all_binary.py
python evaluate_all_model.py

Results:
plots/cam_all_models/
results/binary_mask_all/
mIoU


----------------------------------------------------
⚙️ Model 2: Fully Supervised U-Net
----------------------------------------------------
Run sequence:
python unet.py
python train_unet.py
python test_unet.py

Results:
plots/UNET.png
trained_model/unet_model.pth
mIoU


----------------------------------------------------
⚙️ Model 3: Bounding Box + AffinityNet
----------------------------------------------------
Run sequence:
python bbox_heatmap.py
python bbox_affinity_1.py
python bbox_affinity_2.py
python bbox_affinity_3.py
python bbox_affinity_4.py

Results:
bbox_heatmaps/
binary_mask_bbox/
refined_masks_bbox/
mIoU
