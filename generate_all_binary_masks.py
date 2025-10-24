# We used generative AI in an assistive role for this assessment. 
# Specifically, we used ChatGPT (GPT-4), developed by OpenAI (https://chat.openai.com/), to check the comments in our Python files, 
# as well as to identify grammar and spelling issues in the instruction file. Additionally, 
# we used it to review potential errors in our Python code to reduce bugs and improve overall robustness.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
import numpy as np
from scipy import ndimage, signal
from skimage import color

# ========= Path Configuration =========
base_cam_root = r'plots/cam_all_models'  # Folder containing 20 sets of CAM heatmaps
image_dir = r'oxford-iiit-pet/images'
output_base = r'results/binary_mask_all'
os.makedirs(output_base, exist_ok=True)
thresholds = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]  # Thresholds from 0.1 to 0.9

# ========= Main Loop =========
for model_idx in range(1, 21):
    cam_dir = os.path.join(base_cam_root, f'CAM_model_{model_idx}')
    for threshold in thresholds:
        output_dir = os.path.join(output_base, f'CAM_model_{model_idx}', f'th_{threshold:.1f}')
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(cam_dir):
            if not filename.endswith('.jpg') and not filename.endswith('.png'):
                continue

            base_name = os.path.splitext(filename)[0].replace('CAM_', '').replace('_cam', '')
            cam_path = os.path.join(cam_dir, filename)
            img_path = os.path.join(image_dir, base_name + '.jpg')
            save_path = os.path.join(output_dir, base_name + '_mask.png')

            if not os.path.exists(img_path):
                continue

            # === Load image and CAM
            cam = Image.open(cam_path).convert('L').resize((224, 224))
            cam_np = np.array(cam).astype(np.float32) / 255.0
            img_np = np.array(Image.open(img_path).convert('RGB').resize((224, 224)))

            # === Threshold CAM to binary mask
            mask = (cam_np > threshold).astype(np.uint8)
            mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

            label_im, nb_labels = ndimage.label(mask)
            sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
            if nb_labels > 0:
                largest = (sizes[1:]).argmax() + 1
                mask = (label_im == largest).astype(np.uint8)

            # === Enhance with image edge gradients
            gray = color.rgb2gray(img_np / 255.0)
            grad_x = signal.convolve2d(gray, [[-1, 1]], mode='same', boundary='symm')
            grad_y = signal.convolve2d(gray, [[-1], [1]], mode='same', boundary='symm')
            edge = np.sqrt(grad_x**2 + grad_y**2)
            edge = (edge > 0.1).astype(np.uint8)

            combined = mask.astype(np.float32) + 0.6 * edge
            combined = (combined > 0.5).astype(np.uint8)
            combined = ndimage.binary_closing(combined, structure=np.ones((3, 3))).astype(np.uint8)
            combined = ndimage.binary_fill_holes(combined).astype(np.uint8)

            binary_mask = combined * 255
            Image.fromarray(binary_mask).save(save_path)
            print(f"[Model {model_idx}][Threshold {threshold:.1f}] Saved: {save_path}")
