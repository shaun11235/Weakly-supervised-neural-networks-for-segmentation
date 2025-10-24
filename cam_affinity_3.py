# We used generative AI in an assistive role for this assessment. 
# Specifically, we used ChatGPT (GPT-4), developed by OpenAI (https://chat.openai.com/), to check the comments in our Python files, 
# as well as to identify grammar and spelling issues in the instruction file. Additionally, 
# we used it to review potential errors in our Python code to reduce bugs and improve overall robustness.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from cam_affinity_2 import AffinityNet
from scipy import ndimage

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "trained_model")
os.makedirs(model_dir, exist_ok=True)

image_dir = r"oxford-iiit-pet/images"
binary_mask_dir = "plots/binary_mask_cam"
output_dir = "plots/refined_masks_cam"
model_path = "trained_model/affinitynet_cam.pth"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AffinityNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

NEIGHBOR_SHIFTS = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
def shift(x, dx, dy):
    h, w = x.shape
    shifted = np.zeros_like(x)

    if dy >= 0:
        y1, y2 = 0, h - dy
        y1s, y2s = dy, h
    else:
        y1, y2 = -dy, h
        y1s, y2s = 0, h + dy

    if dx >= 0:
        x1, x2 = 0, w - dx
        x1s, x2s = dx, w
    else:
        x1, x2 = -dx, w
        x1s, x2s = 0, w + dx

    shifted[y1s:y2s, x1s:x2s] = x[y1:y2, x1:x2]
    return shifted


def propagate(seed, affinity, steps=6):
    H, W = seed.shape[1:]
    num_classes = seed.shape[0]
    seed = seed.copy()
    for _ in range(steps):
        new_seed = np.zeros_like(seed)
        total_weight = np.zeros_like(seed)

        for k, (dy, dx) in enumerate(NEIGHBOR_SHIFTS):
            aff = affinity[k]
            for c in range(num_classes):
                shifted = shift(seed[c], dx, dy) * aff
                weight = shift(np.ones_like(seed[c]), dx, dy) * aff
                new_seed[c] += shifted
                total_weight[c] += weight

        seed = new_seed / (total_weight + 1e-6)
    return seed


image_names = sorted(os.listdir(image_dir))

for name in image_names:
    if not name.endswith(".jpg"):
        continue

    img_path = os.path.join(image_dir, name)
    mask_path = os.path.join(binary_mask_dir, name.replace(".jpg", "_mask.png"))
    save_path = os.path.join(output_dir, name.replace(".jpg", "_refined.png"))

    if not os.path.exists(mask_path):
        continue

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    mask_arr = np.array(Image.open(mask_path).resize((224, 224))) / 255.0
    mask_arr = np.array(Image.open(mask_path).convert("L").resize((224, 224)), dtype=np.float32) / 255.0

    fg_seed = ndimage.grey_dilation(mask_arr, size=(5, 5))
    fg_seed = np.clip(fg_seed, 0, 1).astype(np.float32)
    bg_seed = 1.0 - fg_seed

    seed_stack = np.stack([bg_seed, fg_seed], axis=0)

    with torch.no_grad():
        affinity = model(img_tensor)
        affinity_np = affinity.squeeze(0).cpu().numpy()

    refined = propagate(seed_stack, affinity_np)

    fg_prob = refined[1]
    fg_prob -= fg_prob.min()
    fg_prob /= (fg_prob.max() + 1e-6)

    threshold = 0.4
    binary = (fg_prob > threshold).astype(np.uint8)

    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
    binary = ndimage.binary_opening(binary, structure=np.ones((3, 3))).astype(np.uint8)
    binary = ndimage.binary_closing(binary, structure=np.ones((5, 5))).astype(np.uint8)

    label_im, nb_labels = ndimage.label(binary)
    sizes = ndimage.sum(binary, label_im, range(nb_labels + 1))
    mask = np.zeros_like(binary)
    for i in range(1, nb_labels + 1):
        if sizes[i] >= 400:
            mask[label_im == i] = 1

    binary = mask.astype(np.uint8)
    Image.fromarray(binary * 255).save(save_path)
    print(f"Saved refined mask: {save_path}")