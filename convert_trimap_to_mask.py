from PIL import Image
import os
import numpy as np

trimap_dir = "oxford-iiit-pet/annotations/trimaps"
output_dir = "masks/ground_truth_binary"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(trimap_dir):
    if not fname.endswith(".png"):
        continue
    trimap = np.array(Image.open(os.path.join(trimap_dir, fname)))
    binary = np.uint8((trimap == 2) | (trimap == 3)) * 255
    Image.fromarray(binary).save(os.path.join(output_dir, fname))
    print("Saved GT:", fname)
