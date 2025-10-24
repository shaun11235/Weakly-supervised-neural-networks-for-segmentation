# We used generative AI in an assistive role for this assessment. 
# Specifically, we used ChatGPT (GPT-4), developed by OpenAI (https://chat.openai.com/), to check the comments in our Python files, 
# as well as to identify grammar and spelling issues in the instruction file. Additionally, 
# we used it to review potential errors in our Python code to reduce bugs and improve overall robustness.

import os
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import xml.etree.ElementTree as ET

script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(script_dir, "oxford-iiit-pet", "images")
xml_dir = os.path.join(script_dir, "oxford-iiit-pet", "annotations", "xmls")
model_dir = os.path.join(script_dir, "trained_model")
os.makedirs(model_dir, exist_ok=True)
bbox_txt = os.path.join(script_dir, "trained_model", "trimaps_bbox.txt")
output_dir = os.path.join(script_dir, "plots", "bbox_heatmaps")
os.makedirs(output_dir, exist_ok=True)

bbox_lines = []

for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text

    # Extract bbox coordinates from XML
    bndbox = root.find(".//bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    line = f"{filename} {xmin} {ymin} {xmax} {ymax}"
    bbox_lines.append(line)

# Write all bounding boxes to file
with open(bbox_txt, "w") as f:
    for line in bbox_lines:
        f.write(line + "\n")

print(f" Saved {len(bbox_lines)} bounding boxes to {bbox_txt}")

# ========== STEP 2: Generate pseudo CAM heatmaps from bbox ==========
# Load bbox file into a dictionary
bbox_dict = {}
with open(bbox_txt, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 5:
            filename = parts[0]
            bbox = list(map(int, parts[1:]))
            bbox_dict[filename] = bbox

# Define heatmap generator
def generate_heatmap(image_size, bbox, mode="gaussian"):
    w, h = image_size
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2

    y_grid, x_grid = np.ogrid[0:h, 0:w]

    if mode == "gaussian":
        sigma = ((xmax - xmin) + (ymax - ymin)) / 3.5
        heatmap = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
    else:
        max_dist = np.sqrt(((xmax - xmin) / 2)**2 + ((ymax - ymin) / 2)**2)
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        heatmap = np.clip(1 - dist / max_dist, 0, 1)

    heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else np.zeros_like(heatmap)
    return heatmap

for i, (filename, bbox) in enumerate(bbox_dict.items()):

    image_path = os.path.join(image_dir, filename)
    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    heatmap = generate_heatmap((w, h), bbox, mode="gaussian")

    cmap = cm.get_cmap("jet")
    heatmap_color = cmap(heatmap)[:, :, :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_color).resize((w, h))

    blended = Image.blend(image, heatmap_img, alpha=0.5)
    base = os.path.splitext(filename)[0]
    blended.save(os.path.join(output_dir, f"{base}_blend.jpg"))

print(" All pseudo heatmaps have been generated!")

