import os
import numpy as np
from PIL import Image
import matplotlib.cm as cm

# ========== CONFIG ==========
image_dir = r"C:\Users\ROG\OneDrive - University College London\COMP0197\ICA2\oxford-iiit-pet\images\images"  # <- 修改为你本地 images 解压路径
bbox_txt =  r"C:\Users\ROG\OneDrive - University College London\COMP0197\ICA2\oxford-iiit-pet\annotations\annotations\trimaps_bbox.txt"  # 你需要构建一个 bbox 信息的 txt 文件
output_dir = r"C:\Users\ROG\OneDrive - University College London\COMP0197\ICA2\oxford-iiit-pet\heatmaps"
os.makedirs(output_dir, exist_ok=True)

# ========== FUNCTION: 从 bbox 生成改进版热图 ==========
def generate_heatmap(image_size, bbox, mode="gaussian"):
    w, h = image_size
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2

    y_grid, x_grid = np.ogrid[0:h, 0:w]

    if mode == "gaussian":
        # 扩散半径放大，更广泛地覆盖图像
        sigma = ((xmax - xmin) + (ymax - ymin)) / 3.5   #可以调整参数到想要的效果
        heatmap = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
    else:
        max_dist = np.sqrt(((xmax - xmin) / 2)**2 + ((ymax - ymin) / 2)**2)
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        heatmap = np.clip(1 - dist / max_dist, 0, 1)

    heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else np.zeros_like(heatmap)
    return heatmap

# ========== MAIN ==========
# 模拟读取 bbox 文件：每行格式为 "image.jpg xmin ymin xmax ymax"
bbox_dict = {}
with open(bbox_txt, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 5:
            filename = parts[0]
            bbox = list(map(int, parts[1:]))
            bbox_dict[filename] = bbox
limit = 10
for filename, bbox in bbox_dict.items():
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
    # Image.fromarray((heatmap * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{base}_gray.png"))

print("✅ 所有伪CAM热图已生成完毕！")
