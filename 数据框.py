import os
import xml.etree.ElementTree as ET

# ========== CONFIG ==========
xml_dir = r"C:\Users\ROG\OneDrive - University College London\COMP0197\ICA2\oxford-iiit-pet\annotations\annotations\xmls"
output_txt = r"C:\Users\ROG\OneDrive - University College London\COMP0197\ICA2\oxford-iiit-pet\annotations\annotations\trimaps_bbox.txt" 

bbox_lines = []

# 遍历所有 xml 文件
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text

    bndbox = root.find(".//bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    line = f"{filename} {xmin} {ymin} {xmax} {ymax}"
    bbox_lines.append(line)

# 保存为 bbox.txt
with open(output_txt, "w") as f:
    for line in bbox_lines:
        f.write(line + "\n")

print(f"✅ 成功写入 {output_txt}，共 {len(bbox_lines)} 条 bbox")
