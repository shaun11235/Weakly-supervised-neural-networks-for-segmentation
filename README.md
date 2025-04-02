import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ========== CONFIG ==========
image_dir = r"C:\Users\ROG\OneDrive - University College London\COMP0197\ICA2\oxford-iiit-pet\images\images"  # <- 修改为你本地 images 解压路径
list_txt = r"C:\Users\ROG\OneDrive - University College London\COMP0197\ICA2\oxford-iiit-pet\annotations\annotations\list.txt" # <- 修改为你的 list.txt 路径
num_classes = 37
batch_size = 16
epochs = 1
lr = 1e-3
model_save_path = "resnet_pet.pth"
image_path = "test.jpg"
output_cam_path = "CAM.jpg"

# ========== STEP 1: 构造标签映射表 ==========
with open(list_txt, "r") as f:
    lines = f.readlines()[6:]  # 前6行是注释


img_labels = {}
for line in lines:
    parts = line.strip().split()
    img_name = parts[0] + ".jpg"
    class_id = int(parts[1]) - 1
    img_labels[img_name] = class_id

# ========== STEP 2: 自定义 Dataset ==========
class PetDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, label_dict, transform=None):
        self.image_folder = image_folder
        self.label_dict = label_dict
        self.image_names = list(label_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = self.label_dict[img_name]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ========== STEP 3: 数据增强与加载 ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = PetDataset(image_dir, img_labels, transform)
# dataset = Subset(dataset, list(range(50)))  # 仅使用前50张图像测试

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ========== STEP 4: 定义模型 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ========== STEP 5: 训练模型 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), model_save_path)
print("Model saved to", model_save_path)

# ========== STEP 6: 加载模型并生成 CAM ==========
model.eval()
model.load_state_dict(torch.load(model_save_path, map_location="cpu"))

finalconv_name = 'layer4'
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.detach().numpy())

model._modules.get(finalconv_name).register_forward_hook(hook_feature)

img_pil = Image.open(image_path).convert('RGB')
img_tensor = transform(img_pil).unsqueeze(0)
logits = model(img_tensor)
probs = F.softmax(logits, dim=1).data.squeeze()
class_idx = torch.argmax(probs).item()
print(f"Predicted class index: {class_idx}, probability: {probs[class_idx]:.4f}")

params = list(model.parameters())
weight_softmax = params[-2].detach().numpy()
weight_softmax = np.squeeze(weight_softmax)

# ========== 生成 CAM ==========
def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam_img = np.uint8(255 * cam)
    return Image.fromarray(cam_img).resize(img_pil.size, resample=Image.BILINEAR), cam

cam_img_pil, cam_array = returnCAM(features_blobs[0], weight_softmax, class_idx)

jet = cm.get_cmap("jet")
cam_color = jet(cam_array)[:, :, :3]
cam_color = np.uint8(255 * cam_color)
cam_color_img = Image.fromarray(cam_color).resize(img_pil.size)
blended = Image.blend(img_pil, cam_color_img, alpha=0.5)
blended.save(output_cam_path)
print(f"Saved CAM heatmap to {output_cam_path}")



=========================================================直接调用模型=============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ========== 配置 ==========
num_classes = 37
model_save_path = "resnet_pet.pth"
image_path = "test.jpg"              # 测试图像（要放在同一目录下或指定完整路径）
output_cam_path = "CAM.jpg"          # 输出的热力图路径

# ========== 图像预处理 transform（与训练保持一致）==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== 定义模型结构并加载参数 ==========
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
model.eval()

finalconv_name = 'layer4'
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.detach().numpy())

model._modules.get(finalconv_name).register_forward_hook(hook_feature)

img_pil = Image.open(image_path).convert('RGB')
img_tensor = transform(img_pil).unsqueeze(0)
logits = model(img_tensor)
probs = F.softmax(logits, dim=1).data.squeeze()
class_idx = torch.argmax(probs).item()
print(f"Predicted class index: {class_idx}, probability: {probs[class_idx]:.4f}")

params = list(model.parameters())
weight_softmax = params[-2].detach().numpy()
weight_softmax = np.squeeze(weight_softmax)

# ========== 生成 CAM ==========
def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam_img = np.uint8(255 * cam)
    return Image.fromarray(cam_img).resize(img_pil.size, resample=Image.BILINEAR), cam

cam_img_pil, cam_array = returnCAM(features_blobs[0], weight_softmax, class_idx)

jet = cm.get_cmap("jet")
cam_color = jet(cam_array)[:, :, :3]
cam_color = np.uint8(255 * cam_color)
cam_color_img = Image.fromarray(cam_color).resize(img_pil.size)
blended = Image.blend(img_pil, cam_color_img, alpha=0.5)
blended.save(output_cam_path)
print(f"Saved CAM heatmap to {output_cam_path}")
