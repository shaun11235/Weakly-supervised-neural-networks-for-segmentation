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
