import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from torchvision.models import resnet18

# ---------- 数据集类 ----------
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
        return image, label, img_name

# ---------- Hook 和 CAM ----------
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.detach().cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx, img_size):
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam_img = np.uint8(255 * cam)
    return Image.fromarray(cam_img).resize(img_size, resample=Image.BILINEAR), cam

# ---------- 主函数 ----------
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(current_dir, "oxford-iiit-pet", "images")
    list_txt = os.path.join(current_dir, "oxford-iiit-pet", "annotations", "list.txt")
    model_path = os.path.join(current_dir, "trained_model", "resnet_pet.pth")
    output_folder = os.path.join(current_dir, "plots", "cam_heatmaps")
    os.makedirs(output_folder, exist_ok=True)

    # 标签映射
    with open(list_txt, "r") as f:
        lines = f.readlines()[6:]
    img_labels = {line.strip().split()[0] + ".jpg": int(line.strip().split()[1]) - 1 for line in lines}

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = PetDataset(image_dir, img_labels, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型构建（不联网下载预训练权重）
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 37)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 注册 hook
    finalconv_name = 'layer4'
    model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # 提取 softmax 权重
    params = list(model.parameters())
    weight_softmax = params[-2].detach().cpu().numpy()
    weight_softmax = np.squeeze(weight_softmax)

    # 只处理前5张图像
    for i in range(len(dataset)):
        features_blobs.clear()
        img_tensor, label, img_name = dataset[i]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).data.squeeze()
        class_idx = torch.argmax(probs).item()

        img_path = os.path.join(image_dir, img_name)
        img_pil = Image.open(img_path).convert("RGB")

        cam_img_pil, cam_array = returnCAM(features_blobs[0], weight_softmax, class_idx, img_pil.size)
        jet = cm.get_cmap("jet")
        cam_color = jet(cam_array)[:, :, :3]
        cam_color = np.uint8(255 * cam_color)
        cam_color_img = Image.fromarray(cam_color).resize(img_pil.size)
        blended = Image.blend(img_pil, cam_color_img, alpha=0.5)

        save_name = img_name.replace(".jpg", ".png")
        save_path = os.path.join(output_folder, save_name)
        blended.save(save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
