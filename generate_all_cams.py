# We used generative AI in an assistive role for this assessment. 
# Specifically, we used ChatGPT (GPT-4), developed by OpenAI (https://chat.openai.com/), to check the comments in our Python files, 
# as well as to identify grammar and spelling issues in the instruction file. Additionally, 
# we used it to review potential errors in our Python code to reduce bugs and improve overall robustness.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.cm as cm
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import random

# ================= Utility Functions =================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)   # Ensure reproducibility

# Custom Dataset for Oxford-IIIT Pet
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

# Hook to capture intermediate feature maps
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.detach().cpu().numpy())  # Get conv feature maps

# Compute CAM from feature map and softmax weights
def returnCAM(feature_conv, weight_softmax, class_idx, img_size):
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    cam_img = np.uint8(255 * cam)
    return Image.fromarray(cam_img).resize(img_size, resample=Image.BILINEAR), cam

# ================= Main Program =================

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(current_dir, "oxford-iiit-pet", "images")
    list_txt = os.path.join(current_dir, "oxford-iiit-pet", "annotations", "list.txt")
    model_dir = os.path.join(current_dir, "trained_model")
    output_root = os.path.join(current_dir, "plots", "cam_all_models")
    os.makedirs(output_root, exist_ok=True)

    # Load labels from list.txt
    with open(list_txt, "r") as f:
        lines = f.readlines()[6:]

    img_labels = {}
    for line in lines:
        parts = line.strip().split()
        img_name = parts[0] + ".jpg"
        class_id = int(parts[1]) - 1
        img_labels[img_name] = class_id

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = PetDataset(image_dir, img_labels, transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load 20 trained model weights
    model_paths = [os.path.join(model_dir, f"resnet_pet_epoch{i+1}.pth") for i in range(20)]

    # Loop through each model
    for idx, model_path in enumerate(model_paths):
        print(f"Generating CAMs using model: {os.path.basename(model_path)}")
        output_folder = os.path.join(output_root, f"CAM_model_{idx+1}")
        os.makedirs(output_folder, exist_ok=True)

        # Load model
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 37)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        # Reset and register hook to capture feature maps
        global features_blobs
        features_blobs = []
        model._modules.get('layer4').register_forward_hook(hook_feature)

        for i in range(len(dataset)):
            features_blobs.clear()
            img_tensor, label = dataset[i]
            img_name = dataset.image_names[i]
            img_pil = Image.open(os.path.join(image_dir, img_name)).convert("RGB")

            input_tensor = img_tensor.unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).data.squeeze()
            class_idx = torch.argmax(probs).item()

            # Get softmax weights
            params = list(model.parameters())
            weight_softmax = params[-2].detach().cpu().numpy()
            weight_softmax = np.squeeze(weight_softmax)

            # Generate CAM + colorize + blend
            cam_img_pil, cam_array = returnCAM(features_blobs[0], weight_softmax, class_idx, img_pil.size)
            jet = cm.get_cmap("jet")
            cam_color = jet(cam_array)[:, :, :3]
            cam_color = np.uint8(255 * cam_color)
            cam_color_img = Image.fromarray(cam_color).resize(img_pil.size)
            blended = Image.blend(img_pil, cam_color_img, alpha=0.5)

            # Save result
            save_path = os.path.join(output_folder, f"CAM_{img_name}")
            blended.save(save_path)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
