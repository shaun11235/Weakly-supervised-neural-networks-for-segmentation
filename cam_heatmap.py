# We used generative AI in an assistive role for this assessment. 
# Specifically, we used ChatGPT (GPT-4), developed by OpenAI (https://chat.openai.com/), to check the comments in our Python files, 
# as well as to identify grammar and spelling issues in the instruction file. Additionally, 
# we used it to review potential errors in our Python code to reduce bugs and improve overall robustness.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.cm as cm
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

set_seed(42)  


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

# Hook function to extract features from the final convolution layer
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.detach().cpu().numpy())  # ensure on CPU

# Generate the CAM heatmap
def returnCAM(feature_conv, weight_softmax, class_idx, img_size):
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam_img = np.uint8(255 * cam)
    return Image.fromarray(cam_img).resize(img_size, resample=Image.BILINEAR), cam

# ========== Main Code ==========

current_dir = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(current_dir,  "oxford-iiit-pet", "images")
list_txt = os.path.join(current_dir,  "oxford-iiit-pet", "annotations", "list.txt")

# ==== Model and CAM output paths ====
model_dir = os.path.join(current_dir, "trained_model")
os.makedirs(model_dir, exist_ok=True)
model_save_path = os.path.join(model_dir, "resnet_pet_epoch12.pth")

output_folder = os.path.join(current_dir, "plots", "cam_heatmaps")
os.makedirs(output_folder, exist_ok=True)

num_classes = 37
batch_size = 16
epochs = 20
lr = 1e-3

# Step 1: Construct label mapping
with open(list_txt, "r") as f:
    lines = f.readlines()[6:]

img_labels = {}
for line in lines:
    parts = line.strip().split()
    img_name = parts[0] + ".jpg"
    class_id = int(parts[1]) - 1
    img_labels[img_name] = class_id

# Step 2: Data augmentation and loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
dataset = PetDataset(image_dir, img_labels, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 3: Define and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Step 4: Load model and generate CAMs for the whole dataset
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# Register hook to get feature maps from final conv layer
finalconv_name = 'layer4'
model._modules.get(finalconv_name).register_forward_hook(hook_feature)

# Loop through the dataset and generate CAM for each image
for idx in range(len(dataset)):
    features_blobs.clear()  # Clear previous features
    img_tensor, label = dataset[idx]
    img_name = dataset.image_names[idx]
    img_path = os.path.join(image_dir, img_name)
    img_pil = Image.open(img_path).convert("RGB")
    
    input_tensor = img_tensor.unsqueeze(0).to(device)
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).data.squeeze()
    class_idx = torch.argmax(probs).item()
    
    # Get softmax weights
    params = list(model.parameters())
    weight_softmax = params[-2].detach().cpu().numpy()
    weight_softmax = np.squeeze(weight_softmax)

    # Generate CAM
    cam_img_pil, cam_array = returnCAM(features_blobs[0], weight_softmax, class_idx, img_pil.size)
    jet = cm.get_cmap("jet")
    cam_color = jet(cam_array)[:, :, :3]
    cam_color = np.uint8(255 * cam_color)
    cam_color_img = Image.fromarray(cam_color).resize(img_pil.size)
    blended = Image.blend(img_pil, cam_color_img, alpha=0.5)

    # Save CAM
    save_path = os.path.join(output_folder, f"CAM_{img_name}")
    blended.save(save_path)
    print(f"Saved: {save_path}")