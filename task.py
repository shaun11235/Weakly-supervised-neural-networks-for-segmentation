import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import matplotlib.cm as cm
import numpy as np


# ========== Class and Function Definitions ==========
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
def hook_feature(module, input, output):
    features_blobs.append(output.detach().numpy())


# Function to generate the CAM heatmap
# (using the provided image size for resampling)
def returnCAM(feature_conv, weight_softmax, class_idx, img_size):
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam_img = np.uint8(255 * cam)
    return (Image.fromarray(cam_img).resize(img_size, resample=Image.BILINEAR),
            cam)


# ========== Main Code ==========
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Configure paths (using the paths from the first section)
image_dir = os.path.join(current_dir, "dataset", "oxford-iiit-pet", "images")
list_txt = os.path.join(current_dir, "dataset", "oxford-iiit-pet",
                        "annotations", "list.txt")
model_save_path = os.path.join(current_dir, "results", "pth", "resnet_pet.pth")
image_path = os.path.join(current_dir, "dataset", "test.jpg")
output_cam_path = os.path.join(current_dir, "results", "plots", "CAM.jpg")

num_classes = 37
batch_size = 16
epochs = 1
lr = 1e-3

# Step 1: Construct label mapping
with open(list_txt, "r") as f:
    lines = f.readlines()[6:]  # The first 6 lines are comments

img_labels = {}
for line in lines[:50]:  # Use only the first 50 samples (test version)
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
dataset = Subset(dataset, list(range(50)))  # Use only 50 images for testing
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 3: Define and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

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

# Step 4: Load model and generate CAM
model.eval()
model.load_state_dict(torch.load(model_save_path, map_location="cpu"))

finalconv_name = 'layer4'
features_blobs = []
model._modules.get(finalconv_name).register_forward_hook(hook_feature)

img_pil = Image.open(image_path).convert('RGB')
img_tensor = transform(img_pil).unsqueeze(0)
logits = model(img_tensor)
probs = F.softmax(logits, dim=1).data.squeeze()
class_idx = torch.argmax(probs).item()
print(f"Predicted class index: {class_idx}, "
      f"probability: {probs[class_idx]:.4f}")

params = list(model.parameters())
weight_softmax = params[-2].detach().numpy()
weight_softmax = np.squeeze(weight_softmax)

cam_img_pil, cam_array = returnCAM(features_blobs[0], weight_softmax,
                                   class_idx, img_pil.size)

jet = cm.get_cmap("jet")
cam_color = jet(cam_array)[:, :, :3]
cam_color = np.uint8(255 * cam_color)
cam_color_img = Image.fromarray(cam_color).resize(img_pil.size)
blended = Image.blend(img_pil, cam_color_img, alpha=0.5)
blended.save(output_cam_path)
print(f"Saved CAM heatmap to {output_cam_path}")
