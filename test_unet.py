import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from unet import SimpleUNet, FocalLoss, compute_miou
from train_unet import OxfordPetsDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =============== Set device ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============== Check model file ==================
model_path = "unet_model.pth"
if not os.path.exists(model_path):
    print("No trained model found. Please run train_unet.py first.")
    exit()

# =============== Load model ==================
model = SimpleUNet(in_channels=3, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()
print("Model loaded from unet_model.pth")

# =============== Define transforms ==================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# =============== Load test set ==================
test_dataset = OxfordPetsDataset(root_dir="./", split='test', transform=transform,
                                 mask_transform=mask_transform, subset=100)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

# =============== Evaluate model ==================
total_miou = 0.0
count = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        miou = compute_miou(preds, labels, num_classes=2)
        total_miou += miou
        count += 1
avg_miou = total_miou / count
print(f"average mIoU: {avg_miou:.4f}")

# =============== Visualise the first 5 images ==================
import re

def natural_key(filename):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]
valid_ext = ('.jpg', '.jpeg', '.png')
all_image_files = sorted(
    [f for f in os.listdir('./images') if f.lower().endswith(valid_ext)],
    key=natural_key
)
first5_filenames = all_image_files[:5]
fig, axes = plt.subplots(5, 3, figsize=(12, 12))

for i, filename in enumerate(first5_filenames):
    image_path = os.path.join("./images", filename)
    base_name = os.path.splitext(filename)[0]
    mask_path = os.path.join("./annotations/trimaps", base_name + ".png")


    image = Image.open(image_path).convert("RGB")
    if not os.path.exists(mask_path):
        print(f"[WARNING] Mask not found for {filename}, skipping...")
        continue
    mask = Image.open(mask_path)
    mask_np = np.array(mask)
    mask_bin = (mask_np > 1).astype(np.uint8)
    mask_pil = Image.fromarray(mask_bin * 255)

    image_tensor = transform(image).unsqueeze(0).to(device)
    mask_tensor = mask_transform(mask_pil).squeeze(0).long()

    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image_vis = image_tensor[0].cpu() * std[:, None, None] + mean[:, None, None]
    image_vis = torch.clamp(image_vis, 0, 1).permute(1, 2, 0).numpy()

    axes[i, 0].imshow(image_vis)
    axes[i, 0].set_title(f"Image: {filename}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask_tensor.cpu(), cmap='gray')
    axes[i, 1].set_title("Ground Truth")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(pred, cmap='gray')
    axes[i, 2].set_title("Prediction")
    axes[i, 2].axis("off")

    print(f"Inferred and visualised: {filename}")

plt.tight_layout()
plt.savefig("UNET_FIRST5_IMAGES.png")
plt.show()
print("[INFO] Results saved to UNET_FIRST5_IMAGES.png")
