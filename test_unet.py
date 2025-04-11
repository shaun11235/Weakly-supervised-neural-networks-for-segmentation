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

# =============== Visualisation ==================
num_samples = 5
fig, axes = plt.subplots(num_samples, 3, figsize=(12, 12))

for i in range(num_samples):
    image, mask = test_dataset[i]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    pred = output.argmax(dim=1).cpu().numpy()[0]
    mask = mask.numpy()

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = image * std[:, None, None] + mean[:, None, None]
    image = torch.clamp(image, 0, 1)

    axes[i, 0].imshow(image.cpu().squeeze(0).permute(1, 2, 0))
    axes[i, 0].set_title(f'Input Image {i + 1}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(mask, cmap='gray')
    axes[i, 1].set_title(f'Ground Truth {i + 1}')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(pred, cmap='gray')
    axes[i, 2].set_title(f'Predicted {i + 1}')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig("UNET.png")
plt.show()
