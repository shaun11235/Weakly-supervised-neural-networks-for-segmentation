import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from unet import SimpleUNet, FocalLoss, compute_miou, fit_sgd
import time


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================ Data loading =====================
class OxfordPetsDataset(Dataset):
    """
       Custom dataset for Oxford-IIIT Pets, loads images and corresponding segmentation masks.
    """
    def __init__(self, root_dir, split='train', transform=None, mask_transform=None, subset=None):

        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'annotations', 'trimaps')

        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.image_files.sort()

        if subset is not None:
            self.image_files = self.image_files[:subset]

        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        base_name = os.path.splitext(image_name)[0]
        mask_path = os.path.join(self.masks_dir, base_name + '.png')

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        mask = np.array(mask)
        mask = (mask > 1).astype(np.uint8)
        mask = Image.fromarray(mask * 255)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).long().squeeze(0)
        else:
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).long().squeeze(0)

        return image, mask


root_dir = r"./"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# =============== Train test split =================
train_dataset = OxfordPetsDataset(root_dir=root_dir, split='train', transform=transform,
                                  mask_transform=mask_transform, subset=400)
test_dataset = OxfordPetsDataset(root_dir=root_dir, split='test', transform=transform,
                                 mask_transform=mask_transform, subset=100)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)


model = SimpleUNet(in_channels=3, num_classes=2)
criterion = FocalLoss(gamma=2, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
num_epochs = 100
start_time = time.time()

# =============== Train the model ==================
fit_sgd(model, train_loader, optimizer, criterion, num_epochs, device)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

model.eval()
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

# ============== Visualisation of the results of the first five predictions ==============
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
