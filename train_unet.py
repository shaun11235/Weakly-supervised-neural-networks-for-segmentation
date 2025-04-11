import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from unet import SimpleUNet, FocalLoss, fit_sgd
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================ Data loading =====================
class OxfordPetsDataset(Dataset):
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

# =================== Training Logic =====================
if __name__ == "__main__":

    # =============== Set device ==================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # =============== Load training set ==================
    train_dataset = OxfordPetsDataset(root_dir="./", split='train', transform=transform,
                                      mask_transform=mask_transform, subset=400)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    # =============== Load test set (for evaluation during training) ==================
    test_dataset = OxfordPetsDataset(root_dir="./", split='test', transform=transform,
                                     mask_transform=mask_transform, subset=100)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # =============== Model setup ==================
    model = SimpleUNet(in_channels=3, num_classes=2)
    criterion = FocalLoss(gamma=2, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.to(device)

    # =============== Train the model ==================
    num_epochs = 100
    start_time = time.time()
    fit_sgd(model, train_loader, test_loader, optimizer, criterion, num_epochs, device)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    # =============== Save the model ==================
    torch.save(model.state_dict(), "unet_model.pth")
    print("Model saved to unet_model.pth")
