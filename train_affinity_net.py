import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np


#用二值掩码训练affinitynet
image_dir = "dataset/oxford-iiit-pet/images"#原图路径
binary_mask_dir = "binary_mask"#二值掩码所在路径
model_save_path = "affinitynet.pth"#模型保存路径
batch_size = 4
num_epochs = 30
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEIGHBOR_SHIFTS = [(-1, 0),
                   (-1, 1),
                   (0, 1),
                   (1, 1),
                   (1, 0),
                   (1, -1),
                   (0, -1),
                   (-1, -1)]

def generate_affinity_label_from_mask(mask):
    H, W = mask.shape
    affinity = np.full((H, W, 8), -1, dtype=np.int8)

    for idx, (dy, dx) in enumerate(NEIGHBOR_SHIFTS):
        shifted = np.zeros_like(mask)
        if dy >= 0:
            y_src, y_dst = 0, H - dy
            y_src_shift, y_dst_shift = dy, H
        else:
            y_src, y_dst = -dy, H
            y_src_shift, y_dst_shift = 0, H + dy

        if dx >= 0:
            x_src, x_dst = 0, W - dx
            x_src_shift, x_dst_shift = dx, W
        else:
            x_src, x_dst = -dx, W
            x_src_shift, x_dst_shift = 0, W + dx

        shifted[y_src_shift:y_dst_shift, x_src_shift:x_dst_shift] = mask[y_src:y_dst, x_src:x_dst]
        affinity[:, :, idx] = (mask == shifted).astype(np.uint8)

    return affinity

class AffinityNet(nn.Module):
    def __init__(self):
        super(AffinityNet, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.conv_affinity = nn.Conv2d(512, 8, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.conv_affinity(feat)
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return torch.sigmoid(out)


class AffinityFromMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(mask_dir) if f.endswith("_mask.png")]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx].replace("_mask.png", ".jpg")
        img_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).resize((224, 224))
        mask_np = np.array(mask).astype(np.uint8) // 255

        if self.transform:
            img = self.transform(img)

        affinity_label = generate_affinity_label_from_mask(mask_np)
        valid_mask = (affinity_label != -1).astype(np.uint8)

        return img, affinity_label, valid_mask, name


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = AffinityFromMaskDataset(image_dir, binary_mask_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = AffinityNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction="none")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_pos, total_neg = 0.0, 0.0

        for images, affinity_labels, valid_masks, _ in loader:
            images = images.to(device)

            pos_mask = (affinity_labels == 1).float()
            neg_mask = (affinity_labels == 0).float()
            valid_mask = valid_masks.float()


            smoothed_labels = affinity_labels.clone().float()
            smoothed_labels[affinity_labels == 1] = 0.9
            smoothed_labels[affinity_labels == 0] = 0.1

            smoothed_labels = smoothed_labels.permute(0, 3, 1, 2).to(device)
            pos_mask = pos_mask.permute(0, 3, 1, 2).to(device)
            neg_mask = neg_mask.permute(0, 3, 1, 2).to(device)
            valid_mask = valid_mask.permute(0, 3, 1, 2).to(device)

            outputs = model(images)
            loss_map = criterion(outputs, smoothed_labels)

            pos_loss = (loss_map * pos_mask * valid_mask).sum() / (pos_mask.sum() + 1e-6)
            neg_loss = (loss_map * neg_mask * valid_mask).sum() / (neg_mask.sum() + 1e-6)
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pos += pos_loss.item()
            total_neg += neg_loss.item()

        print(f"[Epoch {epoch+1}/{num_epochs}] Total Loss: {total_loss/len(loader):.4f} | Pos: {total_pos/len(loader):.4f} | Neg: {total_neg/len(loader):.4f}")

    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"AffinityNet saved to: {model_save_path}")
