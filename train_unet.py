import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# 从 unet 脚本中导入所需的类和函数
from unet import SimpleUNet, FocalLoss, compute_miou, fit_sgd

class OxfordPetsDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, mask_transform=None, subset=None):
        """
        自定义 Oxford-IIIT Pets 数据集类，用于加载图像及对应的分割 mask。
        任务：将 mask 二值化，即将 mask 中大于1的像素视为 pet（类别1），其他视为背景（类别0）。
        
        参数:
            root_dir: 数据集根目录，应包含 'images' 和 'annotations/trimaps' 文件夹
            split: 数据集划分（此处不做严格区分 train 与 test，可根据需求扩展）
            transform: 对图像进行预处理的转换（例如调整尺寸、归一化等）
            mask_transform: 对 mask 进行预处理的转换（例如调整尺寸，采用最近邻插值）
            subset: 若不为 None，则只加载前 subset 个样本，便于调试
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        
        # 获取 images 文件夹下所有 .jpg 或 .png 文件，并排序
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.image_files.sort()
        
        if subset is not None:
            self.image_files = self.image_files[:subset]
        
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像及对应 mask 的完整路径
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        base_name = os.path.splitext(image_name)[0]
        mask_path = os.path.join(self.masks_dir, base_name + '.png')
        
        # 打开图像和 mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # 将 mask 转换为 numpy 数组，并进行二值化处理：
        # 假设原始 mask 中背景为1，而 pet 或边界为2或3，此处将大于1的像素视为 pet（类别1），其他视为背景（类别0）
        mask = np.array(mask)
        mask = (mask > 1).astype(np.uint8)
        # 转换为 PIL Image 以便后续转换（乘以255便于转换为单通道图像）
        mask = Image.fromarray(mask * 255)
        
        # 对图像和 mask 分别进行转换
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            # mask_transform 后 mask 为 float 张量，阈值0.5二值化，并转换为 long 类型，同时去掉通道维度
            mask = (mask > 0.5).long().squeeze(0)
        else:
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).long().squeeze(0)
        
        return image, mask

if __name__ == '__main__':
    # 固定随机种子，保证结果可复现
    torch.manual_seed(42)
    
    # 设置数据集根目录，请修改为你的实际路径，例如：
    root_dir = r"C:\Users\14629\Desktop\u-net"
    
    # 定义图像的预处理转换：调整尺寸、转换为 Tensor、归一化
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # 定义 mask 的预处理转换：调整尺寸时采用最近邻插值，转换为 Tensor
    mask_transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    # 构造训练集和测试集（这里为了调试分别使用部分样本）
    train_dataset = OxfordPetsDataset(root_dir=root_dir, split='train', transform=transform, mask_transform=mask_transform, subset=100)
    test_dataset = OxfordPetsDataset(root_dir=root_dir, split='test', transform=transform, mask_transform=mask_transform, subset=20)
    
    # 构造 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # 实例化 U-Net 模型、focal loss 损失函数和优化器
    model = SimpleUNet(in_channels=3, num_classes=2)
    criterion = FocalLoss(gamma=2, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 选择计算设备，若有 GPU 则使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置训练轮数
    num_epochs = 5
    
    # 开始训练模型，fit_sgd 内部每次迭代都会输出当前的 loss 和 mIoU
    fit_sgd(model, train_loader, optimizer, criterion, num_epochs, device)
    
    # 训练结束后，在测试集上进行评估，计算平均 mIoU
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
    print(f"测试集平均 mIoU: {avg_miou:.4f}")