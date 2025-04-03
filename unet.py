import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        """
        构建一个简单的 U-Net 模型，仅包含一次池化和一次反卷积操作。
        
        参数:
            in_channels: 输入图像的通道数，默认为 3（RGB 图像）
            num_classes: 输出类别数，默认为 2（宠物和背景）
        """
        super(SimpleUNet, self).__init__()
        
        # 编码器部分
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 池化后进一步提取特征（瓶颈层）
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 解码器部分：使用反卷积进行上采样
        self.upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # 拼接后通道数为 32 (反卷积输出) + 32 (编码器跳跃连接)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # 输出层：1x1 卷积将特征映射到类别数
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        x1 = F.relu(self.conv1(x))      # 第一层卷积
        x2 = F.relu(self.conv2(x1))      # 第二层卷积 (作为后续跳跃连接)
        x_pooled = self.pool(x2)         # 池化下采样
        
        # 瓶颈层
        x3 = F.relu(self.conv3(x_pooled))
        
        # 解码器部分：反卷积上采样
        x_up = self.upconv(x3)
        
        # 拼接编码器中对应的特征 (跳跃连接)
        # 注意：要求输入尺寸合理使得 x_up 与 x2 尺寸一致
        x_cat = torch.cat([x_up, x2], dim=1)
        x4 = F.relu(self.conv4(x_cat))
        x5 = F.relu(self.conv5(x4))
        
        # 输出层（输出 logits）
        out = self.out_conv(x5)
        return out
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        """
        初始化 focal loss 类
        
        参数:
            gamma: 调整难易样本损失的指数因子，这里默认设为 2
            reduction: 损失归约方式，支持 'mean', 'sum' 或 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算 focal loss
        
        参数:
            inputs: 模型输出 logits，尺寸为 (N, C, H, W)
            targets: 真实标签，尺寸为 (N, H, W)，取值为类别索引
        返回:
            根据 reduction 返回 focal loss 值
        """
        # 计算每个像素的对数概率
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # 将目标标签扩展至与 logits 对应的维度
        targets = targets.unsqueeze(1)  # 尺寸变为 (N, 1, H, W)
        # 根据目标标签选择对应类别的对数概率和概率值
        log_pt = log_probs.gather(1, targets)
        pt = probs.gather(1, targets)
        
        # focal loss 公式：FL = - (1 - pt)^gamma * log(pt)
        loss = - (1 - pt) ** self.gamma * log_pt
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
def compute_miou(pred, target, num_classes=2):
    """
    计算单个 mini-batch 的 mIoU
    参数:
        pred: 预测结果，形状为 (N, H, W)，每个像素的类别由 argmax 得到
        target: 真实标签，形状为 (N, H, W)
        num_classes: 类别数，默认为 2
    返回:
        mIoU：所有类别的 IoU 的平均值
    """
    ious = []
    for cls in range(num_classes):
        # 预测为当前类别的像素位置
        pred_inds = (pred == cls)
        # 真实标签为当前类别的像素位置
        target_inds = (target == cls)
        # 交集和并集
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        # 如果该类别在预测和真实中均不存在，则认为 IoU 为 1
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)
    # 计算所有类别的平均 IoU
    return sum(ious) / num_classes

def fit_sgd(model, train_loader, optimizer, criterion, num_epochs, device):
    """
    使用 mini-batch 随机梯度下降（SGD 变种）对模型进行训练，
    并在每一次迭代时计算并输出 mIoU。
    
    参数:
        model: 待训练的 U-Net 模型
        train_loader: 训练数据的 DataLoader，返回 (inputs, labels)
        optimizer: 优化器（例如 torch.optim.Adam）
        criterion: 损失函数（例如 FocalLoss）
        num_epochs: 训练的轮数
        device: 计算设备（如 'cuda' 或 'cpu'）
    """
    model.to(device)
    model.train()
    
    # 遍历所有 epoch
    for epoch in range(num_epochs):
        running_loss = 0.0  # 当前 epoch 内累计的损失
        running_miou = 0.0  # 当前 epoch 内累计的 mIoU
        # 遍历每个 mini-batch
        for i, (inputs, labels) in enumerate(train_loader):
            # 将数据移动到指定设备
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播：计算模型输出
            outputs = model(inputs)
            
            # 计算当前 mini-batch 的损失
            loss = criterion(outputs, labels)
            
            # 反向传播计算梯度
            loss.backward()
            
            # 更新模型参数
            optimizer.step()
            
            # 累计损失
            running_loss += loss.item()
            
            # 计算当前 mini-batch 的 mIoU
            # 获取预测结果，取每个像素上得分最大的类别
            preds = outputs.argmax(dim=1)
            miou = compute_miou(preds, labels, num_classes=2)
            running_miou += miou
            
            # 输出当前迭代的损失和 mIoU
            print(f"Epoch [{epoch+1}/{num_epochs}] Iteration [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f} mIoU: {miou:.4f}")
        
        # 计算当前 epoch 的平均损失和 mIoU
        avg_loss = running_loss / len(train_loader)
        avg_miou = running_miou / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f} Average mIoU: {avg_miou:.4f}")
    
