# We used generative AI in an assistive role for this assessment. 
# Specifically, we used ChatGPT (GPT-4), developed by OpenAI (https://chat.openai.com/), to check the comments in our Python files, 
# as well as to identify grammar and spelling issues in the instruction file. Additionally, 
# we used it to review potential errors in our Python code to reduce bugs and improve overall robustness.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============= A simple U-net ===============
class SimpleUNet(nn.Module):
    """
    A simple U-Net model consisting of an encoder and decoder, with a single pooling layer and a single deconvolution layer.

    in_channels: The number of input channels
    num_classes The number of output classes
    """

    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleUNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x_pooled = self.pool(x2)
        x3 = F.relu(self.conv3(x_pooled))
        x_up = self.upconv(x3)
        x_cat = torch.cat([x_up, x2], dim=1)
        x4 = F.relu(self.conv4(x_cat))
        x5 = F.relu(self.conv5(x4))

        out = self.out_conv(x5)
        return out


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for imbalanced classification tasks.

    gamma: Exponent factor for modulating easy examples, default is 2.
    reduction: Specifies the reduction to apply to the output.
    """

    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.unsqueeze(1)  # Shape becomes (N, 1, H, W)
        log_pt = log_probs.gather(1, targets)
        pt = probs.gather(1, targets)

        loss = - (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ================== mIoU =====================
def compute_miou(pred, target, num_classes=2):
    """
    Computes mean Intersection over Union (mIoU) for a given batch of predictions and ground truth.
    """
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)
    return sum(ious) / num_classes


# ================== sgd =======================
def fit_sgd(model, train_loader, test_loader, optimizer, criterion, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            total_miou = 0.0
            count = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    total_miou += compute_miou(preds, labels, num_classes=2)
                    count += 1
            avg_miou = total_miou / count if count > 0 else 0.0
            print(f"Epoch [{epoch + 1}/{num_epochs}] Test mIoU: {avg_miou:.4f}")