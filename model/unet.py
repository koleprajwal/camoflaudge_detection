import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 1)

        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.upsample(self.relu(self.conv1(x)))
        x = self.upsample(self.relu(self.conv2(x)))
        x = self.upsample(self.relu(self.conv3(x)))
        x = self.conv4(x)

        # Fix size mismatch
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return torch.sigmoid(x)