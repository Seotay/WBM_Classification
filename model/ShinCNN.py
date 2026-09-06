import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3Small(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()

        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.model = models.mobilenet_v3_small(weights=weights)

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 9)

    def forward(self, x):
        return self.model(x)