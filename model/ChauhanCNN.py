import numpy as np
import torch.nn as nn
import torch.nn as nn


import torch
import torch.nn as nn


class Chauhan(nn.Module):
    """
    Input  : 3 x 26 x 26

    Block1 : Conv2D 3   -> 8,   kernel 3x3, MaxPool
    Block2 : Conv2D 8   -> 16,  kernel 3x3, MaxPool
    Block3 : Conv2D 16  -> 32,  kernel 5x5, MaxPool
    Block4 : Conv2D 32  -> 64,  kernel 5x5, MaxPool
    Block5 : Conv2D 64  -> 128, kernel 7x7, MaxPool

    26 -> 13 -> 7 -> 4 -> 2 -> 1

    Flatten : 128 x 1 x 1 = 128
    Dense1  : 128 -> 64
    FC      : 64 -> num_classes
    """

    def __init__(self, emb_dim=128):
        super(Chauhan, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64, 9)

    def extract_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x) # [B, 128]
        x = self.dense1(x)
        x = self.relu1(x) # [B, 64]
        return x

    def forward(self, x):
        x = self.extract_features(x) # [B, 64]
        x = self.fc(x) # [B, 9]
        return x