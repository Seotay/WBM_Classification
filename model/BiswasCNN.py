import numpy as np
import torch.nn as nn

class BiswasCNN(nn.Module):
    """
    Architecture:
        Input  : 3 x 26 x 26
        Conv1  : 3 -> 16,  kernel 3x3, padding 1
        Block1 : 16 -> 64, kernel 3x3, padding 1, MaxPool
        Block2 : 64 -> 128, kernel 3x3, padding 1, MaxPool / AdaptivePool
        FC1    : 512 -> 256
        FC2    : 256 -> 128
        FC3    : 128 -> num_classes

    """
    def __init__(self, emb_dim=512):
        super(BiswasCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(2*2*128, 256)
        self.dense2 = nn.Linear(256, 128)
        self.fc = nn.Linear(128, 9)


    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x) # [BATCH_SIZE, 512 * 1 * 1 ]
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.fc(x)
        return x


