import numpy as np
import torch.nn as nn


class JangCNN(nn.Module):

    def __init__(self, emb_dim=4096):
        super(JangCNN, self).__init__()
        self.conv = nn.Sequential(
            # Block1
            nn. Conv2d(in_channels = 1, out_channels= 8, kernel_size=3, stride=1, padding=1),  # [Batch, 3, 128, 128] -> # [Batch, 8, 128, 128]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [Batch, 8, 128, 128] -> # [Batch, 8, 64, 64]
            
            # Block1
            nn. Conv2d(in_channels = 8, out_channels= 16, kernel_size=3, stride=1, padding=1), # [Batch, 8, 64, 64] -> # [Batch, 16, 64, 64]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [Batch, 16, 64, 64] -> # [Batch, 16, 32, 32]

            # Block2
            nn. Conv2d(in_channels = 16, out_channels= 32, kernel_size=3, stride=1, padding=1),  # [Batch, 16, 32, 32] -> # [Batch, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [Batch, 32, 32, 32] -> # [Batch, 32, 16, 16]
            
            # Block3
            nn. Conv2d(in_channels = 32, out_channels= 64, kernel_size=3, stride=1, padding=1),  # [Batch, 32, 16, 16] -> # [Batch, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [Batch, 64, 16, 16] -> # [Batch, 64, 8, 8]  # 4096
            
            #nn.AdaptiveAvgPool2d((1,1))          # [B, 512, 1, 1]
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(64*8*8, 9)


    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1) # [BATCH_SIZE, 64 * 8 * 8]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

