import numpy as np
import torch.nn as nn

class CNN_Encoder(nn.Module):
    def __init__(self, emb_dim=1024):
        super(CNN_Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn. Conv2d(in_channels = 3, out_channels= 32, kernel_size=3, stride=1, padding=1),  # [Batch, 3, 128, 128] -> # [Batch, 32, 128, 128]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [Batch, 32, 128, 128] -> # [Batch, 32, 64, 64]
            
            # Block1
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # [Batch, 32, 64, 64] -> # [Batch, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),   # [Batch, 64, 32, 32] -> # [Batch, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), # [Batch, 128, 16, 16] -> # [Batch, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Block4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), # [Batch, 256, 8, 8] -> # [Batch, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=4, stride=2) # [Batch, 512, 1, 1]
            #nn.AdaptiveAvgPool2d((1,1))          # [B, 512, 1, 1]

        )
        
    def forward(self, x):
        x = self.conv(x) # [BATCH_SIZE, 512, 1, 1 ]
        x = x.flatten(start_dim=1) # [BATCH_SIZE, 512 * 1 * 1 ]
        return x


class MLP(nn.Module):
    def __init__(self, emb_dim=512, h1_dim = 256, h2_dim=128, out_dim=9, drop_p=0.0):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, h1_dim),
            nn.BatchNorm1d(h1_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_p),

            nn.Linear(h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim),
            nn.LeakyReLU(),

            nn.Linear(h2_dim, out_dim)
        )

    def forward(self, z):
        x = self.classifier(z)
        return x


class CNN_Classifier(nn.Module):
    def __init__(self, emb_dim=512, h1_dim = 256, h2_dim=128, out_dim=9, drop_p=0.0):
        super(CNN_Classifier, self).__init__()
        self.encoder = CNN_Encoder(emb_dim=emb_dim)
        self.classifier = MLP(emb_dim=emb_dim, h1_dim=h1_dim, h2_dim=h2_dim, out_dim=out_dim, drop_p=drop_p)

    def forward(self, x):
        z = self.encoder(x)
        output = self.classifier(z)
        return output