import torch
import torch.nn as nn
import torch.nn.functional as F


class WaferAutoEncoder(nn.Module):
    """
    Autoencoder for wafer map augmentation.

    Input:
        x: [B, 3, H, W]
           one-hot encoded wafer map channels:
           channel 0 = background
           channel 1 = normal die
           channel 2 = defective die

    Output:
        logits: [B, 3, H, W]
                reconstructed categorical logits
        z: latent feature map
    """
    def __init__(self, latent_channels=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   # [B, 32, H, W]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # [B, 32, H/2, W/2]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # [B, 64, H/4, W/4]

            nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1), # [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # [B, 32, H, W]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)    # [B, 3, H, W]
        )

    def forward(self, x):
        z = self.encoder(x)
        logits = self.decoder(z)
        return logits, z

    @torch.no_grad()
    def generate(self, x, noise_std=0.10):
        self.eval()

        z = self.encoder(x)
        noise = torch.randn_like(z) * noise_std
        z_noisy = z + noise

        logits = self.decoder(z_noisy)
        prob = torch.softmax(logits, dim=1)
        gen_map = torch.argmax(prob, dim=1)  # [B, H, W], categorical map
        return gen_map, prob
    
class DepthwiseSeparableConv(nn.Module):
    """
    Basic block in DMC1 model.

    configurations:
        Depthwise Conv 3x3
        Pointwise Conv 1x1
        BatchNorm
        ReLU
        selective MaxPool2d
    """
    def __init__(self, in_channels, out_channels, use_pool=False):
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
    
class DMC1(nn.Module):
    """
    DMC1: Defect-Map Classification Network 1

    논문:
    A Light-Weight Neural Network for Wafer Map Classification
    Based on Data Augmentation

    Input:
        x: [B, 3, 64, 64]

    Output:
        logits: [B, num_classes]
    """
    def __init__(self, emb_dim=128):
        super().__init__()

        # Initial standard convolution
        # Input:  [B, 3, 64, 64]
        # Output: [B, 16, 64, 64]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Depthwise separable convolution blocks
        self.dw1 = DepthwiseSeparableConv(in_channels=16, out_channels=16, use_pool=False)
        self.dw2 = DepthwiseSeparableConv(16, 16, use_pool=True)   # 64 -> 32
        self.dw3 = DepthwiseSeparableConv(16, 32, use_pool=False)
        self.dw4 = DepthwiseSeparableConv(32, 32, use_pool=False)
        self.dw5 = DepthwiseSeparableConv(32, 32, use_pool=False)
        self.dw6 = DepthwiseSeparableConv(32, 32, use_pool=True)   # 32 -> 16

        self.dw7 = DepthwiseSeparableConv(32, 64, use_pool=False)
        self.dw8 = DepthwiseSeparableConv(64, 64, use_pool=True)   # 16 -> 8

        self.dw9 = DepthwiseSeparableConv(64, 128, use_pool=False)
        self.dw10 = DepthwiseSeparableConv(128, 128, use_pool=True) # 8 -> 4

        
        # Average pooling
        self.avgpool = nn.AvgPool2d(kernel_size=4) # 4x4 average pooling -> 1x1 feature ma

        # Fully connected layers
        self.fc1 = nn.Linear(128, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 9)

    def forward(self, x):
        x = self.conv1(x)

        x = self.dw1(x)
        x = self.dw2(x)

        x = self.dw3(x)
        x = self.dw4(x)
        x = self.dw5(x)
        x = self.dw6(x)

        x = self.dw7(x)
        x = self.dw8(x)

        x = self.dw9(x)
        x = self.dw10(x)

        x = self.avgpool(x)          # [B, 128, 1, 1]
        x = torch.flatten(x, 1)      # [B, 128]

        x = self.fc1(x)              # [B, 512]
        x = self.relu(x)
        logits = self.fc2(x)         # [B, 9]

        return logits