import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Basic residual block:
        Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> shortcut -> ReLU
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        #self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                #nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)

        return out

class SpatialMultiHeadAttention(nn.Module):
    """
    Input:
        x: [B, C, H, W]

    Process:
        [B, C, H, W] -> [B, H*W, C]
        Multi-head self-attention
        [B, H*W, C] -> [B, C, H, W]
    """
    def __init__(self, in_channels= 128, out_channels=512, num_heads=2, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1))


    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) # [B, C, H, W] -> [B, H*W, C]
        attn_output, _ = self.attn(tokens, tokens, tokens, need_weights=False) # Query = Key = Value = spatial tokens
        tokens = tokens + attn_output
        out = tokens.transpose(1, 2).reshape(B, C, H, W) # [B, H*W, C] -> [B, C, H, W]
        out = self.downsample(out)

        return out


class ChenCNN(nn.Module):
    """
    Encoder based on:
    Chen et al. (2022), "An Auto-adjusting Weight Model for Imbalanced Wafer Defects Recognition"

    Input:
        [B, 3, 224, 224]

    Architecture:
        Conv7x7, stride=2, 64 channels       -> [B, 64, 112, 112]
        MaxPool3x3, stride=2                 -> [B, 64, 56, 56]
        Residual stage 1: 3 blocks, 64 ch     -> [B, 64, 56, 56]
        Residual stage 2: 4 blocks, 128 ch    -> [B, 128, 28, 28]
        Multi-head Attention                 -> [B, 512, 14, 14]
        Global Average Pooling               -> [B, 512]
    """
    def __init__(self, emb_dim=512, num_heads=2):
        super().__init__()

    
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


        # 3 convolution layers with residual structure, 64 channels
        # 4 convolution layers with residual structure, 128 channels
        # 5 multi-head attention
        # 6 global average pooling
        self.stage1 = self._make_stage(in_channels=64, out_channels=64, num_blocks=3, stride=1)
        self.stage2 = self._make_stage(in_channels=64, out_channels=128, num_blocks=4, stride=2)

        self.attention = SpatialMultiHeadAttention(in_channels=128, out_channels=512, num_heads=num_heads, dropout=0.0) 
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 9)
    
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.stem(x)       # [B, 64, 56, 56]
        x = self.stage1(x)     # [B, 64, 56, 56]
        x = self.stage2(x)     # [B, 128, 28, 28]
        x = self.attention(x)  # [B, 512, 14, 14]

        z = self.global_pool(x)   # [B, 512, 1, 1]
        z = z.flatten(start_dim=1)  # [B, 512]
        z = self.fc(z) # [B, 9]
        return z