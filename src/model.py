
"""
CNN-ViT with Adaptive Attention for Medical Image Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBranch(nn.Module):
    """CNN branch: local feature extraction"""
    def __init__(self, in_channels=1, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_dim, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        return x.flatten(1)


class ViTBranch(nn.Module):
    """ViT branch: global dependency modeling"""
    def __init__(self, img_size=28, patch_size=7, in_channels=1,
                 embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]


class AdaptiveAttention(nn.Module):
    """Adaptive Attention: dynamic fusion of CNN and ViT"""
    def __init__(self, dim=128, reduction=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, cnn_feat, vit_feat):
        concat = torch.cat([cnn_feat, vit_feat], dim=-1)
        weights = self.gate(concat)
        w_cnn = weights[:, 0:1]
        w_vit = weights[:, 1:2]
        fused = w_cnn * cnn_feat + w_vit * vit_feat
        return fused, weights


class CNNViTAdaptive(nn.Module):
    """Full model: CNN-ViT + Adaptive Attention"""
    def __init__(self, num_classes=11, embed_dim=128):
        super().__init__()
        self.cnn_branch = CNNBranch(in_channels=1, out_dim=embed_dim)
        self.vit_branch = ViTBranch(img_size=28, patch_size=7,
                                    in_channels=1, embed_dim=embed_dim)
        self.adaptive_attn = AdaptiveAttention(dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn_branch(x)
        vit_feat = self.vit_branch(x)
        fused, attn_weights = self.adaptive_attn(cnn_feat, vit_feat)
        logits = self.classifier(fused)
        return logits, attn_weights
