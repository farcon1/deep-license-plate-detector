from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision import models

from .plate_text import get_num_classes


class MultiHeadPlateRecognizer(nn.Module):
    def __init__(self, backbone_name: str = "resnet18"):
        super().__init__()
        self.backbone_name = str(backbone_name).lower()
        num_classes: List[int] = get_num_classes()

        if self.backbone_name == "resnet18":
            backbone = models.resnet18(weights=None)
            feat_dim = int(backbone.fc.in_features)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
        elif self.backbone_name == "resnet34":
            backbone = models.resnet34(weights=None)
            feat_dim = int(backbone.fc.in_features)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
        elif self.backbone_name == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(weights=None)
            feat_dim = int(backbone.classifier[-1].in_features)
            self.features = nn.Sequential(
                backbone.features,
                backbone.avgpool,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.dropout = nn.Dropout(p=0.2)
        self.heads = nn.ModuleList([nn.Linear(feat_dim, n) for n in num_classes])

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        feat = self.dropout(feat)
        logits = [head(feat) for head in self.heads]
        return logits


def build_ocr_model(backbone_name: str = "resnet18") -> MultiHeadPlateRecognizer:
    return MultiHeadPlateRecognizer(backbone_name=backbone_name)