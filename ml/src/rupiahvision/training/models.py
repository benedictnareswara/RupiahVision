"""Models compared in v1: a small custom CNN baseline and MobileNetV3 transfer learning."""

from __future__ import annotations

from torch import nn
from torchvision import models

MODEL_NAMES = ("custom_cnn", "mobilenet_v3_small", "mobilenet_v3_large")


class SmallCNN(nn.Module):
    """Five conv blocks (conv-BN-ReLU-pool), global average pooling, dropout, linear.

    The baseline: trained from scratch, so it shows what transfer learning adds.
    """

    def __init__(self, num_classes: int, widths: tuple[int, ...] = (32, 64, 128, 256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        channels = 3
        for width in widths:
            layers += [
                nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(channels, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "custom_cnn":
        return SmallCNN(num_classes)
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    # Replace only the last layer: ImageNet's 1000 classes -> our outputs.
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def parameter_groups(model: nn.Module, lr: float, backbone_lr_scale: float) -> list[dict]:
    """The new classifier learns at `lr`; pretrained features move more slowly."""
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone, "lr": lr * backbone_lr_scale},
        {"params": head, "lr": lr},
    ]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
