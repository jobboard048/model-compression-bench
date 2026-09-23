from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Quantization-friendly CNN with optional Quant/DeQuant stubs."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        width: int = 16,
        extra_block: bool = False,
    ) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        if extra_block:
            layers.extend(
                [
                    nn.Conv2d(width * 2, width * 4, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            feat = width * 4
            spatial = 3 if in_channels == 1 else 4  # 28->3 after 3 pools; 32->4
        else:
            feat = width * 2
            spatial = 7 if in_channels == 1 else 8
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat * spatial * spatial, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        """Fuse Conv-ReLU pairs for better post-training quantization."""
        torch.ao.quantization.fuse_modules(
            self.features,
            [["0", "1"], ["3", "4"]],
            inplace=True,
        )


def teacher_model(in_channels: int = 1, num_classes: int = 10) -> SmallCNN:
    return SmallCNN(num_classes=num_classes, in_channels=in_channels, width=32, extra_block=False)


def student_model(in_channels: int = 1, num_classes: int = 10) -> SmallCNN:
    return SmallCNN(num_classes=num_classes, in_channels=in_channels, width=8, extra_block=False)
