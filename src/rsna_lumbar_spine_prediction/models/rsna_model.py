from __future__ import annotations

import timm
import torch.nn as nn


class RSNA24Model(nn.Module):
    def __init__(self, model_name: str, in_c: int = 42, n_classes: int = 75, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_c,
            num_classes=n_classes,
            global_pool="avg",
        )

    def forward(self, x):
        return self.model(x)
