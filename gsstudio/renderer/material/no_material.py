import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import gsstudio
from gsstudio.renderer.material.base import BaseMaterial
from gsstudio.utils.typing import *


@gsstudio.register("no-material")
class NoMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        n_output_dims: int = 3
        requires_normal: bool = False

    cfg: Config

    def configure(self) -> None:
        self.requires_normal = self.cfg.requires_normal

    def forward(
        self, features: Float[Tensor, "B ... Nf"], **kwargs
    ) -> Float[Tensor, "B ... Nc"]:
        return features

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        color = self(features, **kwargs).clamp(0, 1)
        assert color.shape[-1] >= 3, "Output color must have at least 3 channels"
        if color.shape[-1] > 3:
            gsstudio.warn("Output color has >3 channels, treating the first 3 as RGB")
        return {"albedo": color[..., :3]}
