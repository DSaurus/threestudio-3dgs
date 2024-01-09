from dataclasses import dataclass, field

import torch.nn.functional as F
from diffusers import IFPipeline

import gsstudio
from gsstudio.utils.typing import *

from .base import DiffusersGuidance


def prepare_intermediate_images(self, *args, **kwargs):
    return self.prepared_latents


@gsstudio.register("diffusers-deep-floyd-guidance")
class DiffusersDFGuidance(DiffusersGuidance):
    @dataclass
    class Config(DiffusersGuidance.Config):
        fixed_width: int = 64
        fixed_height: int = 64

    cfg: Config

    def configure(self) -> None:
        super().configure()

    def create_pipe(self):
        HookIFPipeline = type(
            "HookIFPipeline",
            (IFPipeline,),
            {"prepare_intermediate_images": prepare_intermediate_images},
        )
        self.pipe = HookIFPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype,
        ).to(self.device)

    def prepare_latents(
        self, rgb: Float[Tensor, "B H W C"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        assert rgb_as_latents == False, f"No latent space in {self.__class__.__name__}"
        rgb_BCHW = rgb_BCHW * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(
            rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
        )
        return latents

    def prepare_other_conditions(self, **kwargs):
        output = super().prepare_other_conditions(**kwargs)
        output.update({"output_type": "pt"})
        return output
