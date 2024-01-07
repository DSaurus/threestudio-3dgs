from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from threestudio.utils.typing import *

import gsstudio

from .diffusers_guidance import DiffusersGuidance


def prepare_latents(self, *args, **kwargs):
    return self.prepared_latents


@gsstudio.register("diffusers-stable-diffusion-guidance")
class DiffusersStableDiffusionGuidance(DiffusersGuidance):
    @dataclass
    class Config(DiffusersGuidance.Config):
        fixed_width: int = 512
        fixed_height: int = 512
        fixed_latent_width: int = 64
        fixed_latent_height: int = 64

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.pipe.vae.eval()
        for p in self.pipe.vae.parameters():
            p.requires_grad_(False)

    def create_pipe(self):
        HookPipeline = type(
            "HookPipeline",
            (StableDiffusionPipeline,),
            {"prepare_latents": prepare_latents},
        )
        self.pipe = HookPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **self.pipe_kwargs,
        ).to(self.device)

    def prepare_latents(
        self, rgb: Float[Tensor, "B H W C"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 H W"]:
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 H W"]
        if rgb_as_latents:
            if self.cfg.fixed_latent_height > 0 and self.cfg.fixed_latent_width > 0:
                latents = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.fixed_latent_height, self.cfg.fixed_latent_width),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            if self.cfg.fixed_height > 0 and self.cfg.fixed_width > 0:
                rgb_BCHW_512 = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.fixed_height, self.cfg.fixed_width),
                    mode="bilinear",
                    align_corners=False,
                )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.pipe.vae.encode(imgs.to(self.pipe.vae.dtype)).latent_dist
        latents = posterior.sample() * self.pipe.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents,
            (self.cfg.fixed_latent_height, self.cfg.fixed_latent_width),
            mode="bilinear",
            align_corners=False,
        )
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        image = self.pipe.vae.decode(latents.to(self.pipe.vae.dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def prepare_other_conditions(self, **kwargs):
        output = super().prepare_other_conditions(**kwargs)
        output.update({"output_type": "latent"})
        return output
