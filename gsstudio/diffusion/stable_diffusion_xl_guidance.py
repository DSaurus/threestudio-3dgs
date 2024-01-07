from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from threestudio.utils.typing import *

import gsstudio

from .stable_diffusion_guidance import DiffusersStableDiffusionGuidance


def prepare_latents(self, *args, **kwargs):
    return self.prepared_latents


@gsstudio.register("diffusers-stable-diffusion-xl-guidance")
class DiffusersStableDiffusionXLGuidance(DiffusersStableDiffusionGuidance):
    @dataclass
    class Config(DiffusersStableDiffusionGuidance.Config):
        fixed_width: int = 1024
        fixed_height: int = 1024
        fixed_latent_width: int = 128
        fixed_latent_height: int = 128

    cfg: Config

    def create_pipe(self):
        HookPipeline = type(
            "HookPipeline",
            (StableDiffusionXLPipeline,),
            {"prepare_latents": prepare_latents},
        )
        self.pipe = HookPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            variant="fp16" if self.cfg.half_precision_weights else None,
            **self.pipe_kwargs,
        ).to(self.device)
        self.output_type = "latent"
        self.pipe.vae.to(dtype=torch.float32)

    def prepare_text_embeddings(self, prompt_utils, **kwargs):
        return {"prompt": prompt_utils.prompt}
