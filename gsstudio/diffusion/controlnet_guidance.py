from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from controlnet_aux import CannyDetector, NormalBaeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from threestudio.utils.typing import *

import gsstudio

from .stable_diffusion_guidance import DiffusersStableDiffusionGuidance


def prepare_latents(self, *args, **kwargs):
    return self.prepared_latents


@gsstudio.register("diffusers-controlnet-guidance")
class DiffusersControlNetGuidance(DiffusersStableDiffusionGuidance):
    @dataclass
    class Config(DiffusersStableDiffusionGuidance.Config):
        controlnet_model_name_or_path: str = "lllyasviel/control_v11p_sd15_canny"
        preprocessor_model_name_or_path: str = "lllyasviel/Annotators"
        controlnet_conditioning_scale: float = 1.0
        control_type: str = "normal"

        # Canny threshold
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if self.cfg.control_type == "normal":
            self.preprocessor = NormalBaeDetector.from_pretrained(
                self.cfg.preprocessor_model_name_or_path
            )
            self.preprocessor.model.to(self.device)
        elif self.cfg.control_type == "canny":
            self.preprocessor = CannyDetector()

    def create_pipe(self):
        controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_model_name_or_path, torch_dtype=torch.float16
        )
        HookPipeline = type(
            "HookPipeline",
            (StableDiffusionControlNetPipeline,),
            {"prepare_latents": prepare_latents},
        )
        self.pipe = HookPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            controlnet=controlnet,
            **self.pipe_kwargs,
        ).to(self.device)

    def prepare_other_conditions(self, **kwargs):
        output = super().prepare_other_conditions(**kwargs)

        cond_rgb = kwargs["cond_rgb"]
        if self.cfg.control_type == "normal":
            cond_rgb = (
                (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            )
            detected_map = self.preprocessor(cond_rgb)
            control = (
                torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            )
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif self.cfg.control_type == "canny":
            cond_rgb = (
                (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            )
            blurred_img = cv2.blur(cond_rgb, ksize=(5, 5))
            detected_map = self.preprocessor(
                blurred_img, self.cfg.canny_lower_bound, self.cfg.canny_upper_bound
            )
            control = (
                torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            )
            control = control.unsqueeze(-1).repeat(1, 1, 3)
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)

        control = F.interpolate(
            control,
            (self.cfg.fixed_height, self.cfg.fixed_width),
            mode="bilinear",
            align_corners=False,
        )

        output["image"] = control
        output["controlnet_conditioning_scale"] = self.cfg.controlnet_conditioning_scale
        return output
