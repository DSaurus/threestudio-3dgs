from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import gsstudio
from gsstudio.diffusion.pipelines.zero123 import Zero123Pipeline
from gsstudio.utils.typing import *

from .stable_diffusion_guidance import DiffusersStableDiffusionGuidance


def prepare_latents(self, *args, **kwargs):
    return self.prepared_latents


@gsstudio.register("diffusers-stable-zero123-guidance")
class DiffusersStableDiffusionGuidance(DiffusersStableDiffusionGuidance):
    @dataclass
    class Config(DiffusersStableDiffusionGuidance.Config):
        fixed_width: int = 256
        fixed_height: int = 256
        fixed_latent_width: int = 32
        fixed_latent_height: int = 32

        use_stable_zero123: bool = True

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.pipe.vae.eval()
        for p in self.pipe.vae.parameters():
            p.requires_grad_(False)
        self.use_stable_zero123 = self.cfg.use_stable_zero123

    def create_pipe(self):
        HookPipeline = type(
            "HookPipeline",
            (Zero123Pipeline,),
            {"prepare_latents": prepare_latents},
        )
        self.pipe = HookPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=self.weights_dtype,
        ).to(self.device)

    def prepare_text_embeddings(self, prompt_utils, **kwargs):
        return {}

    def get_img_embeds(self, x):
        # x: image tensor in [0, 1]
        x = F.interpolate(x, (256, 256), mode="bilinear", align_corners=False)
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.pipe.feature_extractor(
            images=x_pil, return_tensors="pt"
        ).pixel_values.to(x)
        c = self.pipe.image_encoder(
            x_clip.to(self.pipe.image_encoder.dtype)
        ).image_embeds
        v = self.encode_images(x) / self.pipe.vae.config.scaling_factor
        return c, v

    def get_cam_embeddings(self, elevation, azimuth, radius, default_elevation=0):
        if self.use_stable_zero123:
            cond_elevation = torch.zeros_like(elevation)
            cond_elevation[...] = 90 + default_elevation
            T = torch.stack(
                [
                    torch.deg2rad(-elevation),
                    torch.sin(torch.deg2rad(azimuth)),
                    torch.cos(torch.deg2rad(azimuth)),
                    torch.deg2rad(cond_elevation),
                ],
                dim=-1,
            )
        else:
            # original zero123 camera embedding
            T = torch.stack(
                [
                    torch.deg2rad(-elevation),
                    torch.sin(torch.deg2rad(azimuth)),
                    torch.cos(torch.deg2rad(azimuth)),
                    radius,
                ],
                dim=-1,
            )
        T = T.unsqueeze(1).to(elevation)  # [8, 1, 4]
        return T

    def prepare_other_conditions(self, **kwargs):
        output = super().prepare_other_conditions(**kwargs)
        # output.update({"output_type": "latent"})

        with torch.no_grad():
            if "default_elevation" in kwargs:
                default_elevation = kwargs["default_elevation"]
            else:
                default_elevation = 0
            elevation = kwargs["elevation"]
            azimuth = kwargs["azimuth"]
            radius = kwargs["camera_distances"]
            batch_size = elevation.shape[0]

            T = self.get_cam_embeddings(elevation, azimuth, radius, default_elevation)

            img_embedding, img_latents = self.get_img_embeds(
                kwargs["cond_rgb"].permute(0, 3, 1, 2)
            )
            cc_emb = torch.cat([img_embedding.repeat(batch_size, 1, 1), T], dim=-1)
            cc_emb = self.pipe.clip_camera_projection(cc_emb.to(img_embedding))

            vae_emb = img_latents.repeat(batch_size, 1, 1, 1).to(img_embedding)

            output.update({"image_camera_embeddings": cc_emb, "image_latents": vae_emb})

        return output
