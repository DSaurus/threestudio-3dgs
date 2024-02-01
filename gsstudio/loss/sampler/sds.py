import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

import gsstudio
from gsstudio.utils.typing import *

from .utils import samples_to_noise


def sds_set_timesteps(self, *args, **kwargs):
    self.timesteps = self.sds_timesteps
    self.num_inference_steps = 1


@gsstudio.register("sds-sampler")
class SDSSampler:
    init_sds: bool = False

    def init_sds_sampler(
        self,
    ):
        SDSScheduler = type(
            "SDSScheduler", (DDIMScheduler,), {"set_timesteps": sds_set_timesteps}
        )
        self.sds_scheduler = SDSScheduler.from_config(self.pipe.scheduler.config)
        self.sds_scheduler.config.variance_type = "none"
        self.init_sds = True
        self.alphas_cumprod = self.sds_scheduler.alphas_cumprod.to(self.device)

    def compute_grad_sds(
        self, latents: Float[Tensor, "B 4 64 64"], t: Int[Tensor, "B"], **kwargs
    ):
        if not self.init_sds:
            self.init_sds_sampler()
        original_scheduler = self.pipe.scheduler
        batch_size = latents.shape[0]

        timesteps = t[:1]
        t[:] = t[:1]
        self.sds_scheduler.sds_timesteps = timesteps

        noise = torch.randn_like(latents)
        noisy_latents = self.sds_scheduler.add_noise(latents, noise, t)
        with torch.inference_mode():
            self.pipe.prepared_latents = noisy_latents.to(self.pipe.unet.dtype)
            self.pipe.scheduler = self.sds_scheduler
            pred_latents = self.pipe(**kwargs).images
            if type(pred_latents) != torch.Tensor:
                pred_images = torch.from_numpy(pred_latents)
                pred_latents = self.prepare_latents(pred_images)
            pred_noise = samples_to_noise(
                pred_latents,
                noisy_latents,
                timesteps,
                self.sds_scheduler.alphas_cumprod,
            )
            w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
            sds_grad = w * (pred_noise - noise)
            target = latents - sds_grad

        loss_sds = (
            0.5 * F.mse_loss(latents, target.clone(), reduction="sum") / batch_size
        )

        self.pipe.scheduler = original_scheduler
        return loss_sds
