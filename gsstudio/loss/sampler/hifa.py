import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from threestudio.utils.typing import *

import gsstudio


def hifa_set_timesteps(self, *args, **kwargs):
    self.timesteps = self.hifa_timesteps
    self.num_inference_steps = 1


@gsstudio.register("hifa-sampler")
class HIFASampler:
    init_hifa: bool = False

    def init_hifa_sampler(
        self,
    ):
        HIFAScheduler = type(
            "HIFAScheduler", (DDIMScheduler,), {"set_timesteps": hifa_set_timesteps}
        )
        self.hifa_scheduler = HIFAScheduler.from_config(self.pipe.scheduler.config)
        self.hifa_scheduler.config.variance_type = "none"
        self.init_hifa = True
        self.alphas_cumprod = self.hifa_scheduler.alphas_cumprod.to(self.device)

    def compute_grad_hifa(
        self, latents: Float[Tensor, "B 4 64 64"], t: Int[Tensor, "B"], **kwargs
    ):
        if not self.init_hifa:
            self.init_hifa_sampler()
        original_scheduler = self.pipe.scheduler
        batch_size = latents.shape[0]

        timesteps = t[:1]
        t[:] = t[:1]
        self.hifa_scheduler.hifa_timesteps = timesteps

        noise = torch.randn_like(latents)
        noisy_latents = self.hifa_scheduler.add_noise(latents, noise, t)
        with torch.inference_mode():
            self.pipe.prepared_latents = noisy_latents.to(self.pipe.unet.dtype)
            self.pipe.scheduler = self.hifa_scheduler
            pred_latents = self.pipe(**kwargs).images
            pred_img = self.decode_latents(pred_latents)
            w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
            alpha = (self.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1)
            sigma = ((1 - self.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1)

        input_rgb = kwargs["input_rgb"].permute(0, 3, 1, 2)
        input_rgb = F.interpolate(
            input_rgb, (pred_img.shape[2], pred_img.shape[3]), mode="bilinear"
        )

        with torch.inference_mode():
            hifa_grad = w * (input_rgb - pred_img) * alpha / sigma
            target = input_rgb - hifa_grad

        loss_hifa = (
            0.5 * F.mse_loss(input_rgb, target.clone(), reduction="sum") / batch_size
        )

        self.pipe.scheduler = original_scheduler
        return loss_hifa
