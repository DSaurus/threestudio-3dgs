import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

import gsstudio
from gsstudio.utils.typing import *

from .utils import samples_to_noise


def csd_set_timesteps(self, *args, **kwargs):
    self.timesteps = self.csd_timesteps
    self.num_inference_steps = 1


@gsstudio.register("csd-sampler")
class CSDSampler:
    init_csd: bool = False

    csd_step_s: int = 200
    csd_step_t: int = 50

    def init_csd_sampler(
        self,
    ):
        CSDScheduler = type(
            "CSDScheduler", (DDIMScheduler,), {"set_timesteps": csd_set_timesteps}
        )
        self.csd_scheduler = CSDScheduler.from_config(self.pipe.scheduler.config)
        self.csd_scheduler.config.variance_type = "none"
        self.init_csd = True

        self.alphas_cumprod = self.csd_scheduler.alphas_cumprod.to(self.device)

    def one_step(self, latents, noise, timesteps, **kwargs):
        self.csd_scheduler.csd_timesteps = timesteps
        noisy_latents = self.csd_scheduler.add_noise(latents, noise, timesteps)
        with torch.inference_mode():
            self.pipe.prepared_latents = noisy_latents.to(self.pipe.unet.dtype)
            self.pipe.scheduler = self.csd_scheduler
            pred_latents = self.pipe(**kwargs).images
        pred_noise = samples_to_noise(
            pred_latents, noisy_latents, timesteps, self.csd_scheduler.alphas_cumprod
        )
        return pred_latents, pred_noise

    def compute_grad_csd(
        self, latents: Float[Tensor, "B 4 64 64"], t: Int[Tensor, "B"], **kwargs
    ):
        if not self.init_csd:
            self.init_csd_sampler()
        original_scheduler = self.pipe.scheduler
        batch_size = latents.shape[0]
        with torch.inference_mode():
            timesteps = t[:1]
            timesteps_s = timesteps - self.csd_step_t
            noise = torch.randn_like(latents)
            guidance_scale = kwargs["guidance_scale"]
            # prompt_embeds = kwargs["prompt_embeds"]
            # negative_prompt_embeds = kwargs["negative_prompt_embeds"]

            kwargs["guidance_scale"] = 1.5
            # kwargs["prompt_embeds"] = negative_prompt_embeds
            latents_s = latents
            pred_noise_s = noise
            # for step in range(self.ism_step_s, timesteps_s.item(), self.ism_step_s):
            #     step = torch.Tensor([step]).long().to(timesteps.device)
            #     latents_s, pred_noise_s = self.one_step(
            #         latents_s, pred_noise_s, step, **kwargs
            #     )
            latents_s, pred_noise_s = self.one_step(
                latents_s, pred_noise_s, timesteps_s, **kwargs
            )

            kwargs["guidance_scale"] = 2.0
            # kwargs["prompt_embeds"] = prompt_embeds
            latents_t, pred_noise_t = self.one_step(
                latents_s, pred_noise_s, timesteps, **kwargs
            )

            w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
            csd_score = w * (pred_noise_t - pred_noise_s)
            target = latents - csd_score

        loss_csd = (
            0.5 * F.mse_loss(latents, target.clone(), reduction="sum") / batch_size
        )

        self.pipe.scheduler = original_scheduler
        return loss_csd
