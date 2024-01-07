import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from threestudio.utils.typing import *

import gsstudio


def image_set_timesteps(self, *args, **kwargs):
    super(type(self), self).set_timesteps(*args, **kwargs)

    strength = self.image_timesteps.item() / self.config.num_train_timesteps
    init_timestep = min(
        int(self.num_inference_steps * strength), self.num_inference_steps
    )
    t_start = min(
        max(self.num_inference_steps - init_timestep, 0),
        self.num_inference_steps - self.minimal_steps,
    )
    self.timesteps = self.timesteps[t_start:]


@gsstudio.register("image-sampler")
class ImageSampler:
    init_image: bool = False
    image_sampler_step: int = 20

    def init_image_sampler(
        self,
    ):
        ImageScheduler = type(
            "ImageScheduler",
            (DDIMScheduler,),
            {"set_timesteps": image_set_timesteps},
        )
        self.image_scheduler = ImageScheduler.from_config(self.pipe.scheduler.config)
        self.image_scheduler.minimal_steps = 2

        self.init_image = True

    def compute_grad_image(
        self, latents: Float[Tensor, "B 4 64 64"], t: Int[Tensor, "B"], **kwargs
    ):
        if not self.init_image:
            self.init_image_sampler()
        batch_size = latents.shape[0]

        timesteps = t[:1]
        self.image_scheduler.image_timesteps = timesteps
        self.image_scheduler.set_timesteps(num_inference_steps=self.image_sampler_step)
        t = self.image_scheduler.timesteps[:1]

        noise = torch.randn_like(latents)
        noisy_latents = self.image_scheduler.add_noise(latents, noise, t)
        with torch.inference_mode():
            self.pipe.prepared_latents = noisy_latents.to(self.pipe.unet.dtype)
            self.pipe.scheduler = self.image_scheduler
            pred_latents = self.pipe(
                num_inference_steps=self.image_sampler_step, **kwargs
            ).images
            if type(pred_latents) != torch.Tensor:
                pred_images = torch.from_numpy(pred_latents)
                pred_latents = self.prepare_latents(pred_images)
            img = self.decode_latents(pred_latents)
            # import cv2

            # cv2.imwrite(
            #     ".threestudio_cache/test.jpg",
            #     (img.permute(0, 2, 3, 1)[0].detach().cpu().numpy()[:, :, ::-1] * 255),
            # )
            # exit(0)

        return img.permute(0, 2, 3, 1)
