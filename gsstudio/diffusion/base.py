from dataclasses import dataclass, field

import torch
from diffusers.utils.import_utils import is_xformers_available

import gsstudio
from gsstudio.diffusion.prompt_processors.base import PromptProcessorOutput
from gsstudio.utils.base import BaseObject
from gsstudio.utils.config import C
from gsstudio.utils.misc import cleanup, parse_version
from gsstudio.utils.typing import *


@gsstudio.register("diffusers-guidance")
class DiffusersGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        use_sjc: bool = False
        weighting_strategy: str = "sds"

        view_dependent_prompting: bool = True

    cfg: Config

    def configure(self) -> None:
        gsstudio.info(f"Loading Diffuser Pipeline ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        if not hasattr(self, "pipe_kwargs"):
            self.pipe_kwargs = {}

        self.pipe_kwargs.update(
            {
                "safety_checker": None,
                "feature_extractor": None,
                "requires_safety_checker": False,
                "torch_dtype": self.weights_dtype,
            }
        )

        self.create_pipe()

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                gsstudio.info("PyTorch2.0 uses memory efficient attention by default.")
            elif not is_xformers_available():
                gsstudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        cleanup()
        self.pipe.unet.eval()
        for p in self.pipe.unet.parameters():
            p.requires_grad_(False)

        self.pipe.set_progress_bar_config(disable=True)
        self.num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        gsstudio.info(f"Loaded Diffuser Pipeline!")

    def create_pipe(self):
        raise NotImplementedError

    def prepare_latents(
        self, rgb: Float[Tensor, "B H W C"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        raise NotImplementedError

    def prepare_text_embeddings(self, prompt_utils, **kwargs):
        if (
            "elevation" in kwargs
            and "azimuth" in kwargs
            and "camera_distances" in kwargs
        ):
            elevation = kwargs["elevation"]
            azimuth = kwargs["azimuth"]
            camera_distances = kwargs["camera_distances"]
            view_dependent_prompting = self.cfg.view_dependent_prompting
        else:
            elevation = torch.zeros(1).to(self.device)
            azimuth = torch.zeros(1).to(self.device)
            camera_distances = torch.zeros(1).to(self.device)
            view_dependent_prompting = False

        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting
        )
        batch_size = text_embeddings.shape[0] // 2
        return {
            "prompt_embeds": text_embeddings[:batch_size],
            "negative_prompt_embeds": text_embeddings[batch_size:],
        }

    def prepare_other_conditions(self, **kwargs):
        return {"guidance_scale": self.cfg.guidance_scale}

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        latents = self.prepare_latents(rgb, rgb_as_latents=rgb_as_latents)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        input_cond = {
            "input_rgb": rgb,
        }

        text_cond = self.prepare_text_embeddings(prompt_utils, **kwargs)

        other_cond = self.prepare_other_conditions(**kwargs)

        merged_cond = {**text_cond, **other_cond, **input_cond}

        guidance_out = {
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if hasattr(self, "init_sds"):
            loss_sds = self.compute_grad_sds(latents, t, **merged_cond)
            guidance_out["loss_sds"] = loss_sds

        if hasattr(self, "init_lcm"):
            loss_lcm = self.compute_grad_lcm(latents, t, **merged_cond)
            guidance_out["loss_lcm"] = loss_lcm

        if hasattr(self, "init_ism"):
            loss_ism = self.compute_grad_ism(latents, t, **merged_cond)
            guidance_out["loss_ism"] = loss_ism

        if hasattr(self, "init_hifa"):
            loss_hifa = self.compute_grad_hifa(latents, t, **merged_cond)
            guidance_out["loss_hifa"] = loss_hifa

        if hasattr(self, "init_csd"):
            loss_csd = self.compute_grad_csd(latents, t, **merged_cond)
            guidance_out["loss_csd"] = loss_csd

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
