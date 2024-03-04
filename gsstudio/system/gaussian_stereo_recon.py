import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import gsstudio
from gsstudio.loss.gaussian_loss import gaussian_loss
from gsstudio.loss.general_loss import tv_loss
from gsstudio.representation.base.gaussian import BasicPointCloud
from gsstudio.system.gaussian_base import GaussianBaseSystem
from gsstudio.utils.config import parse_optimizer
from gsstudio.utils.typing import *


@gsstudio.register("gaussian-splatting-stereo-reconstruction-system")
class GaussianSplatting(GaussianBaseSystem):
    @dataclass
    class Config(GaussianBaseSystem.Config):
        visualize_samples: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        out = self(batch)

        guidance_inp = out["comp_rgb"]
        guidance_out = {"loss_rgb": gaussian_loss(guidance_inp, batch["image"])}
        loss_rgb = 0.0
        loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss_rgb += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )
        # if self.cfg.loss["lambda_mask"] > 0.0:
        #     loss_mask = F.l1_loss(out["comp_mask"], batch["mask"].permute(0, 2, 3, 1))
        #     loss += self.C(self.cfg.loss["lambda_mask"]) * loss_mask
        if self.cfg.loss["lambda_scales"] > 0.0:
            scale_mean = torch.mean(self.geometry.get_scaling)
            self.log(f"train/scales", scale_mean)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_mean

        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_loss"] > 0.0
            and batch.__contains__("depth")
        ):
            loss_depth = self.C(self.cfg.loss["lambda_depth_loss"]) * F.l1_loss(
                out["comp_depth"], batch["depth"]
            )
            self.log(f"train/loss_depth", loss_depth)
            loss += loss_depth

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss_rgb.backward(retain_graph=True)
        self.gaussian_update(out)
        if loss > 0:
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": loss_rgb}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": batch["image"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "image" in batch
                else []
            ),
            name="validation_step",
            step=self.global_step,
        )
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": batch["image"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "image" in batch
                else []
            ),
            name="test_step",
            step=self.global_step,
        )
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.global_step,
        )
