import os
from dataclasses import dataclass, field
import threestudio
import torch
from torch.cuda.amp import autocast
import numpy as np
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from threestudio.utils.loss import tv_loss
from ..geometry.gaussian import BasicPointCloud, Camera


@threestudio.register("gaussian-splatting-mvdream-system")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False

        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.requires_grad_(False)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
    
    def configure_optimizers(self):
        optim = self.geometry.optimizer
        ret = {
            "optimizer": optim,
        }
        return ret

    def on_load_checkpoint(self, checkpoint):
        num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()

        for k in list(checkpoint["state_dict"].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {
            "guidance." + k: v for (k, v) in self.guidance.state_dict().items()
        }
        checkpoint["state_dict"] = {**checkpoint["state_dict"], **guidance_state_dict}
        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint["state_dict"].keys()):
            if k.startswith("guidance."):
                checkpoint["state_dict"].pop(k)
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        lr_max_step = self.geometry.cfg.position_lr_max_steps
        scale_lr_max_steps = self.geometry.cfg.scale_lr_max_steps

        if self.global_step < lr_max_step:
            self.geometry.update_xyz_learning_rate(self.global_step)

        if self.global_step < scale_lr_max_steps:
            self.geometry.update_scale_learning_rate(self.global_step)

        bs = batch["c2w"].shape[0]
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        for batch_idx in range(bs):
            fovy = batch["fovy"][batch_idx]
            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=batch["c2w"][batch_idx], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
            )

            # import pdb; pdb.set_trace()
            viewpoint_cam = Camera(
                FoVx=fovy,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )

            with autocast(enabled=False):
                render_pkg = self.renderer(
                    viewpoint_cam,
                    self.background_tensor,
                )
                renders.append(render_pkg["render"])
                viewspace_points.append(render_pkg["viewspace_points"])
                visibility_filters.append(render_pkg["visibility_filter"])
                radiis.append(render_pkg["radii"])
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # render_out = {
        #     "comp_rgb": image,
        # }
        outputs = {
            "render": torch.stack(renders, dim=0),
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
        }
        return outputs

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        out = self(batch)

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["render"].permute(0, 2, 3, 1)
        # import pdb; pdb.set_trace()
        viewspace_point_tensor = out["viewspace_points"]
        guidance_out = self.guidance(
            guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False
        )

        loss_sds = 0.0
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
                loss_sds += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        xyz_mean = None
        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            loss_position = xyz_mean.mean()
            self.log(f"train/loss_position", loss_position)
            loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

        if self.cfg.loss["lambda_opacity"] > 0.0:
            if xyz_mean is None:
                xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            loss_opacity = (
                xyz_mean.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).mean()
            self.log(f"train/loss_opacity", loss_opacity)
            loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_scales"] > 0.0:
            scale_sum = torch.sum(self.geometry.get_scaling)
            self.log(f"train/scales", scale_sum)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        if self.cfg.loss["lambda_tv_loss"] > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(out["render"])
            self.log(f"train/loss_tv", loss_tv)
            loss += loss_tv

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss_sds.backward(retain_graph=True)
        iteration = self.global_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
        if loss > 0:
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": loss_sds}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        save_img = out["render"].permute(0, 2, 3, 1)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": save_img[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "render" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        save_img = out["render"].permute(0, 2, 3, 1)
        print(save_img.shape)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": save_img[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "render" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )