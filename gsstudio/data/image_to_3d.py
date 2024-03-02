import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import gsstudio
from gsstudio.data.text_to_3d import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from gsstudio.data.utils.camera_generation import CameraSampler
from gsstudio.data.utils.camera_utils import matrix2rays, samples2matrix
from gsstudio.data.utils.image_utils import ImageOutput
from gsstudio.data.utils.light_generation import LightSampler
from gsstudio.utils.base import Updateable
from gsstudio.utils.config import parse_structured
from gsstudio.utils.misc import get_rank
from gsstudio.utils.typing import *


@dataclass
class SingleImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False

    rays_d_normalize: bool = True


class SingleImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: SingleImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        self.camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])
        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.default_fovy_deg
        )

        self.elevation = elevation_deg * math.pi / 180
        self.azimuth = azimuth_deg * math.pi / 180
        self.fovy = fovy_deg * math.pi / 180

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                gsstudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]

        self.camera_sampler = CameraSampler()
        self.camera_sampler.batch_size = 1
        self.camera_sampler.disable_perturb()

        self.light_sampler = LightSampler()

        self.image = ImageOutput()
        self.image.white_background = True
        assert os.path.exists(
            self.cfg.image_path
        ), f"Could not find image {self.cfg.image_path}!"
        self.image.frame_image_path = [self.cfg.image_path]
        gsstudio.info(
            f"single image dataset: load image {self.cfg.image_path} {self.height}x{self.width}"
        )
        if self.cfg.requires_depth:
            depth_path = self.cfg.image_path.replace("_rgba.png", "_depth.png")
            assert os.path.exists(depth_path)
            self.image.frame_depth_path = [depth_path]
        if self.cfg.requires_normal:
            normal_path = self.cfg.image_path.replace("_rgba.png", "_normal.png")
            assert os.path.exists(normal_path)
            self.image.frame_normal_path = [normal_path]
        self.image.key_mapping["image"] = "rgb"
        self.image.key_mapping["depth"] = "ref_depth"
        self.image.key_mapping["normal"] = "ref_normal"

        self.set_camera()

        self.image.width = self.width
        self.image.height = self.height
        self.image.load_image()

        self.prev_height = self.height

    def set_camera(self):
        # get directions by dividing directions_unit_focal by focal length
        self.camera_out = self.camera_sampler.sample(
            elevation=self.elevation,
            azimuth=self.azimuth,
            camera_distances=self.camera_distance,
            fovy=self.fovy,
            height=self.height,
            width=self.width,
        )
        (
            self.camera_out.c2w,
            self.camera_out.intrinsic,
            self.camera_out.proj_mtx,
        ) = samples2matrix(**self.camera_out.to_dict())
        self.camera_out.rays_o, self.camera_out.rays_d = matrix2rays(
            normalize=self.cfg.rays_d_normalize, **self.camera_out.to_dict()
        )

        self.light_out = self.light_sampler.sample(self.camera_out.camera_positions[:1])

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        gsstudio.debug(f"Training height: {self.height}, width: {self.width}")

        self.set_camera()

        self.image.width = self.width
        self.image.height = self.height
        self.image.load_image()


class SingleImageIterableDataset(IterableDataset, SingleImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def collate(self, batch) -> Dict[str, Any]:
        batch = {
            **self.camera_out.get_index(0, is_batch=True).to_dict(),
            **self.image.get_index(0, is_batch=True).to_dict(),
            **self.light_out.to_dict(),
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class SingleImageDataset(Dataset, SingleImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        return self.random_pose_generator[index]


@gsstudio.register("single-image-sampler-datamodule")
class SingleImageDataModule(pl.LightningDataModule):
    cfg: SingleImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SingleImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SingleImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = SingleImageDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
