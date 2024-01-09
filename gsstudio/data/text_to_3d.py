import bisect
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import gsstudio
from gsstudio.data.utils.camera_generation import CameraSampler
from gsstudio.data.utils.camera_utils import matrix2rays, samples2matrix
from gsstudio.data.utils.light_generation import LightSampler
from gsstudio.utils.base import Updateable
from gsstudio.utils.config import parse_structured
from gsstudio.utils.typing import *


@dataclass
class RandomCameraDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    batch_uniform_azimuth: bool = True

    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    light_sample_strategy: str = "dreamfusion"

    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy

    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0

    n_val_views: int = 1
    n_test_views: int = 120

    rays_d_normalize: bool = True


class RandomCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.camera_sampler = CameraSampler()
        self.light_sampler = LightSampler()
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                gsstudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.camera_sampler.height: int = self.heights[0]
        self.camera_sampler.width: int = self.widths[0]
        self.camera_sampler.batch_size: int = self.batch_sizes[0]
        self.camera_sampler.elevation_range = self.cfg.elevation_range
        self.camera_sampler.azimuth_range = self.cfg.azimuth_range
        self.camera_sampler.camera_distance_range = self.cfg.camera_distance_range
        self.camera_sampler.fovy_range = self.cfg.fovy_range

        self.light_sampler.light_distance_range = self.cfg.light_distance_range
        self.light_sampler.light_position_perturb = self.cfg.light_position_perturb
        self.light_sampler.light_sample_strategy = self.cfg.light_sample_strategy

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.camera_sampler.height = self.heights[size_ind]
        self.camera_sampler.width = self.widths[size_ind]
        self.camera_sampler.batch_size = self.batch_sizes[size_ind]
        gsstudio.debug(
            f"Training height: {self.camera_sampler.height}, width: {self.camera_sampler.width}, batch_size: {self.camera_sampler.batch_size}"
        )
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        self.camera_sampler.elevation_range = [
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        ]
        self.camera_sampler.azimuth_range = [
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        ]
        # self.camera_distance_range = [
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[0],
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[1],
        # ]
        # self.fovy_range = [
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[0],
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[1],
        # ]

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        camera_out = self.camera_sampler.sample()
        camera_out.c2w, camera_out.intrinsic, camera_out.proj_mtx = samples2matrix(
            **camera_out.to_dict()
        )
        camera_out.rays_o, camera_out.rays_d = matrix2rays(
            normalize=self.cfg.rays_d_normalize, **camera_out.to_dict()
        )

        light_out = self.light_sampler.sample(camera_out.camera_positions)

        return {**camera_out.to_dict(), **light_out.to_dict()}


class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split
        self.camera_sampler = CameraSampler()
        self.light_sampler = LightSampler()

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
        else:
            azimuth_deg = torch.linspace(0, 360.0, self.n_views)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180

        self.camera_sampler.batch_size = self.n_views
        self.camera_sampler.disable_perturb()
        self.camera_out = self.camera_sampler.sample(
            elevation=elevation,
            azimuth=azimuth,
            camera_distances=camera_distances,
            fovy=fovy,
            height=self.cfg.eval_height,
            width=self.cfg.eval_width,
        )
        (
            self.camera_out.c2w,
            self.camera_out.intrinsic,
            self.camera_out.proj_mtx,
        ) = samples2matrix(**self.camera_out.to_dict())
        self.camera_out.rays_o, self.camera_out.rays_d = matrix2rays(
            normalize=self.cfg.rays_d_normalize, **self.camera_out.to_dict()
        )

        self.light_sampler.light_distance_range = self.cfg.light_distance_range
        self.light_sampler.light_position_perturb = self.cfg.light_position_perturb
        self.light_sampler.light_sample_strategy = self.cfg.light_sample_strategy
        self.light_out = self.light_sampler.sample(
            self.camera_out.camera_positions[:1]
        ).get_index(0)

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        camera_out = self.camera_out.get_index(index)
        camera_out["index"] = index
        return {**camera_out, **self.light_out}

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@gsstudio.register("random-camera-sampler-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
