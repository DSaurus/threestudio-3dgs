import json
import math
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pytorch_lightning as pl
import threestudio
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import get_mvp_matrix, get_ray_directions, get_rays
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

import gsstudio
from gsstudio.data.utils.camera_loader import CameraLoader
from gsstudio.utils.typing import *


def inter_pose(pose_0, pose_1, ratio):
    pose_0 = pose_0.detach().cpu().numpy()
    pose_1 = pose_1.detach().cpu().numpy()
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    return pose


@dataclass
class DynamicMultiviewsDataModuleConfig:
    dataroot: str = ""
    train_downsample_resolution: int = 4
    eval_downsample_resolution: int = 4
    train_data_interval: int = 1
    eval_data_interval: int = 1
    batch_size: int = 1
    eval_batch_size: int = 1
    camera_layout: str = "around"
    camera_distance: float = -1
    eval_interpolation: Optional[Tuple[int, int, int]] = None  # (0, 1, 30)
    eval_time_interpolation: Optional[Tuple[float, float]] = None  # (t0, t1)
    time_upsample: bool = False
    sin_interpolation: bool = False

    initial_t0_step: int = 0
    online_load_image: bool = False

    build_dataset_root: str = ""
    build_image_name: str = "images"
    build_json: bool = True

    edit: bool = False
    close_interval: bool = False

    max_train_nums: int = -1
    max_eval_nums: int = -1


class DynamicMultiviewIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: DynamicMultiviewsDataModuleConfig = cfg

        assert self.cfg.batch_size == 1
        if self.cfg.edit:
            assert self.cfg.online_load_image == True
        scale = self.cfg.train_downsample_resolution

        self.camera_loader = CameraLoader(
            self.cfg.dataroot,
            max_nums=self.cfg.max_train_nums,
            interval=self.cfg.train_data_interval,
            scale=scale,
            offline_load=not self.cfg.online_load_image,
        )
        self.camera_loader.set_layout(self.cfg.camera_layout, self.cfg.camera_distance)
        self.cameras = self.camera_loader.cameras
        self.images = self.camera_loader.images

    def __iter__(self):
        while True:
            yield {}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.step = global_step

    def collate(self, batch):
        index = torch.randint(0, self.cameras.c2w.shape[0], (1,)).item()
        camera = self.cameras.get_index(index, is_batch=True)
        image = self.images.get_index(index, is_batch=True)
        if self.cfg.online_load_image:
            camera.gen_rays()
            image.load_image()
        output = {
            **camera.to_dict(),
            **image.to_dict(),
        }
        return output


class DynamicMultiviewDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: DynamicMultiviewsDataModuleConfig = cfg

        assert self.cfg.eval_batch_size == 1
        scale = self.cfg.eval_downsample_resolution

        self.camera_loader = CameraLoader(
            self.cfg.dataroot,
            interval=self.cfg.eval_data_interval,
            max_nums=self.cfg.max_eval_nums,
            scale=scale,
            offline_load=not self.cfg.online_load_image,
        )
        self.camera_loader.set_layout(self.cfg.camera_layout, self.cfg.camera_distance)
        self.cameras = self.camera_loader.cameras
        self.images = self.camera_loader.images

    def __len__(self):
        return self.cameras.c2w.shape[0]

    def __getitem__(self, index):
        camera = self.cameras.get_index(index, is_batch=True)
        image = self.images.get_index(index, is_batch=True)
        if self.cfg.online_load_image:
            camera.gen_rays()
            image.load_image()
        output = {
            **camera.get_index(0, is_batch=False).to_dict(),
            **image.get_index(0, is_batch=False).to_dict(),
            "index": index,
        }
        return output

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cameras.height, "width": self.cameras.width})
        return batch


@gsstudio.register("dynamic-multiview-camera-sampler-datamodule")
class DynamicMultiviewDataModule(pl.LightningDataModule):
    cfg: DynamicMultiviewsDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(DynamicMultiviewsDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = DynamicMultiviewIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DynamicMultiviewDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = DynamicMultiviewDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        if (
            hasattr(dataset.cfg, "online_load_image")
            and dataset.cfg.online_load_image == True
        ):
            return DataLoader(
                dataset,
                num_workers=8,  # type: ignore
                batch_size=batch_size,
                collate_fn=collate_fn,
            )
        else:
            return DataLoader(
                dataset,
                num_workers=0,  # type: ignore
                batch_size=batch_size,
                collate_fn=collate_fn,
            )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=1, collate_fn=self.train_dataset.collate
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
