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
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import get_mvp_matrix, get_ray_directions, get_rays
from threestudio.utils.typing import *
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from gsstudio.data.utils.camera_loader import CameraLoader


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
        # index = torch.randint(0, self.n_frames, (1,)).item()
        dataroot = self.cfg.dataroot
        index = torch.randint(0, self.n_frames, (1,)).item()
        output = {
            **self.cameras.get_index(index, is_batch=True),
            **self.images.get_index(index, is_batch=True),
        }
        return output

        # t0_index = torch.randint(0, len(self.frames_t0), (1,)).item()
        # index = self.frames_t0[t0_index]
        if not self.cfg.online_load_image:
            frame_img = self.frames_img[index : index + 1]
            rays_o = self.rays_o[index : index + 1]
            rays_d = self.rays_d[index : index + 1]
            if len(self.frames_mask_path) > 0:
                mask_img = self.frames_mask[index : index + 1]
            else:
                mask_img = torch.ones_like(frame_img)
        else:
            img = cv2.imread(self.frames_file_path[index])[:, :, ::-1]
            img = cv2.resize(img, (self.frame_w, self.frame_h))
            if len(self.frames_mask_path) > 0:
                mask = cv2.imread(self.frames_mask_path[index])
                mask = cv2.resize(mask, (self.frame_w, self.frame_h))
                mask_img: Float[Tensor, "H W 3"] = (
                    torch.FloatTensor(mask).unsqueeze(0) / 255
                )

            frame_img: Float[Tensor, "H W 3"] = (
                torch.FloatTensor(img).unsqueeze(0) / 255
            )
            intrinsic = self.frames_intrinsic[index]
            frame_direction = get_ray_directions(
                self.frame_h,
                self.frame_w,
                (intrinsic[0, 0], intrinsic[1, 1]),
                (intrinsic[0, 2], intrinsic[1, 2]),
                use_pixel_centers=False,
            ).unsqueeze(0)
            rays_o, rays_d = get_rays(
                frame_direction, self.frames_c2w[index : index + 1], keepdim=True
            )

        return_dict = {
            "index": index,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": self.mvp_mtx[index : index + 1],
            "proj": self.frames_proj[index : index + 1],
            "c2w": self.frames_c2w[index : index + 1],
            "camera_positions": self.frames_position[index : index + 1],
            "light_positions": self.light_positions[index : index + 1],
            "gt_rgb": frame_img,
            "height": self.frame_h,
            "width": self.frame_w,
            "moment": self.frames_moment[index : index + 1],
            "file_path": self.frames_file_path[index],
        }
        if len(self.frames_mask_path) > 0:
            return_dict.update(
                {
                    "frame_mask": mask_img,
                }
            )
        if len(self.frames_bbox) > 0:
            return_dict.update(
                {
                    "frame_bbox": self.frames_bbox[index : index + 1],
                }
            )
        return return_dict


class DynamicMultiviewDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: DynamicMultiviewsDataModuleConfig = cfg

        assert self.cfg.eval_batch_size == 1
        scale = self.cfg.eval_downsample_resolution

        camera_dict = json.load(
            open(os.path.join(self.cfg.dataroot, "transforms.json"), "r")
        )
        assert camera_dict["camera_model"] == "OPENCV"

        frames = camera_dict["frames"]
        frames = frames[:: self.cfg.eval_data_interval]
        self.frames_proj = []
        self.frames_c2w = []
        self.frames_position = []
        self.frames_direction = []
        self.frames_img = []
        self.frames_moment = []
        self.frames_bbox = []
        self.frames_intrinsic = []
        self.frames_path = []

        self.frame_w = frames[0]["w"] // scale
        self.frame_h = frames[0]["h"] // scale
        threestudio.info("Loading frames...")
        self.n_frames = len(frames)

        self.c2w_list = []
        for frame in tqdm(frames):
            extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(
                frame["transform_matrix"], dtype=torch.float32
            )
            c2w = extrinsic
            self.c2w_list.append(c2w)
        self.c2w_list = torch.stack(self.c2w_list, dim=0)

        if self.cfg.camera_layout == "around":
            self.c2w_list[:, :3, 3] -= torch.mean(
                self.c2w_list[:, :3, 3], dim=0
            ).unsqueeze(0)
        elif self.cfg.camera_layout == "front":
            assert self.cfg.camera_distance > 0
            self.c2w_list[:, :3, 3] -= torch.mean(
                self.c2w_list[:, :3, 3], dim=0
            ).unsqueeze(0)
            z_vector = torch.zeros(self.c2w_list.shape[0], 3, 1)
            z_vector[:, 2, :] = -1
            rot_z_vector = self.c2w_list[:, :3, :3] @ z_vector
            rot_z_vector = torch.mean(rot_z_vector, dim=0).unsqueeze(0)
            self.c2w_list[:, :3, 3] -= rot_z_vector[:, :, 0] * self.cfg.camera_distance
        elif self.cfg.camera_layout == "default":
            pass
        else:
            raise ValueError(
                f"Unknown camera layout {self.cfg.camera_layout}. Now support only around and front."
            )

        if not (self.cfg.eval_interpolation is None):
            idx0 = self.cfg.eval_interpolation[0]
            idx1 = self.cfg.eval_interpolation[1]
            moment0 = self.cfg.eval_time_interpolation[0]
            moment1 = self.cfg.eval_time_interpolation[1]
            eval_nums = self.cfg.eval_interpolation[2]
            if self.cfg.time_upsample:
                eval_nums *= 2
            self.get_eval_interpolation(frames, idx0, idx1, moment0, moment1, eval_nums)
        else:
            for idx, frame in tqdm(enumerate(frames)):
                intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
                intrinsic[0, 0] = frame["fl_x"] / scale
                intrinsic[1, 1] = frame["fl_y"] / scale
                intrinsic[0, 2] = frame["cx"] / scale
                intrinsic[1, 2] = frame["cy"] / scale

                frame_path = os.path.join(self.cfg.dataroot, frame["file_path"])
                img = cv2.imread(frame_path)[:, :, ::-1].copy()
                img = cv2.resize(img, (self.frame_w, self.frame_h))
                img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
                self.frames_img.append(img)

                direction: Float[Tensor, "H W 3"] = get_ray_directions(
                    self.frame_h,
                    self.frame_w,
                    (intrinsic[0, 0], intrinsic[1, 1]),
                    (intrinsic[0, 2], intrinsic[1, 2]),
                    use_pixel_centers=False,
                )

                c2w = self.c2w_list[idx]
                camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

                near = 0.01
                far = 100.0
                K = intrinsic
                proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
                proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)

                moment: Float[Tensor, "1"] = torch.zeros(1)
                if frame.__contains__("moment"):
                    moment[0] = frame["moment"]
                else:
                    moment[0] = 0
                if frame.__contains__("bbox"):
                    self.frames_bbox.append(torch.FloatTensor(frame["bbox"]) / scale)

                self.frames_proj.append(proj)
                self.frames_c2w.append(c2w)
                self.frames_position.append(camera_position)
                self.frames_direction.append(direction)
                self.frames_moment.append(moment)
                self.frames_intrinsic.append(intrinsic)
        threestudio.info("Loaded frames.")

    def get_eval_interpolation(self, frames, idx0, idx1, moment0, moment1, eval_nums):
        scale = self.cfg.eval_downsample_resolution

        frame = frames[idx0]
        intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
        intrinsic[0, 0] = frame["fl_x"] / scale
        intrinsic[1, 1] = frame["fl_y"] / scale
        intrinsic[0, 2] = frame["cx"] / scale
        intrinsic[1, 2] = frame["cy"] / scale
        if frame.__contains__("bbox"):
            bbox0 = torch.FloatTensor(frames[idx0]["bbox"]) / scale
            bbox1 = torch.FloatTensor(frames[idx1]["bbox"]) / scale
        for ratio in np.linspace(0, 1, eval_nums):
            img: Float[Tensor, "H W 3"] = torch.zeros((self.frame_h, self.frame_w, 3))
            self.frames_img.append(img)
            direction: Float[Tensor, "H W 3"] = get_ray_directions(
                self.frame_h,
                self.frame_w,
                (intrinsic[0, 0], intrinsic[1, 1]),
                (intrinsic[0, 2], intrinsic[1, 2]),
                use_pixel_centers=False,
            )

            if self.cfg.sin_interpolation:
                space_ratio = (ratio - 0.5) * math.acos(-1)
                space_ratio = math.sin(space_ratio) * 0.5 + 0.5
            else:
                space_ratio = ratio
            c2w = torch.FloatTensor(
                inter_pose(self.c2w_list[idx0], self.c2w_list[idx1], space_ratio)
            )
            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

            near = 0.1
            far = 1000.0
            proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)

            moment: Float[Tensor, "1"] = torch.zeros(1)
            moment[0] = moment0 * (1 - ratio) + moment1 * ratio

            self.frames_proj.append(proj)
            self.frames_c2w.append(c2w)
            self.frames_position.append(camera_position)
            self.frames_direction.append(direction)
            self.frames_moment.append(moment)
            self.frames_intrinsic.append(intrinsic)
            if frame.__contains__("bbox"):
                self.frames_bbox.append(bbox0 * (1 - ratio) + bbox1 * ratio)

    def __len__(self):
        return self.frames_proj.shape[0]

    def __getitem__(self, index):
        return_dict = {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "proj": self.frames_proj[index],
            "c2w": self.frames_c2w[index],
            "intrinsic": self.frames_intrinsic[index],
            "camera_positions": self.frames_position[index],
            "light_positions": self.light_positions[index],
            "gt_rgb": self.frames_img[index],
            "moment": self.frames_moment[index],
        }
        if len(self.frames_bbox) > 0:
            return_dict.update(
                {
                    "frame_bbox": self.frames_bbox[index],
                }
            )
        if len(self.frames_path) > 0:
            return_dict.update(
                {
                    "frame_path": self.frames_path[index],
                }
            )
        if self.cfg.time_upsample:
            return_dict.update(
                {
                    "time_interval": 1.0 / len(self.frames_img),
                    "need_time_interpolation": index % 2 == 0,
                }
            )
        return return_dict

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.frame_h, "width": self.frame_w})
        return batch


@register("dynamic-multiview-camera-datamodule")
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
