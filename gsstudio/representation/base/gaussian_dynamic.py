import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

import gsstudio
from gsstudio.representation.base.dynamic import DynamicBaseModel
from gsstudio.representation.base.gaussian import GaussianBaseModel
from gsstudio.representation.base.utils import BasicPointCloud
from gsstudio.utils.config import C
from gsstudio.utils.typing import *


@gsstudio.register("gaussian-splatting-dynamic")
class GaussianDynamicModel(GaussianBaseModel, DynamicBaseModel):
    @dataclass
    class Config(GaussianBaseModel.Config):
        num_frames: int = 10
        delta_pos_lr: Any = 0.001
        delta_rot_lr: Any = 0.0001

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self._delta_xyz = torch.empty(0)
        self._delta_rot = torch.empty(0)
        self.time_index = 0

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        super().create_from_pcd(pcd, spatial_lr_scale)
        self._delta_xyz = nn.Parameter(
            torch.zeros(self.cfg.num_frames, self._xyz.shape[0], 3).requires_grad_(True)
        )
        self._delta_rot = nn.Parameter(
            torch.zeros(self.cfg.num_frames, self._xyz.shape[0], 3).requires_grad_(True)
        )

    def to(self, device="cpu"):
        super().to(device)
        self._delta_xyz = self._delta_xyz.to(device)
        self._delta_rot = self._delta_rot.to(device)

    def training_setup(self):
        super().training_setup()
        l = self.optimize_list
        training_args = self.cfg
        l.append(
            {
                "params": [self._delta_xyz],
                "lr": C(training_args.delta_pos_lr, 0, 0),
                "name": "normal",
            },
        )
        l.append(
            {
                "params": [self._delta_rot],
                "lr": C(training_args.delta_rot_lr, 0, 0),
                "name": "normal",
            },
        )

    @property
    def get_rotation(self):
        return self.rotation_activation(
            self._rotation + self._delta_rot[self.time_index]
        )

    @property
    def get_xyz(self):
        return self._xyz + self._delta_xyz[self.time_index]
