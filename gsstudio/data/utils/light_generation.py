import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gsstudio.data.utils.data_utils import DataOutput
from gsstudio.utils.typing import *


class LightOutput(DataOutput):
    light_positions: Float[Tensor, "B 3"] = None


class LightSampler:
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    light_sample_strategy: str = "dreamfusion"

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def sample(self, camera_positions):
        # sample light distance from a uniform distribution bounded by light_distance_range
        self.batch_size = camera_positions.shape[0]
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.light_distance_range[1] - self.light_distance_range[0])
            + self.light_distance_range[0]
        )

        if self.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(self.batch_size, 3) * self.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.batch_size) * math.pi * 2 - math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.light_sample_strategy}"
            )

        return LightOutput(light_positions=light_positions)
