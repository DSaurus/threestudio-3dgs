import math
import random

import torch
import torch.nn.functional as F

from gsstudio.data.utils.camera_utils import CameraOutput
from gsstudio.utils.typing import *


class CameraSampler:
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
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
    world_center: Tuple[float, float, float] = (0, 0, 0)

    batch_uniform_azimuth: bool = True
    rays_d_normalize: bool = True

    def disable_perturb(self):
        self.camera_perturb = 0.0
        self.center_perturb = 0.0
        self.up_perturb = 0.0

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def sample(
        self,
        elevation=None,
        azimuth=None,
        camera_distances=None,
        fovy=None,
        fovx=None,
        cx=None,
        cy=None,
        width=None,
        height=None,
    ):
        if elevation is None:
            if random.random() < 0.5:
                # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
                elevation_deg = (
                    torch.rand(self.batch_size)
                    * (self.elevation_range[1] - self.elevation_range[0])
                    + self.elevation_range[0]
                )
                elevation = elevation_deg * math.pi / 180
            else:
                # otherwise sample uniformly on sphere
                elevation_range_percent = [
                    self.elevation_range[0] / 180.0 * math.pi,
                    self.elevation_range[1] / 180.0 * math.pi,
                ]
                # inverse transform sampling
                elevation = torch.asin(
                    (
                        torch.rand(self.batch_size)
                        * (
                            math.sin(elevation_range_percent[1])
                            - math.sin(elevation_range_percent[0])
                        )
                        + math.sin(elevation_range_percent[0])
                    )
                )
        elevation_deg = elevation / math.pi * 180.0

        if azimuth is None:
            # sample azimuth angles from a uniform distribution bounded by azimuth_range
            azimuth_deg: Float[Tensor, "B"]
            if self.batch_uniform_azimuth:
                # ensures sampled azimuth angles in a batch cover the whole range
                azimuth_deg = (
                    torch.rand(self.batch_size) + torch.arange(self.batch_size)
                ) / self.batch_size * (
                    self.azimuth_range[1] - self.azimuth_range[0]
                ) + self.azimuth_range[
                    0
                ]
            else:
                # simple random sampling
                azimuth_deg = (
                    torch.rand(self.batch_size)
                    * (self.azimuth_range[1] - self.azimuth_range[0])
                    + self.azimuth_range[0]
                )
            azimuth = azimuth_deg * math.pi / 180
        azimuth_deg = azimuth / math.pi * 180.0

        if camera_distances is None:
            # sample distances from a uniform distribution bounded by distance_range
            camera_distances: Float[Tensor, "B"] = (
                torch.rand(self.batch_size)
                * (self.camera_distance_range[1] - self.camera_distance_range[0])
                + self.camera_distance_range[0]
            )

        world_center: Float[Tensor, "B 3"] = (
            torch.as_tensor(self.world_center).reshape(1, 3).repeat(self.batch_size, 1)
        )
        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y down, z right
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = (
            torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    -camera_distances * torch.sin(elevation),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                ],
                dim=-1,
            )
            + world_center
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions) + world_center

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(self.batch_size, 3) * 2 * self.camera_perturb
            - self.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.center_perturb
        )
        center = center + center_perturb

        if fovy is None:
            # sample fovs from a uniform distribution bounded by fov_range
            fovy_deg: Float[Tensor, "B"] = (
                torch.rand(self.batch_size) * (self.fovy_range[1] - self.fovy_range[0])
                + self.fovy_range[0]
            )
            fovy = fovy_deg * math.pi / 180

        if height is None:
            height = self.height

        if width is None:
            width = self.width

        if fovx is None:
            f0 = height / (2 * torch.tan(fovy / 2))
            fovx = 2 * torch.atan(width / (2 * f0))

        if cx is None:
            cx = torch.zeros_like(fovx)

        if cy is None:
            cy = torch.zeros_like(fovy)

        return CameraOutput(
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth_deg,
            camera_positions=camera_positions,
            camera_distances=camera_distances,
            camera_centers=center,
            fovx=fovx,
            fovy=fovy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
        )


class CameraInterpolation:
    def __init__(self, camera_sampler: CameraSampler):
        self.camera_sampler = camera_sampler

    def sample(self):
        pass
