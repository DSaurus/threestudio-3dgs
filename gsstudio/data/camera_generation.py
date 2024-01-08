import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from threestudio.utils.typing import *

from gsstudio.renderer.renderer_utils import (
    get_projection_matrix_advanced,
    get_ray_directions,
    get_rays,
)


def samples2matrix(
    camera_positions, camera_centers, fovx, fovy, width, height, **kwargs
):
    batch_size = camera_positions.shape[0]
    up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        None, :
    ].repeat(batch_size, 1)
    lookat: Float[Tensor, "B 3"] = F.normalize(
        camera_centers - camera_positions, dim=-1
    )
    right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
        [torch.stack([right, -up, lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w: Float[Tensor, "B 4 4"] = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0
    intrinsic: Float[Tensor, "B 3 3"] = torch.zeros(batch_size, 3, 3)
    intrinsic[:, 0, 0] = width / (2 * torch.tan(fovx / 2))
    intrinsic[:, 1, 1] = height / (2 * torch.tan(fovy / 2))
    intrinsic[:, 0, 2] = width / 2
    intrinsic[:, 1, 2] = height / 2
    intrinsic[:, 2, 2] = 1.0

    proj_mtx = get_projection_matrix_advanced(0.01, 100, fovx, fovy)
    return c2w, intrinsic, proj_mtx


def matrix2rays(c2w, intrinsic, height, width, normalize=True, **kwargs):
    rays_direction = get_ray_directions(
        H=height,
        W=width,
        focal=(intrinsic[:, 0, 0], intrinsic[:, 1, 1]),
        principal=(intrinsic[:, 0, 2], intrinsic[:, 1, 2]),
    )
    rays_o, rays_d = get_rays(rays_direction, c2w, normalize=normalize)
    return rays_o, rays_d


def intrinsic2proj_mtx(intrinsic, height, width, **kwargs):
    fovx = 2 * torch.atan(width / (2 * intrinsic[:, 0, 0]))
    fovy = 2 * torch.atan(height / (2 * intrinsic[:, 1, 1]))
    cx = 2 * (intrinsic[:, 0, 2] / width) - 1
    cy = 2 * (intrinsic[:, 1, 2] / height) - 1
    return (
        fovx,
        fovy,
        cx,
        cy,
        get_projection_matrix_advanced(0.01, 100, fovx, fovy, cx, cy),
    )


@dataclass
class CameraOutput:
    width: int
    height: int

    elevation_deg: Float[Tensor, "B"] = None
    azimuth_deg: Float[Tensor, "B"] = None
    camera_distances: Float[Tensor, "B"] = None

    camera_positions: Float[Tensor, "B"] = None
    camera_centers: Float[Tensor, "B"] = None

    c2w: Float[Tensor, "B 4 4"] = None
    intrinsic: Float[Tensor, "B 4 4"] = None
    proj_mtx: Float[Tensor, "B 4 4"] = None
    mvp_mtx: Float[Tensor, "B 4 4"] = None
    fovx: Float[Tensor, "B"] = None
    fovy: Float[Tensor, "B"] = None
    cx: Float[Tensor, "B"] = None
    cy: Float[Tensor, "B"] = None

    rays_o: Float[Tensor, "B H W 3"] = None
    rays_d: Float[Tensor, "B H W 3"] = None

    key_list = [
        "elevation",
        "azimuth",
        "camera_positions",
        "camera_centers",
        "camera_distances",
        "c2w",
        "intrinsic",
        "proj_mtx",
        "mvp_mtx",
        "fovx",
        "fovy",
        "cx",
        "cy",
        "rays_o",
        "rays_d",
        "width",
        "height",
    ]
    property_list = [
        "elevation_deg",
        "azimuth_deg",
        "camera_positions",
        "camera_centers",
        "camera_distances",
        "c2w",
        "intrinsic",
        "proj_mtx",
        "mvp_mtx",
        "fovx",
        "fovy",
        "cx",
        "cy",
        "rays_o",
        "rays_d",
        "width",
        "height",
    ]

    def to_dict(self):
        output = {}
        for idx in range(len(self.key_list)):
            key = self.key_list[idx]
            prop = getattr(self, self.property_list[idx])
            if prop is not None:
                output[key] = prop
        return output

    def get_index(self, index):
        output = {}
        for idx in range(len(self.key_list)):
            key = self.key_list[idx]
            prop = getattr(self, self.property_list[idx])
            if prop is not None:
                if isinstance(prop, torch.Tensor):
                    output[key] = prop[index]
                else:
                    output[key] = prop
        return output


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
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = (
            torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )
            + world_center
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions) + world_center
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

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
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.up_perturb
        )
        up = up + up_perturb

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
            c2w=None,
            intrinsic=None,
            proj_mtx=None,
            mvp_mtx=None,
            rays_d=None,
            rays_o=None,
        )


@dataclass
class LightOutput:
    light_positions: Float[Tensor, "B 3"]

    def to_dict(self):
        return {"light_positions": self.light_positions}

    def get_index(self, index):
        return {"light_positions": self.light_positions[index]}


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


class CameraInterpolation:
    def __init__(self, camera_sampler: CameraSampler):
        self.camera_sampler = camera_sampler

    def sample(self):
        pass
