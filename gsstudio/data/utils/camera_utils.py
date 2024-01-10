from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gsstudio.data.utils.data_utils import DataOutput
from gsstudio.renderer.renderer_utils import (
    get_projection_matrix_advanced,
    get_ray_directions,
    get_rays,
)
from gsstudio.utils.typing import *


def convert_gl2cv(C2W, intrinsic, height):
    flip_yz = torch.eye(4, device=C2W.device).unsqueeze(0).repeat(C2W.shape[0], 1, 1)
    flip_yz[:, 2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    intrinsic[:, 1, 1] *= -1
    intrinsic[:, 1, 2] = height - intrinsic[:, 1, 2]
    return C2W, intrinsic


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


class CameraOutput(DataOutput):
    width: int = None
    height: int = None

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
    rays_d_normalize: bool = True

    camera_time: Float[Tensor, "B"] = None
    camera_time_index: Int[Tensor, "B"] = None

    key_mapping = {
        "elevation_deg": "elevation",
        "azimuth_deg": "azimuth",
    }

    def gen_rays(self):
        self.rays_o, self.rays_d = matrix2rays(
            self.c2w,
            self.intrinsic,
            self.height,
            self.width,
            normalize=self.rays_d_normalize,
        )