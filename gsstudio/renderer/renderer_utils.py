import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

import gsstudio
from gsstudio.data.utils.camera_utils import get_projection_matrix_zplus
from gsstudio.utils.typing import *


class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor


def get_fov_gaussian(P):
    tanHalfFovX = 1 / P[0, 0]
    tanHalfFovY = 1 / P[1, 1]
    fovY = math.atan(tanHalfFovY) * 2
    fovX = math.atan(tanHalfFovX) * 2
    return fovX, fovY


def get_cam_info_gaussian(c2w, fovx, fovy, znear, zfar, cx=0, cy=0):
    assert c2w.shape[0] == 1
    world_view_transform = torch.inverse(c2w)[0]

    world_view_transform = world_view_transform.transpose(0, 1).float()
    projection_matrix = (
        get_projection_matrix_zplus(
            znear=znear, zfar=zfar, fovX=fovx, fovY=fovy, cx=cx, cy=cy
        )[0].transpose(0, 1)
    ).to(c2w.device)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return world_view_transform, full_proj_transform, camera_center


def get_projection_matrix(
    fovy: Float[Tensor, "B"], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "B 4 4"]:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = 1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = (far - near) / (far + near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far + near)
    proj_mtx[:, 3, 2] = 1.0
    return proj_mtx


def get_mvp_matrix(
    c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]
) -> Float[Tensor, "B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx


def get_full_projection_matrix(
    c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]
) -> Float[Tensor, "B 4 4"]:
    return (c2w.unsqueeze(0).bmm(proj_mtx.unsqueeze(0))).squeeze(0)


def validate_empty_rays(ray_indices, t_start, t_end):
    if ray_indices.nelement() == 0:
        gsstudio.warn("Empty rays_indices!")
        ray_indices = torch.LongTensor([0]).to(ray_indices)
        t_start = torch.Tensor([0]).to(ray_indices)
        t_end = torch.Tensor([0]).to(ray_indices)
    return ray_indices, t_start, t_end
