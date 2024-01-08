import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

import gsstudio
from gsstudio.utils.typing import *


class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor


# gaussian splatting functions
def convert_gl2cv(C2W, intrinsic, height):
    flip_yz = torch.eye(4, device=C2W.device)
    # flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    intrinsic[1, 1] *= -1
    intrinsic[1, 2] = height - intrinsic[1, 2]
    return C2W, intrinsic


def get_projection_matrix_advanced(
    znear, zfar, fovX, fovY, cx=0.0, cy=0.0, device="cuda"
):
    tanHalfFovY = torch.tan((fovY / 2))
    tanHalfFovX = torch.tan((fovX / 2))
    B = fovX.shape[0]

    P = torch.zeros(B, 4, 4, device=device)

    P[:, 0, 0] = 1.0 / tanHalfFovX
    P[:, 1, 1] = 1.0 / tanHalfFovY
    P[:, 0, 2] = cx
    P[:, 1, 2] = cy
    P[:, 3, 2] = 1
    P[:, 2, 2] = (zfar - znear) / (zfar + znear)
    P[:, 2, 3] = -2 * (zfar * znear) / (zfar + znear)
    return P


def get_fov_gaussian(P):
    tanHalfFovX = 1 / P[0, 0]
    tanHalfFovY = 1 / P[1, 1]
    fovY = math.atan(tanHalfFovY) * 2
    fovX = math.atan(tanHalfFovX) * 2
    return fovX, fovY


def get_cam_info_gaussian(c2w, fovx, fovy, znear, zfar, cx=0, cy=0):
    # c2w = convert_pose(c2w)
    assert c2w.shape[0] == 1
    world_view_transform = torch.inverse(c2w)[0]

    world_view_transform = world_view_transform.transpose(0, 1).cuda().float()
    projection_matrix = (
        get_projection_matrix_advanced(
            znear=znear, zfar=zfar, fovX=fovx, fovY=fovy, cx=cx, cy=cy
        )[0]
        .transpose(0, 1)
        .cuda()
    )
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return world_view_transform, full_proj_transform, camera_center


# NeRF functions


def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )
    if isinstance(fx, torch.Tensor):
        batch_size = fx.shape[0]
        i = i.unsqueeze(0).repeat(batch_size, 1, 1)
        j = j.unsqueeze(0).repeat(batch_size, 1, 1)
        fx = fx.reshape(batch_size, 1, 1)
        fy = fy.reshape(batch_size, 1, 1)
        cx = cx.reshape(batch_size, 1, 1)
        cy = cy.reshape(batch_size, 1, 1)

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1
    )

    return directions


def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
    normalize=True,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


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
