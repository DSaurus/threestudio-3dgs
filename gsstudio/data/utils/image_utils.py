from dataclasses import dataclass

import cv2
import numpy as np
import torch

from gsstudio.data.utils.data_utils import DataOutput
from gsstudio.utils.typing import *


class ImageOutput(DataOutput):
    frame_image_path: List[str] = None
    frame_mask_path: List[str] = None
    frame_normal_path: List[str] = None
    frame_depth_path: List[str] = None
    bbox: Float[Tensor, "B 4"] = None
    image: Float[Tensor, "B H W C"] = None
    mask: Float[Tensor, "B H W C"] = None
    depth: Float[Tensor, "B H W C"] = None
    normal: Float[Tensor, "B H W C"] = None
    width: int = None
    height: int = None

    white_background: bool = False

    key_mapping = {"bbox": "image_bbox"}

    def load_image(self):
        # frame_image_path str or List[str]
        # frame_mask_path str or List[str]
        if isinstance(self.frame_image_path, str):
            frame_image_path = [self.frame_image_path]
        else:
            frame_image_path = self.frame_image_path
        if frame_image_path is not None and len(frame_image_path) > 0:
            self.image = []
            self.mask = []
            for frame_path in frame_image_path:
                if frame_path.endswith(".png"):
                    png_img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                    img = png_img[:, :, :3]

                    mask = png_img[:, :, 3]
                    mask = cv2.resize(mask, (self.width, self.height))
                    mask: Float[Tensor, "H W"] = torch.FloatTensor(mask) / 255
                    mask = mask.unsqueeze(-1)
                    self.mask.append(mask)
                img = cv2.imread(frame_path)[:, :, ::-1].copy()
                img = cv2.resize(img, (self.width, self.height))
                img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
                self.image.append(img)
            self.image = torch.stack(self.image, dim=0)
            if len(self.mask) > 0:
                self.mask = torch.stack(self.mask, dim=0)
            else:
                self.mask = None

        if isinstance(self.frame_mask_path, str):
            frame_mask_path = [self.frame_mask_path]
        else:
            frame_mask_path = self.frame_mask_path
        if frame_mask_path is not None and len(frame_mask_path) > 0:
            self.mask = []
            for mask_path in frame_mask_path:
                mask = cv2.imread(mask_path)
                mask = cv2.resize(mask, (self.width, self.height))
                mask: Float[Tensor, "H W 3"] = torch.FloatTensor(mask) / 255
                mask = mask[:, :, :1]
                self.mask.append(mask)
            self.mask = torch.stack(self.mask, dim=0)

        if self.white_background and self.mask is not None:
            self.image = self.image * self.mask + (1 - self.mask)

        if isinstance(self.frame_normal_path, str):
            frame_normal_path = [self.frame_normal_path]
        else:
            frame_normal_path = self.frame_normal_path
        if frame_normal_path is not None and len(frame_normal_path) > 0:
            self.normal = []
            for normal_path in frame_normal_path:
                normal = cv2.imread(normal_path)[:, :, ::-1].copy()
                normal = cv2.resize(normal, (self.width, self.height))
                normal: Float[Tensor, "H W 3"] = torch.FloatTensor(normal) / 255
                normal = normal[..., :3]
                self.normal.append(normal)
            self.normal = torch.stack(self.normal, dim=0)

        if isinstance(self.frame_depth_path, str):
            frame_depth_path = [self.frame_depth_path]
        else:
            frame_depth_path = self.frame_depth_path
        if frame_depth_path is not None and len(frame_depth_path) > 0:
            self.depth = []
            for depth_path in frame_depth_path:
                if (
                    depth_path.endswith("png")
                    or depth_path.endswith("jpg")
                    or depth_path.endswith("jpeg")
                ):
                    depth = cv2.imread(depth_path)
                    depth = cv2.resize(depth, (self.width, self.height))
                    depth: Float[Tensor, "H W 3"] = torch.FloatTensor(depth) / 255
                    depth = depth[..., :1]
                    self.depth.append(depth)
                elif depth_path.endswith("npy"):
                    depth = np.load(depth_path)
                    depth = cv2.resize(depth, (self.width, self.height))
                    depth = torch.FloatTensor(depth).unsqueeze(-1)
                    self.depth.append(depth)
            self.depth = torch.stack(self.depth, dim=0)
