from dataclasses import dataclass

import cv2
import torch

from gsstudio.data.utils.data_utils import DataOutput
from gsstudio.utils.typing import *


class ImageOutput(DataOutput):
    frame_image_path: List[str] = []
    frame_mask_path: List[str] = []
    bbox: Float[Tensor, "B 4"] = None
    image: Float[Tensor, "B C H W"] = None
    mask: Float[Tensor, "B H W"] = None
    width: int = None
    height: int = None

    key_mapping = {"bbox": "image_bbox"}

    def load_image(self):
        self.image = []
        for frame_path in self.frame_image_path:
            img = cv2.imread(frame_path)[:, :, ::-1].copy()
            img = cv2.resize(img, (self.width, self.height))
            img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
            img = img.permute(2, 0, 1)
            self.image.append(img)
        self.image = torch.stack(self.image, dim=0)
        if self.frame_mask_path is not None and len(self.frame_mask_path) > 0:
            self.mask = []
            for mask_path in self.frame_mask_path:
                mask = cv2.imread(mask_path)
                mask = cv2.resize(mask, (self.width, self.height))
                mask: Float[Tensor, "H W 3"] = torch.FloatTensor(mask) / 255
                mask = mask.permute(2, 0, 1)
                self.mask.append(mask)
            self.mask = torch.stack(self.mask, dim=0)

    def load_single_image(self, index):
        frame_path = self.frame_image_path[index]
        img = cv2.imread(frame_path)[:, :, ::-1].copy()
        img = cv2.resize(img, (self.width, self.height))
        img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
        img = img.permute(2, 0, 1).unsqueeze(0)
        self.image = img
        if self.frame_mask_path is not None:
            mask_path = self.frame_mask_path[index]
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, (self.width, self.height))
            mask: Float[Tensor, "H W 3"] = torch.FloatTensor(mask) / 255
            mask = mask.permute(2, 0, 1).unsqueeze(0)
            self.mask = mask
