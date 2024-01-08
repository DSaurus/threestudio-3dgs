from dataclasses import dataclass

import cv2
import torch
from threestudio.utils.typing import *


class ImageOutput:
    frame_image_path: List[str] = []
    frame_mask_path: List[str] = []
    bbox: Float[Tensor, "B 4"]
    image: Float[Tensor, "B C H W"]
    mask: Float[Tensor, "B H W"]
    width: int = None
    height: int = None

    key_mapping = {}

    def __init__(self, **kwargs):
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

    def to_dict(self):
        output = {}
        for key in dir(self):
            prop = getattr(self, key)
            if prop is not None:
                if key in self.key_mapping:
                    key = self.key_mapping[key]
                output[key] = prop
        return output

    def get_index(self, index):
        output = {}
        for key in dir(self):
            prop = getattr(self, key)
            if prop is not None:
                if key in self.key_mapping:
                    key = self.key_mapping[key]
                if isinstance(prop, torch.Tensor):
                    output[key] = prop[index]
                else:
                    output[key] = prop
        return output

    def load_image(self):
        self.image = []
        for frame_path in self.frame_image_path:
            img = cv2.imread(frame_path)[:, :, ::-1].copy()
            img = cv2.resize(img, (self.width, self.height))
            img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
            img = img.permute(2, 0, 1)
            self.image.append(img)
        self.image = torch.stack(self.image, dim=0)
        if self.frame_mask_path is not None:
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
