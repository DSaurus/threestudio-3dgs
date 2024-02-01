import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import gsstudio
from gsstudio.utils.base import BaseModule
from gsstudio.utils.typing import *


class BaseBackground(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self):
        pass

    def forward(self, dirs: Float[Tensor, "B H W 3"]) -> Float[Tensor, "B H W Nc"]:
        raise NotImplementedError
