from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gsstudio.utils.base import BaseModule
from gsstudio.utils.typing import *


class BaseGeometry(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    @staticmethod
    def create_from(
        other: "BaseGeometry", cfg: Optional[Union[dict, DictConfig]] = None, **kwargs
    ) -> "BaseGeometry":
        raise TypeError(
            f"Cannot create {BaseGeometry.__name__} from {other.__class__.__name__}"
        )

    def export(self, *args, **kwargs) -> Dict[str, Any]:
        return {}


from . import gaussian, gaussian_dynamic, gaussian_dynamic_tensor4d
