import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

import gsstudio


@gsstudio.register("dynamic-base-model")
class DynamicBaseModel:
    moment = None
    time_index = None

    def set_time(self, moment=None, time_index=None):
        self.moment = moment
        self.time_index = time_index
