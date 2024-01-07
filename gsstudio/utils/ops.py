import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from gsstudio.utils.typing import *


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]


def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


class SpecifyGradient(Function):
    # Implementation from stable-dreamfusion
    # https://github.com/ashawkey/stable-dreamfusion
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


trunc_exp = _TruncExp.apply


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


def chunk_batch(func: Callable, chunk_size: int, *args, **kwargs) -> Any:
    if chunk_size <= 0:
        return func(*args, **kwargs)
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    assert (
        B is not None
    ), "No tensor found in args or kwargs, cannot determine batch size."
    out = defaultdict(list)
    out_type = None
    # max(1, B) to support B == 0
    for i in range(0, max(1, B), chunk_size):
        out_chunk = func(
            *[
                arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ],
            **{
                k: arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for k, arg in kwargs.items()
            },
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v)

    if out_type is None:
        return None

    out_merged: Dict[Any, Optional[torch.Tensor]] = {}
    for k, v in out.items():
        if all([vv is None for vv in v]):
            # allow None in return value
            out_merged[k] = None
        elif all([isinstance(vv, torch.Tensor) for vv in v]):
            out_merged[k] = torch.cat(v, dim=0)
        else:
            raise TypeError(
                f"Unsupported types in return value of func: {[type(vv) for vv in v if not isinstance(vv, torch.Tensor)]}"
            )

    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in [tuple, list]:
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged


def shifted_expotional_decay(a, b, c, r):
    return a * torch.exp(-b * r) + c


def shifted_cosine_decay(a, b, c, r):
    return a * torch.cos(b * r + c) + a


def perpendicular_component(x: Float[Tensor, "B C H W"], y: Float[Tensor, "B C H W"]):
    # get the component of x that is perpendicular to y
    eps = torch.ones_like(x[:, 0, 0, 0]) * 1e-6
    return (
        x
        - (
            torch.mul(x, y).sum(dim=[1, 2, 3])
            / torch.maximum(torch.mul(y, y).sum(dim=[1, 2, 3]), eps)
        ).view(-1, 1, 1, 1)
        * y
    )
