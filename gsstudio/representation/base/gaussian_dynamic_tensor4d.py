from dataclasses import dataclass, field

import torch
import torch.nn as nn

import gsstudio
from gsstudio.representation.base.dynamic import DynamicBaseModel
from gsstudio.representation.base.gaussian import GaussianBaseModel
from gsstudio.representation.base.utils import BasicPointCloud
from gsstudio.utils.config import C
from gsstudio.utils.misc import get_device
from gsstudio.utils.typing import *

try:
    import tinycudann
    from threestudio.models.networks import get_encoding, get_mlp
except ImportError:
    gsstudio.info("tinycudann not found, can not use gaussian dynamic tensor4d model.")


class DynamicNetwork(nn.Module):
    def __init__(self, output_dim=3, pos_encoding_config=None, mlp_network_config=None):
        super().__init__()
        self.output_dim = output_dim
        self.enc_xy = get_encoding(2, pos_encoding_config)
        self.enc_xz = get_encoding(2, pos_encoding_config)
        self.enc_yz = get_encoding(2, pos_encoding_config)
        self.enc_xt = get_encoding(2, pos_encoding_config)
        self.enc_yt = get_encoding(2, pos_encoding_config)
        self.enc_zt = get_encoding(2, pos_encoding_config)

        self.pos_encoding_list = [
            self.enc_xy,
            self.enc_xz,
            self.enc_yz,
            self.enc_xt,
            self.enc_yt,
            self.enc_zt,
        ]

        self.feature_network = get_mlp(
            self.enc_xy.n_output_dims * 6 + 6 * 2,
            output_dim,
            mlp_network_config,
        )

    def forward(self, points_list):
        pos_enc_list = []
        for i in range(6):
            pos_enc = self.pos_encoding_list[i](points_list[i])
            pos_enc_list.append(pos_enc)
            pos_enc_list.append(points_list[i])
        enc = torch.cat(pos_enc_list, dim=-1)
        features = self.feature_network(enc).view(
            *points_list[0].shape[:-1], self.output_dim
        )
        return features


@gsstudio.register("gaussian-splatting-dynamic-tensor4d")
class GaussianDynamicT4dModel(GaussianBaseModel, DynamicBaseModel):
    @dataclass
    class Config(GaussianBaseModel.Config):
        num_frames: int = 10
        delta_pos_lr: Any = 0.001
        delta_rot_lr: Any = 0.001

        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "ProgressiveBandHashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
                "start_level": 13,
                "start_step": 500,
                "update_steps": 500,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            }
        )
        need_normalization: bool = False
        bbox_min: Optional[Tuple[float, float, float]] = None
        bbox_max: Optional[Tuple[float, float, float]] = None

    cfg: Config

    @property
    def get_rotation(self):
        _delta_rot = self.query(
            self._xyz.to(get_device()), self.delta_rot_network.to(get_device())
        )
        return self.rotation_activation(self._rotation + _delta_rot)

    @property
    def get_xyz(self):
        _delta_xyz = self.query(
            self._xyz.to(get_device()), self.delta_xyz_network.to(get_device())
        )
        return self._xyz + _delta_xyz.to(self._xyz.device)

    def configure(self) -> None:
        self.delta_xyz_network = DynamicNetwork(
            3, self.cfg.pos_encoding_config, self.cfg.mlp_network_config
        )
        self.delta_rot_network = DynamicNetwork(
            4, self.cfg.pos_encoding_config, self.cfg.mlp_network_config
        )
        self.time_index = 0
        for key in dir(self):
            print(key)
        super().configure()

    def init_normalization(self):
        if self.cfg.bbox_min is not None:
            bbox_min = self.cfg.bbox_min
            bbox_max = self.cfg.bbox_max
        else:
            bbox_min = torch.min(self._xyz, dim=0)[0]
            bbox_max = torch.max(self._xyz, dim=0)[0]
        self.scale_x = bbox_max[0] - bbox_min[0]
        self.scale_y = bbox_max[1] - bbox_min[1]
        self.scale_z = bbox_max[2] - bbox_min[2]
        self.scale = max(max(self.scale_x, self.scale_y), self.scale_z)
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

    def query(
        self,
        points_3d: Float[Tensor, "*N Di"],
        dynamic_network,
    ) -> Dict[str, Float[Tensor, "..."]]:
        if self.cfg.need_normalization:
            points = points_3d.clone()
            points[:, 0] = (points[:, 0] - self.cfg.bbox_min[0]) / self.scale_x
            points[:, 1] = (points[:, 1] - self.cfg.bbox_min[1]) / self.scale_y
            points[:, 2] = (points[:, 2] - self.cfg.bbox_min[2]) / self.scale_z
        else:
            points = points_3d[..., :3]
        points = points.view(-1, 3)

        points_xy = torch.zeros_like(points[:, :2])
        points_xy[:, 0] = points[:, 0]
        points_xy[:, 1] = points[:, 1]

        points_xz = torch.zeros_like(points[:, :2])
        points_xz[:, 0] = points[:, 0]
        points_xz[:, 1] = points[:, 2]

        points_yz = torch.zeros_like(points[:, :2])
        points_yz[:, 0] = points[:, 1]
        points_yz[:, 1] = points[:, 2]

        if self.moment is not None:
            moment = self.moment
        else:
            moment = self.time_index / self.cfg.num_frames * 2 - 1

        points_xt = torch.zeros_like(points[:, :2])
        points_xt[:, 0] = points[:, 0]
        points_xt[:, 1] = moment

        points_yt = torch.zeros_like(points[:, :2])
        points_yt[:, 0] = points[:, 1]
        points_yt[:, 1] = moment

        points_zt = torch.zeros_like(points[:, :2])
        points_zt[:, 0] = points[:, 2]
        points_zt[:, 1] = moment

        points_list = [points_xy, points_xz, points_yz, points_xt, points_yt, points_zt]
        return dynamic_network(points_list)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        super().create_from_pcd(pcd, spatial_lr_scale)
        if self.cfg.need_normalization:
            self.init_normalization()

    def to(self, device="cpu"):
        super().to(device)
        self.delta_xyz_network = self.delta_xyz_network.to(device)
        self.delta_rot_network = self.delta_rot_network.to(device)

    def training_setup(self):
        super().training_setup()
        l = self.optimize_list
        training_args = self.cfg
        l.append(
            {
                "params": self.delta_xyz_network.parameters(),
                "lr": C(training_args.delta_pos_lr, 0, 0),
                "name": "delta_xyz",
            },
        )
        l.append(
            {
                "params": self.delta_rot_network.parameters(),
                "lr": C(training_args.delta_rot_lr, 0, 0),
                "name": "delta_rot",
            },
        )
