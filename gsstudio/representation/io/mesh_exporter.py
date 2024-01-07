from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

import gsstudio
from gsstudio.renderer.background.base import BaseBackground
from gsstudio.renderer.material.base import BaseMaterial
from gsstudio.representation.base.base import BaseGeometry
from gsstudio.representation.io.mesh_exporter_base import Exporter, ExporterOutput
from gsstudio.representation.mesh import Mesh
from gsstudio.utils.typing import *


@gsstudio.register("gaussian-mesh-exporter")
class MeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        fmt: str = "obj"
        save_name: str = "model"
        save_video: bool = True

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)

    def __call__(self) -> List[ExporterOutput]:
        mesh: Mesh = self.geometry.extract_mesh()
        return self.export_obj(mesh)

    def export_obj(self, mesh: Mesh) -> List[ExporterOutput]:
        params = {"mesh": mesh}
        return [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]
