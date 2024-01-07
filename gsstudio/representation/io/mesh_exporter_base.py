from dataclasses import dataclass

import gsstudio
from gsstudio.renderer.background.base import BaseBackground
from gsstudio.renderer.material.base import BaseMaterial
from gsstudio.representation.base import BaseGeometry
from gsstudio.utils.base import BaseObject
from gsstudio.utils.typing import *


@dataclass
class ExporterOutput:
    save_name: str
    save_type: str
    params: Dict[str, Any]


class Exporter(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        save_video: bool = False

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        @dataclass
        class SubModules:
            geometry: BaseGeometry
            material: BaseMaterial
            background: BaseBackground

        self.sub_modules = SubModules(geometry, material, background)

    @property
    def geometry(self) -> BaseGeometry:
        return self.sub_modules.geometry

    @property
    def material(self) -> BaseMaterial:
        return self.sub_modules.material

    @property
    def background(self) -> BaseBackground:
        return self.sub_modules.background

    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        raise NotImplementedError


@gsstudio.register("dummy-exporter")
class DummyExporter(Exporter):
    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        # DummyExporter does not export anything
        return []
