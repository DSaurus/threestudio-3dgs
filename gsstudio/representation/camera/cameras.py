# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

import gsstudio
from gsstudio.utils.typing import *
from gsstudio.utils.base import BaseModule
from gsstudio.representation.camera.rotation_utils import *

# Default values for rotation and translation matrices.
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)

# An input which is a float per batch element
_BatchFloatType = Union[float, Sequence[float], torch.Tensor]

# one or two floats per batch element
_FocalLengthType = Union[
    float, Sequence[Tuple[float]], Sequence[Tuple[float, float]], torch.Tensor
]


class CamerasBase(BaseModule):
    """
    `CamerasBase` implements a base class for all cameras.

    For cameras, there are four different coordinate systems (or spaces)
    - World coordinate system: This is the system the object lives - the world.
    - Camera view coordinate system: This is the system that has its origin on
        the camera and the Z-axis perpendicular to the image plane.
        In PyTorch3D, we assume that +X points left, and +Y points up and
        +Z points out from the image plane.
        The transformation from world --> view happens after applying a rotation (R)
        and translation (T)
    - NDC coordinate system: This is the normalized coordinate system that confines
        points in a volume the rendered part of the object or scene, also known as
        view volume. For square images, given the PyTorch3D convention, (+1, +1, znear)
        is the top left near corner, and (-1, -1, zfar) is the bottom right far
        corner of the volume.
        The transformation from view --> NDC happens after applying the camera
        projection matrix (P) if defined in NDC space.
        For non square images, we scale the points such that smallest side
        has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    - Screen coordinate system: This is another representation of the view volume with
        the XY coordinates defined in image space instead of a normalized space.

    An illustration of the coordinate systems can be found in pytorch3d/docs/notes/cameras.md.

    CameraBase defines methods that are common to all camera models:
        - `get_camera_center` that returns the optical center of the camera in
            world coordinates
        - `get_world_to_view_transform` which returns a 3D transform from
            world coordinates to the camera view coordinates (R, T)
        - `get_full_projection_transform` which composes the projection
            transform (P) with the world-to-view transform (R, T)
        - `transform_points` which takes a set of input points in world coordinates and
            projects to the space the camera is defined in (NDC or screen)
        - `get_ndc_camera_transform` which defines the transform from screen/NDC to
            PyTorch3D's NDC space
        - `transform_points_ndc` which takes a set of points in world coordinates and
            projects them to PyTorch3D's NDC space
        - `transform_points_screen` which takes a set of points in world coordinates and
            projects them to screen space

    For each new camera, one should implement the `get_projection_transform`
    routine that returns the mapping from camera view coordinates to camera
    coordinates (NDC or screen).

    Another useful function that is specific to each camera model is
    `unproject_points` which sends points from camera coordinates (NDC or screen)
    back to camera view or world coordinates depending on the `world_coordinates`
    boolean argument of the function.
    """

    # Used in __getitem__ to index the relevant fields
    # When creating a new camera, this should be set in the __init__
    _FIELDS: Tuple[str, ...] = ()

    # Names of fields which are a constant property of the whole batch, rather
    # than themselves a batch of data.
    # When joining objects into a batch, they will have to agree.
    _SHARED_FIELDS: Tuple[str, ...] = ()

    def get_projection_transform(self) -> Float[Tensor, "B 4 4"]:
        """
        Calculate the projective transformation matrix.

        Args:

        Return:
            a batch of projection matrices of shape (N, 4, 4)
        """
        raise NotImplementedError()

    def unproject_points(self, xy_depth: torch.Tensor, **kwargs):
        """
        Transform input points from camera coordinates (NDC or screen)
        to the world / camera coordinates.

        Each of the input points `xy_depth` of shape (..., 3) is
        a concatenation of the x, y location and its depth.

        For instance, for an input 2D tensor of shape `(num_points, 3)`
        `xy_depth` takes the following form:
            `xy_depth[i] = [x[i], y[i], depth[i]]`,
        for a each point at an index `i`.

        The following example demonstrates the relationship between
        `transform_points` and `unproject_points`:

        .. code-block:: python

            cameras = # camera object derived from CamerasBase
            xyz = # 3D points of shape (batch_size, num_points, 3)
            # transform xyz to the camera view coordinates
            xyz_cam = cameras.get_world_to_view_transform().transform_points(xyz)
            # extract the depth of each point as the 3rd coord of xyz_cam
            depth = xyz_cam[:, :, 2:]
            # project the points xyz to the camera
            xy = cameras.transform_points(xyz)[:, :, :2]
            # append depth to xy
            xy_depth = torch.cat((xy, depth), dim=2)
            # unproject to the world coordinates
            xyz_unproj_world = cameras.unproject_points(xy_depth, world_coordinates=True)
            print(torch.allclose(xyz, xyz_unproj_world)) # True
            # unproject to the camera coordinates
            xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)
            print(torch.allclose(xyz_cam, xyz_unproj)) # True

        Args:
            xy_depth: torch tensor of shape (..., 3).
            world_coordinates: If `True`, unprojects the points back to world
                coordinates using the camera extrinsics `R` and `T`.
                `False` ignores `R` and `T` and unprojects to
                the camera view coordinates.
            from_ndc: If `False` (default), assumes xy part of input is in
                NDC space if self.in_ndc(), otherwise in screen space. If
                `True`, assumes xy is in NDC space even if the camera
                is defined in screen space.

        Returns
            new_points: unprojected points with the same shape as `xy_depth`.
        """
        raise NotImplementedError()

    def get_camera_position(self) -> torch.Tensor:
        """
        Return the 3D location of the camera optical center
        in the world coordinates.

        Setting R or T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        w2v_trans = self.get_world_to_view_transform()
        P = w2v_trans.inverse()
        # the camera center is the translation component (the first 3 elements
        # of the last row) of the inverted world-to-view
        # transform (4x4 RT matrix)
        return P[:, 3, :3]

    def get_world_to_view_transform(self) -> Float[Tensor, "B 4 4"]:
        """
        Return the world-to-view transform.

        Returns:
            a batch of projection matrices of shape (N, 4, 4)
        """
        world_to_view_transform = get_world_to_view_transform(R=self.R, T=self.T)
        return world_to_view_transform
    
    def get_view_to_world_transform(self) -> Float[Tensor, "B 4 4"]:
        """
        Return the view-to-world transform.

        Returns:
            a batch of projection matrices of shape (N, 4, 4)
        """
        view_to_world_transform = get_world_to_view_transform(R=self.R, T=self.T).inverse()
        return view_to_world_transform

    def get_full_projection_transform(self) -> Float[Tensor, "B 4 4"]:
        """
        Return the full world-to-camera transform composing the
        world-to-view and view-to-camera transforms.
        If camera is defined in NDC space, the projected points are in NDC space.
        If camera is defined in screen space, the projected points are in screen space.

        Returns:
            a batch of projection matrices of shape (N, 4, 4)
        """
        world_to_view_transform = self.get_world_to_view_transform()
        view_to_proj_transform = self.get_projection_transform()
        return view_to_proj_transform @ world_to_view_transform
    
    def get_mvp_transform(self, model_transform=None) -> Float[Tensor, "B 4 4"]:
        """
        Return the MVP transform composing the model-to-view, world-to-view and view-to-camera transforms.

        Returns:
            a batch of projection matrices of shape (N, 4, 4)
        """
        world_to_view_transform = self.get_world_to_view_transform()
        view_to_proj_transform = self.get_projection_transform()
        _N = world_to_view_transform.shape[0]
        if model_transform is None:
            _model_transform = torch.eye(4, device=self.device, dtype=torch.float32)[None].repeat(_N, 1, 1)
        elif model_transform.ndim == 2:
            _model_transform = model_transform[None]
        else:
            _model_transform = model_transform
        return view_to_proj_transform @ world_to_view_transform @ _model_transform

    def transform_points(
        self, points, eps: Optional[float] = None
    ) -> torch.Tensor:
        """
        Transform input points from world to camera space.
        If camera is defined in NDC space, the projected points are in NDC space.
        If camera is defined in screen space, the projected points are in screen space.

        For `CamerasBase.transform_points`, setting `eps > 0`
        stabilizes gradients since it leads to avoiding division
        by excessively low numbers for points close to the camera plane.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_proj_transform = self.get_full_projection_transform()
        return transform_points(points, world_to_proj_transform, eps=eps)

    def transform_points_ndc(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transforms points from PyTorch3D world/camera space to NDC space.
        Input points follow the PyTorch3D coordinate system conventions: +X left, +Y up.
        Output points are in NDC space: +X left, +Y up, origin at image center.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_ndc_transform = self.get_full_projection_transform(**kwargs)
        if not self.in_ndc():
            to_ndc_transform = self.get_ndc_camera_transform(**kwargs)
            world_to_ndc_transform = world_to_ndc_transform.compose(to_ndc_transform)

        return world_to_ndc_transform.transform_points(points, eps=eps)

    def transform_points_screen(
        self, points, eps: Optional[float] = None, with_xyflip: bool = True, **kwargs
    ) -> torch.Tensor:
        """
        Transforms points from PyTorch3D world/camera space to screen space.
        Input points follow the PyTorch3D coordinate system conventions: +X left, +Y up.
        Output points are in screen space: +X right, +Y down, origin at top left corner.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.
            with_xyflip: If True, flip x and y directions. In world/camera/ndc coords,
                +x points to the left and +y up. If with_xyflip is true, in screen
                coords +x points right, and +y down, following the usual RGB image
                convention. Warning: do not set to False unless you know what you're
                doing!

        Returns
            new_points: transformed points with the same shape as the input.
        """
        points_ndc = self.transform_points_ndc(points, eps=eps, **kwargs)
        image_size = kwargs.get("image_size", self.get_image_size())
        return get_ndc_to_screen_transform(
            self, with_xyflip=with_xyflip, image_size=image_size
        ).transform_points(points_ndc, eps=eps)

    def is_perspective(self):
        raise NotImplementedError()

    def in_ndc(self):
        """
        Specifies whether the camera is defined in NDC space
        or in screen (image) space
        """
        raise NotImplementedError()

    def get_znear(self):
        return getattr(self, "znear", None)

    def get_image_size(self):
        """
        Returns the image size, if provided, expected in the form of (height, width)
        The image size is used for conversion of projected points to screen coordinates.
        """
        return getattr(self, "image_size", None)
    
    def enable_train(self, attr: Union[str, List[str]]):
        if isinstance(attr, str):
            attr = [attr]
        assert hasattr(self, attr) and type(getattr(self, attr)) == torch.Tensor
        getattr(self, attr).requires_grad = True
        
    def disable_train(self, attr: Union[str, List[str]]):
        if isinstance(attr, str):
            attr = [attr]
        assert hasattr(self, attr) and type(getattr(self, attr)) == torch.Tensor
        getattr(self, attr).requires_grad = False

    def __getitem__(
        self, index: Union[int, List[int], torch.BoolTensor, torch.LongTensor]
    ) -> "CamerasBase":
        """
        Override for the __getitem__ method in TensorProperties which needs to be
        refactored.

        Args:
            index: an integer index, list/tensor of integer indices, or tensor of boolean
                indicators used to filter all the fields in the cameras given by self._FIELDS.
        Returns:
            an instance of the current cameras class with only the values at the selected index.
        """

        kwargs = {}

        tensor_types = {
            # pyre-fixme[16]: Module `cuda` has no attribute `BoolTensor`.
            "bool": (torch.BoolTensor, torch.cuda.BoolTensor),
            # pyre-fixme[16]: Module `cuda` has no attribute `LongTensor`.
            "long": (torch.LongTensor, torch.cuda.LongTensor),
        }
        if not isinstance(
            index, (int, list, *tensor_types["bool"], *tensor_types["long"])
        ) or (
            isinstance(index, list)
            and not all(isinstance(i, int) and not isinstance(i, bool) for i in index)
        ):
            msg = (
                "Invalid index type, expected int, List[int] or Bool/LongTensor; got %r"
            )
            raise ValueError(msg % type(index))

        if isinstance(index, int):
            index = [index]

        if isinstance(index, tensor_types["bool"]):
            # pyre-fixme[16]: Item `List` of `Union[List[int], BoolTensor,
            #  LongTensor]` has no attribute `ndim`.
            # pyre-fixme[16]: Item `List` of `Union[List[int], BoolTensor,
            #  LongTensor]` has no attribute `shape`.
            if index.ndim != 1 or index.shape[0] != len(self):
                raise ValueError(
                    # pyre-fixme[16]: Item `List` of `Union[List[int], BoolTensor,
                    #  LongTensor]` has no attribute `shape`.
                    f"Boolean index of shape {index.shape} does not match cameras"
                )
        elif max(index) >= len(self):
            raise IndexError(f"Index {max(index)} is out of bounds for select cameras")

        for field in self._FIELDS:
            val = getattr(self, field, None)
            if val is None:
                continue

            # e.g. "in_ndc" is set as attribute "_in_ndc" on the class
            # but provided as "in_ndc" on initialization
            if field.startswith("_"):
                field = field[1:]

            if isinstance(val, (str, bool)):
                kwargs[field] = val
            elif isinstance(val, torch.Tensor):
                # In the init, all inputs will be converted to
                # tensors before setting as attributes
                kwargs[field] = val[index]
            else:
                raise ValueError(f"Field {field} type is not supported for indexing")

        kwargs["device"] = self.device
        return self.__class__(**kwargs)


############################################################
#               Perspective Camera Class                   #
############################################################


class PerspectiveCameras(CamerasBase):
    """
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.
    """

    # For __getitem__
    _FIELDS = (
        "K", 
        "fx", "fy", "focal_length",
        "cx", "cy", "principal_point",
        "fov", "fovx", "fovy",
        "R",
        "T"
    )

    _SHARED_FIELDS = ("image_size", "height", "width", "znear", "zfar")
    
    def configure(self, **kwargs):
        """
        Configures the camera with the provided parameters.

        Parameters:
        - kwargs: Keyword arguments containing the camera parameters.
            - Image size: provided as "image_size", ("h", "w") or ("img_h", "img_w")
            - Intrinsic: provided as "K", "intr" or "intrinsic".
              If any of them is provided, the other parameters will be ignored.
              Or you can also provide focal length as "fx", "fy" or "focal_length",
              and principal point as "cx", "cy" or "principal_point".
            - Extrinsic: provided as "extr", "extrinsic", "transform_matrix".
              If any of them is provided, the other parameters will be ignored. 
              Or you can also provide rotation as "R" or "rotation",
              and translation as "T" or "translation".
              If nothing found, use default values (identical).
            - Z near and Z far: provided as "znear", "zfar" or "near", "far.
              If nothing found, use default values (0.01, 1000.0).

        Returns:
        None
        """
        self._N = 1
        # Set image size: serach image_size => h, w => img_h, img_w
        if "image_size" in kwargs:
            _img_size = kwargs["image_size"]
            if isinstance(_img_size, int):
                self.image_size = (_img_size, _img_size)
            elif isinstance(_img_size, (list, tuple)):
                self.image_size = tuple(_img_size)
        elif "h" in kwargs and "w" in kwargs:
            self.image_size = (int(kwargs["h"]), int(kwargs["w"]))
        elif "img_h" in kwargs and "img_w" in kwargs:
            self.image_size = (int(kwargs["img_h"]), int(kwargs["img_w"]))
        if self.image_size is None:
            gsstudio.warning("Image size not provided, using default (1024, 1024)")
            self.image_size = (1024, 1024)
        self.width = self.image_size[1]
        self.height = self.image_size[0]
            
        # K
        self.K = _intr = None
        if "K" in kwargs:
            _intr = kwargs["K"]
        elif "intr" in kwargs:
            _intr = kwargs["intr"]
        elif "intrinsic" in kwargs:
            _intr = kwargs["intrinsic"]
        if _intr is not None:
            _intr = torch.tensor(_intr, dtype=torch.float32)
            if _intr.numel() == 9:
                _intr = _intr.reshape(1, 3, 3)
                self.K = torch.eye(4, device=self.device, dtype=torch.float32)[None]
                self.K[:, :3, :3] = _intr
            elif _intr.numel() == 16:
                self.K = _intr.reshape(1, 4, 4)
            else:
                gsstudio.warning("Invalid K")
        if self.K is not None:
            self.cx = self.K[:, 0, 2]
            self.cy = self.K[:, 1, 2]
            self.principal_point = torch.stack([self.cx, self.cy], dim=1)
            self.fx = self.K[:, 0, 0]
            self.fy = self.K[:, 1, 1]
            self.focal_length = torch.stack([self.fx, self.fy], dim=1)
            self.fovx = 2 * torch.atan(self.image_size[1] / (2 * self.fx))
            self.fovy = 2 * torch.atan(self.image_size[0] / (2 * self.fy))
            self.fov = torch.stack([self.fovx, self.fovy], dim=1)
        else:
            _fx = None
            _fy = None
            if "fx" in kwargs and "fy" in kwargs:
                _fx = float(kwargs["fx"])
                _fy = float(kwargs["fy"])
            elif "fl_x" in kwargs and "fl_y" in kwargs:
                _fx = float(kwargs["fl_x"])
                _fy = float(kwargs["fl_y"])
            elif "focal_length" in kwargs:
                _fx, _fy = float(kwargs["focal_length"])
            if _fx is not None and _fy is not None:
                self.fx = torch.tensor(_fx, device=self.device, dtype=torch.float32).reshape(1, )
                self.fy = torch.tensor(_fy, device=self.device, dtype=torch.float32).reshape(1, )
                self.focal_length = torch.stack([self.fx, self.fy], dim=1)
                self.fovx = 2 * torch.atan(self.image_size[1] / (2 * self.fx))
                self.fovy = 2 * torch.atan(self.image_size[0] / (2 * self.fy))
                self.fov = torch.stack([self.fovx, self.fovy], dim=1)
            else:
                self.fx = self.fy = self.focal_length = None
                self.fovx = self.fovy = self.fov = None
            
            _cx = None
            _cy = None
            if "cx" in kwargs and "cy" in kwargs:
                _cx = float(kwargs["cx"])
                _cy = float(kwargs["cy"])
            elif "pp_x" in kwargs and "pp_y" in kwargs:
                _cx = float(kwargs["pp_x"])
                _cy = float(kwargs["pp_y"])
            if _cx is not None and _cy is not None:
                self.cx = torch.tensor(_cx, device=self.device, dtype=torch.float32).reshape(1, )
                self.cy = torch.tensor(_cy, device=self.device, dtype=torch.float32).reshape(1, )
            else:
                self.cx = torch.tensor(self.image_size[1] / 2, device=self.device, dtype=torch.float32).reshape(1, )
                self.cy = torch.tensor(self.image_size[0] / 2, device=self.device, dtype=torch.float32).reshape(1, )
                self.principal_point = torch.stack([self.cx, self.cy], dim=1)
            
            if self.focal_length is not None:
                self.K = torch.eye(4, device=self.device, dtype=torch.float32)[None]
                self.K[:, 0, 0] = self.fx
                self.K[:, 1, 1] = self.fy
                self.K[:, 0, 2] = self.cx
                self.K[:, 1, 2] = self.cy
            else:
                gsstudio.warning("Intrinsic is not properly set")
                self.K = torch.eye(4, device=self.device, dtype=torch.float32)[None]
                self.fx = self.K[:, 0, 0]
                self.fy = self.K[:, 1, 1]
                self.focal_length = torch.stack([self.fx, self.fy], dim=1)
                self.fovx = 2 * torch.atan(self.image_size[1] / (2 * self.fx))
                self.fovy = 2 * torch.atan(self.image_size[0] / (2 * self.fy))
                self.fov = torch.stack([self.fovx, self.fovy], dim=1)
                    
        # pose
        self.R = None
        self.T = None
        _extr = None
        if "extr" in kwargs:
            _extr = kwargs["extr"]
        elif "extrinsic" in kwargs:
            _extr = kwargs["extrinsic"]
        elif "transform_matrix" in kwargs:
            _extr = kwargs["transform_matrix"]
        if _extr is not None:
            _extr = torch.tensor(_extr, dtype=torch.float32)
            if _extr.numel() == 12:
                _extr = _extr.reshape(1, 3, 4)
                self.R = _extr[:, :3, :3]
                self.T = _extr[:, :3, 3]
            elif _extr.numel() == 16:
                _extr = _extr.reshape(1, 4, 4)
                self.R = _extr[:, :3, :3]
                self.T = _extr[:, :3, 3]
            else:
                gsstudio.warning("Invalid extrinsic")
        if self.R is None:
            _rot = None
            if "R" in kwargs:
                _rot = kwargs["R"]
            elif "rotation" in kwargs:
                _rot = kwargs["rotation"]
            if _rot is not None:
                _rot = torch.tensor(_rot, dtype=torch.float32)
                if _rot.numel() == 9:
                    self.R = _rot.reshape(1, 3, 3)
                elif _rot.numel() == 3:
                    self.R = euler_angles_to_matrix(_rot.reshape(1, 3))
                elif _rot.numel() == 4:
                    self.R = quaternion_to_matrix(_rot.reshape(1, 4))
                elif _rot.numel() == 6:
                    self.R = rotation_6d_to_matrix(_rot.reshape(1, 6))
                else:
                    gsstudio.warning("Invalid rotation")
                    self.R = _R  # default
            else:
                self.R = _R  # default
        if self.T is None:
            _trans = None
            if "T" in kwargs:
                _trans = kwargs["T"]
            elif "translation" in kwargs:
                _trans = kwargs["translation"]
            if _trans is not None:
                _trans = torch.tensor(_trans, dtype=torch.float32)
                if _trans.numel() == 9:
                    self.T = _trans.reshape(1, 3)
                else:
                    gsstudio.warning("Invalid translation")
                    self.T = _T  # default
            else:
                self.T = _T  # default
            
        self.R = self.R.to(self.device)
        self.T = self.T.to(self.device)
        
        if "z_near" in kwargs:
            self.znear = float(kwargs["z_near"])
        elif "near" in kwargs:
            self.znear = float(kwargs["near"])
        else:
            self.znear = 0.01
        if "z_far" in kwargs:
            self.zfar = float(kwargs["z_far"])
        elif "far" in kwargs:
            self.zfar = float(kwargs["far"])
        else:
            self.zfar = 1000.0
        
    def get_projection_transform(self) -> Float[Tensor, "B 4 4"]:
        """
        Calculate the projection matrix using the
        multi-view geometry convention.

        Args:

        Returns:
            a batch of projection matrices of shape (N, 4, 4). 
            
        """
        
        near_fx =  self.znear / self.fx
        near_fy = self.znear / self.fy
        left = - (self.width - self.cx) * near_fx
        right = self.cx * near_fx
        bottom = (self.cy - self.height) * near_fy
        top = self.cy * near_fy

        P = torch.zeros(self._N, 4, 4, dtype=torch.float32, device=self.device)
        z_sign = 1.0
        P[:, 0, 0] = 2.0 * self.znear / (right - left)
        P[:, 1, 1] = 2.0 * self.znear / (top - bottom)
        P[:, 0, 2] = (right + left) / (right - left)
        P[:, 1, 2] = (top + bottom) / (top - bottom)
        P[:, 3, 2] = z_sign
        P[:, 2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[:, 2, 3] = -(2 * self.zfar * self.znear) / (self.zfar - self.znear)
        
        return P       
    
    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        from_ndc: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            from_ndc: If `False` (default), assumes xy part of input is in
                NDC space if self.in_ndc(), otherwise in screen space. If
                `True`, assumes xy is in NDC space even if the camera
                is defined in screen space.
        """
        if world_coordinates:
            to_camera_transform = self.get_full_projection_transform(**kwargs)
        else:
            to_camera_transform = self.get_projection_transform(**kwargs)
        if from_ndc:
            to_camera_transform = to_camera_transform.compose(
                self.get_ndc_camera_transform()
            )

        unprojection_transform = to_camera_transform.inverse()
        xy_inv_depth = torch.cat(
            (xy_depth[..., :2], 1.0 / xy_depth[..., 2:3]), dim=-1  # type: ignore
        )
        return unprojection_transform.transform_points(xy_inv_depth)
    
    def get_ndc_to_screen_transform(
        self,
        with_xyflip: bool = False,
    ) -> Float[Tensor, "B 4 4"]:
        """
        NDC to screen conversion. Following original 3DGS. 
        ndc2Pix: ((v + 1.0) * S - 1.0) * 0.5

        Args:
            with_xyflip: flips x- and y-axis if set to True.

        K = [
                [s/2, 0,   0, (s-1)/2],
                [0,   s/2, 0, (s-1)/2],
                [0,   0,   1,       0],
                [0,   0,   0,       1],
        ]

        """
        
        K = torch.zeros((self._N, 4, 4), device=self.device, dtype=torch.float32)

        K[:, 0, 0] = 0.5 * self.width
        K[:, 1, 1] = 0.5 * self.height
        K[:, 0, 3] = (self.width - 1.0) / 2.0
        K[:, 1, 3] = (self.height - 1.0) / 2.0
        K[:, 2, 2] = 1.0
        K[:, 3, 3] = 1.0
        
        if with_xyflip:
            # flip x, y axis
            xyflip = torch.eye(4, device=self.device, dtype=torch.float32)[None]
            xyflip[0, 0] = -1.0
            xyflip[1, 1] = -1.0
            K = xyflip @ K
        return K

    def get_screen_to_ndc_transform(
        self,
        with_xyflip: bool = False
    ) -> Float[Tensor, "B 4 4"]:
        """
        Inverse of get_ndc_to_screen_transform

        """
        transform = self.get_ndc_to_screen_transform(
            with_xyflip=with_xyflip
        ).inverse()
        return transform

    def to_dict(self, device: Optional[Device] = None) -> Dict[str, Union[torch.Tensor, int, float, tuple]]:
        """
        Returns:
            a dict which contains all the data.
        """

        if device is None:
            device = self.device
        out = {}
        for _f in self._FIELDS:
            if hasattr(self, _f):
                val = getattr(self, _f)
                if isinstance(val, torch.Tensor):
                    out[_f] = val.to(device)
                else:
                    out[_f] = val
        for _f in self._SHARED_FIELDS:
            if hasattr(self, _f):
                val = getattr(self, _f)
                if isinstance(val, torch.Tensor):
                    out[_f] = val.to(device)
                else:
                    out[_f] = val
        
        out["proj_mtx"] = self.get_projection_transform().to(device)
        out["full_proj_mtx"] = self.get_full_projection_transform().to(device)
        out['c2w'] = self.get_view_to_world_transform().to(device)
                    
        return out
    
    def is_perspective(self):
        return True

    def in_ndc(self):
        return self._in_ndc


############################################################
#              Orthographic Camera Class                   #
############################################################


class OrthographicCameras(CamerasBase):
    pass


################################################
# Helper functions for world to view transforms
################################################


def get_world_to_view_transform(
    R: torch.Tensor = _R, T: torch.Tensor = _T
) -> Float[Tensor, "B 4 4"]:
    """
    This function returns a tensor representing the transformation
    matrix to go from world space to view space by applying a rotation and
    a translation.

    For camera extrinsic parameters R (rotation) and T (translation),
    we map a 3D point `X_world` in world coordinates to
    a point `X_cam` in camera coordinates with:
    `X_cam = X_world R + T`

    Args:
        R: (N, 3, 3) matrix representing the rotation.
        T: (N, 3) matrix representing the translation.

    Returns:
        a batch of projection matrices of shape (N, 4, 4)

    """

    if T.shape[0] != R.shape[0]:
        msg = "Expected R, T to have the same batch dimension; got %r, %r"
        gsstudio.error(msg % (R.shape[0], T.shape[0]))
        raise ValueError(msg % (R.shape[0], T.shape[0]))
    if T.dim() != 2 or T.shape[1:] != (3,):
        msg = "Expected T to have shape (N, 3); got %r"
        gsstudio.error(msg % repr(T.shape))
        raise ValueError(msg % repr(T.shape))
    if R.dim() != 3 or R.shape[1:] != (3, 3):
        msg = "Expected R to have shape (N, 3, 3); got %r"
        gsstudio.error(msg % repr(R.shape))
        raise ValueError(msg % repr(R.shape))
    
    transform = torch.eye(4, device=R.device, dtype=torch.float32).unsqueeze(0).repeat(R.shape[0], 1, 1)
    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    return transform


def camera_position_from_spherical_angles(
    distance: float,
    elevation: float,
    azimuth: float,
    degrees: bool = True,
    device: Device = "cpu",
) -> torch.Tensor:
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.

    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
            The inputs distance, elevation and azimuth can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N) or (1)
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.

    The vectors are broadcast against each other so they all have shape (N, 1).

    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    broadcasted_args = convert_to_tensors_and_broadcast(
        distance, elevation, azimuth, device=device
    )
    dist, elev, azim = broadcasted_args
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1)  # add batch dim.
    return camera_position.view(-1, 3)


def look_at_rotation(
    camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device: Device = "cpu"
) -> torch.Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.

    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices
    """
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast(
        camera_position, at, up, device=device
    )
    camera_position, at, up = broadcasted_args
    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)


def look_at_view_transform(
    dist: _BatchFloatType = 1.0,
    elev: _BatchFloatType = 0.0,
    azim: _BatchFloatType = 0.0,
    degrees: bool = True,
    eye: Optional[Union[Sequence, torch.Tensor]] = None,
    at=((0, 0, 0),),  # (1, 3)
    up=((0, 1, 0),),  # (1, 3)
    device: Device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function returns a rotation and translation matrix
    to apply the 'Look At' transformation from world -> view coordinates [0].

    Args:
        dist: distance of the camera from the object
        elev: angle in degrees or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        azim: angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azim is the angle between the projected vector and a
            reference vector at (0, 0, 1) on the reference plane (the horizontal plane).
        dist, elev and azim can be of shape (1), (N).
        degrees: boolean flag to indicate if the elevation and azimuth
            angles are specified in degrees or radians.
        eye: the position of the camera(s) in world coordinates. If eye is not
            None, it will override the camera position derived from dist, elev, azim.
        up: the direction of the x axis in the world coordinate system.
        at: the position of the object(s) in world coordinates.
        eye, up and at can be of shape (1, 3) or (N, 3).

    Returns:
        2-element tuple containing

        - **R**: the rotation to apply to the points to align with the camera.
        - **T**: the translation to apply to the points to align with the camera.

    References:
    [0] https://www.scratchapixel.com
    """

    if eye is not None:
        broadcasted_args = convert_to_tensors_and_broadcast(eye, at, up, device=device)
        eye, at, up = broadcasted_args
        C = eye
    else:
        broadcasted_args = convert_to_tensors_and_broadcast(
            dist, elev, azim, at, up, device=device
        )
        dist, elev, azim, at, up = broadcasted_args
        C = (
            camera_position_from_spherical_angles(
                dist, elev, azim, degrees=degrees, device=device
            )
            + at
        )

    R = look_at_rotation(C, at, up, device=device)
    T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
    return R, T


################################################
# Helper functions for transforming points
################################################

def transform_points(points_batch: torch.Tensor, transform_matrix: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    """
    Use this transform to transform a set of 3D points. Assumes row major
    ordering of the input points.

    Args:
        points_batch: Tensor of shape (P, 3) or (N, P, 3)
        transform_matrix: Tensor of shape (N, 4, 4) or (4, 4)
        eps: If eps!=None, the argument is used to clamp the
            last coordinate before performing the final division.
            The clamping corresponds to:
            last_coord := (last_coord.sign() + (last_coord==0)) *
            torch.clamp(last_coord.abs(), eps),
            i.e. the last coordinates that are exactly 0 will
            be clamped to +eps.

    Returns:
        points_out: points of shape (N, P, 3) or (P, 3) depending
        on the dimensions of the transform
    """
    p_dim = points_batch.dim()
    if p_dim == 2:
        points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
    if p_dim != 3:
        msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
        raise ValueError(msg % repr(points_batch.shape))

    N, P, _3 = points_batch.shape
    ones = torch.ones(N, P, 1, dtype=points_batch.dtype, device=points_batch.device)
    points_batch = torch.cat([points_batch, ones], dim=2)

    if transform_matrix.shape[-2:] != (4, 4):
        msg = "Expected transform_matrix to have shape (N, 4, 4); got %r"
        gsstudio.error(msg % repr(transform_matrix.shape))
        raise ValueError(msg % repr(transform_matrix.shape))
    if transform_matrix.ndim == 3:
        if transform_matrix.shape[0] == 1:
            composed_matrix = transform_matrix.expand(N, -1, -1)
        elif transform_matrix.shape[0] == N:
            composed_matrix = transform_matrix
        else:
            msg = "Expected transform_matrix to have shape (N, 4, 4); got %r"
            gsstudio.error(msg % repr(transform_matrix.shape))
            raise ValueError(msg % repr(transform_matrix.shape))
    elif transform_matrix.ndim == 2:
        composed_matrix = transform_matrix[None].expand(N, -1, -1)
    else:
        msg = "Expected transform_matrix to have shape (N, 4, 4); got %r"
        gsstudio.error(msg % repr(transform_matrix.shape))
        raise ValueError(msg % repr(transform_matrix.shape))
    
    points_out = torch.bmm(composed_matrix, points_batch.transpose(-2, -1)).transpose(-2, -1)  # => (N, P, 4)
    denom = points_out[..., 3:]  # denominator
    if eps is not None:
        denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
        denom = denom_sign * torch.clamp(denom.abs(), eps)
    points_out = points_out[..., :3] / denom

    # When transform is (1, 4, 4) and points is (P, 3) return
    # points_out of shape (P, 3)
    if points_out.shape[0] == 1 and p_dim == 2:
        points_out = points_out.squeeze(0)

    return points_out
