import argparse
import copy
import glob
import json
import math
import os

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from gsstudio.data.utils.camera_utils import perspective, samples2matrix
from gsstudio.data.utils.ray_utils import get_ray_directions, get_rays

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", default="dataset/test", type=str, help="path to dataset"
    )
    parser.add_argument(
        "-t", default=0.05, help="distance between two cameras", type=float
    )
    parser.add_argument("--width", default=512, help="width", type=int)
    parser.add_argument("--height", default=512, help="height", type=int)

    args = parser.parse_args()

    cam_json = {"camera_model": "OPENCV", "orientation_override": "none"}

    frames = []
    os.makedirs(args.dataroot, exist_ok=True)

    index = 0
    circle_x = args.t * 0.5
    for index in range(120):
        camera_positions = torch.zeros((1, 3))
        camera_positions[:, 2] = 1.0
        camera_centers = torch.zeros((1, 3))
        c2w, intrinsic, proj_mtx = samples2matrix(
            camera_positions, camera_centers, 20, 20, args.width, args.height
        )

        fx = intrinsic[:, 0, 0]
        fy = intrinsic[:, 1, 1]
        cx = intrinsic[:, 0, 2]
        cy = intrinsic[:, 1, 2]

        rad = index / 30 * 2 * np.pi

        c2w_right = c2w.clone()
        c2w_right[:, 0, 3] = circle_x - math.cos(rad) * args.t * 1.0
        c2w_right[:, 1, 3] += math.sin(rad) * args.t * 1.0

        intrinsic_right = intrinsic.clone()

        frame = {}
        frame["fl_x"] = fx.item()
        frame["fl_y"] = fy.item()
        frame["cx"] = cx.item()
        frame["cy"] = cy.item()
        frame["w"] = args.width
        frame["h"] = args.height
        frame["transform_matrix"] = c2w_right[0].detach().numpy().tolist()
        frame["moment"] = index // 4 / 30
        index += 1
        frames.append(frame)
        cam_json["frames"] = frames

    json.dump(
        cam_json, open(os.path.join(args.dataroot, "transforms.json"), "w"), indent=4
    )
