import argparse
import copy
import glob
import json
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
        "-l",
        "--left_imgs",
        help="path to all first (left) frames",
        default="datasets/Middlebury/MiddEval3/testH/*/im0.png",
    )
    parser.add_argument(
        "-r",
        "--right_imgs",
        help="path to all second (right) frames",
        default="datasets/Middlebury/MiddEval3/testH/*/im1.png",
    )
    parser.add_argument(
        "-d", "--disparity_npy", help="path to all disparity frames", default=""
    )
    parser.add_argument(
        "-t", default=0.05, help="distance between two cameras", type=float
    )
    parser.add_argument("-s", default=0, help="start frames", type=int)
    parser.add_argument("-e", default=0, help="end frames", type=int)

    args = parser.parse_args()
    left_images = sorted(glob.glob(args.left_imgs, recursive=True))[args.s : args.e]
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))[args.s : args.e]
    disparity_images = sorted(glob.glob(args.disparity_npy, recursive=True))[
        args.s : args.e
    ]
    print(f"Found {len(left_images)} images.")
    print(len(right_images))

    cam_json = {"camera_model": "OPENCV", "orientation_override": "none"}

    frames = []
    os.makedirs(os.path.join(args.dataroot, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.dataroot, "depths"), exist_ok=True)

    index = 0
    for imfile1, imfile2, dfile3 in tqdm(
        list(zip(left_images, right_images, disparity_images))
    ):
        img1 = cv2.imread(imfile1)
        img2 = cv2.imread(imfile2)

        camera_positions = torch.zeros((1, 3))
        camera_positions[:, 2] = 1.0
        camera_centers = torch.zeros((1, 3))
        c2w, intrinsic, proj_mtx = samples2matrix(
            camera_positions, camera_centers, 20, 20, img1.shape[1], img1.shape[0]
        )

        fx = intrinsic[:, 0, 0]
        fy = intrinsic[:, 1, 1]
        cx = intrinsic[:, 0, 2]
        cy = intrinsic[:, 1, 2]

        c2w_right = c2w.clone()
        c2w_right[:, 0, 3] += args.t

        intrinsic_right = intrinsic.clone()

        disparity = np.load(dfile3)
        depth = -fx * args.t / disparity
        depth = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(-1)

        ray_directions = get_ray_directions(
            img1.shape[0], img1.shape[1], (fx, fy), (cx, cy)
        )
        rays_o, rays_d = get_rays(ray_directions, c2w, normalize=False)
        pts = rays_o + rays_d * depth

        pts_color = img1.reshape(-1, 3)

        pts = pts.reshape(-1, 3)

        # projection test
        # pts_2d = perspective(c2w_right, intrinsic, pts.unsqueeze(0))
        # pts_2d = pts_2d.reshape(-1, 2).detach().numpy()

        # draw_img = np.zeros_like(img1)
        # for i in range(pts_2d.shape[0]):
        #     x, y = pts_2d[i]
        #     x = int(x)
        #     y = int(y)
        #     x = min(max(x, 0), draw_img.shape[1] - 1)
        #     y = min(max(y, 0), draw_img.shape[0] - 1)
        #     cv2.circle(draw_img, (x, y), 1, (int(pts_color[i, 0]), int(pts_color[i, 1]), int(pts_color[i, 2])), -1)
        # cv2.imwrite('.threestudio_cache/output.png', draw_img)
        # cv2.imwrite('.threestudio_cache/output2.png', img2)

        if index == 0:
            pts = pts.detach().numpy()
            pts_array = np.array(
                [tuple(x) + tuple(y) for x, y in zip(pts, pts_color)],
                dtype=[
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("blue", "u1"),
                    ("green", "u1"),
                    ("red", "u1"),
                ],
            )
            vertex = PlyElement.describe(pts_array, "vertex")
            PlyData([vertex], text=False).write(os.path.join(args.dataroot, "init.ply"))

        frame = {}
        frame["fl_x"] = fx.item()
        frame["fl_y"] = fy.item()
        frame["cx"] = cx.item()
        frame["cy"] = cy.item()
        frame["w"] = img1.shape[1]
        frame["h"] = img1.shape[0]
        frame["file_path"] = "./images/frame_%05d.jpg" % index
        frame["depth_path"] = "./depths/frame_%05d.npy" % index
        frame["transform_matrix"] = c2w[0].detach().numpy().tolist()
        frame["moment"] = index // 2 / 30
        index += 1
        frames.append(frame)

        frame2 = copy.deepcopy(frame)
        frame2["file_path"] = "./images/frame_%05d.jpg" % index
        frame2["transform_matrix"] = c2w_right[0].detach().numpy().tolist()
        frame["moment"] = index // 2 / 30
        index += 1
        frames.append(frame2)

        cam_json["frames"] = frames

        cv2.imwrite(
            os.path.join(args.dataroot, "images", "frame_%05d.jpg" % (index - 2)), img1
        )
        cv2.imwrite(
            os.path.join(args.dataroot, "images", "frame_%05d.jpg" % (index - 1)), img2
        )
        np.save(
            os.path.join(args.dataroot, "depths", "frame_%05d.npy" % (index - 2)),
            depth[0].detach().numpy()[:, :, 0],
        )
        # np.save(os.path.join(args.dataroot, "depths", "frame_%05d.npy" % (index - 1)), depth[0].detach().numpy()[:, :, 0])
        # if index >= 300:
        #     break
    json.dump(
        cam_json, open(os.path.join(args.dataroot, "transforms.json"), "w"), indent=4
    )
