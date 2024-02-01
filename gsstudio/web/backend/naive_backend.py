import json
import os


def get_suffix(s):
    t = s.split("@")[-1]
    return t


def get_number(s):
    t = s.split("t")[1]
    t, t2 = t.split("-")
    t2 = t2.split(".")[0]
    return int(t) * 1000 + int(t2)


def update():
    exp_list = []
    output_root = "/root/autodl-tmp/threestudio/custom/threestudio-3dgs/gsstudio/web/frontend/gsviewer/outputs"
    for exp_name in os.listdir(output_root):
        test_dir = os.path.join(output_root, exp_name)
        if not os.path.isdir(test_dir):
            continue
        for test_name in os.listdir(test_dir):
            if os.path.exists(
                os.path.join(
                    output_root, exp_name, test_name, "save", "point_cloud.ply"
                )
            ) or os.path.exists(
                os.path.join(
                    output_root, exp_name, test_name, "save", "point_cloud.splat"
                )
            ):
                exp_list.append(os.path.join(exp_name, test_name))
    exp_list.sort(key=get_suffix)
    exp_list.reverse()
    exp_json = {"results": []}
    print(exp_list)
    for exp in exp_list:
        img_name_list = []
        for img_name in os.listdir(os.path.join(output_root, exp, "save")):
            if img_name.endswith(".png"):
                img_name_list.append(img_name)
        img_name_list.sort(key=get_number)
        file_ext = ".splat"
        if os.path.exists(os.path.join(output_root, exp, "save", "point_cloud.ply")):
            file_ext = ".ply"
        url = os.path.join("outputs", exp, "save", "point_cloud" + file_ext)
        single_result = {
            "image": os.path.join("outputs", exp, "save", img_name_list[-1]),
            "url": url,
            "description": os.path.basename(exp),
        }
        exp_json["results"].append(single_result)

    with open(os.path.join(output_root, "exp.json"), "w") as f:
        json.dump(exp_json, f, indent=4)


import time

while True:
    print("update")
    update()
    time.sleep(5)
