python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_mvdream.yaml  --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse"

python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"

python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_background.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"

python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_shading.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_shading.yaml  --train --gpu 0 system.prompt_processor.prompt="A 3D model of an adorable cottage with a thatched roof"

mvdream-sd21-rescale0.5-shading/an_astronaut_riding_a_horse@20231202-130002

python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger" resume=/root/autodl-tmp/threestudio/outputs/gs-sds-generation/a_delicious_hamburger@20231204-000725/ckpts/last.ckpt trainer.max_steps=1

python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger" system.geometry.geometry_convert_from=/root/autodl-tmp/threestudio/outputs/gs-sds-generation/a_delicious_hamburger@20231212-150742/save/point_cloud.ply system.geometry.load_ply_only_vertex=True

python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger" system.geometry.geometry_convert_from="shap-e:a delicious hamburger"

python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --export --gpu 0 system.prompt_processor.prompt="a delicious hamburger"  resume=/root/autodl-tmp/threestudio/outputs/gs-sds-generation/a_delicious_hamburger@20231219-233754/ckpts/last.ckpt

python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_zero123.yaml --train --gpu 0 data.image_path=./load/images/hamburger_rgba.png

python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_zero123.yaml --train --gpu 0 data.image_path=./load/images/anya_front_rgba.png
# import subprocess

# prompt_list = [
#     "a delicious hamburger",
#     "A DSLR photo of a roast turkey on a platter",
#     "A high quality photo of a dragon",
#     "A DSLR photo of a bald eagle",
#     "A bunch of blue rose, highly detailed",
#     "A 3D model of an adorable cottage with a thatched roof",
#     "A high quality photo of a furry corgi",
#     "A DSLR photo of a panda",
#     "a DSLR photo of a cat lying on its side batting at a ball of yarn",
#     "a beautiful dress made out of fruit, on a mannequin. Studio lighting, high quality, high resolution",
#     "a DSLR photo of a corgi wearing a beret and holding a baguette, standing up on two hind legs",
#     "a zoomed out DSLR photo of a stack of pancakes",
#     "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
# ]
# negative_prompt = "oversaturated color, ugly, tiling, low quality, noise, ugly pattern"

# gpu_id = 0
# max_steps = 10
# val_check = 1
# out_name = "gsgen_baseline"
# for prompt in prompt_list:
#     print(f"Running model on device {gpu_id}: ", prompt)
#     command = [
#         "python",
#         "launch.py",
#         "--config",
#         "configs/gaussian_splatting.yaml",
#         "--train",
#         f"system.prompt_processor.prompt={prompt}",
#         f"system.prompt_processor.negative_prompt={negative_prompt}",
#         f"name={out_name}",
#         "--gpu",
#         f"{gpu_id}",
#     ]
#     subprocess.run(command)
