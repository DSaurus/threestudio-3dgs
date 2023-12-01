# threestudio-3dgs
<img src="https://github.com/DSaurus/threestudio-3dgs/assets/24589363/ffe5bc88-4825-4e94-9b62-b944c76cc298" width="200" height="200">
<img src="https://github.com/DSaurus/threestudio-3dgs/assets/24589363/109b9e35-1e05-4f7c-bb87-4edbfb5feb1d" width="200" height="200">

The Gaussian Splatting extension for threestudio. This extension is writen by [Ruizhi Shao](https://github.com/DSaurus) and [Youtian Lin](https://github.com/Linyou). To use it, please install [threestudio](https://github.com/threestudio-project/threestudio) first and then install this extension in threestudio `custom` directory.

## Installation
```
cd custom
git clone https://github.com/DSaurus/threestudio-3dgs.git
cd threestudio-3dgs
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting/submodules
python -m pip install diff-gaussian-rasterization/.
python -m pip install simple-knn/
```

## Quick Start
```
# Gaussian Splatting + SDS Loss
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"
```

## Resume from checkpoints
```
# resume training from the last checkpoint, you may replace last.ckpt with any other checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
```

## Modules Documentation (Coming soon)
