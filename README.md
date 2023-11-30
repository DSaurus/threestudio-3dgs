# threestudio-3dgs
The Gaussian Splatting extension for ThreeStudio. This extension is writen by [Ruizhi Shao](https://github.com/DSaurus) and [Youtian Lin](https://github.com/Linyou). To use it, please install [ThreeStudio](https://github.com/threestudio-project/threestudio) first and then install this extension in ThreeStudio `custom` directory.

# Installation
```
cd custom
git clone https://github.com/DSaurus/threestudio-3dgs.git
cd threestudio-3dgs
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting/submodules
python -m pip install diff-gaussian-rasterization/.
python -m pip install simple-knn/
```

# Quick Start
```
# Gaussian Splatting + SDS Loss
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"
```

# Resume from checkpoints
```
# resume training from the last checkpoint, you may replace last.ckpt with any other checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
```
