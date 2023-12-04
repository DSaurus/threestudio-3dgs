# threestudio-3dgs
<img src="https://github.com/DSaurus/threestudio-3dgs/assets/24589363/ffe5bc88-4825-4e94-9b62-b944c76cc298" width="200" height="200">
<img src="https://github.com/DSaurus/threestudio-3dgs/assets/24589363/109b9e35-1e05-4f7c-bb87-4edbfb5feb1d" width="200" height="200">

The Gaussian Splatting extension for threestudio. This extension is writen by [Ruizhi Shao](https://github.com/DSaurus) and [Youtian Lin](https://github.com/Linyou). To use it, please install [threestudio](https://github.com/threestudio-project/threestudio) first and then install this extension in threestudio `custom` directory.

## Advanced Gaussian Splatting Installation (Recommend)
```
cd custom
git clone https://github.com/DSaurus/threestudio-3dgs.git
cd threestudio-3dgs
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
git clone https://github.com/DSaurus/simple-knn.git
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
```

## Native Gaussian Splatting Installation
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
# Native Gaussian Splatting + SDS Loss
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"

# Advanced Gaussian Splatting with background + SDS Loss
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_background.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"

# Advanced Gaussian Splatting with background and shading + SDS Loss
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_shading.yaml  --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"
```

## Gaussian Splatting + MVDream
Please first install [MVDream extension](https://github.com/DSaurus/threestudio-mvdream), then you can run the following script:
```
# Advanced Gaussian Splatting with background and shading + MVDream
python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_mvdream.yaml  --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse"
```

## Resume from checkpoints
```
# resume training from the last checkpoint, you may replace last.ckpt with any other checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
```

## Load from PLY (coming soon)

## Citation
```
@Article{kerbl3Dgaussians,
    author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal      = {ACM Transactions on Graphics},
    number       = {4},
    volume       = {42},
    month        = {July},
    year         = {2023},
    url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## Acknowledgement
Please also consider citing these work about 3D Gaussian Splatting generation. Their open-source code inspires this project..

```
@article{tang2023dreamgaussian,
    title={DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation},
    author={Tang, Jiaxiang and Ren, Jiawei and Zhou, Hang and Liu, Ziwei and Zeng, Gang},
    journal={arXiv preprint arXiv:2309.16653},
    year={2023}
}
```

```
@article{GaussianDreamer,
    title={GaussianDreamer: Fast Generation from Text to 3D Gaussian Splatting with Point Cloud Priors},
    author={Taoran Yi and Jiemin Fang and Guanjun Wu and Lingxi Xie and Xiaopeng Zhang and Wenyu Liu and Qi Tian and Xinggang Wang},
    journal={arxiv:2310.08529},
    year={2023}
}
```
