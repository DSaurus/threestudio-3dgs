# Gaussian Splatting Studio

The overview of Gaussian Splatting Studio. The goal is bridging 3D/4D reconstruction and generation for general 3D content creation.

# Code Structure

**System**: build the pipeline of
**Dataset** -> **Representation** -> **Renderer** -> **Diffusion Model**/Multi-view Images -> **Losses**

## Representation -> `representation/`

Input: `batch["time"], batch["condition"]`

Output: `batch["xyz"], batch["rotation"], batch["scale"], batch["opacity"], batch["features"]`

Additional Output: `batch["normal"]`

### Basic Representation -> `representation/base/`
- [x] Gaussian Splatting Original Representation -> `representation/base/gaussian.py`
- [ ] Gaussian Splatting with PBR
- [ ] Dynamic Gaussian Splatting -> `representation/base/gaussian_dynamaic.py`
- [ ] MIP Gaussian Splatting
- [ ] Deformable Gaussian Splatting
### Avatar Representation -> `representation/avatar/`
- [ ] Body Utils
- [ ] Head Utils
- [ ] Full-body Avatar Gaussian Splatting
- [ ] Head Avatar Gaussian Splatting
### IO -> `representation/io/`
- [x] Load/Export PLY -> `representation/io/gaussian_io.py`
- [ ] Mesh Extraction -> `representation/io/mesh_utils.py`
    - [x] Marching Cube -> `representation/io/mesh_utils.py`
    - [ ] Poisson Reconstruction -> `representation/io/mesh_utils.py`


## Renderer -> `renderer/`

Input from 3DGS: `batch["xyz"], batch["rotation"], batch["scale"], batch["opacity"], batch["features"], batch["normal"]`

Input from Camera: `batch["camera"] (fov, projection matrix)`

Output: `batch["comp_rgb"], batch["comp_mask"], batch["comp_depth"], batch["comp_normal"]`

### Material -> `renderer/material/`
- [x] Basic Shading Material  `renderer/material/gaussian_material.py`
- [ ] PBR Material

### Background -> `renderer/background/`
- [x] Basic Background -> `renderer/background/gaussian_background.py`

### Rasterizers -> `renderer/rasterizer/`
- [x] Basic Gaussian Splatting Renderer -> `renderer/rasterizer/diff_gaussian_rasterizer.py`
- [x] Gaussian Splatting Renderer with Background -> `renderer/rasterizer/diff_gaussian_rasterizer_background.py`
- [x] Gaussian Splatting Renderer with Shading  -> `renderer/rasterizer/diff_gaussian_rasterizer_shading.py`
- [ ] Gaussian Splatting Renderer with PBR
- [ ] Renderer Utils -> `renderer/rasterizer/renderer_utils.py`
- [x] General Batch Renderer `renderer/rasterizer/batch_renderer.py`
- [ ] General Batch Renderer with time

## System -> `system/`

Training Gaussian Splatting system.

- [x] Basic Gaussian Splatting Generation -> `system/gs_generation.py`
- [ ] Basic Gaussian Splatting Reconstruction
- [ ] Dynamic Gaussian Splatting Generation
- [ ] Dynamic Gaussian Splatting Reconstruction
- [ ] Gaussian Splatting Generation with PBR
- [ ] Gaussian Splatting Reconstruction with PBR
- [ ] Human Avatar Reconstruction
- [ ] Human Avatar Generation
- [ ] Head Avatar Reconstruction
- [ ] Head Avatar Generation
- [x] Image-to-3D Generation -> `system/gs_zero123.py`
- [ ] Image-to-4D Generation
- [ ] Static Gaussian Splatting Editting with Diffusion Model
- [ ] Dynamic Gaussian Splatting Editting with Diffusion Model

## Dataset -> `dataset/`

- [x] Random Camera Dataset (text-to-3D) -> `dataset/uncond.py`
- [x] Random Multi-view Camera Dataset (MVDream) -> `dataset/uncond_multiview.py`
- [x] Image + Random Camera Dataset (image-to-3D) -> `dataset/image.py`
- [ ] Random Camera Dataset with Time (text-to-4D)
- [ ] Image + Random Camera Dataset with Time (image-to-4D)
- [ ] Video + Random Camera Dataset with Time (video-to-4D)
- [ ] Multi-view Camera Dataset (3D Reconstruction, 3D Editing)
- [ ] Multi-view Camera Dataset with Time (4D Reconstruction, 4D Editing)
- [ ] Human + Multi-view Camera Dataset (Human/Avatar Reconstruction)
- [ ] Head + Multi-view Camera Dataset (Head/Avatar Reconstruction)
- [ ] Human + Random Camera Dataset (text-to-3D avatar)

## Diffusion Model -> `diffusion/`
- [x] Stable Diffusion -> `diffusion/diffusers_stable_diffusion.py`
- [x] Stable Diffusion XL -> `diffusion/diffusers_stable_diffusion_xl.py`
- [x] ControlNet -> `diffusion/diffusers_controlnet.py`
- [ ] InstructPix2Pix
- [x] DeepFloyed -> `diffusion/diffusers_deepfloyed.py`
- [x] Stable Video Diffusion -> `diffusion/diffusers_stable_video_diffusion.py`
- [ ] ZeroScope
- [ ] Other Video Diffusion Models such as Animate Anyone
- [ ] MVDream
- [ ] ImageDream
- [ ] Zero123
- [ ] Stable Zero123

## Loss -> `loss/`
### Diffusion Loss -> `loss/sampler`
- [x] SDS loss -> `loss/sampler/sds.py`
- [x] ISM loss -> `loss/sampler/ism.py`
- [x] HiFA loss -> `loss/sampler/hifa.py`
- [x] CSD loss -> `loss/sampler/csd.py`
- [ ] VSD loss -> `loss/sampler/vsd.py`
- [x] DU loss -> `loss/sampler/du.py`
### Reconstruction Loss -> `loss/`
- [x] L1 loss -> `loss/general_loss.py`
- [x] Perceptual loss -> `loss/perceptual/perceptual_loss.py`
- [x] TV loss  -> `loss/general_loss.py`
- [ ] Sparsity loss
- [ ] Regularization loss
- [x] GAN loss -> `loss/gan/gan_loss.py`

## Viewer -> `viewer/`

- [ ] WebUI based viewer. Read ply and show it in real-time.

## Script -> `script/`

- [ ] Removing Background, Depth/Normal Estimation. -> `script/image_preprocess.py`
- [ ] Image to Video (SVD). -> `script/image_to_video.py`
- [ ] SMPL Estimation.

## Documentation

- [ ] Full Documentation.


# Supported Methods

## Reconstruction

### 4D Representation
- [x] [1] Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis
- [x] [2] 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering
- [ ] [3] Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle
- [ ] [4] CoGS: Controllable Gaussian Splatting
- [ ] [5] **Control4D: Efficient 4D Portrait Editing with Text**
### Avatar Representation
- [x] [6] **Animatable Gaussians: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling**
- [ ] [7] **Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians**
- [ ] [8] **GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians**
### General Representation
- [ ] [9] **GPS-Gaussian: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis**
### Geometry Representation
- [ ] [10] PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics
- [ ] [11] SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering
### Rendering Representation
- [ ] [12] Mip-Splatting Alias-free 3D Gaussian Splatting
- [ ] [13] Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing
- [ ] [14] GS-IR: 3D Gaussian Splatting for Inverse Rendering

## Generation
### Object Generation
- [x] [15] DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation
- [x] [16] GaussianDreamer: Fast Generation from Text to 3D Gaussian Splatting with Point Cloud Priors
- [x] [17] LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching
- [ ] [18] HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting
- [ ] [19] CG3D: Compositional Generation for Text-to-3D
### 4D Generation
- [ ] [20] Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models
- [ ] [21] DreamGaussian4D: Generative 4D Gaussian Splatting

### Scene Generation
- [ ] [22] LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes
### Editting
- [ ] [23] Segment Any 3D Gaussians
- [ ] [5] **Control4D: Efficient 4D Portrait Editing with Text**
### Diffusion-based Reconstruction
- [ ] [24] ReconFusion: 3D Reconstruction with Diffusion Priors
