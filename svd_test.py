import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
)
image = load_image("/root/autodl-tmp/threestudio/load/images/anya_front.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(3321)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]
print(frames)
export_to_video(frames, "generated-anya.mp4", fps=7)
