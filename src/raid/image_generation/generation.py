import os
import json
import argparse
from tqdm import tqdm
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline

if torch.cuda.is_available(): device = "cuda"
else: device = "cpu"


def generate_image_flux(pipe, prompt):
    image = pipe(
        prompt,
        height=512,
        width=512,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image


def generate_image_sd(pipe, prompt):
    image = pipe(
        prompt,
        negative_prompt="",
        height=512,
        width=512,
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    return image


args = argparse.ArgumentParser()
args.add_argument("--idx", type=int, default=0, help="index of the job array")
args.add_argument("--tot_idx", type=int, default=10, help="index of the job array")
args.add_argument("--output_path", type=str, default="/work/horizon_ria_elsa/iccv_2025_elsa/cagliari")
args.add_argument("--flux", action="store_true", help="Use FLUX.1-dev model")
args.add_argument("--sd3", action="store_true", help="Use Stable Diffusion v1.5 model")
args = args.parse_args()

with open("image_generation/prompts.json", "r") as f:
    prompts = json.load(f)

if args.tot_idx == 0:
    step = len(prompts)
else:
    step = len(prompts) // args.tot_idx
if args.idx == args.tot_idx - 1:
    prompts = {k: prompts[k] for k in list(prompts.keys())[args.idx*step:]}  # get the prompt for the last job array index
else:
    prompts = {k: prompts[k] for k in list(prompts.keys())[args.idx*step:(args.idx+1)*step]}
print(f"Generating {len(prompts)} images for job array index {args.idx} out of {args.tot_idx}...")
# generate 4.8K images with flux and sd 3

if args.flux:
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    # pipe = pipe.to(device)
    args.output_path = os.path.join(args.output_path, "flux")
    os.makedirs(args.output_path, exist_ok=True)

elif args.sd3:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    args.output_path = os.path.join(args.output_path, "sd3")
    os.makedirs(args.output_path, exist_ok=True)
else:
    raise ValueError("Please specify either --flux or --sd3")

for el in tqdm(prompts):
    id = el
    prompt = prompts[el]
    
    if args.flux:
        image = generate_image_flux(pipe, [prompt])
        image.save(os.path.join(args.output_path, f"{id}_flux.png"))
    elif args.sd3:
        image = generate_image_sd(pipe, [prompt])
        image.save(os.path.join(args.output_path, f"{id}_sd3.png"))
        
print('done!')
