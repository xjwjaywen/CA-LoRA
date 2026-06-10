"""Concept-aware anchor image generation."""
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from tqdm import tqdm


CONCEPT_TEMPLATES = [
    "a photo of a {concept}",
    "a {concept} in the wild",
    "a {concept} with a plain background",
    "a close-up photo of a {concept}",
    "a {concept} in a natural setting",
    "a {concept} from a different angle",
    "a bright photo of a {concept}",
    "a dark photo of a {concept}",
    "a {concept} in an urban environment",
    "a painting of a {concept}",
    "a sketch of a {concept}",
    "a {concept} surrounded by nature",
    "a professional photo of a {concept}",
    "a vintage photo of a {concept}",
    "a {concept} on a white background",
    "a {concept} on a colorful background",
]


def generate_anchors(
    pretrained_model: str,
    concept: str,
    num_anchors: int,
    output_dir: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    resolution: int = 512,
    seed: int = 42,
    device: str = "cuda",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model, torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    images = []
    generator = torch.Generator(device=device)

    for i in tqdm(range(num_anchors), desc="Generating anchors"):
        template = CONCEPT_TEMPLATES[i % len(CONCEPT_TEMPLATES)]
        prompt = template.format(concept=concept)
        generator.manual_seed(seed + i)

        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=resolution,
            width=resolution,
            generator=generator,
        ).images[0]

        image.save(output_path / f"anchor_{i:04d}.png")
        images.append(image)

    del pipe
    torch.cuda.empty_cache()
    print(f"Generated {len(images)} anchor images in {output_path}")
    return images


def generate_class_anchors(
    pretrained_model: str,
    class_name: str,
    num_anchors: int,
    output_dir: str,
    **kwargs,
):
    """DreamBooth-style generic class anchors (baseline comparison)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model, torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = kwargs.get("device", "cuda")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    prompt = f"a photo of a {class_name}"
    generator = torch.Generator(device=device)
    images = []

    for i in tqdm(range(num_anchors), desc="Generating class anchors"):
        generator.manual_seed(kwargs.get("seed", 42) + i)
        image = pipe(
            prompt,
            num_inference_steps=kwargs.get("num_inference_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            height=kwargs.get("resolution", 512),
            width=kwargs.get("resolution", 512),
            generator=generator,
        ).images[0]
        image.save(output_path / f"class_anchor_{i:04d}.png")
        images.append(image)

    del pipe
    torch.cuda.empty_cache()
    return images
