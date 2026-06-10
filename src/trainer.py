"""CA-LoRA training loop."""
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from .losses import compute_feature_losses
from .anchor_selector import (
    load_feature_extractor,
    extract_features,
    AnchorMixingScheduler,
)


class CALoRATrainer:
    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self._load_model()
        self._setup_lora()
        self._load_feature_extractor()

    def _load_model(self):
        model_id = self.config["model"]["pretrained_model"]
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float16
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float16
        ).to(self.device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

    def _setup_lora(self):
        lora_cfg = self.config["lora"]
        lora_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=0.0,
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()

    def _load_feature_extractor(self):
        pass

    @torch.no_grad()
    def _encode_prompt(self, prompt: str, batch_size: int = 1):
        tokens = self.tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0]

    @torch.no_grad()
    def _encode_images(self, pixel_values: torch.Tensor):
        latents = self.vae.encode(pixel_values.to(dtype=torch.float16)).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    @torch.no_grad()
    def precompute_features(self, image_paths: list, batch_size: int = 4) -> torch.Tensor:
        """Encode images to latent space and pool to feature vectors (batched)."""
        from PIL import Image as PILImage
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(self.config["model"]["resolution"],
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.config["model"]["resolution"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        all_features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [PILImage.open(p).convert("RGB") for p in batch_paths]
            tensors = torch.stack([transform(img) for img in images]).to(self.device, dtype=torch.float16)
            latents = self.vae.encode(tensors).latent_dist.sample() * self.vae.config.scaling_factor
            all_features.append(latents.flatten(start_dim=1).float())
            del tensors, latents
            torch.cuda.empty_cache()
        features = torch.cat(all_features, dim=0)
        return F.normalize(features, dim=-1)

    def train(
        self,
        dataloader: DataLoader,
        target_features: torch.Tensor,
        anchor_features: torch.Tensor,
        mixing_scheduler: AnchorMixingScheduler,
    ):
        train_cfg = self.config["training"]
        output_dir = Path(self.config["output"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        optimizer = torch.optim.AdamW(
            self.unet.parameters(), lr=train_cfg["learning_rate"]
        )

        num_steps = train_cfg["num_steps"]
        grad_accum = train_cfg["gradient_accumulation"]
        log_every = self.config["output"]["log_every"]
        save_every = self.config["output"]["save_every"]

        self.unet.train()
        data_iter = iter(dataloader)
        progress = tqdm(range(num_steps), desc="Training CA-LoRA")
        running_losses = {"denoise": 0, "anchor": 0, "diverse": 0}

        for step in progress:
            # Update mixing ratio
            ratio = mixing_scheduler.get_ratio(step)
            dataloader.dataset.set_anchor_ratio(ratio)

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            pixel_values = batch["pixel_values"].to(self.device, dtype=torch.float16)
            prompt = batch["prompt"][0]

            latents = self._encode_images(pixel_values)
            encoder_hidden_states = self._encode_prompt(prompt, latents.shape[0])

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=self.device,
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states
            ).sample
            denoise_loss = F.mse_loss(noise_pred.float(), noise.float())

            # Estimate x_0 from noise prediction in latent space (no VAE decode)
            alpha_prod = self.noise_scheduler.alphas_cumprod.to(self.device)[timesteps]
            alpha_prod = alpha_prod.view(-1, 1, 1, 1)
            pred_x0 = (noisy_latents - (1 - alpha_prod).sqrt() * noise_pred) / alpha_prod.sqrt()

            # Contrastive loss directly in latent space
            pred_features = F.normalize(pred_x0.float().flatten(start_dim=1), dim=-1)

            reg_losses = compute_feature_losses(
                pred_features,
                anchor_features,
                target_features,
                lambda_anchor=train_cfg["lambda_anchor"],
                lambda_diverse=train_cfg["lambda_diverse"],
                temperature=train_cfg["temperature"],
            )

            total_loss = denoise_loss + reg_losses["loss_reg_total"]
            total_loss = total_loss / grad_accum
            total_loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_losses["denoise"] += denoise_loss.item()
            running_losses["anchor"] += reg_losses["loss_anchor"].item()
            running_losses["diverse"] += reg_losses["loss_diverse"].item()

            if (step + 1) % log_every == 0:
                avg = {k: v / log_every for k, v in running_losses.items()}
                progress.set_postfix(
                    denoise=f"{avg['denoise']:.4f}",
                    anchor=f"{avg['anchor']:.4f}",
                    diverse=f"{avg['diverse']:.4f}",
                    mix_ratio=f"{ratio:.2f}",
                )
                running_losses = {k: 0 for k in running_losses}

            if (step + 1) % save_every == 0:
                self.save_lora(output_dir / f"checkpoint-{step+1}")

        self.save_lora(output_dir / "final")
        print(f"Training complete. Model saved to {output_dir / 'final'}")

    def save_lora(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(path)
        print(f"LoRA saved to {path}")

    @torch.no_grad()
    def generate(self, prompt: str, num_images: int = 8, seed: int = 0):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.config["model"]["pretrained_model"],
            unet=self.unet,
            torch_dtype=torch.float16,
        ).to(self.device)

        generator = torch.Generator(device=self.device)
        images = []
        for i in range(num_images):
            generator.manual_seed(seed + i)
            img = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
            images.append(img)

        del pipe
        torch.cuda.empty_cache()
        return images
