"""Evaluation metrics: FID, LPIPS diversity, DINO similarity, CLIP score."""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from scipy import linalg


class Evaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._load_models()

    def _load_models(self):
        # DINOv2 for similarity
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.dino = self.dino.to(self.device).eval()
        self.dino_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # CLIP for text-image alignment
        import open_clip
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def _get_dino_features(self, images):
        if isinstance(images[0], (str, Path)):
            images = [Image.open(p).convert("RGB") for p in images]
        tensors = torch.stack([self.dino_transform(img) for img in images]).to(self.device)
        features = []
        for i in range(0, len(tensors), 32):
            features.append(self.dino(tensors[i:i+32]))
        return torch.cat(features, dim=0)

    @torch.no_grad()
    def _get_clip_image_features(self, images):
        if isinstance(images[0], (str, Path)):
            images = [Image.open(p).convert("RGB") for p in images]
        tensors = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
        features = self.clip_model.encode_image(tensors)
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def _get_clip_text_features(self, texts):
        tokens = self.clip_tokenizer(texts).to(self.device)
        features = self.clip_model.encode_text(tokens)
        return F.normalize(features, dim=-1)

    def compute_dino_similarity(self, generated_images, target_images) -> float:
        gen_feats = self._get_dino_features(generated_images)
        tgt_feats = self._get_dino_features(target_images)
        gen_feats = F.normalize(gen_feats, dim=-1)
        tgt_feats = F.normalize(tgt_feats, dim=-1)
        tgt_centroid = tgt_feats.mean(dim=0, keepdim=True)
        similarities = (gen_feats @ tgt_centroid.T).squeeze(-1)
        return similarities.mean().item()

    def compute_lpips_diversity(self, generated_images) -> float:
        feats = self._get_dino_features(generated_images)
        feats = F.normalize(feats, dim=-1)
        sim_matrix = feats @ feats.T
        mask = ~torch.eye(len(feats), dtype=torch.bool, device=self.device)
        pairwise_dist = 1 - sim_matrix[mask]
        return pairwise_dist.mean().item()

    def compute_clip_score(self, generated_images, prompt: str) -> float:
        img_feats = self._get_clip_image_features(generated_images)
        txt_feats = self._get_clip_text_features([prompt])
        scores = (img_feats @ txt_feats.T).squeeze(-1)
        return scores.mean().item()

    def compute_fid(self, generated_images, reference_images) -> float:
        gen_feats = self._get_dino_features(generated_images).cpu().numpy()
        ref_feats = self._get_dino_features(reference_images).cpu().numpy()

        mu_gen, sigma_gen = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)
        mu_ref, sigma_ref = ref_feats.mean(axis=0), np.cov(ref_feats, rowvar=False)

        diff = mu_gen - mu_ref
        covmean, _ = linalg.sqrtm(sigma_gen @ sigma_ref, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean)
        return float(fid)

    def evaluate_all(self, generated_images, target_images, prompt: str) -> dict:
        results = {}
        results["dino_similarity"] = self.compute_dino_similarity(generated_images, target_images)
        results["lpips_diversity"] = self.compute_lpips_diversity(generated_images)
        results["clip_score"] = self.compute_clip_score(generated_images, prompt)
        results["fid"] = self.compute_fid(generated_images, target_images)

        print("\n=== Evaluation Results ===")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        return results
