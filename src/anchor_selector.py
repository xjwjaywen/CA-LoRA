"""Adaptive anchor selection via CLIP/DINO similarity ranking."""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms


def load_feature_extractor(name: str = "dinov2", device: str = "cuda"):
    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model = model.to(device).eval()
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform

    elif name == "clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model = model.to(device).eval()
        return model.visual, preprocess

    raise ValueError(f"Unknown feature extractor: {name}")


@torch.no_grad()
def extract_features(images, model, transform, device="cuda"):
    if isinstance(images[0], (str, Path)):
        images = [Image.open(p).convert("RGB") for p in images]

    tensors = torch.stack([transform(img) for img in images]).to(device)

    batch_size = 32
    features = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i : i + batch_size]
        feat = model(batch)
        if hasattr(feat, "last_hidden_state"):
            feat = feat.last_hidden_state[:, 0]
        features.append(feat)

    features = torch.cat(features, dim=0)
    features = F.normalize(features, dim=-1)
    return features


def select_anchors(
    target_features: torch.Tensor,
    anchor_features: torch.Tensor,
    strategy: str = "adaptive",
    top_k: int = None,
):
    """
    Select and rank anchors based on similarity to target images.

    Returns indices sorted by relevance and similarity scores.
    """
    target_centroid = F.normalize(target_features.mean(dim=0, keepdim=True), dim=-1)
    similarities = (anchor_features @ target_centroid.T).squeeze(-1)

    if strategy == "random":
        indices = torch.randperm(len(anchor_features))
        scores = torch.ones(len(anchor_features))

    elif strategy == "clip_ranked":
        scores, indices = similarities.sort(descending=True)

    elif strategy == "adaptive":
        # Select a mix: 60% similar (concept-relevant) + 40% diverse (prevent collapse)
        scores, ranked = similarities.sort(descending=True)
        n = len(anchor_features)
        n_similar = int(n * 0.6)

        similar_indices = ranked[:n_similar]

        remaining_indices = ranked[n_similar:]
        if len(remaining_indices) > 0:
            remaining_feats = anchor_features[remaining_indices]
            centroid_sim = similarities[remaining_indices]
            intra_sim = (remaining_feats @ remaining_feats.T).fill_diagonal_(0).mean(dim=1)
            diversity_score = (1 - intra_sim) * 0.7 + centroid_sim * 0.3
            _, div_order = diversity_score.sort(descending=True)
            diverse_indices = remaining_indices[div_order]
        else:
            diverse_indices = torch.tensor([], dtype=torch.long)

        indices = torch.cat([similar_indices, diverse_indices])
        scores = similarities[indices]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if top_k is not None and top_k < len(indices):
        indices = indices[:top_k]
        scores = scores[:top_k]

    return indices, scores


class AnchorMixingScheduler:
    """Dynamic anchor-to-real mixing ratio scheduler."""

    def __init__(
        self,
        total_steps: int,
        ratio_start: float = 0.7,
        ratio_end: float = 0.2,
        schedule: str = "linear",
    ):
        self.total_steps = total_steps
        self.ratio_start = ratio_start
        self.ratio_end = ratio_end
        self.schedule = schedule

    def get_ratio(self, step: int) -> float:
        progress = min(step / max(self.total_steps, 1), 1.0)

        if self.schedule == "linear":
            ratio = self.ratio_start + (self.ratio_end - self.ratio_start) * progress
        elif self.schedule == "cosine":
            ratio = self.ratio_end + (self.ratio_start - self.ratio_end) * (
                1 + np.cos(np.pi * progress)
            ) / 2
        elif self.schedule == "step":
            if progress < 0.33:
                ratio = self.ratio_start
            elif progress < 0.66:
                ratio = (self.ratio_start + self.ratio_end) / 2
            else:
                ratio = self.ratio_end
        else:
            ratio = self.ratio_start

        return float(ratio)
