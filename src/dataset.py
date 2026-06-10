"""Few-shot dataset loading for CUB-200 and MVTec AD."""
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FewShotDataset(Dataset):
    """Few-shot dataset: loads N real images + anchor images with adaptive mixing."""

    def __init__(
        self,
        real_image_paths: list,
        anchor_image_paths: list,
        prompt: str,
        resolution: int = 512,
        anchor_ratio: float = 0.5,
    ):
        self.real_paths = real_image_paths
        self.anchor_paths = anchor_image_paths
        self.prompt = prompt
        self.anchor_ratio = anchor_ratio

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def set_anchor_ratio(self, ratio: float):
        self.anchor_ratio = ratio

    def __len__(self):
        return max(len(self.real_paths) * 100, 1000)

    def __getitem__(self, idx):
        use_anchor = random.random() < self.anchor_ratio and len(self.anchor_paths) > 0

        if use_anchor:
            path = random.choice(self.anchor_paths)
            is_anchor = True
        else:
            path = self.real_paths[idx % len(self.real_paths)]
            is_anchor = False

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return {
            "pixel_values": image,
            "prompt": self.prompt,
            "is_anchor": is_anchor,
        }


def prepare_cub200(data_dir: str, class_name: str, num_shots: int, seed: int = 42):
    """Load few-shot samples from CUB-200-2011."""
    data_path = Path(data_dir) / "CUB_200_2011" / "images"
    if not data_path.exists():
        print(f"CUB-200 not found at {data_path}")
        print("Download from: https://data.caltech.edu/records/65de6-vp158")
        return [], class_name

    class_dirs = sorted(data_path.iterdir())
    target_dir = None
    for d in class_dirs:
        if class_name.lower() in d.name.lower():
            target_dir = d
            break

    if target_dir is None:
        print(f"Class '{class_name}' not found. Available: {[d.name for d in class_dirs[:10]]}...")
        return [], class_name

    all_images = sorted(target_dir.glob("*.jpg"))
    rng = random.Random(seed)
    selected = rng.sample(all_images, min(num_shots, len(all_images)))

    concept = class_name.replace("_", " ").split(".")[-1].strip()
    print(f"CUB-200: loaded {len(selected)} images for '{concept}' from {target_dir.name}")
    return [str(p) for p in selected], concept


def prepare_mvtec(data_dir: str, class_name: str, num_shots: int, seed: int = 42):
    """Load few-shot samples from MVTec AD."""
    data_path = Path(data_dir) / "mvtec_anomaly_detection" / class_name / "train" / "good"
    if not data_path.exists():
        print(f"MVTec AD not found at {data_path}")
        print("Download from: https://www.mvtec.com/company/research/datasets/mvtec-ad")
        return [], class_name

    all_images = sorted(data_path.glob("*.png"))
    rng = random.Random(seed)
    selected = rng.sample(all_images, min(num_shots, len(all_images)))

    print(f"MVTec AD: loaded {len(selected)} images for '{class_name}'")
    return [str(p) for p in selected], class_name


def prepare_flowers(data_dir: str, class_name: str, num_shots: int, seed: int = 42):
    """Load few-shot samples from Oxford Flowers 102."""
    data_path = Path(data_dir) / "flowers102"
    if not data_path.exists():
        print(f"Flowers102 not found at {data_path}")
        return [], class_name

    all_images = sorted(data_path.glob(f"{class_name}/*.jpg"))
    rng = random.Random(seed)
    selected = rng.sample(all_images, min(num_shots, len(all_images)))

    print(f"Flowers102: loaded {len(selected)} images for '{class_name}'")
    return [str(p) for p in selected], class_name


DATASET_LOADERS = {
    "cub200": prepare_cub200,
    "mvtec": prepare_mvtec,
    "flowers": prepare_flowers,
}
