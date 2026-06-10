"""Main training script for CA-LoRA."""
import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.anchor_generator import generate_anchors
from src.anchor_selector import (
    load_feature_extractor,
    extract_features,
    select_anchors,
    AnchorMixingScheduler,
)
from src.dataset import FewShotDataset, DATASET_LOADERS
from src.trainer import CALoRATrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--num_shots", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="ca_lora",
                        choices=["ca_lora", "lora_only", "dreambooth_anchor", "ca_lora_no_diverse",
                                 "ca_lora_no_adaptive", "ca_lora_random_anchor"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override pretrained model path (local or HuggingFace ID)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.class_name:
        config["dataset"]["class_name"] = args.class_name
    if args.num_shots:
        config["dataset"]["num_shots"] = args.num_shots
    if args.model_path:
        config["model"]["pretrained_model"] = args.model_path

    dataset_name = config["dataset"]["name"]
    class_name = config["dataset"]["class_name"]
    num_shots = config["dataset"]["num_shots"]
    data_dir = config["dataset"]["data_dir"]

    method = args.method
    config["output"]["output_dir"] = f"./outputs/{dataset_name}/{class_name}/{method}_shot{num_shots}"

    # --- Step 1: Load few-shot target images ---
    print(f"\n{'='*60}")
    print(f"CA-LoRA Training | method={method} | {dataset_name}/{class_name} | {num_shots}-shot")
    print(f"{'='*60}")

    loader_fn = DATASET_LOADERS[dataset_name]
    real_paths, concept = loader_fn(data_dir, class_name, num_shots)
    if not real_paths:
        print("ERROR: No images loaded. Check your data directory.")
        return

    # --- Step 2: Generate anchor images ---
    anchor_dir = f"./data/anchors/{dataset_name}/{class_name}"

    if method == "lora_only":
        anchor_paths = []
    elif method == "dreambooth_anchor":
        from src.anchor_generator import generate_class_anchors
        anchor_path = Path(f"./data/anchors/{dataset_name}/{class_name}_class")
        if not list(anchor_path.glob("*.png")):
            generate_class_anchors(
                config["model"]["pretrained_model"],
                concept,
                config["anchor"]["num_anchors"],
                str(anchor_path),
                device=args.device,
            )
        anchor_paths = sorted([str(p) for p in anchor_path.glob("*.png")])
    else:
        anchor_path = Path(anchor_dir)
        if not list(anchor_path.glob("*.png")):
            generate_anchors(
                config["model"]["pretrained_model"],
                concept,
                config["anchor"]["num_anchors"],
                anchor_dir,
                num_inference_steps=config["anchor"]["generation_steps"],
                guidance_scale=config["anchor"]["guidance_scale"],
                resolution=config["model"]["resolution"],
                device=args.device,
            )
        anchor_paths = sorted([str(p) for p in anchor_path.glob("*.png")])

    # --- Step 3: Select anchors ---
    if anchor_paths and method not in ("lora_only",):
        feat_model, feat_transform = load_feature_extractor(
            config["training"]["feature_extractor"], args.device
        )
        target_feats = extract_features(
            real_paths, feat_model, feat_transform, args.device
        )
        anchor_feats = extract_features(
            anchor_paths, feat_model, feat_transform, args.device
        )

        if method == "ca_lora_random_anchor":
            strategy = "random"
        elif method == "ca_lora_no_adaptive":
            strategy = "clip_ranked"
        else:
            strategy = config["anchor"]["selection_strategy"]

        selected_idx, scores = select_anchors(
            target_feats, anchor_feats, strategy=strategy
        )
        anchor_paths = [anchor_paths[i] for i in selected_idx.tolist()]
        anchor_feats = anchor_feats[selected_idx]
        print(f"Selected {len(anchor_paths)} anchors (strategy={strategy})")

        del feat_model
        torch.cuda.empty_cache()
    else:
        target_feats = None
        anchor_feats = None

    # --- Step 4: Build dataset ---
    prompt = f"a photo of a {concept}"
    initial_ratio = config["training"]["anchor_ratio_start"] if method != "lora_only" else 0.0

    dataset = FewShotDataset(
        real_image_paths=real_paths,
        anchor_image_paths=anchor_paths,
        prompt=prompt,
        resolution=config["model"]["resolution"],
        anchor_ratio=initial_ratio,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # --- Step 5: Train ---
    trainer = CALoRATrainer(config, device=args.device)

    # Recompute features in latent space for training (anchor selection used DINOv2, training uses VAE latents)
    target_feats = trainer.precompute_features(real_paths)
    if anchor_paths:
        anchor_feats = trainer.precompute_features(anchor_paths)
    else:
        anchor_feats = torch.zeros(1, target_feats.shape[-1], device=args.device)

    # Ablation: disable diversity loss
    if method == "ca_lora_no_diverse":
        config["training"]["lambda_diverse"] = 0.0

    # Ablation: disable contrastive loss for pure LoRA
    if method == "lora_only":
        config["training"]["lambda_anchor"] = 0.0
        config["training"]["lambda_diverse"] = 0.0

    mixing_scheduler = AnchorMixingScheduler(
        total_steps=config["training"]["num_steps"],
        ratio_start=config["training"]["anchor_ratio_start"] if method != "lora_only" else 0.0,
        ratio_end=config["training"]["anchor_ratio_end"] if method != "lora_only" else 0.0,
        schedule=config["training"]["mixing_schedule"],
    )

    trainer.train(dataloader, target_feats, anchor_feats, mixing_scheduler)

    # --- Step 6: Generate & Evaluate ---
    print("\nGenerating evaluation images...")
    eval_cfg = config["evaluation"]
    generated = trainer.generate(prompt, num_images=eval_cfg["num_eval_images"])

    output_dir = Path(config["output"]["output_dir"]) / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(generated):
        img.save(output_dir / f"gen_{i:04d}.png")

    from src.evaluate import Evaluator
    evaluator = Evaluator(device=args.device)
    results = evaluator.evaluate_all(generated, real_paths, prompt)

    results_file = Path(config["output"]["output_dir"]) / "metrics.txt"
    with open(results_file, "w") as f:
        f.write(f"method: {method}\n")
        f.write(f"dataset: {dataset_name}/{class_name}\n")
        f.write(f"num_shots: {num_shots}\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
