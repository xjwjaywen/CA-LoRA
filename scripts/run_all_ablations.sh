#!/bin/bash
# Run all ablation experiments for CA-LoRA paper
# Usage: bash scripts/run_all_ablations.sh [DATASET] [CLASS] [SHOTS] [GPU_ID]

DATASET=${1:-"cub200"}
CLASS=${2:-"001.Black_footed_Albatross"}
SHOTS=${3:-10}
GPU=${4:-0}

export CUDA_VISIBLE_DEVICES=$GPU

echo "============================================"
echo "Running CA-LoRA ablation suite"
echo "Dataset: $DATASET | Class: $CLASS | Shots: $SHOTS | GPU: $GPU"
echo "============================================"

METHODS=(
    "lora_only"              # Baseline: standard LoRA (no anchors)
    "dreambooth_anchor"      # Baseline: DreamBooth-style class anchors
    "ca_lora_random_anchor"  # Ablation: concept anchors + random selection
    "ca_lora_no_diverse"     # Ablation: concept anchors + contrastive, no diversity loss
    "ca_lora_no_adaptive"    # Ablation: concept anchors + contrastive + diversity, fixed ranking
    "ca_lora"                # Full method
)

for method in "${METHODS[@]}"; do
    echo ""
    echo ">>> Running: $method"
    echo "-------------------------------------------"
    python train.py \
        --config configs/default.yaml \
        --dataset "$DATASET" \
        --class_name "$CLASS" \
        --num_shots "$SHOTS" \
        --method "$method" \
        --device cuda
    echo "<<< Done: $method"
    echo ""
done

echo "============================================"
echo "All experiments complete!"
echo "Results in: outputs/$DATASET/$CLASS/"
echo "============================================"
