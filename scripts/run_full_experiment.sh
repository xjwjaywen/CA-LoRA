#!/bin/bash
# Run complete experiments across datasets for the paper
# Designed for 2x L20 GPUs running in parallel

echo "============================================"
echo "CA-LoRA Full Experiment Suite"
echo "============================================"

# --- GPU 0: CUB-200 experiments ---
echo "Starting CUB-200 experiments on GPU 0..."
(
    export CUDA_VISIBLE_DEVICES=0

    # 10-shot experiments
    for method in lora_only dreambooth_anchor ca_lora; do
        python train.py --dataset cub200 --class_name "001.Black_footed_Albatross" \
            --num_shots 10 --method $method --device cuda
    done

    # 5-shot experiments
    for method in lora_only dreambooth_anchor ca_lora; do
        python train.py --dataset cub200 --class_name "001.Black_footed_Albatross" \
            --num_shots 5 --method $method --device cuda
    done

    # Full ablation on 10-shot
    bash scripts/run_all_ablations.sh cub200 "001.Black_footed_Albatross" 10 0

    echo "GPU 0: CUB-200 experiments done!"
) &

# --- GPU 1: MVTec experiments ---
echo "Starting MVTec experiments on GPU 1..."
(
    export CUDA_VISIBLE_DEVICES=1

    # 10-shot experiments
    for method in lora_only dreambooth_anchor ca_lora; do
        python train.py --dataset mvtec --class_name "bottle" \
            --num_shots 10 --method $method --device cuda
    done

    # 5-shot experiments
    for method in lora_only dreambooth_anchor ca_lora; do
        python train.py --dataset mvtec --class_name "bottle" \
            --num_shots 5 --method $method --device cuda
    done

    # Full ablation on 10-shot
    bash scripts/run_all_ablations.sh mvtec bottle 10 1

    echo "GPU 1: MVTec experiments done!"
) &

wait
echo "============================================"
echo "All experiments complete!"
echo "============================================"
