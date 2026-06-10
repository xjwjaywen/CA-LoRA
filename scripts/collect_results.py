"""Collect all experiment results into a summary table."""
import os
from pathlib import Path
from collections import defaultdict


def parse_metrics(metrics_file):
    results = {}
    with open(metrics_file) as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(": ", 1)
                try:
                    results[key] = float(val)
                except ValueError:
                    results[key] = val
    return results


def collect_results(output_root="./outputs"):
    all_results = []
    root = Path(output_root)

    for metrics_file in sorted(root.rglob("metrics.txt")):
        results = parse_metrics(metrics_file)
        all_results.append(results)

    if not all_results:
        print("No results found.")
        return

    # Group by dataset
    by_dataset = defaultdict(list)
    for r in all_results:
        key = f"{r.get('dataset', '?')}"
        by_dataset[key].append(r)

    metrics = ["fid", "lpips_diversity", "dino_similarity", "clip_score"]

    for dataset, results in by_dataset.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")
        header = f"{'Method':<25} {'Shots':>5}"
        for m in metrics:
            header += f" {m:>16}"
        print(header)
        print("-" * 80)

        for r in sorted(results, key=lambda x: (x.get("num_shots", ""), x.get("method", ""))):
            line = f"{r.get('method', '?'):<25} {r.get('num_shots', '?'):>5}"
            for m in metrics:
                val = r.get(m, None)
                if val is not None:
                    line += f" {val:>16.4f}"
                else:
                    line += f" {'N/A':>16}"
            print(line)


if __name__ == "__main__":
    collect_results()
