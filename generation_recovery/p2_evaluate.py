"""Validation-only scoring of the direct-pixel conditional flow baseline."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "work"), str(ROOT / "followup")]

import numpy as np
import torch
import orbitlab as o
from evaluator_v2 import DetailedEvaluator
from p0 import dump, sha
from p1_evaluate import reference, extract_features, diversity, labels_all
from p2_train import RUN_ROOT, hashes
from pixel_model import PixelVelocity


@torch.no_grad()
def sample(model, noise, label):
    z = noise.clone()
    for k in range(64):
        z = z + model(z, torch.full((len(z),), k / 64), label) / 64
    return ((z + 1) / 2).clamp(0, 1)


def evaluate(mode: str, n_per_pair: int):
    if mode not in ("plain", "c4") or n_per_pair not in (8, 80):
        raise ValueError((mode, n_per_pair))
    folder = RUN_ROOT / mode
    run = json.loads((folder / "run.json").read_text())
    if n_per_pair == 80 and run["status"] != "completed":
        raise RuntimeError("Fixed-step run required for full validation")
    if run["source_hashes"] != hashes() or run["checkpoint_sha256"] != sha(folder / "pixel_flow.pt"):
        raise ValueError("P2 checkpoint changed")
    report_path = folder / f"validation_{n_per_pair}_per_pair.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["checkpoint_sha256"] != sha(folder / "pixel_flow.pt"):
            raise ValueError("P2 evaluation source changed")
        return report
    ck = torch.load(folder / "pixel_flow.pt", map_location="cpu", weights_only=True)
    model = PixelVelocity(mode == "c4")
    model.load_state_dict(ck["state_dict"])
    model.eval()
    labels = labels_all(n_per_pair)
    noise_seed = 770161 if n_per_pair == 8 else 770162
    noise = torch.randn((len(labels), 3, 64, 64), generator=torch.Generator().manual_seed(noise_seed))
    images = []
    start = time.perf_counter()
    for first in range(0, len(noise), 8):
        images.append(sample(model, noise[first:first+8], labels[first:first+8]))
        if (first // 8 + 1) % 24 == 0:
            print("P2 evaluate", mode, n_per_pair, first + len(images[-1]), "/", len(noise), flush=True)
    sample_seconds = time.perf_counter() - start
    images = torch.cat(images)
    evaluator = DetailedEvaluator()
    rows = extract_features(images, labels, evaluator)
    result = {}
    for split in ("seen", "ood"):
        selected = [v for v in rows if v["ood"] == (split == "ood")]
        result[split] = {"n": len(selected), **{key: float(np.mean([v[key] for v in selected]))
                                               for key in ("shape_correct", "color_correct", "joint_correct",
                                                           "strict_accepted_and_joint", "template_iou")}}
    diversity_result = None
    if n_per_pair == 80:
        real_x, real_y = reference()
        real = extract_features(real_x, real_y, evaluator)
        diversity_result = diversity(rows, real)
    o.grid(images[:48], folder / f"validation_{n_per_pair}_per_pair.png", 8)
    report = {"mode": mode, "training_status": run["status"], "train_steps": run["steps"],
              "n_per_pair": n_per_pair, "sample_n": len(noise), "noise_seed": noise_seed,
              "sample_seconds": sample_seconds, "raw_model_calls_per_sample": 64 * (4 if mode == "c4" else 1),
              "metrics": result, "diversity": diversity_result,
              "checkpoint_sha256": sha(folder / "pixel_flow.pt"),
              "scope": "Development validation; pilot of a budget-limited model is diagnostic only."
              }
    dump(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plain", "c4"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    for n in (8, 80):
        if n == 80 and json.loads((RUN_ROOT / args.mode / "run.json").read_text())["status"] != "completed":
            continue
        report = evaluate(args.mode, n)
        print("P2 score", args.mode, n, report["metrics"], flush=True)


if __name__ == "__main__":
    main()
