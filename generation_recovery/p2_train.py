"""Bounded direct-pixel flow baseline; entered only after P1 validation."""
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
from torch.nn import functional as F
import orbitlab as o
from common import state_digest
from p0 import dump, sha
from pixel_model import PixelVelocity

DATA = HERE / "data_v1/train.npz"
RUN_ROOT = HERE / "runs/p2_d770101_s0"
P1_SUMMARY = HERE / "evaluation/p1_v1/summary.json"
STEPS = 4000
BATCH = 8
LR = .0003
SECONDS_CAP = 1800


def hashes():
    return {"p2_source_sha256": sha(Path(__file__)),
            "pixel_model_sha256": sha(HERE / "pixel_model.py"),
            "data_sha256": sha(DATA),
            "development_lock_sha256": sha(HERE / "execution_lock_v1.json")}


def train(mode):
    if mode not in ("plain", "c4"):
        raise ValueError(mode)
    if not P1_SUMMARY.exists():
        raise RuntimeError("Complete P1 validation before entering conditional P2")
    folder = RUN_ROOT / mode
    report_path = folder / "run.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["source_hashes"] != hashes() or report["checkpoint_sha256"] != sha(folder / "pixel_flow.pt"):
            raise ValueError("P2 saved run changed")
        return report
    progress_path = folder / "progress.pt"
    if folder.exists() and not progress_path.exists():
        raise FileExistsError("Incomplete P2 run needs inspection")
    folder.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA)
    x = torch.from_numpy(data["images"].copy()).permute(0, 3, 1, 2).float() / 255
    labels = torch.from_numpy(data["labels"].copy()).long()
    torch.manual_seed(91000)
    model = PixelVelocity(mode == "c4")
    initial = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(92000)
    start = time.perf_counter()
    logs = []
    steps = 0
    reason = "fixed_steps_completed"
    estimate = None
    elapsed_before = 0.0
    if progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=True)
        if progress["source_hashes"] != hashes() or progress["initial_raw_weights_sha256"] != initial:
            raise ValueError("P2 progress source or initialization changed")
        model.load_state_dict(progress["state_dict"])
        optimizer.load_state_dict(progress["optimizer"])
        rng.set_state(progress["rng_state"])
        steps = progress["step"]
        logs = progress["logs"]
        elapsed_before = progress["elapsed_seconds"]
        estimate = progress["estimate"]
        print("P2 resume", mode, steps, flush=True)
    for steps in range(steps + 1, STEPS + 1):
        indices = torch.randint(len(x), (BATCH,), generator=rng)
        rotations = torch.randint(4, (BATCH,), generator=rng)
        target = torch.stack([o.rotate(x[i:i+1], int(k))[0] for i, k in zip(indices.tolist(), rotations)])
        target = 2 * target - 1
        noise = torch.randn(target.shape, generator=rng)
        t = torch.rand(BATCH, generator=rng)
        point = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * target
        prediction = model(point, t, labels[indices])
        loss = F.mse_loss(prediction, target - noise)
        if not torch.isfinite(loss):
            raise FloatingPointError("P2 loss became nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(grad):
            raise FloatingPointError("P2 gradient became nonfinite")
        optimizer.step()
        if steps % 100 == 0:
            elapsed = elapsed_before + time.perf_counter() - start
            logs.append({"step": steps, "loss": float(loss.detach()), "seconds": elapsed})
            print("P2", mode, logs[-1], flush=True)
            if steps == 100:
                estimate = elapsed * STEPS / 100
            tmp_path = progress_path.with_suffix(".tmp")
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng_state": rng.get_state(), "step": steps, "logs": logs,
                        "elapsed_seconds": elapsed, "estimate": estimate,
                        "source_hashes": hashes(),
                        "initial_raw_weights_sha256": initial}, tmp_path)
            tmp_path.replace(progress_path)
            if steps == 100:
                if estimate > SECONDS_CAP:
                    reason = "preflight_estimated_over_30_minute_cap"
                    break
            if elapsed >= SECONDS_CAP:
                reason = "local_30_minute_cap"
                break
    seconds = elapsed_before + time.perf_counter() - start
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "optimizer": optimizer.state_dict(), "rng_state": rng.get_state(),
                "mode": mode, "steps": steps, "lr": LR, "batch": BATCH}, folder / "pixel_flow.pt")
    report = {"status": "completed" if steps == STEPS else "budget_limited", "reason": reason,
              "steps": steps, "requested_steps": STEPS, "batch": BATCH, "learning_rate": LR,
              "train_seconds": seconds, "pilot_estimated_full_seconds": estimate,
              "parameters": sum(p.numel() for p in model.parameters()),
              "initial_raw_weights_sha256": initial,
              "raw_evaluations_per_forward": 4 if mode == "c4" else 1,
              "condition": "shape/color at each residual block, [-1,1] image target, conditional flow MSE",
              "source_hashes": hashes(), "checkpoint_sha256": sha(folder / "pixel_flow.pt"),
              "loss_log": logs,
              "budget_interpretation": "A cap-limited model has not completed the proposed fixed-step experiment."}
    dump(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plain", "c4"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(train(args.mode), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
