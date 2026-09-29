"""Two bounded P0 tiny-set fit diagnostics, isolated from historical results."""
from __future__ import annotations

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
import research as r
from evaluator_v2 import DetailedEvaluator, accepted
from p0 import THRESHOLD, dump, sha
from p0_extended import integrate

TINY = HERE / "runs/tiny_v1"
DATA = HERE / "data_v1/train.npz"
AE_STEPS = 6000
FLOW_STEPS = 4000
CAP_SECONDS = 1800


def data():
    a = np.load(DATA)
    x = torch.from_numpy(a["images"].copy()).permute(0, 3, 1, 2).float() / 255
    labels = torch.from_numpy(a["labels"].copy())
    picked = []
    for kind in range(4):
        for color in range(6):
            if kind != color:
                found = ((labels[:, 0] == kind) & (labels[:, 1] == color)).nonzero().flatten()
                assert len(found)
                picked.append(int(found[0]))
    picked.extend(i for i in range(len(x)) if i not in picked and len(picked) < 64)
    assert len(picked) == 64
    return x, labels, picked


def ae_run():
    folder = TINY / "ae"
    report = folder / "run.json"
    if report.exists():
        record = json.loads(report.read_text())
        assert sha(folder / "ae.pt") == record["checkpoint_sha256"]
        return record
    if folder.exists():
        raise FileExistsError("Incomplete tiny AE folder; inspect before retry")
    folder.mkdir(parents=True)
    torch.manual_seed(770111)
    x, y, picked = data()
    tiny = x[picked]
    model = r.ResearchAE("equivariant", 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(770112)
    logs = []
    start = time.perf_counter()
    actual = 0
    for step in range(1, AE_STEPS + 1):
        idx = torch.randint(64, (32,), generator=rng)
        rotation = int(torch.randint(4, (1,), generator=rng))
        target = o.rotate(tiny[idx], rotation)
        prediction = model(target)
        loss = o.reconstruction_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("tiny AE loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(grad):
            raise FloatingPointError("tiny AE gradient")
        optimizer.step()
        actual = step
        if step % 500 == 0:
            logs.append({"step": step, "loss": float(loss.detach()), "seconds": time.perf_counter() - start})
            print("tiny AE", logs[-1], flush=True)
            if logs[-1]["seconds"] >= CAP_SECONDS:
                break
    elapsed = time.perf_counter() - start
    model.eval()
    ev = DetailedEvaluator()
    threshold = json.loads(THRESHOLD.read_text())
    recon, rotated_labels = [], []
    with torch.no_grad():
        for rotation in range(4):
            rotated = o.rotate(tiny, rotation)
            recon.append(model(rotated))
            rotated_labels.extend(y[picked].tolist())
    recon = torch.cat(recon)
    preds = ev.predict(recon)
    strict = [accepted(p, threshold) and p["predicted_shape"] == int(label[0]) and
              p["predicted_color"] == int(label[1]) for p, label in zip(preds, rotated_labels)]
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ck = {"state_dict": state, "size": 64, "model": "equivariant", "width": 16,
          "channels": 16, "seed": 770111, "data_seed": 770101, "n": 64,
          "steps": actual, "picked_indices": picked}
    torch.save(ck, folder / "ae.pt")
    o.grid(torch.cat([tiny[:16], recon[:16]]), folder / "reconstruction.png", 8)
    record = {"status": "completed" if actual == AE_STEPS else "budget_limited", "steps": actual,
              "requested_steps": AE_STEPS, "train_seconds": elapsed,
              "train_reconstruction_strict": float(np.mean(strict)),
              "train_reconstruction_strict_n": len(strict),
              "first_20_indices_one_per_seen_condition": picked[:20],
              "threshold_sha256": sha(THRESHOLD), "data_sha256": sha(DATA),
              "checkpoint_sha256": sha(folder / "ae.pt"), "logs": logs,
              "interpretation": "Train-set memorization only, not generalization."}
    dump(report, record)
    return record


def flow_run():
    ae_record = ae_run()
    folder = TINY / "flow"
    report = folder / "run.json"
    if report.exists():
        record = json.loads(report.read_text())
        assert sha(folder / "flow.pt") == record["checkpoint_sha256"]
        return record
    if folder.exists():
        raise FileExistsError("Incomplete tiny flow folder; inspect before retry")
    folder.mkdir(parents=True)
    ae, ae_ck = r.load_model(TINY / "ae/ae.pt", torch.device("cpu"))
    ae.requires_grad_(False)
    x, labels, picked = data()
    x = x[picked[:20]]
    y = labels[picked[:20]]
    with torch.no_grad():
        z = torch.cat([ae.encode(o.rotate(x, rotation)) for rotation in range(4)])
    ys = y.repeat(4, 1)
    mean = z.mean((0, 1), keepdim=True)
    std = z.std((0, 1), keepdim=True, unbiased=False).clamp_min(.02)
    normalized = (z - mean) / std
    torch.manual_seed(770113)
    flow = o.Velocity(16, True)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=.001, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(770114)
    logs = []
    start = time.perf_counter()
    actual = 0
    for step in range(1, FLOW_STEPS + 1):
        idx = torch.randint(len(z), (128,), generator=rng)
        target = normalized[idx]
        noise = torch.randn(target.shape, generator=rng)
        t = torch.rand(len(idx), generator=rng)
        point = (1 - t[:, None, None]) * noise + t[:, None, None] * target
        loss = F.mse_loss(flow(point, t, ys[idx]), target - noise)
        if not torch.isfinite(loss):
            raise FloatingPointError("tiny flow loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1)
        if not torch.isfinite(grad):
            raise FloatingPointError("tiny flow gradient")
        optimizer.step()
        actual = step
        if step % 500 == 0:
            logs.append({"step": step, "loss": float(loss.detach()), "seconds": time.perf_counter() - start})
            print("tiny flow", logs[-1], flush=True)
            if logs[-1]["seconds"] >= CAP_SECONDS:
                break
    elapsed = time.perf_counter() - start
    flow.eval()
    gen = torch.Generator().manual_seed(770115)
    sample_y = y.repeat_interleave(8, 0)
    noise = torch.randn((len(sample_y), 4, 16), generator=gen)
    threshold = json.loads(THRESHOLD.read_text())
    images = []
    with torch.no_grad():
        for i in range(0, len(noise), 40):
            output = integrate(flow, noise[i:i+40], sample_y[i:i+40], "euler", 64)
            images.append(ae.decode(output * std + mean))
    images = torch.cat(images)
    preds = DetailedEvaluator().predict(images)
    strict = [accepted(p, threshold) and p["predicted_shape"] == int(label[0]) and
              p["predicted_color"] == int(label[1]) for p, label in zip(preds, sample_y)]
    shape = [p["predicted_shape"] == int(label[0]) for p, label in zip(preds, sample_y)]
    ck = {"state_dict": {k: v.detach().cpu() for k, v in flow.state_dict().items()},
          "ae_sha256": sha(TINY / "ae/ae.pt"), "mean": mean, "std": std,
          "steps": actual, "seed": 770113}
    torch.save(ck, folder / "flow.pt")
    o.grid(images[:80], folder / "train_condition_samples.png", 8)
    record = {"status": "completed" if actual == FLOW_STEPS else "budget_limited",
              "steps": actual, "requested_steps": FLOW_STEPS, "train_seconds": elapsed,
              "train_condition_generated_strict": float(np.mean(strict)),
              "train_condition_generated_shape": float(np.mean(shape)),
              "sample_n": len(noise), "conditions": 20,
              "sample_noise_seed": 770115, "data_sha256": sha(DATA),
              "ae_sha256": sha(TINY / "ae/ae.pt"),
              "checkpoint_sha256": sha(folder / "flow.pt"), "logs": logs,
              "interpretation": "Samples condition on labels of the 20 tiny training prototypes; not a held-out accuracy result."}
    dump(report, record)
    return record


if __name__ == "__main__":
    torch.set_num_threads(4)
    print("ae", json.dumps(ae_run(), ensure_ascii=False), flush=True)
    print("flow", json.dumps(flow_run(), ensure_ascii=False), flush=True)
