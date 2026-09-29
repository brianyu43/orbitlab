"""Fresh-development P1 ablation. No historical followup path is writable here."""
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
import research as r
from common import digest_tensor, state_digest
from decoder_study import ControlledDecoder
from evaluator_v2 import DetailedEvaluator, accepted
from models import from_variant
from p0 import THRESHOLD, dump, sha

RUN = HERE / "runs/p1_d770101_s0"
DATA = HERE / "data_v1"
LOCK = HERE / "execution_lock_v1.json"
CAP_SECONDS = 1800


def source_hashes() -> dict:
    return {name: sha(HERE / name) for name in ("p1_train.py", "models.py", "execution_lock_v1.json")}


def verify_previous(folder: Path, checkpoint: str) -> dict | None:
    report = folder / "run.json"
    if report.exists():
        info = json.loads(report.read_text())
        if sha(folder / checkpoint) != info["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {folder}")
        if info["source_hashes"] != source_hashes():
            raise ValueError(f"Source changed since {folder} was trained")
        return info
    if folder.exists() and not (folder / "progress.pt").is_file():
        raise FileExistsError(f"Incomplete run requires inspection: {folder}")
    return None


def save_progress(folder: Path, *, kind: str, step: int, seconds: float,
                  model, optimizer, sampler, logs: list, extra: dict | None = None) -> None:
    record = {"kind": kind, "step": step, "elapsed_seconds": seconds,
              "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
              "optimizer": optimizer.state_dict(), "sampler_state": sampler.get_state(),
              "torch_rng_state": torch.get_rng_state(), "logs": logs,
              "source_hashes": source_hashes(), "train_data_sha256": sha(DATA / "train.npz"),
              "extra": extra or {}}
    path = folder / "progress.pt"
    tmp = folder / "progress.pt.tmp"
    torch.save(record, tmp)
    tmp.replace(path)


def read_progress(folder: Path, kind: str):
    path = folder / "progress.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["kind"] != kind or state["source_hashes"] != source_hashes() or
            state["train_data_sha256"] != sha(DATA / "train.npz")):
        raise ValueError(f"Progress source mismatch: {path}")
    return state


def data(split: str):
    a = np.load(DATA / f"{split}.npz")
    images = torch.from_numpy(a["images"].copy()).permute(0, 3, 1, 2).float() / 255
    labels = torch.from_numpy(a["labels"].copy()).long()
    return images, labels


def ae_train() -> dict:
    folder = RUN / "ae"
    old = verify_previous(folder, "ae.pt")
    if old is not None:
        return old
    folder.mkdir(parents=True, exist_ok=True)
    o.seed_all(0)
    x, _ = data("train")
    model = r.ResearchAE("equivariant", 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(100000)
    progress = read_progress(folder, "ae")
    logs = [] if progress is None else progress["logs"]
    completed = 0 if progress is None else progress["step"]
    prior_seconds = 0 if progress is None else progress["elapsed_seconds"]
    if progress is not None:
        model.load_state_dict(progress["model"])
        optimizer.load_state_dict(progress["optimizer"])
        sampler.set_state(progress["sampler_state"])
        torch.set_rng_state(progress["torch_rng_state"])
        print("P1 AE resumed", completed, flush=True)
    start = time.perf_counter()
    step = completed
    for step in range(completed + 1, 6001):
        indices = torch.randint(len(x), (32,), generator=sampler)
        rotation = int(torch.randint(4, (1,), generator=sampler))
        target = o.rotate(x[indices], rotation)
        loss = o.reconstruction_loss(model(target), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("P1 AE loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(grad):
            raise FloatingPointError("P1 AE gradient")
        optimizer.step()
        if step % 500 == 0:
            elapsed = prior_seconds + time.perf_counter() - start
            logs.append({"step": step, "loss": float(loss.detach()), "seconds": elapsed})
            print("P1 AE", logs[-1], flush=True)
            save_progress(folder, kind="ae", step=step, seconds=elapsed,
                          model=model, optimizer=optimizer, sampler=sampler, logs=logs)
            if elapsed >= CAP_SECONDS:
                break
    seconds = prior_seconds + time.perf_counter() - start
    model.eval()
    vx, vy = data("val")
    evaluator = DetailedEvaluator()
    threshold = json.loads(THRESHOLD.read_text())
    rows = []
    recon = []
    with torch.no_grad():
        for rotation in range(4):
            for first in range(0, len(vx), 64):
                target = o.rotate(vx[first:first + 64], rotation)
                prediction = model(target)
                recon.append(prediction)
                metrics = r.pixel_metrics(prediction, target)
                for j, (p, label) in enumerate(zip(evaluator.predict(prediction), vy[first:first + 64].tolist())):
                    shape = p["predicted_shape"] == label[0]
                    color = p["predicted_color"] == label[1]
                    rows.append({"shape": shape, "color": color, "strict": shape and color and accepted(p, threshold),
                                 "foreground_mae": float(metrics["foreground_mae"][j]),
                                 "mask_iou": float(metrics["mask_iou"][j])})
    validation = {key: float(np.mean([v[key] for v in rows])) for key in rows[0]}
    gate = validation["shape"] >= .9 and validation["color"] >= .9 and \
        validation["foreground_mae"] <= .1 and validation["mask_iou"] >= .7
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "model": "equivariant", "width": 16, "channels": 16, "size": 64,
                  "seed": 0, "data_seed": 770101, "n": 1024, "steps": step}
    torch.save(checkpoint, folder / "ae.pt")
    o.grid(torch.cat([vx[:16], recon[0][:16]]), folder / "validation.png", 8)
    report = {"status": "completed" if step == 6000 else "budget_limited", "steps": step,
              "train_seconds": seconds, "validation": validation,
              "validation_gate_passed": gate, "training_data_sha256": sha(DATA / "train.npz"),
              "validation_data_sha256": sha(DATA / "val.npz"),
              "checkpoint_sha256": sha(folder / "ae.pt"), "source_hashes": source_hashes(),
              "loss_log": logs, "condition": "C4 AE with pixel-mean original decoder"}
    dump(folder / "run.json", report)
    return report


@torch.no_grad()
def pool() -> dict:
    if ae_train()["status"] != "completed":
        raise RuntimeError("AE training did not reach fixed step count")
    path = RUN / "pool.pt"
    ae_path = RUN / "ae/ae.pt"
    if path.exists():
        record = torch.load(path, map_location="cpu", weights_only=True)
        assert record["ae_sha256"] == sha(ae_path)
        assert record["data_sha256"] == sha(DATA / "train.npz")
        return record
    ae, _ = r.load_model(ae_path, torch.device("cpu"))
    x, y = data("train")
    encodings = []
    for rotation in range(4):
        for first in range(0, len(x), 64):
            encodings.append(ae.encode(o.rotate(x[first:first + 64], rotation)))
    z = torch.cat(encodings)
    labels = y.repeat(4, 1)
    mean = z.mean((0, 1), keepdim=True)
    std = z.std((0, 1), keepdim=True, unbiased=False).clamp_min(.02)
    normalized = (z - mean) / std
    record = {"z": normalized, "labels": labels, "mean": mean, "std": std,
              "ae_sha256": sha(ae_path), "data_sha256": sha(DATA / "train.npz"),
              "normalized_sha256": digest_tensor(normalized)}
    torch.save(record, path)
    return record


def decoder_train(kind: str) -> dict:
    if kind not in {"pixel_mean", "logit_mean"}:
        raise ValueError(kind)
    folder = RUN / f"decoder_{kind}"
    old = verify_previous(folder, "decoder.pt")
    if old is not None:
        return old
    item = pool()
    folder.mkdir(parents=True, exist_ok=True)
    o.seed_all(86000)
    model = ControlledDecoder(kind)
    initial = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(87000)
    x, _ = data("train")
    images = torch.cat([o.rotate(x, rotation) for rotation in range(4)])
    z = item["z"]
    progress = read_progress(folder, f"decoder_{kind}")
    logs = [] if progress is None else progress["logs"]
    completed = 0 if progress is None else progress["step"]
    prior_seconds = 0 if progress is None else progress["elapsed_seconds"]
    if progress is not None:
        if progress["extra"]["ae_sha256"] != item["ae_sha256"] or \
                progress["extra"]["latent_sha256"] != item["normalized_sha256"]:
            raise ValueError("Decoder resume pool changed")
        model.load_state_dict(progress["model"])
        optimizer.load_state_dict(progress["optimizer"])
        sampler.set_state(progress["sampler_state"])
        torch.set_rng_state(progress["torch_rng_state"])
        print("P1 decoder resumed", kind, completed, flush=True)
    start = time.perf_counter()
    step = completed
    for step in range(completed + 1, 6001):
        idx = torch.randint(len(z), (32,), generator=sampler)
        target = images[idx]
        loss = o.reconstruction_loss(model(z[idx]), target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"P1 decoder {kind} loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"P1 decoder {kind} gradient")
        optimizer.step()
        if step % 500 == 0:
            elapsed = prior_seconds + time.perf_counter() - start
            logs.append({"step": step, "loss": float(loss.detach()), "seconds": elapsed})
            print("P1 decoder", kind, logs[-1], flush=True)
            save_progress(folder, kind=f"decoder_{kind}", step=step, seconds=elapsed,
                          model=model, optimizer=optimizer, sampler=sampler, logs=logs,
                          extra={"ae_sha256": item["ae_sha256"],
                                 "latent_sha256": item["normalized_sha256"]})
            if elapsed >= CAP_SECONDS:
                break
    elapsed = prior_seconds + time.perf_counter() - start
    model.eval()
    ae, _ = r.load_model(RUN / "ae/ae.pt", torch.device("cpu"))
    vx, vy = data("val")
    ev = DetailedEvaluator()
    threshold = json.loads(THRESHOLD.read_text())
    val_rows = []
    example = None
    with torch.no_grad():
        for rotation in range(4):
            for first in range(0, len(vx), 64):
                target = o.rotate(vx[first:first+64], rotation)
                latent = (ae.encode(target) - item["mean"]) / item["std"]
                prediction = model(latent)
                if example is None:
                    example = torch.cat((target[:8], prediction[:8]))
                pixel = r.pixel_metrics(prediction, target)
                for j, (p, label) in enumerate(zip(ev.predict(prediction), vy[first:first+64].tolist())):
                    joint = p["predicted_shape"] == label[0] and p["predicted_color"] == label[1]
                    val_rows.append({"strict": joint and accepted(p, threshold),
                                     "shape": p["predicted_shape"] == label[0],
                                     "mask_iou": float(pixel["mask_iou"][j]),
                                     "foreground_mae": float(pixel["foreground_mae"][j])})
    validation = {key: float(np.mean([v[key] for v in val_rows])) for key in val_rows[0]}
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "kind": kind, "mean": item["mean"], "std": item["std"],
                  "seed": 0, "ae_sha256": item["ae_sha256"], "steps": step}
    torch.save(checkpoint, folder / "decoder.pt")
    o.grid(example, folder / "validation.png", 8)
    report = {"status": "completed" if step == 6000 else "budget_limited", "steps": step,
              "train_seconds": elapsed, "parameters": sum(p.numel() for p in model.parameters()),
              "initial_weights_sha256": initial, "latent_sha256": item["normalized_sha256"],
              "train_image_sha256": digest_tensor(images), "validation": validation,
              "checkpoint_sha256": sha(folder / "decoder.pt"), "source_hashes": source_hashes(),
              "loss_log": logs}
    dump(folder / "run.json", report)
    return report


def flow_train(variant: str) -> dict:
    folder = RUN / variant
    old = verify_previous(folder, "flow.pt")
    if old is not None:
        return old
    item = pool()
    folder.mkdir(parents=True, exist_ok=True)
    o.seed_all(81000)
    flow = from_variant(variant)
    initial = state_digest(flow)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=.001, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(82000)
    dropout_sampler = torch.Generator().manual_seed(83000)
    z, labels = item["z"], item["labels"]
    progress = read_progress(folder, variant)
    loss_log = [] if progress is None else progress["logs"]
    completed = 0 if progress is None else progress["step"]
    prior_seconds = 0 if progress is None else progress["elapsed_seconds"]
    if progress is not None:
        if progress["extra"]["ae_sha256"] != item["ae_sha256"] or \
                progress["extra"]["latent_sha256"] != item["normalized_sha256"]:
            raise ValueError("Flow resume pool changed")
        flow.load_state_dict(progress["model"])
        optimizer.load_state_dict(progress["optimizer"])
        sampler.set_state(progress["sampler_state"])
        dropout_sampler.set_state(progress["extra"]["dropout_rng_state"])
        torch.set_rng_state(progress["torch_rng_state"])
        print("P1 flow resumed", variant, completed, flush=True)
    start = time.perf_counter()
    step = completed
    for step in range(completed + 1, 4001):
        idx = torch.randint(len(z), (128,), generator=sampler)
        target = z[idx]
        noise = torch.randn(target.shape, generator=sampler)
        t = torch.rand(len(target), generator=sampler)
        point = (1 - t[:, None, None]) * noise + t[:, None, None] * target
        y = labels[idx].clone()
        if variant in {"F2", "F3"}:
            y[torch.rand(len(y), generator=dropout_sampler) < .1] = -1
        loss = F.mse_loss(flow(point, t, y), target - noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"P1 flow {variant} loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"P1 flow {variant} gradient")
        optimizer.step()
        if step % 500 == 0:
            elapsed = prior_seconds + time.perf_counter() - start
            loss_log.append({"step": step, "loss": float(loss.detach()), "seconds": elapsed})
            print("P1 flow", variant, loss_log[-1], flush=True)
            save_progress(folder, kind=variant, step=step, seconds=elapsed,
                          model=flow, optimizer=optimizer, sampler=sampler, logs=loss_log,
                          extra={"ae_sha256": item["ae_sha256"],
                                 "latent_sha256": item["normalized_sha256"],
                                 "dropout_rng_state": dropout_sampler.get_state()})
            if elapsed >= CAP_SECONDS:
                break
    elapsed = prior_seconds + time.perf_counter() - start
    flow.eval()
    with torch.no_grad():
        test_z = torch.randn(8, 4, 16, generator=torch.Generator().manual_seed(770129))
        test_t = torch.rand(8, generator=torch.Generator().manual_seed(770130))
        test_y = torch.tensor([[j % 4, j % 6] for j in range(8)])
        c4_error = float((flow(o.rho(test_z, 1), test_t, test_y) -
                          o.rho(flow(test_z, test_t, test_y), 1)).abs().max())
    if c4_error > 1e-5:
        raise AssertionError("C4 velocity error")
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in flow.state_dict().items()},
                  "variant": variant, "mean": item["mean"], "std": item["std"],
                  "ae_sha256": item["ae_sha256"], "steps": step}
    torch.save(checkpoint, folder / "flow.pt")
    report = {"status": "completed" if step == 4000 else "budget_limited", "steps": step,
              "train_seconds": elapsed, "parameters": sum(p.numel() for p in flow.parameters()),
              "initial_weights_sha256": initial, "latent_sha256": item["normalized_sha256"],
              "flow_c4_max_abs": c4_error, "conditioning": flow.kind,
              "dropout_probability": .1 if variant in {"F2", "F3"} else 0,
              "rng": {"initial_weights": 81000, "batch_noise_time": 82000,
                      "condition_dropout": 83000 if variant in {"F2", "F3"} else None},
              "checkpoint_sha256": sha(folder / "flow.pt"), "source_hashes": source_hashes(),
              "loss_log": loss_log}
    dump(folder / "run.json", report)
    return report


def check_p0():
    required = ["p0_saved_replay_v1.json", "p0_data_c4_v1.json", "p0_clean_renderer_v1.json",
                "p0_solver_v1.json", "p0_latent_v1.json", "p0_diversity_reader_lock_v1.json"]
    for name in required:
        if not (HERE / "reports" / name).is_file():
            raise RuntimeError(f"P0 incomplete: {name}")
    audit = json.loads((HERE / "reports/p0_data_c4_v1.json").read_text())
    reader = json.loads((HERE / "reports/p0_diversity_reader_lock_v1.json").read_text())
    if not audit["c4_under_target_1e_5"] or not reader["orientation_accepted"]:
        raise RuntimeError("P0 numerical or diversity reader gate failed")
    if not (HERE / "runs/tiny_v1/flow/run.json").is_file():
        raise RuntimeError("P0 tiny diagnostics not finished")
    lock = json.loads(LOCK.read_text())
    if lock["data_manifest_sha256"] != sha(DATA / "manifest.json"):
        raise RuntimeError("Development data changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["ae", "decoder", "flow", "all"])
    parser.add_argument("--kind", choices=["pixel_mean", "logit_mean"])
    parser.add_argument("--variant", choices=["F0", "F1", "F2", "F3"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    check_p0()
    if args.stage in ("ae", "all"):
        print("AE", json.dumps(ae_train(), ensure_ascii=False), flush=True)
    if args.stage in ("decoder", "all"):
        for kind in ([args.kind] if args.kind else ["pixel_mean", "logit_mean"]):
            print("decoder", kind, json.dumps(decoder_train(kind), ensure_ascii=False), flush=True)
    if args.stage in ("flow", "all"):
        for variant in ([args.variant] if args.variant else ["F0", "F1", "F2", "F3"]):
            print("flow", variant, json.dumps(flow_train(variant), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
