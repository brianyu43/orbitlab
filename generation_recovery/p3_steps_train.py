"""Continue the two locked P1 flows from step 4000 to step 16000."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "work")]

import torch
from torch.nn import functional as F
import orbitlab as o
from models import from_variant
from p0 import dump, sha
from p1_train import RUN as P1_RUN, DATA, pool, source_hashes as p1_sources

RUN = HERE / "runs/p3_steps16k_d770101_s0"
TARGET_STEPS = 16000


def hashes(variant: str) -> dict:
    return {"p3_train_sha256": sha(Path(__file__)),
            "p3_decision_sha256": sha(HERE / "P3_DECISION_KO.md"),
            "p1_sources": p1_sources(),
            "p1_progress_sha256": sha(P1_RUN / variant / "progress.pt"),
            "p1_checkpoint_sha256": sha(P1_RUN / variant / "flow.pt"),
            "pool_sha256": sha(P1_RUN / "pool.pt"),
            "training_data_sha256": sha(DATA / "train.npz")}


def save_progress(path, *, variant, step, model, optimizer, sampler, dropout_sampler,
                  logs, seconds, provenance):
    record = {"variant": variant, "step": step,
              "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
              "optimizer": optimizer.state_dict(), "sampler_state": sampler.get_state(),
              "dropout_rng_state": dropout_sampler.get_state(),
              "torch_rng_state": torch.get_rng_state(), "logs": logs,
              "elapsed_seconds": seconds, "source_hashes": provenance}
    tmp = path.with_suffix(".tmp")
    torch.save(record, tmp)
    tmp.replace(path)


def train(variant: str):
    if variant not in ("F0", "F3"):
        raise ValueError(variant)
    folder = RUN / variant
    report_path = folder / "run.json"
    provenance = hashes(variant)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["source_hashes"] != provenance or report["checkpoint_sha256"] != sha(folder / "flow.pt"):
            raise ValueError("Saved P3 flow changed")
        return report
    item = pool()
    parent = torch.load(P1_RUN / variant / "progress.pt", map_location="cpu", weights_only=True)
    parent_ck = torch.load(P1_RUN / variant / "flow.pt", map_location="cpu", weights_only=True)
    if parent["step"] != 4000 or parent["source_hashes"] != p1_sources() or \
            parent["train_data_sha256"] != provenance["training_data_sha256"] or \
            parent["extra"]["latent_sha256"] != item["normalized_sha256"] or \
            not all(torch.equal(parent["model"][k], v) for k, v in parent_ck["state_dict"].items()):
        raise ValueError("P1 continuation does not match its fixed 4000-step checkpoint")
    folder.mkdir(parents=True, exist_ok=True)
    model = from_variant(variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    sampler = torch.Generator()
    dropout_sampler = torch.Generator()
    progress_path = folder / "progress.pt"
    progress = (torch.load(progress_path, map_location="cpu", weights_only=True)
                if progress_path.exists() else None)
    state = parent if progress is None else progress
    if progress is not None and (progress["source_hashes"] != provenance or
                                 progress["variant"] != variant):
        raise ValueError("P3 continuation source changed")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    sampler.set_state(state["sampler_state"])
    dropout_sampler.set_state(state["extra"]["dropout_rng_state"] if progress is None
                              else state["dropout_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    completed = state["step"]
    logs = state["logs"] if progress is not None else []
    prior_seconds = state["elapsed_seconds"] if progress is not None else 0.0
    print("P3 start", variant, completed, flush=True)
    start = time.perf_counter()
    z, labels = item["z"], item["labels"]
    for step in range(completed + 1, TARGET_STEPS + 1):
        idx = torch.randint(len(z), (128,), generator=sampler)
        target = z[idx]
        noise = torch.randn(target.shape, generator=sampler)
        t = torch.rand(len(target), generator=sampler)
        point = (1 - t[:, None, None]) * noise + t[:, None, None] * target
        y = labels[idx].clone()
        if variant == "F3":
            y[torch.rand(len(y), generator=dropout_sampler) < .1] = -1
        loss = F.mse_loss(model(point, t, y), target - noise)
        if not torch.isfinite(loss):
            raise FloatingPointError("P3 flow loss nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(grad):
            raise FloatingPointError("P3 flow gradient nonfinite")
        optimizer.step()
        if step % 1000 == 0:
            seconds = prior_seconds + time.perf_counter() - start
            logs.append({"step": step, "loss": float(loss.detach()), "seconds": seconds})
            print("P3", variant, logs[-1], flush=True)
            save_progress(progress_path, variant=variant, step=step, model=model,
                          optimizer=optimizer, sampler=sampler,
                          dropout_sampler=dropout_sampler, logs=logs,
                          seconds=seconds, provenance=provenance)
    model.eval()
    with torch.no_grad():
        ztest = torch.randn(8, 4, 16, generator=torch.Generator().manual_seed(770129))
        ttest = torch.rand(8, generator=torch.Generator().manual_seed(770130))
        ytest = torch.tensor([[j % 4, j % 6] for j in range(8)])
        c4_error = float((model(o.rho(ztest, 1), ttest, ytest) -
                          o.rho(model(ztest, ttest, ytest), 1)).abs().max())
    if c4_error > 1e-5:
        raise AssertionError("P3 C4 velocity error")
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "variant": variant, "mean": item["mean"], "std": item["std"],
                  "ae_sha256": item["ae_sha256"], "steps": TARGET_STEPS}
    torch.save(checkpoint, folder / "flow.pt")
    report = {"status": "completed", "steps": TARGET_STEPS,
              "continued_from_steps": 4000, "additional_seconds": prior_seconds + time.perf_counter() - start,
              "parameters": sum(p.numel() for p in model.parameters()),
              "flow_c4_max_abs": c4_error,
              "checkpoint_sha256": sha(folder / "flow.pt"),
              "source_hashes": provenance, "loss_log": logs,
              "interpretation": "Same development data and initialization as P1; no fresh confirmation."}
    dump(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["F0", "F3", "all"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    for variant in (("F0", "F3") if args.variant == "all" else (args.variant,)):
        report = train(variant)
        print("P3 complete", variant, report["steps"], report["checkpoint_sha256"], flush=True)


if __name__ == "__main__":
    main()
