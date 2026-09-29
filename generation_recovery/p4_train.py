"""Three-by-three independent confirmation training using locked P1/P3 algorithms.

The P1 AE/decoder checkpoint writers embed development-only metadata fields.
This wrapper corrects those fields after training without changing tensor weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "work"), str(ROOT / "followup")]

import torch
import orbitlab as o
import p1_train as p1
import p3_steps_train as p3
from p0 import dump, sha
from p4_prepare import BASE, DATA, LOCK, locked

RUNS = BASE / "runs"
INITIALIZATIONS = (0, 1, 2)


def context_hashes(seed: int, init: int) -> dict:
    return {"p4_train_sha256": sha(Path(__file__)),
            "lock_sha256": sha(LOCK),
            "data_manifest_sha256": sha(DATA / "manifest.json"),
            "data_seed": seed, "initialization": init,
            "initialization_seed_offset": 1000 * init,
            "p1_train_sha256": sha(HERE / "p1_train.py"),
            "p3_steps_train_sha256": sha(HERE / "p3_steps_train.py"),
            "models_sha256": sha(HERE / "models.py"),
            "work_orbitlab_sha256": sha(ROOT / "work/orbitlab.py"),
            "work_research_sha256": sha(ROOT / "work/research.py"),
            "decoder_study_sha256": sha(ROOT / "followup/decoder_study.py")}


def correct_checkpoint(folder: Path, name: str, *, seed: int, init: int, context: dict):
    checkpoint_path = folder / name
    report_path = folder / "run.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text())
    if report["source_hashes"] != context:
        raise ValueError("P4 metadata correction source mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_seed = (0 if name == "ae.pt" else 86000) + 1000 * init
    changed = False
    if checkpoint.get("seed") != expected_seed:
        if sha(checkpoint_path) != report["checkpoint_sha256"]:
            raise ValueError("P4 checkpoint changed before metadata correction")
        checkpoint["seed"] = expected_seed
        changed = True
    if name == "ae.pt" and checkpoint.get("data_seed") != seed:
        if not changed and sha(checkpoint_path) != report["checkpoint_sha256"]:
            raise ValueError("P4 AE checkpoint changed before metadata correction")
        checkpoint["data_seed"] = seed
        changed = True
    if changed:
        temp = checkpoint_path.with_suffix(".corrected.tmp")
        torch.save(checkpoint, temp)
        temp.replace(checkpoint_path)
    if report["checkpoint_sha256"] != sha(checkpoint_path) or \
            not report.get("metadata_corrected_by_p4_wrapper"):
        report["checkpoint_sha256"] = sha(checkpoint_path)
        report["metadata_corrected_by_p4_wrapper"] = True
        report["actual_data_seed"] = seed
        report["actual_initialization_seed"] = expected_seed
        temp = report_path.with_suffix(".corrected.tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temp.replace(report_path)


def train_one(seed: int, init: int):
    lock = locked()
    if seed not in lock["confirmation_seeds"] or init not in INITIALIZATIONS:
        raise ValueError("Unselected P4 run")
    if lock["source_sha256"]["p4_train.py"] != sha(Path(__file__)):
        raise ValueError("P4 trainer changed after lock")
    if not (DATA / "manifest.json").exists():
        raise RuntimeError("Prepare frozen confirmation datasets first")
    folder = RUNS / f"d{seed}_s{init}"
    summary_path = folder / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["context_hashes"] != context_hashes(seed, init):
            raise ValueError("Saved P4 run source changed")
        return summary
    folder.mkdir(parents=True, exist_ok=True)
    original = {"p1_RUN": p1.RUN, "p1_DATA": p1.DATA,
                "p1_sources": p1.source_hashes, "seed_all": o.seed_all,
                "p3_P1_RUN": p3.P1_RUN, "p3_RUN": p3.RUN,
                "p3_sources": p3.p1_sources, "p3_hashes": p3.hashes}
    context = context_hashes(seed, init)
    p1_run = folder / "p1"
    p3_run = folder / "p3"
    try:
        p1.RUN = p1_run
        p1.DATA = DATA / f"d{seed}"
        p1.source_hashes = lambda: context
        o.seed_all = lambda value: original["seed_all"](value + 1000 * init)
        p3.P1_RUN = p1_run
        p3.RUN = p3_run
        p3.p1_sources = p1.source_hashes
        p3.hashes = lambda variant: {"p4_context": context,
                                     "parent_progress_sha256": sha(p1_run / variant / "progress.pt"),
                                     "parent_checkpoint_sha256": sha(p1_run / variant / "flow.pt"),
                                     "pool_sha256": sha(p1_run / "pool.pt")}
        correct_checkpoint(p1_run / "ae", "ae.pt", seed=seed, init=init, context=context)
        ae = p1.ae_train()
        if ae["status"] != "completed" or not ae["validation_gate_passed"]:
            raise RuntimeError("Confirmation AE did not pass prespecified reconstruction gate")
        correct_checkpoint(p1_run / "ae", "ae.pt", seed=seed, init=init, context=context)
        correct_checkpoint(p1_run / "decoder_logit_mean", "decoder.pt", seed=seed, init=init, context=context)
        decoder = p1.decoder_train("logit_mean")
        if decoder["status"] != "completed":
            raise RuntimeError("Confirmation decoder incomplete")
        correct_checkpoint(p1_run / "decoder_logit_mean", "decoder.pt", seed=seed, init=init, context=context)
        flow4 = p1.flow_train("F0")
        if flow4["status"] != "completed":
            raise RuntimeError("Confirmation 4k flow incomplete")
        flow16 = p3.train("F0")
        if flow16["status"] != "completed":
            raise RuntimeError("Confirmation 16k flow incomplete")
        summary = {"status": "trained", "data_seed": seed, "initialization": init,
                   "actual_initialization_seeds": {"ae": 1000 * init,
                                                    "decoder": 86000 + 1000 * init,
                                                    "flow": 81000 + 1000 * init},
                   "shared_sampler_rng_across_initializations": True,
                   "context_hashes": context,
                   "ae_sha256": sha(p1_run / "ae/ae.pt"),
                   "decoder_sha256": sha(p1_run / "decoder_logit_mean/decoder.pt"),
                   "flow4_sha256": sha(p1_run / "F0/flow.pt"),
                   "flow16_sha256": sha(p3_run / "F0/flow.pt"),
                   "metadata_correction": "P1 checkpoint static development fields corrected; tensor weights unchanged."}
        dump(summary_path, summary)
        return summary
    finally:
        p1.RUN, p1.DATA, p1.source_hashes = original["p1_RUN"], original["p1_DATA"], original["p1_sources"]
        o.seed_all = original["seed_all"]
        p3.P1_RUN, p3.RUN, p3.p1_sources, p3.hashes = (original["p3_P1_RUN"], original["p3_RUN"],
                                                       original["p3_sources"], original["p3_hashes"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    seeds = (args.seed,) if args.seed is not None else tuple(locked()["confirmation_seeds"])
    inits = (args.init,) if args.init is not None else INITIALIZATIONS
    for seed in seeds:
        for init in inits:
            result = train_one(seed, init)
            print("P4 trained", seed, init, result["flow16_sha256"], flush=True)


if __name__ == "__main__":
    main()
