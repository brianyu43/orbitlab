"""Freeze candidate and all confirmation configuration before data preparation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from p0 import CONFIRMATION_SEEDS, THRESHOLD, dump, sha
from p4_prepare import BASE, LOCK, PER_PAIR, REFERENCE_PER_PAIR


def create():
    if LOCK.exists():
        raise FileExistsError("Confirmation lock already exists; never revise after test access")
    verified = json.loads((HERE / "evaluation/p3_steps16k_v1/verification.json").read_text())
    if not verified["summary"]["verified"] or \
            not verified["summary"]["F0_16k_diversity_passed"] or \
            verified["summary"]["F3_16k_diversity_passed"]:
        raise ValueError("Development candidate selection evidence changed")
    p1 = json.loads((HERE / "evaluation/p1_v1/summary.json").read_text())
    p3 = json.loads((HERE / "evaluation/p3_steps16k_v1/F0_s1p0/results.json").read_text())
    base = next(x for x in p1["rows"] if x["variant"] == "F0" and
                x["scale"] == 1 and x["decoder"] == "logit_mean")
    if (p3["metrics"]["seen"]["strict_accepted_and_joint"] - base["seen_strict"] < .05 or
            p3["metrics"]["ood"]["strict_accepted_and_joint"] - base["ood_strict"] < .05 or
            not p3["diversity"]["passed"]):
        raise ValueError("Selected 16k F0 development gain or diversity missing")
    sources = {name: sha(ROOT / path) for name, path in {
        "p4_prepare.py": "generation_recovery/p4_prepare.py",
        "p4_train.py": "generation_recovery/p4_train.py",
        "p4_evaluate.py": "generation_recovery/p4_evaluate.py",
        "p1_train.py": "generation_recovery/p1_train.py",
        "p3_steps_train.py": "generation_recovery/p3_steps_train.py",
        "p1_evaluate.py": "generation_recovery/p1_evaluate.py",
        "models.py": "generation_recovery/models.py",
        "work/orbitlab.py": "work/orbitlab.py",
        "work/research.py": "work/research.py",
        "followup/decoder_study.py": "followup/decoder_study.py",
        "followup/evaluator_v2.py": "followup/evaluator_v2.py",
        "diversity_reader": "generation_recovery/reports/p0_diversity_reader_lock_v1.json",
        "P3_DECISION_KO.md": "generation_recovery/P3_DECISION_KO.md",
    }.items()}
    lock = {"status": "frozen_before_confirmation_data", "selection": {
                "baseline": "F0 concat conditional latent flow 4000 steps + logit mean decoder",
                "candidate": "same architecture and initialization continued to 16000 steps + same decoder",
                "guidance": 1.0, "solver": "Euler", "solver_steps": 64,
                "max_candidates": 1},
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "initializations": [0, 1, 2],
            "initialization_seed_offsets": [0, 1000, 2000],
            "noise_seeds": {str(seed): 990201 + i for i, seed in enumerate(CONFIRMATION_SEEDS)},
            "samples_per_pair": PER_PAIR,
            "reference_per_pair": REFERENCE_PER_PAIR,
            "n_pairs": 24, "n_samples_per_model_path": 24 * PER_PAIR,
            "metric": "strict accepted AND requested shape/color joint match",
            "success_thresholds": {"seen_absolute": .30, "ood_absolute": .20,
                                   "paired_seen_gain_pp": 10, "paired_ood_gain_pp": 5,
                                   "no_seed_regression_below_pp": -2,
                                   "seen_and_ood_color_min": .95,
                                   "diversity_coverage_min": .90,
                                   "diversity_tv_max": .20,
                                   "diversity_outside_max": .10,
                                   "strict_subset_min_per_pair": 32},
            "threshold_sha256": sha(THRESHOLD),
            "source_sha256": sources,
            "development_evidence_sha256": {
                "p1_summary": sha(HERE / "evaluation/p1_v1/summary.json"),
                "p3_F0_16k": sha(HERE / "evaluation/p3_steps16k_v1/F0_s1p0/results.json"),
                "p3_verification": sha(HERE / "evaluation/p3_steps16k_v1/verification.json"),
                "p2_pixel_pilot": sha(HERE / "runs/p2_d770101_s0/plain/validation_8_per_pair.json")},
            "confirmation_data_prepared_or_viewed_at_lock": False,
            "claim_limit": "Three fresh synthetic data seeds and three initializations, not human validation or task generalization."}
    dump(LOCK, lock)
    return lock


if __name__ == "__main__":
    lock = create()
    print(json.dumps({"status": lock["status"], "lock_sha256": sha(LOCK),
                      "seeds": lock["confirmation_seeds"]}))
