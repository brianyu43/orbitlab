"""Recompute P3 development scores from immutable rows and replay samples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "work"), str(ROOT / "followup")]

import numpy as np
import torch
from evaluator_v2 import DetailedEvaluator
from models import from_variant
from p0 import dump, sha
from p1_evaluate import (NOISE_SEED, PER_PAIR, diversity, extract_features,
                         integrate, labels_all, reference)
from p3_steps_evaluate import EVAL, SETTINGS
from p3_steps_train import RUN, hashes


def main():
    torch.set_num_threads(4)
    out = EVAL / "verification.json"
    if out.exists():
        print(json.dumps(json.loads(out.read_text())["summary"]))
        return
    ref_x, ref_y = reference()
    real_rows = extract_features(ref_x, ref_y, DetailedEvaluator())
    labels = labels_all(PER_PAIR)
    noise = torch.randn((len(labels), 4, 16), generator=torch.Generator().manual_seed(NOISE_SEED))
    evidence = []
    result_by_variant = {}
    for variant, scale in SETTINGS:
        folder = EVAL / f"{variant}_s{str(scale).replace('.', 'p')}"
        result_path, rows_path, samples_path = (folder / name for name in ("results.json", "rows.json", "samples.pt"))
        result = json.loads(result_path.read_text())
        rows = json.loads(rows_path.read_text())
        samples = torch.load(samples_path, map_location="cpu", weights_only=True)
        if len(rows) != len(labels) or result["rows_sha256"] != sha(rows_path) or \
                result["samples_sha256"] != sha(samples_path) or \
                result["flow_checkpoint_sha256"] != sha(RUN / variant / "flow.pt") or \
                result["steps"] != 16000 or not torch.equal(samples["noise"], noise) or \
                not torch.equal(samples["labels"], labels):
            raise AssertionError("P3 sample or provenance mismatch")
        run = json.loads((RUN / variant / "run.json").read_text())
        if run["source_hashes"] != hashes(variant) or run["checkpoint_sha256"] != result["flow_checkpoint_sha256"]:
            raise AssertionError("P3 run mismatch")
        model = from_variant(variant)
        model.load_state_dict(torch.load(RUN / variant / "flow.pt", map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        with torch.no_grad():
            replay = integrate(model, noise[:8], labels[:8], scale)
        if not torch.allclose(replay, samples["latent"][:8], rtol=0, atol=1e-5):
            raise AssertionError("P3 first-eight latent replay mismatch")
        for split in ("seen", "ood"):
            subset = [r for r in rows if r["ood"] == (split == "ood")]
            for key in ("shape_correct", "color_correct", "joint_correct", "strict_accepted_and_joint", "template_iou"):
                if not np.isclose(np.mean([r[key] for r in subset]), result["metrics"][split][key], atol=1e-12):
                    raise AssertionError("P3 metric mismatch")
        if diversity(rows, real_rows) != result["diversity"]:
            raise AssertionError("P3 diversity mismatch")
        result_by_variant[variant] = result
        evidence.extend({"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
                        for p in (result_path, rows_path, samples_path, RUN / variant / "run.json", RUN / variant / "flow.pt"))
    baseline = result_by_variant["F0"]["metrics"]
    candidate = result_by_variant["F3"]["metrics"]
    comparison = {split: (candidate[split]["strict_accepted_and_joint"] - baseline[split]["strict_accepted_and_joint"]) * 100
                  for split in ("seen", "ood")}
    summary = {"verified": True, "settings": 2, "samples_each": len(labels),
               "F0_16k_diversity_passed": result_by_variant["F0"]["diversity"]["passed"],
               "F3_16k_diversity_passed": result_by_variant["F3"]["diversity"]["passed"],
               "F3_minus_F0_strict_pp": comparison,
               "claim_limit": "One development seed; no fresh-data confirmation."}
    dump(out, {"summary": summary, "checked_artifacts": evidence,
               "verification_source_sha256": sha(Path(__file__))})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
