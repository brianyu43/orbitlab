"""Post-hoc consistency audit of the frozen confirmation, without reselection."""
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
from p1_evaluate import integrate, labels_all
from p4_prepare import BASE, DATA, LOCK, PER_PAIR, locked, prior_orbits
from p4_train import RUNS, INITIALIZATIONS, context_hashes
from p4_evaluate import (ATTRIBUTES, EVAL, KINDS, accepted_diversity,
                         full_diversity, reference)


def main():
    torch.set_num_threads(4)
    out = EVAL / "verification.json"
    if out.exists():
        report = json.loads(out.read_text())
        print(json.dumps({"verified": report["verified"], "model_paths": report["model_paths"]}))
        return
    lock = locked()
    for name, expected in lock["source_sha256"].items():
        path = (HERE / "reports/p0_diversity_reader_lock_v1.json" if name == "diversity_reader"
                else ROOT / name if "/" in name else HERE / name)
        if sha(path) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if sha(LOCK) != json.loads((DATA / "manifest.json").read_text())["lock_sha256"]:
        raise ValueError("Confirmation data lock mismatch")
    manifest = json.loads((DATA / "manifest.json").read_text())
    for item in manifest["files"]:
        if sha(DATA / item["path"]) != item["sha256"]:
            raise ValueError("Confirmation data file changed")
    all_orbits = []
    for seed in lock["confirmation_seeds"]:
        folder = DATA / f"d{seed}"
        for split in ("train", "val", "test", "ood", "reference"):
            metadata = json.loads((folder / f"{split}_metadata.json").read_text())
            all_orbits.extend(item["orbit_sha256"] for item in metadata)
    if len(all_orbits) != 33024 or len(set(all_orbits)) != len(all_orbits) or \
            set(all_orbits) & prior_orbits():
        raise AssertionError("Confirmation orbit uniqueness mismatch")
    summary = json.loads((EVAL / "summary.json").read_text())
    if summary["lock_sha256"] != sha(LOCK) or summary["data_manifest_sha256"] != sha(DATA / "manifest.json"):
        raise ValueError("Confirmation summary source mismatch")
    amendment_path = BASE / "execution_amendment_01.json"
    amendment = json.loads(amendment_path.read_text())
    if amendment["original_lock_sha256"] != sha(LOCK) or \
            amendment["wrapper_sha256"] != sha(HERE / "p4_train_v2.py"):
        raise ValueError("Execution amendment mismatch")
    evaluator = DetailedEvaluator()
    checked_paths = 0
    for seed in lock["confirmation_seeds"]:
        real = reference(seed, evaluator)
        for kind in KINDS:
            combined = []
            init_values = {split: [] for split in ("seen", "ood")}
            for init in INITIALIZATIONS:
                run = RUNS / f"d{seed}_s{init}"
                training = json.loads((run / "training_summary.json").read_text())
                if training["context_hashes"] != context_hashes(seed, init):
                    raise ValueError("Training source mismatch")
                if training.get("execution_amendment_sha256") != sha(amendment_path) or \
                        training.get("execution_wrapper_sha256") != sha(HERE / "p4_train_v2.py"):
                    raise ValueError("Training execution amendment missing")
                for key, path in (("ae_sha256", run / "p1/ae/ae.pt"),
                                  ("decoder_sha256", run / "p1/decoder_logit_mean/decoder.pt"),
                                  ("flow4_sha256", run / "p1/F0/flow.pt"),
                                  ("flow16_sha256", run / "p3/F0/flow.pt")):
                    if training[key] != sha(path):
                        raise ValueError("Training checkpoint mismatch")
                folder = EVAL / f"d{seed}_s{init}" / kind
                result = json.loads((folder / "results.json").read_text())
                rows = json.loads((folder / "rows.json").read_text())
                sample = torch.load(folder / "samples.pt", map_location="cpu", weights_only=True)
                flow_path = run / ("p1/F0/flow.pt" if kind == "baseline_4k" else "p3/F0/flow.pt")
                if len(rows) != 24 * PER_PAIR or result["rows_sha256"] != sha(folder / "rows.json") or \
                        result["samples_sha256"] != sha(folder / "samples.pt") or \
                        result["flow_checkpoint_sha256"] != sha(flow_path) or \
                        result["decoder_checkpoint_sha256"] != training["decoder_sha256"]:
                    raise AssertionError("Confirmation result artifact mismatch")
                expected_noise = torch.randn((24 * PER_PAIR, 4, 16),
                                             generator=torch.Generator().manual_seed(lock["noise_seeds"][str(seed)]))
                expected_labels = labels_all(PER_PAIR)
                if not torch.equal(sample["noise"], expected_noise) or \
                        not torch.equal(sample["labels"], expected_labels):
                    raise AssertionError("Paired confirmation noise/label mismatch")
                flow = from_variant("F0")
                flow.load_state_dict(torch.load(flow_path, map_location="cpu", weights_only=True)["state_dict"])
                flow.eval()
                with torch.no_grad():
                    replay = integrate(flow, expected_noise[:8], expected_labels[:8], 1.0)
                if not torch.allclose(replay, sample["latent"][:8], rtol=0, atol=1e-5):
                    raise AssertionError("Confirmation latent replay mismatch")
                for split in ("seen", "ood"):
                    group = [row for row in rows if row["ood"] == (split == "ood")]
                    if len(group) != result["metrics"][split]["n"]:
                        raise AssertionError("Confirmation row count mismatch")
                    for metric in ("shape_correct", "color_correct", "joint_correct",
                                   "strict_accepted_and_joint", "template_iou"):
                        if not np.isclose(np.mean([row[metric] for row in group]),
                                          result["metrics"][split][metric], atol=1e-12):
                            raise AssertionError("Confirmation metric mismatch")
                    init_values[split].append(result["metrics"][split]["strict_accepted_and_joint"])
                combined.extend(rows)
                checked_paths += 1
            saved = summary["by_data_seed"][str(seed)][kind]
            if full_diversity(combined, real) != saved["all_generation_diversity"] or \
                    accepted_diversity(combined, real) != saved["strict_subset_diversity"]:
                raise AssertionError("Confirmation diversity recomputation mismatch")
            for split in ("seen", "ood"):
                actual = float(np.mean(init_values[split]))
                stored = saved["mean_of_three_initializations"][split]["strict_accepted_and_joint"]
                if not np.isclose(actual, stored, atol=1e-12):
                    raise AssertionError("Confirmation per-seed mean mismatch")
    if checked_paths != 18:
        raise AssertionError("Confirmation model path count mismatch")
    report = {"verified": True, "model_paths": checked_paths,
              "distinct_confirmation_orbits": len(all_orbits),
              "all_frozen_sources_matched": True, "all_data_files_matched": True,
              "all_training_checkpoints_matched": True,
              "paired_noise_and_first_eight_latents_replayed_per_path": True,
              "metrics_and_diversity_recomputed": True,
              "source_sha256": sha(Path(__file__)),
              "summary_sha256": sha(EVAL / "summary.json")}
    dump(out, report)
    print(json.dumps({"verified": True, "model_paths": checked_paths}))


if __name__ == "__main__":
    main()
