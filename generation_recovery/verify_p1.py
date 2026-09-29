"""Independent consistency audit of P1 development training and evaluations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "work"), str(ROOT / "followup")]

import numpy as np
import torch
from common import digest_tensor
from evaluator_v2 import DetailedEvaluator
from models import from_variant
from p0 import dump, sha
from p1_evaluate import EVAL, READER, REFERENCE, NOISE_SEED, PER_PAIR, \
    diversity, extract_features, integrate, labels_all, reference
from p1_train import RUN, source_hashes


def check_close(actual, expected, tag):
    if not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise AssertionError(f"{tag}: {actual} != {expected}")


def main():
    torch.set_num_threads(4)
    out = EVAL / "verification.json"
    if out.exists():
        stored = json.loads(out.read_text())
        for record in stored["checked_artifacts"]:
            if sha(ROOT / record["path"]) != record["sha256"]:
                raise ValueError(f"P1 verification source changed: {record['path']}")
        print(json.dumps({"status": "verified_existing", "candidate_decoders": stored["candidate_decoders"]}))
        return
    summary = json.loads((EVAL / "summary.json").read_text())
    if summary["source_hashes"] != source_hashes() or summary["evaluation_code_sha256"] != sha(HERE / "p1_evaluate.py"):
        raise ValueError("P1 source mismatch")
    checked = []

    def record(path: Path):
        checked.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})

    for path in (RUN / "ae/ae.pt", RUN / "ae/run.json", RUN / "pool.pt",
                 RUN / "decoder_pixel_mean/decoder.pt", RUN / "decoder_pixel_mean/run.json",
                 RUN / "decoder_logit_mean/decoder.pt", RUN / "decoder_logit_mean/run.json",
                 EVAL / "summary.json", READER, REFERENCE):
        record(path)
    ae_run = json.loads((RUN / "ae/run.json").read_text())
    if ae_run["status"] != "completed" or ae_run["steps"] != 6000:
        raise AssertionError("P1 AE fixed-step completion missing")
    if not ae_run["validation_gate_passed"]:
        raise AssertionError("P1 AE validation gate failed; candidates need separate interpretation")
    pixel = json.loads((RUN / "decoder_pixel_mean/run.json").read_text())
    logit = json.loads((RUN / "decoder_logit_mean/run.json").read_text())
    for run in (pixel, logit):
        if run["status"] != "completed" or run["steps"] != 6000:
            raise AssertionError("Decoder fixed-step completion missing")
    for key in ("initial_weights_sha256", "latent_sha256", "train_image_sha256", "parameters"):
        if pixel[key] != logit[key]:
            raise AssertionError(f"Paired decoder control differs: {key}")
    flow_runs = {}
    for variant in ("F0", "F1", "F2", "F3"):
        folder = RUN / variant
        run = json.loads((folder / "run.json").read_text())
        if run["status"] != "completed" or run["steps"] != 4000 or run["source_hashes"] != source_hashes():
            raise AssertionError(f"Flow fixed-step/source issue: {variant}")
        if run["checkpoint_sha256"] != sha(folder / "flow.pt") or run["flow_c4_max_abs"] > 1e-5:
            raise AssertionError(f"Flow checkpoint/C4 issue: {variant}")
        flow_runs[variant] = run
        record(folder / "run.json")
        record(folder / "flow.pt")
    for a, b in (("F0", "F2"), ("F1", "F3")):
        for key in ("initial_weights_sha256", "latent_sha256", "parameters"):
            if flow_runs[a][key] != flow_runs[b][key]:
                raise AssertionError(f"Matched flow control differs: {a}, {b}, {key}")
    if len({r["latent_sha256"] for r in flow_runs.values()}) != 1:
        raise AssertionError("P1 flows have different training latents")
    expected_noise = torch.randn((24 * PER_PAIR, 4, 16), generator=torch.Generator().manual_seed(NOISE_SEED))
    expected_labels = labels_all(PER_PAIR)
    real_x, real_y = reference()
    real_rows = extract_features(real_x, real_y, DetailedEvaluator())
    checked_cases = 0
    checked_decoder_cases = 0
    for variant in ("F0", "F1", "F2", "F3"):
        scales = (1.,) if variant in ("F0", "F1") else (1., 1.5, 2., 3.)
        model = from_variant(variant)
        model.load_state_dict(torch.load(RUN / variant / "flow.pt", map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        for scale in scales:
            folder = EVAL / f"{variant}_s{str(scale).replace('.', 'p')}"
            result_path = folder / "results.json"
            result = json.loads(result_path.read_text())
            sample_path = folder / "samples.pt"
            saved = torch.load(sample_path, map_location="cpu", weights_only=True)
            if (result["samples_sha256"] != sha(sample_path) or
                    result["flow_checkpoint_sha256"] != sha(RUN / variant / "flow.pt") or
                    result["evaluation_code_sha256"] != sha(HERE / "p1_evaluate.py") or
                    result["reference_sha256"] != sha(REFERENCE) or
                    result["diversity_reader_sha256"] != sha(READER)):
                raise AssertionError("P1 evaluation provenance mismatch")
            if not torch.equal(saved["noise"], expected_noise) or not torch.equal(saved["labels"], expected_labels):
                raise AssertionError("P1 paired noise or labels differ")
            if result["noise_sha256"] != digest_tensor(expected_noise):
                raise AssertionError("P1 noise digest differs")
            with torch.no_grad():
                first = integrate(model, saved["noise"][:8], saved["labels"][:8], scale)
            if not torch.allclose(first, saved["latent"][:8], rtol=0, atol=1e-5):
                raise AssertionError("Saved generated latent fails deterministic replay")
            record(sample_path)
            record(result_path)
            checked_cases += 1
            for kind in ("pixel_mean", "logit_mean"):
                block = result["decoders"][kind]
                rows_path = folder / f"{kind}_rows.json"
                if block["rows_sha256"] != sha(rows_path):
                    raise AssertionError("P1 result rows changed")
                rows = json.loads(rows_path.read_text())
                if len(rows) != 24 * PER_PAIR:
                    raise AssertionError("P1 generated row count")
                if [(r["kind"], r["color"]) for r in rows] != [tuple(v) for v in expected_labels.tolist()]:
                    raise AssertionError("P1 generated condition row order")
                for split in ("seen", "ood"):
                    group = [r for r in rows if r["ood"] == (split == "ood")]
                    if len(group) != block["metrics"][split]["n"]:
                        raise AssertionError("P1 seen/OOD row count")
                    for key in ("shape_correct", "color_correct", "joint_correct",
                                "strict_accepted_and_joint", "template_iou"):
                        check_close(np.mean([r[key] for r in group]), block["metrics"][split][key],
                                    f"{variant}/{scale}/{kind}/{split}/{key}")
                for shape in range(4):
                    for color in range(6):
                        group = [r for r in rows if (r["kind"], r["color"]) == (shape, color)]
                        if len(group) != PER_PAIR:
                            raise AssertionError("P1 condition count")
                        key = f"{shape},{color}"
                        for metric in ("shape_correct", "color_correct", "joint_correct",
                                       "strict_accepted_and_joint", "template_iou"):
                            check_close(np.mean([r[metric] for r in group]), block["by_condition"][key][metric],
                                        f"{variant}/{scale}/{kind}/{key}/{metric}")
                computed_diversity = diversity(rows, real_rows)
                if computed_diversity != block["diversity"]:
                    raise AssertionError("P1 diversity recomputation differs")
                record(rows_path)
                checked_decoder_cases += 1
    if checked_cases != 10 or checked_decoder_cases != 20 or len(summary["rows"]) != 20:
        raise AssertionError("P1 ablation matrix incomplete")
    report = {"all_passed": True, "status": "development_validation_verified",
              "flow_cases": checked_cases, "candidate_decoders": checked_decoder_cases,
              "matched_decoder_controls": True, "matched_flow_initializations_by_architecture": True,
              "same_training_latent_and_paired_noise": True,
              "replayed_first_eight_latents_per_case": True,
              "all_rows_and_diversity_recomputed": True,
              "checked_artifacts": checked,
              "claim_limit": "One development data seed. No fresh-data confirmation or human validation."}
    dump(out, report)
    print(json.dumps({"all_passed": True, "flow_cases": checked_cases,
                      "candidate_decoders": checked_decoder_cases}))


if __name__ == "__main__":
    main()
