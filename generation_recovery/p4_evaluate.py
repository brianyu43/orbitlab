"""Frozen confirmation scoring for F0 at 4k versus 16k training steps."""
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
import orbitlab as o
import research as r
from common import digest_tensor
from decoder_study import ControlledDecoder
from evaluator_v2 import DetailedEvaluator
from models import from_variant
from p0 import dump, sha
from p1_evaluate import (READER, categorical, diversity, extract_features,
                         integrate, labels_all)
import p1_evaluate as p1_eval
from p4_prepare import BASE, DATA, LOCK, PER_PAIR, REFERENCE_PER_PAIR, locked
from p4_train import RUNS, INITIALIZATIONS, context_hashes

EVAL = BASE / "evaluation"
KINDS = ("baseline_4k", "candidate_16k")
ATTRIBUTES = ("cx", "cy", "scale", "orientation")


def source_check(lock):
    if lock["source_sha256"]["p4_evaluate.py"] != sha(Path(__file__)):
        raise ValueError("P4 evaluation source differs from frozen protocol")
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["lock_sha256"] != sha(LOCK) or any(
            sha(DATA / entry["path"]) != entry["sha256"] for entry in manifest["files"]):
        raise ValueError("Confirmation data differs from frozen manifest")


def metrics(rows):
    result = {}
    for split in ("seen", "ood"):
        group = [row for row in rows if row["ood"] == (split == "ood")]
        result[split] = {"n": len(group), **{
            key: float(np.mean([row[key] for row in group]))
            for key in ("shape_correct", "color_correct", "joint_correct",
                        "strict_accepted_and_joint", "template_iou")}}
    return result


def reference(seed, evaluator):
    archive = np.load(DATA / f"d{seed}" / "reference.npz")
    x = torch.from_numpy(archive["images"].copy()).permute(0, 3, 1, 2).float() / 255
    y = torch.from_numpy(archive["labels"].copy()).long()
    if len(x) != 24 * REFERENCE_PER_PAIR or not torch.equal(y, labels_all(REFERENCE_PER_PAIR)):
        raise AssertionError("Confirmation diversity reference order changed")
    rows = extract_features(x, y, evaluator)
    if not all(row["strict_accepted_and_joint"] for row in rows):
        raise AssertionError("Clean confirmation reference fails frozen evaluator")
    return rows


def evaluate_one(seed, init, kind, evaluator):
    lock = locked()
    source_check(lock)
    if kind not in KINDS or seed not in lock["confirmation_seeds"] or init not in INITIALIZATIONS:
        raise ValueError("Unselected confirmation setting")
    run = RUNS / f"d{seed}_s{init}"
    training = json.loads((run / "training_summary.json").read_text())
    if training["status"] != "trained" or training["context_hashes"] != context_hashes(seed, init):
        raise ValueError("Training provenance changed")
    flow_path = run / ("p1/F0/flow.pt" if kind == "baseline_4k" else "p3/F0/flow.pt")
    dec_path = run / "p1/decoder_logit_mean/decoder.pt"
    out = EVAL / f"d{seed}_s{init}" / kind
    result_path = out / "results.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["flow_checkpoint_sha256"] != sha(flow_path) or \
                result["decoder_checkpoint_sha256"] != sha(dec_path) or \
                result["evaluation_source_sha256"] != sha(Path(__file__)):
            raise ValueError("Saved confirmation result changed")
        return result
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(flow_path, map_location="cpu", weights_only=True)
    dck = torch.load(dec_path, map_location="cpu", weights_only=True)
    if ck["ae_sha256"] != dck["ae_sha256"] or \
            ck["steps"] != (4000 if kind == "baseline_4k" else 16000):
        raise ValueError("Selected flow and decoder provenance mismatch")
    flow = from_variant("F0")
    flow.load_state_dict(ck["state_dict"])
    flow.eval()
    decoder = ControlledDecoder("logit_mean")
    decoder.load_state_dict(dck["state_dict"])
    decoder.eval()
    labels = labels_all(PER_PAIR)
    noise_seed = lock["noise_seeds"][str(seed)]
    noise = torch.randn((len(labels), 4, 16), generator=torch.Generator().manual_seed(noise_seed))
    start = time.perf_counter()
    with torch.no_grad():
        latent = torch.cat([integrate(flow, noise[i:i+128], labels[i:i+128], 1.0)
                            for i in range(0, len(noise), 128)])
        raw = latent * ck["std"] + ck["mean"]
        images = torch.cat([decoder((raw[i:i+64] - dck["mean"]) / dck["std"])
                            for i in range(0, len(raw), 64)])
    seconds = time.perf_counter() - start
    sample_path = out / "samples.pt"
    torch.save({"noise": noise, "labels": labels, "latent": latent,
                "flow_checkpoint_sha256": sha(flow_path)}, sample_path)
    rows = extract_features(images, labels, evaluator)
    rows_path = out / "rows.json"
    dump(rows_path, rows)
    o.grid(images[:48], out / "samples.png", 8)
    result = {"seed": seed, "initialization": init, "kind": kind,
              "n": len(rows), "samples_per_pair": PER_PAIR,
              "noise_seed": noise_seed, "noise_sha256": digest_tensor(noise),
              "flow_steps": ck["steps"], "raw_model_calls_per_sample": 256,
              "sample_decode_seconds": seconds,
              "flow_checkpoint_sha256": sha(flow_path),
              "decoder_checkpoint_sha256": sha(dec_path),
              "samples_sha256": sha(sample_path), "rows_sha256": sha(rows_path),
              "evaluation_source_sha256": sha(Path(__file__)),
              "lock_sha256": sha(LOCK), "reference_sha256": sha(DATA / f"d{seed}/reference.npz"),
              "metrics": metrics(rows)}
    dump(result_path, result)
    return result


def full_diversity(rows, real):
    old = p1_eval.PER_PAIR
    try:
        p1_eval.PER_PAIR = REFERENCE_PER_PAIR
        return diversity(rows, real)
    finally:
        p1_eval.PER_PAIR = old


def accepted_diversity(rows, real):
    reader = json.loads(READER.read_text())
    accepted_rows = [row for row in rows if row["strict_accepted_and_joint"]]
    conditions = []
    for shape in range(4):
        for color in range(6):
            generated = [row for row in accepted_rows if (row["kind"], row["color"]) == (shape, color)]
            clean = [row for row in real if (row["kind"], row["color"]) == (shape, color)]
            if len(clean) != REFERENCE_PER_PAIR:
                raise AssertionError("Clean confirmation diversity reference incomplete")
            if len(generated) < 32:
                conditions.append({"shape": shape, "color": color, "n": len(generated),
                                   "status": "insufficient_under_32"})
                continue
            features = {}
            for key in ATTRIBUTES:
                if key == "orientation":
                    categories = reader["orientation"][str(shape)]["modulo"]
                    gv = np.asarray([row[key] for row in generated])
                    rv = np.asarray([row[key] for row in clean])
                    outside = 0.0
                else:
                    edges = (reader["center_edges"][key] if key != "scale" else
                             reader["scale_edges_by_kind"][str(shape)])
                    gv, outside = categorical([row[key] for row in generated], edges)
                    rv, _ = categorical([row[key] for row in clean], edges)
                    categories = 3
                gh = np.bincount(gv, minlength=categories)[:categories]
                rh = np.bincount(rv, minlength=categories)[:categories]
                features[key] = {"coverage": float(np.count_nonzero((gh > 0) & (rh > 0)) /
                                                    np.count_nonzero(rh > 0)),
                                 "tv": .5 * float(np.abs(gh / gh.sum() - rh / rh.sum()).sum()),
                                 "outside": outside}
            conditions.append({"shape": shape, "color": color, "n": len(generated),
                               "status": "evaluated", "features": features})
    all_evaluable = all(c["status"] == "evaluated" for c in conditions)
    aggregates = {key: {metric: float(np.mean([c["features"][key][metric] for c in conditions
                                               if c["status"] == "evaluated"]))
                        for metric in ("coverage", "tv", "outside")}
                  for key in ATTRIBUTES} if any(c["status"] == "evaluated" for c in conditions) else {}
    passed = all_evaluable and all(v["coverage"] >= .9 and v["tv"] <= .2 and v["outside"] <= .1
                                 for v in aggregates.values())
    return {"passed": passed,
            "status": "evaluated" if all_evaluable else "insufficient_strict_samples",
            "n_accepted": len(accepted_rows), "aggregate": aggregates,
            "conditions": conditions}


def reconstruction_check(seed, init, evaluator):
    run = RUNS / f"d{seed}_s{init}"
    path = EVAL / f"d{seed}_s{init}" / "reconstruction.json"
    ae_path = run / "p1/ae/ae.pt"
    decoder_path = run / "p1/decoder_logit_mean/decoder.pt"
    if path.exists():
        result = json.loads(path.read_text())
        if result["ae_sha256"] != sha(ae_path) or result["decoder_sha256"] != sha(decoder_path):
            raise ValueError("Saved confirmation reconstruction changed")
        return result
    ae, _ = r.load_model(ae_path, torch.device("cpu"))
    ck = torch.load(decoder_path, map_location="cpu", weights_only=True)
    decoder = ControlledDecoder("logit_mean")
    decoder.load_state_dict(ck["state_dict"])
    decoder.eval()
    result = {}
    for split in ("test", "ood"):
        data = np.load(DATA / f"d{seed}" / f"{split}.npz")
        x = torch.from_numpy(data["images"].copy()).permute(0, 3, 1, 2).float() / 255
        y = torch.from_numpy(data["labels"].copy()).long()
        rows = []
        for rotation in range(4):
            for first in range(0, len(x), 64):
                target = o.rotate(x[first:first+64], rotation)
                with torch.no_grad():
                    latent = ae.encode(target)
                    output = decoder((latent - ck["mean"]) / ck["std"])
                rows.extend(extract_features(output, y[first:first+64], evaluator))
        result[split] = {"n": len(rows), **{
            key: float(np.mean([row[key] for row in rows]))
            for key in ("shape_correct", "color_correct", "joint_correct",
                        "strict_accepted_and_joint", "template_iou")}}
    report = {"seed": seed, "initialization": init,
              "ae_sha256": sha(ae_path), "decoder_sha256": sha(decoder_path),
              "source_sha256": sha(Path(__file__)), "metrics": result,
              "scope": "Fresh clean-image reconstruction diagnostic; separate from generated success."}
    dump(path, report)
    return report


def summarize(all_results, evaluator):
    summary_path = EVAL / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    by_seed = {}
    for seed in locked()["confirmation_seeds"]:
        real = reference(seed, evaluator)
        group = [result for result in all_results if result["seed"] == seed]
        entry = {}
        for kind in KINDS:
            selected = [result for result in group if result["kind"] == kind]
            if len(selected) != len(INITIALIZATIONS):
                raise AssertionError("Confirmation initialization count incomplete")
            rows = []
            for init in INITIALIZATIONS:
                result = next(x for x in selected if x["initialization"] == init)
                rows.extend(json.loads((EVAL / f"d{seed}_s{init}" / kind / "rows.json").read_text()))
            mean = {split: {metric: float(np.mean([x["metrics"][split][metric] for x in selected]))
                             for metric in ("shape_correct", "color_correct", "joint_correct",
                                            "strict_accepted_and_joint", "template_iou")}
                    for split in ("seen", "ood")}
            entry[kind] = {"mean_of_three_initializations": mean,
                           "all_generation_diversity": full_diversity(rows, real),
                           "strict_subset_diversity": accepted_diversity(rows, real)}
        entry["paired_gain_pp"] = {split: 100 * (
            entry["candidate_16k"]["mean_of_three_initializations"][split]["strict_accepted_and_joint"] -
            entry["baseline_4k"]["mean_of_three_initializations"][split]["strict_accepted_and_joint"])
            for split in ("seen", "ood")}
        by_seed[str(seed)] = entry
    average = {split: {kind: float(np.mean([by_seed[str(seed)][kind]["mean_of_three_initializations"][split]["strict_accepted_and_joint"]
                                                for seed in locked()["confirmation_seeds"]]))
                        for kind in KINDS}
               for split in ("seen", "ood")}
    gain = {split: (average[split]["candidate_16k"] - average[split]["baseline_4k"]) * 100
            for split in ("seen", "ood")}
    milestone = (average["seen"]["candidate_16k"] >= .3 and
                 average["ood"]["candidate_16k"] >= .2 and
                 gain["seen"] >= 10 and gain["ood"] >= 5 and
                 all(by_seed[str(seed)]["paired_gain_pp"][split] >= -2
                     for seed in locked()["confirmation_seeds"] for split in ("seen", "ood")) and
                 all(by_seed[str(seed)]["candidate_16k"]["mean_of_three_initializations"][split]["color_correct"] >= .95
                     for seed in locked()["confirmation_seeds"] for split in ("seen", "ood")) and
                 all(by_seed[str(seed)]["candidate_16k"]["all_generation_diversity"]["passed"] and
                     by_seed[str(seed)]["candidate_16k"]["strict_subset_diversity"]["passed"]
                     for seed in locked()["confirmation_seeds"]))
    summary = {"status": "frozen_confirmation", "lock_sha256": sha(LOCK),
               "data_manifest_sha256": sha(DATA / "manifest.json"),
               "n_data_seeds": 3, "n_initializations_per_seed": 3,
               "mean_strict": average, "mean_paired_gain_pp": gain,
               "by_data_seed": by_seed, "milestone_passed": milestone,
               "claim_limit": "Three synthetic data seeds; no human validation or other task generalization."}
    dump(summary_path, summary)
    return summary


def main():
    torch.set_num_threads(4)
    lock = locked()
    source_check(lock)
    evaluator = DetailedEvaluator()
    results = []
    for seed in lock["confirmation_seeds"]:
        for init in INITIALIZATIONS:
            reconstruction_check(seed, init, evaluator)
            for kind in KINDS:
                result = evaluate_one(seed, init, kind, evaluator)
                results.append(result)
                print("P4 score", seed, init, kind,
                      result["metrics"]["seen"]["strict_accepted_and_joint"],
                      result["metrics"]["ood"]["strict_accepted_and_joint"], flush=True)
    summary = summarize(results, evaluator)
    print("P4 summary", json.dumps({"mean_strict": summary["mean_strict"],
                                    "mean_paired_gain_pp": summary["mean_paired_gain_pp"],
                                    "milestone_passed": summary["milestone_passed"]}), flush=True)


if __name__ == "__main__":
    main()
