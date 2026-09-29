"""Locked 80-per-condition development comparison for 16k flow continuation."""
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
from common import digest_tensor
from decoder_study import ControlledDecoder
from evaluator_v2 import DetailedEvaluator
from models import from_variant
from p0 import dump, sha
from p1_evaluate import (NOISE_SEED, PER_PAIR, diversity, extract_features,
                         integrate, labels_all, reference, READER, REFERENCE)
from p1_train import RUN as P1_RUN
from p3_steps_train import RUN, hashes

EVAL = HERE / "evaluation/p3_steps16k_v1"
SETTINGS = (("F0", 1.0), ("F3", 2.0))


def evaluate(variant, guidance, real_rows):
    folder = EVAL / f"{variant}_s{str(guidance).replace('.', 'p')}"
    report_path = folder / "results.json"
    run = json.loads((RUN / variant / "run.json").read_text())
    ck_path = RUN / variant / "flow.pt"
    if run["status"] != "completed" or run["source_hashes"] != hashes(variant) or \
            run["checkpoint_sha256"] != sha(ck_path):
        raise ValueError("P3 training provenance changed")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["evaluation_code_sha256"] != sha(Path(__file__)) or \
                report["flow_checkpoint_sha256"] != sha(ck_path):
            raise ValueError("Existing P3 evaluation changed")
        return report
    folder.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    flow = from_variant(variant)
    flow.load_state_dict(ck["state_dict"])
    flow.eval()
    labels = labels_all(PER_PAIR)
    noise = torch.randn((len(labels), 4, 16), generator=torch.Generator().manual_seed(NOISE_SEED))
    start = time.perf_counter()
    with torch.no_grad():
        latent = torch.cat([integrate(flow, noise[i:i+128], labels[i:i+128], guidance)
                            for i in range(0, len(noise), 128)])
    sample_seconds = time.perf_counter() - start
    sample_path = folder / "samples.pt"
    torch.save({"noise": noise, "labels": labels, "latent": latent,
                "flow_checkpoint_sha256": sha(ck_path)}, sample_path)
    raw = latent * ck["std"] + ck["mean"]
    dec_path = P1_RUN / "decoder_logit_mean/decoder.pt"
    dck = torch.load(dec_path, map_location="cpu", weights_only=True)
    if dck["ae_sha256"] != ck["ae_sha256"]:
        raise ValueError("Flow and decoder use different encoder")
    decoder = ControlledDecoder("logit_mean")
    decoder.load_state_dict(dck["state_dict"])
    decoder.eval()
    with torch.no_grad():
        images = torch.cat([decoder((raw[i:i+64] - dck["mean"]) / dck["std"])
                            for i in range(0, len(raw), 64)])
    rows = extract_features(images, labels, DetailedEvaluator())
    rows_path = folder / "rows.json"
    dump(rows_path, rows)
    metrics = {}
    for split in ("seen", "ood"):
        group = [v for v in rows if v["ood"] == (split == "ood")]
        metrics[split] = {"n": len(group), **{key: float(np.mean([v[key] for v in group]))
                                              for key in ("shape_correct", "color_correct",
                                                          "joint_correct", "strict_accepted_and_joint",
                                                          "template_iou")}}
    o.grid(images[:48], folder / "samples.png", 8)
    report = {"variant": variant, "guidance": guidance,
              "steps": ck["steps"], "n": len(rows), "samples_per_pair": PER_PAIR,
              "noise_seed": NOISE_SEED, "noise_sha256": digest_tensor(noise),
              "sample_seconds": sample_seconds,
              "flow_checkpoint_sha256": sha(ck_path),
              "decoder_checkpoint_sha256": sha(dec_path),
              "samples_sha256": sha(sample_path), "rows_sha256": sha(rows_path),
              "reference_sha256": sha(REFERENCE), "diversity_reader_sha256": sha(READER),
              "evaluation_code_sha256": sha(Path(__file__)),
              "metrics": metrics, "diversity": diversity(rows, real_rows),
              "scope": "Development seed and noise reused from P1; not independent confirmation."}
    dump(report_path, report)
    return report


def main():
    torch.set_num_threads(4)
    x, y = reference()
    real_rows = extract_features(x, y, DetailedEvaluator())
    reports = []
    for variant, guidance in SETTINGS:
        report = evaluate(variant, guidance, real_rows)
        reports.append(report)
        print("P3 validation", variant, guidance, report["metrics"],
              report["diversity"]["aggregate"], flush=True)
    if all(report["diversity"]["passed"] for report in reports):
        print("P3 diversity both passed", flush=True)
    else:
        print("P3 diversity gate failed", flush=True)


if __name__ == "__main__":
    main()
