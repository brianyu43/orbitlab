"""P1 development validation with paired noise and locked shape/diversity screens."""
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
from generation import state_features
import orbitlab as o
from common import digest_tensor
from decoder_study import ControlledDecoder
from evaluator_v2 import DetailedEvaluator, accepted
from models import from_variant, guided
from p0 import THRESHOLD, DEV_SEED, dump, sha, recorded_orbits
from p0_extended import orientation_readout
from p1_train import RUN, DATA, source_hashes

EVAL = HERE / "evaluation/p1_v1"
REFERENCE = HERE / "data_v1/diversity_reference_v1.npz"
READER = HERE / "reports/p0_diversity_reader_lock_v1.json"
NOISE_SEED = 770141
PER_PAIR = 80


def labels_all(n: int) -> torch.Tensor:
    return torch.tensor([[k, c] for _ in range(n) for k in range(4) for c in range(6)], dtype=torch.long)


def reference():
    manifest_path = HERE / "data_v1/diversity_reference_manifest_v1.json"
    if REFERENCE.exists():
        manifest = json.loads(manifest_path.read_text())
        if sha(REFERENCE) != manifest["archive_sha256"]:
            raise ValueError("Diversity reference changed")
        source = np.load(REFERENCE)
        return (torch.from_numpy(source["images"].copy()).permute(0, 3, 1, 2).float() / 255,
                torch.from_numpy(source["labels"].copy()))
    if manifest_path.exists():
        raise FileExistsError("Partial reference creation")
    forbidden = recorded_orbits()
    for split in ("train", "val", "test", "ood"):
        forbidden.update(m["orbit_sha256"] for m in json.loads((DATA / f"{split}_metadata.json").read_text()))
    images = {}
    orbit_hashes = {}
    for split in ("val", "ood"):
        pairs = [(k, c) for k in range(4) for c in range(6) if (k == c) == (split == "ood")]
        bucket = {pair: [] for pair in pairs}
        candidate = 0
        while min(len(values) for values in bucket.values()) < PER_PAIR:
            image, label, _ = o.render_base(candidate, DEV_SEED + 2, 64, split)
            candidate += 1
            digest = __import__("research").orbit_hash(image)
            if digest in forbidden or len(bucket[label]) >= PER_PAIR:
                continue
            forbidden.add(digest)
            orbit_hashes[f"{split}:{label[0]},{label[1]}:{len(bucket[label])}"] = digest
            bucket[label].append(image)
        for pair, values in bucket.items():
            assert len(values) == PER_PAIR
            images[pair] = values
    ordered = []
    for i in range(PER_PAIR):
        for kind in range(4):
            for color in range(6):
                ordered.append(np.rot90(images[(kind, color)][i], i % 4).copy())
    x = np.stack(ordered)
    labels = labels_all(PER_PAIR)
    np.savez_compressed(REFERENCE, images=x, labels=labels.numpy())
    manifest = {"seed": DEV_SEED + 2, "n_per_pair": PER_PAIR, "n": len(x),
                "unique_base_orbits": len(set(orbit_hashes.values())),
                "all_disjoint_from_prior_and_development": True,
                "archive_sha256": sha(REFERENCE),
                "rotations": "image index within each condition modulo 4",
                "purpose": "Development diversity reference only; no model trained on these images."}
    dump(manifest_path, manifest)
    return torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float() / 255, labels


def extract_features(images, labels, evaluator):
    predictions = evaluator.predict(images)
    features = state_features(images)
    threshold = json.loads(THRESHOLD.read_text())
    result = []
    for pred, state, label in zip(predictions, features, labels.tolist()):
        kind, color = label
        orient = orientation_readout(state, evaluator, kind)
        joint = pred["predicted_shape"] == kind and pred["predicted_color"] == color
        result.append({"kind": kind, "color": color, "ood": kind == color,
                       "shape_correct": pred["predicted_shape"] == kind,
                       "color_correct": pred["predicted_color"] == color,
                       "joint_correct": joint,
                       "strict_accepted_and_joint": bool(joint and accepted(pred, threshold)),
                       "template_iou": pred["template_iou"],
                       "cx": pred["cx"], "cy": pred["cy"], "scale": pred["scale"],
                       "orientation": orient})
    return result


def categorical(values, edges):
    a = np.asarray(values)
    return np.searchsorted(np.asarray(edges)[1:-1], a, side="right"), float(np.mean((a < edges[0]) | (a > edges[-1])))


def diversity(generated, real):
    reader = json.loads(READER.read_text())
    assessments = []
    for kind in range(4):
        for color in range(6):
            g = [x for x in generated if (x["kind"], x["color"]) == (kind, color)]
            ref = [x for x in real if (x["kind"], x["color"]) == (kind, color)]
            if len(g) != PER_PAIR or len(ref) != PER_PAIR:
                raise AssertionError("Diversity conditions are not balanced")
            for key in ("cx", "cy", "scale", "orientation"):
                if key == "orientation":
                    categories = reader["orientation"][str(kind)]["modulo"]
                    gv = np.asarray([x[key] for x in g]); rv = np.asarray([x[key] for x in ref])
                    outside = 0.0
                else:
                    edges = reader["center_edges"][key] if key != "scale" else reader["scale_edges_by_kind"][str(kind)]
                    gv, outside = categorical([x[key] for x in g], edges)
                    rv, _ = categorical([x[key] for x in ref], edges)
                    categories = 3
                gh = np.bincount(gv, minlength=categories)[:categories]
                rh = np.bincount(rv, minlength=categories)[:categories]
                covered = int(np.count_nonzero((gh > 0) & (rh > 0)))
                possible = int(np.count_nonzero(rh > 0))
                coverage = covered / possible if possible else 1.0
                total_variation = .5 * float(np.abs(gh / gh.sum() - rh / rh.sum()).sum())
                assessments.append({"kind": kind, "color": color, "attribute": key,
                                    "reference_occupied_bins": possible,
                                    "coverage": coverage, "tv": total_variation, "outside": outside})
    aggregate = {key: {"coverage_mean": float(np.mean([a["coverage"] for a in assessments if a["attribute"] == key])),
                       "tv_mean": float(np.mean([a["tv"] for a in assessments if a["attribute"] == key])),
                       "outside_mean": float(np.mean([a["outside"] for a in assessments if a["attribute"] == key]))}
                 for key in ("cx", "cy", "scale", "orientation")}
    passed = all(a["coverage_mean"] >= .9 and a["tv_mean"] <= .2 and a["outside_mean"] <= .1
                 for a in aggregate.values())
    return {"passed": passed, "aggregate": aggregate, "conditions": assessments,
            "strict_subset_status": "not_evaluated_at_development_screen"}


@torch.no_grad()
def integrate(flow, noise, labels, scale):
    z = noise.clone()
    for step in range(64):
        t = torch.full((len(z),), step / 64)
        z = z + guided(flow, z, t, labels, scale) / 64
    return z


def candidate(variant: str, scale: float, real_rows, evaluator):
    if variant in {"F0", "F1"} and scale != 1:
        raise ValueError("CFG requires a null-trained model")
    folder = EVAL / f"{variant}_s{str(scale).replace('.', 'p')}"
    report_path = folder / "results.json"
    ck_path = RUN / variant / "flow.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    model = from_variant(variant)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    if report_path.exists():
        result = json.loads(report_path.read_text())
        if result["flow_checkpoint_sha256"] != sha(ck_path):
            raise ValueError("Flow changed after evaluation")
        if result["evaluation_code_sha256"] != sha(Path(__file__)):
            raise ValueError("Evaluation code changed after candidate scoring")
        return result
    folder.mkdir(parents=True, exist_ok=True)
    noise = torch.randn((24 * PER_PAIR, 4, 16), generator=torch.Generator().manual_seed(NOISE_SEED))
    labels = labels_all(PER_PAIR)
    sample_path = folder / "samples.pt"
    if sample_path.exists():
        saved = torch.load(sample_path, map_location="cpu", weights_only=True)
        if saved["flow_sha256"] != sha(ck_path) or not torch.equal(saved["noise"], noise):
            raise ValueError("Previous incomplete sample differs")
        latent = saved["latent"]
        seconds = saved["sample_seconds"]
    else:
        start = time.perf_counter()
        latent = torch.cat([integrate(model, noise[i:i+128], labels[i:i+128], scale)
                            for i in range(0, len(noise), 128)])
        seconds = time.perf_counter() - start
        torch.save({"noise": noise, "labels": labels, "latent": latent,
                    "flow_sha256": sha(ck_path), "sample_seconds": seconds}, sample_path)
    result = {"variant": variant, "scale": scale, "n": len(noise), "samples_per_pair": PER_PAIR,
              "flow_checkpoint_sha256": sha(ck_path), "samples_sha256": sha(sample_path),
              "noise_sha256": digest_tensor(noise), "sample_seconds": seconds,
              "raw_model_calls_per_sample": 64 * 4 * (2 if scale != 1 else 1),
              "decoders": {}, "source_hashes": source_hashes(),
              "evaluation_code_sha256": sha(Path(__file__)),
              "evaluator_sha256": sha(ROOT / "followup/evaluator_v2.py"),
              "diversity_reader_sha256": sha(READER),
              "reference_sha256": sha(REFERENCE)}
    raw = latent * ck["std"] + ck["mean"]
    for kind in ("pixel_mean", "logit_mean"):
        path = RUN / f"decoder_{kind}/decoder.pt"
        dck = torch.load(path, map_location="cpu", weights_only=True)
        if dck["ae_sha256"] != ck["ae_sha256"]:
            raise ValueError("Decoder and flow use different encoders")
        decoder = ControlledDecoder(kind)
        decoder.load_state_dict(dck["state_dict"])
        decoder.eval()
        start = time.perf_counter()
        with torch.no_grad():
            images = torch.cat([decoder((raw[i:i+64] - dck["mean"]) / dck["std"])
                                for i in range(0, len(raw), 64)])
        decode_seconds = time.perf_counter() - start
        rows = extract_features(images, labels, evaluator)
        metrics = {}
        for split in ("seen", "ood"):
            subset = [v for v in rows if v["ood"] == (split == "ood")]
            metrics[split] = {"n": len(subset), **{key: float(np.mean([r[key] for r in subset]))
                                                  for key in ("shape_correct", "color_correct", "joint_correct",
                                                              "strict_accepted_and_joint", "template_iou")}}
        by_condition = {}
        for shape in range(4):
            for color in range(6):
                subset = [v for v in rows if (v["kind"], v["color"]) == (shape, color)]
                by_condition[f"{shape},{color}"] = {
                    "n": len(subset),
                    **{key: float(np.mean([r[key] for r in subset]))
                       for key in ("shape_correct", "color_correct", "joint_correct",
                                   "strict_accepted_and_joint", "template_iou")}}
        rows_path = folder / f"{kind}_rows.json"
        if rows_path.exists():
            if json.loads(rows_path.read_text()) != rows:
                raise ValueError("Interrupted evaluation rows disagree with replay")
        else:
            dump(rows_path, rows)
        result["decoders"][kind] = {"checkpoint_sha256": sha(path), "decode_seconds": decode_seconds,
                                    "metrics": metrics, "by_condition": by_condition,
                                    "rows_sha256": sha(rows_path),
                                    "diversity": diversity(rows, real_rows)}
        o.grid(images[:48], folder / f"{kind}_samples.png", 8)
        print("validation", variant, scale, kind,
              metrics["seen"]["strict_accepted_and_joint"],
              metrics["ood"]["strict_accepted_and_joint"],
              result["decoders"][kind]["diversity"]["passed"], flush=True)
    dump(report_path, result)
    return result


def summarize(results):
    out = EVAL / "summary.json"
    if out.exists():
        return json.loads(out.read_text())
    flat = []
    for result in results:
        for decoder, block in result["decoders"].items():
            seen, ood = block["metrics"]["seen"], block["metrics"]["ood"]
            flat.append({"variant": result["variant"], "scale": result["scale"], "decoder": decoder,
                         "seen_strict": seen["strict_accepted_and_joint"],
                         "ood_strict": ood["strict_accepted_and_joint"],
                         "seen_color": seen["color_correct"], "ood_color": ood["color_correct"],
                         "seen_joint": seen["joint_correct"], "ood_joint": ood["joint_correct"],
                         "all_generation_diversity_passed": block["diversity"]["passed"],
                         "raw_model_calls_per_sample": result["raw_model_calls_per_sample"]})
    by_decoder = {d: next(v for v in flat if v["variant"] == "F0" and v["scale"] == 1 and v["decoder"] == d)
                  for d in ("pixel_mean", "logit_mean")}
    for row in flat:
        baseline = by_decoder[row["decoder"]]
        row["paired_seen_gain_pp_vs_F0"] = (row["seen_strict"] - baseline["seen_strict"]) * 100
        row["paired_ood_gain_pp_vs_F0"] = (row["ood_strict"] - baseline["ood_strict"]) * 100
        row["eligible_for_repeats"] = (
            row["paired_seen_gain_pp_vs_F0"] >= 5 and row["paired_ood_gain_pp_vs_F0"] >= -2
            and row["seen_color"] >= .95 and row["ood_color"] >= .95
            and row["all_generation_diversity_passed"])
    report = {"status": "development_validation_only", "rows": flat,
              "eligible": [v for v in flat if v["eligible_for_repeats"]],
              "noise_seed": NOISE_SEED, "samples_per_pair": PER_PAIR,
              "n_independent_data_seeds": 1,
              "selection": "Baseline F0 is paired on labels/noise with every candidate. Strict diversity among passing images remains unverified.",
              "source_hashes": source_hashes(), "evaluation_code_sha256": sha(Path(__file__)),
              "reference_sha256": sha(REFERENCE)}
    dump(out, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["reference", "evaluate"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    ref_x, ref_y = reference()
    print("reference", len(ref_x), sha(REFERENCE), flush=True)
    if args.command == "reference":
        return
    evaluator = DetailedEvaluator()
    real_rows = extract_features(ref_x, ref_y, evaluator)
    if not all(v["strict_accepted_and_joint"] for v in real_rows):
        raise AssertionError("Clean diversity reference failed the frozen screen")
    results = []
    for variant in ("F0", "F1", "F2", "F3"):
        scales = (1.0,) if variant in ("F0", "F1") else (1.0, 1.5, 2.0, 3.0)
        for scale in scales:
            results.append(candidate(variant, scale, real_rows, evaluator))
    report = summarize(results)
    print("summary", json.dumps({"rows": len(report["rows"]), "eligible": report["eligible"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
