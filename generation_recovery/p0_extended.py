"""P0 solver, latent, and diversity-reader diagnostics on saved OrbitLab models."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "work"), str(ROOT / "followup")]

import numpy as np
import torch
from generation import state_features
from evaluator_v2 import DetailedEvaluator, accepted
import orbitlab as o
import research as r
from p0 import OLD, THRESHOLD, DEV_SEED, dump, sha


def old_models():
    ck = torch.load(OLD / "equivariant_fixed/flow.pt", map_location="cpu", weights_only=True)
    ae, _ = r.load_model(OLD / "ae/ae.pt", torch.device("cpu"))
    flow = o.Velocity(16, True)
    flow.load_state_dict(ck["state_dict"])
    return ae.eval(), flow.eval(), ck


@torch.no_grad()
def integrate(flow, noise, labels, method, steps):
    z = noise.clone()
    for k in range(steps):
        t = torch.full((len(z),), k / steps)
        v = flow(z, t, labels)
        if method == "euler":
            z = z + v / steps
        elif method == "heun":
            prediction = z + v / steps
            end = flow(prediction, torch.full((len(z),), (k + 1) / steps), labels)
            z = z + (v + end) / (2 * steps)
        else:
            raise ValueError(method)
    return z


def balanced_noise(n_per_pair=32):
    labels = torch.tensor([[k, c] for _ in range(n_per_pair) for k in range(4) for c in range(6)])
    noise = torch.randn(len(labels), 4, 16, generator=torch.Generator().manual_seed(770118))
    return noise, labels


def score(images, labels, evaluator, threshold):
    rows = []
    for p, (k, c) in zip(evaluator.predict(images), labels.tolist()):
        joint = p["predicted_shape"] == k and p["predicted_color"] == c
        rows.append({"ood": k == c, "shape": p["predicted_shape"] == k,
                     "color": p["predicted_color"] == c, "joint": joint,
                     "strict": bool(joint and accepted(p, threshold)),
                     "template_iou": p["template_iou"]})
    result = {}
    for split in ("seen", "ood"):
        sr = [v for v in rows if v["ood"] == (split == "ood")]
        result[split] = {"n": len(sr), **{key: float(np.mean([v[key] for v in sr]))
                                          for key in ("shape", "color", "joint", "strict", "template_iou")}}
    return result


def solver() -> dict:
    out = HERE / "reports/p0_solver_v1.json"
    if out.exists():
        return json.loads(out.read_text())
    ae, flow, ck = old_models()
    noise, labels = balanced_noise()
    threshold = json.loads(THRESHOLD.read_text())
    evaluator = DetailedEvaluator()
    settings = [("euler", 64), ("heun", 32), ("euler", 256)]
    report = {"source_model_sha256": sha(OLD / "equivariant_fixed/flow.pt"),
              "source_ae_sha256": sha(OLD / "ae/ae.pt"),
              "noise_seed": 770118, "n_per_pair": 32, "device": "cpu", "results": {}}
    latents = {}
    for method, steps in settings:
        start = time.perf_counter()
        images, final_z = [], []
        with torch.no_grad():
            for first in range(0, len(noise), 64):
                z = integrate(flow, noise[first:first + 64], labels[first:first + 64], method, steps)
                raw = z * ck["std"] + ck["mean"]
                final_z.append(z)
                images.append(ae.decode(raw))
        elapsed = time.perf_counter() - start
        image = torch.cat(images)
        key = f"{method}_{steps}"
        latents[key] = torch.cat(final_z)
        report["results"][key] = {"velocity_calls": steps * (2 if method == "heun" else 1),
                                  "raw_network_calls_per_sample": steps * (2 if method == "heun" else 1) * 4,
                                  "generation_and_decode_seconds": elapsed,
                                  **score(image, labels, evaluator, threshold)}
        o.grid(image[:48], HERE / f"reports/p0_solver_{key}.png", 8)
        print("solver", key, report["results"][key], flush=True)
    report["paired_final_latent_mse"] = {
        "euler64_vs_heun32": float((latents["euler_64"] - latents["heun_32"]).square().mean()),
        "euler64_vs_euler256": float((latents["euler_64"] - latents["euler_256"]).square().mean())}
    report["interpretation_limit"] = "Single old development checkpoint and 32 noise samples per condition; not independent model confirmation."
    dump(out, report)
    return report


@torch.no_grad()
def latent() -> dict:
    out = HERE / "reports/p0_latent_v1.json"
    if out.exists():
        return json.loads(out.read_text())
    ae, _, ck = old_models()
    src = ROOT / "followup/data/confirmation_v1/43001"
    data = np.load(src / "train.npz")
    train = torch.from_numpy(data["images"].copy()).permute(0, 3, 1, 2).float() / 255
    labels = torch.from_numpy(data["labels"].copy())
    train_z = []
    for k in range(4):
        for first in range(0, len(train), 64):
            train_z.append(ae.encode(o.rotate(train[first:first + 64], k)))
    train_z = torch.cat(train_z)
    train_y = labels.repeat(4, 1)
    saved = torch.load(OLD / "equivariant_fixed/samples.pt", map_location="cpu", weights_only=True)
    generated_z, generated_y = saved["latents"], saved["labels"]
    check = torch.cat([ae.encode(ae.decode(generated_z[first:first + 64]))
                       for first in range(0, len(generated_z), 64)])
    mean, std = ck["mean"], ck["std"]
    normalized_train = ((train_z - mean) / std).flatten(1)
    normalized_generated = ((generated_z - mean) / std).flatten(1)
    normalized_check = ((check - mean) / std).flatten(1)
    train_norms = normalized_train.norm(dim=1)
    generated_norms = normalized_generated.norm(dim=1)
    roundtrip = (normalized_check - normalized_generated).square().mean(dim=1).sqrt()
    nearest = []
    for first in range(0, len(generated_z), 128):
        query = normalized_generated[first:first + 128]
        batch_label = generated_y[first:first + 128]
        distance = torch.cdist(query, normalized_train).square() / query.shape[1]
        mask = ((batch_label[:, None, :] == train_y[None, :, :]).all(2))
        conditional = torch.where(mask, distance, torch.full_like(distance, float("inf"))).min(1).values
        global_min = distance.min(1).values
        nearest.extend(zip(global_min.tolist(), conditional.tolist()))
    nearest = np.asarray(nearest)
    seen = generated_y[:, 0] != generated_y[:, 1]
    report = {"n_train_rotations": len(train_z), "n_generated": len(generated_z),
              "old_train_data_seed": 43001, "same_encoded_pool_source_as_saved_flow": True,
              "train_norm_median": float(train_norms.median()),
              "generated_norm_median": float(generated_norms.median()),
              "generated_norm_p95": float(torch.quantile(generated_norms, .95)),
              "cycle_normalized_rms_median": float(roundtrip.median()),
              "cycle_normalized_rms_p95": float(torch.quantile(roundtrip, .95)),
              "nearest_normalized_sqdist_global_median": float(np.median(nearest[:, 0])),
              "nearest_normalized_sqdist_seen_same_condition_median": float(np.median(nearest[seen.numpy(), 1])),
              "nearest_normalized_sqdist_ood_same_condition_finite_n": int(np.isfinite(nearest[~seen.numpy(), 1]).sum()),
              "note": "Descriptors of saved generated codes, not a proof that a code is on/off a learned manifold. OOD pairs have no same-condition training codes."}
    dump(out, report)
    return report


def orientation_readout(features, evaluator, expected_kind):
    mask = features[0]
    intersection = (evaluator.templates * mask).sum((1, 2))
    union = np.maximum(evaluator.templates, mask).sum((1, 2))
    scores = intersection / np.maximum(union, 1)
    shape_scores = np.where(evaluator.kinds == expected_kind, scores, -1)
    raw = int(np.argmax(shape_scores) % 4)
    return raw % (2 if expected_kind == 3 else 4)


def diversity_reader() -> dict:
    out = HERE / "reports/p0_diversity_reader_lock_v1.json"
    if out.exists():
        return json.loads(out.read_text())
    evaluator = DetailedEvaluator()
    data_root = HERE / "data_v1"
    all_features, all_labels = [], []
    orientation = {str(k): {"correct": 0, "n": 0, "modulo": 2 if k == 3 else 4} for k in range(4)}
    for split in ("val", "ood"):
        data = np.load(data_root / f"{split}.npz")
        x = torch.from_numpy(data["images"].copy()).permute(0, 3, 1, 2).float() / 255
        labels = data["labels"]
        for rotation in range(4):
            for feature, label in zip(state_features(o.rotate(x, rotation)), labels):
                kind = int(label[0])
                predicted = orientation_readout(feature, evaluator, kind)
                orientation[str(kind)]["n"] += 1
                orientation[str(kind)]["correct"] += predicted == rotation % orientation[str(kind)]["modulo"]
                all_features.append({"cx": float(feature[2]), "cy": float(feature[3]),
                                     "scale": float(feature[5]), "kind": kind})
                all_labels.append([int(x) for x in label])
    for row in orientation.values():
        row["accuracy"] = row["correct"] / row["n"]
    measures = {}
    for key in ("cx", "cy"):
        values = np.asarray([x[key] for x in all_features])
        measures[key] = [float(v) for v in np.quantile(values, [.01, 1 / 3, 2 / 3, .99])]
    scale = {}
    for kind in range(4):
        values = np.asarray([x["scale"] for x in all_features if x["kind"] == kind])
        scale[str(kind)] = [float(v) for v in np.quantile(values, [.01, 1 / 3, 2 / 3, .99])]
    report = {"reader": "template orientation conditional on known shape; shape 3 has 180-degree visual symmetry",
              "orientation": orientation,
              "orientation_accepted": all(v["accuracy"] >= .95 for v in orientation.values()),
              "center_edges": {key: values for key, values in measures.items()},
              "scale_edges_by_kind": scale,
              "source": "Development validation/held-out-renderer clean images, all C4 rotations; no generated quality score used",
              "source_manifest_sha256": sha(data_root / "manifest.json"),
              "count": len(all_features),
              "out_of_edge_guard_proposed_max": .10,
              "histogram_guard": {"reference_occupied_bin_coverage_min": .90, "total_variation_max": .20},
              "note": "Shape 3 uses two distinguishable orientation bins; a four-bin claim would be incorrect."
              }
    dump(out, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["solver", "latent", "diversity", "all"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    for name, fn in (("solver", solver), ("latent", latent), ("diversity", diversity_reader)):
        if args.command in (name, "all"):
            print(name, json.dumps(fn(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
