"""Isolated P0 checks for the single-shape generation recovery study.

Historical followup files are read only. Every new artifact is written below
generation_recovery/ and all completed inputs are checked before reuse.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "work"), str(ROOT / "followup")]

import numpy as np
import torch
from common import original_orbits, o, r
from evaluator_v2 import DetailedEvaluator, accepted

PLAN = ROOT / "planning/generation_recovery_v1/protocol.json"
OLD = ROOT / "followup/runs/confirmation_v1/d43001_s0"
THRESHOLD = ROOT / "followup/reports/evaluator_calibration_v1/locked_threshold.json"
DEV_SEED = 770101
CONFIRMATION_SEEDS = (770201, 770202, 770203)
SPLIT_SIZES = {"train": 1024, "val": 256, "test": 256, "ood": 256}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def recorded_orbits() -> set[str]:
    seen = set(original_orbits())
    for folder in (ROOT / "followup/data", ROOT / "followup/reports/evaluator_audit_v1",
                   ROOT / "followup/reports/evaluator_calibration_v1"):
        if not folder.exists():
            continue
        for path in folder.rglob("*_metadata.json"):
            data = json.loads(path.read_text())
            if isinstance(data, list):
                seen.update(v["orbit_sha256"] for v in data if isinstance(v, dict) and "orbit_sha256" in v)
        for path in folder.glob("*sources.json"):
            data = json.loads(path.read_text())
            if isinstance(data, list):
                seen.update(v["orbit_sha256"] for v in data if isinstance(v, dict) and "orbit_sha256" in v)
    return seen


def prepare() -> dict:
    cfg_path = HERE / "execution_lock_v1.json"
    data_root = HERE / "data_v1"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg["development_seed"] != DEV_SEED or cfg["confirmation_seeds"] != list(CONFIRMATION_SEEDS):
            raise ValueError("Existing seed lock does not match source")
        manifest = json.loads((data_root / "manifest.json").read_text())
        for entry in manifest["files"]:
            if sha(data_root / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Prepared data changed: {entry['path']}")
        return {"status": "verified_existing", "lock": str(cfg_path), "manifest": str(data_root / "manifest.json"),
                "files": len(manifest["files"]), "unique_base_orbits": manifest["unique_base_orbits"]}
    if data_root.exists() or cfg_path.exists():
        raise FileExistsError("Incomplete P0 preparation; inspect before continuing")
    plan = json.loads(PLAN.read_text())
    if sha(THRESHOLD) != plan["threshold_sha256"]:
        raise ValueError("Evaluator lock changed")
    seen = recorded_orbits()
    before = len(seen)
    data_root.mkdir(parents=True)
    files = []
    counts = {}
    for split, n in SPLIT_SIZES.items():
        arrays, labels, metadata = [], [], []
        candidate = 0
        while len(arrays) < n:
            array, label, info = o.render_base(candidate, DEV_SEED, 64, split)
            candidate += 1
            digest = r.orbit_hash(array)
            if digest in seen:
                continue
            seen.add(digest)
            arrays.append(array)
            labels.append(label)
            metadata.append({**info, "orbit_sha256": digest})
        npz = data_root / f"{split}.npz"
        meta = data_root / f"{split}_metadata.json"
        np.savez_compressed(npz, images=np.stack(arrays), labels=np.asarray(labels, dtype=np.int64))
        dump(meta, metadata)
        files.extend([{"path": path.name, "sha256": sha(path)} for path in (npz, meta)])
        counts[split] = {"n": n, "candidates": candidate,
                         "seen_pairs": sorted({f"{a},{b}" for a, b in labels})}
        if any((a == b) != (split == "ood") for a, b in labels):
            raise AssertionError(f"Unexpected composition in {split}")
    unique = len(seen) - before
    if unique != sum(SPLIT_SIZES.values()):
        raise AssertionError("Base image orbit collision")
    manifest = {"seed": DEV_SEED, "existing_orbits_excluded": before, "unique_base_orbits": unique,
                "counts": counts, "files": files, "all_c4_orbits_disjoint": True,
                "not_independent_orbits": "The four rotations of each image remain one base scene."}
    dump(data_root / "manifest.json", manifest)
    cfg = {"status": "development_locked", "development_seed": DEV_SEED,
           "confirmation_seeds": list(CONFIRMATION_SEEDS),
           "confirmation_data_not_prepared_or_viewed": True,
           "plan_sha256": sha(PLAN), "threshold_sha256": sha(THRESHOLD),
           "data_manifest_sha256": sha(data_root / "manifest.json"),
           "development_data": str(data_root.relative_to(ROOT)),
           "selection": "Development validation only. Final settings and code hashes are locked before confirmation test access.",
           "source_sha256": sha(Path(__file__)), "created_at_unix": time.time()}
    dump(cfg_path, cfg)
    return {"status": "prepared", "lock": str(cfg_path), "manifest": str(data_root / "manifest.json"),
            "files": len(files), "unique_base_orbits": unique, "excluded_prior_orbits": before}


def saved_replay() -> dict:
    out = HERE / "reports/p0_saved_replay_v1.json"
    if out.exists():
        return json.loads(out.read_text())
    flow_folder = OLD / "equivariant_fixed"
    run = json.loads((flow_folder / "run.json").read_text())
    metrics = json.loads((flow_folder / "metrics.json").read_text())
    ae_run = json.loads((OLD / "ae/run.json").read_text())
    if sha(flow_folder / "flow.pt") != run["checkpoint_sha256"] != metrics["flow_sha256"]:
        raise AssertionError("Flow checkpoint SHA mismatch")
    if sha(OLD / "ae/ae.pt") != ae_run["checkpoint_sha256"]:
        raise AssertionError("AE checkpoint SHA mismatch")
    source = torch.load(flow_folder / "samples.pt", map_location="cpu", weights_only=True)
    ae, _ = r.load_model(OLD / "ae/ae.pt", torch.device("cpu"))
    with torch.no_grad():
        images = torch.cat([ae.decode(source["latents"][i:i+32]) for i in range(0, 96, 32)])
    predictions = DetailedEvaluator().predict(images)
    threshold = json.loads(THRESHOLD.read_text())
    with (flow_folder / "samples.csv").open(newline="") as f:
        stored = list(csv.DictReader(f))[:96]
    comparisons = []
    for i, (p, label, original) in enumerate(zip(predictions, source["labels"][:96].tolist(), stored)):
        joint = p["predicted_shape"] == label[0] and p["predicted_color"] == label[1]
        strict = bool(joint and accepted(p, threshold))
        comparisons.append({"sample": i, "stored_joint": original["joint_correct"] == "True",
                            "replayed_joint": joint,
                            "stored_strict": original["strict_accepted_and_joint"] == "True",
                            "replayed_strict": strict,
                            "template_iou_abs_error": abs(p["template_iou"] - float(original["template_iou"]))})
    agree = sum(a["stored_joint"] == a["replayed_joint"] and a["stored_strict"] == a["replayed_strict"]
                for a in comparisons)
    report = {"n_redecoded": 96, "classification_agreement": agree,
              "max_template_iou_abs_error": max(x["template_iou_abs_error"] for x in comparisons),
              "source_flow_sha256": sha(flow_folder / "flow.pt"), "source_ae_sha256": sha(OLD / "ae/ae.pt"),
              "source_samples_sha256": sha(flow_folder / "samples.pt"),
              "source_metrics_sha256": sha(flow_folder / "metrics.json"),
              "device": "cpu", "note": "Saved latent -> AE -> locked evaluator on first 96 stored samples; no historical files written."}
    dump(out, report)
    return report


def audit_data_and_c4() -> dict:
    out = HERE / "reports/p0_data_c4_v1.json"
    if out.exists():
        return json.loads(out.read_text())
    prep = prepare()
    data_root = HERE / "data_v1"
    h = []
    pair_counts = {}
    ranges = {}
    for split, n in SPLIT_SIZES.items():
        data = np.load(data_root / f"{split}.npz")
        images, labels = data["images"], data["labels"]
        meta = json.loads((data_root / f"{split}_metadata.json").read_text())
        assert len(images) == len(labels) == len(meta) == n
        assert np.issubdtype(images.dtype, np.integer)
        assert images.shape[1:] == (64, 64, 3)
        assert labels.min() >= 0 and labels[:, 0].max() < 4 and labels[:, 1].max() < 6
        split_counts = {}
        for a, lab, record in zip(images, labels, meta):
            digest = r.orbit_hash(a)
            assert digest == record["orbit_sha256"]
            assert list(lab) == [record["kind"], record["color"]]
            assert (record["kind"] == record["color"]) == (split == "ood")
            h.append(digest)
            key = f"{lab[0]},{lab[1]}"
            split_counts[key] = split_counts.get(key, 0) + 1
        pair_counts[split] = split_counts
        ranges[split] = [int(images.min()), int(images.max())]
    assert len(h) == len(set(h)) == sum(SPLIT_SIZES.values())
    source = torch.load(OLD / "equivariant_fixed/flow.pt", map_location="cpu", weights_only=True)
    ae, _ = r.load_model(OLD / "ae/ae.pt", torch.device("cpu"))
    flow = o.Velocity(16, True)
    flow.load_state_dict(source["state_dict"])
    x = torch.from_numpy(np.load(data_root / "val.npz")["images"][:8].copy()).permute(0, 3, 1, 2).float() / 255
    z = torch.randn(8, 4, 16, generator=torch.Generator().manual_seed(770119))
    t = torch.rand(8, generator=torch.Generator().manual_seed(770120))
    y = torch.tensor([[i % 4, (i // 4) % 6] for i in range(8)])
    with torch.no_grad():
        latent_error = (ae.encode(o.rotate(x, 1)) - o.rho(ae.encode(x), 1)).abs().max().item()
        decoder_error = (ae.decode(o.rho(z, 1)) - o.rotate(ae.decode(z), 1)).abs().max().item()
        flow_error = (flow(o.rho(z, 1), t, y) - o.rho(flow(z, t, y), 1)).abs().max().item()
    report = {"prepared_data": prep, "base_orbits_unique": len(h), "pair_counts": pair_counts,
              "pixel_ranges": ranges, "label_order": "shape index 0-3, color index 0-5",
              "c4_max_abs": {"encoder": latent_error, "decoder": decoder_error, "flow": flow_error},
              "c4_under_target_1e_5": all(v <= 1e-5 for v in (latent_error, decoder_error, flow_error)),
              "note": "Development data and prior checkpoint; CPU FP32. Historical test files not changed."}
    dump(out, report)
    return report


def clean_renderer_screen() -> dict:
    out = HERE / "reports/p0_clean_renderer_v1.json"
    if out.exists():
        return json.loads(out.read_text())
    ev = DetailedEvaluator()
    threshold = json.loads(THRESHOLD.read_text())
    forbidden = recorded_orbits()
    forbidden.update(v["orbit_sha256"] for split in SPLIT_SIZES
                     for v in json.loads((HERE / "data_v1" / f"{split}_metadata.json").read_text()))
    rows = []
    for split in ("test", "ood"):
        images, labels = [], []
        counts = {(k, c): 0 for k in range(4) for c in range(6) if (k == c) == (split == "ood")}
        candidate = 0
        while min(counts.values()) < 32:
            array, label, _ = o.render_base(candidate, DEV_SEED + 1, 64, split)
            candidate += 1
            digest = r.orbit_hash(array)
            if digest in forbidden or counts[label] >= 32:
                continue
            forbidden.add(digest)
            counts[label] += 1
            images.append(array)
            labels.append(label)
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
        preds = ev.predict(x)
        for p, label in zip(preds, labels):
            rows.append({"split": split, "label": list(label), "shape_correct": p["predicted_shape"] == label[0],
                         "color_correct": p["predicted_color"] == label[1],
                         "strict_accepted_and_joint": bool(accepted(p, threshold) and
                                                           p["predicted_shape"] == label[0] and p["predicted_color"] == label[1])})
    results = {}
    for split in ("test", "ood"):
        sr = [v for v in rows if v["split"] == split]
        results[split] = {"n": len(sr), **{key: float(np.mean([v[key] for v in sr])) for key in
                                              ("shape_correct", "color_correct", "strict_accepted_and_joint")}}
    report = {"results": results, "source": "known synthetic renderer with fresh images; reference screen check, not a learned model",
              "threshold_sha256": sha(THRESHOLD), "all_base_orbits_unique": True}
    dump(out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "replay", "audit", "renderer", "all"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command in ("prepare", "all"):
        print("prepare", json.dumps(prepare(), ensure_ascii=False), flush=True)
    if args.command in ("replay", "all"):
        print("replay", json.dumps(saved_replay(), ensure_ascii=False), flush=True)
    if args.command in ("audit", "all"):
        print("audit", json.dumps(audit_data_and_c4(), ensure_ascii=False), flush=True)
    if args.command in ("renderer", "all"):
        print("renderer", json.dumps(clean_renderer_screen(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
