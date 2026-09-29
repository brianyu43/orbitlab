"""Prepare three disjoint confirmation datasets only after the final protocol lock."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "work")]

import numpy as np
import orbitlab as o
import research as r
from p0 import (SPLIT_SIZES, CONFIRMATION_SEEDS, THRESHOLD,
                dump, recorded_orbits, sha)

BASE = HERE / "confirmation_v1"
LOCK = BASE / "lock.json"
DATA = BASE / "data"
PER_PAIR = 128
REFERENCE_PER_PAIR = 3 * PER_PAIR


def locked() -> dict:
    lock = json.loads(LOCK.read_text())
    if lock["source_sha256"]["p4_prepare.py"] != sha(Path(__file__)) or \
            lock["threshold_sha256"] != sha(THRESHOLD) or \
            lock["confirmation_seeds"] != list(CONFIRMATION_SEEDS) or \
            lock["samples_per_pair"] != PER_PAIR or \
            lock["reference_per_pair"] != REFERENCE_PER_PAIR:
        raise ValueError("Confirmation preparation differs from frozen protocol")
    return lock


def prior_orbits() -> set[str]:
    seen = recorded_orbits()
    for split in SPLIT_SIZES:
        seen.update(x["orbit_sha256"] for x in json.loads(
            (HERE / "data_v1" / f"{split}_metadata.json").read_text()))
    source = np.load(HERE / "data_v1/diversity_reference_v1.npz")
    for image in source["images"]:
        seen.add(r.orbit_hash(image))
    return seen


def prepare():
    lock = locked()
    manifest_path = DATA / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["lock_sha256"] != sha(LOCK) or any(
                sha(DATA / record["path"]) != record["sha256"] for record in manifest["files"]):
            raise ValueError("Confirmation data or lock changed")
        return manifest
    DATA.mkdir(parents=True, exist_ok=True)
    forbidden = prior_orbits()
    prior_count = len(forbidden)
    files = []
    counts = {}
    for seed in CONFIRMATION_SEEDS:
        folder = DATA / f"d{seed}"
        folder.mkdir(exist_ok=True)
        counts[str(seed)] = {}
        for split, n in SPLIT_SIZES.items():
            archive = folder / f"{split}.npz"
            metadata_path = folder / f"{split}_metadata.json"
            if archive.exists() != metadata_path.exists():
                raise FileExistsError("Partial confirmation split requires inspection")
            if archive.exists():
                meta = json.loads(metadata_path.read_text())
                arrays = np.load(archive)
                if len(meta) != n or len(arrays["images"]) != n or \
                        any(r.orbit_hash(a) != m["orbit_sha256"] for a, m in zip(arrays["images"], meta)):
                    raise ValueError("Existing confirmation split changed")
                if any(m["orbit_sha256"] in forbidden for m in meta):
                    raise ValueError("Confirmation split orbit repeats prior data")
                forbidden.update(m["orbit_sha256"] for m in meta)
                candidate = None
            else:
                images, labels, meta = [], [], []
                candidate = 0
                while len(images) < n:
                    image, label, info = o.render_base(candidate, seed, 64, split)
                    candidate += 1
                    orbit = r.orbit_hash(image)
                    if orbit in forbidden:
                        continue
                    forbidden.add(orbit)
                    images.append(image)
                    labels.append(label)
                    meta.append({**info, "orbit_sha256": orbit})
                np.savez_compressed(archive, images=np.stack(images), labels=np.asarray(labels, dtype=np.int64))
                dump(metadata_path, meta)
            files.extend({"path": str(p.relative_to(DATA)), "sha256": sha(p)} for p in (archive, metadata_path))
            counts[str(seed)][split] = {"n": n, "candidates": candidate}
        archive = folder / "reference.npz"
        metadata_path = folder / "reference_metadata.json"
        if archive.exists() != metadata_path.exists():
            raise FileExistsError("Partial confirmation reference requires inspection")
        if archive.exists():
            arrays = np.load(archive)
            meta = json.loads(metadata_path.read_text())
            if len(arrays["images"]) != 24 * REFERENCE_PER_PAIR or len(meta) != 24 * REFERENCE_PER_PAIR or \
                    any(r.orbit_hash(a) != m["orbit_sha256"] for a, m in zip(arrays["images"], meta)):
                raise ValueError("Existing confirmation reference changed")
            if any(m["orbit_sha256"] in forbidden for m in meta):
                raise ValueError("Confirmation reference orbit repeats prior data")
            forbidden.update(m["orbit_sha256"] for m in meta)
        else:
            buckets = {(k, c): [] for k in range(4) for c in range(6)}
            candidate = 0
            while min(len(v) for v in buckets.values()) < REFERENCE_PER_PAIR:
                split = "ood" if candidate % 6 == 0 else "val"
                image, label, info = o.render_base(candidate, seed + 10000, 64, split)
                candidate += 1
                orbit = r.orbit_hash(image)
                if orbit in forbidden or len(buckets[label]) >= REFERENCE_PER_PAIR:
                    continue
                forbidden.add(orbit)
                buckets[label].append((image, {**info, "orbit_sha256": orbit}))
            images, labels, meta = [], [], []
            for i in range(REFERENCE_PER_PAIR):
                for k in range(4):
                    for c in range(6):
                        image, info = buckets[(k, c)][i]
                        images.append(np.rot90(image, i % 4).copy())
                        labels.append((k, c))
                        meta.append(info)
            np.savez_compressed(archive, images=np.stack(images), labels=np.asarray(labels, dtype=np.int64))
            dump(metadata_path, meta)
        files.extend({"path": str(p.relative_to(DATA)), "sha256": sha(p)} for p in (archive, metadata_path))
        counts[str(seed)]["reference"] = {"n": 24 * REFERENCE_PER_PAIR,
                                           "per_pair": REFERENCE_PER_PAIR}
    report = {"lock_sha256": sha(LOCK), "confirmation_seeds": list(CONFIRMATION_SEEDS),
              "prior_orbits_excluded": prior_count,
              "unique_new_orbits": len(forbidden) - prior_count,
              "all_new_c4_orbits_disjoint": True,
              "counts": counts, "files": files,
              "statement": "Test images are prepared after configuration lock; preparation does not evaluate models."}
    dump(manifest_path, report)
    return report


if __name__ == "__main__":
    report = prepare()
    print(json.dumps({"unique_new_orbits": report["unique_new_orbits"],
                      "files": len(report["files"])}))
