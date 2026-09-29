"""P4 execution-only wrapper fixing the missing continuation provenance key.

All weights, data, optimizer states, and evaluation configuration are unchanged.
The original frozen p4_train.py remains immutable and still checks its lock.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import p1_train as p1
import p3_steps_train as p3
import p4_train as original
from p0 import sha
from p4_prepare import BASE, LOCK, locked

AMENDMENT = BASE / "execution_amendment_01.json"


def amend_one(seed: int, init: int):
    amendment = json.loads(AMENDMENT.read_text())
    if amendment["original_lock_sha256"] != sha(LOCK) or \
            amendment["wrapper_sha256"] != sha(Path(__file__)) or \
            locked()["source_sha256"]["p4_train.py"] != sha(HERE / "p4_train.py"):
        raise ValueError("P4 execution amendment/source changed")
    parent_train = p3.train

    def with_provenance(variant):
        old_hashes = p3.hashes
        p3.hashes = lambda name: {**old_hashes(name),
                                  "training_data_sha256": sha(p1.DATA / "train.npz")}
        try:
            return parent_train(variant)
        finally:
            p3.hashes = old_hashes

    p3.train = with_provenance
    try:
        result = original.train_one(seed, init)
    finally:
        p3.train = parent_train
    path = original.RUNS / f"d{seed}_s{init}" / "training_summary.json"
    saved = json.loads(path.read_text())
    saved["execution_amendment_sha256"] = sha(AMENDMENT)
    saved["execution_wrapper_sha256"] = sha(Path(__file__))
    temp = path.with_suffix(".amended.tmp")
    temp.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    seeds = (args.seed,) if args.seed is not None else tuple(locked()["confirmation_seeds"])
    inits = (args.init,) if args.init is not None else original.INITIALIZATIONS
    for seed in seeds:
        for init in inits:
            result = amend_one(seed, init)
            print("P4 trained", seed, init, result["flow16_sha256"], flush=True)


if __name__ == "__main__":
    main()
