# OrbitLab replication and result testing

The real-world environment is macOS 27 arm64, M5 Pro/64GB, Python 3.13.14, PyTorch 2.14.0 MPS FP32. The version is in requirements.lock.txt, and the device per run is in run.json. It can be tested/deduced on the CPU, but the learning matrices specify mps. We do not guarantee that the time and micro-values on other devices are the same.

## Check saved results

```bash
cd /Users/xavier/Documents/dev/orbitlab
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/final_metrics.py
.venv/bin/python scripts/plot_results.py
.venv/bin/python scripts/build_report.py
```

The aggregation command does not restart the learning. It updates the CSV/report/figure aggregation outputs. It does not modify the original checkpoints and measurement CSVs in runs/. You can preserve the distribution version by first checking the release manifest or aggregating from a separate copy.

## Full learning in a new folder and new environment

After unpacking the result package into a new folder, create a new execution folder. Do not copy the existing runs/, completed marker, and visual review approval records. Instead of using the command to delete the original results, use a new directory.

```bash
mkdir ../orbitlab-reproduction
cp -R work scripts tests configs data requirements.lock.txt ../orbitlab-reproduction/
cd ../orbitlab-reproduction
mkdir reports runs
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip check
.venv/bin/python work/orbitlab.py doctor --device mps --out reports/doctor_mps.json
.venv/bin/python work/orbitlab.py smoke --device mps --out reports/smoke_mps.json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_matrix.py
.venv/bin/python scripts/quality_gate.py
.venv/bin/python scripts/run_matrix.py --manifest configs/generation_ae_repair.json
.venv/bin/python scripts/quality_gate.py configs/generation_ae_repair.json
```

run_matrix.py checks the exact task settings in the config and the hash of the completion marker. It does not overwrite the failed folder that only contains leftovers. In case of failure, it inspects the logs and uses a new output path and a new config. The resume CLI, which saves the optimizer/RNG but does not implement a resume that continues from the intermediate step, was not implemented. The first quality_gate.py failed during the initial run because B1 seed2 returned a termination code of 1. This is a result of the baseline test, so the logs are preserved and the scheduled 6,000-step correction is executed. If the correction gate fails, it does not proceed to generation.

Verify that all the creation entry metrics passed even in the new run, and **directly view the six newly created** repair_*/validation/val_reconstruction.png files. Check that the shape and position are preserved and that the output is not just the background. Only after actual review should the new record below be written. Do not reuse the pass records from the original run.

```bash
# Run only after inspecting all six new reconstruction grids and confirming that each passes
.venv/bin/python - <<'PY'
import json
from pathlib import Path
Path('reports/repair_visual_review.json').write_text(json.dumps({
    'passed': True,
    'review_method': 'Reviewed all six newly generated repair validation reconstruction grids',
    'images': [f'runs/repair_{m}_n1024_s{s}/validation/val_reconstruction.png'
               for s in range(3) for m in ['aug','equivariant']]
}, indent=2))
PY
.venv/bin/python scripts/postprocess_matrix.py
.venv/bin/python scripts/run_generation.py
.venv/bin/python scripts/nn_reference.py
.venv/bin/python scripts/final_metrics.py
.venv/bin/python scripts/plot_results.py
```

The new run interprets the new CSV/JSON/diagram. Since the description sentences in build_report.py and the presentation manuscript include observations and judgments from the 2026-09-23 run, we do not write them as a new run report before reviewing them based on the new equipment/results.

The main AE comparison is 3,000 steps, and the AE for generation qualification is 6,000 steps. The flow is 4,000 steps, and the initial noise seed is 53000+training_seed, n576, Euler 0/16/32/64. The time control AE is fixed at 26.938 seconds, and the step count varies on new equipment. The exploration flow time control is set to the B2 seed0 learning time of the new run. To compare based on the exact step count, please also refer to the actual step count in the saved run.json file.

Other GPU learning is not run in parallel with learning and time measurement. To recreate the entire dataset, run work/research.py prepare instead of copying data into an empty new execution folder. NPZ file hashes may vary depending on the recompressed metadata, so image/orbit hashes and split metadata are also checked.

## Play the saved creation model from another path

The original flow checkpoint records the absolute path of AE. The helper below creates a separate sample flow copy that only modifies the path after verifying the AE hash without changing the original.

```bash
.venv/bin/python scripts/replay_generation.py \
  --flow runs/flow_equivariant_s0/flow.pt \
  --ae runs/repair_equivariant_n1024_s0/ae.pt \
  --out replay_equivariant_s0 --device mps --seed 53000 --n 576
```

If MPS is not available --device cpu is used. For small path checks --n 24 can be used, but it is not equivalent to 576 evaluations of quality verification. This time, 24 were replayed by CPU to verify the actual path for generating a new sample with a weight of path changes, as shown in reports/portability_replay/.

## Rules for reading records

- AE train n1024 and latent probe fitting n256 are different. probe train/val/test/OOD are respectively 256 base scenes×4 rotations.
- AE bootstrap is based on base scene. The base_scene_ci95 key for generation uses the name of the common helper, and in this case it is CI that re-collects the generation noise samples. The rotation copy samples were not counted as independent outputs.
- Fixed-rho error and GIF are not guaranteed by the correct rotation in B1. B1 is compared with the residual/decoded error of learned A90.
- flow_time_* is a control of the flow learning time for seed0 only. It is not mixed with the average generated by three seed weeks.
- peak_process_rss_bytes is the peak of the AE process, mps_driver_allocated_bytes is the allocation at the last time point. It is not read as the flow/system overall peak.
- code_hashes also includes auxiliary files that have not been used during learning. AE's orbitlab.py/research.py is consistent with the current version. It does not mistakenly interpret the previous hashes of generation auxiliary files developed separately during AE learning as the final flow execution version. The code_sha256 in flow run.json is consistent with the final work/generation.py.

## Presentation materials

The presentation manuscript is deliverables/TALK_6MIN_KO.md, and the generated code is .slides-build/build.mjs. Slides 3/5 contain editable native charts and embedded workbooks. The chart values are rounded to 6 decimal places, and the overall statistics are preserved in JSON/CSV. Reproducibility is achieved using the Codex Presentations artifact-tool runtime. This dedicated runtime is not required for a standard Python experimental environment. The final PPTX, notes, PNG, and package verification receipts are preserved.

This record is a specified local verification range. It does not claim that a full 43 re-learning was completed twice in a new environment, nor does it guarantee bitwise matching per GPU.
