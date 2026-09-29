# Reproduction and recovery

Run from `/Users/xavier/Documents/dev/orbitlab` with its `.venv/bin/python`. This completion layer imports the preserved `work/`, `followup/`, and `generation_recovery/` implementation and checkpoint dependencies. It is **not a standalone clean-install distribution**. Exact versions and dependency hashes are recorded in the final provenance files and individual protocols.

## Read results without training

```sh
.venv/bin/python completion_v2/summarize.py
.venv/bin/python completion_v2/aggregate_results.py
```

`live_summary.json` is a mutable progress view; `aggregate.json` is a descriptive result aggregation. Completion requires matching expected counts and verification receipts, not merely an existing output folder. See `scope_audit.json` for the final closure status.

## Verification

```sh
.venv/bin/python completion_v2/verify_new_data.py
.venv/bin/python completion_v2/verify_new_results.py perception
.venv/bin/python completion_v2/verify_perception_detail.py
.venv/bin/python completion_v2/verify_new_results.py dynamics
.venv/bin/python completion_v2/verify_counterfactual.py
.venv/bin/python completion_v2/dynamics_geometry.py
.venv/bin/python completion_v2/verify_new_results.py generation
.venv/bin/python completion_v2/verify_svib.py
.venv/bin/python completion_v2/generation_reconstruction.py
.venv/bin/python completion_v2/dynamics_truth_geometry.py
```

Existing per-unit verification receipts are reused. For a genuinely new replay, copy the release to a new directory and remove only the copied verification receipts, or use the independent release verifier. Do not delete historical experiment outputs. Saved prediction and checkpoint hashes are checked; pixel/latent replays cover the fixed sample counts recorded in each receipt. Dynamics replays all stored episodes and counterfactuals. These checks do not independently retrain all models, establish scientific correctness, or replace independent human evaluation.

## Recover unfinished execution

Do not launch two writers for the same branch. Inspect `execution/journal.jsonl`, driver logs, and process status first.

```sh
# Generation data/development, if not already complete:
.venv/bin/python completion_v2/generation_run.py prepare
.venv/bin/python completion_v2/generation_run.py train --seed 880101 --init 0
# Host MPS access is required for the nine confirmation paths:
.venv/bin/python completion_v2/generation_host_driver.py

# Initial perception:
.venv/bin/python completion_v2/perception_evaluate.py prepare
.venv/bin/python completion_v2/perception.py --kind global --seed 0
.venv/bin/python completion_v2/perception.py --kind crop --seed 0
.venv/bin/python completion_v2/driver.py perception
# Corrected full-pose, raster-detail follow-up:
.venv/bin/python completion_v2/perception_detail_driver.py

# Dynamics development then confirmation:
.venv/bin/python completion_v2/dynamics.py --kind one --seed 640031 --init 0
.venv/bin/python completion_v2/dynamics.py --kind multi --seed 640031 --init 0
.venv/bin/python completion_v2/dynamics.py --kind multi_bounded --seed 640031 --init 0
.venv/bin/python completion_v2/dynamics_prepare.py
.venv/bin/python completion_v2/driver.py dynamics

# Official SVIB archive must already match svib/data_manifest.json:
.venv/bin/python completion_v2/svib_prepare.py
# This intentionally fails without host MPS, rather than silently changing device:
.venv/bin/python completion_v2/svib_run.py
```

The download archive is `svib_dsprites_hard.archive` (gzip tar, despite its generic suffix). Its exact SHA-256 and official origin are in `svib/data_manifest.json`. Only Hard dSprites Single Atomic was packed/evaluated; the other downloaded tasks were not trained. Do not treat downloading the archive as reproducing all tasks. `svib_data_audit.py` verifies the arrays, semantic shape swap, and exact source-image train/test overlap.

Protocols lock source hashes. If a frozen source differs, stop and create an explicit amendment/new version rather than editing the protocol to bypass the check. `PERCEPTION_AMENDMENT.md` corrects the initial geometric/raster symmetry interpretation; original scores and files remain intact. `generation/device_amendment.json` records confirmation-only MPS acceleration and its numerical probe.

## Statistical units and comparison boundaries

- Generation and dynamics: three new dataset seeds × three initializations; aggregate initializations within data seed first. Development data are excluded from confirmation summaries.
- Perception studies: a shared historical training dataset; three initializations and three new evaluation seeds. The second study uses fresh evaluation seeds and changes both crop-value representation and pose supervision relative to the first study. Its binary/alpha arms are paired; cross-study differences are descriptive, not a clean single-factor comparison.
- SVIB: one official 64,000/8,000 split, three initializations. Full split does not mean full suite or published baseline reproduction. Fixed five epochs; no convergence claim.
- Dynamics physics reference is repeated for bookkeeping across initialization paths; these identical predictions are not independent learned replications.
- Generated samples share noise within data seed across initializations and arms. Per-seed pooled diversity contains 384 samples per condition but is not 384 independent noise draws.
- Original people/AI ratings retain their provenance. No new independent human ratings were collected in this execution.
