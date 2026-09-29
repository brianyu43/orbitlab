# Public distribution and reproducibility

Publication was requested on 2026-09-29. This adds a public source-and-evidence
distribution to the local project; it does not close the active follow-up plan.

## Included and excluded

Included are implementation and experiment scripts, selected source snapshots,
tests, frozen protocols, the small original synthetic dataset, research reports,
selected figures, completion-v2 aggregates/audits, and verified R1/R3 summaries.
The 2026-09-30 KST update includes verified R2/R4 and matched-disk reports,
external Single Atomic results, all six measurement-reader calibrations,
source/protocol/replay evidence, and the user's requested stop boundary.
Multiple Atomic plain/C4 models stopped at 5/20 epochs with local resumable
checkpoints. Other external primary tasks and the required conditional
confirmations remain unfinished. All experiment and recovery processes have
exited; no automatic restart is authorized. See
[the stop record](research_v3/STOPPED_KO.md). Partial scores are not final findings.

`PUBLICATION_MANIFEST.json` records the SHA-256 and size of every other published
file at the initial-publication commit. It describes that snapshot only.
It remains unchanged and can be checked at commit `cb9c250`.
`PUBLICATION_LATEST_MANIFEST.json` records the updated distribution, excluding
itself. `PUBLICATION_LATEST_VALIDATION.json` records the checks for this update.

Private conversation exports, original iCloud documents, participant review
packets, third-party repository mirrors, environments/caches, live process
records, later datasets, checkpoints, raw predictions and ZIP releases remain
local. No complete checkpoint/data release is available here at this time.

Historical documents may link to local-only artifacts or contain original
machine paths. These are provenance, not download URLs. An old report's local
ZIP link does not mean a GitHub Release asset has been uploaded.

## What a fresh clone can verify

The three test commands in [the public README](.github/README.md) use only the
published source, original dataset, and dependencies. They check C4 equivariance,
split provenance, metrics, object edits and reference dynamics.

Later verification receipts describe actual local replays. A clone cannot rerun
those score checks without the omitted datasets, weights and predictions.
Historical drivers may import the published source but still need local artifacts.
In particular, R5's diagnostic renderer imports Spriteworld from the local SVIB
checkout, which is not vendored in this public repository. Official source URLs
and the inspected commit are recorded in `research_v3/r5/references/sources.json`.
That reference receipt describes its original intake time, not live download
progress. Later CLEVR intake has completed locally; dataset payloads remain
excluded from this distribution.

Do not regenerate historical splits and call them the same experiment: some
samplers reject overlaps against earlier scene families not all distributed
here. Fresh training needs new output directories, explicit data provenance, and
a new protocol. Do not overwrite frozen protocols to bypass hash checks.

`RELEASE_MANIFEST.json` and `completion_v2/artifact_manifest.json` describe earlier
local releases, not the smaller GitHub distribution. Existing research files
remain unchanged. The new `.github/README.md` supplies a current overview while
preserving the historical root README, following GitHub's
[README location rules](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes).

Older notes excluding external publication describe an earlier authorization
boundary. Publication is separately authorized and does not change experiments.
Local verification is not independent replication or an independent human study.

## Rights

Public availability does not imply a new blanket open-source license. No
repository-wide license grant is added in this snapshot. Third-party materials
retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
