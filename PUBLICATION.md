# Public distribution and reproducibility

Publication was requested on 2026-09-29. This adds a public source-and-evidence
distribution to the local project; it does not close the active follow-up plan.

## Included and excluded

Included are implementation and experiment scripts, selected source snapshots,
tests, frozen protocols, the small original synthetic dataset, research reports,
selected figures, completion-v2 aggregates/audits, and verified R1/R3 summaries.
R2 source and preflights are included; R2 remains incomplete at this publication
boundary. Partial scores are not presented as final findings.

`PUBLICATION_MANIFEST.json` records the SHA-256 and size of every other published
file at the initial-publication commit. It describes that snapshot only.

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
