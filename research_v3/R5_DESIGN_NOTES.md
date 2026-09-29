# R5 preparation notes — training/evaluation protocol not frozen yet

The authoritative scope remains `completion_v2/NEXT_RESEARCH_PLAN_KO.md` R5.
Preparation below does not complete the external benchmark or integrated model.

## Verified sources and existing local data

- Official project: https://systematic-visual-imagination.github.io/
- Official paper: https://arxiv.org/html/2311.09064v1
- Official code: `followup/external/svib`, commit
  `7eec57ef3c2ea7a2c7389cb6cd470eb3600fa57c` (read-only).
- The paper's Appendix A.6 Question 4 explicitly states dataset CC0. Code
  LICENSE is CC0-1.0. Copies and exact sources are in `r5/references/`.
- All four Hard dSprites tasks have 64,000 train and 8,000 test episodes.
  Existing approximately2GB archive SHA is locked by `r5/intake_protocol.json`.
- `r5_intake.py` preserves all PNG/JSON/mask bytes in task-specific payload.bin
  plus SQLite offset/length/hash indexes. 288,000 episodes,1,728,000 members
  were compared byte-for-byte against a second full archive pass. Approximately
  3GB prepared payload+index storage. Source archives and old arrays unchanged.
- This is lossless data preparation. Full decoded semantic, factor-binding and
  split audits remain pending. `r5_intake.Store` reads raw files or RGB images.
- Official new Hard archive IDs: CLEVR `1NkbROrTu38_Wydp_3z7v_1AtSkgXEF0s`
  (project says13GB), CLEVRTex `1YgJwN6Efl7kXlXRRKiX8g9BxIKBye1zx` (20GB).
  Neither has been downloaded. A streamed selective task intake can preserve
  selected original members while avoiding keeping all four tasks twice.

## Required experiment distinctions

1. Hard dSprites Single Atomic: validation-selected5/10/20epochs, plain/C4/object
   models and input-copy control, identical train examples/update counts.
   Prepare a deterministic train-only validation partition before training and
   lock the policy. The official test was seen in previous research; don't call
   it new unseen project-level evidence. Report changed/preserved/unchanged scene
   errors and actual object attributes, not full-frame MSE alone.
2. Additional dSprites Multiple Atomic and both Non-Atomic variants are now
   available locally. Then one CLEVR task, then one CLEVRTex task. Keep native
   image formats, official splits and provenance. Select specific3Dtask(s) before
   reading model results. Any input labels or mask supervision must be disclosed
   and matched or explicitly treated as an extra-information arm.
3. Independent attributes need a calibrated reader. Do not score predicted
   pixels by simply reusing the input labels. If true masks/ROIs are used only
   for an evaluation diagnostic, name that privileged localization explicitly.
4. The same moving-disk data must train reader, transition and neural decoder
   before integration. Current R2 polygons and R3 disks cannot be directly
   connected and called a matched integrated model. Use true-state/force/renderer
   swaps and counterfactual action/non-target response diagnostics.
5. Independent human evaluation is conditional on improved results and actual
   available participants. Keep the existing9human/231AI records untouched;
   neither AI scoring nor our plot inspection is new human validation.

## Symmetry audit is necessary

Official AppendixD.1.1 says Multiple Non-Atomic uses object position quadrants
in color/material and size updates. A quarter-turn can change these target
attributes, so forced image-space C4 equivariance need not match the task rule.
Read the exact source lookup ordering and quadrant convention before a numeric
audit. Do not presume C4helps every benchmark, or conflate an implementation
commutation test with symmetry of the official target function. In3Dscenes,
image rotations also differ from physical scene rotations.

## Cost / unfinished design

Historical Single Atomic native128px training used MPS, batch32,10,000updates.
Measured prior plain times435–484seconds and C4times630–655seconds (three initial-
ization runs). These are prior measurements, not new hardware timing. The new
object architecture needs100-update timing and numerical/device checks.

A20epoch comparison across every task and three initializations could exceed
the initial12hourR5cap; estimate the entire queue before dispatch. Do not silently
reduce epochs/data to label the original scope complete. First lock development
and any confirmation rule, preserve planned vs completed units, and report real
resource limits. None of the R5models, validation splits or training criteria
have been implemented/frozen in this preparation step.
