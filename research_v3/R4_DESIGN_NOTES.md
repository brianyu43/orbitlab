# R4 implementation notes — not an executed result or frozen protocol

The authoritative requirement remains `completion_v2/NEXT_RESEARCH_PLAN_KO.md`,
R4. This note records source inspection while the R2 decoder finishes. Do not
claim R4 is complete from this note or start confirmation before the locked gate.

## Sources already inspected

- `completion_v2/generation_run.py`: baseline6KAE,6Kdecoder,16Kflow; paired uniform
  versus area/edge decoder,128samples per factor pair,384reference samples/pair.
- `completion_v2/generation_reconstruction.py`: reconstruction grouped by actual
  renderer radius, not a radius inferred from generated pixels.
- `followup/decoder_study.py`: `ControlledDecoder('logit_mean')` and existing
  `oracle_codes(labels, metadata, rotation)` provide genuine privileged renderer
  factors as a4×16regular representation. Do not call an encoder latent a ground-
  truth latent, or an image-conditioned numerical inversion a proven optimum.
- `work/orbitlab.py`: shape2 is the arrow; canonical radius is uniform.13–.24 of64px,
  center uniform.32–.68; exact90degree rotations are applied after PIL2/Lanczos
  rendering. `rho` rolls the four latent blocks.
- `generation_recovery/p4_prepare.py`, `p1_evaluate.py`, `p4_evaluate.py`: historical
  orbit exclusion, frozen strict evaluator, categorical diversity and strict-
  subset diversity. Do not mutate their paths/protocols or regenerate old data.

## Constraints for the forthcoming implementation

Use889101development,889201–203conditional confirmation;1024training scenes,
whole-C4-orbit separation, all4rotations within each scene's split. Keep original
strict thresholds and diversity-reader bins. Matched64-step generation noise is
4×16for every candidate; generated sizes remain estimates. Ground-truth size is
available only for reconstruction diagnostics.

Four primary candidates remain: current decoder, area/edge loss, decoder retaining
spatial feature maps, and an explicit coordinate decoder. Preserve the same
baseline encoder and latent normalization for the current/area-edge loss
comparison. A native spatial encoder or coordinate supervision changes more than
one factor; disclose that information budget and parameter/update/time costs.

Evaluate true renderer-factor/oracle decoding, actual encoder-based decoding, and
flow-based generation separately. The existing semantic oracle and an arbitrary
learned embedding are different coordinate systems. If separate decoders must be
trained for those roles, disclose that their weights differ; the difference is
not a pure intervention on one fixed decoder. A same-decoder, target-assisted
latent inversion may be an additional upper-bound diagnostic but cannot replace
true factor-code decoding or be called a global optimum.

The selection gate must use actual-encoder small-arrow reconstruction (>=80%),
actual generation strict improvement (>=10percentage points over the baseline),
and both all-image and accepted-subset historical diversity gates. Oracle-only
success cannot pass the end-to-end gate. Keep per-condition sample counts and
scene-level units; rotations are not independent scenes. On failure, diagnose
without automatically expanding3data×3initializations. On success, lock baseline
and one candidate before those confirmations.

Read `generation_recovery/p0.py` and evaluator sources again before writing the
final data/training/evaluation protocol. Do not copy the old fixed1800second
trainer cap or import-time scripts into a new live driver without inspection.
Measure100updates for each new architecture and account for the12hour/30GBstage
cap. Nothing in this note reduces the original R4/R5 scope.
