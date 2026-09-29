# OrbitLab completion-v2 technical report

The prescribed experiments and primary replays are complete. Capability limitations remain: unconstrained generation diversity, occlusion, and autonomous object-count generalization. These are findings, not missing runs. The next research proposal is separate from executed experiments. These are separate diagnostic systems, not one jointly trained end-to-end world model; see `INFORMATION_BUDGET.md` for inputs, learned parts and privileged priors.

## Experiment inventory

| Study | Completed units | Verified |
| --- | --- | --- |
| perception | 90 | 90 |
| perception_detail | 90 | 90 |
| dynamics | 1404 | 1404 |
| generation | 22 | 22 |
| svib | 6 | 6 |

A unit is a saved model/condition evaluation, not an independent dataset. Dynamics additionally has 18 development units. Generation's 22 units include four development comparisons and 18 confirmation comparisons. New training comprises 50 generation components (including unselected diagnostic decoders), 12 perception readers, 30 dynamics models including development, and six SVIB models: 98 components. Historical controllers/evaluators are reused rather than counted as new training.

## Generation

Historical diagnosis used 18,432 rotated reconstructions from nine paths, with renderer radius thirds. Small/medium/large arrow test strict acceptance was 0.1143/0.7053/0.9451. Rotations are not independent scenes. The new development comparison crossed uniform versus shape/color/true-size stratified flow sampling with standard versus foreground/edge-normalized decoder loss. AE and each decoder: 6,000 updates; each flow: 16,000. The alternative loss is per-image foreground MSE + 0.1 background MSE + 0.25 summed spatial-gradient MSE normalized by foreground area.

The locked candidate is stratified flow with standard decoder. No development candidate passed both diversity gates, so subsequent 3-data-seed × 3-initialization results are diagnostic replication, not a validated successful selection. Data seeds 880201–880203 are separate from development 880101. Confirmation AE/decoder training used an explicitly recorded MPS FP32 CPU-output bridge; flow/evaluation remained CPU. Paired arms share AE/normalization and noise. Forward/gradient accelerator probes passed; this does not guarantee identical CPU/MPS optimization trajectories.

| Variant | Seen strict | Unseen strict | Seen shape | Unseen shape | Seen color | Unseen color |
| --- | --- | --- | --- | --- | --- | --- |
| stratified_flow/standard_decoder | 39.67% | 32.16% | 72.60% | 68.62% | 99.49% | 99.33% |
| uniform_flow/standard_decoder | 39.75% | 31.38% | 72.57% | 68.08% | 99.34% | 98.89% |

Paired strict-rate differences, candidate minus baseline in percentage points, averaging initializations within each data seed:

| Data seed | Seen change (pp) | Unseen change (pp) |
| --- | --- | --- |
| 880201 | -0.12 | -0.26 |
| 880202 | -0.08 | +1.11 |
| 880203 | -0.07 | +1.50 |

Individual confirmation diversity passes: all-image 14/18, strict-subset 0/18. Pooled diversity passes: all-image 5/6, strict-subset 0/6. Pooling is within data seed over initializations with shared noise; it is not independent noise expansion. Gates remain original and unchanged. Per-condition coverage, total variation, out-of-range and insufficient-count diagnostics are in `aggregate.json`.

Post-hoc confirmation reconstruction compares both trained decoder losses on identical real encoded inputs, stratified by true renderer radius; it did not change candidate selection.

| Decoder | Small arrow test | Medium arrow test | Large arrow test | Small arrow OOD |
| --- | --- | --- | --- | --- |
| standard | 14.43% | 74.40% | 91.30% | 13.93% |
| area_edge | 36.82% | 90.34% | 93.72% | 28.86% |

36,864 rotated reconstruction rows are retained; 64 images per each of 18 decoder paths were replayed. Values are pooled descriptive rates, not independent-scene confidence estimates.

## Perception and editing

| Study/model | 2 objects | Unseen pairs | 3 objects | 4 objects | Occlusion |
| --- | --- | --- | --- | --- | --- |
| perception/global | 0.07% | 0.26% | 0.00% | 0.00% | 0.00% |
| perception/crop | 78.21% | 76.48% | 67.62% | 61.72% | 12.57% |
| perception_detail/binary | 78.73% | 82.55% | 74.22% | 67.53% | 15.89% |
| perception_detail/alpha | 100.00% | 100.00% | 100.00% | 100.00% | 12.34% |

Both initial arms use known six-color palette directions, distinct object colors, no ground-truth inference masks, supervised shape/radius/center/pose, and the same CNN. Global masks are resized 64→33; crop preserves a native 33px neighborhood and predicts an offset. Alignment, sampling resolution and effective pixel-scale coordinate-loss weighting all change: coordinate normalization is 31.5 globally versus 8 for crops, producing 15.50390625 times the coordinate penalty per squared pixel error in crop arms. This is not an isolated cropping effect. Six models, 6,000 updates each, share the original 1,024-scene training dataset. Three evaluation seeds do not supply independent training-set replications. Evaluations use RGB/clicks, a reused analytic or learned controller, and the known renderer.

The initial protocol's assertion of exact zigzag raster symmetry was incorrect. Original modulo-two pose training and all outputs are retained. The supplemental geometric-equivalence audit's “unobservable” field names are withdrawn as an interpretation; they are not information-theoretic evidence. `renderer_detail_audit.json` directly compares raw rasters and preprocessing. The follow-up freezes binary/intensity crops with full four-way pose supervision, paired architecture/seeds/budget, fresh evaluation seeds 885201–885203 and prior source/target orbit exclusion. Its effect is representation of raster detail, not proof of abstract semantic pose understanding. New renderer tests are proposed separately.

## Autonomous dynamics

Fresh data seeds 882201–882203 × three initializations; three trained arms at 4,000 updates: one-step, eight-step recurrent loss, and eight-step with bounded feedback. Inference-only projections provide additional controls. Velocity bounds derive from training velocity extrema; walls/radii are known priors. The validation-selected arm is multi, and all prespecified arms are retained. Evaluation has no teacher forcing after initialization, uses 61 future steps, and separates oracle from RGB-measured initial states.

| Model | 2 objects: 16-step px | 2 objects: 61-step px | 4 objects: 61-step px | 4 objects: 61-step failure |
| --- | --- | --- | --- | --- |
| one | 5.131 | 15.92 | 1.697e+22 | 0.35% |
| one_bounded | 4.796 | 15.59 | 17.53 | 0.00% |
| multi | 2.26 | 11.56 | 1.15e+36 | 99.65% |
| multi_projected | 2.717 | 13.8 | 17.41 | 0.00% |
| multi_bounded | 2.516 | 12.16 | 16.99 | 0.00% |
| force_wall | 1.055 | 8.039 | 13.79 | 0.00% |

Table: variable force, oracle input. Position error is endpoint absolute coordinate error over color-matched objects, averaged per scene and then per unit/data seed. Nonfinite endpoint errors are excluded from the MAE denominator; missing-object counts and cumulative nonfinite/finite-blowup rates are separate. Therefore finite MAE alone is insufficient. The finite-blowup threshold is any position coordinate exceeding absolute 64px up to the horizon. Geometry diagnostics separately record wall penetration and interobject overlap with estimated input radii. Projection is not learned physical understanding.

| Model | Wall >1px episodes (finite) | Overlap >1px episodes (finite) |
| --- | --- | --- |
| one | 76.74% | 92.36% |
| multi | 100.00% | 98.38% |
| multi_bounded | 0.00% | 99.83% |
| force_wall | 0.00% | 99.48% |

Geometry table: variable-force/oracle/count4, cumulative to 61 steps, finite trajectories only. Simulator truth across all 39 confirmation dataset groups has zero >1px wall/overlap episode rates under the same diagnostic (`dynamics_truth_geometry.json`). Thus no catastrophic numeric blow-up does not imply physically valid contact dynamics.

Counterfactuals negate the prescribed targeted action while preserving initial input; all saved counterfactual rollouts were replayed.

| Model | 16-step target response MAE | 16-step nontarget response MAE |
| --- | --- | --- |
| one | 7.814 | 2.275 |
| multi | 3.007 | 2.293 |
| multi_bounded | 3.463 | 2.325 |
| force_wall | 1.763 | 1.695 |

Table: variable-force/oracle/test at 16 steps. Zero-response errors are 13.12px (target) and 1.695px (nontarget). Direct target response gains do not establish correct indirect interaction effects. Target and nontarget response errors are compared against zero-response baselines. The physics reference knows net force/drag, walls and simulator discretization, omits disk-disk interaction, and is not a learned comparator with identical prior information.

## Official external data

SVIB Hard dSprites Single Atomic, official 64,000 training and 8,000 test pairs at 128px; five complete shuffled epochs, batch 32, Adam 0.0005, 10,000 updates, final checkpoint, no test-driven selection. Paired initialization and sampling orders; C4 costs more compute. One fixed split, three initializations, no convergence claim. The C4 commutation check validates the model implementation, not equivariance or identifiability of the benchmark target rule; no separate rotated-SVIB test distribution was evaluated.

| Model | Image MSE | Changed-region MSE | Preserved-region MSE | Unchanged-scene MSE |
| --- | --- | --- | --- | --- |
| plain | 0.0191 | 0.1449 | 0.00429 | 0.008049 |
| c4 | 0.01398 | 0.09717 | 0.004291 | 0.009303 |

Identity image MSE: 0.02313. C4 versus plain mean image-MSE reduction: 26.79%. Of 8,000 scenes, 6,007 change and 1,993 have equal shapes and are unchanged. Changed-pixel metrics exclude unchanged scenes; preserved and unchanged-scene damage are not omitted. Metadata verified shape swap and other-factor preservation for all 72,000 pairs, no exact source RGB overlap, and 12 test object attribute combinations absent from training. This is not full 12-task SVIB reproduction, published model-budget reproduction, LPIPS evaluation or independent human validation.

## Verification, preservation and limitations

Checkpoints, datasets, protocols and result artifacts have SHA-256 provenance. A separate data audit recomputed C4 hashes for 44,032 procedural dataset images for the generation study (including reference images), 3,840 perception sources and 17,685 targets, and 6,720 dynamics initial physical/image states. Cross-family/prior overlap was zero; multiple targets of one source remain the same family. Saved metrics are reaggregated and checkpoint inference is replayed with explicit sample counts/tolerances in receipts. Dynamics replays all stored original and counterfactual trajectories; perception replays eight image inferences per unit, all command states, and eight rendered predictions; generation rescores every sample and replays 128 latents/64 images per unit; SVIB aggregates all 8,000 rows and CPU-replays 64 images per model, with C4 commutation checks. This is not an independently implemented evaluator or full clean-install retraining.

Historical artifacts are preserved. The 1,324-file original manifest matches. The 70,514-file followup manifest had four preexisting documentation/status mismatches; final auditing requires exactly those mismatches and unchanged starting hashes. They are not silently repaired. Original 9 human/231 AI ratings remain distinguishable, with no new independent human evaluation.

Reports use descriptive data-seed means/ranges, not narrow pseudo-replicated confidence intervals. Physical controls repeated across model-initialization paths do not add independent evidence. See `REPRODUCE.md`, `PERCEPTION_AMENDMENT.md`, the final scope audit, and the Korean concrete next-stage plan.
