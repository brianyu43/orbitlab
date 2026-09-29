# OrbitLab experiment results — 15th round completed report

2026-09-23 · M5 Pro / 64GB · macOS 27 arm64 · Python 3.13.14 · PyTorch 2.14.0 / MPS FP32

**Rotation consistency was confirmed, but the C4 model did not outperform on every reconstruction and generation metric.** In AE learning at the same time, restoration of B1 was good, and in fixed-step flow generation, the condition matching rate of B2 was high. Generated images still have substantial blur and shape errors. We actually performed the required outputs for 15 task slots, and this does not mean that we learned for 15 days or 15×90 minutes.

## 1. Original materials and work actually performed

I copied the `OrbitLab_starter.zip` and `OrbitLab_Mac_실험계획서.docx` files from iCloud Drive to `sources/originals/` and verified their SHA-256, ZIP CRC, and paths. I checked the 11 files in the ZIP and the 299 paragraphs in the Word file. I preserved the [original list and hashes](../../reports/source_manifest.json), [original audit](reports/CODE_AUDIT_EN.md), and [recovered conversation](sources/conversation_plan.md).

The original `orbitlab.py`, `probe_latents.py`, and `extract_dino.py` were not modified. Additional implementations are located in `research.py`, `probes.py`, and `generation.py`. The work included **43 training runs**, including 29 AE comparisons, 6 AE corrections for generation entry, and 8 flow runs. Separately, the original 200-step pilot run and 1,000-step calibration run were performed, along with CPU replay tests for doctor/smoke/unit tests and the saved checkpoint. DINO and dynamics extensions were not performed within this 15-run range.

## 2. Design, data, entry criteria

We used 4 types of 64×64 RGB shapes and 6 colors. The data seed was fixed to 42. `(shape,color)=(0,0),(1,1),(2,2),(3,3)` was excluded from the training process. The train pool is 4,096, with 256 samples for validation/test/OOD, and the C4 orbit hash for 4,864 scenes is all different. We removed duplicates between 4 train candidates and 1 test candidate. The train 256/1,024/4,096 are used by overlapping the front part of the fixed pool. [Data specification](../../data/manifest.json)

Initialization seed 0/1/2, AE batch 32, AdamW lr 0.001, FP32, latent total 64 dimensions, fixed final checkpoint rules were set. The external minibatch original of B1/B2 and the rotation noise distribution are identical. B2 evaluates four directions internally. The main experiment runs 6 times, with 3,000 steps; data size, parameters, and time control are separate search comparisons. Parameter control B2 width=31 is +1.15% compared to B1, but the number of parameters is not exactly the same.

The MAE of the original 200-step pilot was blurred to 0.2335. After validation-only calibration, the main comparison was fixed at 3,000 steps. The generation entry criterion was that the validation MSE < 0.5 times the black baseline, the image MAE ≤ 0.10, IoU ≥ 0.70, the accuracy of the shape/color evaluator for the reconstructed image was ≥ 0.90, and the shape preservation of the fixed grid. This **reconstruction image evaluator criterion differs from the latent linear probe 90%.** The latent shape probe failed to achieve 90%.

The 3,000-step B1 seed-2 reconstruction achieved 88.28% color accuracy, below the entry threshold. Without lowering the threshold, both the two-model × 3 seed models were retrained on 6,000-steps and passed all six models. This supplementary decision was made during validation before opening the test/OOD results. The supplementary models are exclusively for generation and did not replace the 3,000-step AE table below. [Number gate](../../reports/quality_gate_repair.json), [Actual image review](../../reports/repair_visual_review.json)

## 3. AE comparison: Data efficiency and cost

Each cell is the average ± sample standard deviation of the 3 initial seeds. test/OOD consists of 256 independent scenes × 4 rotations for each. The raw data CSV and the 95% CI of each model are located in `runs/<id>/analysis/`. Bootstrap first averages the original scenes from four rotations and then resamples them 1,000 times. The seed standard deviation and scene CI measure different levels of uncertainty.

| Comparison | Parameters | Mean steps | Training time (s) | Test foreground MAE ↓ | Test IoU ↑ | OOD foreground MAE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| B1 rotation enhancement | 315,075 | 3,000 | 13.83 ± 0.16 | 0.0697 ± 0.0289 | 0.8945 ± 0.0052 | 0.0707 ± 0.0260 |
| B2 C4 | 118,419 | 3,000 | 28.75 ± 0.04 | 0.0716 ± 0.0058 | 0.8298 ± 0.0158 | 0.0777 ± 0.0076 |
| B2 parameter control | 318,699 | 3,000 | 54.79 ± 0.01 | 0.0576 ± 0.0034 | 0.8781 ± 0.0030 | 0.0643 ± 0.0019 |
| B1 time control | 315,075 | 5,667 | 26.94 ± 0.00 | 0.0446 ± 0.0012 | 0.9378 ± 0.0032 | 0.0459 ± 0.0012 |
| B2 time control | 118,419 | 2,765 | 26.94 ± 0.00 | 0.0727 ± 0.0085 | 0.8170 ± 0.0051 | 0.0788 ± 0.0105 |

The actual learning time for time control is approximately 26.94 seconds on both sides. In this budget, B1 performed about 5,667 steps, and B2 performed about 2,765 steps. The overall MAE is 0.04457 versus 0.07268. In this implementation, baseline B2 is slower despite having fewer parameters. The overall MAE of the expanded B2 improved, but the learning time increased to 54.79 seconds. These two comparisons are not mixed in a single fairness table.

| Number of scenes | B1 test overview MAE | B2 test overview MAE | B2−B1 paired average |
| --- | --- | --- | --- |
| 256 | 0.0654 ± 0.0025 | 0.1221 ± 0.0217 | +0.05669 |
| 1024 | 0.0697 ± 0.0289 | 0.0716 ± 0.0058 | +0.00183 |
| 4096 | 0.0622 ± 0.0185 | 0.0594 ± 0.0033 | -0.00279 |

In 256 cases, B2 performed worse than all seeds, and in 1,024/4,096 cases, the order of seed performance changed. Therefore, **this experiment does not support the universal data efficiency advantage of the C4 configuration.** The same steps are not the same epoch for different data sizes. Single data seeds and three initializations do not claim population significance.

![Number of data and actual seed value](../../reports/figures/data_efficiency.png)

## 4. Expression and intervention: distinguishing accurate symmetry and meaning

The test rotation error of this C4 encoder is 0, and the maximum decoder error is approximately 1.19e-7. This is a structural verification of the shared encoder and symmetry-preserving decoder. This design does not replicate all group-convolution layers in the paper. [C4 formulas and limitations](reports/C4_MATH_EN.md), [Cohen·Welling original paper](https://proceedings.mlr.press/v48/cohenc16.html)

The probe was standardized and fitted with **256 scenes × 4 rotations before training**, and C or ridge alpha was selected using 256 validation samples. Do not confuse the n=1,024 for AE training and n=256 for probe fitting. The test/OOD, CI, and rank results for full/m0/m2/m13 were saved in [CSV](../../reports/representation_results.csv). The B2 full shape accuracy is 42.2/43.0/45.3% for each seed, and the color accuracy is 98.4/96.5/97.7%. The direction probe for m0 is 25% for all three seeds.

When only m0 is left from seed 0, the overall MAE increased from 0.07825 to 0.29146, and the center error increased from 0.00605 to 0.14627. As shown in the bottom row of the figure, a clump with rotational symmetry merged with the contour. The B3 invariant-only also had a test IoU of 0.3435, which was lower. This aligns with the result of losing position and direction. However, removing the band creates latent variables outside the learning distribution, so it cannot be interpreted as a pure meaning separation causal experiment. Even if the linear shape is lower in m1/m3, this does not provide evidence of non‑linear information.

![Input, full restoration, keep only m0](../../runs/primary_equivariant_n1024_s0/analysis/band_m0.png)

In general AE, the slot shift determined by the human is not treated as the correct rotation, but compared with the A90 learned from the train. The closure below is the four‑fold synthesis of standardized coordinates. The full‑matrix closure is also sensitive to directions that do not occupy data. For example, the in‑data closure of B3 is almost 0 while the matrix closure is 0.866, showing the difference.

| Execution | A90 relative MSE | A⁴ relative MSE within data | Frobenius relative to matrix A⁴−I | Rotation answer decode MSE |
| --- | --- | --- | --- | --- |
| primary_aug_n1024_s0 | 0.0339 | 0.0229 | 0.745 | 0.00175 |
| primary_equivariant_n1024_s0 | 2.41e-12 | 3.86e-11 | 0.000147 | 0.00257 |
| primary_aug_n1024_s1 | 0.0252 | 0.0159 | 0.873 | 0.00169 |
| primary_equivariant_n1024_s1 | 1.02e-12 | 1.63e-11 | 6.12e-05 | 0.00195 |
| primary_aug_n1024_s2 | 0.0204 | 0.0124 | 0.785 | 0.00270 |
| primary_equivariant_n1024_s2 | 7.74e-13 | 1.24e-11 | 4.15e-05 | 0.00206 |
| diagnostic_plain_n1024_s0 | 0.0752 | 0.0644 | 1.08 | 0.00614 |
| diagnostic_invariant_n1024_s0 | 2.23e-14 | 3.56e-13 | 0.866 | 0.02999 |

Even with a small residual error in B2 A90, the rotation answer decode MSE originally includes the recovery error. The large fixed-rho error in B1 was not used as the basis for performance inferiority.

## 5. Generate flow in fixed AE

We fixed the encoder/decoder of the 6,000-step AE and verified the actual weight invariance. Each flow has 4,000 optimizer steps, a batch size of 128, and the number of parameters on both sides is the same. B2 averages the velocity four times. We did not standardize each slot separately but used the batch+group common channel average and standard deviation. We learned the vector field of the straight path from latent to Gaussian and integrated the Euler ODE. This small experiment uses the principle of [Flow Matching](https://arxiv.org/abs/2210.02747) and is not a replication of large RAEs or world models.

For each model/seed, **the same 576 initial noise** was reused on Gaussian/16/32/64 steps. For each of the 24 conditions, 24 samples were used, resulting in 480 seen samples and 96 OOD samples. The noise SHA between B1/B2 also matches. The total of 8 flow × 4 conditions = 18,432 evaluation rows does not mean there are 18,432 independent noise samples. It means that the same noise was evaluated repeatedly.

In the clean 512-frame × 4 rotation of the independent renderer seed, the evaluator reported 100% for both shape and color, with 0 duplicates in the train orbit. Since the generated output is blurry, we cannot guarantee that this accuracy is transferred to the previous one. `valid` means the silhouette template IoU is ≥0.65 and not empty, which is not the human-verified success rate of the generation.

**64-step results, 3-seed average±SD, ratio 0-1:**

| Model / Condition | Shape | Color | Shape+Color | Form Valid | Valid & Shape+Color | Center coverage |
| --- | --- | --- | --- | --- | --- | --- |
| B1 / seen | 0.351 ± 0.019 | 0.963 ± 0.032 | 0.340 ± 0.016 | 0.728 ± 0.016 | 0.247 ± 0.021 | 1.000 ± 0.000 |
| B1 / ood | 0.319 ± 0.074 | 0.920 ± 0.061 | 0.295 ± 0.081 | 0.681 ± 0.069 | 0.205 ± 0.049 | 0.625 ± 0.108 |
| B2 / seen | 0.577 ± 0.030 | 0.985 ± 0.011 | 0.570 ± 0.036 | 0.844 ± 0.012 | 0.522 ± 0.033 | 1.000 ± 0.000 |
| B2 / ood | 0.552 ± 0.089 | 0.976 ± 0.016 | 0.545 ± 0.100 | 0.837 ± 0.016 | 0.507 ± 0.074 | 0.958 ± 0.036 |

Moving from Gaussian to flow improves condition matching. The simultaneous seen matching rate for B2 is 57.0%, while for B1 it is 34.0%. However, when both validity and condition matching are simultaneously required, the rates are approximately 52.2% and 24.7% respectively. Therefore, **the relative improvement in condition control is observed but not considered sufficient for high-quality generation completion.** OOD is not a new shape or color, but rather a combination of four previously observed factors. The condition distributions for OOD and seen are also not identical.

| ODE steps | B1 seen | B2 seen | B1 OOD | B2 OOD |
| --- | --- | --- | --- | --- |
| Gaussian | 0.043 ± 0.010 | 0.049 ± 0.014 | 0.014 ± 0.006 | 0.017 ± 0.006 |
| 16 | 0.333 ± 0.015 | 0.550 ± 0.027 | 0.292 ± 0.048 | 0.559 ± 0.084 |
| 32 | 0.342 ± 0.016 | 0.565 ± 0.036 | 0.302 ± 0.068 | 0.559 ± 0.105 |
| 64 | 0.340 ± 0.016 | 0.570 ± 0.036 | 0.295 ± 0.081 | 0.545 ± 0.100 |

The ODE steps increase condition did not always improve accuracy. The latent differences between 16→32 and 32→64 decreased, but since the scales of B1 and B2 latents were different, the raw latent MSE was not used for model quality comparison. The coupled-noise ODE equivariance error for the B2 seed set was 0, and the decoder exchange error was approximately 1.19e-7. The fixed-rho GIF for B1 lacks structural rotation guarantees. The B2 GIF is presented as evidence of rotation control.

![Gaussian/ODE and condition match](../../reports/figures/generation_conditions.png)

Time control flow is a search comparison of seed 0. It has matched approximately 10.57 seconds of the **flow learning part**. This is not a comparison of the entire pipeline, which also includes the training time of the previous AE or the cost of ODE inference.

| Model | Actual steps | Learning stage | seen simultaneous match | OOD simultaneous match |
| --- | --- | --- | --- | --- |
| aug | 5125 | 10.568 | 0.338 | 0.354 |
| equivariant | 3808 | 10.570 | 0.562 | 0.490 |

## 6. New sample, recent neighbors, failure cases

For the fixed 64-step run, we saved the first 48 samples, the 16 with the lowest template fit, and at least eight nearest-training-image comparisons. In B2, small stains, rounded corners, incorrect shapes, and inaccurate colors remained. In B1, the combination of separate pieces and shapes was more frequently observed. The image review list and range are found in [Visual Review Records](../../reports/result_visual_review.json).

![B2 fixed first 48 pieces](../../runs/flow_equivariant_s0/samples/samples_64.png)
![B2: 16 samples with the lowest template fit](../../runs/flow_equivariant_s0/samples/lowest_template_fit.png)
![B1: 16 samples with the lowest template fit](../../runs/flow_aug_s0/samples/lowest_template_fit.png)
![Generated sample above, recent-neighbor learning image below](../../runs/flow_equivariant_s0/samples/nearest_train_pairs.png)

The recent closest distance is the exact FP32 RGB MSE compared to all four rotations of 1,024 train scenes. The average distance from independent real tests is 0.00907, and OOD is 0.01087. The average seen of the generated is B1 0.01140, B2 0.00896. Less than 1e-7 copies were observed in the 64-step 3,456 images of the main generation. This pixel-based criterion almost cannot exclude all copied images or prove distribution learning. Especially blurring and background can reduce the distance.

The valid and condition-matching sample of seen filled out a 4×4 centroid grid for both models. OOD coverage was 0.625 for B1 and 0.958 for B2. The coverage represents the position grid of the entire condition, not an indication of the diversity of the overall distribution. The ratio of the area outside the reference region and the scale/centroid standard deviation were also preserved in `samples/metrics.json`. The occupancy of the generated seen was 0.0997 for B1 and 0.1094 for B2, which was larger than the actual test value of 0.0911. The observed position and size distributions also do not match completely.

## 7. Resources, reproduction, limitations

The benchmark measured 100 steps three times after 20 warm-up steps and synchronized MPS. During the training period, model/data setup and final evaluation were not included, and the speed of that segment was not used as the total execution time. The maximum process RSS among the 35 recorded baseline and complement AEs was 0.908 GiB. This is not the peak memory of the entire device including MPS, and the peak RSS of the flow was not measured separately. MPS driver allocation is the last time value of AE. Environment fallback was not used.

The range of single data generation seed, three initialization seed, and small synthetic C4 world is maintained. Parameter control is approximation matching, and time control is the value measured on this Mac. The shape evaluation metric for generation was only independently verified on clean data. The probe was limited to 256 train scenes. We did not lower thresholds after inspecting test results or replace only unfavorable seeds. The 3,000-step and 6,000-step AE results are not combined with the same learning budget.

Code, settings, original, checkpoint, CSV, environment lock, and hashes were preserved. [Reproduce command](REPRODUCE.md), [15th session evidence report](reports/SESSION_COMPLETION_EN.md), [Completion check](../../reports/completion_checks.json), [6-minute presentation](../../deliverables/OrbitLab_6min.pptx) can be reviewed together. The CPU replay test copies the absolute path of the storage flow from a separate copy and checks the path to generate samples with the same weight. This was not a second full run of all 43 trainings on another Mac.

## 8. The decision of the following experiment

The 15-round assessment covers the following: confirming the lack of universal data efficiency superiority in H1, verifying the loss in position and direction of invariant pooling in H2, assessing relative improvement in the alignment of generation conditions in H3 and the incomplete quality, and checking the difference in parameter count and actual costs in H4. Upon further execution, the appropriate order is to first assess the **independence of human evaluation/evaluation tool robustness for the generated outputs**, followed by repeated data seeding and fixing the overall AE+flow cost control. Before attaching larger models or DINO/world models, the current shape errors are used as the target for measurement. This subsequent step is an suggestion and was not included in the number of completed experiments this time.
