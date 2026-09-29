# Original code thanks — 2026-09-23

## Collection and range

We copied the top-level iCloud Drive files `OrbitLab_starter.zip` (31,141 bytes) and `OrbitLab_Mac_실험계획서.docx` (53,491 bytes). The original SHA-256 hash and the list of 11 ZIP files are in `source_manifest.json`. The compressed file path and CRC checks passed. The original files are `sources/extracted/66642ae290950e82/orbitlab_starter`, and the executable copy is `work`. We extracted 299 paragraphs from the Word XML file to `sources/word_plan_extracted.txt` and compared the PLAN in the ZIP file with the 15-time schedule and more than 512 generated items, as well as the 16/32/64 step requirements.

I read all three original Python files, the README, PLAN, subsequent implementation instructions, requirements, and validation JSON. The original generates synthetic data without downloading or uploading data/weights. The selected DINO script includes a separate weight download and is not executed in the core experiments.

## Structure checked

- The Encoder shares `h(R_-k x)` in four directions. It does not implement the layers of the group convolution paper.
- The Decoder is symmetrized as `mean_k R_k d(z_k)`. Since the mean follows the sigmoid function of the common decoder, the symmetry is accurate, but the image may become blurred.
- B1 total latent=64, B2=4×16 but the independent encoder/decoder latent head size is different. The basic parameters are B1 315,075, B2 118,419.
- Flow takes shape/color one-hot as condition and averages the velocity field in C4. The batch+group common channel normalization of train latent is replaced by rho.
- The original distinguishes holdouts based on the train/val/test/ood RNG namespace, but does not remove pixel/orbit duplicates.
- The original learning saves the last fixed-step checkpoint. Validation is not used for intermediate selection. Probe uses train standardization and val regularization selection.
- The original AE checkpoint does not have optimizer/RNG/step, so there is no accurate restart learning function. The original analysis is the average of indicators per batch, and there are no per-scene statistics.
- The B1 fixed-rho GIF of the original `sample` is not guaranteed by meaningful rotation effects. Rotation control of B1 must be judged by the learned A90 results.

## Original actual execution

The original doctor/smoke was passed through MPS. Forward parity maximum error 1.19e-7, gradient parity 7.45e-9. Encoder error 0, decoder 5.96e-8, flow 0. Original 32×32, n=256, 200-step pilot foreground MAE=0.2335, and the actual restoration grid was blurred. Execution success and quality success are distinguished.

## Follow-up modification

The original orbitlab.py was modified to add subsequent features to `research.py`, `probes.py`, and `generation.py` without changing the original orbitlab.py.

- First fix up a maximum train pool of 4,096 and remove duplicates using orbit SHA-256. Then generate 256 each for independent ID val/test and combined OOD. Removed 4 train duplicates and 1 test duplicate. Nested n=256/1024/4096 are prefixes of the same frozen pool.
- Mini-batch samples are separated from model initialization according to the seed. B1/B2 receive the same original scene and external rotation minibatch. Due to the structural accuracy of B2, external rotation is duplicated, but the cost is measured.
- Store MSE/PSNR/foreground MAE/IoU/center error in CSV per sample and bundle C4 4 into base-scene and bootstrap.
- parameter match is set to B2 CNN width=31 (318,699, +1.15% compared to B1). latent 64 dimensions are retained.
- Maintain the final-step rules and store benchmarks, time, exposure counts, and code/data/checkpoint hashes for each run. Separate time budget matching and comparison into separate executions.
- added full/m0/m2/m13 probes, learned A90 and real decoded action, component removal and amplification, independent renderer-state evaluator, raw Gaussian/16/32/64-step flow generation, exact pixel train-neighbor search.
- The evaluator obtained 100% of shape/color in 512 clean base scenes×4 rotations of separate seeds. Since the accuracy of the blurry generated image is not guaranteed, we also report the template IoU and failed grid.

## Quality standards before the confirmed experiment

In the 64×64 preliminary training 1,000-step, the MAE before validation for B1/B2 was 0.0915/0.1015 respectively. Shape preservation was improved, but the contours of B2 were more blurred. Based on this validation, the final full experiment is fixed at 3,000 steps for both sides. test/OOD did not use this decision.

For the generation entry, validation must confirm that the overall MSE is less than half of the black baseline, foreground MAE ≤ 0.10, mean mask IoU ≥ 0.70, and the shape/color alignment of the reconstruction from the independent evaluator is ≥ 0.90. If the baseline fails, the generation results will not be claimed as quality success, and the cause will be recorded.

## Final execution results

29 comparisons of AE and 6 supplementary AE for generation, and 8 flows were actually completed. The 3,000-step B1 seed2 restoration color gate failure (88.28%) was preserved, and after both sides were 6,000-step supplemented, the six models passed the standard. No claim that the final latent shape linear probe achieved 90% is made. Although the conditions for actual generation were consistent, B2 was high, but there were still blurry and shape errors. The consistency of the three original codes in bytes and execution checkpoints/CSV/partitioning were confirmed in the final audit.
