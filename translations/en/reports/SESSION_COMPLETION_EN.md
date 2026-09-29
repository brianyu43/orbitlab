# 15th episode requirements and completion evidence

2026-09-23. Compared to the original PLAN_KO.md Chapter 8 and the Word plan. The 15th iteration is based on the work slots. It is based on actual execution and output, and the scheduled time has not been converted into actual labor/learning time.

| Episode | Performance and Judgment | Evidence |
| --- | --- | --- |
| 1 | Original MPS forward/backward parity and smoke passage | original_doctor_mps.json, original_smoke_mps.json |
| 2 | Orbit duplicate removal from 4,864 base scenes, split/factor verification | ../data/manifest.json, input_32.png, c4_orbits.png, unit test |
| 3 | Original 32px, n256, batch16, 200-step pilot execution. Blurred with MAE 0.2335 in the foreground | ../runs/original_pilot_eq/, ../runs/original_pilot_eq_analysis/ |
| 4 | C4 encoder/decoder, synthesis, DFT, description of the range of guarantees and unit tests | C4_MATH_KO.md, ../tests/test_research.py, unit_tests.log |
| 5 | B1 n1024 3,000-step baseline learning·fixed/worst recovery test | ../runs/primary_aug_n1024_s0/ |
| 6 | B2 n1024 3,000-step comparison, foreground metrics and visual results | ../runs/primary_equivariant_n1024_s0/, ae_results.csv |
| 7 | B3 invariant-only, Fourier energy and removal/amplification intervention and probes | ../runs/diagnostic_invariant_n1024_s0/analysis/, B2 band_*.png, band_interventions.json |
| 8 | train standardization/compatibility, validation regularization, independent test/OOD and A90 closure | representation_results.csv, 8 analysis/probes.json, learned_action.png |
| 9 | Weekly comparison seed 1/2 repetition. Three seed material and CI | ../configs/experiment_matrix.json, ae_summary.json, each *_samples.csv |
| 10 | n256/4096 Each 3 seeds, 3 times parameter control, 6 times time control | benchmark.json, ae_results.csv, figures/fairness.png |
| 11 | 3,000-step B1 seed2 entry failed. All 6,000-step complementation of both models passed, then frozen AE flow 6 times | quality_gate.json, quality_gate_repair.json, repair_visual_review.json, flow run.json |
| 12 | Generate 576 identical noise 576 for each model/seed in Gaussian/16/32/64-step. B2 GIF rotation control | 8 samples/samples_*.png, samples.pt, latent_rotation_*.gif for flow |
| 13 | shape/color, occupancy, centroid, coverage, accurate train NN, worst grid test | generation_results.csv, generation_summary.json, nn_reference.json, result_visual_review.json |
| 14 | Original/data/checkpoint hash, sample count, AE freeze, noise pairing, actual resource·time·constraint audit | completion_checks.json, final_audit.json, ../RESULTS_KO.md Section 7 |
| 15 | Description·Conclusion·Reproduce command·6 chapter presentation·6 minute manuscript·Storage result package | ../README.md, ../REPRODUCE.md, ../deliverables/, ../RELEASE_MANIFEST.json |

We distinguish between completion and hypothesis success. We observed that H1's universal data efficiency advantage is not yet realized, H2's invariant pooling loss in position and direction is relative improvement compared to observation, H3's condition alignment is imperfect, and H4's parameter sharing and difference in actual costs are observed. The second Mac's full retraining, flow peak RSS, human judgment on outputs, and repeated data seed sampling were not performed and were not used as evidence of completion.

The results of the 3,000-step original experiment and the 6,000-step generation correction were separately preserved. The top 3,000 fixed steps remaining in the correction config were corrected to 6,000, and all actual execution tasks were set to 6,000. The original files and config_metadata_correction.json were preserved together before the modifications.
