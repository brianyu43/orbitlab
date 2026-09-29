# OrbitLab

Local experiment comparing C4 rotation expression, restoration, and conditional latent flow. On 2026-09-23, we completed the 15th iteration output of the original plan.

**The restoration at the same time was good for B1 rotation enhancement AE, and the B2 C4 model had higher flow shape and color conditions matching. The blur and shape errors of the product remain.**

- [Results report: Design·measurement·failure·limitations](RESULTS_EN.md)
- [Proof of completion of 15th session](reports/SESSION_COMPLETION_EN.md)
- [Reproduce command and save result playback](REPRODUCE.md)
- [6-minute presentation PowerPoint](../../deliverables/OrbitLab_6min.pptx) · [Presentation manuscript](deliverables/TALK_6MIN_EN.md)
- [Original·Code·Results·Weight ZIP](../../deliverables/OrbitLab_15sessions_2026-09-23.zip) · [Content hash list](../../RELEASE_MANIFEST.json)
- [Full Execution Plan](EXECUTION_PLAN_EN.md) · [Original Code Audit](reports/CODE_AUDIT_EN.md) · [C4 Formulas](reports/C4_MATH_EN.md)

Actual execution: AE comparison 29 times + AE complementation for generation 6 times + flow 8 times = 43 times. Separate pilot/calibration and diagnostic tests were not included in this count. The 15th iteration is a work list and does not claim the actual required time to be 15×90 minutes.

| 3 seed average | B1 enhancement AE | B2 C4 AE |
| --- | --- | --- |
| MAE preview of the test at the same time ↓ | 0.0446 | 0.0727 |
| 64-step generation seen shape and color match simultaneously ↑ | 34.0% | 57.0% |
| 64-step generation combination OOD simultaneous matching ↑ | 29.5% | 54.5% |

Generation accuracy is the evaluation judgment of the template evaluator verified by a clean renderer. The possibility of errors in blurry generated images and the worst case scenarios are presented in the report. AE time control and generation comparison are different experiments.

`sources/originals/` contains the original ZIP and Word files recovered from iCloud. `work/` contains the executable code, `data/` contains the fixed split with orbit deduplication, `configs/` contains the configuration files, `runs/` contains the weights, CSV, images, and GIFs, and `reports/` contains the audit, aggregation, and verification records. The original Python 3 files were preserved exactly as they were.

```bash
cd /Users/xavier/Documents/dev/orbitlab
.venv/bin/python scripts/verify_artifacts.py
```

The training was performed in the local MPS. The DINO and world model, which is a selection extension, were not executed. The performance of large pre-trained models or generalization to real images are not claimed.
