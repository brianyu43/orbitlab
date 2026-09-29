# OrbitLab

OrbitLab is a local experiment comparing C4 rotation-aware representations, reconstruction, and conditional latent flow. The original 15-session plan was completed on 2026-09-23.

**At matched training time, the B1 rotation-augmentation AE reconstructed images better; the B2 C4 model followed shape-and-color generation conditions more often. Generated images still showed blur and shape errors.**

- [Results: design, measurements, failures, and limitations](translations/en/RESULTS_EN.md)
- [Evidence for completing the 15 sessions](translations/en/reports/SESSION_COMPLETION_EN.md)
- [Reproduction commands and replay of saved results](translations/en/REPRODUCE.md)
- [Six-minute presentation](deliverables/OrbitLab_6min.pptx) · [reviewed English speaker script](deliverables/TALK_6MIN_EN.md)
- [Originals, code, results, and weights archive](deliverables/OrbitLab_15sessions_2026-09-23.zip) · [file-hash manifest](RELEASE_MANIFEST.json)
- [Full execution plan](translations/en/EXECUTION_PLAN_EN.md) · [original-code audit](translations/en/reports/CODE_AUDIT_EN.md) · [C4 mathematics](translations/en/reports/C4_MATH_EN.md)
- [Follow-up research and historical material in English](translations/en/TRANSLATION_NOTES.md)
- [Generation success recovery: reviewed English summary](generation_recovery/README_EN.md)

The work included 29 AE comparison runs, six additional AE runs to qualify generation, and eight flow runs: 43 training runs in all. Separate pilots, calibration, and diagnostic checks are outside that count. “15 sessions” refers to the work plan, not a claim that each session consumed exactly 90 minutes.

| Mean over three seeds | B1 augmented AE | B2 C4 AE |
| --- | ---: | ---: |
| Matched-time test foreground MAE ↓ | 0.0446 | 0.0727 |
| 64-step generation: seen shape-and-color match ↑ | 34.0% | 57.0% |
| 64-step generation: held-out combination match ↑ | 29.5% | 54.5% |

Generation accuracy here is the judgment of a template evaluator checked on clean rendered images. The report discusses possible errors on blurry generations and shows worst cases. The matched-time AE comparison and generation comparison are separate experiments.

`sources/originals/` preserves the ZIP and Word files recovered from iCloud. `work/` holds executable code; `data/` the fixed, rotation-orbit-deduplicated splits; `configs/` run settings; `runs/` checkpoints, CSVs, images, and GIFs; and `reports/` audits, aggregate results, and verification records. The three original Python files were preserved unchanged.

```bash
cd /Users/xavier/Documents/dev/orbitlab
.venv/bin/python scripts/verify_artifacts.py
```

Training ran locally on MPS. The optional DINO and world-model extensions were not run. No result is claimed for large pretrained models or real-image generalization.
