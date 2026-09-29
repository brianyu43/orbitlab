# OrbitLab Generation Success Recovery Experiment

The original project and `followup/` results are preserved; new experiments, data, and reports are kept separately in this folder. **The final confirmation run is complete.** Generation success improved substantially, but the prespecified size-diversity criterion for successful images failed, so the final research milestone was not met.

Suggested reading order:

1. [Final results and limitations](confirmation_v1/FINAL_REPORT_EN.md) · [summary figure](confirmation_v1/figures/confirmation_overview.png)
2. [P0 diagnosis](reports/P0_FINDINGS_EN.md)
3. [P1 condition and decoder ablations](reports/P1_FINDINGS_EN.md)
4. [P2 direct-pixel baseline and P3 training-duration experiment](reports/P2_P3_FINDINGS_EN.md)
5. [Locked final configuration](confirmation_v1/lock.json) · [execution fix for a provenance field](confirmation_v1/execution_amendment_01.json) · [raw results](confirmation_v1/evaluation/summary.json) · [recomputation check](confirmation_v1/evaluation/verification.json)

The headline numbers are paired strict-pass rates across three new synthetic-data seeds × three initializations. The mean rate for seen combinations rose from 22.01% to 40.20% when flow training increased from 4,000 to 16,000 steps; for excluded combinations it rose from 16.02% to 32.70%. Diversity across all generated images passed, but the size distributions among strictly successful images exceeded the threshold for all three seeds, and shape 2/color 0 had fewer than the minimum successful samples per condition. We make no claim about human judgments or generalization to real-world data.

Recorded checks are available for [P1](evaluation/p1_v1/verification.json), [P3](evaluation/p3_steps16k_v1/verification.json), [the pre-confirmation training audit](confirmation_v1/pre_evaluation_training_audit.json), and [the final confirmation](confirmation_v1/evaluation/verification.json). The work used four CPU threads, without an independent package installation or external compute resources.
