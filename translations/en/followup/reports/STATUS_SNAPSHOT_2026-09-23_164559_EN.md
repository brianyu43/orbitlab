# OrbitLab progress status — 2026-09-23 16:45 KST

This is the status at the time of checking the actual execution sessions 52909·55594 and the saved progress records. The overall goal is in progress.

- 31 detailed tasks: completed 26, in progress 3, waiting 1, actual person judgment waiting 1. The proportion is not interpreted by the remaining time proportion because the size of the task is different.
- Comparison of observation lengths 1·2·4·8: 72 models trained and learning results playback verification completed. 3 sets of data × 4 types of length evaluation predictions generated completed.
- Last playback verification queue: completed step 17/20. A separate verification step 4 was also completed before this queue. Currently, 997102 data, length 4, C05_C06 verification is in progress.
- Subsequent aggregation pipeline: waiting_for_active_verification_queue. It is currently running to perform aggregation, re-aggregation, table and diagram generation, and numerical verification in sequence after verification is complete. The final observation length conclusion has not yet been confirmed.
- 24 models of the external SVIB public example comparison and verification completed. C4 is better than the general model, but the overall pixel error was larger than the comparison of directly reproducing the original by the average of all four alphas. It is not a full official benchmark replication.
- Completed comparison with 9 nearby prior research. Does not claim that the originality of the new structure has been proven.
- The original distribution material of 1,324 files, 201,125,244 bytes, matched both the initial hash and size.

Current verified abilities: The strict automatic success rate for single shape generation is 15.6% for this combination and 11.4% for the new combination. The entire scene editing starting from RGB had a strict success rate of 0 in the verified sample. The speed error within a movement stage was lower in both independent data sets than the general interaction model in the structure that aligned symmetry, but higher than the reference using known physical equations, and a long future was projected. Since it is a score for different tasks, it does not combine with the accuracy of a single model.

The remaining tasks include collecting observation lengths, summarizing the overall conclusions, final figures, short research notes, easy explanations, replication bundles, and a comprehensive review. 240 person evaluations are prepared, but actual person responses are not yet available.

[Current capability and basis](CURRENT_CAPABILITY_EN.md) · [Observation length execution status](../../../../followup/reports/dynamics_observation_length_v1/execution_sessions.json) · [SVIB results](svib_preview_shape_swap_v1/RESULTS_EN.md)
