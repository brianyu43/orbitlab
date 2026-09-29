# Observation length 1·2·4·8 comparison progress record

The initial scene of the three existing independent data sets was reused, and the action time was shifted to t=7. The final state, action, and future are predicted with equal observation lengths. The new independent data sets are not counted as three additional sets. This document is a output snapshot, and execution status is checked separately in the session.

It played 6,720 trajectories and 53,760 RGB frames. All RGB measurements, 53,760 length-based reference estimates, and aggregations were examined. Even if the masked input is changed, the calculation is the same, and a pre-check was performed to verify whether the model size and initial weights per length are the same.

72/72 new video estimation models have completed training, and 72/72 separate training, development prediction playback verification have been completed. The training is fixed at the final 4,000 steps, and the model is not selected based on intermediate scores.

| Data / Observation duration | One-step comparison completed / Verification | Long future·opposite action completed / Verification |
| --- | --- | --- |
| 640031 / 1 page | 156/156 · 156/156 | 1404/1404 · 1404/1404 |
| 640031 / 2 pages | 156/156 · 156/156 | 1404/1404 · 1404/1404 |
| 640031 / 4 pages | 156/156 · 156/156 | 1404/1404 · 1404/1404 |
| 640031 / 8 pages | 156/156 · 156/156 | 1404/1404 · 1404/1404 |
| 997101 / 1 page | 156/156 · 156/156 | 1404/1404 · 1404/1404 |
| 997101 / 2 pages | 156/156 · 156/156 | 1404/1404 · 1404/1404 |
| 997101 / 4 pages | 156/156 · 156/156 | 1404/1404 · 1404/1404 |
| 997101 / 8 pages | 156/156 · 156/156 | 1404/1404 · 1404/1404 |
| 997102 / 1 page | 156/156 · 156/156 | 1404/1404 · 819/1404 |
| 997102 / 2 pages | 156/156 · 0/156 | 1404/1404 · 0/1404 |
| 997102 / 4 pages | 156/156 · 0/156 | 1404/1404 · 0/1404 |
| 997102 / 8 pages | 156/156 · 0/156 | 1404/1404 · 0/1404 |

Step-by-step contrast converts perception and environmental inputs into the correct answers, dividing the causes, and compares the effects of opposite behaviors between the 1/4/8/16/32/61-step and long-term future. During observation, the contact category only uses the L−1 observed transitions, so the composition of participants may vary depending on the length. The main comparison of the length effect is between matching the same overall scene.

The performance of speed and environment estimated in a single video, as well as the benefits and limitations of long observation, are not confirmed before the overall evaluation, reproduction, and aggregation by length are complete. Currently, passing input and model verification does not mean achieving the accuracy of a world model.

Remaining tasks: model learning/reproduction completion, connection of all data and lengths, long-term and opposite behavior evaluation and reproduction, length-based pair comparison aggregation and report. Subsequently, integrate with the results of environmental information missing. External SVIB, prior research, real human judgment, and final research bundles also remain separately.

[Input Verification](../../../../../followup/reports/dynamics_observation_length_v1/input_verification.json) · [Learning Input Pre-flight](../../../../../followup/reports/dynamics_observation_length_v1/model_preflight.json) · [Execution Plan](../../planning_C07_OBSERVATION_LENGTH_EN.md) · [Execution Sessions](../../../../../followup/reports/dynamics_observation_length_v1/execution_sessions.json)
