# Environmental information missing comparison progress record

Current state and actions are accurately provided, and only the combined or resisted inputs are used to re-learn from scratch. The information at the learning and evaluation stages is identical, and this is not a comparison where only the input of the full information model is deleted post-hoc. This document is a snapshot of the saved output, and whether it is actually running is verified separately during the work session.

| Work | Document creation time |
| --- | --- |
| Learning new missing condition | 81/81 completed |
| Comparison of all information evaluation | 27/27 completed |
| Full conditions learning·evaluation | 108/108 completed |
| Separate playback verification | 108/108 completed |
| Replayed transition prediction | 9,815,040; not independent sample size |
| Combined hidden rotation model learning comparison | 18/18 pairs check, among them 18 pairs with completely the same weight |
| Same input, different answer configuration | Completed three cases and free movement equation·square loss decomposition test |

We compare three independent data sets with three initializations and three interaction/rotation models. The training data only use variable_force and do not select different models depending on the evaluation environment. Therefore, it is a comparison with the table of C03 that was learned separately for each environment. The 27 total information comparisons reuse the existing weights, but the 81 missing conditions are learned anew.

If the combined force is removed, the observation force vector that causes rotation in the state alone, in the rotating and co-rotating structures, is zero. When the same weight is applied, the pre-test that the function and gradient are identical was passed. This does not mean that there is no hidden actual force or that the evaluation distribution is rotationally symmetric.

In the three configuration cases, the inputs for the current state, behavior, and observation environment were the same, but the next answer differed. The minimum squared loss was examined by weighting these two cases equally. This value is not claimed to be the lower bound of the error for the entire random evaluation data set.

Training, replay, aggregation by data seed, and numerical report checks for the missing-environment-information comparison are complete. The [results report](RESULTS_EN.md) explains effects and limits by condition. Experiments on observation lengths 1/2/4/8 at a common time point, external evaluation, human judgment, and final research outputs were still outstanding at this progress snapshot.

[Detailed execution plan](../../planning_C07_CONTEXT_OMISSION_EN.md) · [Input/calculation inspection](../../../../../followup/reports/dynamics_context_omission_v1/preflight.json) · [Information deficiency example](identifiability/EXPLANATION_EN.md) · [Learning progress](../../../../../followup/reports/dynamics_context_omission_v1/progress.json) · [Replication progress](../../../../../followup/reports/dynamics_context_omission_v1/verification_progress.json)
