# Independent data repeat progress record

2026-09-23. The number below is based on the storage output at the time of document creation, and whether it is in progress is verified separately for the actual work session. The entire C07 and subsequent goals are underway.

Completed the learning of independent data repetition, C03–C06 evaluation, and aggregation verification and comparison reports. [Result Report](RESULTS_EN.md). The observation length experiment and environmental information missingness comparison are separate.

## Conditions to maintain the comparison

The original data seed is 640031, and new seeds 997101 and 997102 are added. The order of three model initialization and sampling of training samples is the same as the original. Since the design crosses the data seed and initialization, it is not counted as nine independent data sets. The original learning weights are not transferred and are learned from scratch in the new training set.

State prediction consists of three sets of 126 total: three environments × seven models × three initializations × new data. It maintained the same 12,000 steps·batch 64·Adam 0.001·hidden 48·random object order as before. Video estimation consists of three initializations for each CNN and measurement+correction MLP, with two data sets, totaling 12. It maintained the 4,000 steps·batch 32·Adam 0.0005 and the scope of use of known color, physics rules.

## Actual completion and remaining work

| Item | Completed evidence | Remaining range |
| --- | --- | --- |
| New World Materials | 4,480 trajectories, 222,080 states, 17,920 RGB images played | Distinguish from the model evaluation status below |
| Behavior change·symmetry | Opposite behavior 1,024, rotation 1,824, object sequence change 608 inspection | Distinguish from the long-term behavior evaluation status below |
| Data independence | 6,720 initial physical/video rotation orbits combined with existing and new ones are separated without overlap | It does not mean generalizing to a wider real-world environment |
| Video input and non-learning reference | RGB measurement 17,920 images, separation of correct/inference files, verification of reference line indicators | Completion of connection comparison of independent repetition |
| New video estimator | 12/12 learning and validation, train/validation prediction 11,520 CPU playback | Independent repetition test/OOD and long-term evaluation completed |
| New state predictor | 126/126 learning/holding evaluation completed, 126 of them played verified | Integrated comparison of independent repetitions completed |
| Observation length·information missing | Example of separate detailed planning and identification limit configuration | 1/2/4/8 frames are not executed; information missing refer to separate progress records |

Separate records are kept for the evaluation of the connection between the train/validation development verification and the hold-out condition. When the data changes, the validation error does not be reported as the final conclusion for improving test performance or independent verification. The completed hold-out predictions of the state model are intermediate results before the comparison of all data seeds is complete. Only the good initializations are selected and stopped, or the original baseline is not lowered.

Currently, `dynamics_repeat_eval.py` connects the C04 component comparison between new weights and new inputs, and the C05/C06 autonomous prediction. It confirmed that the input path and decoding match when compared to the development predictions, which have saved 2,304 new validation inferences. The overall connection evaluation is only executed after verifying the state models of 63 data bundles and video models from the corresponding validation receipts. The completion status of the connection evaluation and separate playback is as follows, and the aggregation status across data bundles was recorded at the top of the document.

Video input comparison for data 997101: 156/156 conditions saved, separate playback verification completed. Long-term/behavior changes for data 997101: 1404/1404 conditions saved, separate playback verification completed. Video input comparison for data 997102: 156/156 conditions saved, separate playback verification completed. Long-term/behavior changes for data 997102: 1404/1404 conditions saved, separate playback verification completed.

[Progressing environmental information omission comparison](../dynamics_context_omission_v1/PROGRESS_EN.md) · [Detailed execution plan for observation length](../../planning_C07_OBSERVATION_LENGTH_EN.md)

[Data verification](../../../../../followup/reports/dynamics_repeats_v1/data_verification.json) · [Video input verification](../../../../../followup/reports/dynamics_repeats_v1/observation_data_verification.json) · [Video model verification](../../../../../followup/reports/dynamics_repeats_v1/observation_development_verification.json) · [Number of state learning counts](../../../../../followup/reports/dynamics_repeats_v1/state_progress.json) · [State playback verification](../../../../../followup/reports/dynamics_repeats_v1/state_verification_progress.json) · [Complete C07 execution plan](../../planning_C07_EXECUTION_EN.md)

The entire range of the SVIB external evaluation, pre-research comparison, actual person judgment, final research memo, drawing, and replication bundle is also maintained.
