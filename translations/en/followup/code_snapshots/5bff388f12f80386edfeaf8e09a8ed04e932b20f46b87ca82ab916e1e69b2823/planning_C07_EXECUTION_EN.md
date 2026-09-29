# C07: The boundary between independent repetition and observation and environmental information

2026-09-23. This is a follow-up execution after checking the C05/C06 results. Only examples of the same past and different future configurations have already been executed. The independent repetition, observation length, and condition missing learning below have not yet been executed. Only the successful models are selected and the verification scope is not reduced.

| Order | Execution unit | Completion evidence |
| --- | --- | --- |
| C07.1 | Independent data repetition setting fixed | 2 new generated seeds that do not overlap with current data, hash of existing settings, learning budget, input contract, model list, main comparison and failure criteria |
| C07.2 | Repeat new data and existing model protocol | Repeat new train/val/test/OOD/count data, trajectory replay, rotation/partition duplicate check, same budget learning of existing three environments×seven methods×three initializations, video estimator repeat |
| C07.3 | Reconfirmation of a step-by-step·long future·behavior change | Scoring such as C03–C06, results by data seed and results by initialization separation, aggregation including numerical failures, results by object count·property combination·external force·direction effect |
| C07.4 | Comparison of observation lengths 1/2/4/8 | Only the past observation quantities were changed in separate data that aligned the time point before action, while the final state, future, action, and split remained the same, with explicit indication of the same budget learning and input/model capacity differences per length |
| C07.5 | Compare missing environment information | Give the true current state and compare net force hidden, drag hidden, both hidden, and full information. Training and evaluation must receive the same information; zeroing a trained model’s input afterward is not a fair retraining control. |
| C07.6 | Separate identifiability from learning failure | Recheck existing examples of different environments yielding the same observation. Distinguish cases with no unique answer because information is missing from cases where a model fails despite sufficient information. |
| C07.7 | Replay, aggregation, research conclusions | Report all new prediction replay, pre-processing and data exposure tests for train-only, uncertainty at the data seed level, confirmed effects and unconfirmed effects |

In the observation length experiment, a design is used that shifts the common action time point to t=7 and provides only the last L frames. Therefore, L=4 is also a distribution different from t=3 of the original C04, and L=4 comparisons at the same new reference time point are learned together. The existing data are not called the past because future frames are added to them. The model's convolution input channel number, whether known original color, physics rules are used, and the speed estimation limits of a single image are disclosed.

Independent repetition distinguishes model exploration that increases success rates from separate analysis. First, we verify whether the dispersion, long-term error, and failure of non-target responses for C05/C06 persist even in new data. The detailed setting of the observation length and condition missing experiment is fixed in a separate configuration file before viewing the respective data and results. The composition counterexample of observation deficiency is not expanded into the error limit of the entire random data set.

SVIB external evaluation (B06), close-up prior research comparison (D01), full core picture (D02), final research memo (D03), replication bundle (D04), real human judgment (A04) remain unchanged.
