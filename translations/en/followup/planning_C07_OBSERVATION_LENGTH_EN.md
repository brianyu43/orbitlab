# C07.4 Execution unit: Seeing the same moment and changing only the path of the past

2026-09-23. This is a follow-up pre-design, and data generation and learning have not yet been executed at the time of this document. This step is not replaced by the completion of C07 independent repetition or comparison of environmental information omissions.

## How to maintain the same prediction task

The initial state, environment, split, and action sizes of the existing independent data 640031·997101·997102 are reused, but the action time is shifted from t=3 to t=7 during action. The first 8 states are observations before action. In the simulator, a new rollout is generated, and the train creates 8 observations + 16 future observations, while val/test/OOD creates 8 observations + 61 future observations. The original action application state after t=4 is not directly copied.

The final state t=7 of each scene, the action, future answer, and train/val/test/OOD split are the same across all observation lengths. The input consists of the last L videos, with L values of 1/2, 4, and 8. Since changing the evaluation time point may alter collisions and position distributions, L=4 must also be learned from scratch in this new t=7 task. The four results from the original t=3 are not used with the L=4 value of this comparison.

This study is a separate observational research that matches the initial scenes of the existing data. It does not count the newly developed trajectories as having three additional independent data seed sets. Instead, it analyzes the differences in length of the same scene by matching them.

## Input contracts that distinguish information quantity and model capacity

Use an input template with 8 identical time positions for all Ls. Fill in the RGB values of the front portion that are not observed with 0, and indicate whether each time position is observable separately. The actual video, state, and measurement values obscured are not used in the model preprocessing. The method of calculating speed by using hidden frames in L=1 is prohibited.

In this design, the model structure, number of parameters, and initial weights for each L can be accurately matched. The CNN receives 8×RGB and 8 observation status channels, and the measurement-based MLP also receives 8 time positions and observation status as vectors of the same size. This eliminates the effect of the input layer expanding with the length of the observation. The difference in effective input number due to masking is a subject of study. Since it differs from the original C04 CNN structure and input, we do not interpret the absolute change in scores solely as the observation length effect.

## Comparison estimate and budget

- Each is trained separately with a correction MLP using direct RGB CNN and known one-color estimation. The range of using known renderers and physical equations is kept open.
- Data 3 × input length 4 × method 2 × initialization 3 = train 72 new estimators on 4,000 fixed steps, batch 32, with Adam 0.0005.
- Correctly match the initialization and train sample extraction order for each L. Do not select the number of training iterations or checkpoints based on the results of different Ls.
- The goal is the sum of the t=7 object state and the environment's resistance. Only the train correct answers are used as supervision signals. The RGB model does not provide the correct number of objects, ID, and masks.
- The non-learning measurement reference assumes that only position, radius, and color exist at L=1, while speed, resultant force, and resistance are treated as pre-declared baseline values. At L=2, speed is estimated using the final displacement, but the resultant force and resistance are not claimed to have been identified. The estimation equations for L≥3 and the insufficient degrees of freedom, decay, and contact processing rules are fixed to separate settings before viewing the results.
- Fix and connect the already verified variable_force motion predictor. Do not change the predictor based on the environment name. Calculate the state and environment comparison for the correct answer t=7 together to distinguish the error of the observer from the error of the motion predictor.

## Evaluation and completion evidence

1. Check whether the final state, action, and future answers are the same in all Ls, and whether the model input does not change even if the hidden frame is changed.
2. Play back the data trajectory and observation RGB and check for overlapping rotation trajectories during learning and retention. The correspondence between the original data and the intended initial scene is specified separately.
3. State and environment estimation, one-step component replacement, 1/4/8/16/32/61-step autonomous prediction, and comparison of identical past and opposite behavioral reactions with existing indicators. Includes failure color existence detection and numerical dispersion.
4. Preserve results regarding contact status between object count, non-learning properties, non-learning external forces, and observation. The contact status during observation is for scoring purposes and is not used as a correct filter for model input or reference estimators.
5. Match the differences across lengths between the same scenes and the same learning initialization, and aggregate three independent data seed sets separately. We do not assume that longer observations are always better.
6. Play back all new estimation and connection predictions and aggregations, and distinguish the speed ambiguity of a single video and the existing same past and different future configuration examples and actual errors.

The configuration file that fixes specific input preprocessing, loss weights, identification estimators, and learning code hashes serves as the benchmark for actual execution. Execution completion cannot be claimed solely based on this design document.
