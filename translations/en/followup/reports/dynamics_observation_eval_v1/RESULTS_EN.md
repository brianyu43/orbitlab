# The result of estimating the status and environment from four past videos

2026-09-23. We completed video inference for C04 and a step-by-step connection comparison. The result of predicting the entire future independently is not yet available.

## Input and learning

Four RGB images of t=0,1,2,3 were input, and the final object state and constant power and resistance were estimated. The behavior is applied to the first movement after t=3, so there is no behavior effect in the four input scenes yet. The commands are provided as the target color and speed change amount. Since the objects in the world are of different colors, this information can be verified in the video. The model does not provide hidden ID, number of correct answers, state, or environment type.

Each RGB direct CNN and shared MLP that corrects video measurements were trained with three initializations. Only four initial images were used from the 768 trajectories in the train of three environments, and the batch size was set to 32, the learning rate to 0.0005, and the fixed number of steps to 4,000. The supervisory signal is the sum of the t=3 state in the train and the sum of the resistance. The CNN was trained on MPS with 1,112,103 parameters, and the correction MLP was trained on CPU with 31,369 parameters. The comparison is not that the parameters, time, and prior knowledge are set the same.

Video measurement aligns the center and size using the known six colors, black background, circular shape, and radial alpha rules. Even when estimating sum and resistance with four displacement vectors, the simulator uses the eight substep integration rules. This strong non-learning reference is not called a model that discovers object concepts or physical laws on its own. MLP uses this reference as input and initial estimation.

The RGB direct CNN had a coordinate error of about 0.64 pixels during training, but about 3.80 pixels during validation. A significant generalization gap was observed in the current data, model, and training budget. It does not extend beyond the impossibility of the image model in general.

## Estimating the status and environment of the withheld material

This is a general test with 128 different external force environments. CNN/MLP uses three initialization means, and the physical-based estimation is identical without learning initialization. Position and velocity are conditional absolute errors with respect to the detected correct color slot. The presence of object color was also separately indicated.

| Method | All object color existence detection match | Position error (px) | Speed error (px/frame) | Total error (px/frame²) | Resistance error |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB Direct CNN | 96.1% | 3.70707 | 0.8781 | 0.02245 | 0.00418 |
| Video measurement + learning correction | 100.0% | 0.03913 | 0.0505 | 0.00239 | 0.00068 |
| Video measurement + physical formula | 100.0% | 0.00022 | 0.0207 | 0.00912 | 0.00274 |
| Correct answer location + physical formula (privilege comparison) | 100.0% | 0.00000 | 0.0203 | 0.00890 | 0.00249 |

The measurement's position error, which is known from the rule of drawing a circle, is very small. However, if there is a collision during observation, the assumption of matching the external force according to free motion is broken. In this environment's test, the physical speed error for 119 non-contact trajectories was about 0.00035, and for 9 trajectories with past contact, it was about 0.2893 pixels/frame. Even if the exact position is provided, the error in the contact interval is almost the same, so in this case, it is not only a problem of pixel position measurement.

Learning correction reduced the accuracy error of the general test, but it could not improve both position and speed compared to non-learning reference. Performance from external forces that were not present in learning should also be considered separately.

## What influences one-step prediction

The general interaction and state of the variable_force learned previously were fixed, and the joint rotation model of rotation and state with external force was established. Regardless of the environment being evaluated, the same weights are used, so the model is not selected by the actual environment name. The comparison of constant speed and known external force/wall reflection was also made. The latter omits collisions between objects.

Below is the speed error of the joint rotation model. In four comparisons that swapped the correct state and the correct environment, only the initial information was changed. The behavior is directed toward the target color predicted by the model, and if the target is missed, the failure is also recorded.

| State/Environment provided method | RGB CNN | Measurement + learning correction | Measurement + physical formula |
| --- | ---: | ---: | ---: |
| Correct answer status + Correct answer environment | 0.0246 | 0.0246 | 0.0246 |
| Estimated state + Correct answer environment | 0.8823 | 0.0638 | 0.0395 |
| Correct answer status + estimated environment | 0.0371 | 0.0266 | 0.0475 |
| Estimated state + Estimated environment | 0.8833 | 0.0638 | 0.0529 |

In current RGB CNNs, the impact of reading location and speed errors was much greater than the error caused by environmental differences. However, we do not interpret the difference in indicators as an independent or additive causal effect. Errors in components can interact.

This value was evaluated only for t=3→4. The average trajectory transition of the entire C03 hold-out set is 0.0411, and the distribution at the evaluation time differs, so the 0.0246 difference in the correct state comparison is not interpreted as an improvement in the new model's performance. The model weights are the same.

## Number of objects and non-learning external forces

This is the speed error of a joint rotation predictor that uses both the estimated state and the estimated environment. The missing correct color slot was filled with the 0 state and calculated, and the detection and count indicators are present together with the material.

| Condition | RGB CNN | Measurement + Learning Correction | Measurement + Physical Equation | Correct state + Correct environment |
| --- | ---: | ---: | ---: | ---: |
| General test | 0.8833 | 0.0638 | 0.0529 | 0.0246 |
| Unlearned size·color | 0.8454 | 0.0914 | 0.0822 | 0.0594 |
| Three objects | 0.8951 | 0.1770 | 0.2085 | 0.1481 |
| Your object | 0.9277 | 0.1886 | 0.2247 | 0.1349 |
| Non-learning external force | 0.8578 | 0.1421 | 0.0665 | 0.0421 |

Learning correction was worse than physical-based reference in non-learning external forces. The correction gains under general conditions cannot be directly transferred to non-learning environments. The RGB CNN's detection accuracy for all color presence in the four objects dropped to 7.3%. The circle video measurement was also well detected under this condition, but even in comparison to the correct state, future prediction errors remained, indicating that the number of motion models must be distinguished from generalization issues.

## The limits of information that can be retained even if the video is perfect

In a trajectory at a constant speed where the sum of a=gamma*v is constant, different resistances and the sum of a can create the same past. After the same action, the future changes. In the actual renderer, four past images were completely composed of two identical environments and tested. This is a specific boundary case and does not claim the identifiability of the entire set of general evaluation data. [Explanation and numbers](../observation_identifiability_v1/EXPLANATION_EN.md)

## Verification and next step

- Input contract: 2,240 trajectories·RGB 8,960 images and confirmed the same past at the action point, 512 counterfactuals.
- Non-learning reference: All RGBs were re-run and 2,240 estimated and 6,720 indicator rows, 3,384 aggregates/intervals were checked.
- Learning model: 5,760 train/validation predictions of 6 checkpoints were replayed on the CPU and train-specific input, sample, and optimizer steps were checked.
- Hold-up connection comparison: 15,360 estimated rows under 156 conditions and 307,200 step-ahead predictions, and 93,456 aggregated/grouped ones were re-verified. Since non-learning estimation and answer comparison were repeated, this number does not imply that there are as many independent learning or scenes as this.
- The evaluation target is the original trajectory of 13 hold groups totaling 1,280, and one data generation seed. We do not interpret the three learning initialization and trajectory bootstrap segments as population significance.
- In the following C05/C06, only the t=3 estimated state and environment are entered once, and a maximum of 61 steps of prediction are performed, while evaluating the non-target object reaction based on opposite behavior in the same past. Subsequently, in C07, the evaluation of independent repetition and the boundary of observation length and condition omission is conducted.

[Overall results](../../../../../followup/reports/dynamics_observation_eval_v1/summary.json) · [Replay verification](../../../../../followup/reports/dynamics_observation_eval_v1/verification.json) · [Component comparison diagram](../../../../../followup/reports/dynamics_observation_eval_v1/state_context_substitution.png) · [Learning development results](../../../../../followup/reports/dynamics_observation_models_v1/development_summary.json)
