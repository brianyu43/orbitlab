# A long future without giving the correct answer and behavior change

2026-09-23. Completed the fixed model evaluation and reproduction of C05/C06. A momentary prediction accuracy did not guarantee the stability of a long future or the accuracy of the behavior effects on other objects. Achieving a stable world model does not mean that.

## Execution range

Provided the state and environment obtained from the initial four frames (t=0..3) once and predicted them autonomously up to 61 steps (t=64). No correct position, speed, count, or environment were supplied in the middle. The actions were applied only to the first transition and the remaining actions were zero. The correct initial state and environment are the explicit diagnostic reference group. CNN, video measurement + learning correction, and video measurement + physical reasoning used the fixed estimates of C04.

It retained all seven existing learning methods, three initializations, constant speed, and external force/wall reflection references. It used the same variable_force weights for all environments. The learning algorithm learned by a single step loss, and the learning trajectory ends at t=19. The current 32/61 step does not exceed the time range of the learning data, but it is not a verification experiment where the data generation seed is independently changed.

The original withheld trajectory used 1,280 instances, of which 512 were identical past/opposite action responses. This is the 1,404 condition for the 4 inputs × 3 initialization × 13 groups × 9 predictors. The number of prediction instances repeated on the same data is not counted as an independent sample size. The method for measuring the circle uses known color, shape, and integration rules, and the physical reference excludes collisions between objects.

## How much wrong will you be if it takes a long time?

The joint rotation (joint_exact) model's position mean absolute error is the general test of 128 trajectories of varying external forces. The unit is pixels, and the smaller it is, the better. After calculating the finite prediction mean for each seed, the three values are averaged. Missing correct objects are filled with 0 coordinates and included in the error, and the error for the existence judgment and the detected objects are also separately provided in the data.

| Initial Input | 1st Stage | 16th Stage | 61st Stage | 61st Stage Limited Ratio |
| --- | ---: | ---: | ---: | ---: |
| Correct answer initial state·environment | 0.022 | 2.363 | 14.206 | 100.0% |
| RGB Direct CNN | 4.187 | 11.655 | 16.845 | 100.0% |
| Video measurement + learning correction | 0.080 | 2.753 | 14.517 | 100.0% |
| Video measurement+physical | 0.052 | 2.813 | 1.24e+05 | 100.0% |

Even if the correct answer information is provided, errors accumulate, so the problem cannot be solved using only the delayed model. Direct video CNN has large errors from the start. Conversely, the fact that the values are finite does not mean the prediction is accurate or physically plausible. In specific initializations of video measurement + physical equations, very large finite errors were also observed.

The following are the nine full prediction models for video measurement+learning correction inputs. Large values were not hidden or cut out. A finite proportion was shown together to ensure that the error, excluding failed cases, does not appear overly good.

| Predictor | 16-step position error | 61-step position error | 61-step finite ratio |
| --- | ---: | ---: | ---: |
| flat | 5.744 | 16.421 | 100.0% |
| object | 4.723 | 18.996 | 100.0% |
| interaction | 3.837 | 2.49e+12 | 100.0% |
| augmented | 3.719 | 18.851 | 100.0% |
| wrong_exact | 4.727 | 19.511 | 100.0% |
| joint_exact | 2.753 | 14.517 | 100.0% |
| relaxed | 3.686 | 16.922 | 100.0% |
| inertial | 7.794 | 52.884 | 100.0% |
| force_wall | 1.508 | 8.455 | 100.0% |

13 blocked groups · Basic predictions of all methods. Among 138,240 paths, 750 paths became limited in their values. This denominator is the number of repeated comparison paths and not the number of independent scenes. After the initial failure, it retained NaN and a failure indicator. Even if the limited values are very large, you cannot assess stability using only the dispersion ratio.

## Long prediction under new conditions

This is the result of video measurement + learning correction input, a joint rotation model. Without mixing the three conditions, each aesthetic properties, object number, and external force were evaluated separately.

| Conditions | 16-stage location error | 61-stage location error | Wall penetration exceeding 1px or failure to reach 61-stage level |
| --- | ---: | ---: | ---: |
| General | 2.753 | 14.517 | 5.2% |
| Unlearned size·color | 3.115 | 15.060 | 11.7% |
| Three objects | 5.667 | 2.77e+17 | 17.7% |
| Your object | 1005.432 | 1.56e+25 | 59.9% |
| Non-learning external force | 5.187 | 16.376 | 5.7% |

The color and existence are retained as they were in the initial estimation. Therefore, the success of object tracking is not measured by the retention of the color slot. The position of the predicted same-color object is measured separately to see if it is actually closest to that object, and this correspondence is for scoring purposes. The prediction is not reordered by the correct position sequence.

## How reliable is the collision point

The model does not directly output collision events. It calculated contact candidates based on close proximity between two frames and speed changes outside of free movement. The speed residual of 0.1 pixels/frame and distance tolerance of 1.5 pixels were fixed before execution. It does not count as a collision success if there is no speed response simply because an object overlaps or passes through.

The correct answer applies the same judgment to 78,080 trajectory transfers and compared it to actual simulator events. Below is the performance of this judgment tool itself, not the performance of the learning model.

| Contact type | TP | FP | FN | Precision | Reproducibility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Between objects | 2065 | 251 | 25 | 89.2% | 98.8% |
| Wall | 6696 | 277 | 3 | 96.0% | 100.0% |

The candidate assessment is not perfect. Below is the evaluation of the initial contact point, including its limitations, as presented. It is a general test, video measurement + learning correction input, and a 61-step interval. The timing error is conditional in cases where both sides have events and the values are limited, and it must be considered together with the event omission and success within one frame.

| Predictor | Contact | First-point error (conditional frame) | Within 1 frame of the correct event | Missing correct event or numerical failure |
| --- | --- | ---: | ---: | ---: |
| interaction | pair | 10.591 | 8.2% | 51.1% |
| interaction | wall | 4.081 | 53.4% | 0.3% |
| joint_exact | pair | 12.287 | 5.0% | 53.9% |
| joint_exact | wall | 1.607 | 70.8% | 0.0% |
| force_wall | pair | 8.589 | 7.8% | 58.4% |
| force_wall | wall | 0.865 | 86.2% | 0.0% |

Even force_wall, which does not calculate collisions between objects, can generate pair candidates. This is because if another object is present near the instant the speed changes due to wall reflection, it can satisfy the pair candidate rule together. It is not interpreted as evidence of object interactions learned from this. The reason for separately evaluating non-target reactions to behavioral changes is also here.

## If you change your behavior, does the reaction of other objects also match?

Only the sign of the change in speed given to the target was reversed in the same initial estimate and environment. It predicted the difference between two futures and compared it with the actual difference between the two futures. Since object collisions transmit the influence of behavior to other objects, unconditionally preserving the non-ideal is not the correct answer. The 512 response data for this evaluation are from two object scenes.

The following is the location response error in the variable external force test for video measurement + learning correction input. The “no response” comparison assumes that even if behavior changes, the predicted future remains the same. The lower it is, the better.

| Predictor | Stage | Target response error | Target non-response comparison | Non-target response error | Non-target non-response comparison |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint_exact | 16 | 4.190 | 13.356 | 2.224 | 1.510 |
| joint_exact | 61 | 19.175 | 15.244 | 13.515 | 11.143 |
| force_wall | 16 | 2.413 | 13.356 | 1.510 | 1.510 |
| force_wall | 61 | 12.007 | 15.244 | 11.143 | 11.143 |

Currently, the joint rotation model predicts the target's own behavioral response better than the passive control in the 16th stage, but the response transmitted to the non-target was worse than this control. In the 61st stage, the superiority in the target's response was not maintained either. The external force and wall references omit interactions between objects, so the non-target response is exactly 0. The non-target response error in this reference is identical to the passive control, which aligns with the implementation intention.

## Verification and remaining range

Stored 8,432,640 basic predictions and 3,373,056 transfer and counter-action predictions. Replayed from the initial input. The main location, speed, count, spatial response, overall success, and action response errors were calculated using separate code, while the remaining metrics were replayed using fixed scoring code. Checked color-based action transmission from input, unused count answers, continuous numerical failures, data, weights, and code hashes.

The overall values are in `all_metrics.csv`, and the scene-level bootstrap intervals are in `aggregate.json`. We grouped the three initializations of the same scene together and preserved the denominator of the finite-sample error. This is the uncertainty within a single data generation seed and is not evidence of independent data replication or population significance.

The evaluation work for C05/C06 has been completed, but it does not claim that long-term prediction ability has been achieved. For C07, there are still gaps in the independent repetition, observation length, condition missingness, SVIB external evaluation, completion of preliminary research, actual human judgment, and final replication bundle.

[All metrics](../../../../../followup/reports/dynamics_autonomous_v1/all_metrics.csv) · [Aggregation and intervals](../../../../../followup/reports/dynamics_autonomous_v1/aggregate.json) · [Prediction replay verification](../../../../../followup/reports/dynamics_autonomous_v1/verification.json) · [Aggregation audit](../../../../../followup/reports/dynamics_autonomous_v1/aggregate_verification.json) · [Long-horizon curves](../../../../../followup/reports/dynamics_autonomous_v1/long_horizon_curves.png) · [Number failure](../../../../../followup/reports/dynamics_autonomous_v1/stability_overview.png) · [Behavior response](../../../../../followup/reports/dynamics_autonomous_v1/counterfactual_response.png) · [Fixed first case video](../../../../../followup/reports/dynamics_autonomous_v1/autonomous_examples.gif)
