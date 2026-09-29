# One-step exercise prediction in correct state: Complete this comparison

2026-09-23. Evaluated 63 learning models for 3 environments × 7 methods × 3 initialization. Reused the 9 previously verified checkpoints and trained 54 new ones. This comparison is C03, and C04, which detects the state from video, and C05, which predicts the continuous future, are separate stages.

## Common input and comparison conditions

It provided the actual object state, behavior, known total power and resistance coefficients at every moment. Two objects were transitioned uniformly along the train trajectory, using batch 64, Adam 0.001, and a common 12,000 steps. We did not select the best test among several checkpoints. We matched the raw initial weights of the symmetric contrast group with the sample flow. The number of parameters and actual calculation time are different, and this is not a comparison that aligns the time. There is one data generation seed, and three initializations.

The model for each object does not see other objects. The interaction model combines the messages of other objects. The constraint that only rotates the state rotates the state while fixing the directional external force. The joint constraint of state and external force also changes the external force when the coordinate system is changed. In an environment without directional external force, the two constraints become the same.

## General test

It is the average absolute error in speed, and the unit is pixels/frames. The lower it is, the better. First, calculate the average within the trajectory, and the learning model displayed three initialization averages.

| Method | Directionless environment | Fixed gravity | Variable external force |
| --- | ---: | ---: | ---: |
| Entire scene MLP | 0.1195 | 0.1141 | 0.1114 |
| Independent object MLP | 0.0749 | 0.0926 | 0.0818 |
| Object interaction | 0.0580 | 0.0846 | 0.0698 |
| Status·External force rotation enhancement | 0.0476 | 0.0667 | 0.0583 |
| Restriction that only rotates the state | 0.0406 | 0.0699 | 0.0631 |
| Status·External force rotating together restriction | 0.0406 | 0.0571 | 0.0411 |
| State rotation restriction and general model combination | 0.0407 | 0.0591 | 0.0560 |
| Reference to speed of sound | 0.0533 | 0.0873 | 0.0865 |
| Refer to known external force/wall reflection | 0.0130 | 0.0145 | 0.0129 |

## Changing external force: Restraining conditions and object number

| Method | General test | Combination of non-learned properties | Three objects | Four objects | Non-learned external force |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entire scene MLP | 0.1114 | 0.1171 | 0.1599 | 0.1865 | 0.1476 |
| Independent object MLP | 0.0818 | 0.0874 | 0.1049 | 0.1144 | 0.1342 |
| Object interaction | 0.0698 | 0.0769 | 0.3800 | 0.8963 | 0.1292 |
| Status·External force rotation enhancement | 0.0583 | 0.0624 | 0.2281 | 0.4170 | 0.0804 |
| Constraint that only rotates the state | 0.0631 | 0.0702 | 0.1462 | 0.2376 | 0.1133 |
| State·External force together rotating constraint | 0.0411 | 0.0475 | 0.1162 | 0.1821 | 0.0640 |
| State rotation restrictions and general model combination | 0.0560 | 0.0634 | 0.1824 | 0.3181 | 0.1192 |
| Reference to acceleration | 0.0865 | 0.0908 | 0.1078 | 0.1179 | 0.1406 |
| Refer to known external force/wall reflection | 0.0129 | 0.0158 | 0.0317 | 0.0419 | 0.0239 |

## What was improved and what was left

The common test for variable external forces and the joint rotation constraint in non-learned external forces showed that both initializations were less error-prone than the general interaction, rotation enhancement, state-only rotation, and hybrid models. However, in the four objects, neither initialization was better than the model that only rotates the state. It does not extend the effect under specific conditions to all generalized capabilities.

The increase in the number of objects has significantly increased the error in several interaction models. The structure of summing without normalizing the message by the number of messages is a candidate cause, but because a comparative experiment was not conducted only by changing the aggregation method, it cannot be confirmed as the cause. The independent object model is a structure that excludes interaction, so it may be less sensitive to changes in the number of objects, but it does not mean that a collision model is sufficient.

The average includes many moments without collisions. The speed error of the variable external force test, which separates the collision moments, is as follows.

| Method | Non-contact | Contact between objects | Contact only with the wall |
| --- | ---: | ---: | ---: |
| Object interaction | 0.0406 | 1.1305 | 0.3076 |
| Status·External force rotating together constraint | 0.0219 | 1.1184 | 0.1306 |
| Reference to speed of sound | 0.0320 | 1.1413 | 0.7066 |
| Refer to known external force/wall reflection | 0.0000 | 1.1287 | 0.0000 |

The external force and wall reflection reference uses the known integration rules and boundaries of the simulator directly and ignores the contact between objects. It is not a model that learns the physical laws through learning. Since this strong non‑learned reference is more accurate on the overall average, the learning model does not claim that it predicts this world overall better.

## Verification and uncertainty

- 1,720,320 transition predictions for 63 models were played back. optimizer·sample·initialization·reuse hash and 2,730 metric aggregation tests passed.
- The physical reference was used to recalculate 200,704 individual spherical objects using the original simulator, and 12,022 indicator rows and 260 aggregates/regions were examined.
- In the total 63 data sources in CSV, 910 and 52 paired comparisons were separately examined for the average, initialization standard deviation, and bootstrap intervals of 455 conditions.
- Separated the initial three values and sample standard deviation from the scene resampling interval. The interval is conditional on these three models, and it is not an independent data seed repetition or a population significance test.
- It is a step-by-step prediction as long as the correct state returns every moment. We did not verify the performance of long rollout, visual recognition, environmental inference, and counterfactual prediction here.

## Next step

Using the same past observation length, position, velocity, and environment are estimated from the video, and error accumulation is measured through continuous prediction that does not provide the correct state again. The possibility of object failure and external forces are separately evaluated, as well as the response of non-target objects to behavioral changes.

[Aggregate all conditions](../../../../../followup/reports/dynamics_state_study_v1/aggregate.json) · [Material comparison table](../../../../../followup/reports/dynamics_state_study_v1/comparison_rows.csv) · [Model replay verification](../../../../../followup/reports/dynamics_state_study_v1/verification.json) · [Aggregate audit](../../../../../followup/reports/dynamics_state_study_v1/aggregate_verification.json) · [Physical reference verification](../../../../../followup/reports/dynamics_state_baselines_v1/verification.json)
