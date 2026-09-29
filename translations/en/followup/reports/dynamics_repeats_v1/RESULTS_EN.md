# Exercise prediction rechecked from 3 independent data sets

The design that rotates both the state and the external forces reduced the speed error in a single step compared to the general interaction model for all three data sets. The reduction range was 16.8~41.1%. The error of the physical reference that directly calculates the known force and the wall was smaller for all three data sets.

Data that outperformed the non-learning physical equations in video measurement learning correction were 2/3 of the general conditions and 1/3 of the non-learning external forces. The superiority observed in one data could not be generalized into a general conclusion.

The long future was not stable. The general conditions of the joint rotating predictor 61-step position error in video measurement+learning correction were 14.51707, 3.863e+06, and 9.909e+21 pixels per data. The value includes the large dispersion that remains finite. Therefore, it does not claim that the gain of one step led to the completion of the long-term world model's accuracy.

We compared the existing data 640031 with the new data 997101·997102. In the new dataset, we trained 126 state prediction models and 12 video estimation models from scratch. We retained the existing initialization order, sample extraction sequence, number of training iterations, model list, and evaluation rules. This report is the independent repeated part of C07, and the experiment with missing observation lengths and environmental information is separate.

## Comparison units and verification

Each table's one column represents a single independent data generation seed. Each cell is the mean of the fixed three learning initializations, and the initializations were counted as nine independent data samples, not nine. Non-learning references are calculated only once per dataset. The full CSV contains the mean for each dataset, the standard deviation between initializations, and the standard deviation, minimum, and maximum values across the three data sets. Only three datasets are used to claim population significance or generalization of the actual images.

We played back 4,480 new trajectories and 17,920 video clips, and confirmed that there is no overlap between the existing materials and the rotating orbits. We verified 126 new state models, 312 video connection comparison conditions, and 2,808 conditions for long-term future and counter-action. The basic transfer of the long-term future is 16,865,280, and the response transfer is 6,746,112, which are calculated repeatedly for the same scene and model, not independent scenes. We also confirmed 327,680 one-step transfer of the new physical reference using separate reference codes.

Upon examining the very small difference sign in the aggregation, a sign reversal was detected. The prediction and aggregation CSV were preserved, and a precise sum of positive integers was used for separate sign verification. The number of such extremely small negative numbers is not interpreted as a practical effect or significance. [Numerical Audit Record](../../../../../followup/reports/dynamics_repeats_v1/across_data/aggregate_numeric_audit_note.json)

## One-step prediction that provides the correct answer status every moment

The unit is the average absolute speed error (px/frame), and the lower it is, the better. It is the C03 condition using all hold transitions.

| Condition / Model | 640031 | 997101 | 997102 |
| --- | ---: | ---: | ---: |
| General / General interaction | 0.06982 | 0.06657 | 0.06226 |
| General / Status·External force joint rotation | 0.04112 | 0.05539 | 0.04807 |
| General / Known external force/wall physics reference | 0.01286 | 0.01460 | 0.01415 |
| Your object / General interaction | 0.89629 | 0.80063 | 0.86345 |
| Four objects / state·external force co-rotation | 0.18207 | 0.25187 | 0.18520 |
| Your object / Reference to known external force·wall physics | 0.04190 | 0.04123 | 0.04542 |
| Non-learning external force / general interaction | 0.12923 | 0.11379 | 0.11005 |
| Non-learned external force / State·External force joint rotation | 0.06404 | 0.07574 | 0.06598 |
| Unlearned external force / Refer to known external force·wall physics | 0.02388 | 0.02363 | 0.02321 |

Under general conditions, the error reduction rates for co-rotation were 41.1%, 16.8%, and 22.8% per dataset. Out of the 3/3 datasets where known external forces and wall references had smaller errors than co-rotation, 3 were used. This reference does not account for simulator integration and boundary conditions, and it omits collisions between objects. It does not interpret the learned physical laws or general model superiority.

## Read from the four videos and predict the next moment

The same common rotating prediction engine transmitted the estimated state and environment together. Since it is a single speed error (t=3→4), it is not compared directly with the overall transition average across the entire range due to direct performance improvement. Measurements of the circular object and the physical equations use known color, shape, and integration rules.

| Conditions / How to enter video | 640031 | 997101 | 997102 |
| --- | ---: | ---: | ---: |
| General / Direct CNN | 0.88334 | 0.87787 | 0.85333 |
| General / Video measurement + learning correction | 0.06379 | 0.09647 | 0.08716 |
| General / Video measurement+Physical | 0.05293 | 0.21997 | 0.09896 |
| Four objects / Direct CNN | 0.92765 | 0.92993 | 0.87158 |
| Four objects / video measurement + learning correction | 0.18857 | 0.26334 | 0.25552 |
| Four objects / video measurement+physical formula | 0.22465 | 0.41864 | 0.49327 |
| Non-learning external force / Direct CNN | 0.85781 | 0.91753 | 0.87769 |
| Non-learning external force / Video measurement + learning correction | 0.14209 | 0.14052 | 0.12288 |
| Non-learning external force / Video measurement+physical formula | 0.06648 | 0.32376 | 0.10464 |

The results of attaching a measuring device for a circular object are not called performance because they learn concepts and physics on their own using only RGB. Whether the learning correction is better than physical methods must be judged by both conditions and the values per data set. The correct input state/environment change and the results of all comparables and all prediction models were preserved in observation_metrics.csv.

## A long future predicted by ourselves

This is the position mean absolute error (px) of the joint rotation predictor for video measurement + learning correction. Since the values are finite, the prediction mean for each initialization was first calculated. The large finite variance was not well suppressed, and the analogous failure was not converted to an 0 error. The finite sample sizes per initialization were left in autonomous_inputs.csv.

| Condition / Prediction length | 640031 | 997101 | 997102 |
| --- | ---: | ---: | ---: |
| General / 1st stage | 0.08046 | 0.08274 | 0.15985 |
| General / 16 steps | 2.75296 | 3.97278 | 3.51065 |
| General / 61st stage | 14.51707 | 3.863e+06 | 9.909e+21 |
| Your object / 1st stage | 0.23841 | 0.20716 | 0.38255 |
| Four objects / 16 steps | 1005.43227 | 2.516e+09 | 4.879e+09 |
| Four objects / 61st stage | 1.561e+25 | 2.771e+34 | 1.022e+33 |
| Non-learning external force / 1st stage | 0.14855 | 0.11018 | 0.24313 |
| Non-learning external force / 16 stages | 5.18685 | 5.89037 | 4.11042 |
| Non-learning external force / 61st stage | 16.37562 | 4.077e+07 | 3.893e+23 |

| Conditions / 61st stage diagnosis | 640031 | 997101 | 997102 |
| --- | ---: | ---: | ---: |
| General / Limited path ratio | 1.00000 | 1.00000 | 1.00000 |
| General / Wall invasion 1px over or failure rate of the number | 0.05208 | 0.14583 | 0.28646 |
| General / Overall status success rate | 0.00000 | 0.00000 | 0.00000 |
| The limited path ratio of the four objects/values | 1.00000 | 0.94792 | 0.77604 |
| Over 1px object/wall penetration or failure rate of measurement | 0.59896 | 0.80729 | 0.58333 |
| Your object / Success rate of the whole state | 0.00000 | 0.00000 | 0.00000 |
| Non-learned external force / limited path ratio of values | 1.00000 | 1.00000 | 1.00000 |
| Out of skill force / Wall penetration exceeding 1px or failure rate of the number | 0.05729 | 0.12240 | 0.32552 |
| Non-learning external force / Overall status success rate | 0.00000 | 0.00000 | 0.00000 |

Even if large values remain finite, they are not accurate predictions. Since the average is sensitive to some large deviations, we presented both the finite ratio and the overall state success/wall penetration indicators together. The initial answer information is also included in the entire CSV file, as are the values of the remaining eight prediction models. We do not infer long-term stability from a single step improvement.

## The reaction when you change your behavior

General conditions, video measurement + learning correction, joint rotation predictor waiting. The table shows the value obtained by subtracting the response location error from the “even if you change your behavior, the future remains unchanged” non‑response comparison error. If the value is negative, it is better than this comparison; if it is positive, it is worse. Accuracy of behavior transfer to other objects due to object collisions is considered separately.

| Target / Prediction length | 640031 | 997101 | 997102 |
| --- | ---: | ---: | ---: |
| Target action / 16 steps | -9.16574 | -8.23902 | -8.60613 |
| Other object / 16th stage | 0.71417 | 0.88537 | 0.72710 |
| Target action / 61st stage | 3.93086 | 2.115e+08 | 2.595e+22 |
| Other object / 61st stage | 2.37151 | 3.188e+08 | 2.388e+22 |

The response error is conditional in cases where both predictions are limited, and the no-response comparison is calculated for all response scenes. The simple mean difference under conditional conditions with analogous cases is not a paired comparison of the same sample, so superiority cannot be claimed solely based on this difference. The two original errors and the number of limited samples can be verified in the entire CSV. The accuracy of responses per object and per time is not extended beyond natural language understanding or general causal inference abilities.

## Remaining research stages

The effects and failures confirmed in the independent repetition are preserved, and a comparison of observation lengths at the common behavior time points of 1/2/4/8 is conducted, along with a re-learning process to recover and recover missing information on convergence and resistance. The cases where answers cannot be determined from the same observation and the cases where the model failed to learn are distinguished. External SVIB evaluations, comparisons with nearby prior research, real-person judgments, and final drawings, research notes, and replication bundles are also remaining.

[State prediction overall](../../../../../followup/reports/dynamics_repeats_v1/across_data/state_metrics.csv) · [Video comparison overall](../../../../../followup/reports/dynamics_repeats_v1/across_data/observation_metrics.csv) · [Long-term future/behavior change overall](../../../../../followup/reports/dynamics_repeats_v1/across_data/autonomous_metrics.csv) · [State aggregation verification](../../../../../followup/reports/dynamics_repeats_v1/across_data/state_verification.json) · [Video aggregation verification](../../../../../followup/reports/dynamics_repeats_v1/across_data/observation_verification.json) · [Long-term future aggregation verification](../../../../../followup/reports/dynamics_repeats_v1/across_data/autonomous_verification.json)
