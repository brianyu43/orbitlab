# Exercise prediction when environmental information is hidden

In a model that rotates both state and external forces together, masking the combined (gravity + wind) effect and retraining it, the one-step speed error in the general test increased by 51.3%, 32.8%, and 30.8% per data point. The error also worsened in both non-learned external forces. This observation indicates that environmental direction information contributes to the one-step advantage of this model.

The effect of masking only resistance varied depending on the data and evaluation conditions. Furthermore, models that did not rotate the external force direction together were actually better at hiding the combined effect of non-learned external forces. The performance of finite learning models did not always improve just because they had more information.

## What did you keep the same?

Input the exact current object state and behavior. It is not a video recognition or long-term prediction score. It was re-learned from scratch in each of the three conditions that cover both convergence and resistance, and the same masking was applied to both learning and evaluation. The 27 total information conditions reused the already verified variable_force weights. We completed the evaluation of 81 new learning instances and 108 total conditions.

All models were trained only on variable_force data and were applied directly to other environments. Each model followed 12,000 steps, initial weights, sample selection, and object order changes. Each table's row is a single independent data generation seed, and the numbers are the means of three fixed initializations. The independent data consists of three sets, not nine sets.

We played back 9,815,040 transfer predictions for 108 conditions, 784,224 rows per scene and per contact, and the aggregated data. Since the same scene was repeated across multiple conditions for evaluation, the number of independent samples differs. All environmental, contact categories, and train/validation diagnoses were preserved in the CSV, and the table below shows the hold-out conditions for variable_force.

## Co-rotating model: The effect of environmental information

Speed average absolute error (px/frame), the lower it is, the better. It is a one-step evaluation that provides the correct state again every moment, and differs from autonomous prediction that accumulates errors over time.

| Evaluation conditions / Provided information | 640031 | 997101 | 997102 |
| --- | ---: | ---: | ---: |
| General test / All information | 0.04112 | 0.05539 | 0.04807 |
| General test / Hide the score | 0.06221 | 0.07358 | 0.06286 |
| General test / Resistance hiding | 0.04393 | 0.05371 | 0.04921 |
| General test / Both hidden | 0.06143 | 0.06846 | 0.06220 |
| Your object / whole information | 0.18207 | 0.25187 | 0.18520 |
| Your object / combined hidden | 0.22251 | 0.25202 | 0.19315 |
| Your object / Resistance hiding | 0.21829 | 0.21668 | 0.26449 |
| Four objects / Both hidden | 0.20827 | 0.24113 | 0.21725 |
| Non-learning external force / All information | 0.06404 | 0.07574 | 0.06598 |
| Non-learning external force / Hidden combined ability | 0.09986 | 0.10464 | 0.09233 |
| Non-learned external force / Resistance concealment | 0.06894 | 0.07285 | 0.06950 |
| Unlearned external force / Both hidden | 0.09860 | 0.09935 | 0.09341 |

When the combined effort was hidden, the error of the unlearned external force increased by 55.9%, 38.2%, and 39.9% per data set. The change when resistance was hidden was inconsistent. The learning resistance range, fixed budget, and results are limited to that structure, and this does not mean that resistance information is generally unnecessary.

## In the incorrectly applied rotation restrictions, there are other phenomena

The two models below use the same rotational average structure. The model that rotates only under a state condition fixes the external force vector, while the coupled rotation model rotates both the external force and the state together. They were compared under non-learning external force conditions.

| Model / Provided information | 640031 | 997101 | 997102 |
| --- | ---: | ---: | ---: |
| Only state rotation / Full information | 0.11330 | 0.11651 | 0.11179 |
| Only status rotates / Compulsory hidden | 0.09986 | 0.10464 | 0.09233 |
| Co-rotation / Full information | 0.06404 | 0.07574 | 0.06598 |
| Co-rotation / Co-operation hiding | 0.09986 | 0.10464 | 0.09233 |

When the combined force is applied, the input external force vector becomes zero, so the function of the two rotational structures becomes identical. In the 18 corresponding learning pairs between combined force concealment and overall environment concealment, the weights were completely the same, and this equivalence was also confirmed in prediction comparison. At this point, the superiority of the two structures is not interpreted as separate experimental effects. It does not mean that the concealed actual external force is zero or that the evaluation distribution itself is rotational symmetric.  The phenomenon of information being erased in a structure that only rotates in

and scores improving shows that the structure may be using the provided directional information incorrectly or not generalizing it appropriately. It does not extend to conclusions that the value of information has become negative in the optimal predictor or to a proof of the cause for all models.

## Cases where you can't solve because you don't have information and cases where you failed to learn

In the three pairs separately constructed, even if the current state, behavior, and visible environmental inputs were the same, the next answer differed depending on the hidden environment. A model that gives the same answer to the same input cannot achieve both answers. The minimum squared error for each pair was tested using closed‑form physical equations and numerical methods.

These three pairs are a concrete example of ambiguity caused by information deficiency. They do not indicate how frequently such cases occur in random evaluation data or calculate the optimal error limit for the entire dataset. Even in the current state, traces of past environments may remain. Therefore, the entire average error difference cannot be explained as indeterminability.

## Next connection

The learning, evaluation, and aggregation of environmental information missing-comparison has been completed. By linking the currently ongoing observation lengths 1/2/4/8 experiments, we verify which information is actually recovered by looking at the past more closely. The C07 entire set, SVIB external evaluation, comparison with prior research, actual human judgment, and final research bundle are still not completed.

[Overall statistics](../../../../../followup/reports/dynamics_context_omission_v1/across_data/metrics.csv) · [Difference in the same initialization](../../../../../followup/reports/dynamics_context_omission_v1/across_data/paired.csv) · [Aggregation verification](../../../../../followup/reports/dynamics_context_omission_v1/across_data/verification.json) · [Same input, different answer examples](identifiability/EXPLANATION_EN.md)
