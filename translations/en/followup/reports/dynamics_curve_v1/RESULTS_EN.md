# Learning curve and physical baseline of exercise models

2026-09-23. Trained 9 models from the same initializations for 1,000/6,000/12,000 steps. We confirmed that the 1,000-step weights exactly matched the uniform sample pilot, and we did not initialize the optimizer or random number in the middle. This is not an additional 9 independent initializations, but a development experiment that extended the same learning path as the existing 3 environments × 3 method.

## Verified curve

The values are the average one-step speed of MAE per validation scene, measured in pixels/frames per unit. Lower values are better. It provides the correct state at every moment, so it is not a continuous future prediction result.

| Environment | Model | 1,000 steps | 6,000 steps | 12,000 steps |
| --- | --- | ---: | ---: | ---: |
| No external force | Normal interaction | 0.1246 | 0.0726 | 0.0565 |
| No external force | Only rotates in state | 0.1276 | 0.0433 | 0.0379 |
| No external force | Environment also rotates | 0.1276 | 0.0433 | 0.0379 |
| Gravitational constant | General interaction | 0.1082 | 0.0781 | 0.0734 |
| Fixed gravity | Only state rotates | 0.1421 | 0.0699 | 0.0708 |
| Fixed gravity | Environment also rotates | 0.1372 | 0.0601 | 0.0644 |
| Changing external force | General interaction | 0.1295 | 0.0740 | 0.0850 |
| Changing external force | Only state rotates | 0.1522 | 0.0687 | 0.0719 |
| Changing external force | The environment also rotates | 0.1442 | 0.0519 | 0.0517 |

The amount of learning actually had an effect. The constant speed baseline under no external force was 0.0543, the fixed gravity was 0.0859, and the changing external force was 0.0909. At 12,000 steps, several models exceeded this baseline. On the other hand, even if the learning sample error continues to decrease, the validation error increased from 6,000→12,000 under some conditions. There is no evidence that all conditions improve just by increasing the learning time.

## A stronger simple baseline

While known cumulative, decay, and wall reflections are calculated, a reference line has been implemented that **excludes collisions between objects**. It is a non-academic reference that only knows the time intervals and boundary rules of the same simulator, and it is not a model that discovered physics from the video. It is distinguished from a reference line that is provided with the correct answer input information and even algorithm knowledge.

This method almost accurately calculates the moments of contact-free moments and moments with wall reflection only. The total speed error of the changing external force was 0.0152, which was smaller than the learning model above. If only we look at the transition with collisions between objects, it increased to 1.247. Therefore, the learning model should not only show the overall average but also show how well it predicts the actual interaction. [Full values of the physical baseline](../../../../../followup/reports/dynamics_physical_baselines_v1/summary.json)

## Verification and progress status

The validation predictions for 27 checkpoints were re-evaluated, resulting in 110,592 results and 324 totals. Adam step, random state, initial weights, learning samples, and predefined learning error samples were also tested. The physical baseline was compared to 24,576 independent reference points, where the simulator was run for each object individually, and confirmed wall/external force/damping consistency and object-to-object collision omissions. [Learning curve verification](../../../../../followup/reports/dynamics_curve_v1/verification.json) · [Physical baseline verification](../../../../../followup/reports/dynamics_physical_baselines_v1/verification.json)

The next state comparison has been fixed to 7 methods × 3 environments × 3 initializations, with a common 12,000 steps. The model does not select the best validation checkpoint for each model. The validated 9 seed-0 models are reused, and the remaining 54 are newly trained. The test involves evaluating combinations of unobserved properties, increased object count, and new external force conditions, and it is still before the entire process is completed. [Fixed settings](../../../../../followup/configs/dynamics_state_study_v1.json)
