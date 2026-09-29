# The intervention experiment that gave the correct object state

2026-09-23. Completed 6 learning sessions and prediction playback verification. This experiment is a problem where you learn the rules to apply when you already know the object's properties and location. It does not include the ability to recognize objects in images.

## What did you compare?

Both models were given the same input: maximum four objects' shape, color, coordinates, size, direction, and existence status, the display of the object to be changed, and commands for color/rotation/translation. One model combines the status of the four objects into a single vector for processing (flat), while the other model applies the same neural network to each object (object). Both models predict the change in the properties of all objects. They did not include a correct rule that unconditionally preserves non-target objects into the learning model.

Each model has exactly 34,991 learning parameters. They used the same task dataset, same random shuffling, same optimizer and loss, and 6,000 steps. Since the structures are different, the initial weights are not the same. They compared three initializations, and there is only one data generation seed. They do not claim that they ran on a CPU and adjusted the calculation time to be the same.

There are two objects in the learning scene. By mixing all four positions, including the empty space, each time, we ensured that an object and a target command appeared sufficiently in each position. Color changes, 90-degree rotations, and movements were each learned, but pairs that rotated after a color change were excluded from the learning process. Combination evaluation applies the same model twice consecutively and does not include a middle correct state.

## Results

Below is the percentage that satisfies **target change + preservation of the remaining properties of the target + preservation of non-target objects + preservation of empty spaces**. Categorical properties required exact matching, coordinates were within 1 pixel on each axis, and radii were within 0.5 pixels. The average was calculated for each scene, and the average was then averaged over 3 training runs.

| Evaluation condition | Process the entire scene as a vector | Shared model per object |
| --- | ---: | ---: |
| Two objects, general test | 99.54% | 99.93% |
| Includes shapes and color combinations excluding two objects from learning | 98.83% | 100.00% |
| Three objects | 10.74% | 100.00% |
| Four objects | 0.78% | 100.00% |
| Provides the correct state of two hidden objects | 100.00% | 100.00% |

The aesthetic manipulation synthesis of the two objects was 100% in both models. Therefore, the difference between the two methods is not claimed to exist in the manipulation synthesis itself. The difference mainly occurred under the condition of increasing the number of objects. In the three objects, the success rate of changing the target of flat is 86.8%, but the preservation of non-target is 29.4%, so if only the target is scored, the problem is missed.

This overall success rate is sensitive to the tolerance. When the coordinate/radius tolerance is increased from 1/0.5 pixel to 2/1 pixel in the post-sensitivity test, the success rates of the three objects in the flat configuration increase from 10.7% to 52.5%, and in the 4/2 pixel configuration, they rise to 91.6%. The average absolute error of the target coordinates for the three objects is 0.84 pixels for the flat configuration and 0.05 pixels for the object configuration, while the changes in the non-target coordinates are 0.60 pixels and 0.04 pixels, respectively. Therefore, the results must be interpreted in light of the cumulative effects of small position and size errors. No fixed primary criteria were changed. [Tolerance Sensitivity](../../../../../followup/reports/object_oracle_study_v1/tolerance_sensitivity.json)

Since we provided all correct states even under the occlusion condition, it does not mean that 100% of those conditions correspond to the object that is obscured being identified as an image. The shared model per object does not receive other objects as inputs, so it is the expected property in this design that is less affected by changes in the number of objects. It does not mean that the superiority of the task is due to the interaction between objects, such as collisions. It is used to confirm the implementation of simple independent interventions and as a baseline for the perception stage.

The overall success of the identity baseline that does not change anything was 0%, while the analytic baseline that followed explicit answer rules was 100%. The latter is not a learning result.

## Verification and next step

All 1,792 basic scenes and 4,820 original intervention pairs were rendered by the renderer. The task set that added commands for object count and occlusion conditions contains 6,356 tasks. The correct transformation was fully verified through separate formula implementations, and 26,272 evaluation predictions for 6 models and 2 reference lines were rendered. 2,304 scene-level metrics were aggregated, and the same learning samples and position exposure were passed, along with model hash checks. Consistency in the model positions for each object was also verified.

The next step is a model that only takes RGB inputs to divide and represent objects. The correct mask is used only for evaluation and not for training inputs or loss calculations. If objects are incorrectly divided or properties are incorrectly read, the high performance of this experiment cannot be transferred to the overall system's capabilities.

- [Summary](../../../../../followup/reports/object_oracle_study_v1/summary.json), [Verification record](../../../../../followup/reports/object_oracle_study_v1/verification.json), [Run-by table](../../../../../followup/reports/object_oracle_study_v1/runs.csv)
- [Examples of three objects in flat](../../../../../followup/runs/object_oracle_study_v1/flat_s0/count3_examples.png)
- [Example of the same conditions for models per object](../../../../../followup/runs/object_oracle_study_v1/object_s0/count3_examples.png)
- The examples are the first 12 tasks of each condition. Each set is in the order of input, answer, and prediction, and they are not selected randomly for good appearance.
