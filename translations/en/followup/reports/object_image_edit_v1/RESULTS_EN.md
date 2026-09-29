# Actual editing result after entering the picture and target point markers

2026-09-23. We completed the connection of the fixed model for B05 and the evaluation of the new verification data. We connected the existing encoder, supervised learning reader, and intervention model without any training. This does not mean that we have actually achieved the ability to stably edit objects.

## What the experiment asks

Can you read the object state in the image, change the color, direction, and position of the object pointed by a dot, and then redraw the image? The model only provided RGB values for the (x,y) coordinates of the dot over the visible object and explicit editing commands. It is not a natural language command. The correct answers—state, object count, mask, and ID—are not passed to the image model.

I created 2,961 tasks by applying color changes, 90-degree rotations, 5-pixel moves, and aesthetic color combinations that are possible after color changes and rotations, to 640 new scenes. Each scene has 128 images for the general/non-learning combination, three objects, four objects, and occlusion. There was no overlap between the existing 8,148 object scenes/editing rotation orbits and the new source family. Multiple commands in the same scene are not counted as independent samples.

The model leaves four slots with a prediction existence score out of five candidates, resulting in a maximum of four object capacities. The number of objects is not taken from the correct answer. The target is selected as the object closest to the point of the prediction center. The coordinate-based evaluation response before editing is retained even after editing. After coloring, rotation is performed sequentially in the same prediction slot without inserting the intermediate correct answer.

Each encoder uses one initialization, and the matching intervention model uses three initializations. In each method, we compared the identity/explicit transformation rules and the learned intervention models for each object. Separately, we evaluated the correct state+target and correct state+point selection comparison groups. The original encoder only learned RGB, but the state reader is a supervised learning model that uses the train correct attributes.

## Selection of the target and overall success

Below are three initialization means connected to the intervention model learned. The selected answer is the evaluation based on the coordinate transformation before editing. For full success, all target selection, target modification, remaining properties of the target, non‑target properties, and number of objects must be correct. A 1‑pixel tolerance on each coordinate axis and a 0.5‑pixel radius are allowed.

| Conditions | Select flat target | Select Slot target | Command successful based on flat estimated status | Command successful based on Slot estimated status | Successful for all actual answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two objects test | 93.5% | 92.2% | 100.0% | 99.2% | 0 observations for both methods |
| Unlearned shapes and colors | 95.1% | 83.9% | 99.7% | 99.2% | 0 observations for both methods |
| Three objects | 87.8% | 94.5% | 99.9% | 100.0% | 0 observations for both methods |
| Four objects | 66.1% | 84.1% | 100.0% | 100.0% | 0 observations for both methods |
| Hidden | 57.0% | 39.3% | 98.0% | 67.9% | 0 observations of both methods |

Command success based on the estimated state means that the model has recognized the shape and position it incorrectly as the starting state and has only performed the requested changes. It does not mean that it matches the actual answer. For example, if it already incorrectly read the shape of another object, even if it preserves that error, the preservation based on the estimated state will still succeed.

In all RGB methods, three initializations, three interventions, and five conditions, there were only 0 strict total successes. This is an observation of this sample and does not mean that the success probability of the population is exactly 0.

## If you give the correct state to the same command model

The learning intervention model that provided the correct state and the target ID for the correct answer achieved 100% overall success in all three initializations of the five new conditions. The explicit rule comparison group correctly matched both the state and the output image. In contrast, even with a correct state, the rule that selects the center closest to the target score only achieved 92.2% success in the masking condition. In this case, there are also limitations in the target selection rule beyond recognition.

Even applying explicit answer transformation rules to the estimated state results in zero total successes. Therefore, the error state that the current system perceives as only making the controller perfect cannot be resolved by this approach. It does not conclude that the encoder lacks information in the first place, nor does it conclude that other readers or learning methods fail.

## Matching target properties by command type

This is the test result for two objects that only scored the properties that need to change for the actual target, excluding the overall preservation conditions. Claiming success with editing based on only this value misses the error.

| Command | flat | Slot Attention |
| --- | ---: | ---: |
| Color change | 93.5% | 92.2% |
| 90 degrees rotation | 24.5% | 22.7% |
| Movement | 15.6% | 21.6% |
| Rotation after color change | 24.0% | 22.4% |
| Unlearned color combination | 93.2% | 94.0% |

## Error in output picture

I saved 68,103 outputs as actual prediction states and played them all back. Using evaluation response, I did not match the drawing's front-back order to the correct answer. The prediction slot order is the order in which the drawings are drawn, and depth was not learned separately. Coordinate and size rounding and limits were applied only when drawing the drawings, and the state score was calculated as the original real number.

| Conditions | Flat view error | Slot view error | Reference showing the original picture as it is |
| --- | ---: | ---: | ---: |
| Two objects test | 0.244 | 0.236 | 0.168 |
| Unlearned shape and color | 0.250 | 0.246 | 0.174 |
| Three objects | 0.297 | 0.235 | 0.119 |
| Your object | 0.342 | 0.276 | 0.089 |
| Hide | 0.315 | 0.351 | 0.178 |

Setting RGB values to 0 to 1, the average absolute error was calculated from the intersection of the input and correct preview areas. The current editing output is significantly larger in error than that of a reference that returns the input unchanged on average. This serves as a contrast to the fact that the input return does not execute commands, making it an editing tool that cannot be considered useful solely due to high preservation.

## Covering and error analysis

The results for object count by category and command type, as well as the visible proportion of the target, were left in four sections in analysis.json. Additional failure classifications are displayed in the order of first failure: No detection → Target selection → Change property → Target remaining property → Non-target property → Number of instances. This order is a reporting classification and does not represent the independent causal effect of each cause.

Even in occlusion, all targets were required to have at least one visible hard pixel. Therefore, it is not the restoration performance of completely occluded targets. The geometry-only handling of overlapping objects and the lack of depth learning are also limitations of interpretation. It was executed from a single new data generation seed, and the rules of the learning material and the renderer are the same synthetic world.

## Verification and next step

I played back 640 data files, 2,961 edits, and 2,560 global rotations. I re-verified 1,280 RGB encoders, 3,840 reading predictions, 68,103 edits/outputs under 115 comparison conditions, and 17,388 scene-specific indicators and segments. The fixed code before evaluation on the reader and controller remained unchanged.

The connection and verification evaluation for B05 has been completed, and the performance limits are recorded as they are. The full failure analysis for B07 is concluded by linking the existing separation errors with this intervention error. The independent SVIB external evaluation and video-based movement and long-term prediction are also continued.

[Overall summary](../../../../../followup/reports/object_image_edit_v1/summary.json) · [Initialization/Scene and Failure Analysis](../../../../../followup/reports/object_image_edit_v1/analysis.json) · [Data verification](../../../../../followup/reports/object_image_edit_v1/data_verification.json) · [Prediction/Output verification](../../../../../followup/reports/object_image_edit_v1/verification.json) · [Comparison chart](../../../../../followup/reports/object_image_edit_v1/capability_curves.png)
