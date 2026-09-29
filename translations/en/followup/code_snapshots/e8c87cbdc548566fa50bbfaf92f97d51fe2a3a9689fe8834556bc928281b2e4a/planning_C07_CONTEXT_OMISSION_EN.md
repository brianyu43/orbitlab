# C07.5 Execution unit: When the current state is unknown and only environment information is available

2026-09-23. Below is the preliminary implementation plan for the subsequent comparison. At the time of document preparation, the corresponding re-learning has not yet been implemented. The existing independent repetition, C07.4 observation length comparison, and SVIB do not reduce the range of the final research outputs.

## Questions and input

Currently, it provides the correct position, speed, radius, color, existence, and behavior of the object, and when it is masked by the combination or resistance of the environment, what can be predicted? It tests for a lack of environmental information by removing video recognition errors. Unlike the contrast that only changes the hidden value to 0 during inference, it masks it identically from the beginning of learning.

Maintain the 3D environmental input and the total model capacity. The combined hiding is set to zero for the first two coordinates, the resistance hiding is set to zero for the last coordinate, and all three coordinates are set to zero. The mask type is fixed in the execution settings and does not add environmental names or hidden truth values as input. The value 0 is a standardized input that prevents the model from observing the corresponding coordinate. This setting does not interpret it as converting the physical world to one where resistance or actual force is zero.

## Fixed comparison

- Independent data: existing 640031 and new 997101·997102. Maintain the train/val/test/OOD split within the same batch.
- Learning environment: only use variable_force train. Do not choose different weights for each evaluation environment.
- Model: Three types of general interaction, only state rotation, and state·external force joint rotation. These three use the same interaction structure to compare the relationship between symmetry and external force information. The original seven model comparison of C07 independent repetition is retained separately as is.
- Information conditions: Hide all information and hide the combined ability and hide both.
- Initialization: Weight initialization, sample extraction, and object order changes similar to the existing 0/1/2. hidden 48, 12,000 steps, batch 64, Adam 0.001, gradient clip 1.
- New learning: Data 3 × model 3 × initialization 3 × missing condition 3 = 81. The 27 total information comparisons reuse the variable_force checkpoint that has already been learned and validated and hash it.
- Evaluation: Provides the current state of the entire translation for all 13 hold groups. The first comparative indicators are the MAE of single-step speed per scene, and the auxiliary indicators are the location MAE and the error per layer for contact/non-contact/object-to-object/wall contact. Since different environments are evaluated using the same variable_force model, the C03 table with separate learning for each environment is not directly mixed with the table using separate learning for each environment.
- Preserves both learning loss, validation, and fixed-learning-end checkpoint from insufficient and sufficient contrastive learning. Does not change to a good seed or time point.

## Structural properties to be verified

When the observation sum is zero, only the state with the same internal weights is rotating; the rotating model and the co‑rotating model both compute the same mathematical function in the current implementation because the environment vector that rotates is zero. The consistency of the calculated, gradient, and matched learning results can be verified. This consistency does not mean that the hidden actual external force has disappeared or that the conditional distribution of the data is necessarily rotationally symmetric.

## If the answer is not determined in one way

Create a consistent current state and behavior without collisions or walls, and calculate the next state in two different environments using only hidden combinations or resistances. If the model receives the same input but the next answer is different, a deterministic predictor that only uses that input cannot accurately match both answers. The minimum squared loss, weighted equally for both cases, is proportional to the square of the difference between the two answers, and it is directly verified using the same normalization and object mask.

This configuration example does not extend the error limit to the Bayes error of the entire random evaluation data. It distinguishes this example from the existing video examples with the same past and different future scenarios, and from the example that only provides the current state. Even a model with sufficient information can be incorrect, and it does not explain its failure as a result of insufficient information.

## Completion criteria

1. Set the hash of the settings, original materials, reuse checkpoint, and execution code before learning.
2. Complete 81 new learning and 108 fixed checkpoint evaluations by condition.
3. Replay the learning sample, initialization, optimizer step and all hold-out predictions and per-row scores.
4. The data seed is aggregated in independent units by a repeating process that matches the initialization. It preserves results that do not have a hidden effect or appear in the opposite direction.
5. Separate and report the conclusions of the composition examples and actual random data. C07.4 Record the observation length and the overall completion of the subsequent goals separately.
