# Results of reading object state in RGB representation

2026-09-23. We trained the initializations of the status readers for two fixed RGB encoders. We completed the training of a total of 6 readers and 6,912 train/validation prediction playback verification. The results of performing actual commands on the image are not yet available.

## Method and evaluation range

We fixed the 12,000-step encoder of the existing flat/Slot Attention and input five slots extracted from RGB. The shared MLP reader predicts the shape, color, coordinates, size, direction, and existence of each slot. Each reader uses 22,800 learnable parameters, Adam 0.001, batch 128, and 6,000 steps. We supervised with train ground‑truth properties and normalized only in the train phase. The image encoder is initialized per method, and the three initializations are repeated for each reader.

It does not provide the number of correct objects during evaluation. The objects are selected based on the prediction existence probability, the distance between the prediction-centered and correct-centered points is calculated, and then the attributes are scored. Missing objects are considered attribute errors. Coordinates must be within 1 pixel of each axis, and the size must be within 0.5 pixels of the radius. All attributes and the number of objects in a single image must match for the entire success.

## validation results

This is the average of the initial readings of the same 128 plates. It is not interpreted as the result of new independent verification data.

| Item | flat | Slot Attention |
| --- | ---: | ---: |
| Object matching | 82.0% | 65.1% |
| Shape matching | 34.6% | 40.6% |
| Color matching | 91.5% | 88.8% |
| Direction matching | 26.3% | 22.4% |
| Coordinates are within 1 pixel of each axis | 13.0% | 25.7% |
| All properties of a single object match | 0.78% | 1.95% |
| All objects, properties, and counts in the scene match | 0/128, all three initializations | 0/128, all three initializations |

The train shape accuracy of flat is 99.6%, but the validation is 34.6%. A large generalization gap was observed in the current training settings. Based on only these detector results, we cannot definitively conclude that the encoder does not contain any information corresponding to this. Furthermore, interpreting a success rate of 0 for observation as an exact probability of success in the population does not make sense.

## Direction evaluation modification and verification

The first evaluation code compared the direction error vectors to ensure they were exactly the same values. There was a slight rounding error in the sin/cos components of the renderer, causing errors even when the direction was 90 degrees. We preserved the existing models, learning, prediction, and first evaluation records, and added a separate evaluator that classifies directions into four categories. The table above only uses the modified evaluations.

Yes, we examined the actual direction and incorrect direction, and the verification tool reconfirmed the direction scores with a separate atan2 calculation. We examined all 20 candidate options for the prediction replay, train-specific normalization and sampling, optimizer step, initial weights, and minimum cost allocation of the 6 models, as well as the aggregation of modified indicators. The performance has not changed due to model retraining.

## Next step and argument boundaries

The existing intervention model that provided the correct answer status showed high success rates, but the step to obtain that status in RGB is still weak. The next step is to connect the target point marking and the intervention model to the prediction state of the fixed reader, separately evaluating target selection, target change, and non-target preservation. There are remaining verification data, changes in object count, masking, and manipulation synthesis evaluation. B05 is underway.

- [Modified statistics](../../../../../followup/reports/object_readout_pose_corrected_v1/summary.json)
- [Independent playback verification](../../../../../followup/reports/object_readout_pose_corrected_v1/verification.json)
- [Execution of overall image intervention](../../planning_B05_EXECUTION_EN.md)
