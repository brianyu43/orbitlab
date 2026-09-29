# Failure of object representation: from separation to actual intervention

2026-09-23. We compiled examples of object count, concealment, manipulation synthesis, and error in the scope of B07. This is a failure analysis of the compared fixed model and the synthesis world, and it does not extend beyond the performance limitations of general Slot Attention.

## Background and object separation

In the two object tests of 12k Slot Attention, the native mask IoU was 0.086, and the mask that left an RGB background was 0.628. The fact that the background occupied a large portion of the prediction area was related to the reduction in native score. The evidence that the post-processing learned object concepts by exploiting the data characteristic of a black background is not.

The four IoUs of the same fixed post-processing object were 0.742 and 0.427 in the occlusion scene. In the occlusion scene, the simple connection area merged objects in 97.7% of the scenes. The Slot+foreground area had a merging rate of 28.1%, but the percentage of object fragments was 70.7%. Just having less merging cannot be considered a good separation. We retained the separation indicators, restoration indicators, and error types together.

The material for this result is in [separation evaluation](object_layers_heldout_v1/RESULTS_EN.md), [aggregation, fragmentation, and shading extent aggregation](../../../../followup/reports/object_failure_audit_v1/summary.json), and [re-verification](../../../../followup/reports/object_failure_audit_v1/verification.json).

## Does separation lead to a state expression that can be edited?

In the actual editing of the new verification scene, when the number of objects increased from two to four, the target selection changed from flat 93.5% to 66.1% for Slot 92.2% to 84.1%. The consistency of the four object counts for Slot was 81.3%, but editing that matched all properties and the entire scene had zero observations in both methods. It is important to distinguish between the ability to count objects or select candidates near the target, and the ability to accurately represent the properties of those objects.

The inputs for this comparison are new RGB data, and the reader is a fixed MLP trained in supervised learning in the correct state. The image samples in the previous mask evaluation are not image samples, so we do not interpret the results as causal relationships between the two metrics or individual scene correlations.

## In Gariim, it becomes difficult to choose the target as well.

In the new occlusion scene, target selection was 57.0% for flat objects and 39.3% for slots. Slots failed to select any object candidates present in the scene at 32.0%. Even when the correct state was achieved, only 92.2% followed the rule of selecting the center closest to the point. Therefore, both expression recognition and target selection rules require separate examination.

The post-counts are calculated by dividing the visible area of the target by the proportion of the visible area. Each scene contains multiple commands, but the sample size is the original scene size. The average is the number of initial readings of the three readers.

| Visible ratio of the target | Number of scenes | Flat target selection | Slot target selection |
| --- | ---: | ---: | ---: |
| Less than 50% | 13 | 61.5% | 15.4% |
| 50~75% | 32 | 50.0% | 36.5% |
| 75~90% | 15 | 62.2% | 40.0% |
| More than 90% | 68 | 58.3% | 45.1% |

This segment is not a paired comparison of scenes that have been manipulated again, but rather a technical statistical measure of the degree of obscuration observed. Since there is also a small sample variation in the segment and differences in object shape and arrangement, it is not possible to infer causal effects solely from the obscuration ratio. Since all objects have visible hard pixels, a complete obscuration restoration was not evaluated.

## Command synthesis error

In the two object tests, the actual target's change properties matched the color change flat/Slot 93.5/92.2%, rotation 24.5/22.7%, and movement 15.6/21.6%. After color, the rotation was 24.0/22.4%. One should not conclude that the operation synthesis was not learned only because the low success rate of color+rotation was observed. Even from single rotation, directional recognition errors are significant.

Applying the same learning intervention model to the correct state and correct target resulted in 100% success for both initializations across all five new conditions, including single and composite commands. Even applying explicit correct‑answer conversion rules to the estimated state, the total success for actual correct answers was 0. The current pipeline recognition state error did not disappear even after replacing the controller with the correct‑answer rule. This is a diagnosis of a fixed encoder/decoder configuration.

We separated the score indicating that non-target objects were preserved compared to the estimated state from the score indicating that the actual answer's non-target properties matched. The electrons can still succeed even if they preserve the incorrect shape. The main overall success criterion includes the latter.

## Error examples and reproduction range

The [step-by-step GIF](../../../../followup/reports/object_image_edit_v1/edit_sequence.gif) that shows the first color + rotation task selected in the general test, with the three objects and masking, shows the recognition immediately, after color change, and after rotation. Only the white dot in the input is displayed as the target. The success/failure level was shown, and the example was not selected, and the reader/intervention model seed was fixed to 0. The model did not provide a middle answer. [Comparison curve](../../../../followup/reports/object_image_edit_v1/capability_curves.png), [PNG last step](../../../../followup/reports/object_image_edit_v1/edit_sequence_frame_2.png) were also stored.

[Intervention Evaluation Report](object_image_edit_v1/RESULTS_EN.md), [180 combination/10 failure classification/8 masking intervals](../../../../followup/reports/object_image_edit_v1/analysis.json), [2,138 additional aggregation verification](../../../../followup/reports/object_image_edit_v1/analysis_verification.json) were checked. The original prediction/image 68,103 cases and the indicators/intervals 17,388 cases also passed [independent re-verification](../../../../followup/reports/object_image_edit_v1/verification.json).

The experiment aimed at better modeling is a separate follow-up research from this failure result. The termination of B05/B07 means that the execution and analysis of the fixed comparison have been completed, and it does not mean the achievement of the entire study or the development of useful editing capabilities. External SVIB evaluation, video state and environment inference, long-term future and counterfactual prediction, and independent repetitions remain.
