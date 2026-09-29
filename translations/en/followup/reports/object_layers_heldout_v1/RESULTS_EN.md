# Fixed evaluation of the model for finding objects in images

2026-09-23. The current model implementation, restoration, object assignment, and masking evaluation for B04 has been completed. **This does not mean that stable object understanding or image manipulation capabilities have been achieved.** The initial evaluation fixed both models and evaluated them, and changing the target for B05 or preserving non-targets is the next step.

## What did you distinguish?

The 12,000-step flat/Slot Attention model was fixed. Input is RGB, and the number of correct masks, properties, and actual objects is not specified in the model or post-processing. Both models have 5 learning slots, which provides the capacity to represent up to four objects and the background in evaluation. Since flat has 216,324 parameters and Slot Attention has 148,100 parameters, the comparison is not based on the same parameter budget.

In the same validation, we distinguished assignments made using encoder attention, decoder mask, and RGB contribution scores. A common post-processing step was also applied, where only the portion of the input RGB brightness greater than 0.05 was retained for each assignment. This criterion was based on values used in the existing training loss and was not adjusted to match the ground truth. The original mask results are preserved, and the post-processing is reported using a separate method.

In validation, the original decoder mask for Slot Attention consisted of approximately 91.7% of the corresponding region being the actual background. The input attention also was similar, at 91.3%. By leaving the foreground, overlapping regions improved the decoder scores from 0.090 to 0.621 and the attention scores from 0.087 to 0.619. Therefore, it is difficult to consider that only changing the output decoder is sufficient, and it must also be checked for background separation and object allocation. [validation raw data](../../../../../followup/reports/object_layers_validation_v1/summary.json)

## The results of the material that was withheld

Below is the IoU of the mask that best matches the **visible area** of the actual individual object. 1 is perfect, not the command success rate or the percentage accuracy. Each condition is a different scene with 128 images. The post‑processing method and checkpoints were fixed before seeing this result and all transformations were reported.

| Evaluation conditions | flat original mask | Slot original mask | flat + RGB background | Slot + RGB background | Reference line of connection area |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test of two objects | 0.143 | 0.086 | 0.759 | 0.628 | 0.997 |
| Unobserved shape and color combination | 0.144 | 0.094 | 0.745 | 0.605 | 0.999 |
| Three objects | 0.137 | 0.103 | 0.720 | 0.707 | 0.998 |
| Four objects | 0.113 | 0.113 | 0.645 | 0.742 | 0.998 |
| Two objects covered by each other | 0.082 | 0.071 | 0.384 | 0.427 | 0.342 |

The characteristic of the black background task helps in RGB foreground preprocessing. It does not interpret learned background understanding as generalizing to a complex background. The fact that the better side of two objects was not always the better side of all four objects was not the result of a paired experiment where only objects were added to the same initial scene, so the cause of this difference cannot be definitively determined by slot competition.

The IoU of the restored image's foreground was 0.394/Slot 0.528 for the test, 0.347/0.417 for the four objects, and 0.464/0.650 for the masked region. Although the restoration of the entire image's foreground in the masked region appeared good, the ability to accurately separate individual objects was low. [Restore/Failure Analysis](../../../../../followup/reports/object_failure_audit_v1/summary.json)

## Types of errors

In the occlusion condition, **fusion** was mainly assigned the same prediction area to both actual objects, with the connection area baseline at 97.7%, flat+foreground at 73.4%, and Slot+foreground at 28.1%. Conversely, **residual** was the case where 80% of the pixels visible on the actual object did not fall into a single prediction area, with the average for each object being 3.5%, 41.0%, and 70.7%. The Slot model had a high error rate in dividing the object into multiple pieces instead of fusing them. If different errors are judged only by a single score, they may be missed.

The degree of occlusion was divided into how much of the original object's area is visible. Scenes where the proportion of the least visible object is less than 50%, 50~75%, 75~90%, or 90~100% have 34, 58, 24, and 12 instances, respectively. The connection area method can also have relatively high IoU even in severely obscured regions. This is because the prediction of combining two objects into one overlaps well with the object with a large area, and it does not mean that the obscured object has been recovered. This evaluation did not measure the ability to infer the shape of the hidden area.

![Object separation by heldout conditions and degree of concealment](../../../../../followup/reports/object_failure_audit_v1/heldout_segmentation.png)

The error bar is the 95% interval of the scene re-sampled under a fixed model. It does not include uncertainty between model initialization. [Expandable SVG](../../../../../followup/reports/object_failure_audit_v1/heldout_segmentation.svg)

## Verification and next connection

1,536 validation and hold data learning model outputs, and 6,912 object correspondences were reproduced and verified. The 1,280 restoration lines and the 5,760 error analysis lines for hold data were also separately verified. We confirmed that there are actual pixels visible for each object under all conditions. [Fixed Evaluation Verification](../../../../../followup/reports/object_layers_heldout_v1/verification.json) · [Error Analysis Verification](../../../../../followup/reports/object_failure_audit_v1/verification.json)

In B05, the evaluation assesses whether, when using this expression, the properties of the target can be changed while preserving other objects. To determine the target, the points/indicators provided by the actual user and the state that the model must infer are distinguished, and the correct object ID or the entire correct mask are not passed to the image model input. The existing correct state model B03 is retained with a separate upper‑bound diagnosis. Since error analysis of image‑based intervention and manipulation synthesis remains, it is not counted as completed with B05/B07.
