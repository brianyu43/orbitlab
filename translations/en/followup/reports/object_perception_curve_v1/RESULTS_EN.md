# Additional learning results of object recognition models

2026-09-23. After continuing the weights, Adam state, learning sample random numbers, and slot initialization random numbers from the existing 3,000-step model, the model was trained up to 6,000/12,000 steps. The additional training for the two models is 9,000 steps each, totaling 18,000 steps. The existing pilot model was not replaced with a new initialization. Below is a development diagnosis for one initialization and 128 validation samples, and the test/new object count/masking results have not yet been opened.

| Model | Cumulative learning steps | Overlapping foreground images | Overlapping object regions | Foreground ARI |
| --- | ---: | ---: | ---: | ---: |
| flat | 3,000 | 0.3924 | 0.0933 | 0.8121 |
| flat | 6,000 | 0.3709 | 0.1024 | 0.8422 |
| flat | 12,000 | 0.3968 | 0.1390 | 0.8142 |
| Slot Attention | 3,000 | 0.4551 | 0.0720 | 0.7659 |
| Slot Attention | 6,000 | 0.4835 | 0.0793 | 0.6830 |
| Slot Attention | 12,000 | 0.5226 | 0.0896 | 0.5912 |

Overlap in the image foreground compares the parts of the object in the reconstructed image to the correct foreground. Overlap in the object region is the IoU when the output masks best match the actual individual objects. Both are perfect when 1 is achieved, but they evaluate different capabilities. ARI is not the command success rate or the accuracy percentage.

Slot Attention scores increased, indicating a higher ability to recover the approximate foreground of the image, but the scores for identifying individual objects remained low. In the examples, the masks for each slot covered a wide background, and the shape of the recovered figures was also round and blurry. The foreground ARI, on the contrary, decreased. Therefore, it cannot be concluded that object detection progressed solely due to learning loss or the recovery of the image.

The [simple connection area baseline](../object_component_diagnostic_v1/RESULTS_EN.md) in the same validation achieved an overlap of 0.9973 between object areas without a ground truth mask. In this easy separation condition, the failure of the learning model was substantial. In contrast, the baseline fails to resolve general object understanding because it merges the overlapping objects.

## Verification

The 512 validation outputs from the four checkpoints were recalculated on the CPU. The parent model and optimizer hashes, Adam step, and two random streams were regenerated from the original seed, and the initial weights, identical training samples, and all evaluation rows and averages were checked. A small floating‑point difference between CPU/MPS was recorded. The hard mask of 12,000‑step Slot Attention differed by one pixel among the 128×64×64 pixels, and the existing metrics were recalculated based on the stored MPS mask. [Full verification](../../../../../followup/reports/object_perception_curve_v1/verification.json)

The model's input is RGB, and the ground truth mask was used only for evaluation. Both models used the same step, image, and sampling, but the number of parameters differs with 216,324 flat parameters for Slot Attention and 148,100 for Slot Attention, and the computational time is not the same. [Material aggregation](../../../../../followup/reports/object_perception_curve_v1/summary.json) · [Slot Attention 12,000-step example](../../../../../followup/runs/object_perception_curve_v1/slot_s0_step12000/val_examples.png)

## Next diagnosis

1. Measure the encoder's attention and decoder's mask separately at the same checkpoint to determine at which stage the separation failure occurs.
2. The common postprocessing that only uses the foreground obtained from RGB is applied to both models and the baseline to measure the effect of background leakage. It does not cover the original mask indicator.
3. The changes in input normalization, position expression, and separation structure are recorded as independent experiments. After locking in the final settings, connect the test/object count/obscuring conditions and image intervention to the test/obscuring conditions.

B04 is in progress. It does not draw any conclusion that it will never improve in more steps, nor does it draw any conclusion that the current object representation is sufficient.
