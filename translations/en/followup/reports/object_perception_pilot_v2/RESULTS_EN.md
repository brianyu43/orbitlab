# Two times development learning of RGB object representation

2026-09-23. The correct mask and properties were trained without inputting them during training or loss calculation, and were learned solely through RGB reconstruction. Each version was trained on a single flat/Slot Attention model, and only 128 validation images were evaluated. The entire B04 is not considered complete or independent test performance.

In the first version, both models collapsed against a black background at 1,000 steps. The foreground MAE was 0.452, and the IoU before restoration was 0. The code, checkpoints, and evaluations were preserved.

The second version used a foreground/background balance loss calculated using linear RGB output and input brightness instead of sigmoid and was trained for 3,000 steps. Since both modifications and increased training data volume are present simultaneously, this comparison is not a separation of the causal effects of each factor. Within the same version, the image, sample order, training step, and slot capacities of the two models were matched. The parameters are 216,324 for Flat and 148,100 for Slot Attention, and this comparison is not a match of the same time/parameter values.

| Validation indicators | Flat | Slot Attention |
| --- | ---: | ---: |
| Recovery view MAE, the lower the better | 0.154 | 0.122 |
| IoU view before restoration, the higher the better | 0.392 | 0.455 |
| Foreground ARI, separation between object foreground | 0.812 | 0.766 |
| All pixels ARI, background included | 0.002 | 0.009 |
| One-to-one correspondence between object and slot mask IoU | 0.093 | 0.072 |
| 3,000-step learning time, record value | 167 seconds | 182 seconds |

The IoU of the restored foreground and the corresponding mask IoU of the slot are different indicators. The former shows how much the combined output image recovers the original foreground, while the latter shows how much each individual slot takes up the actual object area.

The output image has improved compared to the first version, but it still resembles a blurry color blob. Each slot's mask contains a wide background area mixed in. If you only isolate the foreground, the two objects may enter different regions, causing ARI to increase, but that does not mean the actual object contours have been separated. We do not claim object detection success by comparing the low response mask IoU with the actual mask image together.

We checked the checkpoint, data, and aggregation hash for four versions of Model 4. The metrics were recalculated from the stored mask and the RGB predictions were replayed by the CPU. The maximum image error for the second version of CPU/MPS was less than 0.0002, and the hard mask mismatch was 0. Verification success indicates the consistency of the outputs and is not a criterion for passing processing when dealing with low model quality.

## Tasks to do next

1. Preserve the same v2 checkpoint and optimizer/RNG to measure a longer learning curve. The model is trained from scratch from the beginning, so previous calculations are not discarded. The validation set is used to monitor convergence and mask status, while the test set is evaluated after the final protocol is finalized.
2. For foreground and background loss, separate diagnosis is performed for the mask's area and RGB values per slot. The next stage is not determined by a single ARI for the foreground. Any necessary structural changes are recorded by comparing them with the same conditions as the separate version.
3. Once the object representation is established, connect the property reading and image input intervention. The part that uses the map learning probe is divided into unsupervised representation learning. Objects with different number of parts, occlusion, and similar colors are evaluated under different failure conditions.
4. Independent repetition, external SVIB and environmental conditions and future predictions remain unchanged in the overall plan.

[Summary](../../../../../followup/reports/object_perception_pilot_v2/summary.json), [Verification](../../../../../followup/reports/object_perception_pilot_v2/verification.json), [Slot Attention input, restoration, mask examples](../../../../../followup/runs/object_perception_pilot_v2/slot_s0/val_examples.png), [Flat examples](../../../../../followup/runs/object_perception_pilot_v2/flat_s0/val_examples.png)
