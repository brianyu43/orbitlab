# What the simple video processing benchmark has told us

2026-09-23. This is a diagnosis targeting 128 validation images of the two objects that were currently off-target. It is not the final performance of the learning model or the image editing performance.

From the input RGB, I found the connected area of pixels with brightness greater than 0.05, and I cut the blurred edges based on the brightness of each area itself. I did not input the correct answer mask, the actual number of objects, the color palette, or the object location. I only used the capacity of the task that allows up to four objects. I saved the settings before running and did not adjust the criteria to match the correct answer.

| Indicator | Connection area baseline |
| --- | ---: |
| Object area overlap, 1 is perfect | 0.9973 |
| Full view ARI | 1.0000 |
| Total pixel ARI | 0.9986 |
| Matching the number of visible objects | 128/128 |

In the same validation, the overlap of the object regions in the 3,000-step learning model was flat 0.0933 and Slot Attention 0.0720. Therefore, in this easy separation condition, the region can be found very accurately even without the correct mask. There is no evidence to explain the failure of the learning model as the inherent irreducibility of the data. We need to diagnose the curve that learns longer and the separation method of the model.

The connection area method allows two adjacent objects to be merged into a single entity. In the actual example of a red/green square that is actually touching, it was tested to verify the failure of merging into a single object. It does not mean that it obtains a generalized object representation that can be applied to occlusion, the same color, or complex backgrounds. It is left as a contrast to argue for the contribution of new object representation research solely based on the high scores of easy data at present.

[Raw material and aggregation](../../../../../followup/reports/object_component_diagnostic_v1/summary.json) · [Video playback, rotation, independent overlap calculation verification](../../../../../followup/reports/object_component_diagnostic_v1/verification.json) · [Input and detection results](../../../../../followup/reports/object_component_diagnostic_v1/examples.png)
