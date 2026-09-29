# Construction and verification of data on exercise, collision and environmental conditions

2026-09-23. C01/C02 data and simulator verification completed. **The performance results of the future prediction neural network are not yet available.**

## Composition

On a 64×64 black screen, a circular object of six colors with a radius of 4/5/6 pixels moves. It is a different task from the polygons in the B phase, and both masses are 1. The color does not affect physics, while the radius affects collisions. Collisions are elastic collisions, and walls are reflection boundaries. It is calculated 8 times per frame, and overlapping is resolved by position projection. This is a fixed discrete simulator and does not represent an accurate collision solution for continuous time or the results verified in the real world. The processing order of contact that occurs simultaneously is fixed by a continuous object ID.

In the case of no external forces, there are three types of fixed gravity pulling downward, and gravity, wind, and isotropic damping changing. Here, wind is defined as a constant acceleration. It does not refer to the velocity field of the actual air flow or drag force.

| Category | Trajectory count | Object count | Storage frame |
| --- | ---: | ---: | ---: |
| train, 256 pieces per environment | 768 | 2 | 20 |
| validation, 64 per environment | 192 | 2 | 65 |
| test, 128 pieces per environment | 384 | 2 | 65 |
| Size and color combination excluded from learning | 384 | 2 | 65 |
| Increase in object count | 384 | 3 or 4 | 65 |
| New direction and range of external force | 128 | 2 | 65 |
| Total | **2,240** | | |

First, observe the first 4 frames, and apply a velocity change command to one object at t=3. The train only stores the next 16 frames. The evaluation includes the next 61 frames, allowing us to measure a long future. The frame, rotation, and action transformations of each trajectory belong to the same split group. There was no overlap between the initial scene video and the rotation trajectories of the state, action, and environment. Only the RGB and split data of the first 4 frames are stored, and the future video is regenerated from the state.

## Relationship confirmed

`next_state(rotated_state, rotated_action, rotated_force) = rotate(next_state(original_state, original_action, original_force))`

In the 912 trajectories that were also rotated to the direction of the external force, the maximum difference in stored float64 calculations was 0. In contrast, in the 96 counterexamples where gravity was fixed downward and only the state and action were rotated by 90 degrees, the maximum difference in the next state coordinates was approximately 0.05. Therefore, experiments can be conducted to distinguish which symmetry is forced into the model in a directional environment. This finite test is not interpreted as a generalization for any arbitrary physical system.

When both gravity and wind are at a constant acceleration, the state change depends only on the **sum** of the two. It was also verified that other decompositions that produce the same sum also yield the same trajectory. Therefore, the identification targets for subsequent video inference are the resultant force and damping, and the claim that individual values of gravity and wind were determined from the video is not made.

## Verification results

- Played all 2,240 basic trajectories, 111,040 status frames, and 8,960 observation videos.
- The error in the kinetic energy of motion per frame for trajectories without external forces was a maximum of 6.22×10⁻¹⁵. This is a comparison that reflects the energy changed by the action at t=3. Since the wall changes the momentum, the total momentum conservation was confirmed in a test of two object units without a wall.
- The maximum overlap/wall penetration of all storage states was less than 3.56×10⁻¹⁵ pixels.
- In the test/new external force conditions, 512 response trajectories were added where only the direction of the command was reversed. The four front frames observed are identical, and only the command direction differs thereafter. In the 419 cases, non-target objects also exhibited different motion through collisions. Therefore, in subsequent counterfactual predictions, if non-target objects are unconditionally fixed, the incorrect answer will be obtained.
- Changed the object storage order, rotation of the circular renderer, free movement, accumulation of external forces, collision between two objects, wall reflection unit check were also passed.

The full replay record is in [verification.json](../../../../../followup/reports/dynamics_world_v1/verification.json), the generation conditions are in the [locked configuration](../../../../../followup/configs/dynamics_world_v1.json), and the data are in the [manifest](../../../../../followup/data/dynamics_world_v1/manifest.json). The [video with reversed actions](../../../../../followup/data/dynamics_world_v1/isotropic/test/example.gif) lets you compare the left and right trajectories.

## Next execution

1. Compare the constant speed reference line, general scene model, object-by-object model, and interaction model in C03. In symmetric comparisons, the same basic network, weights, and sample are used, and the incorrect condition that only rotates the state is separated from the correct condition that rotates the state due to external forces.
2. The results from the provided correct state status and the estimated state results from the video are divided into C04. Only the past of the same length is observed.
3. Evaluate the long-term future, changes in external forces, increase in object number, and changes in behavior in C05/C06. Do not judge the existence of the current response trajectory by the success of model prediction.
