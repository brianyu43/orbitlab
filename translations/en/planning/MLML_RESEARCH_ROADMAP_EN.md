# OrbitLab Follow-up Research Proposal — Symmetry based on combination expression and conditions

Writing and literature review: 2026-09-23. **This is an proposal for discussing the research direction, and the following follow-up experiments have not yet been implemented.** We will retain the existing 15th-round results and distribution ZIP files unchanged. The calculation scale is based on the principle of performing initial small verification on the existing Mac.

## 1. Recommended central questions

**Under what conditions does the use of symmetry help in the creation and prediction of new combinations? Can it also maintain this advantage when there are directional environmental conditions?**

In simple terms, this is a study that creates the color and shape combination that a model trained separately on red arrows and blue Ls sees for the first time, and that anticipates the subsequent scene when only one object moves. What is important here is whether the model distinguishes and uses the properties of the object, its position and direction, and the forces of the environment.

The width is obtained by establishing it as a single object → multiple objects → changes over time → an external benchmark. The depth is obtained through a comparison that separates the causes one by one, the mathematical definition of correct symmetry, intervention experiments that know the correct answer, and measurements of failure conditions. Instead of implementing all expansions at once, each stage is carried out to justify the next question.

## 2. Points connected to MLML and existing research

KAIST Machine Learning and Mind Lab presents structured representations, systematic generalization, and generative world models and plans as formal research topics. Therefore, the direction of developing OrbitLab into a problem of reassembling objects, properties, and transformation rules aligns with the research topic. This is a judgment based on publicly published research, and it does not mean that the laboratory's evaluation has been pre-confirmed. [Introduction to Public Research](https://mlml.kaist.ac.kr/research)

| Related research | Key points confirmed | What OrbitLab needs to clarify further |
| --- | --- | --- |
| [SysBinder](https://arxiv.org/abs/2211.01177v3) | Representation that hierarchically divides objects and their properties | Difference between directional latent and object-specific latent, preservation of other properties when properties are changed |
| [Dreamweaver](https://arxiv.org/abs/2501.14174v5) | Future generation by dividing and recombining objects, properties, and movements in video | Additional effects and failure conditions when structural symmetry is added |
| [SVIB](https://arxiv.org/abs/2311.09064) | Benchmark for evaluating generalization of unobserved combinations of visual world models | Does the same conclusion hold outside of self-figure data |
| [NEO: Learning to Theorize](https://arxiv.org/abs/2605.03413v3) | Inferring combinable rules from observations | Can we apply rules learned in one scene to another scene |
| [SEN](https://proceedings.mlr.press/v162/park22a.html) | World model combining learned representations and equivariant transitions | Do not claim the combination of symmetry and world model itself as a source of originality |
| [Invariant Slot Attention](https://proceedings.mlr.press/v202/biza23a.html) | Object representation utilizing object reference coordinates and geometric structures | Clearly distinguishing the difference in methods for incorporating geometric information per object |
| [Soft Geometric Inductive Bias](https://arxiv.org/abs/2512.15493v1) | A soft geometric bias considering the conditions where exact symmetry breaks down | Not simply treating it as a new claim that "it's good to relax strong constraints" |
| [GeoCo-SAVi](https://arxiv.org/abs/2609.06628v2) | Expression for explicitly editing the position, size, etc. of objects | Shows differences in combination generalization and dynamics beyond the editing capability itself |

Dreamweaver·NEO reviewed the paper and experimental settings together. Some related research is a preliminary survey focused on abstracts and official publication pages. Specifically, Soft Geometric Inductive Bias and GeoCo-SAVi are based on the arXiv materials reviewed. Before finalizing the method, an additional review is needed to compare the paper and public code of the closest research.

Currently, the most promising differentiation candidate is **connecting it in an experimental framework that generalizes the ability to observe the cause of symmetry breaking, the extent to which environmental conditions can be observed, and combinations that are not yet observed**. This does not yet mean that this combination has been confirmed to be unexplored. If the same experiment and conclusion appear in the closest related paper, the question is narrowed or redefined as a contribution to replication and analysis.

## 3. First, separate the cause of the current results

[In the existing results](../RESULTS_EN.md), the simultaneous matching rate of B2's generation shape and color was 57.0%, and B1 was 34.0%. If form validity is also required, it is 52.2% and 24.7%. On the other hand, in the same AE learning time, B1 restoration was better. This inconsistency is the starting point for subsequent research.

It is still difficult to definitively establish that “symmetry improved the generation performance” as a single cause. The constraints on expression, decoder, flow, and computational load were all different, and the evaluator was only independently verified on clean ground-truth diagrams. There was only one data generation seed.

### A. Verification from the generator evaluator

1. Create a separate evaluation set that applies blurring, contour deformation, fragment separation, and color mixing to the correct figure. Samples that are severely deformed and difficult to identify are not forced to be graded by their original shape and are classified as unidentifiable.
2. From the existing generated products, randomly select model×seed×seen/OOD and create samples for human review to hide the source. For example, 2 models×3 seeds×2 splits×20 sheets=240 sheets. Do not claim the accuracy of human evaluation before the human review is completed.
3. Shape and color accuracy, form validity, ratio that cannot be determined, and the consistency between the person and the evaluator are reported separately. If possible, the consistency between the two evaluators is also recorded.
4. Check the distribution and diversity of position, size, and direction for each condition. The distribution is not evaluated solely by the coverage of positions or the proximity pixel distance combined with all conditions.

**Judgment:** It distinguishes whether the relative advantage of B2 is maintained even if the evaluation method changes, and whether the evaluator was more favorable for a specific type of blurry figure. Even if the difference disappears, it is preserved as a result of identifying the reason for the evaluation.

### B. Only change the flow on the same expression

First, fix the same C4 AE checkpoint and compare the following three.

| Change factor | Comparison model | Fixed factor |
| --- | --- | --- |
| flow's symmetry constraint | general flow / rotation augmentation or consistency loss flow / accurate C4 flow | AE, latent normalization, learning scene, condition, evaluation noise |

Distinguish between actual calculation time comparisons like step comparisons and also record inference time. Also set the budget for hyperparameter selection. After checking the results of one AE, repeat it with other AE seeds and data seeds.

**Design error to be caution:** If you arbitrarily attach a slot shift to the latent of a general AE to create “C4 flow,” there is no guarantee that this shift corresponds to image rotation. Therefore, a simple 2×2 table of general AE/C4 AE and general flow/C4 flow will not automatically become a valid comparative experiment. It should be compared first within expressions that contain known rotation effects.

### C. Divide the restoration bottleneck and the generation bottleneck

- The current B2 decoder averages pixel outputs from multiple directions. We verify whether this is the cause of blurriness and whether the same phenomenon occurs in other equivariant decoders. We do not make a priori assumption that the average is the cause.
- Set the error when decoding the latent of the actual correct image as the reference line. Compare whether the generation error already exists in AE or was added in flow.
- It has separate oracle conditions that directly provide the correct attributes, location, and direction. This is a diagnostic comparison that bypasses visual expression learning and is not mixed with the competitive model performance of the same input conditions.
- For the new design, a new test pool is created. The existing tests that have already been reviewed are left as development reference materials. After the small pilot, the main conclusions are to be verified by 3 data seeds × 3 initialization seeds as the basic plan.

**Output of this stage:** A comparative table by cause explaining 57% vs. 34%, an analysis of evaluation errors, and a performance graph comparing total AE+flow learning and generation costs. The goal is not victory in a predetermined direction, but interpretable differences.

## 4. Expand with the intervention of several objects

Currently, the four latent bundles of B2 correspond to four rotation directions. They are not object slots that represent four objects. To handle multiple objects, a separate structure is needed to distinguish and track the objects.

The first task is two objects that do not overlap. Subsequently, add one axis at a time to a set of objects with overlapping, three or more objects, or combinations of relationships that are not observed. To test the generalization of the number of objects, provide sufficient slot capacity to hold the test objects from the beginning for all models.

For example, in a scene with a red arrow and a blue L, ask for the following:

- Change the color of the red arrow to green only.
- Rotate one object by 90 degrees around its center.
- Move only one object.
- Apply the two operations that were separately observed in the learning.

Using a renderer that preserves the generation state of the same scene, it generates the exact answer before and after the intervention. It measures together how much the target object changes precisely and how much other objects and properties remain unchanged. After removing arbitrary latent variables, the interpretation is clearer than observing that the shape has changed.

Manipulating objects is an intervention that changes the actual state. It is distinct from a symmetric transformation that rotates the entire coordinate system. Specifically, if only one colliding object is rotated, the subsequent physical outcome can change. We do not assume that all manipulations are equivalent even if they alter the order of operations. While color changes and rotations around a central axis of a single object can be designed to be interchangeable, the movement and rotation of fixed world coordinates are generally not.

The initial diagnosis is performed by an oracle that provides the correct mask and state, and then transferred to the condition of finding objects from the image. The two results are clearly separated, and the result using the mask is not called supervised object detection. All competing models provide the same observation information.

External validation is connected from the representative tasks of [SVIB official benchmark](https://systematic-visual-imagination.github.io/). It distinguishes results that follow the official split and evaluation methods from search results reduced to a small budget. If a separate configuration is used to show all combinations as overlapping images, it is reported separately with strict non‑observation combination settings. Whether the test combination is pre‑exposed can change the conclusion.

## 5. The candidate to develop it the most deeply: environmental conditions and world models

A world model is a model that predicts the future state based on current observations and actions. Initially, it determines the speed and environmental conditions using 2–4 past frames and predicts the next frame. Then, it inputs its own predictions again to generate a long future.

### Three environments of the core experiment

| Environment | Questions | Conditions for accurate comparison |
| --- | --- | --- |
| Simple motion that does not distinguish direction | Does the precise rotational constraint help with the combination generalization? | Designed to conform to the rotational rules from boundaries, objects, and observations |
| Fix the gravity in the downward direction or the wind in one direction | What error occurs if you only rotate the object and force the same rule? | Adjust the strength of the directional effect and the observation information separately |
| Provide gravitational and wind direction as inputs or infer from the past | Does generalizing even when expressing environmental conditions restore generalization? | Provide the same environmental information to both general models and augmented models |

For example, gravity acts downward; if you only rotate the object's orientation by 90 degrees, the future scene does not necessarily need to be identical to the original future simply rotated by 90 degrees. You must define the relationship that arises when both the state and the direction of gravity and its behavior are rotated together.

The relationship that the candidate model must satisfy for state s, action a, environmental conditions c, and 90° rotation R is as follows.

`T(Rs, Ra, Rc) = R T(s, a, c)`

This is a relationship that also transforms environmental conditions together. It is different from forcing the same relationship unconditionally on the fixed gravitational field `T(Rs, Ra, c)`. It distinguishes between the symmetry that recovers when all the environmental directions are expressed, and the cases where recovery is impossible when information is hidden. Cases where only the symmetry of the observed image is broken due to lighting and obscuration are also handled separately from violations of dynamical symmetry.

**Hypothesis to be verified:** The success or failure of strong symmetry constraints depends not only on the strength of the constraint but also on how environmental variables are expressed and transformed. A structure that expresses the conditions together may be advantageous in the first-time combination of objects, environmental directions, and long prediction intervals. This is a hypothesis that has not yet been verified.

Simply adding a simple gravitational vector to the input can cause the effect to disappear, or if a general enhancement model yields the same performance, the fact is taken as a conclusion. A weak comparison model is not chosen unless a precise symmetry model is advantageous.

### Minimum comparison and evaluation

- Baseline: constant-speed extrapolation, transition of the latent of the entire scene, transition of objects, rotation enhancement model, accurate symmetry model, constrained model, model that transforms environmental conditions together. Add the necessary comparison from the results of the previous stage.
- Input control: In experiments where the environmental conditions are known, all learning models receive the same conditions. In experiments where the conditions are inferred, they receive the same number of past frames. Unidentifiable futures are treated as problems similar to an oracle that receives the correct conditions, considering uncertainty.
- Division: It measures the separation of color/shape combinations not learned, object arrangement relationships, environmental direction, and combinations of operations. It does not present only the total score that changes multiple difficulty levels at once.
- Length: For example, use the exploration strategy that is learned within 16 frames and then extended to 32/64 frames. The values are set as a pilot and fixed before the verification experiment.
- Indicators: object position and velocity, maintaining identity, collision time and subsequent motion, effect of target object intervention, appropriate response of non-target objects, error per prediction interval. After the collision, it may be the correct answer that other objects actually change, so there is no requirement for conservation in general.
- Statistics: Consecutive frames of the same trajectory are not counted as independent samples. It distinguishes uncertainty between trajectory and scene units and between learning seed.

Here, claims about the cause are limited to the range of synthetic environments that can be manipulated. Causality is not asserted solely based on video prediction scores. Behavioral planning or NEO-style rule transfer is treated as a selection expansion following the verification of basic dynamics and intervention predictions.

## 6. Small supplementary tasks that add mathematical depth

If the expression E, which is completely invariant under rotation, satisfies `E(Rx)=E(x)` and the decoder is deterministic, it cannot distinguish inputs from different rotation trajectories and recover them. The minimum average error of the possible common output p for pixel-wise squared error is as follows.

`min_p mean_g ||g x - p||² = mean_g ||g x - mean_h(h x)||²`

The average video is optimal, and the rotational variation around it is the limit of loss. This is a standard minimum‑squared‑error formula, so it does not claim to be a new theorem. By simply deriving it and comparing the actual limits for each shape with the loss of B3, we can quantitatively explain “why is it difficult to recover when we eliminate directional information.” If there are constraints on the decoder output, the limit may not be achieved.

This formula is for the mean of the uniform rotation and the general squared error. The corresponding MSE must be calculated separately, not by comparing it to the existing MAE directly. The B2 expression or distribution outside the band does not unconditionally apply this lower bound to the entire result after band removal.

## 7. Progress order and completion criteria

| Step | Key task | Result to take to the next step |
| --- | --- | --- |
| 1. Causes of current phenomena | Evaluation instrument verification, flow comparison of the same AE, decoder bottleneck, multiple data seed | Reproducibility of relative advantages and causes, uncertainty, cost |
| 2. Objects·Combinations·Interventions | Pair of correct interventions for two objects, expressions for each object, representative SVIB tasks | Measure both the change of the target and the preservation of properties in combinations that are not observed |
| 3. Conditions and the Future | Directionless environment → Fixed forces → Provision/inference of conditions | Curve showing which conditions provide which constraints that are beneficial and harmful |
| 4. Research Package | Most strong conclusion repetition, closest method comparison, failure case | Paper format notes, execution settings, replication data, limitations |

If the results of step 1 are explained by an evaluation bias, then after correcting that evaluation, the generation superiority claim and subsequent priority are modified. If image-based object separation fails in step 2, a bottleneck is reported by distinguishing it from the oracle results. If only condition inputs are sufficient in step 3, the conclusion is drawn that the need for a complex new architecture is low.

In the existing Mac, evaluation, small renderer, cause separation, and small dynamic pilot are performed from the beginning. After measuring time and memory with short learning for each candidate, the total repeat budget is calculated. Reproducing the size of the original Dreamweaver paper is a separate project with a GPU size, and reduction and transformation are distinct from reproducing the original paper. We do not promise completion dates for subsequent work that has not yet been measured for execution time or cost.

The final bundle that is good for the research lab to review is as follows.

1. Page-by-page problem definition: What is being asked anew, and which assumptions of existing research are being tested.
2. Comparison table with the 3-5 closest methods: input information·symmetry·intervention·generalization·calculation amount.
3. Three key pictures: separation of causes, performance according to changes in conditions, the combination you see for the first time and the failure curve of the long future.
4. Short intervention video: Display the changed variables and the answer/prediction together.
5. Research notes on pages 4–6, replication code, setting, materials, and failure cases.

First, the initial candidate is the **flow comparison experiment in AE, identical to the evaluation device verification**. Based on this result, we recommend that after establishing the current observations as reliable, we advance the research to a central study focusing on a **world model that includes the intervention of multiple objects and environmental conditions**.
